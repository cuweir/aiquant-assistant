import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select
import asyncio
from .data_fetcher import data_fetcher_instance  # Keep for potential other uses
from ..core.config import settings
from ..llm_providers.base import LLMStrategy
from ..db.session import SessionLocal
from ..db.models import AnalysisResult, Symbol, Strategy, HistoricalOhlcv, Position
from ..utils.formatters import format_price_dynamically
from .parameter_manager import ParameterManager

from .backtest.db_data_fetcher import fetch_df_from_postgres
from ..strategies.multi_indicator_strategy import MultiIndicatorStrategy
from .order_executor import OrderExecutor


class AnalysisService:
    def __init__(self, param_manager: ParameterManager, order_executor: OrderExecutor, llm_strategy: LLMStrategy):
        """
        Initializes the AnalysisService with its required dependencies.
        """
        self.param_manager = param_manager
        self.order_executor = order_executor
        self.llm_strategy = llm_strategy

    def _get_or_create_symbol(self, db: Session, symbol_name: str) -> Symbol:
        """Finds an existing symbol or creates a new one in the database."""
        symbol = db.query(Symbol).filter(Symbol.name == symbol_name).first()
        if not symbol:
            print(f"Symbol '{symbol_name}' not found in DB, creating new entry.")
            symbol = Symbol(name=symbol_name)
            db.add(symbol)
            db.commit()
            db.refresh(symbol)
        return symbol

    def _get_or_create_strategy(self, db: Session, strategy_name: str, config: dict) -> Strategy:
        """Finds an existing strategy or creates a new one in the database."""
        strategy = db.query(Strategy).filter(Strategy.strategy_name == strategy_name).first()
        if not strategy:
            print(f"Strategy '{strategy_name}' not found in DB, creating new entry.")
            strategy = Strategy(
                strategy_name=strategy_name,
                description="Volatility Adaptive Signal Confluence Strategy V7",
                config=config
            )
            db.add(strategy)
            db.commit()
            db.refresh(strategy)
        return strategy
    def _fetch_data_from_db(
            self, db: Session, symbol_id: int, timeframe: str, limit: int
    ) -> pd.DataFrame | None:
        """
        Fetches OHLCV data directly from the local historical database.
        """
        print(f"Fetching data from LOCAL DB for symbol_id={symbol_id}, timeframe={timeframe}, limit={limit}")

        # Build the query to get the latest 'limit' candles
        stmt = (
            select(HistoricalOhlcv)
            .where(HistoricalOhlcv.symbol_id == symbol_id, HistoricalOhlcv.timeframe == timeframe)
            .order_by(HistoricalOhlcv.open_time.desc())
            .limit(limit)
        )

        # Use pandas to execute the query and load data into a DataFrame
        # The connection can be extracted from the session
        df = pd.read_sql_query(stmt, db.bind)

        if df.empty:
            print(f"Warning: No data found in local DB for symbol_id={symbol_id}, timeframe={timeframe}")
            return None

        # --- Data Formatting (Crucial to match the old format) ---
        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)

        # Convert numeric types from Decimal to float for pandas_ta
        cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # The data is fetched in descending order, so we reverse it to get chronological order.
        df.sort_index(ascending=True, inplace=True)

        return df

    async def generate_comprehensive_analysis(self, symbol_name: str) -> Dict[str, Any] | None:
        """
        Orchestration method for the complete trading lifecycle.
        It checks for open positions and decides whether to open, monitor, or close a trade.
        """
        print(f"\n--- Analyzing {symbol_name} ---")
        db: Session = SessionLocal()
        try:
            symbol_record = self._get_or_create_symbol(db, symbol_name)

            # [STATEFUL LOGIC] Check if we already have an open position for this symbol
            open_position = db.query(Position).filter(
                Position.symbol_id == symbol_record.id,
                Position.is_open == True
            ).first()

            if open_position:
                # If a position is open, our only job is to check for an exit signal.
                await self._handle_open_position(db, symbol_record, open_position)
            else:
                # If no position is open, we look for a new entry signal.
                await self._handle_no_position(db, symbol_record)

        except Exception as e:
            print(f"  > An unexpected error occurred during analysis for {symbol_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            db.close()

    async def _handle_no_position(self, db: Session, symbol: Symbol):
        """Logic for when we are flat and looking for a new entry."""
        print(f"  > No open position for {symbol.name}. Looking for entry signals.")

        symbol_params = self.param_manager.get_params_for_symbol(symbol.name)
        trading_strategy = MultiIndicatorStrategy(params=symbol_params)

        # Fetch data
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = end_date - pd.Timedelta(days=100)
        df_signal = fetch_df_from_postgres(symbol.name, settings.SIGNAL_TIMEFRAME, start_date, end_date)
        df_regime = fetch_df_from_postgres(symbol.name, settings.TREND_TIMEFRAME_SHORT, start_date, end_date)
        if df_signal is None or df_regime is None: return

        # Generate signal
        strategy_result = await trading_strategy.generate_signals(df_signal, df_regime)
        if not strategy_result or strategy_result['overall_signal'] != "POTENTIAL_BUY":
            print(f"  > No valid entry signal found for {symbol.name}.")
            return

        print(f"  > POTENTIAL_BUY signal found for {symbol.name}!")

        # --- [EXECUTION LOGIC] ---
        # 1. Calculate order size (simplified for now, using a fixed amount)
        # A full implementation would use the risk management logic we discussed.
        order_amount = 0.001  # Example: buy 0.001 BTC

        # 2. Place the market buy order
        entry_order = await self.order_executor.create_market_order(symbol.name, 'buy', order_amount)
        if not entry_order:
            print(f"  > FAILED to place entry order for {symbol.name}. Aborting.")
            return

        # 3. Place the stop loss order
        sl_price = strategy_result['risk_management']['suggested_sl']
        sl_order = await self.order_executor.create_stop_loss_order(symbol.name, 'sell', order_amount, sl_price)
        if not sl_order:
            print(f"  > CRITICAL: FAILED to place stop loss order for {symbol.name}. Manual intervention required!")
            # In a real system, this would trigger an emergency alert.
            # We should try to close the position we just opened for safety.
            await self.order_executor.close_market_position(symbol.name, 'LONG', order_amount)
            return

        # 4. [PERSISTENCE] Save the new position to the database
        new_position = Position(
            symbol_id=symbol.id,
            is_open=True,
            position_side='LONG',
            entry_price=entry_order['price'],
            quantity=entry_order['amount'],
            entry_order_id=entry_order['id'],
            stop_loss_order_id=sl_order['id'],
            initial_stop_loss_price=sl_price
        )
        db.add(new_position)
        db.commit()
        print(f"  > Successfully opened and persisted new LONG position for {symbol.name}.")

    async def _handle_open_position(self, db: Session, symbol: Symbol, position: Position):
        """Logic for when we have a position and are only checking for exits."""
        print(f"  > Found open LONG position for {symbol.name}. Checking for exit signals.")

        symbol_params = self.param_manager.get_params_for_symbol(symbol.name)
        trading_strategy = MultiIndicatorStrategy(params=symbol_params)

        # Fetch data
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = end_date - pd.Timedelta(days=100)
        df_signal = fetch_df_from_postgres(symbol.name, settings.SIGNAL_TIMEFRAME, start_date, end_date)
        df_regime = fetch_df_from_postgres(symbol.name, settings.TREND_TIMEFRAME_SHORT, start_date, end_date)
        if df_signal is None or df_regime is None: return

        # --- Check for Exit Signal ---
        # The exit signal is the trend reversal (death cross)
        # We need to calculate the MAs to check this condition.
        is_high_vol = df_signal['ATR_14'].iloc[-1] > df_signal['ATR_14'].rolling(100).mean().iloc[-1]
        if is_high_vol:
            ma_short = df_signal['close'].rolling(symbol_params['high_vol_ma_short']).mean().iloc[-1]
            ma_long = df_signal['close'].rolling(symbol_params['high_vol_ma_long']).mean().iloc[-1]
        else:
            ma_short = df_signal['close'].rolling(symbol_params['low_vol_ma_short']).mean().iloc[-1]
            ma_long = df_signal['close'].rolling(symbol_params['low_vol_ma_long']).mean().iloc[-1]

        if ma_short < ma_long:
            print(f"  > Exit signal (Death Cross) found for {symbol.name}. Closing position.")

            # --- [EXECUTION LOGIC] ---
            # 1. Close the position with a market order
            close_order = await self.order_executor.close_market_position(
                symbol.name, 'LONG', float(position.quantity)
            )
            if not close_order:
                print(f"  > CRITICAL: FAILED to close position for {symbol.name}. Manual intervention required!")
                return

            # 2. Cancel the now-redundant stop loss order
            await self.order_executor.cancel_order(position.stop_loss_order_id, symbol.name)

            # 3. [PERSISTENCE] Update the position status in the database
            position.is_open = False
            db.commit()
            print(f"  > Successfully closed and updated position for {symbol.name}.")
        else:
            print(f"  > No exit signal. Holding position for {symbol.name}.")
            # [FAULT TOLERANCE] We can add the SL sync logic here if needed.
            # E.g., check if the SL order still exists on the exchange. If not, recreate it.

    # ... [ _build_llm_prompt, get_all_analyses_from_db, close_llm_resources remain unchanged ] ...
    def _build_llm_prompt(self, symbol: str, timeframe: str, result: Dict[str, Any], df: pd.DataFrame) -> str:
        # This function should be reviewed to ensure it uses the new strategy_result structure
        price_str = format_price_dynamically(result['current_price'])
        prompt = f"""
        Cryptocurrency Analysis Request for {symbol} ({timeframe}):

        - **Price:** {price_str}
        - **Calculated Signal:** {result['overall_signal']} (Score: {result['total_score']})
        - **Suggested Stop Loss (SL):** {format_price_dynamically(result.get('suggested_sl')) if result.get('suggested_sl') else 'N/A'}

        **AI Analyst Task:**
        You are a professional, data-driven crypto analyst. Your advice must be concise and actionable.
        1.  Assess the signal: Is it strong? What are the key confirming factors in the recent data?
        2.  Provide Final Suggestion: **[Strong Buy / Buy / Hold / Avoid]**
        3.  Justify in 1-2 sentences.
        """
        return prompt.strip()

    def get_all_analyses_from_db(self, db: Session, skip: int = 0, limit: int = 20) -> List[AnalysisResult]:
        """Fetches a paginated list of analysis results from the database."""
        return db.query(AnalysisResult).options(
            joinedload(AnalysisResult.symbol),
            joinedload(AnalysisResult.strategy)
        ).order_by(AnalysisResult.timestamp.desc()).offset(skip).limit(limit).all()

    async def close_llm_resources(self):
        await self.llm_strategy.close_clients()

