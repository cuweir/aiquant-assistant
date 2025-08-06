import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select
import asyncio
from .data_fetcher import data_fetcher_instance  # Keep for potential other uses
from .trading_service import TradingService
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
    """
       The "Scout" of the system.
       Its only job is to analyze the market and generate a signal,
       then pass it to the TradingService for a decision.
       """
    def __init__(self, param_manager: ParameterManager, trading_service: TradingService, llm_strategy: LLMStrategy):
        self.param_manager = param_manager
        self.trading_service = trading_service  # <-- Injected dependency
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

    async def generate_comprehensive_analysis(self, symbol_name: str):
        print(f"\n--- Analyzing {symbol_name} ---")
        db: Session = SessionLocal()
        try:
            symbol_record = self._get_or_create_symbol(db, symbol_name)
            symbol_params = self.param_manager.get_params_for_symbol(symbol_name)
            trading_strategy = MultiIndicatorStrategy(params=symbol_params)

            end_date = pd.Timestamp.now(tz='UTC')
            start_date = end_date - pd.Timedelta(days=100)
            df_signal = fetch_df_from_postgres(symbol_name, settings.SIGNAL_TIMEFRAME, start_date, end_date)
            df_regime = fetch_df_from_postgres(symbol_name, settings.TREND_TIMEFRAME_SHORT, start_date, end_date)
            if df_signal is None or df_regime is None: return

            # 1. Generate the raw signal
            strategy_result = await trading_strategy.generate_signals(df_signal, df_regime)
            if not strategy_result: return

            print(f"  > AnalysisService: Generated signal for {symbol_name}: {strategy_result.get('overall_signal')}")

            # 2. [NEW] Pass the signal to the TradingService for a decision
            await self.trading_service.process_signal(db, symbol_record, strategy_result)

            # 3. (Optional) LLM analysis and DB logging can still happen here if needed,
            # for example, to log every signal generated, regardless of execution.
            # For now, we keep it simple.

        except Exception as e:
            print(f"  > An unexpected error occurred during analysis for {symbol_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            db.close()

    def get_all_analyses_from_db(self, db: Session, skip: int = 0, limit: int = 20) -> List[AnalysisResult]:
        """Fetches a paginated list of analysis results from the database."""
        return db.query(AnalysisResult).options(
            joinedload(AnalysisResult.symbol),
            joinedload(AnalysisResult.strategy)
        ).order_by(AnalysisResult.timestamp.desc()).offset(skip).limit(limit).all()

    async def close_llm_resources(self):
        await self.llm_strategy.close_clients()

