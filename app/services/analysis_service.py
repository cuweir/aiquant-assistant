import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session, joinedload
import asyncio
from .data_fetcher import data_fetcher_instance
from ..core.config import settings
from ..llm_providers import get_llm_strategy
from ..strategies.multi_indicator_strategy import MultiIndicatorStrategy
from ..db.session import SessionLocal
from ..db.models import AnalysisResult, Symbol, Strategy
from ..utils.formatters import format_price_dynamically


class AnalysisService:
    def __init__(self):
        self.data_fetcher = data_fetcher_instance
        self.llm_strategy = get_llm_strategy(settings)
        self.trading_strategy = MultiIndicatorStrategy()

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
                description="Multi-Indicator Scoring Strategy with ATR-based exits",
                config=config
            )
            db.add(strategy)
            db.commit()
            db.refresh(strategy)
        return strategy

    async def generate_comprehensive_analysis(
            self,
            symbol_name: str,
            signal_timeframe: str = settings.SIGNAL_TIMEFRAME,  # <-- Use config as default
            trend_timeframe_short: str = settings.TREND_TIMEFRAME_SHORT,  # <-- Use config as default
            trend_timeframe_long: str = settings.TREND_TIMEFRAME_LONG  # <-- Use config as default
    ) -> Dict[str, Any] | None:
        """
        Orchestration method for MTFA.
        Accepts optional timeframe overrides, otherwise uses defaults from settings.
        """
        limit_needed = 200

        df_signal_task = self.data_fetcher.fetch_ohlcv(symbol_name, signal_timeframe, limit=limit_needed)
        df_trend_short_task = self.data_fetcher.fetch_ohlcv(symbol_name, trend_timeframe_short, limit=limit_needed)
        df_trend_long_task = self.data_fetcher.fetch_ohlcv(symbol_name, trend_timeframe_long, limit=limit_needed)

        df_signal, df_trend_short, df_trend_long = await asyncio.gather(
            df_signal_task, df_trend_short_task, df_trend_long_task
        )

        if any(df is None or df.empty for df in [df_signal, df_trend_short, df_trend_long]):
            print(f"Could not fetch sufficient OHLCV data for all MTFA timeframes for {symbol_name}.")
            return None

        strategy_result = await self.trading_strategy.generate_signals(df_signal, df_trend_short, df_trend_long)
        if not strategy_result:
            print(f"Strategy failed to generate signals for {symbol_name}.")
            return None

        # Correctly define variables from strategy_result
        final_signal = strategy_result['overall_signal']
        original_score = strategy_result['total_score']
        current_price = strategy_result['current_price']

        price_str = format_price_dynamically(current_price)
        print(
            f"Analysis for {symbol_name} ({signal_timeframe}): Price={price_str}, Score={original_score:.1f}, Signal: {final_signal}")

        ai_suggestion = "AI analysis not triggered due to neutral or filtered signal."
        should_call_llm = final_signal in ["POTENTIAL_BUY", "POTENTIAL_SELL"]

        if should_call_llm:
            print(f"  Signal is valid. Querying LLM...")
            prompt = self._build_llm_prompt(symbol_name, signal_timeframe, strategy_result, df_signal)
            ai_suggestion = await self.llm_strategy.generate_analysis(prompt)

        db: Session = SessionLocal()
        try:
            strategy_config = {
                "atr_sl_multiplier": settings.ATR_STOP_LOSS_MULTIPLIER,
                "rr_ratio_tp1": settings.RISK_REWARD_RATIO_TP1,
                "rr_ratio_tp2": settings.RISK_REWARD_RATIO_TP2
            }
            symbol_record = self._get_or_create_symbol(db, symbol_name)
            strategy_record = self._get_or_create_strategy(db, "multi_indicator_v1.2_mtfa", strategy_config)

            db_current_price = float(current_price) if pd.notna(current_price) else None
            db_composite_score = float(original_score) if pd.notna(original_score) else None
            db_suggested_sl = float(strategy_result['suggested_sl']) if pd.notna(
                strategy_result['suggested_sl']) else None
            db_suggested_tp = float(strategy_result['suggested_tp']) if pd.notna(
                strategy_result['suggested_tp']) else None

            # Check if essential data is present before creating DB object
            if db_current_price is None or db_composite_score is None:
                print(f"Error: Price or Score is None for {symbol_name}, skipping DB save.")
                return None

            db_analysis_result = AnalysisResult(
                timestamp=pd.Timestamp.now(tz='UTC').to_pydatetime(),
                symbol_id=symbol_record.id,
                strategy_id=strategy_record.id,
                timeframe=signal_timeframe,
                current_price=db_current_price,
                composite_score=db_composite_score,
                overall_signal=final_signal,
                suggested_sl=db_suggested_sl,
                suggested_tp1=strategy_result['suggested_tp1'],
                suggested_tp=strategy_result['suggested_tp2'],
                llm_queried=should_call_llm,
                llm_analysis=ai_suggestion,
                indicator_details=strategy_result['signals_details']
            )

            db.add(db_analysis_result)
            db.commit()
            db.refresh(db_analysis_result)
            print(f"Analysis for {symbol_name} successfully saved to database.")

            return {
                "timestamp": db_analysis_result.timestamp,
                "symbol": symbol_name,
                "timeframe": signal_timeframe,
                "local_signal": final_signal,
                "price": float(current_price),
                "stop_loss": float(strategy_result['suggested_sl']) if strategy_result['suggested_sl'] else None,
                "take_profit_1": float(db_analysis_result.suggested_tp1) if db_analysis_result.suggested_tp1 else None,
                "take_profit": float(strategy_result['suggested_tp']) if strategy_result['suggested_tp'] else None,
                "ai_analysis": ai_suggestion,
                "rsi": next((d['value'] for d in strategy_result['signals_details'] if d['indicator'] == 'RSI'),
                            float('nan')),
                "details": {
                    "composite_score": original_score,
                    "individual_signals_details": strategy_result['signals_details']
                }
            }
        except Exception as e:
            print(f"Database Error: Failed to save analysis for {symbol_name}. Error: {e}")
            import traceback;
            traceback.print_exc()
            db.rollback()
            return None
        finally:
            db.close()

    def _build_llm_prompt(self, symbol: str, timeframe: str, result: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Builds a detailed and high-quality prompt for the LLM based on the strategy results.
        This method is now fully implemented.
        """
        price_str = format_price_dynamically(result['current_price'])

        # Build a summary of key contributing signals
        prompt_indicator_summary_for_llm = ""
        significant_signal_count = 0
        if result.get('signals_details'):
            for detail in result['signals_details']:
                if detail.get("score_change", 0) != 0:  # Only include signals that contributed to the score
                    if significant_signal_count < 4:  # Limit to ~4 key contributing signals for brevity
                        val_str = f"{detail['value']:.2f}" if isinstance(detail['value'], float) else str(
                            detail['value'])
                        prompt_indicator_summary_for_llm += f"          - {detail['indicator']} ({detail['signal']}): {val_str} (Score: {detail['score_change']:+})\n"
                        significant_signal_count += 1

        if not prompt_indicator_summary_for_llm:
            prompt_indicator_summary_for_llm = "          - No strong individual indicator signals detected.\n"

        # Build the main prompt string
        prompt = f"""
            Cryptocurrency Analysis Request for {symbol} ({timeframe}):
    
            **1. Core Signal Data:**
               - **Price:** {price_str}
               - **Calculated Signal:** {result['overall_signal']} (Total Score: {result['total_score']})
        """
        suggested_sl = result.get('suggested_sl')
        suggested_tp1 = result.get('suggested_tp1')
        suggested_tp2 = result.get('suggested_tp2')

        if suggested_sl is not None and suggested_tp2 is not None:  # Check for the final TP
            prompt += f"""           - **Suggested Stop Loss (SL):** {format_price_dynamically(suggested_sl)}
                   - **Suggested Take Profit 1 (TP1 @ 1R):** {format_price_dynamically(suggested_tp1)}
                   - **Suggested Take Profit 2 (TP2 @ {settings.RISK_REWARD_RATIO_TP2}R):** {format_price_dynamically(suggested_tp2)}
        """

        prompt += f"""
            **2. Key Contributing Indicator Signals:**
            {prompt_indicator_summary_for_llm}
            **3. Recent Market Data with Indicators (last 5 periods):**
               ```
               {df.iloc[-5:].to_string(float_format=lambda x: format_price_dynamically(x))}
               ```
    
            **AI Analyst Task:**
    
            You are a professional, data-driven crypto analyst. Your advice must be concise, actionable, and based *only* on the data provided.
    
            1.  **Assess the Signal:** Briefly evaluate the `Calculated Signal`. Is the score strong? Do the contributing indicators show clear alignment (confluence) or are there mixed signals (divergence)?
            2.  **Validate Exit Levels:** Review the `Suggested Stop Loss (SL)` and `Take Profit (TP1, TP2)`. Are they placed at logical levels? Provide your **final suggested SL and TP prices**.
            3.  **Provide Final Suggestion:** Based on everything, give a single, clear trading suggestion from this list:
                **[Strong Buy / Buy / Hold / Sell / Strong Sell / Avoid]**
            4.  **Justify:** In 1-2 sentences, explain *why* you made that suggestion.
            5.  **Identify Key Factor:** Mention the single most important risk to watch for OR a key confirmation that would strengthen the signal.
    
            **Format your response clearly using Markdown.**
        """
        return prompt.strip()

    def get_all_analyses_from_db(self, db: Session, skip: int = 0, limit: int = 20) -> List[AnalysisResult]:
        """Fetches a paginated list of analysis results from the database."""
        # 2. Use joinedload directly after importing it.
        return db.query(AnalysisResult).options(
            joinedload(AnalysisResult.symbol),
            joinedload(AnalysisResult.strategy)
        ).order_by(AnalysisResult.timestamp.desc()).offset(skip).limit(limit).all()

    async def close_llm_resources(self):
        await self.llm_strategy.close_clients()


analysis_service = AnalysisService()