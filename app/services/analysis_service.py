import pandas as pd
from typing import Dict, Any

from .data_fetcher import data_fetcher_instance
from ..core.config import settings
from ..llm_providers import get_llm_strategy
from ..strategies.multi_indicator_strategy import MultiIndicatorStrategy
from ..utils.formatters import format_price_dynamically

# This will be replaced by a database layer later.
analysis_cache: Dict[str, Any] = {}


class AnalysisService:
    def __init__(self):
        self.data_fetcher = data_fetcher_instance
        self.llm_strategy = get_llm_strategy(settings)
        # In a more complex system, you might choose the strategy based on config
        self.trading_strategy = MultiIndicatorStrategy()

    async def generate_comprehensive_analysis(self, symbol: str, timeframe: str) -> dict | None:
        # 1. Fetch data
        limit_needed = 200  # A safe buffer for indicators
        df_ohlcv = await self.data_fetcher.fetch_ohlcv(symbol, timeframe, limit=limit_needed)
        if df_ohlcv is None or df_ohlcv.empty:
            print(f"No OHLCV data for {symbol} ({timeframe}).")
            return None

        # 2. Generate signals from strategy
        strategy_result = await self.trading_strategy.generate_signals(df_ohlcv)
        if not strategy_result:
            print(f"Strategy failed to generate signals for {symbol} ({timeframe}).")
            return None

        # 3. Log the technical analysis result
        price_str = format_price_dynamically(strategy_result['current_price'])
        print(
            f"Analysis for {symbol} ({timeframe}): Price={price_str}, Score={strategy_result['total_score']}, Signal: {strategy_result['overall_signal']}")
        for detail in strategy_result['signals_details']:
            score_chg = detail.get('score_change', 0)
            if score_chg != 0:  # Only log signals that contributed to the score
                print(f"    - {detail['indicator']} ({detail['signal']}): Score Change={score_chg:+}")

        # 4. Decide whether to call LLM
        ai_suggestion = "AI analysis not triggered due to neutral local score."
        should_call_llm = strategy_result['overall_signal'] in ["POTENTIAL_BUY", "POTENTIAL_SELL"]

        if should_call_llm:
            print(f"  Local score met threshold. Querying LLM...")
            prompt = self._build_llm_prompt(symbol, timeframe, strategy_result, df_ohlcv)
            ai_suggestion = await self.llm_strategy.generate_analysis(prompt)
        else:
            print(f"  Local score is neutral. Skipping LLM query.")

        # 5. Assemble final analysis data and cache it
        final_analysis = {
            "timestamp": pd.Timestamp.now(tz='UTC'),
            "symbol": symbol, "timeframe": timeframe,
            "local_signal": strategy_result['overall_signal'],
            "price": strategy_result['current_price'],
            "stop_loss": strategy_result['suggested_sl'],
            "take_profit": strategy_result['suggested_tp'],
            "ai_analysis": ai_suggestion,
            "rsi": next((d['value'] for d in strategy_result['signals_details'] if d['indicator'] == 'RSI'),
                        float('nan')),
            "details": {
                "composite_score": strategy_result['total_score'],
                "individual_signals_details": strategy_result['signals_details'],
                "llm_queried": should_call_llm
            }
        }

        cache_key = f"{symbol}_{timeframe}_COMPOSITE"
        analysis_cache[cache_key] = final_analysis
        print(f"Analysis generation finished for {symbol} ({timeframe}). LLM Queried: {should_call_llm}")
        return final_analysis

    def _build_llm_prompt(self, symbol, timeframe, result, df) -> str:
        # Logic to build the prompt, extracted for clarity
        # This is essentially the same prompt building logic as before.
        # ...
        return "This is a generated prompt based on strategy results."

    async def get_all_cached_analyses(self) -> dict:
        return analysis_cache

    async def close_llm_resources(self):
        await self.llm_strategy.close_clients()


# Global instance for dependency injection
analysis_service = AnalysisService()