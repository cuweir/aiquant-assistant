from __future__ import annotations

import pandas as pd
import pandas_ta as ta # For technical indicators like RSI
from .data_fetcher import data_fetcher_instance
from ..core.llm_client import llm_client_instance
from ..core.config import settings

# --- Storage for recent AI analysis (in-memory for simplicity) ---
# In a real app, this would be a database or a more robust cache.
ai_analysis_cache = {} # Key: symbol_timeframe_signaltype, Value: analysis text

class TradingLogicService:
    async def calculate_rsi(self, df: pd.DataFrame, period: int = settings.RSI_PERIOD) -> pd.Series | None:
        if df is not None and 'close' in df.columns and len(df) >= period:
            return df.ta.rsi(length=period)
        return None

    async def check_rsi_signal(self, symbol: str, timeframe: str) -> dict | None:
        """
        Checks for RSI overbought/oversold conditions.
        If a condition is met, it prepares data and a prompt for LLM analysis.
        """
        df = await data_fetcher_instance.fetch_ohlcv(symbol, timeframe, limit=100 + settings.RSI_PERIOD)
        if df is None or df.empty:
            return None

        rsi_series = await self.calculate_rsi(df)
        if rsi_series is None or rsi_series.empty:
            return None

        latest_rsi = rsi_series.iloc[-1]
        signal_type = None
        current_price = df['close'].iloc[-1]

        if latest_rsi > settings.RSI_OVERBOUGHT:
            signal_type = "RSI_OVERBOUGHT"
        elif latest_rsi < settings.RSI_OVERSOLD:
            signal_type = "RSI_OVERSOLD"

        if signal_type:
            print(f"Local signal triggered: {symbol} ({timeframe}) - {signal_type} at RSI {latest_rsi:.2f}, Price: {current_price}")

            # Prepare data for LLM
            ohlcv_summary = df.iloc[-10:].to_string() # Last 10 candles as string for LLM

            prompt = f"""
            Analyze the following cryptocurrency signal for {symbol} on the {timeframe} timeframe:
            Signal Type: {signal_type}
            Current RSI({settings.RSI_PERIOD}): {latest_rsi:.2f}
            Current Price: {current_price}
            Recent Market Data (last 10 periods):
            {ohlcv_summary}

            Task:
            1. As a professional crypto analyst, assess the reliability of this {signal_type} signal.
            2. Is this a good opportunity to consider a trade (long for oversold, short for overbought), or should one wait for more confirmation?
            3. What are the key factors (e.g., volume, price action context, broader market sentiment if you can infer) that support or contradict this signal?
            4. Provide a concise recommendation: [Enter Long / Consider Short / Hold & Observe / Ignore Signal]
            5. Briefly state your reasoning and any immediate risks.

            Format your response clearly.
            """
            ai_suggestion = await llm_client_instance.generate_analysis(prompt)
            cache_key = f"{symbol}_{timeframe}_{signal_type}"
            ai_analysis_cache[cache_key] = {
                "timestamp": pd.Timestamp.now(),
                "local_signal": signal_type,
                "rsi": latest_rsi,
                "price": current_price,
                "prompt": prompt, # For debugging
                "ai_analysis": ai_suggestion
            }
            return ai_analysis_cache[cache_key]
        return None

    async def get_cached_analysis(self, symbol: str, timeframe: str, signal_type: str):
        cache_key = f"{symbol}_{timeframe}_{signal_type}"
        return ai_analysis_cache.get(cache_key)

    async def get_all_cached_analyses(self):
        return ai_analysis_cache

# Global instance
trading_logic_service_instance = TradingLogicService()