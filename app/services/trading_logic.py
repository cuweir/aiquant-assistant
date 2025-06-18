from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # For technical indicators like RSI
# Make sure this import matches your file structure for data_fetcher
# If data_fetcher.py is in the same 'services' directory:
from .data_fetcher import data_fetcher_instance
# If llm_client.py is in 'core' directory:
from ..core.llm_client import llm_client_instance  # This was your original import
from ..core.config import settings
import datetime  # For consistent timestamping if needed, pandas uses it too.

# --- Storage for recent AI analysis (in-memory for simplicity) ---
ai_analysis_cache = {}  # Key: symbol_timeframe_signaltype, Value: analysis dict


class TradingLogicService:
    # No __init__ needed in your version as llm_client_instance is global

    async def calculate_rsi(self, df: pd.DataFrame, period: int = settings.RSI_PERIOD) -> pd.Series | None:
        if df is not None and 'close' in df.columns and len(df) >= period:
            return df.ta.rsi(length=period)
        return None

    async def check_rsi_signal(self, symbol: str, timeframe: str) -> dict | None:
        """
        Checks for RSI overbought/oversold conditions.
        If a condition is met, it prepares data and a prompt for LLM analysis.
        """
        print(
            f"[{pd.Timestamp.now(tz='UTC')}] INFO: Checking signal for {symbol} on timeframe {timeframe}")  # Added log

        df = await data_fetcher_instance.fetch_ohlcv(symbol, timeframe, limit=100 + settings.RSI_PERIOD)
        if df is None or df.empty:
            print(
                f"[{pd.Timestamp.now(tz='UTC')}] WARNING: No OHLCV data fetched for {symbol} on {timeframe}.")  # Changed to print WARNING
            return None

        rsi_series = await self.calculate_rsi(df)
        if rsi_series is None or rsi_series.empty:
            print(
                f"[{pd.Timestamp.now(tz='UTC')}] WARNING: Could not calculate RSI for {symbol} on {timeframe}.")  # Changed to print WARNING
            return None

        latest_rsi = rsi_series.iloc[-1]
        signal_type = None
        current_price = df['close'].iloc[-1]

        print(
            f"[{pd.Timestamp.now(tz='UTC')}] INFO: Data for {symbol} ({timeframe}): Latest RSI = {latest_rsi:.2f}, Price = {current_price:.4f}")  # Added log

        if latest_rsi > settings.RSI_OVERBOUGHT:
            signal_type = "RSI_OVERBOUGHT"
        elif latest_rsi < settings.RSI_OVERSOLD:
            signal_type = "RSI_OVERSOLD"
        else:  # Explicitly log when no signal is found
            print(
                f"[{pd.Timestamp.now(tz='UTC')}] INFO: No RSI signal for {symbol} ({timeframe}). RSI {latest_rsi:.2f} is within thresholds ({settings.RSI_OVERSOLD} - {settings.RSI_OVERBOUGHT}).")

        if signal_type:
            print(
                f"[{pd.Timestamp.now(tz='UTC')}] INFO: Local signal triggered: {symbol} ({timeframe}) - {signal_type} at RSI {latest_rsi:.2f}, Price: {current_price:.4f}")

            ohlcv_summary = df.iloc[-10:].to_string()

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
            # Using llm_client_instance directly as per your original structure
            ai_suggestion = await llm_client_instance.generate_analysis(prompt)

            if ai_suggestion.lower().startswith("error:"):
                print(
                    f"[{pd.Timestamp.now(tz='UTC')}] ERROR: LLM failed to generate analysis for {symbol} ({timeframe}), Signal: {signal_type}. LLM Response: {ai_suggestion}")
                # Decide if you want to cache this error or return None.
                # If you return None here, main.py will raise the 404.
                # return None # <--- Uncomment if you want LLM failure to result in the 404 from client

            cache_key = f"{symbol}_{timeframe}_{signal_type}"
            # Ensure consistent timestamping for the cache
            timestamp_now = pd.Timestamp.now(tz='UTC')
            analysis_data = {
                "timestamp": timestamp_now,  # Use a consistent, timezone-aware timestamp
                "local_signal": signal_type,
                "rsi": latest_rsi,
                "price": current_price,
                "prompt": prompt,
                "ai_analysis": ai_suggestion,
                "symbol": symbol,  # Adding these as they are useful for the response model
                "timeframe": timeframe
            }
            ai_analysis_cache[cache_key] = analysis_data
            print(
                f"[{pd.Timestamp.now(tz='UTC')}] INFO: AI analysis completed and cached for {symbol} ({timeframe}), Signal: {signal_type}.")
            return analysis_data

        # This part is reached if signal_type remained None
        return None

    async def get_cached_analysis(self, symbol: str, timeframe: str, signal_type: str):
        cache_key = f"{symbol}_{timeframe}_{signal_type}"
        return ai_analysis_cache.get(cache_key)

    async def get_all_cached_analyses(self):
        return ai_analysis_cache


# Global instance
trading_logic_service_instance = TradingLogicService()