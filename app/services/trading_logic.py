from __future__ import annotations
import pandas as pd
import pandas_ta as ta
from .data_fetcher import data_fetcher_instance
from ..core.config import settings  # Import the global settings instance
from ..llm_providers import get_llm_strategy  # Import the factory function
from ..llm_providers.base import LLMStrategy  # Import the base strategy type

ai_analysis_cache = {}


class TradingLogicService:
    def __init__(self):
        # Use the factory to get the LLM strategy instance based on config
        self.llm_strategy: LLMStrategy = get_llm_strategy(settings)  # Pass the settings instance
        # The print statement below should now reflect the type from the factory
        print(f"TradingLogicService initialized with LLM Strategy: {type(self.llm_strategy).__name__}")

    async def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (no changes here, keep your indicator calculation logic) ...
        if df is None or df.empty:
            return pd.DataFrame()

        df[f'RSI_{settings.RSI_PERIOD}'] = df.ta.rsi(length=settings.RSI_PERIOD)
        macd = df.ta.macd(fast=settings.MACD_FAST_PERIOD, slow=settings.MACD_SLOW_PERIOD,
                          signal=settings.MACD_SIGNAL_PERIOD)
        if macd is not None and not macd.empty:
            df[
                f'MACD_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'] = macd.iloc[
                                                                                                                 :, 0]
            df[
                f'MACDH_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'] = macd.iloc[
                                                                                                                  :, 1]
            df[
                f'MACDS_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'] = macd.iloc[
                                                                                                                  :, 2]
        df[f'EMA_{settings.MA_SHORT_PERIOD}'] = df.ta.ema(length=settings.MA_SHORT_PERIOD)
        df[f'EMA_{settings.MA_LONG_PERIOD}'] = df.ta.ema(length=settings.MA_LONG_PERIOD)
        bbands = df.ta.bbands(length=settings.BBANDS_PERIOD, std=settings.BBANDS_STD_DEV)
        if bbands is not None and not bbands.empty:
            df[f'BBL_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}'] = bbands.iloc[:, 0]
            df[f'BBM_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}'] = bbands.iloc[:, 1]
            df[f'BBU_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}'] = bbands.iloc[:, 2]
        if 'volume' not in df.columns:
            df['volume'] = 0
        return df

    async def _get_indicator_signals_and_score(self, df_with_indicators: pd.DataFrame) -> tuple[dict, int]:
        # ... (no changes here, keep your scoring logic) ...
        signals = {}
        total_score = 0
        if df_with_indicators.empty or len(df_with_indicators) < 2:
            print("Warning: Not enough data for signal/score calculation (need at least 2 rows for previous candle).")
            return signals, total_score  # Return empty signals and zero score

        latest = df_with_indicators.iloc[-1]
        previous = df_with_indicators.iloc[-2]

        rsi_key = f'RSI_{settings.RSI_PERIOD}'
        if rsi_key in latest and pd.notna(latest[rsi_key]):
            signals['RSI_VALUE'] = latest[rsi_key]
            if latest[rsi_key] < settings.RSI_OVERSOLD:
                signals['RSI_SIGNAL'] = "OVERSOLD_BUY"
                total_score += settings.WEIGHT_RSI_SIGNAL
            elif latest[rsi_key] > settings.RSI_OVERBOUGHT:
                signals['RSI_SIGNAL'] = "OVERBOUGHT_SELL"
                total_score -= settings.WEIGHT_RSI_SIGNAL
            else:
                signals['RSI_SIGNAL'] = "NEUTRAL"

        macd_line_key = f'MACD_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        signal_line_key = f'MACDS_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        if all(k in latest and pd.notna(latest[k]) for k in [macd_line_key, signal_line_key]) and \
                all(k in previous and pd.notna(previous[k]) for k in [macd_line_key, signal_line_key]):
            signals['MACD_LINE'] = latest[macd_line_key]
            signals['MACD_SIGNAL_LINE'] = latest[signal_line_key]
            if previous[macd_line_key] < previous[signal_line_key] and latest[macd_line_key] > latest[signal_line_key]:
                signals['MACD_CROSS'] = "GOLDEN_CROSS_BUY"
                total_score += settings.WEIGHT_MACD_CROSS
            elif previous[macd_line_key] > previous[signal_line_key] and latest[macd_line_key] < latest[
                signal_line_key]:
                signals['MACD_CROSS'] = "DEATH_CROSS_SELL"
                total_score -= settings.WEIGHT_MACD_CROSS
            else:
                signals['MACD_CROSS'] = "NO_CROSS"

        short_ma_key = f'EMA_{settings.MA_SHORT_PERIOD}'
        long_ma_key = f'EMA_{settings.MA_LONG_PERIOD}'
        if all(k in latest and pd.notna(latest[k]) for k in [short_ma_key, long_ma_key]) and \
                all(k in previous and pd.notna(previous[k]) for k in [short_ma_key, long_ma_key]):
            signals[short_ma_key] = latest[short_ma_key]
            signals[long_ma_key] = latest[long_ma_key]
            if previous[short_ma_key] < previous[long_ma_key] and latest[short_ma_key] > latest[long_ma_key]:
                signals['MA_CROSS'] = "GOLDEN_CROSS_BUY"
                total_score += settings.WEIGHT_MA_CROSS
            elif previous[short_ma_key] > previous[long_ma_key] and latest[short_ma_key] < latest[long_ma_key]:
                signals['MA_CROSS'] = "DEATH_CROSS_SELL"
                total_score -= settings.WEIGHT_MA_CROSS
            else:
                signals['MA_CROSS'] = "NO_CROSS"

        price_close = latest.get('close')
        bbu_key = f'BBU_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}'
        bbl_key = f'BBL_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}'
        if price_close is not None and pd.notna(price_close) and \
                all(k in latest and pd.notna(latest[k]) for k in [bbu_key, bbl_key]):
            signals['BB_UPPER'] = latest[bbu_key]
            signals['BB_LOWER'] = latest[bbl_key]
            if price_close > latest[bbu_key]:
                signals['BB_SIGNAL'] = "BREAK_UPPER"
                total_score += settings.WEIGHT_BBANDS_BREAKOUT
            elif price_close < latest[bbl_key]:
                signals['BB_SIGNAL'] = "BREAK_LOWER"
                total_score -= settings.WEIGHT_BBANDS_BREAKOUT
            else:
                signals['BB_SIGNAL'] = "INSIDE_BANDS"

        volume_latest = latest.get('volume')
        if volume_latest is not None and pd.notna(volume_latest):
            avg_volume_period = 5
            if len(df_with_indicators) > avg_volume_period:
                avg_volume = df_with_indicators['volume'].iloc[-avg_volume_period:-1].mean()
                if pd.notna(avg_volume):  # Ensure avg_volume is not NaN
                    signals['VOLUME'] = volume_latest
                    signals['AVG_VOLUME_5'] = avg_volume
                    if volume_latest > avg_volume * 1.5:
                        signals['VOLUME_SIGNAL'] = "HIGH_VOLUME"
                    else:
                        signals['VOLUME_SIGNAL'] = "NORMAL_OR_LOW_VOLUME"
        return signals, total_score

    async def generate_comprehensive_analysis(self, symbol: str, timeframe: str) -> dict | None:
        # ... (logic for fetching data, calculating indicators, getting score - no change) ...
        # Ensure this part uses `self.llm_strategy`
        limit_needed = max(settings.MACD_SLOW_PERIOD, settings.MA_LONG_PERIOD, settings.BBANDS_PERIOD,
                           settings.RSI_PERIOD) + 50
        df_ohlcv = await data_fetcher_instance.fetch_ohlcv(symbol, timeframe, limit=limit_needed)

        if df_ohlcv is None or df_ohlcv.empty or len(df_ohlcv) < limit_needed - 40:
            print(
                f"Not enough OHLCV data for {symbol} ({timeframe}). Required: ~{limit_needed - 40}, Got: {len(df_ohlcv) if df_ohlcv is not None else 0}")
            return None

        df_with_indicators = await self._calculate_indicators(df_ohlcv.copy())
        if df_with_indicators.empty or len(df_with_indicators.iloc[-1].dropna()) < 5:
            print(f"Could not calculate sufficient indicators for {symbol} ({timeframe}).")
            return None

        individual_signals, total_score = await self._get_indicator_signals_and_score(df_with_indicators)

        latest_candle = df_with_indicators.iloc[-1]
        current_price = latest_candle.get('close')
        if current_price is None or pd.isna(current_price):
            print(f"Error: Current price is missing for {symbol} ({timeframe}). Skipping analysis.")
            return None

        overall_signal_label = "HOLD_OBSERVE"
        if total_score >= settings.BUY_SCORE_THRESHOLD:
            overall_signal_label = "POTENTIAL_BUY"
        elif total_score <= settings.SELL_SCORE_THRESHOLD:
            overall_signal_label = "POTENTIAL_SELL"

        print(
            f"Analysis for {symbol} ({timeframe}): Price={current_price:.4f}, Score={total_score}, Signal: {overall_signal_label}")
        # print(f"Individual Signals: {individual_signals}") # Can be verbose

        ohlcv_indicator_summary = df_with_indicators.iloc[-10:].to_string(float_format="%.4f")
        prompt = f"""
        Cryptocurrency Analysis Request for {symbol} ({timeframe}):

        Current Market Data:
        - Price: {current_price:.4f}
        - Overall Calculated Signal: {overall_signal_label} (based on a composite score of {total_score})
        - Individual Indicator States:
        """
        for k, v in individual_signals.items():
            if isinstance(v, float):
                v_str = f"{v:.4f}"
            else:
                v_str = str(v)
            prompt += f"          - {k}: {v_str}\n"
        prompt += f"""
        Recent Market Data with Indicators (last 10 periods):
        {ohlcv_indicator_summary}

        Task for AI Crypto Analyst:
        1. Review the overall calculated signal ({overall_signal_label}) and the composite score ({total_score}).
        2. Examine the individual indicator states. Are there any strong confluences or divergences?
        3. Based on all the provided data (current price, signals, recent history), provide a concise trading suggestion:
           [Strong Buy / Buy / Hold & Observe / Sell / Strong Sell / Risky - Avoid]
        4. Briefly explain your reasoning, highlighting the most influential factors and any potential risks or confirmations to watch for (e.g., volume patterns, upcoming news if inferable - though focus on provided TA).
        5. If 'Hold & Observe', specify what key changes or confirmations would shift your view.

        Please be concise and actionable.
        """
        # CRITICAL: Use the llm_strategy instance
        ai_suggestion = await self.llm_strategy.generate_analysis(prompt)

        cache_key = f"{symbol}_{timeframe}_COMPOSITE"
        analysis_data = {
            "timestamp": pd.Timestamp.now(tz='UTC'),
            "symbol": symbol,
            "timeframe": timeframe,
            "local_signal": overall_signal_label,
            "rsi": individual_signals.get('RSI_VALUE', float('nan')),
            "price": current_price,
            "ai_analysis": ai_suggestion,
            "details": {
                "composite_score": total_score,
                "individual_signals": individual_signals,
            }
        }
        ai_analysis_cache[cache_key] = analysis_data
        return analysis_data

    async def get_cached_analysis(self, symbol: str, timeframe: str):
        cache_key = f"{symbol}_{timeframe}_COMPOSITE"
        return ai_analysis_cache.get(cache_key)

    async def get_all_cached_analyses(self):
        return ai_analysis_cache

    async def close_llm_resources(self):
        """Method to call during application shutdown to close LLM strategy resources."""
        if hasattr(self.llm_strategy, 'close_clients') and callable(self.llm_strategy.close_clients):
            await self.llm_strategy.close_clients()
        else:
            print(f"LLM strategy {type(self.llm_strategy).__name__} does not have a callable 'close_clients' method.")


trading_logic_service_instance = TradingLogicService()  # Instance created here