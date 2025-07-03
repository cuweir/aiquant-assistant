from __future__ import annotations
import pandas as pd
import pandas_ta as ta
from .data_fetcher import data_fetcher_instance
from ..core.config import settings
from ..llm_providers import get_llm_strategy
from ..llm_providers.base import LLMStrategy

ai_analysis_cache = {}

def format_price_dynamically(price: float) -> str:
    """
    Dynamically formats the price based on its value to ensure appropriate precision.
    """
    if price is None or not isinstance(price, (int, float)) or pd.isna(price):
        return "N/A"
    
    if price >= 100:
        return f"{price:.2f}"  # e.g., 65432.10
    elif price >= 1:
        return f"{price:.3f}"  # e.g., 5.123
    elif price >= 0.01:
        return f"{price:.4f}"  # e.g., 0.5231
    else:
        return f"{price:.6f}"  # e.g., 0.001234

class TradingLogicService:
    def __init__(self):
        self.llm_strategy: LLMStrategy = get_llm_strategy(settings)
        print(f"TradingLogicService initialized with LLM Strategy: {type(self.llm_strategy).__name__}")

    async def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (This method remains unchanged from your last correct version) ...
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
        df[f'ATR_{settings.ATR_PERIOD}'] = df.ta.atr(length=settings.ATR_PERIOD)

        if 'volume' not in df.columns:
            df['volume'] = 0
        return df

    async def _get_indicator_signals_and_score(self, df_with_indicators: pd.DataFrame) -> tuple[list[dict], int]:
        """
        Analyzes the latest row of the DataFrame with indicators to generate signals and a composite score.
        Returns a list of dictionaries (each detailing a signal and its score contribution) and the total_score.
        """
        signals_details = []  # List to store dicts like {"indicator": "RSI", "signal": "OVERSOLD_BUY", "value": 25.0, "score_change": +1}
        total_score = 0

        if df_with_indicators.empty or len(df_with_indicators) < 2:
            print("Warning: Not enough data for signal/score calculation.")
            signals_details.append(
                {"indicator": "DataCheck", "signal": "NotEnoughData", "value": len(df_with_indicators),
                 "score_change": 0})
            return signals_details, total_score

        latest = df_with_indicators.iloc[-1]
        previous = df_with_indicators.iloc[-2]

        # 1. RSI Signal
        rsi_key = f'RSI_{settings.RSI_PERIOD}'
        rsi_value = latest.get(rsi_key)
        if rsi_value is not None and pd.notna(rsi_value):
            signal_text = "NEUTRAL"
            score_change = 0
            if rsi_value < settings.RSI_OVERSOLD:
                signal_text = "OVERSOLD_BUY"
                score_change = settings.WEIGHT_RSI_SIGNAL
            elif rsi_value > settings.RSI_OVERBOUGHT:
                signal_text = "OVERBOUGHT_SELL"
                score_change = -settings.WEIGHT_RSI_SIGNAL  # Negative for sell
            total_score += score_change
            signals_details.append(
                {"indicator": "RSI", "signal": signal_text, "value": rsi_value, "score_change": score_change})

        # 2. MACD Signal (Crossover)
        macd_line_key = f'MACD_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        signal_line_key = f'MACDS_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        latest_macd = latest.get(macd_line_key)
        latest_signal = latest.get(signal_line_key)
        prev_macd = previous.get(macd_line_key)
        prev_signal = previous.get(signal_line_key)

        if all(pd.notna(val) for val in [latest_macd, latest_signal, prev_macd, prev_signal]):
            signal_text = "NO_CROSS"
            score_change = 0
            if prev_macd < prev_signal and latest_macd > latest_signal:
                signal_text = "GOLDEN_CROSS_BUY"
                score_change = settings.WEIGHT_MACD_CROSS
            elif prev_macd > prev_signal and latest_macd < latest_signal:
                signal_text = "DEATH_CROSS_SELL"
                score_change = -settings.WEIGHT_MACD_CROSS  # Negative for sell
            total_score += score_change
            signals_details.append({"indicator": "MACD_Cross", "signal": signal_text,
                                    "value": f"MACD:{latest_macd:.2f},Sig:{latest_signal:.2f}",
                                    "score_change": score_change})

        # 3. MA Crossover Signal (EMA example)
        short_ma_key = f'EMA_{settings.MA_SHORT_PERIOD}'
        long_ma_key = f'EMA_{settings.MA_LONG_PERIOD}'
        latest_short_ma = latest.get(short_ma_key)
        latest_long_ma = latest.get(long_ma_key)
        prev_short_ma = previous.get(short_ma_key)
        prev_long_ma = previous.get(long_ma_key)

        if all(pd.notna(val) for val in [latest_short_ma, latest_long_ma, prev_short_ma, prev_long_ma]):
            signal_text = "NO_CROSS"
            score_change = 0
            if prev_short_ma < prev_long_ma and latest_short_ma > latest_long_ma:
                signal_text = "GOLDEN_CROSS_BUY"
                score_change = settings.WEIGHT_MA_CROSS
            elif prev_short_ma > prev_long_ma and latest_short_ma < latest_long_ma:
                signal_text = "DEATH_CROSS_SELL"
                score_change = -settings.WEIGHT_MA_CROSS  # Negative for sell
            total_score += score_change
            signals_details.append({"indicator": "MA_Cross", "signal": signal_text,
                                    "value": f"S:{latest_short_ma:.2f},L:{latest_long_ma:.2f}",
                                    "score_change": score_change})

        # 4. Bollinger Bands Signal
        price_close = latest.get('close')
        bbu_key = f'BBU_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}'
        bbl_key = f'BBL_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}'
        latest_bbu = latest.get(bbu_key)
        latest_bbl = latest.get(bbl_key)

        if price_close is not None and pd.notna(price_close) and \
                all(pd.notna(val) for val in [latest_bbu, latest_bbl]):
            signal_text = "INSIDE_BANDS"
            score_change = 0
            if price_close > latest_bbu:
                signal_text = "BREAK_UPPER"
                score_change = settings.WEIGHT_BBANDS_BREAKOUT  # Assumed positive for breakout
            elif price_close < latest_bbl:
                signal_text = "BREAK_LOWER"
                score_change = -settings.WEIGHT_BBANDS_BREAKOUT  # Assumed negative for breakdown
            total_score += score_change
            signals_details.append({"indicator": "BollingerBands", "signal": signal_text,
                                    "value": f"P:{format_price_dynamically(price_close)},U:{format_price_dynamically(latest_bbu)},L:{format_price_dynamically(latest_bbl)}",
                                    "score_change": score_change})

        # 5. Volume Confirmation (Informational, not directly adding to score in this simplified version)
        volume_latest = latest.get('volume')
        if volume_latest is not None and pd.notna(volume_latest):
            avg_volume_period = 5
            signal_text = "VOLUME_DATA_UNAVAILABLE"  # Default
            vol_value_str = f"Vol:{volume_latest:.2f}"
            if len(df_with_indicators) > avg_volume_period + 1:  # +1 because we look at [-avg-1:-1]
                avg_volume = df_with_indicators['volume'].iloc[-(avg_volume_period + 1):-1].mean()
                if pd.notna(avg_volume) and avg_volume > 0:
                    vol_value_str = f"Vol:{volume_latest:.2f},Avg5P:{avg_volume:.2f}"
                    if volume_latest > avg_volume * 1.5:
                        signal_text = "HIGH_VOLUME"
                    else:
                        signal_text = "NORMAL_OR_LOW_VOLUME"
                else:  # avg_volume is NaN or 0
                    signal_text = "VOLUME_AVG_NOT_CALCULABLE"
            else:  # Not enough data for average
                signal_text = "NOT_ENOUGH_DATA_FOR_VOLUME_AVG"
            signals_details.append({"indicator": "Volume", "signal": signal_text, "value": vol_value_str,
                                    "score_change": 0})  # Score change is 0 for volume here

        return signals_details, total_score

    async def generate_comprehensive_analysis(self, symbol: str, timeframe: str) -> dict | None:
        limit_needed = max(settings.MACD_SLOW_PERIOD, settings.MA_LONG_PERIOD, settings.BBANDS_PERIOD, settings.RSI_PERIOD, settings.ATR_PERIOD) + 50
        df_ohlcv = await data_fetcher_instance.fetch_ohlcv(symbol, timeframe, limit=limit_needed)

        if df_ohlcv is None or df_ohlcv.empty or len(df_ohlcv) < limit_needed - 40:
            print(f"Not enough OHLCV data for {symbol} ({timeframe})...")
            return None

        df_with_indicators = await self._calculate_indicators(df_ohlcv.copy())
        if df_with_indicators.empty or len(df_with_indicators.iloc[-1].dropna()) < 5:  # Check if latest row has values
            print(f"Could not calculate sufficient indicators for {symbol} ({timeframe})...")
            return None

        # This now returns a list of dicts and the total_score
        individual_signals_details, total_score = await self._get_indicator_signals_and_score(df_with_indicators)

        latest_candle = df_with_indicators.iloc[-1]
        current_price = latest_candle.get('close')
        current_atr = latest_candle.get(f'ATR_{settings.ATR_PERIOD}')
        if current_price is None or pd.isna(current_price):
            print(f"Error: Current price is missing or NaN for {symbol} ({timeframe}). Skipping analysis.")
            return None

        overall_signal_label = "HOLD_OBSERVE_NEUTRAL_SCORE"
        ai_suggestion = "AI analysis not triggered due to neutral local score or error."
        suggested_sl = None
        suggested_tp = None
        should_call_llm = False

        if total_score >= settings.BUY_SCORE_THRESHOLD:
            overall_signal_label = "POTENTIAL_BUY"
            should_call_llm = True
        elif total_score <= settings.SELL_SCORE_THRESHOLD:
            overall_signal_label = "POTENTIAL_SELL"
            should_call_llm = True
            # Use the new helper function for logging
            price_str = format_price_dynamically(current_price)
            print(
                f"Analysis for {symbol} ({timeframe}): Price={price_str}, Total Score={total_score}, Calculated Signal: {overall_signal_label}")
            print("  Detailed Indicator Signals & Score Contributions:")
            if individual_signals_details:
                for signal_detail in individual_signals_details:
                    indicator = signal_detail.get("indicator", "N/A")
                    signal = signal_detail.get("signal", "N/A")
                    value_info = signal_detail.get("value", "N/A")
                    score_chg = signal_detail.get("score_change", 0)
                    if isinstance(value_info, float):
                        value_str = f"{value_info:.2f}"
                    else:
                        value_str = str(value_info)
                    print(f"    - {indicator}: Signal='{signal}', Value(s)='{value_str}', Score Change={score_chg:+}")
            else:
                print("    No individual signals details generated.")

        if should_call_llm:
            if current_atr is not None and pd.notna(current_atr) and current_atr > 0:
                if overall_signal_label == "POTENTIAL_BUY":
                    suggested_sl = current_price - (settings.ATR_STOP_LOSS_MULTIPLIER * current_atr)
                    suggested_tp = current_price + (settings.ATR_TAKE_PROFIT_MULTIPLIER * current_atr)
                elif overall_signal_label == "POTENTIAL_SELL":
                    suggested_sl = current_price + (settings.ATR_STOP_LOSS_MULTIPLIER * current_atr)
                    suggested_tp = current_price - (settings.ATR_TAKE_PROFIT_MULTIPLIER * current_atr)
                print(f"  ATR-based exit levels calculated: SL={suggested_sl:.2f}, TP={suggested_tp:.2f} (ATR={current_atr:.2f})")
            else:
                print("  Warning: Could not calculate ATR-based exit levels (ATR value is missing, NaN, or zero).")

            print(f"  Local score ({total_score}) met threshold. Querying LLM for {symbol}...")
            ohlcv_indicator_summary = df_with_indicators.iloc[-5:].to_string(float_format=lambda x: format_price_dynamically(x) if x > 0.00001 else f"{x:.8f}")

            prompt_indicator_summary_for_llm = ""
            significant_signal_count = 0
            for detail in individual_signals_details:
                if detail.get("score_change", 0) != 0:  # Only include signals that contributed to the score
                    if significant_signal_count < 4:  # Limit to ~4 key contributing signals for brevity
                        val_str = f"{detail['value']:.2f}" if isinstance(detail['value'], float) else str(
                            detail['value'])
                        prompt_indicator_summary_for_llm += f"          - {detail['indicator']} ({detail['signal']}): {val_str} (Score: {detail['score_change']:+})\n"
                        significant_signal_count += 1
            if not prompt_indicator_summary_for_llm:
                prompt_indicator_summary_for_llm = "          - No strong contributing indicator signals.\n"

            prompt = f"""
            Cryptocurrency Analysis Request for {symbol} ({timeframe}):

            Key Data:
            - Price: {format_price_dynamically(current_price)}
            - Calculated Signal: {overall_signal_label} (Total Score: {total_score})
            """
            if suggested_sl is not None and suggested_tp is not None:
                prompt += f"""- Suggested Stop Loss (SL): {suggested_sl:.2f} (based on {settings.ATR_STOP_LOSS_MULTIPLIER} * ATR)
                                - Suggested Take Profit (TP): {suggested_tp:.2f} (based on {settings.ATR_TAKE_PROFIT_MULTIPLIER} * ATR)
            """
            prompt += f"""
            - Key Contributing Indicator Signals:
            {prompt_indicator_summary_for_llm}
            Recent Market Data with Indicators (last 5 periods):
            {ohlcv_indicator_summary}

            AI Analyst Task:
            1. Briefly assess the `Calculated Signal` ({overall_signal_label}, Score: {total_score}) considering the key contributing indicators.
            2. Validate or adjust the suggested Stop Loss and Take Profit levels. Are they reasonable given the chart context (e.g., recent support/resistance)? Provide your final suggested SL and TP prices.
               [Strong Buy / Buy / Hold / Sell / Strong Sell / Avoid]
            3. Based on ALL data, provide a VERY CONCISE trading suggestion:
            4. Give a 1-2 sentence justification for your suggestion, focusing on the most critical factors.
            5. Mention 1 key risk OR 1 key confirmation to watch.

            TARGET OUTPUT LENGTH: Under 180 words. Be extremely brief and direct.
            """
            ai_suggestion = await self.llm_strategy.generate_analysis(prompt)
        else:
            print(f"  Local score ({total_score}) is neutral or below threshold for {symbol}. Skipping LLM query.")

        # Extract RSI_VALUE from the list of dicts for the AIAnalysisOutput schema
        rsi_val_for_output = float('nan')
        for detail in individual_signals_details:
            if detail.get("indicator") == "RSI" and "value" in detail:
                rsi_val_for_output = detail["value"]
                break

        cache_key = f"{symbol}_{timeframe}_COMPOSITE"
        analysis_data = {
            "timestamp": pd.Timestamp.now(tz='UTC'),
            "symbol": symbol,
            "timeframe": timeframe,
            "local_signal": overall_signal_label,
            "rsi": rsi_val_for_output,  # Use the extracted RSI value
            "price": current_price,
            "ai_analysis": ai_suggestion,
            "stop_loss": suggested_sl,
            "take_profit": suggested_tp,
            "details": {  # This is for internal caching, not necessarily for API response unless schema is updated
                "composite_score": total_score,
                "individual_signals_details": individual_signals_details,  # Store the new detailed list
                "llm_queried": should_call_llm
            }
        }
        ai_analysis_cache[cache_key] = analysis_data
        print(
            f"Comprehensive AI Analysis generation attempt finished for {symbol} ({timeframe}). LLM Queried: {should_call_llm}")
        return analysis_data

    # get_cached_analysis, get_all_cached_analyses, close_llm_resources remain the same
    async def get_cached_analysis(self, symbol: str, timeframe: str):
        cache_key = f"{symbol}_{timeframe}_COMPOSITE"
        return ai_analysis_cache.get(cache_key)

    async def get_all_cached_analyses(self):
        return ai_analysis_cache

    async def close_llm_resources(self):
        if hasattr(self.llm_strategy, 'close_clients') and callable(self.llm_strategy.close_clients):
            await self.llm_strategy.close_clients()
        else:
            print(f"LLM strategy {type(self.llm_strategy).__name__} does not have a callable 'close_clients' method.")


trading_logic_service_instance = TradingLogicService()