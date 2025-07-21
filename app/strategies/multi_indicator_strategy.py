import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List, Tuple

from .base_strategy import TradingStrategy
from ..core.config import settings
from ..utils.formatters import format_price_dynamically


class MultiIndicatorStrategy(TradingStrategy):
    """
    A strategy that combines multiple technical indicators (RSI, MACD, MA, BBands, Volume)
    to generate a composite score and trading signals.
    """

    async def generate_signals(self, df_signal: pd.DataFrame, df_trend: pd.DataFrame) -> Dict[str, Any]:
        """
        Public method to generate signals. It orchestrates the internal calculation
        and scoring logic.
        """
        # 1. Determine the long-term trend from the trend DataFrame
        long_term_trend, trend_details = self._get_long_term_trend(df_trend)

        # 2. Calculate indicators and score on the signal DataFrame
        df_with_indicators = self._calculate_indicators(df_signal.copy())
        df_clean = df_with_indicators.dropna()

        if len(df_clean) < 2:
            print(f"Warning: Not enough clean signal data for {df_signal.attrs.get('symbol', 'N/A')}.")
            return {}

        signals_details, total_score = self._get_indicator_signals_and_score(df_clean)
        # Add trend info to the details
        signals_details.append({
            "indicator": "Trend_Filter",
            "signal": long_term_trend,
            "value": trend_details,
            "score_change": 0
        })

        # 3. Apply the trend filter to the score and signal
        final_signal_label, is_filtered = self._apply_trend_filter(total_score, long_term_trend)

        latest_candle = df_clean.iloc[-1]
        current_price = latest_candle.get('close')
        atr_key = f'ATRr_{settings.ATR_PERIOD}'
        current_atr = latest_candle.get(atr_key)

        if current_price is None or pd.isna(current_price):
            return {}
        # 4. Determine exit levels based on the *final* filtered signal
        # Only calculate exits if the signal is not neutral/filtered

        suggested_sl, suggested_tp = None, None
        if not is_filtered and final_signal_label in ["POTENTIAL_BUY", "POTENTIAL_SELL"]:
            _, suggested_sl, suggested_tp = self._determine_overall_signal_and_exits(
                current_price, current_atr, pre_approved_signal=final_signal_label
            )

        return {
            "signals_details": signals_details,
            "total_score": total_score,  # Return the final score after filtering
            "current_price": current_price,
            "overall_signal": final_signal_label,
            "is_filtered": is_filtered,
            "suggested_sl": suggested_sl,
            "suggested_tp": suggested_tp,
        }

    def _get_long_term_trend(self, df_trend: pd.DataFrame) -> Tuple[str, str]:
        """Determines the trend based on the long-term DataFrame."""
        if df_trend is None or df_trend.empty:
            return "NEUTRAL", "Trend data unavailable"

        trend_ema_key = f"EMA_{settings.TREND_FILTER_PERIOD}"
        df_trend[trend_ema_key] = df_trend.ta.ema(length=settings.TREND_FILTER_PERIOD)

        df_trend_clean = df_trend.dropna()
        if df_trend_clean.empty:
            return "NEUTRAL", "Not enough trend data for EMA"

        latest_trend_candle = df_trend_clean.iloc[-1]
        price = latest_trend_candle.get('close')
        trend_ema = latest_trend_candle.get(trend_ema_key)

        if price is None or pd.isna(price) or pd.isna(trend_ema):
            return "NEUTRAL", "Could not calculate trend EMA"

        trend_details = f"Price:{format_price_dynamically(price)} vs EMA({settings.TREND_FILTER_PERIOD}):{format_price_dynamically(trend_ema)}"

        if price > trend_ema:
            return "UPTREND", trend_details
        elif price < trend_ema:
            return "DOWNTREND", trend_details
        else:
            return "SIDEWAYS", trend_details

    def _apply_trend_filter(self, score: float, trend: str) -> Tuple[str, bool]:
        """Applies the trend filter logic."""
        is_buy_signal = score >= settings.BUY_SCORE_THRESHOLD
        is_sell_signal = score <= settings.SELL_SCORE_THRESHOLD

        if is_buy_signal and trend == "DOWNTREND":
            print(f"FILTERED: BUY signal (score: {score}) ignored due to DOWNTREND.")
            return "HOLD_FILTERED_BY_TREND", True  # Reset score

        if is_sell_signal and trend == "UPTREND":
            print(f"FILTERED: SELL signal (score: {score}) ignored due to UPTREND.")
            return "HOLD_FILTERED_BY_TREND", True  # Reset score

        # If signal is aligned with trend, or trend is neutral, let it pass
        if is_buy_signal: return "POTENTIAL_BUY", False
        if is_sell_signal: return "POTENTIAL_SELL", False

        return "HOLD_OBSERVE_NEUTRAL_SCORE", False

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Private helper method to calculate and append all necessary indicators to the DataFrame.
        This version includes detailed debugging.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # pandas_ta's append=True argument adds the column directly to the dataframe.
        # This simplifies the indicator calculation block.
        df.ta.rsi(length=settings.RSI_PERIOD, append=True)

        macd = df.ta.macd(fast=settings.MACD_FAST_PERIOD, slow=settings.MACD_SLOW_PERIOD,
                          signal=settings.MACD_SIGNAL_PERIOD)
        if macd is not None and not macd.empty:
            df = df.join(macd)

        df.ta.ema(length=settings.MA_SHORT_PERIOD, append=True)
        df.ta.ema(length=settings.MA_LONG_PERIOD, append=True)

        bbands = df.ta.bbands(length=settings.BBANDS_PERIOD, std=settings.BBANDS_STD_DEV)
        if bbands is not None and not bbands.empty:
            df = df.join(bbands)
        else:
            print(f"Warning: Bollinger Bands calculation failed. Creating empty columns.")
            for col in [f'BBL_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}',
                        f'BBM_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}',
                        f'BBU_{settings.BBANDS_PERIOD}_{settings.BBANDS_STD_DEV}']:
                df[col] = pd.NA

        df.ta.atr(length=settings.ATR_PERIOD, append=True)

        if 'volume' not in df.columns:
            df['volume'] = 0
        return df

    def _get_indicator_signals_and_score(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], float]:
        """
        Analyzes the latest data using a STATEFUL scoring model.
        """
        signals_details: List[Dict[str, Any]] = []
        total_score = 0.0  # Use float for score now

        if len(df) < 2:
            signals_details.append(
                {"indicator": "DataCheck", "signal": "INSUFFICIENT_DATA_ROWS", "value": len(df), "score_change": 0})
            return signals_details, total_score

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # 1. MA State & Event
        short_ma_key, long_ma_key = f'EMA_{settings.MA_SHORT_PERIOD}', f'EMA_{settings.MA_LONG_PERIOD}'
        latest_short, latest_long = latest.get(short_ma_key), latest.get(long_ma_key)
        prev_short, prev_long = previous.get(short_ma_key), previous.get(long_ma_key)
        if all(pd.notna(v) for v in [latest_short, latest_long, prev_short, prev_long]):
            state_score, event_score = 0, 0
            state_signal, event_signal = "NEUTRAL_STATE", "NO_EVENT"
            # State Score
            if latest_short > latest_long:
                state_signal = "GOLDEN_STATE"
                state_score = settings.WEIGHT_MA_STATE
            elif latest_short < latest_long:
                state_signal = "DEATH_STATE"
                state_score = -settings.WEIGHT_MA_STATE
            # Event Score
            if prev_short < prev_long and latest_short > latest_long:
                event_signal = "GOLDEN_CROSS"
                event_score = settings.WEIGHT_MA_EVENT
            elif prev_short > prev_long and latest_short < latest_long:
                event_signal = "DEATH_CROSS"
                event_score = -settings.WEIGHT_MA_EVENT

            total_score += state_score + event_score
            signals_details.append({"indicator": "MA", "signal": f"{state_signal} & {event_signal}",
                                    "value": f"S:{latest_short:.2f},L:{latest_long:.2f}",
                                    "score_change": state_score + event_score})
        else:
            signals_details.append(
                {"indicator": "MA", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        # 2. MACD State & Event
        macd_key = f'MACD_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        signal_key = f'MACDs_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        latest_macd, latest_signal = latest.get(macd_key), latest.get(signal_key)
        prev_macd, prev_signal = previous.get(macd_key), previous.get(signal_key)
        if all(pd.notna(v) for v in [latest_macd, latest_signal, prev_macd, prev_signal]):
            state_score, event_score = 0, 0
            state_signal, event_signal = "NEUTRAL_STATE", "NO_EVENT"
            # State Score
            if latest_macd > latest_signal:
                state_signal = "BULLISH_STATE"
                state_score = settings.WEIGHT_MACD_STATE
            elif latest_macd < latest_signal:
                state_signal = "BEARISH_STATE"
                state_score = -settings.WEIGHT_MACD_STATE
            # Event Score
            if prev_macd < prev_signal and latest_macd > latest_signal:
                event_signal = "GOLDEN_CROSS"
                event_score = settings.WEIGHT_MACD_EVENT
            elif prev_macd > prev_signal and latest_macd < latest_signal:
                event_signal = "DEATH_CROSS"
                event_score = -settings.WEIGHT_MACD_EVENT

            total_score += state_score + event_score
            signals_details.append({"indicator": "MACD", "signal": f"{state_signal} & {event_signal}",
                                    "value": f"M:{latest_macd:.2f},S:{latest_signal:.2f}",
                                    "score_change": state_score + event_score})
        else:
            signals_details.append(
                {"indicator": "MACD", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        # 3. RSI State (Extreme and Trend)
        rsi_key = f'RSI_{settings.RSI_PERIOD}'
        rsi_value = latest.get(rsi_key)
        if pd.notna(rsi_value):
            extreme_score, trend_score = 0, 0
            extreme_signal, trend_signal = "NEUTRAL_EXTREME", "NEUTRAL_TREND"
            # Extreme Score
            if rsi_value < settings.RSI_OVERSOLD:
                extreme_signal = "OVERSOLD"
                extreme_score = settings.WEIGHT_RSI_EXTREME
            elif rsi_value > settings.RSI_OVERBOUGHT:
                extreme_signal = "OVERBOUGHT"
                extreme_score = -settings.WEIGHT_RSI_EXTREME
            # Trend Score
            if 50 < rsi_value <= settings.RSI_OVERBOUGHT:
                trend_signal = "BULLISH_ZONE"
                trend_score = settings.WEIGHT_RSI_TREND
            elif settings.RSI_OVERSOLD <= rsi_value < 50:
                trend_signal = "BEARISH_ZONE"
                trend_score = -settings.WEIGHT_RSI_TREND

            total_score += extreme_score + trend_score
            signals_details.append(
                {"indicator": "RSI", "signal": f"{extreme_signal} & {trend_signal}", "value": rsi_value,
                 "score_change": extreme_score + trend_score})
        else:
            signals_details.append(
                {"indicator": "RSI", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        # 4. Bollinger Bands State (Breakout)
        price_close = latest.get('close')
        std_dev_str = f"{float(settings.BBANDS_STD_DEV):.1f}"
        bbu_key, bbl_key = f'BBU_{settings.BBANDS_PERIOD}_{std_dev_str}', f'BBL_{settings.BBANDS_PERIOD}_{std_dev_str}'
        latest_bbu, latest_bbl = latest.get(bbu_key), latest.get(bbl_key)
        if pd.notna(price_close) and all(pd.notna(val) for val in [latest_bbu, latest_bbl]):
            score_change = 0
            signal_text = "INSIDE_BANDS"
            if price_close > latest_bbu:
                signal_text = "BREAK_UPPER"
                score_change = settings.WEIGHT_BBANDS_BREAKOUT
            elif price_close < latest_bbl:
                signal_text = "BREAK_LOWER"
                score_change = -settings.WEIGHT_BBANDS_BREAKOUT
            total_score += score_change
            signals_details.append({"indicator": "BollingerBands", "signal": signal_text,
                                    "value": f"P:{format_price_dynamically(price_close)},U:{format_price_dynamically(latest_bbu)},L:{format_price_dynamically(latest_bbl)}",
                                    "score_change": score_change})
        else:
            signals_details.append(
                {"indicator": "BollingerBands", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        # 5. Volume Confirmation
        volume_latest = latest.get('volume')
        if pd.notna(volume_latest):
            avg_volume_period = 5
            signal_text = "NOT_ENOUGH_DATA_FOR_AVG"
            vol_value_str = f"Vol:{volume_latest:.2f}"
            if len(df) > avg_volume_period + 1:
                avg_volume = df['volume'].iloc[-(avg_volume_period + 1):-1].mean()
                if pd.notna(avg_volume) and avg_volume > 0:
                    vol_value_str = f"Vol:{volume_latest:.2f},Avg5P:{avg_volume:.2f}"
                    signal_text = "HIGH_VOLUME" if volume_latest > avg_volume * 1.5 else "NORMAL_OR_LOW_VOLUME"
                else:
                    signal_text = "VOLUME_AVG_NOT_CALCULABLE"
            signals_details.append(
                {"indicator": "Volume", "signal": signal_text, "value": vol_value_str, "score_change": 0})
        else:
            signals_details.append(
                {"indicator": "Volume", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        return signals_details, total_score

    def _determine_overall_signal_and_exits(self, current_price: float, current_atr: float,
        pre_approved_signal: str) -> Tuple[str, float | None, float | None]:
        """
        Calculates exit prices for an already approved signal.
        """
        suggested_sl = None
        suggested_tp = None

        if pd.notna(current_atr) and current_atr > 0:
            stop_loss_distance = settings.ATR_STOP_LOSS_MULTIPLIER * current_atr
            take_profit_distance = stop_loss_distance * settings.RISK_REWARD_RATIO

            if pre_approved_signal == "POTENTIAL_BUY":
                suggested_sl = current_price - stop_loss_distance
                suggested_tp = current_price + take_profit_distance
            elif pre_approved_signal == "POTENTIAL_SELL":
                suggested_sl = current_price + stop_loss_distance
                suggested_tp = current_price - take_profit_distance

        return pre_approved_signal, suggested_sl, suggested_tp