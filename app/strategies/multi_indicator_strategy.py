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

    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Public method to generate signals. It orchestrates the internal calculation
        and scoring logic.
        """
        # 1. Calculate all indicators on the DataFrame
        df_with_indicators = self._calculate_indicators(df.copy())

        # 2. Drop all rows that have ANY NaN value after indicator calculation.
        df_clean = df_with_indicators.dropna()

        if len(df_clean) < 2:
            print(
                f"Warning: After dropping NaN rows, not enough data remains for signal generation. Original rows: {len(df)}, Clean rows: {len(df_clean)}")
            return {}

        # 3. Generate signals and score using the cleaned DataFrame
        signals_details, total_score = self._get_indicator_signals_and_score(df_clean)

        latest_candle = df_clean.iloc[-1]
        current_price = latest_candle.get('close')

        atr_key = f'ATRr_{settings.ATR_PERIOD}'
        current_atr = latest_candle.get(atr_key)
        if current_price is None or pd.isna(current_price):
            return {}

        # 4. Determine overall signal and exit levels
        overall_signal, suggested_sl, suggested_tp = self._determine_overall_signal_and_exits(
            total_score, current_price, current_atr
        )

        # 5. Return the final, structured result
        return {
            "signals_details": signals_details,
            "total_score": total_score,
            "current_price": current_price,
            "overall_signal": overall_signal,
            "suggested_sl": suggested_sl,
            "suggested_tp": suggested_tp,
        }

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

    def _get_indicator_signals_and_score(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
        """
        Analyzes the latest row of the DataFrame with indicators to generate signals and a composite score.
        This version has more robust handling for NaN values.
        """
        signals_details: List[Dict[str, Any]] = []
        total_score = 0
        if len(df) < 2:
            signals_details.append(
                {"indicator": "DataCheck", "signal": "INSUFFICIENT_DATA_ROWS", "value": len(df), "score_change": 0})
            return signals_details, total_score

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # 1. RSI Signal
        rsi_key = f'RSI_{settings.RSI_PERIOD}'
        rsi_value = latest.get(rsi_key)
        if pd.notna(rsi_value):
            score_change = 0;
            signal_text = "NEUTRAL"
            if rsi_value < settings.RSI_OVERSOLD:
                signal_text = "OVERSOLD_BUY";
                score_change = settings.WEIGHT_RSI_SIGNAL
            elif rsi_value > settings.RSI_OVERBOUGHT:
                signal_text = "OVERBOUGHT_SELL";
                score_change = -settings.WEIGHT_RSI_SIGNAL
            total_score += score_change
            signals_details.append(
                {"indicator": "RSI", "signal": signal_text, "value": rsi_value, "score_change": score_change})
        else:
            signals_details.append(
                {"indicator": "RSI", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        # 2. MACD Signal (Crossover)
        macd_line_key = f'MACD_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        signal_line_key = f'MACDs_{settings.MACD_FAST_PERIOD}_{settings.MACD_SLOW_PERIOD}_{settings.MACD_SIGNAL_PERIOD}'
        latest_macd = latest.get(macd_line_key)
        latest_signal = latest.get(signal_line_key)
        prev_macd = previous.get(macd_line_key)
        prev_signal = previous.get(signal_line_key)

        if all(pd.notna(val) for val in [latest_macd, latest_signal, prev_macd, prev_signal]):
            score_change = 0;
            signal_text = "NO_CROSS"
            if prev_macd < prev_signal and latest_macd > latest_signal:
                signal_text = "GOLDEN_CROSS_BUY";
                score_change = settings.WEIGHT_MACD_CROSS
            elif prev_macd > prev_signal and latest_macd < latest_signal:
                signal_text = "DEATH_CROSS_SELL";
                score_change = -settings.WEIGHT_MACD_CROSS
            total_score += score_change
            signals_details.append({"indicator": "MACD_Cross", "signal": signal_text,
                                    "value": f"MACD:{latest_macd:.2f},Sig:{latest_signal:.2f}",
                                    "score_change": score_change})
        else:
            signals_details.append(
                {"indicator": "MACD_Cross", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        # 3. MA Crossover Signal
        short_ma_key, long_ma_key = f'EMA_{settings.MA_SHORT_PERIOD}', f'EMA_{settings.MA_LONG_PERIOD}'
        latest_short_ma, latest_long_ma = latest.get(short_ma_key), latest.get(long_ma_key)
        prev_short_ma, prev_long_ma = previous.get(short_ma_key), previous.get(long_ma_key)

        if all(pd.notna(val) for val in [latest_short_ma, latest_long_ma, prev_short_ma, prev_long_ma]):
            score_change = 0
            signal_text = "NO_CROSS"
            if prev_short_ma < prev_long_ma and latest_short_ma > latest_long_ma:
                signal_text = "GOLDEN_CROSS_BUY"
                score_change = settings.WEIGHT_MA_CROSS
            elif prev_short_ma > prev_long_ma and latest_short_ma < latest_long_ma:
                signal_text = "DEATH_CROSS_SELL"
                score_change = -settings.WEIGHT_MA_CROSS
            total_score += score_change
            signals_details.append({"indicator": "MA_Cross", "signal": signal_text,
                                    "value": f"S:{latest_short_ma:.2f},L:{latest_long_ma:.2f}",
                                    "score_change": score_change})
        else:
            signals_details.append(
                {"indicator": "MA_Cross", "signal": "DATA_UNAVAILABLE", "value": "NaN", "score_change": 0})

        # 4. Bollinger Bands Signal
        price_close = latest.get('close')
        std_dev_str = f"{float(settings.BBANDS_STD_DEV):.1f}"
        bbu_key = f'BBU_{settings.BBANDS_PERIOD}_{std_dev_str}'
        bbl_key = f'BBL_{settings.BBANDS_PERIOD}_{std_dev_str}'
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
            signal_text = "NOT_ENOUGH_DATA_FOR_AVG";
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

    def _determine_overall_signal_and_exits(self, total_score: int, current_price: float, current_atr: float) -> Tuple[
        str, float | None, float | None]:
        """Determines the final signal and exit prices based on score and ATR."""
        suggested_sl = None
        suggested_tp = None

        is_buy_signal = total_score >= settings.BUY_SCORE_THRESHOLD
        is_sell_signal = total_score <= settings.SELL_SCORE_THRESHOLD

        if (is_buy_signal or is_sell_signal) and pd.notna(current_atr) and current_atr > 0:
            stop_loss_distance = settings.ATR_STOP_LOSS_MULTIPLIER * current_atr
            take_profit_distance = stop_loss_distance * settings.RISK_REWARD_RATIO

            if is_buy_signal:
                overall_signal = "POTENTIAL_BUY"
                suggested_sl = current_price - stop_loss_distance
                suggested_tp = current_price + take_profit_distance
            else:  # is_sell_signal
                overall_signal = "POTENTIAL_SELL"
                suggested_sl = current_price + stop_loss_distance
                suggested_tp = current_price - take_profit_distance
        else:  # No strong signal or ATR is invalid
            overall_signal = "HOLD_OBSERVE_NEUTRAL_SCORE"

        return overall_signal, suggested_sl, suggested_tp