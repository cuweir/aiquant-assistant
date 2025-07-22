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

    async def generate_signals(
            self,
            df_signal: pd.DataFrame,
            df_trend_short: pd.DataFrame,
            df_trend_long: pd.DataFrame
    ) -> Dict[str, Any]:

        # 1. Determine multi-level trends
        trend_long, trend_details_long = self._get_trend(df_trend_long, settings.TREND_FILTER_PERIOD_LONG)
        trend_short, trend_details_short = self._get_trend(df_trend_short, settings.TREND_FILTER_PERIOD_SHORT)

        # 2. Determine the current market regime
        market_regime = self._determine_market_regime(trend_long, trend_short)

        # 3. Calculate base indicators and score on the signal DataFrame
        df_with_indicators = self._calculate_indicators(df_signal.copy())
        df_clean = df_with_indicators.dropna()

        if len(df_clean) < 2: return {}

        signals_details, base_score = self._get_indicator_signals_and_score(df_clean)

        signals_details.append({"indicator": f"Trend_Filter_{settings.TREND_TIMEFRAME_LONG}", "signal": trend_long,
                                "value": trend_details_long, "score_change": 0})
        signals_details.append({"indicator": f"Trend_Filter_{settings.TREND_TIMEFRAME_SHORT}", "signal": trend_short,
                                "value": trend_details_short, "score_change": 0})
        signals_details.append({"indicator": "Market_Regime", "signal": market_regime, "value": "", "score_change": 0})

        # 4. Apply regime-specific rules to adjust score and determine final signal
        final_signal, final_score = self._apply_regime_rules(base_score, market_regime, signals_details)

        # 5. Determine exits and assemble final result
        latest_candle = df_clean.iloc[-1]
        current_price = latest_candle.get('close')
        atr_key = f'ATRr_{settings.ATR_PERIOD}'
        current_atr = latest_candle.get(atr_key)

        if current_price is None or pd.isna(current_price): return {}

        suggested_sl, suggested_tp1, suggested_tp2 = None, None, None
        if final_signal in ["POTENTIAL_BUY", "POTENTIAL_SELL"]:
            _, suggested_sl, suggested_tp1, suggested_tp2 = self._determine_overall_signal_and_exits(
                current_price, current_atr, pre_approved_signal=final_signal
            )

        return {
            "signals_details": signals_details,
            "total_score": final_score,
            "current_price": current_price,
            "overall_signal": final_signal,
            "suggested_sl": suggested_sl,
            "suggested_tp1": suggested_tp1,  # <-- ADDED
            "suggested_tp2": suggested_tp2
        }

    @staticmethod
    def _get_trend(df_trend: pd.DataFrame, period: int) -> Tuple[str, str]:
        if df_trend is None or df_trend.empty: return "NEUTRAL", "Trend data unavailable"
        trend_ema_key = f"EMA_{period}"
        df_trend[trend_ema_key] = df_trend.ta.ema(length=period)
        df_trend_clean = df_trend.dropna(subset=['close', trend_ema_key])
        if df_trend_clean.empty: return "NEUTRAL", "Not enough trend data for EMA"
        latest = df_trend_clean.iloc[-1]
        price, trend_ema = latest.get('close'), latest.get(trend_ema_key)
        if pd.isna(price) or pd.isna(trend_ema): return "NEUTRAL", "Could not calculate trend EMA"
        details = f"Price:{format_price_dynamically(price)} vs EMA({period}):{format_price_dynamically(trend_ema)}"
        if price > trend_ema:
            return "UPTREND", details
        elif price < trend_ema:
            return "DOWNTREND", details
        else:
            return "SIDEWAYS", details

    @staticmethod
    def _determine_market_regime(trend_long: str, trend_short: str) -> str:
        if trend_long == "UPTREND" and trend_short == "UPTREND": return "STRONG_BULL"
        if trend_long == "DOWNTREND" and trend_short == "DOWNTREND": return "STRONG_BEAR"
        if trend_long == "UPTREND" and trend_short in ["DOWNTREND", "SIDEWAYS"]: return "BULLISH_PULLBACK"
        if trend_long == "DOWNTREND" and trend_short in ["UPTREND", "SIDEWAYS"]: return "BEARISH_RALLY"
        return "CHOPPY"

    @staticmethod
    def _apply_regime_rules(score: float, regime: str, details: List[Dict]) -> Tuple[str, float]:
        final_score = score
        if regime == "STRONG_BULL":
            if score > 0:
                final_score += settings.REGIME_STRONG_TREND_BONUS
            else:
                final_score = 0
        elif regime == "STRONG_BEAR":
            if score < 0:
                final_score -= settings.REGIME_STRONG_TREND_BONUS
            else:
                final_score = 0
        elif regime == "BULLISH_PULLBACK":
            if score < 0: final_score = 0
            rsi_detail = next((d for d in details if d['indicator'] == 'RSI'), None)
            macd_detail = next((d for d in details if d['indicator'] == 'MACD'), None)
            if rsi_detail and "OVERSOLD" in rsi_detail['signal']:
                final_score += settings.REGIME_BULLISH_PULLBACK_RSI_BONUS
            if macd_detail and "GOLDEN_CROSS" in macd_detail['signal']:
                final_score += settings.REGIME_BULLISH_PULLBACK_MACD_BONUS
        elif regime == "BEARISH_RALLY":
            if score > 0: final_score = 0
            rsi_detail = next((d for d in details if d['indicator'] == 'RSI'), None)
            macd_detail = next((d for d in details if d['indicator'] == 'MACD'), None)
            if rsi_detail and "OVERBOUGHT" in rsi_detail['signal']:
                final_score -= settings.REGIME_BULLISH_PULLBACK_RSI_BONUS
            if macd_detail and "DEATH_CROSS" in macd_detail['signal']:
                final_score -= settings.REGIME_BULLISH_PULLBACK_MACD_BONUS
        elif regime == "CHOPPY":
            final_score = 0

        if final_score >= settings.BUY_SCORE_THRESHOLD: return "POTENTIAL_BUY", final_score
        if final_score <= settings.SELL_SCORE_THRESHOLD: return "POTENTIAL_SELL", final_score

        if regime != "CHOPPY" and score != final_score: return "HOLD_REGIME_ADJUSTED", final_score
        if regime == "CHOPPY": return "HOLD_CHOPPY_MARKET", final_score
        return "HOLD_OBSERVE_NEUTRAL_SCORE", final_score

    @staticmethod
    def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        df.ta.rsi(length=settings.RSI_PERIOD, append=True)
        macd = df.ta.macd(fast=settings.MACD_FAST_PERIOD, slow=settings.MACD_SLOW_PERIOD,
                          signal=settings.MACD_SIGNAL_PERIOD)
        if macd is not None and not macd.empty: df = df.join(macd)
        df.ta.ema(length=settings.MA_SHORT_PERIOD, append=True)
        df.ta.ema(length=settings.MA_LONG_PERIOD, append=True)
        bbands = df.ta.bbands(length=settings.BBANDS_PERIOD, std=settings.BBANDS_STD_DEV)
        if bbands is not None and not bbands.empty: df = df.join(bbands)
        df.ta.atr(length=settings.ATR_PERIOD, append=True)
        if 'volume' not in df.columns: df['volume'] = 0
        return df

    @staticmethod
    def _get_indicator_signals_and_score(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], float]:
        signals_details: List[Dict[str, Any]] = []
        total_score = 0.0
        if len(df) < 2: return signals_details, total_score
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

    @staticmethod
    def _determine_overall_signal_and_exits(current_price: float, current_atr: float, pre_approved_signal: str) -> \
    Tuple[str, float | None, float | None]:
        """
        Calculates exit prices for an already approved signal.
        Now returns SL, TP1, and TP2.
        """
        suggested_sl, suggested_tp1, suggested_tp2 = None, None, None

        if pd.notna(current_atr) and current_atr > 0:
            stop_loss_distance = settings.ATR_STOP_LOSS_MULTIPLIER * current_atr

            take_profit_1_distance = stop_loss_distance * settings.RISK_REWARD_RATIO_TP1
            take_profit_2_distance = stop_loss_distance * settings.RISK_REWARD_RATIO_TP2

            if pre_approved_signal == "POTENTIAL_BUY":
                suggested_sl = current_price - stop_loss_distance
                suggested_tp1 = current_price + take_profit_1_distance
                suggested_tp2 = current_price + take_profit_2_distance
            elif pre_approved_signal == "POTENTIAL_SELL":
                suggested_sl = current_price + stop_loss_distance
                suggested_tp1 = current_price - take_profit_1_distance
                suggested_tp2 = current_price - take_profit_2_distance
        return pre_approved_signal, suggested_sl, suggested_tp1, suggested_tp2
