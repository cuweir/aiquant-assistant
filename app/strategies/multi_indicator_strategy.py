# app/strategies/multi_indicator_strategy.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any

from app.strategies.base_strategy import TradingStrategy


class MultiIndicatorStrategy(TradingStrategy):
    """
    This is the complete and correct real-time analysis version of our successful
    V7 Volatility Adaptive backtesting strategy. It includes both Stop Loss
    and a dynamic Take Profit signal logic.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the strategy with a specific parameter set.
        """
        if not params:
            raise ValueError("Strategy parameters cannot be None or empty.")
        self.p = params
        # Set default values for all possible parameters to ensure robustness
        self.p.setdefault('regime_ma_period', 200)
        self.p.setdefault('buy_score_threshold', 3)
        self.p.setdefault('slope_lookback_period', 5)
        self.p.setdefault('slope_min_threshold', 0.0)
        self.p.setdefault('vol_atr_period', 14)
        self.p.setdefault('vol_atr_ma_period', 100)
        self.p.setdefault('low_vol_ma_short', 25)
        self.p.setdefault('low_vol_ma_long', 80)
        self.p.setdefault('low_vol_adx_threshold', 30)
        self.p.setdefault('low_vol_atr_sl_multiplier', 2.0)
        self.p.setdefault('high_vol_ma_short', 15)
        self.p.setdefault('high_vol_ma_long', 40)
        self.p.setdefault('high_vol_adx_threshold', 30)
        self.p.setdefault('high_vol_atr_sl_multiplier', 2.5)
        self.p.setdefault('rsi_period', 14)
        self.p.setdefault('rsi_oversold', 40)
        self.p.setdefault('macd_fast', 12)
        self.p.setdefault('macd_slow', 26)
        self.p.setdefault('macd_signal', 9)
        self.p.setdefault('adx_period', 14)

    def _calculate_indicators(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculates all necessary indicators using pandas_ta.
        """
        indicators = {}

        # Regime Filter
        indicators['regime_ma'] = df_regime['close'].ta.sma(length=self.p['regime_ma_period'])

        # Volatility Regime
        vol_atr = df_signal.ta.atr(length=self.p['vol_atr_period'], append=True)
        indicators['vol_atr'] = vol_atr
        indicators['vol_atr_ma'] = vol_atr.ta.sma(length=self.p['vol_atr_ma_period'])

        # Low Volatility Indicators
        indicators['low_vol_ma_short'] = df_signal['close'].ta.sma(length=self.p['low_vol_ma_short'])
        indicators['low_vol_ma_long'] = df_signal['close'].ta.sma(length=self.p['low_vol_ma_long'])
        adx_low = df_signal.ta.adx(length=self.p['adx_period'])
        if adx_low is not None and not adx_low.empty:
            indicators['low_vol_adx'] = adx_low[f'ADX_{self.p["adx_period"]}']

        # High Volatility Indicators
        indicators['high_vol_ma_short'] = df_signal['close'].ta.sma(length=self.p['high_vol_ma_short'])
        indicators['high_vol_ma_long'] = df_signal['close'].ta.sma(length=self.p['high_vol_ma_long'])
        adx_high = df_signal.ta.adx(length=self.p['adx_period'])
        if adx_high is not None and not adx_high.empty:
            indicators['high_vol_adx'] = adx_high[f'ADX_{self.p["adx_period"]}']

        # Common Indicators
        indicators['rsi'] = df_signal.ta.rsi(length=self.p['rsi_period'])
        macd_df = df_signal.ta.macd(fast=self.p['macd_fast'], slow=self.p['macd_slow'], signal=self.p['macd_signal'])
        if macd_df is not None and not macd_df.empty:
            indicators['macd'] = macd_df[f'MACD_{self.p["macd_fast"]}_{self.p["macd_slow"]}_{self.p["macd_signal"]}']
            indicators['macd_signal'] = macd_df[
                f'MACDs_{self.p["macd_fast"]}_{self.p["macd_slow"]}_{self.p["macd_signal"]}']

        return indicators

    async def generate_signals(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, Any] | None:
        """
        Main analysis function, mirroring the V7 backtest logic completely.
        """
        if len(df_signal) < self.p['low_vol_ma_long'] or len(df_regime) < self.p['regime_ma_period']:
            return {"overall_signal": "INSUFFICIENT_DATA", "total_score": 0,
                    "current_price": df_signal['close'].iloc[-1]}

        indicators = self._calculate_indicators(df_signal, df_regime)

        # Get latest and previous values for signal evaluation
        latest = {name: series.iloc[-1] for name, series in indicators.items()}
        previous = {name: series.iloc[-2] for name, series in indicators.items()}
        latest_close = df_signal['close'].iloc[-1]

        # --- Strategy Logic ---

        # 1. Regime Filter
        is_bull_regime = df_regime['close'].iloc[-1] > latest.get('regime_ma', float('inf'))
        if not is_bull_regime:
            return {"overall_signal": "REGIME_FILTER_BEARISH", "total_score": 0, "current_price": latest_close}

        # 2. Volatility Adaptive Parameters
        is_high_vol = latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0)
        if is_high_vol:
            ma_short, ma_long, adx_val, adx_thresh, atr_sl_mult, ma_short_series = (
                latest.get('high_vol_ma_short'), latest.get('high_vol_ma_long'), latest.get('high_vol_adx'),
                self.p['high_vol_adx_threshold'], self.p['high_vol_atr_sl_multiplier'], indicators['high_vol_ma_short']
            )
        else:
            ma_short, ma_long, adx_val, adx_thresh, atr_sl_mult, ma_short_series = (
                latest.get('low_vol_ma_short'), latest.get('low_vol_ma_long'), latest.get('low_vol_adx'),
                self.p['low_vol_adx_threshold'], self.p['low_vol_atr_sl_multiplier'], indicators['low_vol_ma_short']
            )

        # 3. Confluence Scoring
        buy_score = 0
        if ma_short > ma_long: buy_score += 1
        if previous.get('macd') < previous.get('macd_signal') and latest.get('macd') > latest.get('macd_signal'):
            buy_score += 2
        if previous.get('rsi') < self.p['rsi_oversold'] and latest.get('rsi') > self.p['rsi_oversold']:
            buy_score += 1

        final_signal = "NEUTRAL"
        if buy_score >= self.p['buy_score_threshold']:
            if adx_val > adx_thresh:
                # Slope Confirmation
                y_values = ma_short_series.iloc[-self.p['slope_lookback_period']:].values
                if len(y_values) == self.p['slope_lookback_period']:
                    x_values = np.arange(len(y_values))
                    slope = np.polyfit(x_values, y_values, 1)[0]
                    if slope > self.p['slope_min_threshold']:
                        final_signal = "POTENTIAL_BUY"

        # 4. [FIX] Calculate Exits and provide complete information
        suggested_sl = None
        take_profit_condition = "Trend reversal (short MA crosses below long MA)"  # This is our dynamic TP

        if final_signal == "POTENTIAL_BUY":
            suggested_sl = latest_close - (latest.get('vol_atr', 0) * atr_sl_mult)

        return {
            "overall_signal": final_signal,
            "total_score": buy_score,
            "current_price": latest_close,
            "suggested_sl": suggested_sl,
            "take_profit_condition": take_profit_condition
        }