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

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [NEW] A robust data cleaning pre-processing step.
        """
        # Forward-fill any missing values. This is a common practice for time-series data.
        df.ffill(inplace=True)
        # Drop any remaining NaN rows that might exist at the beginning of the series
        df.dropna(inplace=True)
        return df

    def _calculate_indicators(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculates all necessary indicators after cleaning the data.
        """
        print(f"  > Data before cleaning: signal={len(df_signal)}, regime={len(df_regime)}")

        # [CRITICAL FIX] Apply data cleaning before any calculations
        df_signal = self._clean_data(df_signal.copy())
        df_regime = self._clean_data(df_regime.copy())

        print(f"  > Data after cleaning: signal={len(df_signal)}, regime={len(df_regime)}")
        print(f"  > Required: signal>={self.p['low_vol_ma_long']}, regime>={self.p['regime_ma_period']}")

        # Check if data is still sufficient after cleaning
        if len(df_signal) < self.p['low_vol_ma_long'] or len(df_regime) < self.p['regime_ma_period']:
            print(f"  > INSUFFICIENT DATA after cleaning: signal={len(df_signal)}<{self.p['low_vol_ma_long']} or regime={len(df_regime)}<{self.p['regime_ma_period']}")
            return {}  # Return empty dict if not enough data

        indicators = {}
        print(f"  > Starting indicator calculations...")

        # [CRITICAL FIX] Call ta methods on the DataFrame, not the Series.

        # Regime Filter
        try:
            indicators['regime_ma'] = df_regime.ta.sma(length=self.p['regime_ma_period'])
            print(f"  > regime_ma calculated: {len(indicators['regime_ma'])} values")
        except Exception as e:
            print(f"  > Error calculating regime_ma: {e}")
            return {}

        # Volatility Regime
        try:
            # .ta.atr() returns a Series.
            vol_atr = df_signal.ta.atr(length=self.p['vol_atr_period'], append=False)
            if vol_atr is not None and not vol_atr.empty:
                indicators['vol_atr'] = vol_atr
                print(f"  > vol_atr calculated: {len(vol_atr)} values")
                # To calculate the SMA of the ATR, we add it to the original DataFrame temporarily.
                df_with_atr = df_signal.copy()
                df_with_atr['atr'] = vol_atr
                # Now we can call .ta.sma() on the DataFrame, specifying the 'atr' column.
                indicators['vol_atr_ma'] = df_with_atr.ta.sma(close='atr', length=self.p['vol_atr_ma_period'])
                print(f"  > vol_atr_ma calculated: {len(indicators['vol_atr_ma'])} values")
            else:
                print(f"  > vol_atr calculation failed or empty")
        except Exception as e:
            print(f"  > Error calculating vol_atr: {e}")
            return {}

        # Low Volatility Indicators
        indicators['low_vol_ma_short'] = df_signal.ta.sma(length=self.p['low_vol_ma_short'])
        indicators['low_vol_ma_long'] = df_signal.ta.sma(length=self.p['low_vol_ma_long'])
        adx_low = df_signal.ta.adx(length=self.p['adx_period'])
        if adx_low is not None and not adx_low.empty:
             indicators['low_vol_adx'] = adx_low[f'ADX_{self.p["adx_period"]}']

        # High Volatility Indicators
        indicators['high_vol_ma_short'] = df_signal.ta.sma(length=self.p['high_vol_ma_short'])
        indicators['high_vol_ma_long'] = df_signal.ta.sma(length=self.p['high_vol_ma_long'])
        adx_high = df_signal.ta.adx(length=self.p['adx_period'])
        if adx_high is not None and not adx_high.empty:
            indicators['high_vol_adx'] = adx_high[f'ADX_{self.p["adx_period"]}']

        # Common Indicators
        indicators['rsi'] = df_signal.ta.rsi(length=self.p['rsi_period'])
        macd_df = df_signal.ta.macd(fast=self.p['macd_fast'], slow=self.p['macd_slow'], signal=self.p['macd_signal'])
        if macd_df is not None and not macd_df.empty:
            indicators['macd'] = macd_df[f'MACD_{self.p["macd_fast"]}_{self.p["macd_slow"]}_{self.p["macd_signal"]}']
            indicators['macd_signal'] = macd_df[f'MACDs_{self.p["macd_fast"]}_{self.p["macd_slow"]}_{self.p["macd_signal"]}']

        return indicators

    async def generate_signals(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, Any] | None:
        if len(df_signal) < self.p['low_vol_ma_long'] or len(df_regime) < self.p['regime_ma_period']:
            return {"overall_signal": "INSUFFICIENT_DATA", "current_price": df_signal['close'].iloc[-1]}

        indicators = self._calculate_indicators(df_signal, df_regime)
        if not indicators:
            return {"overall_signal": "INSUFFICIENT_CLEAN_DATA", "current_price": df_signal['close'].iloc[-1]}

        print(f"  > Indicators calculated: {list(indicators.keys())}")

        # Check for NaN values in the last few values of each indicator
        for name, series in indicators.items():
            if len(series) >= 2:
                last_val = series.iloc[-1]
                prev_val = series.iloc[-2]
                print(f"  > {name}: last={last_val}, prev={prev_val}, last_is_nan={pd.isna(last_val)}, prev_is_nan={pd.isna(prev_val)}")
            else:
                print(f"  > {name}: insufficient data (length={len(series)})")

        latest = {name: series.iloc[-1] for name, series in indicators.items() if pd.notna(series.iloc[-1])}
        previous = {name: series.iloc[-2] for name, series in indicators.items() if pd.notna(series.iloc[-2])}
        latest_close = df_signal['close'].iloc[-1]

        print(f"  > Latest values: {list(latest.keys())}")
        print(f"  > Previous values: {list(previous.keys())}")

        if not latest:
            print(f"  > ERROR: No valid latest values found!")
            return {"overall_signal": "NO_VALID_INDICATORS", "current_price": latest_close}

        snapshot = {
            "price": latest_close,
            "regime_ma": latest.get('regime_ma'),
            "is_bull_regime": df_regime['close'].iloc[-1] > latest.get('regime_ma', float('inf')),
            "vol_atr": latest.get('vol_atr'),
            "vol_atr_ma": latest.get('vol_atr_ma'),
            "is_high_vol": latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0),
            "components": {}  # To be filled below
        }

        print(f"  > Current regime price: {df_regime['close'].iloc[-1]}")
        print(f"  > Regime MA: {latest.get('regime_ma')}")
        print(f"  > Is bull regime: {snapshot['is_bull_regime']}")

        # --- Strategy Logic ---
        is_bull_regime = snapshot["is_bull_regime"]
        if not is_bull_regime:
            return {"overall_signal": "REGIME_FILTER_BEARISH", "total_score": 0, "current_price": latest_close, "snapshot": snapshot}

        is_high_vol = snapshot["is_high_vol"]
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

        # --- Exit Signal Check ---
        print(f"  > MA Short: {ma_short}, MA Long: {ma_long}")
        print(f"  > Exit signal check: ma_short < ma_long = {ma_short < ma_long}")

        if ma_short < ma_long:
            print(f"  > EXIT SIGNAL triggered - but still filling components for analysis")
            # Fill components even for exit signals to provide complete analysis
            snapshot['components']['trend'] = {
                "value": f"MA({self.p['low_vol_ma_short'] if not is_high_vol else self.p['high_vol_ma_short']}) > MA({self.p['low_vol_ma_long'] if not is_high_vol else self.p['high_vol_ma_long']})",
                "result": False, "score": 0}

            # Add other components for completeness
            c2_momentum = previous.get('macd', 0) < previous.get('macd_signal', 0) and latest.get('macd', 0) > latest.get('macd_signal', 0)
            snapshot['components']['momentum'] = {"value": f"MACD Cross Up", "result": c2_momentum, "score": 2 if c2_momentum else 0}

            c3_pullback = previous.get('rsi', 0) < self.p['rsi_oversold'] and latest.get('rsi', 0) > self.p['rsi_oversold']
            snapshot['components']['pullback'] = {"value": f"RSI Cross Up {self.p['rsi_oversold']}", "result": c3_pullback, "score": 1 if c3_pullback else 0}

            snapshot['total_score'] = 0  # Exit signal overrides any positive score

            return {
                "overall_signal": "NEUTRAL",
                "exit_signal": "EXIT_LONG",
                "current_price": latest_close,
                "snapshot": snapshot
            }

        # --- Entry Logic ---
        final_signal = "NEUTRAL"
        buy_score = 0

        # Scoring Component 1: Trend
        c1_trend = ma_short > ma_long
        if c1_trend: buy_score += 1
        snapshot['components']['trend'] = {
            "value": f"MA({self.p['low_vol_ma_short'] if not is_high_vol else self.p['high_vol_ma_short']}) > MA({self.p['low_vol_ma_long'] if not is_high_vol else self.p['high_vol_ma_long']})",
            "result": c1_trend, "score": 1 if c1_trend else 0}

        # Scoring Component 2: Momentum
        c2_momentum = previous.get('macd', 0) < previous.get('macd_signal', 0) and latest.get('macd', 0) > latest.get(
            'macd_signal', 0)
        if c2_momentum: buy_score += 2
        snapshot['components']['momentum'] = {"value": f"MACD Cross Up", "result": c2_momentum,
                                              "score": 2 if c2_momentum else 0}

        # Scoring Component 3: Pullback
        c3_pullback = previous.get('rsi', 0) < self.p['rsi_oversold'] and latest.get('rsi', 0) > self.p['rsi_oversold']
        if c3_pullback: buy_score += 1
        snapshot['components']['pullback'] = {"value": f"RSI Cross Up {self.p['rsi_oversold']}", "result": c3_pullback,
                                              "score": 1 if c3_pullback else 0}

        snapshot['total_score'] = buy_score

        if buy_score >= self.p['buy_score_threshold']:
            c4_strength = adx_val > adx_thresh
            snapshot['components']['strength'] = {"value": f"ADX > {adx_thresh}", "result": c4_strength}
            if c4_strength:
                y_values = ma_short_series.iloc[-self.p['slope_lookback_period']:].values
                if len(y_values) == self.p['slope_lookback_period']:
                    slope = np.polyfit(np.arange(len(y_values)), y_values, 1)[0]
                    c5_slope = slope > self.p['slope_min_threshold']
                    snapshot['components']['slope'] = {"value": f"Slope > {self.p['slope_min_threshold']}",
                                                       "result": c5_slope, "slope_value": slope}
                    if c5_slope:
                        final_signal = "POTENTIAL_BUY"

        suggested_sl = latest_close - (latest.get('vol_atr', 0) * atr_sl_mult)

        return {
            "overall_signal": final_signal,
            "exit_signal": None,
            "current_price": latest_close,
            "risk_management": {
                "suggested_sl": suggested_sl,
                "take_profit_condition": "Trend reversal (short MA crosses below long MA)"
            },
            "snapshot": snapshot
        }