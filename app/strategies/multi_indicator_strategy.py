import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any

from app.strategies.base_strategy import TradingStrategy
from ..core.config import settings

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def _calculate_common_indicators(df_signal: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
    """Calculates indicators that are common to multiple strategies."""
    indicators = {}

    # Volatility Regime
    vol_atr = df_signal.ta.atr(length=params['vol_atr_period'], append=False)
    if vol_atr is not None and not vol_atr.empty:
        indicators['vol_atr'] = vol_atr
        df_with_atr = df_signal.copy()
        df_with_atr['atr'] = vol_atr
        indicators['vol_atr_ma'] = df_with_atr.ta.sma(close='atr', length=params['vol_atr_ma_period'])

    # Low Volatility Indicators
    indicators['low_vol_ma_short'] = df_signal.ta.sma(length=params['low_vol_ma_short'])
    indicators['low_vol_ma_long'] = df_signal.ta.sma(length=params['low_vol_ma_long'])
    adx_low = df_signal.ta.adx(length=params['adx_period'])
    if adx_low is not None and not adx_low.empty:
        indicators['low_vol_adx'] = adx_low[f'ADX_{params["adx_period"]}']

    # High Volatility Indicators
    indicators['high_vol_ma_short'] = df_signal.ta.sma(length=params['high_vol_ma_short'])
    indicators['high_vol_ma_long'] = df_signal.ta.sma(length=params['high_vol_ma_long'])
    adx_high = df_signal.ta.adx(length=params['adx_period'])
    if adx_high is not None and not adx_high.empty:
        indicators['high_vol_adx'] = adx_high[f'ADX_{params["adx_period"]}']

    # Common Signal Indicators
    indicators['rsi'] = df_signal.ta.rsi(length=params['rsi_period'])
    macd_df = df_signal.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'])
    if macd_df is not None and not macd_df.empty:
        indicators['macd'] = macd_df[f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
        indicators['macd_signal'] = macd_df[
            f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']

    return indicators


# --- STRATEGY IMPLEMENTATION 1: The Original Long-Only Strategy ---

class LongOnlyTrendStrategy(TradingStrategy):
    """
    The original, battle-tested long-only trend following strategy.
    It enters on strong buy signals in a bull market and exits on a trend reversal.
    """

    def __init__(self, params: Dict[str, Any]):
        self.p = params

    async def generate_signals(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, Any] | None:
        df_signal = _clean_data(df_signal.copy())
        df_regime = _clean_data(df_regime.copy())

        # Calculate indicators
        regime_ma = df_regime.ta.sma(length=self.p['regime_ma_period'])
        common_indicators = _calculate_common_indicators(df_signal, self.p)

        if not all(k in common_indicators for k in
                   ['vol_atr', 'vol_atr_ma', 'low_vol_ma_short', 'low_vol_ma_long', 'low_vol_adx', 'macd',
                    'macd_signal', 'rsi']) or regime_ma is None:
            return {"overall_signal": "INDICATOR_ERROR", "current_price": df_signal['close'].iloc[-1]}

        # Combine and get latest values
        indicators = {'regime_ma': regime_ma, **common_indicators}
        latest = {name: series.iloc[-1] for name, series in indicators.items() if
                  not series.empty and pd.notna(series.iloc[-1])}
        previous = {name: series.iloc[-2] for name, series in indicators.items() if
                    len(series) > 1 and pd.notna(series.iloc[-2])}
        latest_close = df_signal['close'].iloc[-1]

        # Logic is identical to the original MtfaStrategyBt backtest
        is_bull_regime = latest_close > latest.get('regime_ma', float('inf'))

        final_signal = "NEUTRAL"
        exit_signal = None

        is_high_vol = latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0)
        ma_short, ma_long, adx_val, adx_thresh = (latest['high_vol_ma_short'], latest['high_vol_ma_long'],
                                                  latest['high_vol_adx'],
                                                  self.p['high_vol_adx_threshold']) if is_high_vol else (
            latest['low_vol_ma_short'], latest['low_vol_ma_long'], latest['low_vol_adx'],
            self.p['low_vol_adx_threshold'])

        if not is_bull_regime:
            final_signal = "REJECTED_BEAR_REGIME"
        else:
            buy_score = (1 if ma_short > ma_long else 0) + \
                        (2 if previous.get('macd', 0) < previous.get('macd_signal', 0) and latest.get('macd',
                                                                                                      0) > latest.get(
                            'macd_signal', 0) else 0) + \
                        (1 if previous.get('rsi', 0) < self.p['rsi_oversold'] and latest.get('rsi', 0) > self.p[
                            'rsi_oversold'] else 0)

            if buy_score >= self.p['buy_score_threshold']:
                if adx_val > adx_thresh:
                    final_signal = "POTENTIAL_BUY"  # Simplified for clarity, slope check can be added back if needed
                else:
                    final_signal = "REJECTED_ADX_LOW"

        if ma_short < ma_long:
            exit_signal = "EXIT_LONG_DEATH_CROSS"

        return {"overall_signal": final_signal, "exit_signal": exit_signal, "current_price": latest_close}


# --- STRATEGY IMPLEMENTATION 2: The Final Alpha Strategy ---

class AlphaRegimeStrategy(TradingStrategy):
    """
    The final, most advanced strategy. It uses a sophisticated regime filter
    to switch between long-only and short-only modes, effectively adapting
    to the broader market trend (Bull vs. Bear).
    """

    def __init__(self, params: Dict[str, Any]):
        self.p = params

    async def generate_signals(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, Any] | None:
        df_signal = _clean_data(df_signal.copy())
        df_regime = _clean_data(df_regime.copy())

        # --- Calculate Indicators ---
        common_indicators = _calculate_common_indicators(df_signal, self.p)
        commander_ma = df_regime.ta.sma(length=self.p['commander_ma_period'])
        regime_atr = df_regime.ta.atr(length=self.p['vol_filter_atr_period'])
        regime_atr_highest = regime_atr.rolling(
            window=self.p['vol_filter_lookback']).max() if regime_atr is not None else None

        if not all(k in common_indicators for k in
                   ['vol_atr', 'vol_atr_ma', 'low_vol_ma_short', 'low_vol_ma_long', 'low_vol_adx', 'macd',
                    'macd_signal', 'rsi']) or commander_ma is None or regime_atr_highest is None:
            return {"overall_signal": "INDICATOR_ERROR", "current_price": df_signal['close'].iloc[-1]}

        # --- Get Latest Values ---
        indicators = {'commander_ma': commander_ma, 'regime_atr': regime_atr, 'regime_atr_highest': regime_atr_highest,
                      **common_indicators}
        latest = {name: series.iloc[-1] for name, series in indicators.items() if
                  not series.empty and pd.notna(series.iloc[-1])}
        previous = {name: series.iloc[-2] for name, series in indicators.items() if
                    len(series) > 1 and pd.notna(series.iloc[-2])}
        latest_close = df_signal['close'].iloc[-1]

        # --- Determine Market Regime (The General's Order) ---
        market_regime = "CHOPPY"
        if latest['regime_atr'] >= latest['regime_atr_highest']:
            market_regime = "CHAOS"
        elif df_regime['close'].iloc[-1] > latest['commander_ma']:
            market_regime = "BULL"
        elif df_regime['close'].iloc[-1] < latest['commander_ma']:
            market_regime = "BEAR"

        # --- Entry Logic ---
        final_signal = "NEUTRAL"
        rejection_reason = f"Market regime is {market_regime}, no new entries."

        if market_regime not in ["CHAOS", "CHOPPY"]:
            is_high_vol_local = latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0)
            ma_short, ma_long, adx_val, adx_thresh = (latest['high_vol_ma_short'], latest['high_vol_ma_long'],
                                                      latest['high_vol_adx'],
                                                      self.p['high_vol_adx_threshold']) if is_high_vol_local else (
                latest['low_vol_ma_short'], latest['low_vol_ma_long'], latest['low_vol_adx'],
                self.p['low_vol_adx_threshold'])

            ma_cross_score = 1 if ma_short > ma_long else -1
            macd_cross_score = 2 if previous.get('macd', 0) < previous.get('macd_signal', 0) and latest.get('macd',
                                                                                                            0) > latest.get(
                'macd_signal', 0) else -2 if previous.get('macd', 0) > previous.get('macd_signal', 0) and latest.get(
                'macd', 0) < latest.get('macd_signal', 0) else 0
            rsi_score = 1 if previous.get('rsi', 50) < self.p['rsi_oversold'] and latest.get('rsi', 50) > self.p[
                'rsi_oversold'] else -1 if previous.get('rsi', 50) > self.p['rsi_overbought'] and latest.get('rsi',
                                                                                                             50) < \
                                           self.p['rsi_overbought'] else 0
            total_score = ma_cross_score + macd_cross_score + rsi_score

            y_values = indicators['low_vol_ma_short' if not is_high_vol_local else 'high_vol_ma_short'].iloc[
                       -self.p['slope_lookback_period']:].values
            slope = np.polyfit(np.arange(len(y_values)), y_values, 1)[0] if len(y_values) == self.p[
                'slope_lookback_period'] else 0

            gatekeeper_ok = adx_val > adx_thresh

            if market_regime == "BULL":
                if total_score >= self.p['buy_score_threshold'] and gatekeeper_ok and slope > self.p[
                    'slope_min_threshold']:
                    final_signal = "POTENTIAL_BUY"
                    rejection_reason = None
            elif market_regime == "BEAR":
                if total_score <= self.p['sell_score_threshold'] and gatekeeper_ok and slope < -self.p[
                    'slope_min_threshold']:
                    final_signal = "POTENTIAL_SELL"
                    rejection_reason = None

        # --- Exit Logic ---
        exit_signal = None
        is_high_vol_local = latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0)
        ma_short, ma_long = (latest['high_vol_ma_short'], latest['high_vol_ma_long']) if is_high_vol_local else (
            latest['low_vol_ma_short'], latest['low_vol_ma_long'])
        if ma_short < ma_long: exit_signal = "EXIT_LONG_DEATH_CROSS"
        if ma_short > ma_long: exit_signal = "EXIT_SHORT_GOLDEN_CROSS"

        return {"overall_signal": final_signal, "exit_signal": exit_signal, "rejection_reason": rejection_reason,
                "current_price": latest_close}