import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Tuple

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
    [FINAL PRODUCTION VERSION]
    The final, most advanced strategy. It uses a sophisticated regime filter
    to switch between long-only and short-only modes, and provides detailed
    reasons for its decisions.
    """

    def __init__(self, params: Dict[str, Any]):
        self.p = params
        # Set defaults for any missing parameters to ensure robustness
        self.p.setdefault('commander_ma_period', 200)
        self.p.setdefault('vol_filter_atr_period', 100)
        self.p.setdefault('vol_filter_lookback', 50)
        self.p.setdefault('buy_score_threshold', 3)
        self.p.setdefault('sell_score_threshold', -3)
        # ... other defaults can be added here if needed

    async def generate_signals(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, Any] | None:
        df_signal = _clean_data(df_signal.copy())
        df_regime = _clean_data(df_regime.copy())

        indicators = self._calculate_all_indicators(df_signal, df_regime)
        if not indicators:
            return {"overall_signal": "INDICATOR_ERROR", "current_price": df_signal['close'].iloc[-1]}

        latest = {name: s.iloc[-1] for name, s in indicators.items() if not s.empty and pd.notna(s.iloc[-1])}
        previous = {name: s.iloc[-2] for name, s in indicators.items() if len(s) > 1 and pd.notna(s.iloc[-2])}
        latest_close = df_signal['close'].iloc[-1]

        market_regime = self._get_market_regime(latest, df_regime)

        score_details = self._get_score_details(latest, previous, indicators)

        final_signal, rejection_reason = self._get_final_decision(market_regime, score_details)

        exit_signal = self._get_exit_signal(latest)
        suggested_sl = self._calculate_stop_loss(final_signal, latest_close, latest)

        return {
            "overall_signal": final_signal,
            "exit_signal": exit_signal,
            "rejection_reason": rejection_reason,
            "current_price": latest_close,
            "market_regime": market_regime,
            "score_details": score_details,
            "risk_management": {"suggested_sl": suggested_sl}
        }

    def _calculate_all_indicators(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, pd.Series]:
        i = {}
        # Signal TF
        i['vol_atr'] = df_signal.ta.atr(length=self.p['vol_atr_period'], append=False)
        i['low_vol_ma_short'] = df_signal.ta.sma(length=self.p['low_vol_ma_short'])
        i['low_vol_ma_long'] = df_signal.ta.sma(length=self.p['low_vol_ma_long'])
        i['high_vol_ma_short'] = df_signal.ta.sma(length=self.p['high_vol_ma_short'])
        i['high_vol_ma_long'] = df_signal.ta.sma(length=self.p['high_vol_ma_long'])
        adx_df = df_signal.ta.adx(length=self.p['adx_period'])
        if adx_df is not None: i['adx'] = adx_df[f'ADX_{self.p["adx_period"]}']
        i['rsi'] = df_signal.ta.rsi(length=self.p['rsi_period'])
        macd_df = df_signal.ta.macd(fast=self.p['macd_fast'], slow=self.p['macd_slow'], signal=self.p['macd_signal'])
        if macd_df is not None:
            i['macd'] = macd_df[f'MACD_{self.p["macd_fast"]}_{self.p["macd_slow"]}_{self.p["macd_signal"]}']
            i['macd_signal'] = macd_df[f'MACDs_{self.p["macd_fast"]}_{self.p["macd_slow"]}_{self.p["macd_signal"]}']
        df_with_atr = df_signal.copy();
        df_with_atr['atr'] = i.get('vol_atr')
        i['vol_atr_ma'] = df_with_atr.ta.sma(close='atr', length=self.p['vol_atr_ma_period'])
        # Regime TF
        i['commander_ma'] = df_regime.ta.sma(length=self.p['commander_ma_period'])
        regime_atr = df_regime.ta.atr(length=self.p['vol_filter_atr_period'])
        if regime_atr is not None:
            i['regime_atr'] = regime_atr
            i['regime_atr_highest'] = regime_atr.rolling(window=self.p['vol_filter_lookback']).max()
        return i

    def _get_market_regime(self, latest: Dict, df_regime: pd.DataFrame) -> str:
        if latest.get('regime_atr', 0) >= latest.get('regime_atr_highest', float('inf')): return "CHAOS"
        if df_regime['close'].iloc[-1] > latest.get('commander_ma', float('inf')): return "BULL"
        if df_regime['close'].iloc[-1] < latest.get('commander_ma', 0): return "BEAR"
        return "CHOPPY"

    def _get_score_details(self, latest: Dict, previous: Dict, indicators: Dict) -> Dict[str, Any]:
        details = {}
        is_high_vol = latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0)
        ma_short, ma_long, adx, adx_thresh = (latest.get('high_vol_ma_short', 0), latest.get('high_vol_ma_long', 0),
                                              latest.get('adx', 0),
                                              self.p['high_vol_adx_threshold']) if is_high_vol else (
            latest.get('low_vol_ma_short', 0), latest.get('low_vol_ma_long', 0), latest.get('adx', 0),
            self.p['low_vol_adx_threshold'])

        details['ma_cross_score'] = 1 if ma_short > ma_long else -1
        details['macd_cross_score'] = 2 if previous.get('macd', 0) < previous.get('macd_signal', 0) and latest.get(
            'macd', 0) > latest.get('macd_signal', 0) else -2 if previous.get('macd', 0) > previous.get('macd_signal',
                                                                                                        0) and latest.get(
            'macd', 0) < latest.get('macd_signal', 0) else 0
        details['rsi_score'] = 1 if previous.get('rsi', 50) < self.p['rsi_oversold'] and latest.get('rsi', 50) > self.p[
            'rsi_oversold'] else -1 if previous.get('rsi', 50) > self.p['rsi_overbought'] and latest.get('rsi', 50) < \
                                       self.p['rsi_overbought'] else 0
        details['total_score'] = details['ma_cross_score'] + details['macd_cross_score'] + details['rsi_score']

        ma_series = indicators['high_vol_ma_short' if is_high_vol else 'low_vol_ma_short']
        y_vals = ma_series.iloc[-self.p['slope_lookback_period']:].values
        details['slope'] = np.polyfit(np.arange(len(y_vals)), y_vals, 1)[0] if len(y_vals) == self.p[
            'slope_lookback_period'] else 0

        details['adx'] = adx
        details['adx_ok'] = adx > adx_thresh
        details['slope_ok_long'] = details['slope'] > self.p['slope_min_threshold']
        details['slope_ok_short'] = details['slope'] < -self.p['slope_min_threshold']

        return details

    def _get_final_decision(self, regime: str, scores: Dict) -> Tuple[str, str | None]:
        if regime in ["CHAOS", "CHOPPY"]:
            return "NEUTRAL", f"Market regime is '{regime}', all new entries are forbidden."

        if regime == "BULL":
            if scores['total_score'] < self.p['buy_score_threshold']:
                return "NEUTRAL", f"Score of {scores['total_score']} did not meet BUY threshold of {self.p['buy_score_threshold']}."
            if not scores['adx_ok']:
                return "REJECTED", f"ADX ({scores['adx']:.2f}) is below strength threshold."
            if not scores['slope_ok_long']:
                return "REJECTED", f"MA slope ({scores['slope']:.2f}) is not positive."
            return "POTENTIAL_BUY", None  # All checks passed

        if regime == "BEAR":
            if scores['total_score'] > self.p['sell_score_threshold']:
                return "NEUTRAL", f"Score of {scores['total_score']} did not meet SELL threshold of {self.p['sell_score_threshold']}."
            if not scores['adx_ok']:
                return "REJECTED", f"ADX ({scores['adx']:.2f}) is below strength threshold."
            if not scores['slope_ok_short']:
                return "REJECTED", f"MA slope ({scores['slope']:.2f}) is not negative."
            return "POTENTIAL_SELL", None  # All checks passed

        return "NEUTRAL", "No valid trading regime detected."

    def _get_exit_signal(self, latest: Dict) -> str | None:
        is_high_vol = latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0)
        ma_short, ma_long = (latest.get('high_vol_ma_short', 0),
                             latest.get('high_vol_ma_long', 0)) if is_high_vol else (latest.get('low_vol_ma_short', 0),
                                                                                     latest.get('low_vol_ma_long', 0))
        if ma_short < ma_long: return "EXIT_LONG_DEATH_CROSS"
        if ma_short > ma_long: return "EXIT_SHORT_GOLDEN_CROSS"
        return None

    def _calculate_stop_loss(self, signal: str, price: float, latest: Dict) -> float | None:
        if signal not in ["POTENTIAL_BUY", "POTENTIAL_SELL"]: return None
        is_high_vol = latest.get('vol_atr', 0) > latest.get('vol_atr_ma', 0)
        atr_mult_key = 'high_vol_atr_sl_multiplier' if is_high_vol else 'low_vol_atr_sl_multiplier'
        atr_mult = self.p.get(atr_mult_key, 2.0)
        risk_dist = latest.get('vol_atr', 0) * atr_mult
        return price - risk_dist if signal == "POTENTIAL_BUY" else price + risk_dist