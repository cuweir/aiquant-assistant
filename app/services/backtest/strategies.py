# services/backtest/strategies.py

import backtrader as bt
import numpy as np
from ...core.config import settings


class BuyAndHold(bt.Strategy):
    """
    A simple benchmark strategy that buys on the first available bar and holds.
    """

    def __init__(self):
        self.bought = False

    def next(self):
        if not self.bought:
            size = (self.broker.get_cash() * 0.95) / self.data.close[0]
            self.buy(size=size)
            self.bought = True


class MtfaStrategyBt(bt.Strategy):
    """
    This is the ORIGINAL strategy, using only a trend reversal (death cross) as an exit signal.
    """
    params = (
        ('regime_ma_period', 200),
        ('buy_score_threshold', 3),
        ('slope_lookback_period', 5),
        ('slope_min_threshold', 0.0),
        ('vol_atr_period', 14),
        ('vol_atr_ma_period', 100),

        ('low_vol_ma_short', 25), ('low_vol_ma_long', 80),
        ('low_vol_adx_threshold', 30), ('low_vol_atr_sl_multiplier', 2.0),

        ('high_vol_ma_short', 15), ('high_vol_ma_long', 40),
        ('high_vol_adx_threshold', 30), ('high_vol_atr_sl_multiplier', 2.5),

        ('rsi_period', 14), ('rsi_oversold', 40),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('adx_period', 14),
    )

    def __init__(self):
        self.d_signal = self.datas[0];
        self.d_regime = self.datas[1];
        self.dataclose = self.d_signal.close
        self.regime_ma = bt.indicators.SimpleMovingAverage(self.d_regime, period=self.p.regime_ma_period)
        self.vol_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.vol_atr_period)
        self.vol_atr_ma = bt.indicators.SimpleMovingAverage(self.vol_atr, period=self.p.vol_atr_ma_period)
        self.low_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_short)
        self.low_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_long)
        self.low_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period)
        self.high_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_short)
        self.high_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_long)
        self.high_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period)
        self.rsi = bt.indicators.RSI(self.d_signal, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(self.d_signal, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow,
                                       period_signal=self.p.macd_signal)
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.order = None;
        self.stop_loss_price = None  # This version uses a conceptual SL, not a real order

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()} - {txt}') # Suppress logging for cleaner comparison runs

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        self.order = None

    def next(self):
        if self.order: return

        is_bull_regime = self.d_regime.close[0] > self.regime_ma[0]

        if not self.position:
            if is_bull_regime:
                is_high_volatility = self.vol_atr[0] > self.vol_atr_ma[0]
                if is_high_volatility:
                    ma_short, ma_long, adx, adx_thresh, atr_sl_mult = (
                        self.high_vol_ma_short, self.high_vol_ma_long, self.high_vol_adx,
                        self.p.high_vol_adx_threshold, self.p.high_vol_atr_sl_multiplier
                    )
                else:
                    ma_short, ma_long, adx, adx_thresh, atr_sl_mult = (
                        self.low_vol_ma_short, self.low_vol_ma_long, self.low_vol_adx,
                        self.p.low_vol_adx_threshold, self.p.low_vol_atr_sl_multiplier
                    )

                buy_score = 0
                if ma_short[0] > ma_long[0]: buy_score += 1
                if self.macd_crossover[0] > 0: buy_score += 2
                if self.rsi[-1] < self.p.rsi_oversold and self.rsi[0] > self.p.rsi_oversold: buy_score += 1

                if buy_score >= self.p.buy_score_threshold:
                    if adx.adx[0] > adx_thresh:
                        y_values = ma_short.get(size=self.p.slope_lookback_period)
                        if len(y_values) < self.p.slope_lookback_period: return
                        slope = np.polyfit(np.arange(len(y_values)), y_values, 1)[0]
                        if slope > self.p.slope_min_threshold:
                            self.stop_loss_price = self.dataclose[0] - (self.vol_atr[0] * atr_sl_mult)
                            self.order = self.buy()
        else:
            if self.position.size > 0:
                is_high_vol = self.vol_atr[0] > self.vol_atr_ma[0]
                ma_short = self.high_vol_ma_short if is_high_vol else self.low_vol_ma_short
                ma_long = self.high_vol_ma_long if is_high_vol else self.low_vol_ma_long

                # The ONLY exit condition is the death cross
                if ma_short[0] < ma_long[0]:
                    self.order = self.close()


class MtfaStrategyFinal(bt.Strategy):
    """
    [V4 - FINAL STRATEGY]
    Uses a powerful volatility filter on a higher timeframe to switch between
    "Normal Mode" and "Safe Mode".
    - In "Safe Mode" (high volatility spikes): All new buy signals are ignored.
    - In "Normal Mode": The original, trend-following strategy is active.
    """
    params = (
        # Same entry and exit params as the original strategy
        ('regime_ma_period', 200), ('buy_score_threshold', 3),
        ('slope_lookback_period', 5), ('slope_min_threshold', 0.0),
        ('vol_atr_period', 14), ('vol_atr_ma_period', 100),
        ('low_vol_ma_short', 25), ('low_vol_ma_long', 80),
        ('low_vol_adx_threshold', 30), ('low_vol_atr_sl_multiplier', 2.0),
        ('high_vol_ma_short', 15), ('high_vol_ma_long', 40),
        ('high_vol_adx_threshold', 30), ('high_vol_atr_sl_multiplier', 2.5),
        ('rsi_period', 14), ('rsi_oversold', 40),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('adx_period', 14),

        # [NEW] Params for the Volatility Filter
        ('vol_filter_atr_period', 100),  # Long-term ATR on the 4h chart
        ('vol_filter_lookback', 50),  # How far back to look for a new ATR high
    )

    def __init__(self):
        # Indicators
        self.d_signal = self.datas[0];
        self.d_regime = self.datas[1];
        self.dataclose = self.d_signal.close;
        self.vol_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.vol_atr_period);
        self.vol_atr_ma = bt.indicators.SimpleMovingAverage(self.vol_atr, period=self.p.vol_atr_ma_period);
        self.low_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_short);
        self.low_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_long);
        self.low_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.high_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_short);
        self.high_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_long);
        self.high_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.rsi = bt.indicators.RSI(self.d_signal, period=self.p.rsi_period);
        self.macd = bt.indicators.MACD(self.d_signal, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow,
                                       period_signal=self.p.macd_signal);
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal);

        # [NEW] Volatility Filter Indicator
        self.regime_ma = bt.indicators.SimpleMovingAverage(self.d_regime, period=self.p.regime_ma_period)

        self.regime_atr = bt.indicators.AverageTrueRange(self.d_regime, period=self.p.vol_filter_atr_period)
        self.regime_atr_highest = bt.indicators.Highest(self.regime_atr, period=self.p.vol_filter_lookback)

        # State
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def next(self):
        if self.order: return

        # --- [THE ULTIMATE FILTER] ---
        # If the 4h ATR just made a new N-period high, enter "Safe Mode" and do not trade.
        # We check [-1] because we want to react to the breakout on the *next* bar.
        is_safe_mode = self.regime_atr[-1] >= self.regime_atr_highest[-1]

        # --- Entry Logic ---
        if not self.position:
            if is_safe_mode:
                return  # Do not enter new trades in safe mode

            # If not in safe mode, proceed with the original entry logic
            is_bull_regime_price = self.d_regime.close[0] > self.regime_ma[0]
            if is_bull_regime_price:
                is_high_vol_local = self.vol_atr[0] > self.vol_atr_ma[0]
                ma_short, ma_long, adx, adx_thresh = (self.high_vol_ma_short, self.high_vol_ma_long, self.high_vol_adx,
                                                      self.p.high_vol_adx_threshold) if is_high_vol_local else (
                    self.low_vol_ma_short, self.low_vol_ma_long, self.low_vol_adx, self.p.low_vol_adx_threshold)

                buy_score = (1 if ma_short[0] > ma_long[0] else 0) + (2 if self.macd_crossover[0] > 0 else 0) + (
                    1 if self.rsi[-1] < self.p.rsi_oversold and self.rsi[0] > self.p.rsi_oversold else 0)

                if buy_score >= self.p.buy_score_threshold:
                    if adx.adx[0] > adx_thresh:
                        y_values = ma_short.get(size=self.p.slope_lookback_period)
                        if len(y_values) < self.p.slope_lookback_period: return
                        if np.polyfit(np.arange(len(y_values)), y_values, 1)[0] > self.p.slope_min_threshold:
                            self.order = self.buy()

        # --- Exit Logic (remains the same as the original strategy) ---
        elif self.position.size > 0:
            is_high_vol_local = self.vol_atr[0] > self.vol_atr_ma[0]
            ma_short, ma_long = (self.high_vol_ma_short, self.high_vol_ma_long) if is_high_vol_local else (
                self.low_vol_ma_short, self.low_vol_ma_long)
            if ma_short[0] < ma_long[0]:
                self.order = self.close()


class ShortTermStrategy(bt.Strategy):
    """
    [V5 - SHORT-TERM EXPERIMENT]
    Adapts the final strategy for shorter timeframes (e.g., 15m signal, 1h regime).
    - Uses proportionally scaled parameters.
    - Employs a fixed, quick take-profit exit.
    """
    params = (
        # Parameters will be scaled down from the runner
        ('regime_ma_period', 50),  # 200 / 4
        ('buy_score_threshold', 3),
        ('slope_lookback_period', 5),
        ('slope_min_threshold', 0.0),
        ('vol_atr_period', 14),
        ('vol_atr_ma_period', 25),  # 100 / 4
        ('low_vol_ma_short', 6),  # 25 / 4
        ('low_vol_ma_long', 20),  # 80 / 4
        ('low_vol_adx_threshold', 30),
        ('low_vol_atr_sl_multiplier', 2.0),
        ('high_vol_ma_short', 4),  # 15 / 4
        ('high_vol_ma_long', 10),  # 40 / 4
        ('high_vol_adx_threshold', 30),
        ('high_vol_atr_sl_multiplier', 2.5),
        ('rsi_period', 14),
        ('rsi_oversold', 40),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('adx_period', 14),

        # Short-term specific exit param
        ('quick_tp_rr_ratio', 1.5),
    )

    def __init__(self):
        # Indicators with scaled params
        self.d_signal = self.datas[0];
        self.d_regime = self.datas[1];
        self.dataclose = self.d_signal.close;
        self.regime_ma = bt.indicators.SimpleMovingAverage(self.d_regime, period=self.p.regime_ma_period)
        self.vol_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.vol_atr_period)
        self.vol_atr_ma = bt.indicators.SimpleMovingAverage(self.vol_atr, period=self.p.vol_atr_ma_period)
        self.low_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_short)
        self.low_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_long)
        self.low_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period)
        self.high_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_short)
        self.high_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_long)
        self.high_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period)
        self.rsi = bt.indicators.RSI(self.d_signal, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(self.d_signal, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow,
                                       period_signal=self.p.macd_signal)
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        # State
        self.order = None
        self.trade_state = {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status == order.Completed and order.isbuy():
            is_high_vol = self.vol_atr[0] > self.vol_atr_ma[0]
            atr_sl_mult = self.p.high_vol_atr_sl_multiplier if is_high_vol else self.p.low_vol_atr_sl_multiplier
            risk_dist = self.vol_atr[0] * atr_sl_mult
            self.trade_state = {
                'stop_price': order.executed.price - risk_dist,
                'tp_price': order.executed.price + (risk_dist * self.p.quick_tp_rr_ratio)
            }
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed: self.trade_state = {}

    def next(self):
        if self.order: return

        # --- In-Trade Management: Quick Exit ---
        if self.position:
            if self.dataclose[0] <= self.trade_state['stop_price']:
                self.order = self.close()
            elif self.dataclose[0] >= self.trade_state['tp_price']:
                self.order = self.close()
            return

        # --- Entry Logic ---
        is_bull_regime = self.dataclose[0] > self.regime_ma[0]
        if not is_bull_regime: return

        is_high_vol = self.vol_atr[0] > self.vol_atr_ma[0]
        ma_short, ma_long, adx, adx_thresh = (self.high_vol_ma_short, self.high_vol_ma_long, self.high_vol_adx,
                                              self.p.high_vol_adx_threshold) if is_high_vol else (self.low_vol_ma_short,
                                                                                                  self.low_vol_ma_long,
                                                                                                  self.low_vol_adx,
                                                                                                  self.p.low_vol_adx_threshold)

        buy_score = (1 if ma_short[0] > ma_long[0] else 0) + (2 if self.macd_crossover[0] > 0 else 0) + (
            1 if self.rsi[-1] < self.p.rsi_oversold and self.rsi[0] > self.p.rsi_oversold else 0)

        if buy_score >= self.p.buy_score_threshold:
            if adx.adx[0] > adx_thresh:
                y_values = ma_short.get(size=self.p.slope_lookback_period)
                if len(y_values) < self.p.slope_lookback_period: return
                if np.polyfit(np.arange(len(y_values)), y_values, 1)[0] > self.p.slope_min_threshold:
                    self.order = self.buy()


class MtfaStrategyFinalBilateral(bt.Strategy):
    params = (
        ('regime_ma_period', 200),
        ('buy_score_threshold', 3), ('sell_score_threshold', -3),  # Symmetrical thresholds
        ('slope_lookback_period', 5), ('slope_min_threshold', 0.0),
        ('vol_atr_period', 14), ('vol_atr_ma_period', 100),
        ('low_vol_ma_short', 25), ('low_vol_ma_long', 80),
        ('low_vol_adx_threshold', 30), ('low_vol_atr_sl_multiplier', 2.0),
        ('high_vol_ma_short', 15), ('high_vol_ma_long', 40),
        ('high_vol_adx_threshold', 30), ('high_vol_atr_sl_multiplier', 2.5),
        ('rsi_period', 14), ('rsi_oversold', 40), ('rsi_overbought', 60),  # Symmetrical RSI
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('adx_period', 14),
        ('vol_filter_atr_period', 100), ('vol_filter_lookback', 50),
    )

    def __init__(self):
        # Indicators
        self.d_signal = self.datas[0];
        self.d_regime = self.datas[1];
        self.dataclose = self.d_signal.close;
        self.vol_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.vol_atr_period);
        self.vol_atr_ma = bt.indicators.SimpleMovingAverage(self.vol_atr, period=self.p.vol_atr_ma_period);
        self.low_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_short);
        self.low_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_long);
        self.low_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.high_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_short);
        self.high_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_long);
        self.high_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.rsi = bt.indicators.RSI(self.d_signal, period=self.p.rsi_period);
        self.macd = bt.indicators.MACD(self.d_signal, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow,
                                       period_signal=self.p.macd_signal);
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal);

        # Volatility Filter
        self.regime_atr = bt.indicators.AverageTrueRange(self.d_regime, period=self.p.vol_filter_atr_period)
        self.regime_atr_highest = bt.indicators.Highest(self.regime_atr, period=self.p.vol_filter_lookback)

        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def next(self):
        if self.order: return

        is_safe_mode = self.regime_atr[0] >= self.regime_atr_highest[-1]

        if not self.position:
            if is_safe_mode: return

            is_high_vol_local = self.vol_atr[0] > self.vol_atr_ma[0]
            ma_short, ma_long, adx, adx_thresh = (self.high_vol_ma_short, self.high_vol_ma_long, self.high_vol_adx,
                                                  self.p.high_vol_adx_threshold) if is_high_vol_local else (
                self.low_vol_ma_short, self.low_vol_ma_long, self.low_vol_adx, self.p.low_vol_adx_threshold)

            # --- Symmetrical Scoring ---
            ma_cross_score = 1 if ma_short[0] > ma_long[0] else -1
            macd_cross_score = 2 if self.macd_crossover[0] > 0 else -2 if self.macd_crossover[0] < 0 else 0

            rsi_buy_signal = self.rsi[-1] < self.p.rsi_oversold and self.rsi[0] > self.p.rsi_oversold
            rsi_sell_signal = self.rsi[-1] > self.p.rsi_overbought and self.rsi[0] < self.p.rsi_overbought
            rsi_score = 1 if rsi_buy_signal else -1 if rsi_sell_signal else 0

            total_score = ma_cross_score + macd_cross_score + rsi_score

            y_values = ma_short.get(size=self.p.slope_lookback_period)
            if len(y_values) < self.p.slope_lookback_period: return
            slope = np.polyfit(np.arange(len(y_values)), y_values, 1)[0]

            # --- Entry Logic ---
            if total_score >= self.p.buy_score_threshold and adx.adx[
                0] > adx_thresh and slope > self.p.slope_min_threshold:
                self.order = self.buy()
            elif total_score <= self.p.sell_score_threshold and adx.adx[
                0] > adx_thresh and slope < -self.p.slope_min_threshold:
                self.order = self.sell()

        # --- Symmetrical Exit Logic ---
        else:
            is_high_vol_local = self.vol_atr[0] > self.vol_atr_ma[0]
            ma_short, ma_long = (self.high_vol_ma_short, self.high_vol_ma_long) if is_high_vol_local else (
                self.low_vol_ma_short, self.low_vol_ma_long)

            if self.position.size > 0 and ma_short[0] < ma_long[0]:  # Exit Long on Death Cross
                self.order = self.close()
            elif self.position.size < 0 and ma_short[0] > ma_long[0]:  # Exit Short on Golden Cross
                self.order = self.close()


class MtfaStrategyAlpha(bt.Strategy):
    params = (
        ('buy_score_threshold', 3), ('sell_score_threshold', -3),
        ('slope_lookback_period', 5), ('slope_min_threshold', 0.0),
        ('vol_atr_period', 14), ('vol_atr_ma_period', 100),
        ('low_vol_ma_short', 25), ('low_vol_ma_long', 80),
        ('low_vol_adx_threshold', 25),
        ('high_vol_ma_short', 15), ('high_vol_ma_long', 40),
        ('high_vol_adx_threshold', 25),
        ('rsi_period', 14), ('rsi_oversold', 35), ('rsi_overbought', 65),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('adx_period', 14),
        ('commander_ma_period', 200),
        ('vol_filter_atr_period', 100), ('vol_filter_lookback', 50),
    )

    def __init__(self):
        # Indicators
        self.d_signal = self.datas[0];
        self.d_regime = self.datas[1];
        self.dataclose = self.d_signal.close;
        self.vol_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.vol_atr_period);
        self.vol_atr_ma = bt.indicators.SimpleMovingAverage(self.vol_atr, period=self.p.vol_atr_ma_period);
        self.low_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_short);
        self.low_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_long);
        self.low_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.high_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_short);
        self.high_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_long);
        self.high_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.rsi = bt.indicators.RSI(self.d_signal, period=self.p.rsi_period);
        self.macd = bt.indicators.MACD(self.d_signal, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow,
                                       period_signal=self.p.macd_signal);
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal);

        # The "Supreme Commander" Indicator
        self.commander_ma = bt.indicators.SimpleMovingAverage(self.d_regime, period=self.p.commander_ma_period)

        # The Volatility Filter
        self.regime_atr = bt.indicators.AverageTrueRange(self.d_regime, period=self.p.vol_filter_atr_period)
        self.regime_atr_highest = bt.indicators.Highest(self.regime_atr, period=self.p.vol_filter_lookback)

        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def get_market_regime(self):
        is_high_volatility = self.regime_atr[0] >= self.regime_atr_highest[-1]
        if is_high_volatility:
            return "CHAOS"

        if self.d_regime.close[0] > self.commander_ma[0]:
            return "BULL"
        elif self.d_regime.close[0] < self.commander_ma[0]:
            return "BEAR"

        return "CHOPPY"

    def next(self):
        if self.order: return

        market_regime = self.get_market_regime()

        if not self.position:
            if market_regime in ["CHAOS", "CHOPPY"]:
                return

            is_high_vol_local = self.vol_atr[0] > self.vol_atr_ma[0]
            ma_short, ma_long, adx, adx_thresh = (self.high_vol_ma_short, self.high_vol_ma_long, self.high_vol_adx,
                                                  self.p.high_vol_adx_threshold) if is_high_vol_local else (
                self.low_vol_ma_short, self.low_vol_ma_long, self.low_vol_adx, self.p.low_vol_adx_threshold)

            ma_cross_score = 1 if ma_short[0] > ma_long[0] else -1
            macd_cross_score = 2 if self.macd_crossover[0] > 0 else -2 if self.macd_crossover[0] < 0 else 0
            rsi_score = 1 if (self.rsi[-1] < self.p.rsi_oversold and self.rsi[0] > self.p.rsi_oversold) else -1 if (
                        self.rsi[-1] > self.p.rsi_overbought and self.rsi[0] < self.p.rsi_overbought) else 0
            total_score = ma_cross_score + macd_cross_score + rsi_score

            y_values = ma_short.get(size=self.p.slope_lookback_period)
            if len(y_values) < self.p.slope_lookback_period: return
            slope = np.polyfit(np.arange(len(y_values)), y_values, 1)[0]

            if market_regime == "BULL":
                if total_score >= self.p.buy_score_threshold and adx.adx[
                    0] > adx_thresh and slope > self.p.slope_min_threshold:
                    self.order = self.buy()
            elif market_regime == "BEAR":
                if total_score <= self.p.sell_score_threshold and adx.adx[
                    0] > adx_thresh and slope < -self.p.slope_min_threshold:
                    self.order = self.sell()

        else:
            is_high_vol_local = self.vol_atr[0] > self.vol_atr_ma[0]
            ma_short, ma_long = (self.high_vol_ma_short, self.high_vol_ma_long) if is_high_vol_local else (
                self.low_vol_ma_short, self.low_vol_ma_long)

            if self.position.size > 0 and ma_short[0] < ma_long[0]:
                self.order = self.close()
            elif self.position.size < 0 and ma_short[0] > ma_long[0]:
                self.order = self.close()


class MtfaStrategyOmega(bt.Strategy):
    params = (
        ('buy_score_threshold', 3), ('sell_score_threshold', -3),
        ('slope_lookback_period', 5), ('slope_min_threshold', 0.0),
        ('vol_atr_period', 14), ('vol_atr_ma_period', 100),
        ('low_vol_ma_short', 25), ('low_vol_ma_long', 80),
        ('low_vol_adx_threshold', 25),
        ('high_vol_ma_short', 15), ('high_vol_ma_long', 40),
        ('high_vol_adx_threshold', 25),
        ('rsi_period', 14), ('rsi_oversold', 35), ('rsi_overbought', 65),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('adx_period', 14),
        ('commander_ma_period', 200),
        ('vol_filter_atr_period', 100), ('vol_filter_lookback', 50),

        # [NEW] Sharpe Ratio Enhancement Params
        ('risk_per_trade', 0.02),  # Risk 2% of portfolio value per trade
        ('chandelier_atr_period', 22),  # ATR period for Chandelier Exit
        ('chandelier_atr_mult', 3.0),  # ATR multiplier for Chandelier Exit
    )

    def __init__(self):
        # Indicators
        self.d_signal = self.datas[0];
        self.d_regime = self.datas[1];
        self.dataclose = self.d_signal.close;
        self.vol_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.vol_atr_period);
        self.vol_atr_ma = bt.indicators.SimpleMovingAverage(self.vol_atr, period=self.p.vol_atr_ma_period);
        self.low_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_short);
        self.low_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_long);
        self.low_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.high_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_short);
        self.high_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_long);
        self.high_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period);
        self.rsi = bt.indicators.RSI(self.d_signal, period=self.p.rsi_period);
        self.macd = bt.indicators.MACD(self.d_signal, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow,
                                       period_signal=self.p.macd_signal);
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal);
        self.commander_ma = bt.indicators.SimpleMovingAverage(self.d_regime, period=self.p.commander_ma_period)
        self.regime_atr = bt.indicators.AverageTrueRange(self.d_regime, period=self.p.vol_filter_atr_period)
        self.regime_atr_highest = bt.indicators.Highest(self.regime_atr, period=self.p.vol_filter_lookback)

        # [NEW] Chandelier Exit Indicators
        self.chandelier_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.chandelier_atr_period)
        self.highest_high = bt.indicators.Highest(self.d_signal.high, period=self.p.chandelier_atr_period)
        self.lowest_low = bt.indicators.Lowest(self.d_signal.low, period=self.p.chandelier_atr_period)

        self.order = None
        self.trade_state = {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed: self.trade_state = {}

    def get_market_regime(self):
        if self.regime_atr[0] >= self.regime_atr_highest[-1]: return "CHAOS"
        if self.d_regime.close[0] > self.commander_ma[0]: return "BULL"
        if self.d_regime.close[0] < self.commander_ma[0]: return "BEAR"
        return "CHOPPY"

    def next(self):
        if self.order: return
        market_regime = self.get_market_regime()

        # --- In-Trade Management: Chandelier Exit ---
        if self.position:
            if self.position.size > 0:  # Long position
                chandelier_exit = self.highest_high[0] - self.chandelier_atr[0] * self.p.chandelier_atr_mult
                if self.dataclose[0] < chandelier_exit:
                    self.order = self.close()
            elif self.position.size < 0:  # Short position
                chandelier_exit = self.lowest_low[0] + self.chandelier_atr[0] * self.p.chandelier_atr_mult
                if self.dataclose[0] > chandelier_exit:
                    self.order = self.close()
            return

        # --- Entry Logic ---
        if market_regime in ["CHAOS", "CHOPPY"]: return

        is_high_vol_local = self.vol_atr[0] > self.vol_atr_ma[0]
        ma_short, ma_long, adx, adx_thresh = (self.high_vol_ma_short, self.high_vol_ma_long, self.high_vol_adx,
                                              self.p.high_vol_adx_threshold) if is_high_vol_local else (
            self.low_vol_ma_short, self.low_vol_ma_long, self.low_vol_adx, self.p.low_vol_adx_threshold)

        total_score = (1 if ma_short[0] > ma_long[0] else -1) + \
                      (2 if self.macd_crossover[0] > 0 else -2 if self.macd_crossover[0] < 0 else 0) + \
                      (1 if (self.rsi[-1] < self.p.rsi_oversold and self.rsi[0] > self.p.rsi_oversold) else -1 if (
                                  self.rsi[-1] > self.p.rsi_overbought and self.rsi[0] < self.p.rsi_overbought) else 0)

        y_values = ma_short.get(size=self.p.slope_lookback_period)
        if len(y_values) < self.p.slope_lookback_period: return
        slope = np.polyfit(np.arange(len(y_values)), y_values, 1)[0]

        # [NEW] Dynamic Position Sizing Calculation
        atr_sl_mult = self.p.high_vol_atr_sl_multiplier if is_high_vol_local else self.p.low_vol_atr_sl_multiplier
        stop_dist = self.vol_atr[0] * atr_sl_mult
        risk_amount = self.broker.getvalue() * self.p.risk_per_trade
        position_size = risk_amount / stop_dist

        if market_regime == "BULL":
            if total_score >= self.p.buy_score_threshold and adx.adx[
                0] > adx_thresh and slope > self.p.slope_min_threshold:
                self.order = self.buy(size=position_size)
        elif market_regime == "BEAR":
            if total_score <= self.p.sell_score_threshold and adx.adx[
                0] > adx_thresh and slope < -self.p.slope_min_threshold:
                self.order = self.sell(size=position_size)