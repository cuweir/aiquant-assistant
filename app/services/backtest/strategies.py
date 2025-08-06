# services/backtest/strategies.py

import backtrader as bt
import numpy as np

class BuyAndHold(bt.Strategy):
    """
    A simple benchmark strategy that buys on the first available bar and holds.
    """
    def __init__(self):
        # A flag to ensure we only buy once.
        self.bought = False

    def next(self):
        # The 'next' method is called for each bar of data.
        # We check if we have already bought. If not, we execute the buy order.
        if not self.bought:
            # Calculate the size based on the current bar's close price
            # and almost all available cash (e.g., 95% to account for commission)
            size = (self.broker.get_cash() * 0.95) / self.data.close[0]
            self.buy(size=size)
            # Set the flag to true so we don't buy again.
            self.bought = True

class MtfaStrategyBt(bt.Strategy):
    """
    FINAL PRODUCTION VERSION: Long-Only, Volatility Adaptive Specialist (V7 - Robust).
    This is the most reliable and validated version of our strategy. It focuses on its
    strength: capturing gains in bull/choppy markets, while inherently controlling
    risk in bear markets.
    """
    params = (
        # These are the champion parameters found from our successful 1h optimization.
        # They will be passed in from the runner.
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
        # ... (The __init__ method from the stable V7 version) ...
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
        self.stop_loss_price = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

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
                if self.dataclose[0] < self.stop_loss_price:
                    self.order = self.close()
                elif ma_short[0] < ma_long[0]:
                    self.order = self.close()