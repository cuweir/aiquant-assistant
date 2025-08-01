import backtrader as bt
import numpy as np

class MtfaStrategyBt(bt.Strategy):
    """
    EVOLUTION 5 (Final): Adding a Slope Filter to curb over-trading.
    The goal is to only enter when the short-term trend has strong upward momentum.
    """
    params = (
        # --- Regime Filter Parameters ---
        ('regime_ma_period', 200),
        # --- Confluence Scoring Parameters ---
        ('score_ma_short_period', 20),
        ('score_ma_long_period', 50),
        ('score_macd_fast', 12),
        ('score_macd_slow', 26),
        ('score_macd_signal', 9),
        ('score_rsi_period', 14),
        ('score_rsi_oversold', 40),
        # --- Entry Threshold ---
        ('buy_score_threshold', 3),
        # --- Risk Management Parameters ---
        ('atr_period', 14),
        ('atr_sl_multiplier', 2.5),
        # --- Slope Filter Parameters ---
        ('slope_lookback_period', 5),
        ('slope_min_threshold', 0.0),
        # --- [NEW] Add all other parameters from the optimizer grid here ---
        # Even if they are not used in this specific simple strategy version,
        # defining them prevents "unexpected keyword argument" errors.
        ('adx_period', 14),
        ('adx_threshold', 25),
        ('risk_reward_ratio_tp', 1.5)  # From older strategy versions
    )

    def __init__(self):
        # ... (Data Aliases and Indicator definitions remain mostly the same) ...
        self.d_signal = self.datas[0]
        self.d_regime = self.datas[1]
        self.dataclose = self.d_signal.close
        self.regime_ma = bt.indicators.SimpleMovingAverage(self.d_regime.close, period=self.p.regime_ma_period)
        self.score_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal.close, period=self.p.score_ma_short_period)
        self.score_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal.close, period=self.p.score_ma_long_period)
        self.rsi = bt.indicators.RSI(self.d_signal.close, period=self.p.score_rsi_period)
        self.macd = bt.indicators.MACD(self.d_signal.close, period_me1=self.p.score_macd_fast, period_me2=self.p.score_macd_slow, period_signal=self.p.score_macd_signal)
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.atr_period)

        # --- State Variables ---
        self.order = None
        self.stop_loss_price = None

    # log, notify_order, notify_trade methods remain unchanged
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
                buy_score = 0
                if self.score_ma_short > self.score_ma_long: buy_score += 1
                if self.macd_crossover > 0 and self.macd_crossover[-1] != self.macd_crossover[0]: buy_score += 2
                if self.rsi[-1] < self.p.score_rsi_oversold and self.rsi[0] > self.p.score_rsi_oversold: buy_score += 1

                if buy_score >= self.p.buy_score_threshold:
                    # [NEW] FINAL CONFIRMATION: Check the slope of the short-term MA
                    # Get the last N values of the short MA
                    y_values = self.score_ma_short.get(size=self.p.slope_lookback_period)
                    # Use a simple linear regression to find the slope
                    # A positive slope means the MA is trending upwards
                    x_values = np.arange(len(y_values))
                    slope = np.polyfit(x_values, y_values, 1)[0]

                    if slope > self.p.slope_min_threshold:
                        risk_per_share = self.atr[0] * self.p.atr_sl_multiplier
                        self.stop_loss_price = self.dataclose[0] - risk_per_share

                        self.log(
                            f'BUY SIGNAL (Score={buy_score}, Slope={slope:.2f}), Price={self.dataclose[0]:.2f}, SL={self.stop_loss_price:.2f}')
                        self.order = self.buy()
        else:  # Exiting a position
            if self.position.size > 0:
                if self.dataclose[0] < self.stop_loss_price:
                    self.log(f'STOP LOSS HIT, Price={self.dataclose[0]:.2f}')
                    self.order = self.close()
                elif self.score_ma_short < self.score_ma_long:
                    self.log(f'TREND WEAKENED (Death Cross), TAKE PROFIT, Price={self.dataclose[0]:.2f}')
                    self.order = self.close()