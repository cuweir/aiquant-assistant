import backtrader as bt


class MtfaStrategyBt(bt.Strategy):
    """
    EVOLUTION 3: Removing fixed Take Profit, re-introducing signal reversal as the exit.
    The goal is to let winning trades run longer to improve the Profit Factor.
    """
    params = (
        ('ma_short_period', 20),
        ('ma_long_period', 50),
        ('adx_period', 14),
        ('adx_threshold', 25),
        ('atr_period', 14),
        ('atr_sl_multiplier', 2.0), # We keep the initial Stop Loss
    )

    def __init__(self):
        self.d_signal = self.datas[0]
        self.dataclose = self.d_signal.close
        self.order = None
        self.stop_loss_price = None # To keep track of the SL price

        self.ma_short = bt.indicators.SimpleMovingAverage(self.d_signal.close, period=self.p.ma_short_period)
        self.ma_long = bt.indicators.SimpleMovingAverage(self.d_signal.close, period=self.p.ma_long_period)
        self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=self.p.adx_period)
        self.atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.atr_period)

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

        if not self.position:  # Not in the market
            # Entry signal
            is_strong_trend = self.adx.adx[0] > self.p.adx_threshold
            is_golden_cross = self.crossover > 0

            if is_strong_trend and is_golden_cross:
                # [NEW] Calculate initial stop loss price
                risk_per_share = self.atr[0] * self.p.atr_sl_multiplier
                self.stop_loss_price = self.dataclose[0] - risk_per_share

                self.log(f'BUY CREATE, Price={self.dataclose[0]:.2f}, Initial SL={self.stop_loss_price:.2f}')
                self.order = self.buy()

        else:  # Already in the market
            # Exit Logic
            is_death_cross = self.crossover < 0

            # 1. Check for initial stop loss
            if self.dataclose[0] < self.stop_loss_price:
                self.log(f'STOP LOSS HIT, Price={self.dataclose[0]:.2f}')
                self.order = self.close()
            # 2. [NEW] Check for trend reversal (death cross) as the profit-taking signal
            elif is_death_cross:
                self.log(f'TREND REVERSAL (DEATH CROSS), TAKE PROFIT, Price={self.dataclose[0]:.2f}')
                self.order = self.close()
