import backtrader as bt
from ...core import config

# We import the settings object to get all strategy parameters
settings = config.settings


class MtfaStrategyBt(bt.Strategy):
    """
    Backtrader adaptation of our Multi-Indicator, Multi-Timeframe Analysis Strategy.
    """
    params = (
        # --- Trend Filter Parameters ---
        ('trend_filter_period_short', settings.TREND_FILTER_PERIOD_SHORT),
        ('trend_filter_period_long', settings.TREND_FILTER_PERIOD_LONG),

        # --- Indicator Parameters (for Signal Timeframe) ---
        ('rsi_period', settings.RSI_PERIOD),
        ('rsi_overbought', settings.RSI_OVERBOUGHT),
        ('rsi_oversold', settings.RSI_OVERSOLD),
        ('macd_fast_period', settings.MACD_FAST_PERIOD),
        ('macd_slow_period', settings.MACD_SLOW_PERIOD),
        ('macd_signal_period', settings.MACD_SIGNAL_PERIOD),
        ('ma_short_period', settings.MA_SHORT_PERIOD),
        ('ma_long_period', settings.MA_LONG_PERIOD),
        ('bbands_period', settings.BBANDS_PERIOD),
        ('bbands_std_dev', settings.BBANDS_STD_DEV),

        # --- Scoring Weights ---
        ('weight_ma_state', settings.WEIGHT_MA_STATE),
        ('weight_ma_event', settings.WEIGHT_MA_EVENT),
        ('weight_macd_state', settings.WEIGHT_MACD_STATE),
        ('weight_macd_event', settings.WEIGHT_MACD_EVENT),
        ('weight_rsi_extreme', settings.WEIGHT_RSI_EXTREME),
        ('weight_bbands_breakout', settings.WEIGHT_BBANDS_BREAKOUT),

        # --- Thresholds ---
        ('buy_score_threshold', settings.BUY_SCORE_THRESHOLD),
        ('sell_score_threshold', settings.SELL_SCORE_THRESHOLD),

        # --- Exit Strategy Parameters ---
        ('atr_period', settings.ATR_PERIOD),
        ('atr_sl_multiplier', settings.ATR_STOP_LOSS_MULTIPLIER),
        ('risk_reward_ratio_tp', settings.RISK_REWARD_RATIO_TP2),  # Using final TP for this example
    )

    def __init__(self):
        # --- Data Aliases for Clarity ---
        # Backtrader passes data feeds in the order they are added.
        # We assume: 0=Signal (15m), 1=ShortTrend (1h), 2=LongTrend (4h)
        self.d_signal = self.datas[0]
        self.d_trend_short = self.datas[1]
        self.d_trend_long = self.datas[2]

        # --- Trend Indicators on Trend Timeframes ---
        self.trend_ema_short = bt.ind.EMA(self.d_trend_short.close, period=self.p.trend_filter_period_short)
        self.trend_ema_long = bt.ind.EMA(self.d_trend_long.close, period=self.p.trend_filter_period_long)

        # --- Indicators on the Signal Timeframe ---
        self.rsi = bt.ind.RSI(self.d_signal.close, period=self.p.rsi_period)

        # [FIX] Corrected the keyword arguments for the MACD indicator.
        # backtrader uses 'period_me1', 'period_me2', and 'period_signal'.
        self.macd = bt.ind.MACD(self.d_signal.close,
                                period_me1=self.p.macd_fast_period,
                                period_me2=self.p.macd_slow_period,
                                period_signal=self.p.macd_signal_period)

        self.ma_short = bt.ind.EMA(self.d_signal.close, period=self.p.ma_short_period)
        self.ma_long = bt.ind.EMA(self.d_signal.close, period=self.p.ma_long_period)
        self.bbands = bt.ind.BollingerBands(self.d_signal.close,
                                            period=self.p.bbands_period,
                                            devfactor=self.p.bbands_std_dev)

        # --- Exit Indicator ---
        self.atr = bt.ind.ATR(self.d_signal, period=self.p.atr_period)

        # --- Crossover Indicators for Events ---
        self.macd_cross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        self.ma_cross = bt.ind.CrossOver(self.ma_short, self.ma_long)

        self.order = None  # To keep track of pending orders

    def log(self, txt, dt=None):
        """ Logging function for this strategy """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.Status[order.status]}')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # We can only trade if we are not in the market
        if self.position:
            return

        # === 1. Determine Market Regime (from longer timeframes) ===
        is_uptrend_short = self.d_trend_short.close[0] > self.trend_ema_short[0]
        is_uptrend_long = self.d_trend_long.close[0] > self.trend_ema_long[0]

        market_regime = "NEUTRAL"
        if is_uptrend_short and is_uptrend_long:
            market_regime = "STRONG_BULL"
        elif not is_uptrend_short and not is_uptrend_long:
            market_regime = "STRONG_BEAR"
        elif is_uptrend_long and not is_uptrend_short:
            market_regime = "BULLISH_PULLBACK"
        # (Other regimes like BEARISH_RALLY can be added)

        # === 2. Calculate Score (from signal timeframe) ===
        score = 0

        # MA State
        if self.ma_short[0] > self.ma_long[0]:
            score += self.p.weight_ma_state
        else:
            score -= self.p.weight_ma_state

        # MACD State
        if self.macd.macd[0] > self.macd.signal[0]:
            score += self.p.weight_macd_state
        else:
            score -= self.p.weight_macd_state

        # MA Crossover Event
        if self.ma_cross[0] > 0:
            score += self.p.weight_ma_event
        elif self.ma_cross[0] < 0:
            score -= self.p.weight_ma_event

        # MACD Crossover Event
        if self.macd_cross[0] > 0:
            score += self.p.weight_macd_event
        elif self.macd_cross[0] < 0:
            score -= self.p.weight_macd_event

        # RSI Extreme
        if self.rsi[0] < self.p.rsi_oversold:
            score += self.p.weight_rsi_extreme
        elif self.rsi[0] > self.p.rsi_overbought:
            score -= self.p.weight_rsi_extreme

        # Bollinger Bands Breakout
        if self.d_signal.close[0] > self.bbands.top[0]:
            score += self.p.weight_bbands_breakout
        elif self.d_signal.close[0] < self.bbands.bot[0]:
            score -= self.p.weight_bbands_breakout

        # === 3. Apply Regime Adjustments ===
        # (This is a simplified version of the logic)
        if market_regime == "STRONG_BULL" and score > 0:
            score *= settings.REGIME_STRONG_TREND_BONUS
        elif market_regime == "BULLISH_PULLBACK" and self.rsi[0] < 40:  # Look for buy-the-dip
            score += settings.REGIME_BULLISH_PULLBACK_RSI_BONUS
        # (Add logic for BEAR regimes)

        # === 4. Entry Logic ===
        if score >= self.p.buy_score_threshold:
            atr_value = self.atr[0]
            stop_loss_price = self.d_signal.close[0] - self.p.atr_sl_multiplier * atr_value
            take_profit_price = self.d_signal.close[0] + (
                        self.p.atr_sl_multiplier * atr_value * self.p.risk_reward_ratio_tp)

            # Use bracket order for simultaneous SL and TP
            # Note: backtrader's bracket orders are basic. For more complex scenarios,
            # one might manage SL/TP orders manually in notify_trade.
            self.buy_bracket(
                slprice=stop_loss_price,
                price=self.d_signal.close[0],  # The entry price
                stopexec=bt.Order.Stop,
                limitprice=take_profit_price,
                limitexec=bt.Order.Limit
            )
            self.log(
                f'BUY BRACKET CREATED: Price={self.d_signal.close[0]:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}')

        elif score <= self.p.sell_score_threshold:
            atr_value = self.atr[0]
            stop_loss_price = self.d_signal.close[0] + self.p.atr_sl_multiplier * atr_value
            take_profit_price = self.d_signal.close[0] - (
                        self.p.atr_sl_multiplier * atr_value * self.p.risk_reward_ratio_tp)

            self.sell_bracket(
                slprice=stop_loss_price,
                price=self.d_signal.close[0],
                stopexec=bt.Order.Stop,
                limitprice=take_profit_price,
                limitexec=bt.Order.Limit
            )
            self.log(
                f'SELL BRACKET CREATED: Price={self.d_signal.close[0]:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}')