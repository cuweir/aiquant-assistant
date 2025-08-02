import backtrader as bt
import numpy as np

class MtfaStrategyBt(bt.Strategy):
    """
    EVOLUTION 5 (Final): Adding a Slope Filter to curb over-trading.
    The goal is to only enter when the short-term trend has strong upward momentum.
    """
    params = (
        # --- Base Parameters ---
        ('regime_ma_period', 200),
        ('buy_score_threshold', 3),
        ('atr_period', 14),
        ('slope_lookback_period', 5),
        ('slope_min_threshold', 0.0),

        # --- [NEW] Parameters now define ranges for different volatility states ---
        # Low Volatility Parameters
        ('low_vol_ma_short', 25),
        ('low_vol_ma_long', 60),
        ('low_vol_adx_threshold', 28),
        ('low_vol_atr_sl_multiplier', 2.0),

        # High Volatility Parameters
        ('high_vol_ma_short', 15),
        ('high_vol_ma_long', 40),
        ('high_vol_adx_threshold', 22),
        ('high_vol_atr_sl_multiplier', 3.0),

        # Volatility Regime Parameters
        ('vol_atr_period', 14),  # ATR for volatility calculation
        ('vol_atr_ma_period', 100),  # Moving average of ATR
    )

    def __init__(self):
        # --- Data & Aliases ---
        self.d_signal = self.datas[0]
        self.d_regime = self.datas[1]
        self.dataclose = self.d_signal.close

        # --- [FIX] All indicators are now defined once and for all here ---

        # Regime Filter
        self.regime_ma = bt.indicators.SimpleMovingAverage(self.d_regime, period=self.p.regime_ma_period)

        # Volatility Regime Indicators
        self.vol_atr = bt.indicators.AverageTrueRange(self.d_signal, period=self.p.vol_atr_period)
        self.vol_atr_ma = bt.indicators.SimpleMovingAverage(self.vol_atr, period=self.p.vol_atr_ma_period)

        # --- Indicators for the LOW volatility parameter set ---
        self.low_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_short)
        self.low_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.low_vol_ma_long)
        self.low_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=14)

        # --- Indicators for the HIGH volatility parameter set ---
        self.high_vol_ma_short = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_short)
        self.high_vol_ma_long = bt.indicators.SimpleMovingAverage(self.d_signal, period=self.p.high_vol_ma_long)
        self.high_vol_adx = bt.indicators.AverageDirectionalMovementIndex(self.d_signal, period=14)

        # --- Common Indicators (their periods don't change) ---
        self.rsi = bt.indicators.RSI(self.d_signal, period=14)
        self.macd = bt.indicators.MACD(self.d_signal, period_me1=12, period_me2=26, period_signal=9)
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

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

        # === [STEP 1] Determine Volatility Regime ===
        is_high_volatility = self.vol_atr[0] > self.vol_atr_ma[0]

        # === [STEP 2] Select the correct, pre-calculated indicator set ===
        if is_high_volatility:
            ma_short = self.high_vol_ma_short
            ma_long = self.high_vol_ma_long
            adx = self.high_vol_adx
            adx_thresh = self.p.high_vol_adx_threshold
            atr_sl_mult = self.p.high_vol_atr_sl_multiplier
        else:  # Low Volatility
            ma_short = self.low_vol_ma_short
            ma_long = self.low_vol_ma_long
            adx = self.low_vol_adx
            adx_thresh = self.p.low_vol_adx_threshold
            atr_sl_mult = self.p.low_vol_atr_sl_multiplier

        # === [STEP 3] Execute Strategy Logic (using the selected indicators) ===
        is_bull_regime = self.d_regime.close[0] > self.regime_ma[0]

        if not self.position:
            if is_bull_regime:
                buy_score = 0
                if ma_short[0] > ma_long[0]: buy_score += 1
                if self.macd_crossover[0] > 0: buy_score += 2
                if self.rsi[-1] < 40 and self.rsi[0] > 40: buy_score += 1

                if buy_score >= self.p.buy_score_threshold:
                    if adx.adx[0] > adx_thresh:
                        y_values = ma_short.get(size=self.p.slope_lookback_period)
                        x_values = np.arange(len(y_values))
                        if len(y_values) < self.p.slope_lookback_period: return
                        slope = np.polyfit(x_values, y_values, 1)[0]

                        if slope > self.p.slope_min_threshold:
                            risk = self.vol_atr[0] * atr_sl_mult
                            self.stop_loss_price = self.dataclose[0] - risk
                            self.log(
                                f'BUY SIGNAL ({"High" if is_high_volatility else "Low"} Vol), Score={buy_score}, Slope={slope:.2f}')
                            self.order = self.buy()
        else:  # Exiting a position
            if self.position.size > 0:
                if self.dataclose[0] < self.stop_loss_price:
                    self.log('STOP LOSS HIT')
                    self.order = self.close()
                elif ma_short[0] < ma_long[0]:
                    self.log('TREND WEAKENED (Death Cross)')
                    self.order = self.close()