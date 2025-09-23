from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .schemas import LabelConfig, LabelMethod, LabelSeries


class LabelBuilder:
    def __init__(self, config: LabelConfig):
        self.config = config

    def build(self, price_df: pd.DataFrame) -> LabelSeries:
        if price_df.empty:
            raise ValueError("Price DataFrame is empty")
        if self.config.method == LabelMethod.FIXED_HORIZON:
            series = self._build_fixed_horizon(price_df)
        elif self.config.method == LabelMethod.TRIPLE_BARRIER:
            series = self._build_triple_barrier(price_df)
        elif self.config.method == LabelMethod.MULTI_STATE:
            series = self._build_multi_state(price_df)
        else:
            raise ValueError(f"Unsupported label method {self.config.method}")
        metadata: Dict[str, object] = {
            "method": self.config.method.value,
            "horizon": self.config.horizon,
            "fingerprint": self.config.fingerprint(),
        }
        if self.config.method == LabelMethod.MULTI_STATE:
            metadata["states"] = self._multi_state_labels()
            metadata["thresholds"] = self._multi_state_thresholds()
        return LabelSeries(series, metadata=metadata)

    def _build_fixed_horizon(self, price_df: pd.DataFrame) -> pd.Series:
        horizon = self.config.horizon
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
        close = price_df["close"]
        future_return = close.shift(-horizon) / close - 1
        labels = pd.Series(0, index=close.index, dtype=int)
        take_profit = self.config.take_profit or 0.0
        stop_loss = self.config.stop_loss or 0.0
        neutral_zone = self.config.neutral_zone or 0.0

        if self.config.directional:
            labels[future_return > take_profit] = 1
            labels[future_return < -stop_loss] = -1
        else:
            labels[(future_return > take_profit)] = 1
            labels[(future_return < -stop_loss)] = -1
            in_neutral = future_return.abs() <= neutral_zone
            labels[in_neutral] = 0

        labels.iloc[-horizon:] = np.nan
        return labels

    def _build_triple_barrier(self, price_df: pd.DataFrame) -> pd.Series:
        horizon = self.config.horizon
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
        take_profit = self.config.take_profit or 0.02
        stop_loss = self.config.stop_loss or 0.02
        close = price_df["close"].values
        high = price_df.get("high", price_df["close"]).values
        low = price_df.get("low", price_df["close"]).values
        n = len(price_df)
        labels = np.full(n, np.nan)

        for idx in range(n):
            start_price = close[idx]
            if np.isnan(start_price):
                continue
            upper = start_price * (1 + take_profit)
            lower = start_price * (1 - stop_loss)
            outcome = 0
            for step in range(1, horizon + 1):
                future_idx = idx + step
                if future_idx >= n:
                    break
                if high[future_idx] >= upper:
                    outcome = 1
                    break
                if low[future_idx] <= lower:
                    outcome = -1
                    break
            labels[idx] = outcome
        labels[-horizon:] = np.nan
        return pd.Series(labels, index=price_df.index, dtype=float)

    def _build_multi_state(self, price_df: pd.DataFrame) -> pd.Series:
        horizon = self.config.horizon
        if horizon <= 0:
            raise ValueError("Horizon must be positive")

        thresholds = self._multi_state_thresholds()
        states = self._multi_state_labels()

        close = price_df["close"].astype(float)
        future_return = close.shift(-horizon) / close - 1

        base_state = "ranging"
        label_array = pd.Series(base_state, index=close.index, dtype="object")

        strong_tp = thresholds["strong_take_profit"]
        weak_tp = thresholds["take_profit"]
        neutral = thresholds["neutral_zone"]
        weak_sl = thresholds["stop_loss"]
        strong_sl = thresholds["strong_stop_loss"]

        label_array[future_return >= strong_tp] = "strong_bullish"
        grinding_up_mask = (future_return >= weak_tp) & (future_return < strong_tp)
        label_array[grinding_up_mask] = "grinding_up"

        label_array[future_return <= -strong_sl] = "strong_bearish"
        grinding_down_mask = (future_return <= -weak_sl) & (future_return > -strong_sl)
        label_array[grinding_down_mask] = "grinding_down"

        ranging_mask = future_return.abs() <= neutral
        label_array[ranging_mask] = "ranging"

        label_array.iloc[-horizon:] = np.nan

        categorical = pd.Categorical(label_array, categories=states, ordered=True)
        return pd.Series(categorical, index=close.index, dtype="category")

    def _multi_state_thresholds(self) -> Dict[str, float]:
        take_profit = self.config.take_profit or 0.01
        stop_loss = self.config.stop_loss or 0.01
        strong_take_profit = self.config.strong_take_profit or max(take_profit * 3, take_profit)
        strong_stop_loss = self.config.strong_stop_loss or max(stop_loss * 3, stop_loss)
        neutral_zone = self.config.neutral_zone or min(take_profit, stop_loss) / 2

        if strong_take_profit < take_profit:
            raise ValueError("strong_take_profit must be >= take_profit")
        if strong_stop_loss < stop_loss:
            raise ValueError("strong_stop_loss must be >= stop_loss")

        return {
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "neutral_zone": neutral_zone,
            "strong_take_profit": strong_take_profit,
            "strong_stop_loss": strong_stop_loss,
        }

    @staticmethod
    def _multi_state_labels() -> list[str]:
        return [
            "strong_bearish",
            "grinding_down",
            "ranging",
            "grinding_up",
            "strong_bullish",
        ]
