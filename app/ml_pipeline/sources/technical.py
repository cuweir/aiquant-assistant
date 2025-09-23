from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pandas_ta as ta

from ..schemas import FeatureSourceSpec
from .base import FeatureSource, register_source


@register_source("technical_indicators")
class TechnicalIndicatorSource(FeatureSource):
    """Generates a collection of technical indicators using pandas_ta."""

    def __init__(self, spec: FeatureSourceSpec):
        super().__init__(spec)
        self._indicator_params = self._resolve_params(spec.params)

    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        df = price_df.copy()
        outputs: Dict[str, pd.Series] = {}

        if "rsi" in self._indicator_params:
            length = self._indicator_params["rsi"].get("length", 14)
            outputs[f"rsi_{length}"] = df.ta.rsi(length=length)

        if "macd" in self._indicator_params:
            params = self._indicator_params["macd"]
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            signal = params.get("signal", 9)
            macd_df = df.ta.macd(fast=fast, slow=slow, signal=signal)
            if macd_df is not None and not macd_df.empty:
                outputs[f"macd_{fast}_{slow}"] = macd_df.iloc[:, 0]
                outputs[f"macds_{fast}_{slow}_{signal}"] = macd_df.iloc[:, 1]
                outputs[f"macdh_{fast}_{slow}_{signal}"] = macd_df.iloc[:, 2]

        if "adx" in self._indicator_params:
            length = self._indicator_params["adx"].get("length", 14)
            adx_df = df.ta.adx(length=length)
            if adx_df is not None and not adx_df.empty:
                outputs[f"adx_{length}"] = adx_df.iloc[:, 0]

        if "atr" in self._indicator_params:
            length = self._indicator_params["atr"].get("length", 14)
            outputs[f"atr_{length}"] = df.ta.atr(length=length)

        if "bollinger_bandwidth" in self._indicator_params:
            params = self._indicator_params["bollinger_bandwidth"]
            length = params.get("length", 20)
            std = params.get("std", 2)
            bb = df.ta.bbands(length=length, std=std)
            if bb is not None and not bb.empty:
                upper = bb.iloc[:, 0]
                lower = bb.iloc[:, 2]
                mid = bb.iloc[:, 1]
                bandwidth = (upper - lower) / mid
                outputs[f"bb_width_{length}_{std}"] = bandwidth

        feature_df = pd.DataFrame(outputs)
        feature_df = feature_df.loc[df.index]
        return feature_df

    @staticmethod
    def _resolve_params(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not params:
            return {
                "rsi": {"length": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "adx": {"length": 14},
                "atr": {"length": 14},
                "bollinger_bandwidth": {"length": 20, "std": 2},
            }
        indicators = params.get("indicators")
        if isinstance(indicators, list):
            resolved: Dict[str, Dict[str, Any]] = {}
            for entry in indicators:
                if isinstance(entry, dict) and "name" in entry:
                    name = entry["name"]
                    resolved[name] = {k: v for k, v in entry.items() if k != "name"}
            return resolved or TechnicalIndicatorSource._resolve_params({})
        return params
