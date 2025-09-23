from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from hashlib import sha1
from typing import Any, Dict, List, Optional

import pandas as pd


class FeatureSourceKind(Enum):
    TECHNICAL = auto()
    MARKET_MICRO = auto()
    ONCHAIN = auto()
    SENTIMENT = auto()
    CUSTOM = auto()


class ScalingMethod(Enum):
    NONE = "none"
    ZSCORE = "zscore"
    MINMAX = "minmax"


class FillMethod(Enum):
    FILL_FORWARD = "ffill"
    FILL_BACKWARD = "bfill"
    DROP = "drop"


class LabelMethod(Enum):
    FIXED_HORIZON = "fixed_horizon"
    TRIPLE_BARRIER = "triple_barrier"
    MULTI_STATE = "multi_state"


@dataclass(slots=True)
class FeatureSourceSpec:
    name: str
    kind: FeatureSourceKind
    timeframe: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = f"{self.name}|{self.kind.value}|{self.timeframe}|{sorted(self.params.items())}"
        return sha1(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.name,
            "timeframe": self.timeframe,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSourceSpec":
        return cls(
            name=data["name"],
            kind=FeatureSourceKind[data["kind"]],
            timeframe=data.get("timeframe"),
            params=data.get("params", {}),
        )


@dataclass(slots=True)
class FeatureConfig:
    sources: List[FeatureSourceSpec]
    lags: List[int] = field(default_factory=list)
    scaling: ScalingMethod = ScalingMethod.NONE
    fill_method: FillMethod = FillMethod.FILL_FORWARD
    resample_rule: Optional[str] = None

    def fingerprint(self) -> str:
        src_fp = ",".join(spec.fingerprint() for spec in self.sources)
        payload = f"{src_fp}|{sorted(self.lags)}|{self.scaling.value}|{self.fill_method.value}|{self.resample_rule}"
        return sha1(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": [spec.to_dict() for spec in self.sources],
            "lags": list(self.lags),
            "scaling": self.scaling.name,
            "fill_method": self.fill_method.name,
            "resample_rule": self.resample_rule,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureConfig":
        return cls(
            sources=[FeatureSourceSpec.from_dict(item) for item in data.get("sources", [])],
            lags=data.get("lags", []),
            scaling=ScalingMethod[data.get("scaling", ScalingMethod.NONE.name)],
            fill_method=FillMethod[data.get("fill_method", FillMethod.FILL_FORWARD.name)],
            resample_rule=data.get("resample_rule"),
        )


@dataclass(slots=True)
class LabelConfig:
    method: LabelMethod
    horizon: int
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    neutral_zone: Optional[float] = None
    directional: bool = True
    strong_take_profit: Optional[float] = None
    strong_stop_loss: Optional[float] = None

    def fingerprint(self) -> str:
        payload = (
            f"{self.method.value}|{self.horizon}|{self.take_profit}|{self.stop_loss}|"
            f"{self.neutral_zone}|{self.directional}|{self.strong_take_profit}|{self.strong_stop_loss}"
        )
        return sha1(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.name,
            "horizon": self.horizon,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "neutral_zone": self.neutral_zone,
            "directional": self.directional,
            "strong_take_profit": self.strong_take_profit,
            "strong_stop_loss": self.strong_stop_loss,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelConfig":
        return cls(
            method=LabelMethod[data["method"]],
            horizon=data["horizon"],
            take_profit=data.get("take_profit"),
            stop_loss=data.get("stop_loss"),
            neutral_zone=data.get("neutral_zone"),
            directional=data.get("directional", True),
            strong_take_profit=data.get("strong_take_profit"),
            strong_stop_loss=data.get("strong_stop_loss"),
        )


@dataclass(slots=True)
class FeatureMatrix:
    dataframe: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "FeatureMatrix":
        return FeatureMatrix(self.dataframe.copy(), metadata=self.metadata.copy())


@dataclass(slots=True)
class LabelSeries:
    series: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "LabelSeries":
        return LabelSeries(self.series.copy(), metadata=self.metadata.copy())


@dataclass(slots=True)
class DatasetBundle:
    symbol: str
    timeframe: str
    features: FeatureMatrix
    labels: LabelSeries
    meta: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = (
            f"{self.symbol}|{self.timeframe}|{self.features.metadata.get('fingerprint')}|"
            f"{self.labels.metadata.get('fingerprint')}"
        )
        return sha1(payload.encode()).hexdigest()


@dataclass(slots=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    def as_slice(self) -> Dict[str, pd.Timestamp]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
        }
