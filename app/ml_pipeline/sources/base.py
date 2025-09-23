from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import pandas as pd

from ..schemas import FeatureSourceSpec


class FeatureSource(ABC):
    """Base interface for all feature sources."""

    def __init__(self, spec: FeatureSourceSpec):
        self.spec = spec

    @abstractmethod
    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame indexed by timestamp containing the engineered features."""
        raise NotImplementedError


FeatureSourceRegistry: Dict[str, Type[FeatureSource]] = {}


def register_source(name: str):
    def decorator(cls: Type[FeatureSource]) -> Type[FeatureSource]:
        FeatureSourceRegistry[name] = cls
        return cls

    return decorator


def build_source(spec: FeatureSourceSpec) -> FeatureSource:
    cls = FeatureSourceRegistry.get(spec.name)
    if cls is None:
        raise KeyError(f"Feature source '{spec.name}' is not registered")
    return cls(spec)
