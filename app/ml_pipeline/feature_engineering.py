from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .schemas import (
    FeatureConfig,
    FeatureMatrix,
    FillMethod,
    ScalingMethod,
)
from .sources.base import build_source


class FeatureEngineer:
    """Builds aligned feature matrices from price data and registered sources."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def build(self, price_df: pd.DataFrame) -> FeatureMatrix:
        if price_df.empty:
            raise ValueError("Price DataFrame is empty")
        base_df = price_df.sort_index()
        if isinstance(base_df.index, pd.DatetimeIndex) and base_df.index.tz is not None:
            base_df = base_df.copy()
            base_df.index = base_df.index.tz_convert("UTC").tz_localize(None)
        feature_frames: List[pd.DataFrame] = []
        for spec in self.config.sources:
            source = build_source(spec)
            feature_df = source.compute(base_df)
            feature_frames.append(feature_df)

        combined = pd.concat(feature_frames, axis=1)
        combined = combined.loc[base_df.index]

        drop_columns = [col for col in combined.columns if combined[col].isna().all()]
        if drop_columns:
            combined = combined.drop(columns=drop_columns)
            if combined.empty or combined.shape[1] == 0:
                raise ValueError(
                    "All generated feature columns are empty. External data sources may not overlap with price data."
                )

        if self.config.lags:
            combined = self._append_lags(combined, self.config.lags)

        filled = self._fill_missing(combined)
        if filled.empty:
            raise ValueError(
                "Feature matrix is empty after missing value handling. "
                "Check external feature coverage (e.g. funding/open interest/sentiment files)."
            )
        combined = filled
        scaling_stats: Dict[str, Tuple[float, float]] | None = None
        if self.config.scaling != ScalingMethod.NONE:
            combined, scaling_stats = self._scale(combined)

        if isinstance(combined.index, pd.DatetimeIndex) and combined.index.tz is None:
            combined = combined.tz_localize("UTC")

        metadata = {
            "columns": list(combined.columns),
            "scaling": self.config.scaling.value,
            "fill_method": self.config.fill_method.value,
            "lags": self.config.lags,
            "fingerprint": self.config.fingerprint(),
        }
        if scaling_stats:
            metadata["scaling_stats"] = {k: (float(v[0]), float(v[1])) for k, v in scaling_stats.items()}
        return FeatureMatrix(combined, metadata=metadata)

    def _append_lags(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        lag_frames = [df]
        for lag in sorted(set(lags)):
            lagged = df.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
            lag_frames.append(lagged)
        return pd.concat(lag_frames, axis=1)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.fill_method == FillMethod.FILL_FORWARD:
            filled = df.ffill()
        elif self.config.fill_method == FillMethod.FILL_BACKWARD:
            filled = df.bfill()
        elif self.config.fill_method == FillMethod.DROP:
            filled = df.dropna()
        else:
            raise ValueError(f"Unsupported fill method: {self.config.fill_method}")
        return filled.dropna()

    def _scale(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        stats: Dict[str, Tuple[float, float]] = {}
        scaled = df.copy()
        for col in scaled.columns:
            series = scaled[col]
            if self.config.scaling == ScalingMethod.ZSCORE:
                mean = series.mean()
                std = series.std(ddof=0)
                std = std if std != 0 else 1.0
                scaled[col] = (series - mean) / std
                stats[col] = (mean, std)
            elif self.config.scaling == ScalingMethod.MINMAX:
                min_val = series.min()
                max_val = series.max()
                span = max_val - min_val
                span = span if span != 0 else 1.0
                scaled[col] = (series - min_val) / span
                stats[col] = (min_val, max_val)
            else:
                raise ValueError(f"Unsupported scaling: {self.config.scaling}")
        return scaled, stats
