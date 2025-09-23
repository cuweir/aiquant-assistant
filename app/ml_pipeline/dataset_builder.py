from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .feature_engineering import FeatureEngineer
from .labeling import LabelBuilder
from .schemas import (
    DatasetBundle,
    FeatureConfig,
    FeatureMatrix,
    LabelConfig,
    LabelSeries,
    WalkForwardSplit,
)


class DatasetBuilder:
    def __init__(self, feature_config: FeatureConfig, label_config: LabelConfig):
        self.feature_engineer = FeatureEngineer(feature_config)
        self.label_builder = LabelBuilder(label_config)
        self.feature_config = feature_config
        self.label_config = label_config

    def build(self, symbol: str, timeframe: str, price_df: pd.DataFrame) -> DatasetBundle:
        features = self.feature_engineer.build(price_df)
        labels = self.label_builder.build(price_df)
        aligned_features, aligned_labels = self._align(features, labels)
        meta = {
            "feature_fingerprint": features.metadata.get("fingerprint"),
            "label_fingerprint": labels.metadata.get("fingerprint"),
            "rows": len(aligned_features.dataframe),
            "feature_config": self.feature_config.to_dict(),
            "label_config": self.label_config.to_dict(),
        }
        return DatasetBundle(symbol=symbol, timeframe=timeframe, features=aligned_features, labels=aligned_labels, meta=meta)

    def _align(self, features: FeatureMatrix, labels: LabelSeries) -> tuple[FeatureMatrix, LabelSeries]:
        def _normalize(idx: pd.Index) -> pd.Index:
            if isinstance(idx, pd.DatetimeIndex):
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                return idx
            return idx

        feature_idx = _normalize(features.dataframe.index)
        label_idx = _normalize(labels.series.dropna().index)
        common_values = feature_idx.intersection(label_idx)

        feature_mask = feature_idx.isin(common_values)
        label_mask = _normalize(labels.series.index).isin(common_values)

        feature_df = features.dataframe.loc[feature_mask].copy()
        label_series = labels.series.loc[label_mask].copy()

        if isinstance(feature_df.index, pd.DatetimeIndex) and feature_df.index.tz is None:
            feature_df.index = feature_df.index.tz_localize("UTC")
        if isinstance(label_series.index, pd.DatetimeIndex) and label_series.index.tz is None:
            label_series.index = label_series.index.tz_localize("UTC")

        aligned_features = FeatureMatrix(feature_df, metadata=features.metadata | {"aligned_rows": len(feature_df)})
        aligned_labels = LabelSeries(label_series, metadata=labels.metadata | {"aligned_rows": len(label_series)})
        return aligned_features, aligned_labels


def save_bundle(bundle: DatasetBundle, base_path: Path) -> Path:
    dataset_dir = base_path / bundle.symbol.replace("/", "-") / bundle.timeframe
    dataset_dir.mkdir(parents=True, exist_ok=True)
    feature_path = dataset_dir / "features.parquet"
    label_path = dataset_dir / "labels.parquet"
    meta_path = dataset_dir / "meta.json"
    bundle.features.dataframe.to_parquet(feature_path)
    bundle.labels.series.to_frame("label").to_parquet(label_path)
    metadata = {
        "symbol": bundle.symbol,
        "timeframe": bundle.timeframe,
        "bundle_fingerprint": bundle.fingerprint(),
        "feature_metadata": bundle.features.metadata,
        "label_metadata": bundle.labels.metadata,
        "meta": bundle.meta,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    return dataset_dir


def load_bundle(path: Path) -> DatasetBundle:
    meta_path = path / "meta.json"
    metadata = json.loads(meta_path.read_text())
    feature_df = pd.read_parquet(path / "features.parquet")
    label_series = pd.read_parquet(path / "labels.parquet")["label"]
    features = FeatureMatrix(feature_df, metadata=metadata["feature_metadata"])
    labels = LabelSeries(label_series, metadata=metadata["label_metadata"])
    return DatasetBundle(
        symbol=metadata["symbol"],
        timeframe=metadata["timeframe"],
        features=features,
        labels=labels,
        meta=metadata.get("meta", {}),
    )


def walk_forward_split(bundle: DatasetBundle, train_size: int, val_size: int, step_size: Optional[int] = None) -> List[WalkForwardSplit]:
    index = bundle.features.dataframe.index
    total = len(index)
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive")
    step = step_size or val_size
    splits: List[WalkForwardSplit] = []
    cursor = 0
    while cursor + train_size + val_size <= total:
        train_start = index[cursor]
        train_end = index[cursor + train_size - 1]
        val_start = index[cursor + train_size]
        val_end = index[cursor + train_size + val_size - 1]
        splits.append(WalkForwardSplit(train_start, train_end, val_start, val_end))
        cursor += step
    return splits
