from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .artifacts import ModelArtifact
from .model_registry import load_model
from .schemas import FeatureConfig, LabelConfig


@dataclass(slots=True)
class LoadedModel:
    artifact: ModelArtifact
    metadata: Dict[str, object]
    metrics: Dict[str, object]
    run_id: str
    feature_config: FeatureConfig
    label_config: LabelConfig


class PredictionService:
    """Loads a trained model artifact and exposes convenience inference helpers."""

    def __init__(self, model_name: str, base_path: Optional[Path] = None, run_id: Optional[str] = None) -> None:
        artifact, metadata, metrics, resolved_run = load_model(model_name, base_path=base_path, run_id=run_id)
        feature_cfg = FeatureConfig.from_dict(metadata["feature_config"])
        label_cfg = LabelConfig.from_dict(metadata["label_config"])
        self.model = LoadedModel(
            artifact=artifact,
            metadata=metadata,
            metrics=metrics,
            run_id=resolved_run,
            feature_config=feature_cfg,
            label_config=label_cfg,
        )

    def predict_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        required_cols = self.model.artifact.feature_columns
        missing = [col for col in required_cols if col not in features.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        ordered = features[required_cols]
        augmented = self._augment_with_clusters(ordered)
        classifier = self.model.artifact.classifier
        proba = classifier.predict_proba(augmented)
        labels = self.model.artifact.class_labels()
        return pd.DataFrame(proba, index=ordered.index, columns=labels)

    def predict_classes(self, features: pd.DataFrame) -> pd.Series:
        proba = self.predict_probabilities(features)
        top_class = proba.idxmax(axis=1)
        return top_class

    def _augment_with_clusters(self, features: pd.DataFrame) -> pd.DataFrame:
        clusterer = self.model.artifact.clusterer
        if clusterer is None:
            return features[self.model.artifact.augmented_feature_columns]

        distances = clusterer.transform(features)
        labels = clusterer.predict(features)
        dist_cols = [f"cluster_dist_{i}" for i in range(distances.shape[1])]
        dist_df = pd.DataFrame(distances, index=features.index, columns=dist_cols)
        label_df = pd.DataFrame({"cluster_label": labels}, index=features.index)
        augmented = pd.concat([features, dist_df, label_df], axis=1)
        return augmented[self.model.artifact.augmented_feature_columns]
