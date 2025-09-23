from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from .artifacts import ModelArtifact
from .schemas import DatasetBundle, FeatureConfig, LabelConfig


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters controlling the model training pipeline."""

    random_state: int = 42
    n_splits: int = 5
    min_split_size: int = 400
    cluster_count: int = 6
    learning_rate: float = 0.05
    max_iter: int = 500
    max_depth: Optional[int] = None
    max_leaf_nodes: Optional[int] = 31
    early_stopping: bool = True
    validation_fraction: float = 0.1


@dataclass(slots=True)
class TrainingResult:
    artifact: ModelArtifact
    metrics: Dict[str, float]
    cv_metrics: List[Dict[str, float]]
    feature_importances: Dict[str, float]
    class_distribution: Dict[str, float]


class ModelTrainer:
    """Coordinates dataset preparation, cross-validation, and model fitting."""

    def __init__(
        self,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        self.feature_config = feature_config
        self.label_config = label_config
        self.training_config = training_config or TrainingConfig()

    def train(self, bundle: DatasetBundle) -> TrainingResult:
        feature_df, label_series = self._prepare_dataset(bundle)
        if feature_df.empty:
            raise ValueError("Prepared feature DataFrame is empty after cleaning")

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(label_series)
        n_classes = len(label_encoder.classes_)

        cv_metrics = self._run_cross_validation(feature_df, label_series, label_encoder, n_classes)
        aggregated_metrics = self._aggregate_metrics(cv_metrics)

        clusterer = self._fit_clusterer(feature_df)
        augmented_features = self._augment_with_clusters(feature_df, clusterer)

        classifier = self._fit_classifier(augmented_features, encoded_labels)

        feature_importances = self._extract_feature_importances(classifier, augmented_features.columns)
        training_metrics = self._evaluate_predictions(
            label_series,
            label_encoder.inverse_transform(classifier.predict(augmented_features)),
            classifier.predict_proba(augmented_features),
            label_encoder,
            n_classes,
        )

        final_metrics = aggregated_metrics | {f"train_{k}": v for k, v in training_metrics.items()}

        class_dist = (label_series.value_counts(normalize=True).sort_index().to_dict())
        class_dist = {str(k): float(v) for k, v in class_dist.items()}

        artifact = ModelArtifact(
            classifier=classifier,
            clusterer=clusterer,
            label_encoder=label_encoder,
            feature_columns=list(feature_df.columns),
            augmented_feature_columns=list(augmented_features.columns),
            metadata={
                "feature_config": self.feature_config.to_dict(),
                "label_config": self.label_config.to_dict(),
                "training_config": asdict(self.training_config),
                "class_labels": label_encoder.classes_.tolist(),
            },
        )

        return TrainingResult(
            artifact=artifact,
            metrics=final_metrics,
            cv_metrics=cv_metrics,
            feature_importances=feature_importances,
            class_distribution=class_dist,
        )

    def _prepare_dataset(self, bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.Series]:
        feature_df = bundle.features.dataframe.copy()
        labels = bundle.labels.series.copy()

        valid_mask = labels.notna()
        feature_df = feature_df.loc[valid_mask]
        labels = labels.loc[valid_mask]

        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.dropna(axis=0, how="any", inplace=True)
        labels = labels.loc[feature_df.index]

        if hasattr(labels, "astype") and str(labels.dtype) == "category":
            labels = labels.astype(str)
        else:
            labels = labels.astype(str)

        return feature_df, labels

    def _run_cross_validation(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        label_encoder: LabelEncoder,
        n_classes: int,
    ) -> List[Dict[str, float]]:
        n_samples = len(features)
        min_split = max(self.training_config.min_split_size, n_classes * 5)
        potential_splits = min(self.training_config.n_splits, max(0, n_samples // min_split))
        if potential_splits < 2:
            return []

        tscv = TimeSeriesSplit(n_splits=potential_splits)
        folds: List[Dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(features), start=1):
            X_train = features.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_test = labels.iloc[test_idx]

            clusterer = self._fit_clusterer(X_train)
            X_train_aug = self._augment_with_clusters(X_train, clusterer)
            X_test_aug = self._augment_with_clusters(X_test, clusterer)

            classifier = self._fit_classifier(
                X_train_aug,
                label_encoder.transform(y_train),
            )

            proba = classifier.predict_proba(X_test_aug)
            predictions = label_encoder.inverse_transform(classifier.predict(X_test_aug))
            fold_metrics = self._evaluate_predictions(
                y_test,
                predictions,
                proba,
                label_encoder,
                n_classes,
            )
            fold_metrics["fold"] = float(fold_idx)
            folds.append(fold_metrics)
        return folds

    def _fit_clusterer(self, features: pd.DataFrame) -> MiniBatchKMeans:
        clusterer = MiniBatchKMeans(
            n_clusters=self.training_config.cluster_count,
            random_state=self.training_config.random_state,
            batch_size=2048,
            n_init="auto",
        )
        clusterer.fit(features)
        return clusterer

    def _augment_with_clusters(self, features: pd.DataFrame, clusterer: MiniBatchKMeans) -> pd.DataFrame:
        distances = clusterer.transform(features)
        labels = clusterer.predict(features)
        dist_columns = [f"cluster_dist_{i}" for i in range(distances.shape[1])]
        dist_df = pd.DataFrame(distances, index=features.index, columns=dist_columns)
        label_df = pd.DataFrame({"cluster_label": labels}, index=features.index)
        return pd.concat([features, dist_df, label_df], axis=1)

    def _fit_classifier(self, features: pd.DataFrame, labels: np.ndarray) -> HistGradientBoostingClassifier:
        classifier = HistGradientBoostingClassifier(
            learning_rate=self.training_config.learning_rate,
            max_iter=self.training_config.max_iter,
            max_depth=self.training_config.max_depth,
            max_leaf_nodes=self.training_config.max_leaf_nodes,
            early_stopping=self.training_config.early_stopping,
            validation_fraction=self.training_config.validation_fraction,
            random_state=self.training_config.random_state,
        )
        classifier.fit(features, labels)
        return classifier

    def _evaluate_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        proba: np.ndarray,
        label_encoder: LabelEncoder,
        n_classes: int,
    ) -> Dict[str, float]:
        y_true_encoded = label_encoder.transform(y_true)
        y_pred_encoded = label_encoder.transform(y_pred)
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
        try:
            metrics["log_loss"] = float(log_loss(y_true_encoded, proba, labels=np.arange(n_classes)))
        except ValueError:
            metrics["log_loss"] = float("nan")
        metrics["support"] = float(len(y_true))
        unique, counts = np.unique(y_pred_encoded, return_counts=True)
        pred_dist = {label_encoder.inverse_transform([cls])[0]: float(count / len(y_pred)) for cls, count in zip(unique, counts)}
        metrics["predicted_distribution"] = pred_dist
        return metrics

    def _extract_feature_importances(
        self,
        classifier: HistGradientBoostingClassifier,
        feature_names: pd.Index,
    ) -> Dict[str, float]:
        if not hasattr(classifier, "feature_importances_"):
            return {}
        importances = classifier.feature_importances_
        return {name: float(score) for name, score in zip(feature_names, importances)}

    @staticmethod
    def _aggregate_metrics(cv_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not cv_metrics:
            return {}
        keys = {k for metric in cv_metrics for k in metric.keys() if k not in {"fold", "predicted_distribution"}}
        aggregated: Dict[str, float] = {}
        for key in keys:
            values = [metric[key] for metric in cv_metrics if key in metric]
            if not values:
                continue
            aggregated[f"cv_{key}"] = float(np.nanmean(values))
        return aggregated
