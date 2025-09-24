#!/usr/bin/env python
"""Train the ML dynamic decision model from a saved dataset bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.ml_pipeline.dataset_builder import load_bundle
from app.ml_pipeline.model_registry import save_model
from app.ml_pipeline.schemas import FeatureConfig, LabelConfig
from app.ml_pipeline.trainer import ModelTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML model from dataset bundle")
    parser.add_argument("dataset", type=Path, help="Path to dataset directory containing meta.json")
    parser.add_argument("model_name", type=str, help="Name to register the trained model under")
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"), help="Directory to store trained models")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward splits for cross validation")
    parser.add_argument("--cluster-count", type=int, default=6, help="Number of unsupervised clusters to learn")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for gradient boosting")
    parser.add_argument("--max-iter", type=int, default=500, help="Maximum boosting iterations")
    parser.add_argument("--run-id", type=str, default=None, help="Optional custom run identifier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.dataset)

    feature_cfg_data = bundle.meta.get("feature_config")
    label_cfg_data = bundle.meta.get("label_config")
    if not feature_cfg_data or not label_cfg_data:
        raise ValueError("Dataset bundle is missing feature or label configuration metadata.")

    feature_config = FeatureConfig.from_dict(feature_cfg_data)
    label_config = LabelConfig.from_dict(label_cfg_data)

    training_config = TrainingConfig(
        n_splits=args.n_splits,
        cluster_count=args.cluster_count,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
    )

    trainer = ModelTrainer(feature_config, label_config, training_config)
    result = trainer.train(bundle)

    metrics: Dict[str, float] = result.metrics.copy()
    metrics["class_distribution"] = result.class_distribution

    metadata = {
        "symbol": bundle.symbol,
        "timeframe": bundle.timeframe,
        "rows": bundle.meta.get("rows"),
        "feature_fingerprint": bundle.meta.get("feature_fingerprint"),
        "label_fingerprint": bundle.meta.get("label_fingerprint"),
        "feature_config": feature_config.to_dict(),
        "label_config": label_config.to_dict(),
        "cv_metrics": result.cv_metrics,
    }

    model_dir = save_model(
        args.model_name,
        result.artifact,
        metadata=metadata,
        metrics=metrics,
        base_path=args.output_dir,
        run_id=args.run_id,
    )

    print("Model training complete.")
    print(f"  > Saved model to: {model_dir}")
    print("  > Metrics:")
    for key, value in metrics.items():
        print(f"    - {key}: {value}")


if __name__ == "__main__":
    main()
