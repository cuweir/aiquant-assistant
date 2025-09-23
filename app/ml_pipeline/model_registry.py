from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from .artifacts import ModelArtifact

DEFAULT_MODEL_ROOT = Path("data/models")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def _model_root(base_path: Optional[Path] = None) -> Path:
    root = base_path or DEFAULT_MODEL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_text(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def save_model(
    name: str,
    artifact: ModelArtifact,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    base_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Persist a trained model bundle to disk."""

    root = _model_root(base_path)
    run_identifier = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_dir = root / name / run_identifier
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_dir / "model.joblib")
    _write_text(model_dir / "metadata.json", metadata)
    _write_text(model_dir / "metrics.json", metrics)
    (root / name / "latest.txt").write_text(run_identifier)
    return model_dir


def _resolve_run_id(name: str, base_path: Optional[Path], run_id: Optional[str]) -> Tuple[Path, str]:
    root = _model_root(base_path)
    model_root = root / name
    if run_id is None:
        latest_marker = model_root / "latest.txt"
        if latest_marker.exists():
            run_id = latest_marker.read_text().strip()
        else:
            candidates = sorted((p.name for p in model_root.iterdir() if p.is_dir()), reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No saved runs for model '{name}'")
            run_id = candidates[0]
    target_dir = model_root / run_id
    if not target_dir.exists():
        raise FileNotFoundError(f"Model run directory missing: {target_dir}")
    return target_dir, run_id


def load_model(
    name: str,
    base_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Tuple[ModelArtifact, Dict[str, Any], Dict[str, Any], str]:
    """Load a previously saved model bundle along with metadata and metrics."""

    model_dir, resolved_id = _resolve_run_id(name, base_path, run_id)
    artifact: ModelArtifact = joblib.load(model_dir / "model.joblib")
    metadata_path = model_dir / "metadata.json"
    metrics_path = model_dir / "metrics.json"
    metadata: Dict[str, Any] = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metrics: Dict[str, Any] = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return artifact, metadata, metrics, resolved_id


def list_runs(name: str, base_path: Optional[Path] = None) -> List[str]:
    """Return the available run identifiers for a given model name."""

    root = _model_root(base_path)
    model_root = root / name
    if not model_root.exists():
        return []
    return sorted((p.name for p in model_root.iterdir() if p.is_dir()), reverse=True)
