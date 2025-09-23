from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ModelArtifact:
    """Serializable container bundling everything required for inference."""

    classifier: Any
    clusterer: Optional[Any]
    label_encoder: Any
    feature_columns: List[str]
    augmented_feature_columns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def class_labels(self) -> List[str]:
        labels = getattr(self.label_encoder, "classes_", None)
        return labels.tolist() if labels is not None else []

