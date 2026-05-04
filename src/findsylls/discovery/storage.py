"""Persistence helpers for discovery pipelines and fitted models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np


PIPELINE_FILENAME = "discovery_pipeline.joblib"
METADATA_FILENAME = "discovery_metadata.json"


def compute_intrinsic_fit_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute standard intrinsic clustering metrics for fitted embeddings."""
    metrics: Dict[str, Any] = {
        "n_samples": int(embeddings.shape[0]),
        "n_features": int(embeddings.shape[1]) if embeddings.ndim > 1 else 1,
        "n_clusters": int(len(np.unique(labels))),
    }

    if embeddings.size == 0 or labels.size == 0:
        metrics.update(
            {
                "silhouette": None,
                "davies_bouldin": None,
                "calinski_harabasz": None,
                "status": "empty_input",
            }
        )
        return metrics

    n_samples = embeddings.shape[0]
    n_clusters = len(np.unique(labels))

    if n_clusters < 2 or n_samples < 2 or n_clusters >= n_samples:
        metrics.update(
            {
                "silhouette": None,
                "davies_bouldin": None,
                "calinski_harabasz": None,
                "status": "insufficient_cluster_structure",
            }
        )
        return metrics

    try:
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    except ImportError as exc:
        raise ImportError(
            "Intrinsic discovery metrics require scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    metrics["silhouette"] = float(silhouette_score(embeddings, labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(embeddings, labels))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))
    metrics["status"] = "ok"
    return metrics


def save_discovery_pipeline(
    pipeline: Any,
    output_dir: str | Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Persist a fitted discovery pipeline and its metadata sidecar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / PIPELINE_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    if not overwrite and (model_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Discovery artifacts already exist in {output_dir}. Set overwrite=True to replace them."
        )

    joblib.dump(pipeline, model_path)

    payload: Dict[str, Any] = {
        "pipeline_class": type(pipeline).__name__,
        "method": getattr(pipeline, "method", None),
        "model_kwargs": getattr(pipeline, "model_kwargs", {}),
        "fit_metrics": getattr(pipeline, "fit_metrics", None),
    }
    if metadata:
        payload["metadata"] = metadata

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return output_dir


def load_discovery_pipeline(output_dir: str | Path):
    """Load a persisted discovery pipeline from disk."""
    output_dir = Path(output_dir)
    model_path = output_dir / PIPELINE_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Missing discovery pipeline artifact: {model_path}")

    pipeline = joblib.load(model_path)

    if not hasattr(pipeline, "fit_metrics"):
        pipeline.fit_metrics = None
    if not hasattr(pipeline, "_fit_metrics"):
        pipeline._fit_metrics = getattr(pipeline, "fit_metrics", None)

    return pipeline
