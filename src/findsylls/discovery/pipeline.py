"""Pipeline orchestration for syllable discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .dispatch import get_discovery_model
from .storage import (
    compute_intrinsic_fit_metrics,
    load_discovery_pipeline as load_persisted_pipeline,
    save_discovery_pipeline,
)
from .types import DiscoveryResult


class DiscoveryPipeline:
    """Orchestrate clustering-based syllable discovery from embeddings."""

    def __init__(self, method: str = "kmeans", model_kwargs: Optional[Dict[str, Any]] = None):
        self.method = method
        self.model_kwargs = model_kwargs or {}
        self.model = get_discovery_model(method, **self.model_kwargs)
        self.fit_metrics: Optional[Dict[str, Any]] = None

    @property
    def fitted_model(self):
        """Expose the wrapped fitted estimator for persistence/debugging."""
        return self.model._model

    def fit(self, embeddings: np.ndarray) -> "DiscoveryPipeline":
        self.model.fit(embeddings)
        labels = self.model.predict(embeddings)
        self.fit_metrics = compute_intrinsic_fit_metrics(embeddings, labels)
        return self

    def fit_from_iterator(self, embedding_chunks: Iterable[np.ndarray]) -> "DiscoveryPipeline":
        """Fit model incrementally from embedding chunks."""
        if not self.model.supports_streaming:
            raise NotImplementedError(
                f"Model '{self.method}' does not support streaming fit. "
                "Use fit(...) with an in-memory matrix."
            )

        saw_data = False
        for chunk in embedding_chunks:
            if chunk is None or chunk.size == 0:
                continue
            saw_data = True
            self.model.partial_fit(chunk)

        if not saw_data:
            raise ValueError("No non-empty embedding chunks were provided")

        self.fit_metrics = None

        return self

    def fit_from_storage(
        self,
        manifest_path: str,
        chunk_size: int = 10000,
    ) -> "DiscoveryPipeline":
        """Fit model from storage manifest without loading all embeddings at once."""
        from ..embedding.storage import iter_embeddings_from_manifest

        return self.fit_from_iterator(
            iter_embeddings_from_manifest(manifest_path=manifest_path, chunk_size=chunk_size)
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict(embeddings)

    def predict_from_iterator(self, embedding_chunks: Iterable[np.ndarray]) -> np.ndarray:
        """Predict labels from embedding chunks and concatenate outputs."""
        labels = []
        for chunk in embedding_chunks:
            if chunk is None or chunk.size == 0:
                continue
            labels.append(self.model.predict(chunk))
        if not labels:
            return np.array([], dtype=np.int64)
        return np.concatenate(labels)

    def predict_from_storage(
        self,
        manifest_path: str,
        chunk_size: int = 10000,
    ) -> np.ndarray:
        """Predict labels from storage manifest in chunks."""
        from ..embedding.storage import iter_embeddings_from_manifest

        return self.predict_from_iterator(
            iter_embeddings_from_manifest(manifest_path=manifest_path, chunk_size=chunk_size)
        )

    def discover(self, embeddings: np.ndarray) -> DiscoveryResult:
        self.fit(embeddings)
        labels = self.model.predict(embeddings)
        return DiscoveryResult(
            labels=labels,
            num_clusters=int(len(np.unique(labels))),
            model_name=self.method,
            fit_metrics=self.fit_metrics,
            metadata={"model_kwargs": self.model_kwargs, "fit_metrics": self.fit_metrics},
        )

    def discover_from_storage(
        self,
        manifest_path: str,
        chunk_size: int = 10000,
    ) -> DiscoveryResult:
        """
        Fit and predict from storage-backed embedding chunks.

        For non-streaming models, this method raises NotImplementedError.
        """
        self.fit_from_storage(manifest_path=manifest_path, chunk_size=chunk_size)
        labels = self.predict_from_storage(manifest_path=manifest_path, chunk_size=chunk_size)
        return DiscoveryResult(
            labels=labels,
            num_clusters=int(len(np.unique(labels))) if labels.size > 0 else 0,
            model_name=self.method,
            fit_metrics=self.fit_metrics,
            metadata={
                "model_kwargs": self.model_kwargs,
                "manifest_path": manifest_path,
                "chunk_size": chunk_size,
                "fit_metrics": self.fit_metrics,
            },
        )

    def save(self, output_dir: str | Path, *, metadata: Optional[Dict[str, Any]] = None, overwrite: bool = False):
        """Persist the fitted discovery pipeline and metadata sidecar."""
        return save_discovery_pipeline(self, output_dir, metadata=metadata, overwrite=overwrite)

    @classmethod
    def load(cls, output_dir: str | Path) -> "DiscoveryPipeline":
        """Load a persisted discovery pipeline."""
        pipeline = load_persisted_pipeline(output_dir)
        if not isinstance(pipeline, cls):
            raise TypeError(
                f"Expected a persisted {cls.__name__}, got {type(pipeline).__name__}"
            )
        return pipeline
