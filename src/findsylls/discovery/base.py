"""Base class for discovery models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseDiscoveryModel(ABC):
    """Abstract interface for syllable discovery models."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray):
        """Fit model on embedding matrix."""

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict labels for embedding matrix."""

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and return labels in one call."""
        self.fit(embeddings)
        return self.predict(embeddings)

    def save(self, output_dir: str, **kwargs):
        """Persist the fitted model using the discovery storage helpers."""
        from .storage import save_discovery_pipeline

        return save_discovery_pipeline(self, output_dir, **kwargs)

    @property
    def supports_streaming(self) -> bool:
        """Whether model supports incremental/streaming fitting."""
        return False

    def partial_fit(self, embeddings: np.ndarray):
        """Incremental fit hook for streaming-capable models."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support partial_fit"
        )
