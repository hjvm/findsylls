"""Agglomerative discovery model."""

import numpy as np

from .base import BaseDiscoveryModel


class AgglomerativeDiscovery(BaseDiscoveryModel):
    """Cluster embeddings using scikit-learn AgglomerativeClustering."""

    def __init__(self, n_clusters: int = 50, **kwargs):
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError as exc:
            raise ImportError(
                "AgglomerativeDiscovery requires scikit-learn. Install with: pip install scikit-learn"
            ) from exc

        self._model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)

    def fit(self, embeddings: np.ndarray):
        self._model.fit(embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        # AgglomerativeClustering does not support out-of-sample prediction.
        # We expose labels for fitted data and require same input length.
        if not hasattr(self._model, "labels_"):
            raise RuntimeError("Model must be fitted before predict")
        if embeddings.shape[0] != self._model.labels_.shape[0]:
            raise ValueError(
                "AgglomerativeDiscovery can only return labels for fitted data. "
                "Pass the same embeddings used in fit()."
            )
        return self._model.labels_
