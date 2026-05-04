"""K-means discovery model."""

import numpy as np

from .base import BaseDiscoveryModel


class KMeansDiscovery(BaseDiscoveryModel):
    """Cluster embeddings using scikit-learn KMeans."""

    def __init__(self, n_clusters: int = 50, random_state: int = 0, **kwargs):
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "KMeansDiscovery requires scikit-learn. Install with: pip install scikit-learn"
            ) from exc

        self._model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)

    def fit(self, embeddings: np.ndarray):
        self._model.fit(embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict(embeddings)
