"""MiniBatch K-means discovery model for streaming corpora."""

import numpy as np

from .base import BaseDiscoveryModel


class MiniBatchKMeansDiscovery(BaseDiscoveryModel):
    """Cluster embeddings using scikit-learn MiniBatchKMeans."""

    def __init__(self, n_clusters: int = 50, random_state: int = 0, batch_size: int = 4096, **kwargs):
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:
            raise ImportError(
                "MiniBatchKMeansDiscovery requires scikit-learn. Install with: pip install scikit-learn"
            ) from exc

        self._model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            **kwargs,
        )

    @property
    def supports_streaming(self) -> bool:
        return True

    def fit(self, embeddings: np.ndarray):
        self._model.fit(embeddings)
        return self

    def partial_fit(self, embeddings: np.ndarray):
        self._model.partial_fit(embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict(embeddings)
