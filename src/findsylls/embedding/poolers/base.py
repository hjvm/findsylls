"""Base pooler interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BasePooler(ABC):
    """Poolers aggregate frame features over syllable spans."""

    @abstractmethod
    def pool(
        self,
        features: np.ndarray,
        syllables: List[Tuple[float, float, float]],
        fps: float,
        **kwargs,
    ) -> np.ndarray:
        """Return pooled embeddings with shape (num_syllables, embedding_dim)."""
