"""Base contracts for embedding layer components."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BasePooler(ABC):
    """Abstract interface for pooling frame-level features into syllable embeddings."""

    @abstractmethod
    def pool(
        self,
        features: np.ndarray,
        syllables: List[Tuple[float, float, float]],
        fps: float,
        **kwargs,
    ) -> np.ndarray:
        """Pool frame features into per-syllable embeddings."""


class BaseEmbeddingPipeline(ABC):
    """Abstract interface for embedding pipelines."""

    @abstractmethod
    def embed_audio(self, audio_path: str, **kwargs):
        """Embed a single audio file."""

    @abstractmethod
    def embed_corpus(self, audio_files, **kwargs):
        """Embed a batch of audio files."""

    @abstractmethod
    def embed_corpus_to_storage(self, audio_files, output_dir: str, **kwargs):
        """Embed a batch and write artifacts to disk incrementally."""
