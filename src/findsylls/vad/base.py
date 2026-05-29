from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseSAD(ABC):
    """Abstract base for Speech Activity Detection (VAD) backends."""

    @abstractmethod
    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Return list of (start_s, end_s) speech region pairs, sorted by start time."""
        ...
