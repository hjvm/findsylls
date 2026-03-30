"""
Base class for feature extraction methods.

Feature extractors convert raw audio into frame-level representations
that can be used for various tasks (segmentation, embedding, classification).

All feature extractors must:
- Take audio (np.ndarray) and sample rate as input
- Return features as (N, D) numpy array where N is number of frames
- Specify their frame rate in Hz
"""

from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    Feature extractors produce frame-level representations from audio
    that can be used with various downstream tasks.
    
    Examples:
        - MFCC with deltas (classical speech features)
        - Mel-spectrograms (time-frequency representations)
        - HuBERT embeddings (learned contextualized features)
        - Sylber features (syllable-level learned features)
    """
    
    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract features from audio.
        
        Args:
            audio: Audio waveform (mono, float32)
            sr: Sample rate in Hz
        
        Returns:
            Features of shape (N, D) where N is number of frames
            and D is feature dimensionality
        """
        pass
    
    @property
    @abstractmethod
    def frame_rate(self) -> float:
        """
        Feature frame rate in Hz (frames per second).
        
        This is used to convert frame indices to time in seconds.
        """
        pass


__all__ = ['FeatureExtractor']
