"""
Feature extraction utilities and dispatcher.

Provides:
- CustomCallableExtractor: Wrap any function as a FeatureExtractor
- get_extractor(): Factory function to get extractors by name
"""

from typing import Callable
import numpy as np

from .base import FeatureExtractor
from .hubert import HuBERTExtractor
from .mfcc import MFCCExtractor
from .melspectrogram import MelSpectrogramExtractor
from .sylber import SylberFeatureExtractor
from .vg_hubert import VGHuBERTFeatureExtractor


class CustomCallableExtractor(FeatureExtractor):
    """
    Wrap any callable as a FeatureExtractor.
    
    Allows using custom feature extraction functions with the standard interface.
    
    Args:
        extract_fn: Function that takes (audio, sr) and returns features (N, D)
        frame_rate: Frame rate of extracted features in Hz
    
    Example:
        >>> def my_features(audio, sr):
        ...     # Custom feature extraction
        ...     return features  # (N, D) array
        >>> 
        >>> extractor = CustomCallableExtractor(my_features, frame_rate=50.0)
        >>> features = extractor.extract(audio, sr=16000)
    """
    
    def __init__(self, extract_fn: Callable, frame_rate: float):
        self._extract_fn = extract_fn
        self._frame_rate = frame_rate
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Call wrapped function."""
        return self._extract_fn(audio, sr)
    
    @property
    def frame_rate(self) -> float:
        """Return specified frame rate."""
        return self._frame_rate


def get_extractor(feature_type: str, **kwargs) -> FeatureExtractor:
    """
    Factory function to get feature extractor by name.
    
    Args:
        feature_type: Type of features to extract
                     - 'hubert' or 'hub': Vanilla HuBERT
                     - 'sylber': Sylber's fine-tuned HuBERT
                     - 'vg-hubert' or 'vghubert': VG-HuBERT
                     - 'mfcc': MFCC features
                     - 'mel', 'melspec', 'melspectrogram': Mel-spectrogram
        **kwargs: Additional arguments passed to extractor constructor
    
    Returns:
        FeatureExtractor instance
    
    Example:
        >>> extractor = get_extractor('hubert', layer=9)
        >>> extractor = get_extractor('sylber')
        >>> extractor = get_extractor('mfcc', n_mfcc=13, include_deltas=True)
    """
    feature_type = feature_type.lower().replace('-', '').replace('_', '')
    
    if feature_type in ['hubert', 'hub']:
        return HuBERTExtractor(**kwargs)
    elif feature_type in ['sylber']:
        return SylberFeatureExtractor(**kwargs)
    elif feature_type in ['vghubert']:
        return VGHuBERTFeatureExtractor(**kwargs)
    elif feature_type in ['mfcc']:
        return MFCCExtractor(**kwargs)
    elif feature_type in ['mel', 'melspec', 'melspectrogram']:
        return MelSpectrogramExtractor(**kwargs)
    else:
        raise ValueError(
            f"Unknown feature type: {feature_type}. "
            f"Available: 'hubert', 'sylber', 'vg-hubert', 'mfcc', 'mel'"
        )


__all__ = [
    'FeatureExtractor',
    'HuBERTExtractor',
    'SylberFeatureExtractor',
    'VGHuBERTFeatureExtractor',
    'MFCCExtractor',
    'MelSpectrogramExtractor',
    'CustomCallableExtractor',
    'get_extractor',
]
