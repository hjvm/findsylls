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
from typing import Tuple
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

    @property
    def supports_attention(self) -> bool:
        """
        Whether this extractor can return a CLS-style attention trace.

        Extractors that do not expose attention leave this as False.
        """
        return False

    @property
    def has_cls_token(self) -> bool:
        """
        Whether this extractor injects a CLS token at position 0.

        When True, raw attention has shape [n_heads, T+1, T+1] and CLS
        attention is extracted via attention[:, 0, 1:].
        When False (default), raw attention has shape [n_heads, T, T] and
        saliency is derived via attention.sum(1) - diagonal (no-CLS formula
        from Peng & Harwath 2022).
        """
        return False

    @property
    def is_dp_mincut_calibrated(self) -> bool:
        """
        Whether this extractor's features are calibrated for the SyllableLM DP
        MinCut parameters (delta=0.0033, quantile=0.75 at 8.33 Hz).

        Defaults to False for all extractors. Override to True only for
        Data2Vec2-based extractors (or any extractor for which you have
        explicitly re-tuned delta/quantile). MinCutSegmenter(use_reference=True)
        checks this property and suppresses its calibration warning when True.
        """
        return False

    def extract_with_attention(
        self,
        audio: np.ndarray,
        sr: int,
        return_raw: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract frame features plus attention weights aligned by frame.

        Args:
            audio: Audio waveform (mono, float32)
            sr: Sample rate in Hz
            return_raw: If True, return raw multi-head attention [n_heads, seq_len, src_len].
                       If False (default), return pre-aggregated 1-D attention [seq_len].
                       Raw mode is for downstream aggregation (segmente/envelope consume raw data).
            **kwargs: Additional extractor-specific parameters

        Returns:
            Tuple (features, attention) where:
                - features has shape (N, D)
                - attention has shape (n_heads, N, src_len) if return_raw=True,
                  else (N,) if return_raw=False
        
        Raises:
            RuntimeError: If extractor does not support attention
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support attention extraction"
        )

    def release(self) -> None:
        """Release extractor-held resources.

        Heavy extractors can override this to free model state after a workflow
        boundary (e.g., end of a corpus pass).
        """
        return None


__all__ = ['FeatureExtractor']
