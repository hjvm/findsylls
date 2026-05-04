"""
Abstract base class for envelope computation methods.

This module defines the core interface that all envelope computation
methods should implement, enabling modular mixing of envelope methods
with segmentation algorithms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from ..features.base import FeatureExtractor


class EnvelopeComputer(ABC):
    """
    Abstract base class for envelope computation methods.
    
    Envelope computers take raw audio and produce an amplitude envelope
    that can be used by envelope-based segmentation algorithms (like
    Billauer's peakdetect).
    
    This enables mixing-and-matching different envelope methods with
    different segmentation algorithms in a modular way.
    
    All envelope computers should:
    - Take audio (numpy array) and sample rate as input
    - Return envelope (1D array) and corresponding time points
    - Be combinable with any envelope-based segmentation algorithm
    
    Examples:
        - Hilbert envelope
        - Theta oscillator envelope
        - Spectral band subtraction (SBS)
        - Lowpass filtered envelope
        
    Usage:
        >>> from findsylls.envelope.base import EnvelopeComputer
        >>> from findsylls.envelope.hilbert import HilbertEnvelope
        >>> 
        >>> computer = HilbertEnvelope()
        >>> envelope, times = computer.compute(audio, sr=16000)
    """
    
    @abstractmethod
    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute amplitude envelope from audio.
        
        Args:
            audio: Audio waveform (mono, float32)
            sr: Sample rate in Hz
        
        Returns:
            Tuple of (envelope, times) where:
            - envelope: 1D array of amplitude values
            - times: 1D array of time points in seconds
        """
        pass


class PseudoEnvelope(EnvelopeComputer, ABC):
    """Shared base class for feature-derived pseudo-envelope computers.

    Pseudo-envelopes are not independent segmentation algorithms. They expose a
    1-D trace derived from canonical segmentation internals for diagnostics,
    plotting, and envelope-based experiments.
    """

    def __init__(self, feature_extractor: FeatureExtractor, normalize: bool = True):
        self.feature_extractor = feature_extractor
        self.normalize = normalize

    def _normalize_envelope(self, envelope: np.ndarray) -> np.ndarray:
        envelope = np.asarray(envelope, dtype=np.float32)
        if not self.normalize:
            return envelope
        env_min = float(envelope.min()) if envelope.size > 0 else 0.0
        env_max = float(envelope.max()) if envelope.size > 0 else 0.0
        if env_max > env_min:
            return (envelope - env_min) / (env_max - env_min)
        return np.zeros_like(envelope, dtype=np.float32)

    def _frame_times(self, num_frames: int, frame_rate: Optional[float] = None) -> np.ndarray:
        if frame_rate is None:
            frame_rate = float(self.feature_extractor.frame_rate)
        return (np.arange(num_frames, dtype=np.float32) / float(frame_rate)).astype(np.float32)
