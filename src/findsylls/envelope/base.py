"""
Abstract base class for envelope computation methods.

This module defines the core interface that all envelope computation
methods should implement, enabling modular mixing of envelope methods
with segmentation algorithms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


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
