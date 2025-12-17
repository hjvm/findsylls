"""
Base classes for syllable segmentation methods.

This module defines the abstract base classes for two types of segmentation approaches:

1. EnvelopeBasedSegmenter: Classical signal processing methods (Methods 1-7)
   - Take raw audio as input
   - Compute method-specific envelope/feature curve internally
   - Apply peak/valley detection or similar algorithms
   - Signal processing approach (fast, no GPU needed)

2. End2EndSegmenter: End-to-end neural methods (Methods 8-11)
   - Take raw audio as input
   - Process through learned representations (typically transformers)
   - Produce syllable boundaries end-to-end
   - Neural network approach (GPU-accelerated, pre-trained models)

Both types work from raw audio but differ in their approach: classical methods
compute interpretable envelopes as an intermediate step, while end-to-end models
learn the entire segmentation pipeline from data.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Protocol
import numpy as np


class SegmenterProtocol(Protocol):
    """
    Protocol for duck-typing segmenter objects.
    
    Any object with a segment() method returning List[Tuple[float, float, float]]
    can be used as a segmenter.
    """
    
    def segment(self, **kwargs) -> List[Tuple[float, float, float]]:
        """Segment audio into syllables."""
        ...


class BaseSegmenter(ABC):
    """
    Abstract base class for all segmentation methods.
    
    All segmenters must:
    - Implement segment() method
    - Return list of (start, nucleus, end) tuples in seconds
    - Accept raw audio as input
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize segmenter.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    @abstractmethod
    def segment(self, **kwargs) -> List[Tuple[float, float, float]]:
        """
        Segment audio into syllables.
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
            - start: syllable onset time
            - nucleus: approximate syllable nucleus/peak time
            - end: syllable offset time
        """
        pass
    
    def _validate_output(self, segments: List[Tuple[float, float, float]]) -> None:
        """
        Validate output format.
        
        Args:
            segments: List of (start, nucleus, end) tuples
            
        Raises:
            AssertionError: If segments are invalid
        """
        for i, (start, nucleus, end) in enumerate(segments):
            assert start <= nucleus <= end, \
                f"Invalid segment {i}: start={start}, nucleus={nucleus}, end={end}"
            assert start >= 0, f"Negative start time in segment {i}: {start}"
    
    def __call__(self, **kwargs) -> List[Tuple[float, float, float]]:
        """Allow using segmenter as callable."""
        return self.segment(**kwargs)


class EnvelopeBasedSegmenter(BaseSegmenter):
    """
    Base class for envelope-based segmentation methods (Methods 1-7).
    
    These are classical signal processing methods that:
    1. Take raw audio as input
    2. Compute a method-specific envelope/feature curve internally
    3. Apply peak/valley detection or similar algorithms
    4. Return syllable boundaries
    
    The envelope computation and segmentation are typically separate steps,
    making these methods interpretable and tuneable.
    
    Examples:
        - Peaks and valleys (Räsänen et al. 2018)
        - Theta oscillator (Räsänen et al. 2018)
        - Mermelstein (1975)
        - VSeg (Gammatone filterbank)
        - TCSSC (time-correlated spectral subband centroids)
    """
    
    @abstractmethod
    def segment(
        self, 
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        envelope: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment from raw audio or pre-computed envelope.
        
        Args:
            audio: Raw audio waveform (if envelope not provided)
            sr: Sample rate (if audio provided)
            envelope: Pre-computed envelope (optional, for backward compatibility)
            times: Time array for envelope (optional, for backward compatibility)
            **kwargs: Method-specific parameters
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        
        Note: Methods compute their own envelopes internally when working from audio.
        The envelope/times parameters are for backward compatibility with existing code.
        """
        pass


class End2EndSegmenter(BaseSegmenter):
    """
    Base class for end-to-end neural segmentation methods (Methods 8-11).
    
    These are neural network models that:
    1. Take raw audio as input
    2. Process through learned representations (typically transformer-based)
    3. Produce syllable boundaries end-to-end
    4. Often include pre-trained weights
    
    Unlike envelope-based methods, these learn the entire segmentation
    pipeline from data, making them more powerful but less interpretable.
    
    Examples:
        - Sylber (self-supervised syllabic distillation)
        - VG-HuBERT + MinCut (visually-grounded representations)
        - SD-HuBERT (self-distillation)
        - biLSTM (supervised learning)
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        device: str = 'cpu',
        cache: bool = True
    ):
        """
        Initialize end-to-end segmenter.
        
        Args:
            sample_rate: Target sample rate for audio processing
            device: Device for neural network ('cpu', 'cuda', 'mps')
            cache: Whether to cache model after first load
        """
        super().__init__(sample_rate)
        self.device = device
        self.cache = cache
        self._model = None  # Lazy loading
    
    @abstractmethod
    def segment(
        self, 
        audio: np.ndarray, 
        sr: int = 16000,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment from raw audio using end-to-end neural model.
        
        Args:
            audio: Raw audio waveform
            sr: Sample rate
            **kwargs: Method-specific parameters
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        pass
    
    def _lazy_load_model(self):
        """
        Lazy load model on first use.
        
        Subclasses should override this to load their specific model.
        Should set self._model to the loaded model.
        """
        raise NotImplementedError("Subclass must implement _lazy_load_model()")
    
    def _get_device(self) -> str:
        """
        Get appropriate device for model.
        
        Returns:
            Device string ('cpu', 'cuda', or 'mps')
        """
        if self.device == 'cuda':
            try:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                return 'cpu'
        elif self.device == 'mps':
            try:
                import torch
                return 'mps' if torch.backends.mps.is_available() else 'cpu'
            except ImportError:
                return 'cpu'
        return 'cpu'
