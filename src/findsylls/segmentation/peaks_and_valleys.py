"""
Peaks and valleys segmentation.

Generic peak detection algorithm using Eli Billauer's peakdetect.
This is the core segmentation algorithm used by multiple envelope-based methods:
- Theta oscillator (Räsänen et al. 2018) - uses this with theta envelope
- Generic envelope segmentation - can use this with any envelope (SBS, Hilbert, etc.)

Note: When using theta envelope specifically, this implements the complete
theta oscillator method from Räsänen et al. (2018), not a separate method.
"""

import numpy as np
from typing import List, Tuple, Optional

from .base import EnvelopeBasedSegmenter
from .peakdetect_segmenter import segment_peakdetect


def segment_peaks_and_valleys(envelope: np.ndarray, times: np.ndarray, **kwargs) -> List[Tuple[float, float, float]]:
    """
    Segment envelope using peak detection.
    
    This is just an alias to segment_peakdetect() for backward compatibility.
    """
    return segment_peakdetect(envelope, times, **kwargs)


class PeaksAndValleysSegmenter(EnvelopeBasedSegmenter):
    """
    Segment based on peaks and valleys in amplitude envelope.
    
    This is a classical signal processing method that:
    1. Takes raw audio (or pre-computed envelope)
    2. Computes amplitude envelope if needed
    3. Detects peaks (syllable nuclei) and valleys (boundaries)
    4. Merges nearby valleys and filters by duration
    
    Args:
        sample_rate: Target sample rate (default: 16000)
        envelope_method: Method for computing envelope (default: 'hilbert')
        delta: Minimum peak/valley height difference (default: 0.01)
        lookahead: Samples to look ahead for peak detection (auto-computed if not set)
        min_syllable_dur: Minimum syllable duration in seconds (default: 0.05)
        onset: Time threshold for adding initial/final valleys (default: 0.05)
        merge_valley_tol: Time tolerance for merging nearby valleys (default: 0.05)
    
    Example:
        >>> segmenter = PeaksAndValleysSegmenter(delta=0.02)
        >>> segments = segmenter.segment(audio=audio, sr=16000)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        envelope_method: str = 'hilbert',
        delta: float = 0.01,
        lookahead: Optional[int] = None,
        min_syllable_dur: float = 0.05,
        onset: float = 0.05,
        merge_valley_tol: float = 0.05
    ):
        super().__init__(sample_rate)
        self.envelope_method = envelope_method
        self.delta = delta
        self.lookahead = lookahead
        self.min_syllable_dur = min_syllable_dur
        self.onset = onset
        self.merge_valley_tol = merge_valley_tol
    
    def segment(
        self,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        envelope: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment from audio or pre-computed envelope.
        
        Args:
            audio: Raw audio waveform (if envelope not provided)
            sr: Sample rate (if audio provided)
            envelope: Pre-computed envelope (optional, for backward compatibility)
            times: Time array for envelope (optional, for backward compatibility)
            **kwargs: Override default parameters
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        # Compute envelope if not provided
        if envelope is None or times is None:
            if audio is None:
                raise ValueError("Must provide either audio or (envelope, times)")
            
            # Import envelope computation
            from ..envelope import get_amplitude_envelope
            import librosa
            
            sr = sr or self.sample_rate
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            # Compute envelope
            envelope_method = kwargs.get('envelope_method', self.envelope_method)
            envelope, times = get_amplitude_envelope(
                audio, sr, method=envelope_method
            )
        
        # Merge instance params with kwargs
        params = {
            'delta': kwargs.get('delta', self.delta),
            'min_syllable_dur': kwargs.get('min_syllable_dur', self.min_syllable_dur),
            'onset': kwargs.get('onset', self.onset),
            'merge_valley_tol': kwargs.get('merge_valley_tol', self.merge_valley_tol),
        }
        
        # Add lookahead only if explicitly set
        if 'lookahead' in kwargs:
            params['lookahead'] = kwargs['lookahead']
        elif self.lookahead is not None:
            params['lookahead'] = self.lookahead
        
        # Call existing function
        segments = segment_peaks_and_valleys(envelope, times, **params)
        
        # Validate output
        self._validate_output(segments)
        
        return segments
    
    def __repr__(self) -> str:
        return (
            f"PeaksAndValleysSegmenter("
            f"envelope_method='{self.envelope_method}', "
            f"delta={self.delta}, "
            f"min_syllable_dur={self.min_syllable_dur})"
        )
