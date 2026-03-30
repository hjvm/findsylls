"""
Peakdetect segmentation using Eli Billauer's algorithm.

This module provides both functional and object-oriented APIs for envelope-based
segmentation using Eli Billauer's peak detection algorithm (from findpeaks).
This is used by multiple envelope-based methods (theta, SBS, Hilbert, etc.)
to detect syllable nuclei (peaks) and boundaries (valleys).

This is the ONLY place in the library that imports from findpeaks.peakdetect.
All other modules should use these wrappers.

Functional API:
    from findsylls.envelope.theta import theta_oscillator_envelope
    from findsylls.segmentation.peakdetect_segmenter import segment_peakdetect
    
    envelope, times = theta_oscillator_envelope(audio, sr)
    syllables = segment_peakdetect(envelope, times)

Object-Oriented API (for mixing envelopes with algorithms):
    from findsylls.segmentation.peakdetect_segmenter import PeakdetectSegmenter
    from findsylls.segmentation.custom_segmenters import EnvelopeComputer
    
    class HilbertEnvelope(EnvelopeComputer):
        def compute(self, audio, sr):
            from findsylls.envelope.dispatch import get_amplitude_envelope
            return get_amplitude_envelope(audio, sr, method='hilbert')
    
    segmenter = PeakdetectSegmenter(HilbertEnvelope(), delta=0.02)
    segments = segmenter.segment(audio, sr)
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from findpeaks.peakdetect import peakdetect

from .base import EnvelopeBasedSegmenter
from ..envelope.base import EnvelopeComputer


def segment_peakdetect(envelope: np.ndarray, times: np.ndarray, **kwargs) -> List[Tuple[float, float, float]]:
    """
    Segment envelope into syllable-like units using Billauer's peak detection algorithm.
    
    Args:
        envelope: Amplitude envelope (1D array)
        times: Time array corresponding to envelope samples (in seconds)
        **kwargs: Segmentation parameters:
            - delta: Minimum peak/valley height difference (default: 0.01)
            - min_syllable_dur: Minimum syllable duration in seconds (default: 0.05)
            - onset: Time threshold for adding initial/final valleys (default: 0.05)
            - merge_valley_tol: Time tolerance for merging nearby valleys (default: 0.05)
            - min_amplitude_threshold: Minimum amplitude as fraction of max envelope (default: 0.0)
                                      Filters peaks in silent regions. E.g., 0.1 = 10% of max.
            - lookahead: Samples to look ahead (auto-computed from min_syllable_dur if None)
    
    Returns:
        List of (start, nucleus, end) tuples in seconds
    """
    delta = kwargs.get("delta", 0.01)
    min_syllable_dur = kwargs.get("min_syllable_dur", 0.05)
    onset = kwargs.get("onset", 0.05)
    merge_tol = kwargs.get("merge_valley_tol", 0.05)
    min_amplitude_threshold = kwargs.get("min_amplitude_threshold", 0.0)  # NEW: Filter low-amplitude peaks

    # Auto-compute lookahead if not explicitly provided
    if 'lookahead' in kwargs:
        # Use explicit value if provided (for testing/comparison)
        lookahead = kwargs['lookahead']
    else:
        # Calculate lookahead based on min_syllable_dur.
        # Use half the min duration to avoid finding multiple peaks per syllable.
        lookahead_time = min_syllable_dur / 2.0  # e.g., 0.025s for default 0.05s
        
        # Convert time to samples based on envelope sampling rate
        dt = times[1] - times[0] if len(times) > 1 else 0.01  # Time per envelope sample
        lookahead = max(1, int(lookahead_time / dt))
    
    # NEW: Calculate amplitude threshold (relative to max envelope value)
    # min_amplitude_threshold is a fraction (e.g., 0.1 = 10% of max amplitude)
    if min_amplitude_threshold > 0:
        envelope_max = np.max(envelope)
        amplitude_cutoff = min_amplitude_threshold * envelope_max
    else:
        amplitude_cutoff = -np.inf  # No filtering

    raw_peaks, raw_valleys = peakdetect(envelope, lookahead=lookahead, delta=delta, x_axis=times)
    peaks = np.array([p[0] for p in raw_peaks])
    valleys_times = np.array([v[0] for v in raw_valleys])
    valleys_vals = np.array([v[1] for v in raw_valleys])
    if peaks.size == 0 or valleys_times.size == 0:
        return []
    diffs = np.diff(valleys_times)
    break_idxs = np.nonzero(diffs > merge_tol)[0] + 1
    groups = np.split(np.arange(len(valleys_times)), break_idxs)
    merged_valleys = []
    for grp in groups:
        sub_vals = valleys_vals[grp]
        best_idx = grp[np.argmin(sub_vals)]
        merged_valleys.append(valleys_times[best_idx])
    valleys = np.array(merged_valleys)
    if valleys[0] > onset:
        valleys = np.insert(valleys, 0, 0.0)
    if valleys[-1] < times[-1] - onset:
        valleys = np.append(valleys, times[-1])
    syllables = []
    for i in range(1, len(valleys)):
        left, right = valleys[i-1], valleys[i]
        mid_peaks = peaks[(peaks > left) & (peaks < right)]
        if mid_peaks.size == 0:
            continue
        best_peak = max(mid_peaks, key=lambda tsec: envelope[np.argmin(np.abs(times - tsec))])
        
        # Get amplitude at peak (nucleus)
        peak_amplitude = envelope[np.argmin(np.abs(times - best_peak))]
        
        # Filter by duration AND amplitude
        if (right - left) >= min_syllable_dur and peak_amplitude >= amplitude_cutoff:
            syllables.append((left, best_peak, right))
    return syllables


class PeakdetectSegmenter(EnvelopeBasedSegmenter):
    """
    Apply Billauer's peak detection algorithm to any feature extraction method.
    
    This segmenter separates feature extraction from segmentation algorithm, allowing
    you to mix-and-match:
    - Classical envelopes (SBS, Theta, Hilbert) + peak detection
    - Neural features (Sylber, VG-HuBERT) + peak detection
    - Any custom envelope + peak detection
    
    Args:
        envelope_computer: Feature extractor - EnvelopeComputer instance or callable
                          that returns (envelope/features, times). Can be:
                          - Classical: HilbertEnvelope(), ThetaEnvelope(), etc.
                          - Neural: Sylber feature extractor, VG-HuBERT feature extractor
                          - Custom: Any callable(audio, sr) -> (features, times)
        delta: Minimum peak/valley height difference (default: 0.01)
        lookahead: Samples to look ahead for peak detection (auto-computed if None)
        min_syllable_dur: Minimum syllable duration in seconds (default: 0.05)
        onset: Time threshold for adding initial/final valleys (default: 0.05)
        merge_valley_tol: Time tolerance for merging nearby valleys (default: 0.05)
        min_amplitude_threshold: Minimum amplitude threshold as fraction of max envelope
                                 amplitude (default: 0.0). Filters peaks in silent regions.
                                 E.g., 0.1 means peaks must be at least 10% of max amplitude.
    
    Examples:
        >>> # Classical envelope
        >>> from findsylls.envelope.theta import ThetaEnvelope
        >>> segmenter = PeakdetectSegmenter(ThetaEnvelope(f=5, Q=0.5), delta=0.02)
        >>> segments = segmenter.segment(audio=audio, sr=16000)
        
        >>> # Or use pre-computed envelope
        >>> envelope, times = theta_envelope_function(audio, sr)
        >>> segments = segmenter.segment(envelope=envelope, times=times)
        
        >>> # Neural features as envelope (future extension)
        >>> # class VGHubertFeatureExtractor(EnvelopeComputer):
        >>> #     def compute(self, audio, sr):
        >>> #         features = vg_hubert.extract(audio, sr)
        >>> #         return features, times
        >>> # segmenter = PeakdetectSegmenter(VGHubertFeatureExtractor())
    """
    
    def __init__(
        self,
        envelope_computer: Optional[Union[EnvelopeComputer, callable]] = None,
        delta: float = 0.01,
        lookahead: Optional[int] = None,
        min_syllable_dur: float = 0.05,
        onset: float = 0.05,
        merge_valley_tol: float = 0.05,
        min_amplitude_threshold: float = 0.0,
    ):
        super().__init__()
        self.envelope_computer = envelope_computer
        self.delta = delta
        self.lookahead = lookahead
        self.min_syllable_dur = min_syllable_dur
        self.onset = onset
        self.merge_valley_tol = merge_valley_tol
        self.min_amplitude_threshold = min_amplitude_threshold
    
    def segment(
        self,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        envelope: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio using envelope + Billauer's peak detection.
        
        Supports two modes:
        1. From raw audio: segment(audio=audio, sr=sr) 
           - Computes envelope using self.envelope_computer
        2. From pre-computed envelope: segment(envelope=env, times=times)
           - Uses provided envelope directly
        
        Args:
            audio: Raw audio waveform (required if envelope not provided)
            sr: Sample rate (required if audio provided)
            envelope: Pre-computed envelope (optional, for pre-processing)
            times: Time array for envelope (required if envelope provided)
            **kwargs: Override segmentation parameters (delta, min_syllable_dur, etc.)
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        # Mode 1: Compute envelope from audio
        if envelope is None:
            if audio is None or sr is None:
                raise ValueError("Must provide either (audio, sr) or (envelope, times)")
            if self.envelope_computer is None:
                raise ValueError("Must provide envelope_computer or pre-computed envelope")
            
            # Compute envelope using configured computer
            if hasattr(self.envelope_computer, 'compute'):
                envelope, times = self.envelope_computer.compute(audio, sr)
            else:
                envelope, times = self.envelope_computer(audio, sr)
        
        # Mode 2: Use pre-computed envelope
        else:
            if times is None:
                raise ValueError("Must provide times array when using pre-computed envelope")
        
        # Build kwargs - use instance params unless overridden
        peak_kwargs = {
            'delta': kwargs.get('delta', self.delta),
            'min_syllable_dur': kwargs.get('min_syllable_dur', self.min_syllable_dur),
            'onset': kwargs.get('onset', self.onset),
            'merge_valley_tol': kwargs.get('merge_valley_tol', self.merge_valley_tol),
            'min_amplitude_threshold': kwargs.get('min_amplitude_threshold', self.min_amplitude_threshold),
        }
        
        # Only include lookahead if explicitly set
        lookahead = kwargs.get('lookahead', self.lookahead)
        if lookahead is not None:
            peak_kwargs['lookahead'] = lookahead
        
        # Apply peak detection algorithm
        return segment_peakdetect(envelope, times, **peak_kwargs)


# Legacy names for backward compatibility
segment_billauer = segment_peakdetect


__all__ = [
    "segment_peakdetect",
    "segment_billauer", 
    "PeakdetectSegmenter",
]
