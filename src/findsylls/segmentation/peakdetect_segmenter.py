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
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from findpeaks.peakdetect import peakdetect

from .base import EnvelopeBasedSegmenter
from ..envelope.base import EnvelopeComputer

if TYPE_CHECKING:
    from ..vad.base import BaseSAD


def segment_peakdetect(envelope: np.ndarray, times: np.ndarray, **kwargs) -> List[Tuple[float, float, float]]:
    """
    Segment envelope into syllable-like units using Billauer's peak detection algorithm.

    Args:
        envelope: Amplitude envelope (1D array)
        times: Time array corresponding to envelope samples (in seconds)
        **kwargs: Segmentation parameters:
            - delta: Minimum peak/valley height difference (default: 0.01)
            - min_syllable_dur: Minimum syllable duration in seconds (default: 0.05)
            - max_syllable_dur: Maximum syllable duration in seconds (default: None, no cap)
            - merge_valley_tol: Time tolerance for merging nearby valleys (default: 0.05)
            - amplitude_ratio_tol: Shallow-valley filter — valleys whose amplitude exceeds
                                   this fraction of the local peak maximum are merged.
                                   E.g., 0.4 removes valleys shallower than 40% of local max.
                                   (default: None, disabled)
            - min_amplitude_threshold: Minimum amplitude as fraction of max envelope (default: 0.0)
                                      Filters peaks in silent regions. E.g., 0.1 = 10% of max.
            - lookahead: Samples to look ahead (auto-computed from min_syllable_dur if None)

    Returns:
        List of (start, nucleus, end) tuples in seconds
    """
    envelope = np.asarray(envelope)
    times = np.asarray(times)

    if envelope.ndim != 1:
        raise ValueError(
            f"segment_peakdetect expects a 1-D envelope, got array with shape {envelope.shape}."
        )
    if times.ndim != 1:
        raise ValueError(
            f"segment_peakdetect expects a 1-D times array, got array with shape {times.shape}."
        )
    if envelope.shape[0] != times.shape[0]:
        raise ValueError(
            "segment_peakdetect expects envelope and times arrays to have identical length; "
            f"got {envelope.shape[0]} and {times.shape[0]}."
        )

    delta = kwargs.get("delta", 0.01)
    min_syllable_dur = kwargs.get("min_syllable_dur", 0.05)
    max_syllable_dur = kwargs.get("max_syllable_dur", None)
    add_boundary_valleys = kwargs.get("add_boundary_valleys", False)
    merge_tol = kwargs.get("merge_valley_tol", 0.05)
    amplitude_ratio_tol = kwargs.get("amplitude_ratio_tol", None)
    min_amplitude_threshold = kwargs.get("min_amplitude_threshold", 0.0)

    if 'lookahead' in kwargs:
        lookahead = kwargs['lookahead']
    else:
        lookahead_time = min_syllable_dur / 2.0
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        lookahead = max(1, int(lookahead_time / dt))

    if min_amplitude_threshold > 0:
        amplitude_cutoff = min_amplitude_threshold * np.max(envelope)
    else:
        amplitude_cutoff = -np.inf

    raw_peaks, raw_valleys = peakdetect(envelope, lookahead=lookahead, delta=delta, x_axis=times)
    peak_times = np.array([p[0] for p in raw_peaks])
    peak_vals = np.array([p[1] for p in raw_peaks])
    valleys_times = np.array([v[0] for v in raw_valleys])
    valleys_vals = np.array([v[1] for v in raw_valleys])

    if peak_times.size == 0:
        return []
    if valleys_times.size == 0 and not add_boundary_valleys:
        return []

    # Shallow-valley filter: remove valleys that don't dip below
    # amplitude_ratio_tol * local_max (local_max = adjacent peak amplitudes).
    if amplitude_ratio_tol is not None and peak_vals.size > 0:
        keep = np.ones(len(valleys_times), dtype=bool)
        for i, (vt, vv) in enumerate(zip(valleys_times, valleys_vals)):
            left_peaks = peak_vals[peak_times < vt]
            right_peaks = peak_vals[peak_times > vt]
            left_max = float(left_peaks[-1]) if left_peaks.size > 0 else 0.0
            right_max = float(right_peaks[0]) if right_peaks.size > 0 else 0.0
            local_max = max(left_max, right_max)
            if local_max > 0 and vv > amplitude_ratio_tol * local_max:
                keep[i] = False
        valleys_times = valleys_times[keep]
        valleys_vals = valleys_vals[keep]

    if valleys_times.size == 0 and not add_boundary_valleys:
        return []

    # Boundary valley insertion — before merge so that any natural valley close to
    # an endpoint is absorbed into the boundary via the merge step below.
    # Inserting at times[0] and times[-1] unconditionally lets merge_valley_tol
    # decide whether a nearby natural valley should replace the boundary position.
    # Boundary-spanning segments are exempt from max_syllable_dur (see loop below).
    if add_boundary_valleys:
        valleys_times = np.concatenate(([times[0]], valleys_times, [times[-1]]))
        valleys_vals = np.concatenate(([envelope[0]], valleys_vals, [envelope[-1]]))

    # Time-based merge: group consecutive close valleys, keep deepest.
    diffs = np.diff(valleys_times)
    break_idxs = np.nonzero(diffs > merge_tol)[0] + 1
    groups = np.split(np.arange(len(valleys_times)), break_idxs)
    merged_valleys = []
    for grp in groups:
        sub_vals = valleys_vals[grp]
        best_idx = grp[np.argmin(sub_vals)]
        merged_valleys.append(valleys_times[best_idx])
    valleys = np.array(merged_valleys)

    syllables = []
    for i in range(1, len(valleys)):
        left, right = valleys[i-1], valleys[i]
        mid_peaks = peak_times[(peak_times > left) & (peak_times < right)]
        if mid_peaks.size == 0:
            continue
        best_peak = max(mid_peaks, key=lambda tsec: envelope[np.argmin(np.abs(times - tsec))])
        peak_amplitude = envelope[np.argmin(np.abs(times - best_peak))]

        dur = right - left
        # Segments spanning the chunk endpoints are exempt from max_syllable_dur:
        # that cap targets spurious long interior spans, not intentional SAD boundaries.
        is_boundary_seg = add_boundary_valleys and (
            left == times[0] or right == times[-1]
        )
        if (dur >= min_syllable_dur
                and peak_amplitude >= amplitude_cutoff
                and (is_boundary_seg or max_syllable_dur is None or dur <= max_syllable_dur)):
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
        max_syllable_dur: Maximum syllable duration in seconds (default: None, no cap)
        merge_valley_tol: Time tolerance for merging nearby valleys (default: 0.05)
        amplitude_ratio_tol: Shallow-valley filter. Valleys whose amplitude exceeds this
                             fraction of the local peak maximum are merged (removed as
                             boundaries). E.g., 0.4 removes valleys shallower than 40% of
                             local max. (default: None, disabled)
        min_amplitude_threshold: Minimum amplitude threshold as fraction of max envelope
                                 amplitude (default: 0.0). Filters peaks in silent regions.
        sample_rate: Target sample rate (default: 16000).
        sad: Optional SAD backend for speech-region chunking (default: None).
        add_utterance_boundaries: Insert boundary valleys at region onset/offset before
                                  peak detection so the algorithm can produce segments
                                  covering the full speech region (default: True).

    Examples:
        >>> # Classical envelope
        >>> from findsylls.envelope.theta import ThetaEnvelope
        >>> segmenter = PeakdetectSegmenter(ThetaEnvelope(f=5, Q=0.5), delta=0.02)
        >>> segments = segmenter.segment(audio=audio, sr=16000)

        >>> # Or use pre-computed envelope
        >>> envelope, times = theta_envelope_function(audio, sr)
        >>> segments = segmenter.segment(envelope=envelope, times=times)
    """

    def __init__(
        self,
        envelope_computer: Optional[Union[EnvelopeComputer, callable]] = None,
        delta: float = 0.01,
        lookahead: Optional[int] = None,
        min_syllable_dur: float = 0.05,
        max_syllable_dur: Optional[float] = None,
        merge_valley_tol: float = 0.05,
        amplitude_ratio_tol: Optional[float] = None,
        min_amplitude_threshold: float = 0.0,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, sad=sad,
                         add_utterance_boundaries=add_utterance_boundaries)
        self.envelope_computer = envelope_computer
        self.delta = delta
        self.lookahead = lookahead
        self.min_syllable_dur = min_syllable_dur
        self.max_syllable_dur = max_syllable_dur
        self.merge_valley_tol = merge_valley_tol
        self.amplitude_ratio_tol = amplitude_ratio_tol
        self.min_amplitude_threshold = min_amplitude_threshold

    def _segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        if self.envelope_computer is None:
            raise ValueError("Must provide envelope_computer or use segment(envelope=..., times=...)")
        if hasattr(self.envelope_computer, 'compute'):
            envelope, times = self.envelope_computer.compute(audio, sr)
        else:
            envelope, times = self.envelope_computer(audio, sr)
        return self._segment_from_envelope(np.asarray(envelope), np.asarray(times))

    def _segment_from_envelope(
        self, envelope: np.ndarray, times: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        envelope = np.asarray(envelope)
        times = np.asarray(times)

        if envelope.ndim != 1:
            raise ValueError(
                "PeakdetectSegmenter requires a 1-D envelope. "
                f"Got shape {envelope.shape}; use an explicit envelope method/pseudo-envelope first."
            )
        if times.ndim != 1:
            raise ValueError(
                f"PeakdetectSegmenter requires a 1-D times array; got shape {times.shape}."
            )
        if envelope.shape[0] != times.shape[0]:
            raise ValueError(
                "PeakdetectSegmenter requires envelope and times arrays of equal length; "
                f"got {envelope.shape[0]} and {times.shape[0]}."
            )

        peak_kwargs = {
            'delta': self.delta,
            'min_syllable_dur': self.min_syllable_dur,
            'max_syllable_dur': self.max_syllable_dur,
            'merge_valley_tol': self.merge_valley_tol,
            'amplitude_ratio_tol': self.amplitude_ratio_tol,
            'min_amplitude_threshold': self.min_amplitude_threshold,
        }
        if self.lookahead is not None:
            peak_kwargs['lookahead'] = self.lookahead
        if self.add_utterance_boundaries:
            peak_kwargs['add_boundary_valleys'] = True

        return segment_peakdetect(envelope, times, **peak_kwargs)


__all__ = [
    "segment_peakdetect",
    "PeakdetectSegmenter",
]
