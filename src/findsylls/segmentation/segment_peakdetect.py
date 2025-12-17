"""
Peakdetect utility function for envelope-based segmentation.

This module provides segment_peakdetect(), the library's wrapper around
Eli Billauer's peak detection algorithm (from findpeaks). This is used by
multiple envelope-based methods (theta, SBS, Hilbert, etc.) to detect
syllable nuclei (peaks) and boundaries (valleys).

This is the ONLY place in the library that imports from findpeaks.peakdetect.
All other modules should use this wrapper.

Usage:
    from findsylls.envelope.theta import theta_oscillator_envelope
    from findsylls.segmentation.segment_peakdetect import segment_peakdetect
    
    envelope, times = theta_oscillator_envelope(audio, sr)
    syllables = segment_peakdetect(envelope, times)
"""

import numpy as np
from typing import List, Tuple
from findpeaks.peakdetect import peakdetect


def segment_peakdetect(envelope: np.ndarray, times: np.ndarray, **kwargs) -> List[Tuple[float, float, float]]:
    delta = kwargs.get("delta", 0.01)
    min_syllable_dur = kwargs.get("min_syllable_dur", 0.05)
    onset = kwargs.get("onset", 0.05)
    merge_tol = kwargs.get("merge_valley_tol", 0.05)

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
        if (right - left) >= min_syllable_dur:
            syllables.append((left, best_peak, right))
    return syllables


# Legacy names for backward compatibility
segment_billauer = segment_peakdetect
segment_peaks_and_valleys = segment_peakdetect


__all__ = ["segment_peakdetect", "segment_billauer", "segment_peaks_and_valleys"]
