import numpy as np
from findpeaks.peakdetect import peakdetect
from typing import List, Tuple

def segment_peaks_and_valleys(envelope: np.ndarray, times: np.ndarray, **kwargs) -> List[Tuple[float, float, float]]:
    lookahead = kwargs.get("lookahead", 1)
    delta = kwargs.get("delta", 0.01)
    min_syllable_dur = kwargs.get("min_syllable_dur", 0.05)
    onset = kwargs.get("onset", 0.05)
    merge_tol = kwargs.get("merge_valley_tol", 0.05)
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
