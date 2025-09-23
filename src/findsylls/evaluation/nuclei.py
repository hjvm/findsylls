from typing import List, Tuple, Dict

def evaluate_nuclei(peaks: List[float], vocalic_intervals: List[Tuple[float, float]], window: float = 0.05) -> Dict:
    matched_peaks = set(); matched_intervals = set(); extra_peaks = set()
    for i, (start, end) in enumerate(vocalic_intervals):
        peaks_in_range = [p for p in peaks if (start - window) <= p <= (end + window)]
        if len(peaks_in_range) == 0:
            continue
        elif len(peaks_in_range) == 1:
            matched_peaks.add(peaks_in_range[0]); matched_intervals.add(i)
        else:
            matched_peaks.add(peaks_in_range[0]); matched_intervals.add(i); extra_peaks.update(peaks_in_range[1:])
    TP = len(matched_intervals)
    FP = len(extra_peaks) + len([p for p in peaks if p not in matched_peaks and p not in extra_peaks])
    FN = len(vocalic_intervals) - TP
    TER = (FP + FN) / len(vocalic_intervals) if vocalic_intervals else 0.0
    unmatched_peaks = [p for p in peaks if p not in matched_peaks]
    missed_segments = [vocalic_intervals[i] for i in range(len(vocalic_intervals)) if i not in matched_intervals]
    return {"TP": TP, "Ins": FP, "Del": FN, "Sub": None, "TER": TER, "matches": sorted(matched_peaks), "insertions": sorted(unmatched_peaks), "deletions": missed_segments, "substitutions": None}
