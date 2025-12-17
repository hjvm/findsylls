"""High‑level evaluation orchestration.

This module provides two entry points:

evaluate_syllable_segmentation  (legacy style) – works with a combined
    syllable segmentation returning peaks + (start, end) spans.
evaluate_segmentation (current pipeline) – evaluates nuclei and boundaries/spans
    against any specified tiers. All tiers are specified via the `tiers` dict.

Notes on data structures:
    * extract_syllable_intervals returns a dict: {"intervals": [...], "deleted": [...]}.
        A plain truthiness check on that dict is always True (because it always
        has the two keys) even when both lists are empty. We therefore must
        explicitly examine the length of the "intervals" list (and optionally
        deleted) to decide whether there is usable reference data. Previously the
        code used `if not reference_syllables:` which incorrectly treated empty
        reference content as present, leading the boundary/span evaluators to run
        on empty data structures. We now guard with a helper `_is_empty_ref`.
    * A tier index of None means "skip this evaluation".
    * A tier index of -1 for syllables triggers (future) synthetic syllable
        generation (currently returns an empty list in the parser stub, so the
        evaluation will be skipped gracefully).
    * The 'phone' tier is used for nuclei evaluation (vocalic intervals).

Tier indexing is zero‑based; given a TextGrid with tiers [words, syllables, phones]
you should pass: tiers={'phone': 2, 'syllable': 1, 'word': 0}.
"""

from typing import List, Tuple, Dict, Union, Optional
from .boundaries import evaluate_syllable_boundaries
from .nuclei import evaluate_nuclei
from .spans import evaluate_syllable_spans
from ..parsing.textgrid_parser import extract_vocalic_intervals, extract_syllable_intervals, generate_syllable_intervals


def _is_empty_ref(ref: Optional[Dict]) -> bool:
        """Return True if a reference dict is None or contains no kept intervals.

        The reference structure produced by `extract_syllable_intervals` always has
        keys `intervals` and `deleted`. We consider it empty if both lists are
        empty (no active intervals to score) – in that case downstream boundary
        and span evaluation should be skipped (return None) so the flattening code
        does not emit misleading zero rows.
        """
        if ref is None:
                return True
        if not isinstance(ref, dict):  # Defensive; unexpected type
                return False
        intervals = ref.get("intervals", [])
        deleted = ref.get("deleted", [])
        return len(intervals) == 0 and len(deleted) == 0

def evaluate_syllable_segmentation(
    peaks: List[float],
    predicted_syllables: List[Tuple[float, float]],
    textgrid_path: str,
    phone_tier: Union[str, int],
    syllable_tier: Optional[Union[str, int]] = None,
    tolerance: float = 0.05,
) -> Dict:
    """Legacy convenience wrapper retained for backwards compatibility.

    Parameters
    ----------
    peaks : list of float
        Predicted nucleus (peak) times in seconds.
    predicted_syllables : list of (start, end)
        Predicted syllable spans.
    textgrid_path : str
        Path to TextGrid file.
    phone_tier : int
        Zero‑based index of the phone tier (for vocalic interval extraction).
    syllable_tier : int | None
        Zero‑based syllable tier index, -1 to invoke synthetic generation, or
        None to skip syllable boundary/span evaluation.
    tolerance : float
        Boundary matching tolerance in seconds.
    """
    vocalic_intervals = extract_vocalic_intervals(textgrid_path, phone_tier)
    nuclei_eval = evaluate_nuclei(peaks, vocalic_intervals, window=tolerance)
    if syllable_tier is None:
        return {"nuclei": nuclei_eval, "boundaries": None, "spans": None}
    if syllable_tier == -1:
        reference_syllables = generate_syllable_intervals(textgrid_path, phone_tier)
    else:
        reference_syllables = extract_syllable_intervals(textgrid_path, syllable_tier)
    if _is_empty_ref(reference_syllables):
        boundary_eval = None
        span_eval = None
    else:
        boundary_eval = evaluate_syllable_boundaries(predicted_syllables, reference_syllables, tolerance=tolerance)
        span_eval = evaluate_syllable_spans(predicted_syllables, reference_syllables, tolerance=tolerance)
    return {"nuclei": nuclei_eval, "boundaries": boundary_eval, "spans": span_eval}

def evaluate_segmentation(
    peaks: List[float],
    spans: List[Tuple[float, float]],
    textgrid_path: str,
    phone_tier: Optional[int] = None,
    syllable_tier: Optional[int] = None,
    word_tier: Optional[int] = None,
    tiers: Optional[Dict[str, int]] = None,
    tolerance: float = 0.05,
) -> Dict:
    """Evaluate a predicted segmentation against TextGrid references.

    Parameters
    ----------
    peaks : list of float
        Predicted nucleus times (for nuclei evaluation).
    spans : list of (start, end)
        Predicted segment spans (for boundary/span evaluation).
    textgrid_path : str
        Path to TextGrid file.
    phone_tier : int, optional
        Zero-based phone tier index (legacy parameter for backward compatibility).
        Prefer using tiers={'phone': index} instead.
    syllable_tier : int, optional
        Zero-based syllable tier index (legacy parameter for backward compatibility).
        Prefer using tiers={'syllable': index} instead.
    word_tier : int, optional
        Zero-based word tier index (legacy parameter for backward compatibility).
        Prefer using tiers={'word': index} instead.
    tiers : dict, optional
        Mapping of tier names to indices, e.g., {'phone': 2, 'syllable': 1, 'word': 0}.
        This is the preferred way to specify tiers. The 'phone' tier is used for
        nuclei evaluation. All other tiers generate '{name}_boundaries' and
        '{name}_spans' evaluation keys.
    tolerance : float
        Boundary matching tolerance in seconds.

    Returns
    -------
    dict
        Keys include 'nuclei' (if phone tier specified) and dynamically generated
        keys like '{tier_name}_boundaries' and '{tier_name}_spans' for each tier.
    """
    result: Dict[str, Optional[Dict]] = {}

    # Collect all tiers (merging legacy params with new tiers dict)
    all_tiers = {}
    if phone_tier is not None:
        all_tiers['phone'] = phone_tier
    if syllable_tier is not None:
        all_tiers['syllable'] = syllable_tier
    if word_tier is not None:
        all_tiers['word'] = word_tier
    if tiers is not None:
        all_tiers.update(tiers)

    # Extract phone tier for nuclei evaluation and synthetic syllable generation
    phone_tier_index = all_tiers.get('phone')

    # Nuclei evaluation (requires phone tier)
    if phone_tier_index is None:
        nuclei_eval = None
    else:
        vocalic_intervals = extract_vocalic_intervals(textgrid_path, phone_tier_index)
        nuclei_eval = evaluate_nuclei(peaks, vocalic_intervals, window=tolerance)
    result["nuclei"] = nuclei_eval

    # Evaluate boundaries and spans for each specified tier
    for tier_name, tier_index in all_tiers.items():
        if tier_name == 'phone':
            # Phone tier is only used for nuclei, skip boundary/span evaluation
            continue
            
        if tier_index == -1 and tier_name == 'syllable':
            # Special case: generate synthetic syllables (requires phone tier)
            if phone_tier_index is None:
                reference_intervals = None
            else:
                reference_intervals = generate_syllable_intervals(textgrid_path, phone_tier_index)
        else:
            reference_intervals = extract_syllable_intervals(textgrid_path, tier_index)
        
        if _is_empty_ref(reference_intervals):
            boundary_eval = None
            span_eval = None
        else:
            boundary_eval = evaluate_syllable_boundaries(spans, reference_intervals, tolerance=tolerance)
            span_eval = evaluate_syllable_spans(spans, reference_intervals, tolerance=tolerance)
        
        result[f"{tier_name}_boundaries"] = boundary_eval
        result[f"{tier_name}_spans"] = span_eval

    return result
