from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

def build_reference_with_deletions(reference: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[Tuple[float, float], bool]]:
    deleted_set = set(reference.get("deleted", []))
    all_intervals = reference.get("intervals", [])
    return [((start, end), (start, end) not in deleted_set) for (start, end) in all_intervals + reference.get("deleted", [])]

def temporal_iou(pred_span: Tuple[float, float], ref_span: Tuple[float, float]) -> float:
    """
    Calculate Intersection-over-Union for temporal spans.
    
    Args:
        pred_span: Predicted (start, end) tuple
        ref_span: Reference (start, end) tuple
    
    Returns:
        IoU value between 0 and 1
    """
    intersection_start = max(pred_span[0], ref_span[0])
    intersection_end = min(pred_span[1], ref_span[1])
    intersection = max(0, intersection_end - intersection_start)
    
    union_start = min(pred_span[0], ref_span[0])
    union_end = max(pred_span[1], ref_span[1])
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0

def evaluate_syllable_spans(predicted: List[Tuple[float, float]], reference: Dict[str, List[Tuple[float, float]]], tolerance: float = 0.05) -> Dict:
    """
    Evaluate predicted syllable spans using Hungarian matching with IoU-based cost.
    
    Uses the Hungarian (Munkres) algorithm to find globally optimal assignment between
    predicted and reference spans based on temporal Intersection-over-Union (IoU).
    This avoids the order-dependent issues of greedy matching.
    
    Args:
        predicted: List of (start, end) tuples for predicted spans
        reference: Dict with 'intervals' and optionally 'deleted' keys
        tolerance: Maximum distance (in seconds) for both boundaries to be considered a match
    
    Returns:
        Dict with TP, Ins, Del, Sub counts, TER metric, and lists of matches/insertions/deletions/substitutions
    """
    all_refs = build_reference_with_deletions(reference)
    
    if len(predicted) == 0 or len(all_refs) == 0:
        # Handle edge cases
        deletions = [ref for (ref, active) in all_refs if active]
        insertions = predicted
        TP = 0; Sub = 0; Ins = len(insertions); Del = len(deletions)
        TER = (Sub + Ins + Del) / max(TP + Sub + Del, 1)
        return {"TP": TP, "Ins": Ins, "Del": Del, "Sub": Sub, "TER": TER, 
                "matches": [], "insertions": insertions, "deletions": deletions, "substitutions": []}
    
    # Build cost matrix (1 - IoU for minimization)
    n_pred = len(predicted)
    n_ref = len(all_refs)
    cost_matrix = np.ones((n_pred, n_ref))
    
    for i, pred in enumerate(predicted):
        for j, (ref, is_active) in enumerate(all_refs):
            if not is_active:
                # High cost for deleted references (should not match)
                cost_matrix[i, j] = 1.0
            else:
                # Cost is 1 - IoU (lower IoU = higher cost)
                cost_matrix[i, j] = 1 - temporal_iou(pred, ref)
    
    # Solve optimal assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Classify matched pairs based on tolerance
    matches = []
    substitutions = []
    matched_pred = set()
    matched_ref = set()
    
    for i, j in zip(row_ind, col_ind):
        pred = predicted[i]
        ref, is_active = all_refs[j]
        
        if not is_active:
            # Matched to deleted reference - treat as unmatched
            continue
        
        matched_pred.add(i)
        matched_ref.add(j)
        
        # Check if both boundaries are within tolerance
        if abs(pred[0] - ref[0]) <= tolerance and abs(pred[1] - ref[1]) <= tolerance:
            matches.append(ref)
        else:
            substitutions.append(pred)
    
    # Unmatched predictions are insertions
    insertions = [predicted[i] for i in range(n_pred) if i not in matched_pred]
    
    # Unmatched active references are deletions
    deletions = [ref for j, (ref, is_active) in enumerate(all_refs) if j not in matched_ref and is_active]
    
    TP = len(matches)
    Sub = len(substitutions)
    Ins = len(insertions)
    Del = len(deletions)
    TER = (Sub + Ins + Del) / max(TP + Sub + Del, 1)
    
    return {"TP": TP, "Ins": Ins, "Del": Del, "Sub": Sub, "TER": TER, 
            "matches": matches, "insertions": insertions, "deletions": deletions, "substitutions": substitutions}
