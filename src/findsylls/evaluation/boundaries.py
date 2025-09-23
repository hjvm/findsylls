from typing import List, Tuple, Dict
from ..config.constants import DEFAULT_TOLERANCE

def spans_to_boundaries(spans: List[Tuple[float, float]], tolerance: float = 0) -> List[float]:
    if not spans:
        return []
    boundaries = [spans[0][0]]
    for start, end in spans:
        prev = boundaries[-1]
        if abs(start - prev) <= tolerance:
            boundaries[-1] = (start + prev) / 2
        else:
            boundaries.append(start)
        boundaries.append(end)
    return sorted(set(boundaries))

def evaluate_syllable_boundaries(predicted: List[Tuple[float, float]], reference: Dict[str, List[Tuple[float, float]]], tolerance: float = DEFAULT_TOLERANCE) -> Dict:
    all_refs = []
    good_spans = reference.get("intervals", [])
    bad_spans = reference.get("deleted", [])
    all_refs.extend([(b, True) for b in spans_to_boundaries(good_spans, tolerance=tolerance)])
    all_refs.extend([(b, False) for b in spans_to_boundaries(bad_spans, tolerance=tolerance)])
    pred_boundaries = spans_to_boundaries(predicted, tolerance=tolerance)
    ref_match = {}; pred_match = {}
    for i, pred in enumerate(pred_boundaries):
        closest = None; min_dist = float('inf')
        for j, (ref, _) in enumerate(all_refs):
            dist = abs(pred - ref)
            if dist <= tolerance and dist < min_dist:
                closest = j; min_dist = dist
        if closest is not None:
            pred_match[i] = closest
    for j, (ref, _) in enumerate(all_refs):
        closest = None; min_dist = float('inf')
        for i, pred in enumerate(pred_boundaries):
            dist = abs(pred - ref)
            if dist <= tolerance and dist < min_dist:
                closest = i; min_dist = dist
        if closest is not None:
            ref_match[j] = closest
    matches = []; used_preds = set(); used_refs = set()
    for i, j in pred_match.items():
        if ref_match.get(j) == i:
            ref_time, is_active = all_refs[j]
            if is_active:
                matches.append(pred_boundaries[i])
            used_preds.add(i); used_refs.add(j)
    insertions = [pred_boundaries[i] for i in range(len(pred_boundaries)) if i not in used_preds]
    deletions = [all_refs[j][0] for j in range(len(all_refs)) if j not in used_refs and all_refs[j][1] is True]
    TP = len(matches); Ins = len(insertions); Del = len(deletions)
    TER = (Ins + Del) / max(TP + Del, 1)
    return {"TP": TP, "Ins": Ins, "Del": Del, "Sub": None, "TER": TER, "matches": matches, "insertions": insertions, "deletions": deletions, "substitutions": None}
