from typing import List, Tuple, Dict
from scipy.spatial.distance import euclidean

def build_reference_with_deletions(reference: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[Tuple[float, float], bool]]:
    deleted_set = set(reference.get("deleted", []))
    all_intervals = reference.get("intervals", [])
    return [((start, end), (start, end) not in deleted_set) for (start, end) in all_intervals + reference.get("deleted", [])]

def evaluate_syllable_spans(predicted: List[Tuple[float, float]], reference: Dict[str, List[Tuple[float, float]]], tolerance: float = 0.05) -> Dict:
    all_refs = build_reference_with_deletions(reference)
    ref_used = [False] * len(all_refs)
    pred_used = [False] * len(predicted)
    matches = []; substitutions = []
    for i, pred in enumerate(predicted):
        best_idx = None; best_dist = float('inf')
        for j, (ref, _) in enumerate(all_refs):
            if ref_used[j]:
                continue
            dist = euclidean(pred, ref)
            if dist < best_dist:
                best_dist = dist; best_idx = j
        if best_idx is not None:
            ref_used[best_idx] = True; pred_used[i] = True
            ref, is_active = all_refs[best_idx]
            if not is_active:
                continue
            if abs(pred[0] - ref[0]) <= tolerance and abs(pred[1] - ref[1]) <= tolerance:
                matches.append(ref)
            else:
                substitutions.append(pred)
    deletions = [ref for (ref, active), used in zip(all_refs, ref_used) if not used and active]
    insertions = [pred for pred, used in zip(predicted, pred_used) if not used]
    TP = len(matches); Sub = len(substitutions); Ins = len(insertions); Del = len(deletions)
    TER = (Sub + Ins + Del) / max(TP + Sub + Del, 1)
    return {"TP": TP, "Ins": Ins, "Del": Del, "Sub": Sub, "TER": TER, "matches": matches, "insertions": insertions, "deletions": deletions, "substitutions": substitutions}
