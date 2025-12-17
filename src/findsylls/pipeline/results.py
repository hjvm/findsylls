import os, pandas as pd
from typing import List

def flatten_results(results: List[dict]) -> pd.DataFrame:
    """Flatten evaluation results into a tidy DataFrame.
    
    Dynamically detects all evaluation keys in the results (e.g., 'nuclei',
    'syllable_boundaries', 'word_spans', etc.) rather than relying on a
    fixed EVAL_METHODS constant.
    """
    flattened = []
    for res in results:
        # Copy metadata fields (both standard and any extras like 'dataset', 'method', etc.)
        metadata_keys = {"audio_file", "tg_file", "envelope", "segmentation", "dataset", "method", "method_type"}
        flat = {k: res.get(k) for k in metadata_keys if k in res}
        
        # Iterate over all keys in result that contain evaluation data
        # (skip metadata keys and look for dict values containing metrics)
        for key, eval_res in res.items():
            if key in metadata_keys or eval_res is None:
                continue
            if not isinstance(eval_res, dict):
                continue
            # This is an evaluation result (e.g., nuclei, syllable_boundaries, etc.)
            row = flat.copy()
            row["eval_method"] = key
            for k, v in eval_res.items():
                row[k] = v
            flattened.append(row)
    df = pd.DataFrame(flattened)
    if not df.empty and "audio_file" in df.columns:
        df["file_id"] = df["audio_file"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df

def aggregate_results(results_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    summary = results_df.groupby("eval_method").aggregate({"TP": "sum", "Ins": "sum", "Del": "sum", "Sub": "sum"})
    # Reference total (ground truth): TP + Del + Sub (excludes Insertions)
    total = summary["TP"] + summary["Del"] + summary["Sub"]
    precision = summary["TP"] / (summary["TP"] + summary["Ins"] + summary["Sub"])
    recall = summary["TP"] / total
    f1 = 2 * (precision * recall) / (precision + recall)
    # Normalize TP, Del, Sub by reference total (these sum to 1.0)
    # Ins normalized separately as it's not part of reference
    summary["TP"] = summary["TP"] / total
    summary["Del"] = summary["Del"] / total
    summary["Sub"] = summary["Sub"] / total
    summary["Ins"] = summary["Ins"] / total  # Shown as proportion of reference
    summary["TER"] = summary["Ins"] + summary["Del"] + summary["Sub"]
    summary["Total"] = total
    summary["Precision"] = precision
    summary["Recall"] = recall
    summary["F1"] = f1
    summary.reset_index(inplace=True)
    summary["dataset"] = dataset_name
    summary["envelope"] = results_df["envelope"].iloc[0] if "envelope" in results_df.columns else None
    summary["segmentation"] = results_df["segmentation"].iloc[0] if "segmentation" in results_df.columns else None
    return summary
