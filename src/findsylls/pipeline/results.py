import os, pandas as pd
from typing import List
from ..config.constants import EVAL_METHODS

def flatten_results(results: List[dict]) -> pd.DataFrame:
    flattened = []
    for res in results:
        flat = {"audio_file": res.get("audio_file"), "tg_file": res.get("tg_file"), "envelope": res.get("envelope"), "segmentation": res.get("segmentation")}
        for method in EVAL_METHODS:
            row = flat.copy(); eval_res = res.get(method, None)
            if eval_res is None: continue
            row["eval_method"] = method
            for k, v in eval_res.items():
                row[k] = v
            flattened.append(row)
    df = pd.DataFrame(flattened)
    df["file_id"] = df["audio_file"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df

def aggregate_results(results_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    summary = results_df.groupby("eval_method").aggregate({"TP": "sum", "Ins": "sum", "Del": "sum", "Sub": "sum"})
    total = summary.sum(axis=1)
    precision = summary["TP"] / (summary["TP"] + summary["Ins"] + summary["Sub"])
    recall = summary["TP"] / (summary["TP"] + summary["Del"] + summary["Sub"])
    f1 = 2 * (precision * recall) / (precision + recall)
    summary = summary.div(total, axis=0)
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
