"""Corpus-level TextGrid label attachment and discovery-label metrics.

These helpers bridge post-hoc evaluation and early-injection corpus labeling.
They accept wav/textgrid inputs as either explicit lists or glob patterns and
use the same alignment rules as the main evaluation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json

import numpy as np
import pandas as pd

from ..audio.utils import match_wavs_to_textgrids
from ..pipeline.manifests import build_label_manifest, join_corpus_manifests
from ..parsing.textgrid_parser import parse_textgrid_intervals


ManifestLike = Union[pd.DataFrame, List[Dict[str, Any]], str, Path]


def _coerce_manifest_frame(manifest: ManifestLike) -> pd.DataFrame:
    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            return pd.read_csv(path, sep=sep)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported manifest format: {path.suffix}")
    return pd.DataFrame(list(manifest))


def _textgrid_lookup(
    wav_paths: Sequence[str] | str,
    textgrid_paths: Sequence[str] | str,
    tg_suffix_to_strip: Optional[str] = None,
) -> Dict[str, Path]:
    matched_tg, matched_wav = match_wavs_to_textgrids(
        wav_paths,
        textgrid_paths,
        tg_suffix_to_strip=tg_suffix_to_strip,
    )
    return {Path(wav).stem: Path(tg) for tg, wav in zip(matched_tg, matched_wav)}


def _row_textgrid_labels(
    row: pd.Series,
    tg_path: Path,
    textgrid_tier_index: int,
    primary_label_mode: str,
    textgrid_overlap_threshold: float,
    textgrid_overlap_min_sec: float,
    start_column: str,
    peak_column: str,
    end_column: str,
) -> Dict[str, Any]:
    intervals = parse_textgrid_intervals(str(tg_path), textgrid_tier_index)
    start = float(row[start_column])
    peak = float(row[peak_column])
    end = float(row[end_column])

    attached: List[str] = []
    best_label: Optional[str] = None
    best_overlap = -1.0
    peak_label: Optional[str] = None

    for interval_start, interval_end, interval_label in intervals:
        label = (interval_label or "").strip()
        if not label:
            continue
        overlap = min(end, interval_end) - max(start, interval_start)
        if overlap <= 0:
            continue
        interval_duration = max(1e-6, float(interval_end) - float(interval_start))
        if (overlap >= textgrid_overlap_min_sec) or ((overlap / interval_duration) >= textgrid_overlap_threshold):
            attached.append(label)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
        if (interval_start <= peak <= interval_end) and peak_label is None:
            peak_label = label

    if primary_label_mode == "sequence":
        primary_label = " ".join(attached)
    elif primary_label_mode == "nucleus":
        primary_label = peak_label or best_label or ""
    elif primary_label_mode == "max-overlap":
        primary_label = best_label or peak_label or ""
    elif primary_label_mode == "first":
        primary_label = attached[0] if attached else (best_label or peak_label or "")
    else:
        raise ValueError(
            f"Unknown primary_label_mode '{primary_label_mode}'. "
            "Use 'sequence', 'nucleus', 'max-overlap', or 'first'."
        )

    return {
        "tg_labels": attached,
        "tier_labels_concat": " ".join(attached),
        "primary_label": primary_label,
        "primary_label_peak": peak_label or "",
        "primary_label_max_overlap": best_label or "",
        "textgrid_path": str(tg_path),
        "label_attached": True,
        "label_source": "textgrid",
    }


def attach_textgrid_labels_to_manifest(
    manifest: ManifestLike,
    wav_paths: Optional[Sequence[str] | str] = None,
    textgrid_paths: Optional[Sequence[str] | str] = None,
    *,
    file_manifest: Optional[ManifestLike] = None,
    audio_path_column: str = "audio_path",
    file_id_column: str = "file_id",
    start_column: str = "start",
    peak_column: str = "peak",
    end_column: str = "end",
    textgrid_tier_index: int = 0,
    primary_label_mode: str = "sequence",
    textgrid_overlap_threshold: float = 0.5,
    textgrid_overlap_min_sec: float = 0.03,
    tg_suffix_to_strip: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Attach gold labels from TextGrids to a corpus manifest.

    This is the post-hoc path: take an existing corpus manifest (for example,
    a discovery manifest with per-syllable spans) and enrich it with TextGrid
    labels. The same function can be called before writing the manifest to disk
    for the early-injection path.
    """
    df = _coerce_manifest_frame(manifest)
    if audio_path_column not in df.columns:
        if file_manifest is None:
            raise ValueError(f"Manifest must include '{audio_path_column}' or supply file_manifest")
        joined = join_corpus_manifests(df, file_manifest=file_manifest)
        df = joined
    if audio_path_column not in df.columns:
        raise ValueError(f"Manifest must include '{audio_path_column}' after joining file_manifest")
    for required in (start_column, peak_column, end_column):
        if required not in df.columns:
            raise ValueError(f"Manifest must include '{required}'")

    if textgrid_paths is None:
        raise ValueError("textgrid_paths is required to attach labels")

    source_wavs = wav_paths if wav_paths is not None else df[audio_path_column].dropna().astype(str).tolist()
    tg_lookup = _textgrid_lookup(source_wavs, textgrid_paths, tg_suffix_to_strip=tg_suffix_to_strip)

    enriched_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        audio_path = str(row_dict[audio_path_column])
        tg_path = tg_lookup.get(Path(audio_path).stem)
        if tg_path is not None and tg_path.exists():
            row_dict.update(
                _row_textgrid_labels(
                    row,
                    tg_path,
                    textgrid_tier_index=textgrid_tier_index,
                    primary_label_mode=primary_label_mode,
                    textgrid_overlap_threshold=textgrid_overlap_threshold,
                    textgrid_overlap_min_sec=textgrid_overlap_min_sec,
                    start_column=start_column,
                    peak_column=peak_column,
                    end_column=end_column,
                )
            )
        else:
            row_dict.setdefault("tg_labels", [])
            row_dict.setdefault("tier_labels_concat", "")
            row_dict.setdefault("primary_label", "")
            row_dict.setdefault("primary_label_peak", "")
            row_dict.setdefault("primary_label_max_overlap", "")
            row_dict.setdefault("textgrid_path", "")
            row_dict.setdefault("label_attached", False)
            row_dict.setdefault("label_source", "")
        enriched_rows.append(row_dict)

    enriched = pd.DataFrame(enriched_rows)
    if output_path is not None:
        build_label_manifest(enriched, output_path=output_path)
    return enriched


def compute_discovery_label_metrics(
    manifest: ManifestLike,
    *,
    cluster_column: str = "cluster_label",
    label_column: str = "primary_label",
) -> Dict[str, Any]:
    """Compute label-aware discovery metrics from an attached corpus manifest.

    Metrics are based on discovered cluster ids versus attached syllable labels.
    The majority label in each cluster is used as the cluster's pseudo-ground-truth
    for cluster-level precision/recall/F1.
    """
    df = _coerce_manifest_frame(manifest)
    if cluster_column not in df.columns:
        raise ValueError(f"Manifest must include '{cluster_column}'")
    if label_column not in df.columns:
        raise ValueError(f"Manifest must include '{label_column}'")

    df = df.copy()
    df = df[df[cluster_column].notna()]
    df[cluster_column] = df[cluster_column].astype(int)
    df[label_column] = df[label_column].fillna("").astype(str)
    df = df[df[label_column].str.len() > 0]

    total_labeled = int(len(df))
    if total_labeled == 0:
        return {
            "clusters": [],
            "labels": [],
            "cluster_purity": 0.0,
            "label_purity": 0.0,
            "label_norm_mutual_info": 0.0,
            "mutual_information": 0.0,
            "label_entropy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "total_labeled_syllables": 0,
        }

    cluster_to_labels: Dict[int, List[str]] = {}
    label_counts = df[label_column].value_counts().to_dict()
    cluster_counts = df[cluster_column].value_counts().to_dict()

    for _, row in df.iterrows():
        cluster_to_labels.setdefault(int(row[cluster_column]), []).append(str(row[label_column]))

    labels_set = sorted(label_counts.keys())
    cluster_ids = sorted(cluster_to_labels.keys())
    contingency = np.zeros((len(cluster_ids), len(labels_set)), dtype=float)
    cluster_index = {cluster: i for i, cluster in enumerate(cluster_ids)}
    label_index = {label: j for j, label in enumerate(labels_set)}

    for cluster, labels in cluster_to_labels.items():
        i = cluster_index[cluster]
        for label in labels:
            contingency[i, label_index[label]] += 1.0

    # Cluster-wise statistics
    cluster_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    macro_f1_values: List[float] = []
    weighted_f1_sum = 0.0
    weighted_support = 0

    for cluster in cluster_ids:
        row_idx = cluster_index[cluster]
        row_counts = contingency[row_idx]
        cluster_support = int(cluster_counts.get(cluster, 0))
        labeled_support = int(row_counts.sum())
        if labeled_support == 0:
            continue
        majority_idx = int(np.argmax(row_counts))
        majority_label = labels_set[majority_idx]
        majority_count = float(row_counts[majority_idx])
        precision = majority_count / labeled_support if labeled_support > 0 else 0.0
        label_total = float(label_counts.get(majority_label, 0))
        recall = majority_count / label_total if label_total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        cluster_purity = precision

        cluster_rows.append(
            {
                "cluster": cluster,
                "n_syllables": cluster_support,
                "n_labeled_syllables": labeled_support,
                "majority_label": majority_label,
                "majority_count": majority_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "purity": cluster_purity,
                "support": labeled_support,
                "label_distribution": {label: int(row_counts[label_index[label]]) for label in labels_set if row_counts[label_index[label]] > 0},
            }
        )
        macro_f1_values.append(f1)
        weighted_f1_sum += f1 * labeled_support
        weighted_support += labeled_support

    # Label-side stats
    for label in labels_set:
        col_idx = label_index[label]
        col_counts = contingency[:, col_idx]
        label_support = float(col_counts.sum())
        best_cluster_idx = int(np.argmax(col_counts)) if len(col_counts) else -1
        best_cluster = cluster_ids[best_cluster_idx] if best_cluster_idx >= 0 and len(cluster_ids) > 0 else None
        best_count = float(col_counts[best_cluster_idx]) if best_cluster_idx >= 0 and len(cluster_ids) > 0 else 0.0
        label_purity = (best_count / label_support) if label_support > 0 else 0.0
        label_rows.append(
            {
                "label": label,
                "count": label_support,
                "best_cluster": best_cluster,
                "best_cluster_count": best_count,
                "purity": label_purity,
            }
        )

    cluster_purity = float(sum(row["majority_count"] for row in cluster_rows) / total_labeled) if total_labeled else 0.0
    label_purity = float(sum(row["best_cluster_count"] for row in label_rows) / total_labeled) if total_labeled else 0.0

    # Mutual information / entropy
    Pcp = contingency / float(total_labeled)
    Pc = Pcp.sum(axis=1, keepdims=True)
    Pp = Pcp.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = Pc @ Pp
        valid = (Pcp > 0) & (denom > 0)
        ratio = np.divide(Pcp, denom, out=np.ones_like(Pcp), where=valid)
        mutual_information = float(np.sum(np.where(valid, Pcp * np.log(ratio), 0.0)))
        label_entropy = float(-np.sum(np.where(Pp > 0, Pp * np.log(Pp), 0.0)))
    label_norm_mutual_info = (mutual_information / label_entropy) if label_entropy > 0 else 0.0

    return {
        "clusters": sorted(cluster_rows, key=lambda d: d["purity"], reverse=True),
        "labels": sorted(label_rows, key=lambda d: d["purity"], reverse=True),
        "cluster_purity": cluster_purity,
        "label_purity": label_purity,
        "label_norm_mutual_info": label_norm_mutual_info,
        "mutual_information": mutual_information,
        "label_entropy": label_entropy,
        "macro_f1": float(np.mean(macro_f1_values)) if macro_f1_values else 0.0,
        "weighted_f1": float(weighted_f1_sum / weighted_support) if weighted_support > 0 else 0.0,
        "total_labeled_syllables": total_labeled,
        "n_clusters": int(len(cluster_rows)),
        "n_labels": int(len(label_rows)),
    }


__all__ = [
    "attach_textgrid_labels_to_manifest",
    "compute_discovery_label_metrics",
    "export_discovery_label_artifacts",
    "load_discovery_label_artifacts",
]


def export_discovery_label_artifacts(
    manifest: ManifestLike,
    output_base: Union[str, Path],
    *,
    cluster_column: str = "cluster_label",
    label_column: str = "primary_label",
    intrinsic_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Export corpus-level discovery artifacts split by metric category.

    This writes one file per category so downstream analysis can load only the
    piece it needs:
    - `<base>_syllables.csv`: syllable-level manifest with labels and clusters
    - `<base>_cluster_metrics.csv`: per-cluster purity/precision/recall/F1/support
    - `<base>_label_metrics.csv`: per-label purity/best-cluster/support
    - `<base>_global_metrics.json`: dataset-level summary, plus optional intrinsic metrics
    """
    df = _coerce_manifest_frame(manifest)
    metrics = compute_discovery_label_metrics(
        df,
        cluster_column=cluster_column,
        label_column=label_column,
    )

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    syllables_path = output_base.parent / f"{output_base.name}_syllables.csv"
    cluster_path = output_base.parent / f"{output_base.name}_cluster_metrics.csv"
    label_path = output_base.parent / f"{output_base.name}_label_metrics.csv"
    summary_path = output_base.parent / f"{output_base.name}_global_metrics.json"

    df.to_csv(syllables_path, index=False)
    pd.DataFrame(metrics.get("clusters", [])).to_csv(cluster_path, index=False)
    pd.DataFrame(metrics.get("labels", [])).to_csv(label_path, index=False)

    summary = {
        "cluster_purity": metrics.get("cluster_purity", 0.0),
        "label_purity": metrics.get("label_purity", 0.0),
        "label_norm_mutual_info": metrics.get("label_norm_mutual_info", 0.0),
        "mutual_information": metrics.get("mutual_information", 0.0),
        "label_entropy": metrics.get("label_entropy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "total_labeled_syllables": metrics.get("total_labeled_syllables", 0),
        "n_clusters": metrics.get("n_clusters", 0),
        "n_labels": metrics.get("n_labels", 0),
    }
    if intrinsic_metrics:
        summary["intrinsic_metrics"] = intrinsic_metrics
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return {
        "syllables_csv": str(syllables_path),
        "cluster_metrics_csv": str(cluster_path),
        "label_metrics_csv": str(label_path),
        "global_metrics_json": str(summary_path),
    }


def load_discovery_label_artifacts(output_base: Union[str, Path]) -> Dict[str, Any]:
    """Load split discovery artifact files into a single in-memory bundle."""
    output_base = Path(output_base)
    syllables_path = output_base.parent / f"{output_base.name}_syllables.csv"
    cluster_path = output_base.parent / f"{output_base.name}_cluster_metrics.csv"
    label_path = output_base.parent / f"{output_base.name}_label_metrics.csv"
    summary_path = output_base.parent / f"{output_base.name}_global_metrics.json"

    bundle: Dict[str, Any] = {
        "paths": {
            "syllables_csv": str(syllables_path),
            "cluster_metrics_csv": str(cluster_path),
            "label_metrics_csv": str(label_path),
            "global_metrics_json": str(summary_path),
        },
        "syllables": pd.DataFrame(),
        "cluster_metrics": pd.DataFrame(),
        "label_metrics": pd.DataFrame(),
        "global_metrics": {},
    }

    if syllables_path.exists():
        bundle["syllables"] = pd.read_csv(syllables_path)
    if cluster_path.exists():
        bundle["cluster_metrics"] = pd.read_csv(cluster_path)
    if label_path.exists():
        bundle["label_metrics"] = pd.read_csv(label_path)
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            bundle["global_metrics"] = json.load(handle)

    return bundle