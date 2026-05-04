"""Corpus manifest helpers for segmentation, embedding, discovery, and label stages.

These helpers keep each pipeline stage self-contained while allowing later
joins by file_id and segment_id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


ManifestLike = Union[pd.DataFrame, List[Dict[str, Any]], str, Path]


def _coerce_frame(data: ManifestLike) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            return pd.read_csv(path, sep=sep)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported manifest format: {path.suffix}")
    return pd.DataFrame(list(data))


def build_file_manifest(
    file_rows: Sequence[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build a file-level manifest with `file_id` -> `audio_path` lookup rows."""
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(file_rows):
        row = dict(record)
        row.setdefault("file_id", index)
        if "audio_path" not in row:
            raise ValueError("Each file row must include audio_path")
        rows.append(
            {
                "file_id": int(row["file_id"]),
                "audio_path": str(row["audio_path"]),
                **{k: v for k, v in row.items() if k not in {"file_id", "audio_path"}},
            }
        )

    frame = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame


def build_segmentation_manifest(
    file_segments: Sequence[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build a normalized segmentation manifest.

    Expected input per file:
      {
        'file_id': int,
        'audio_path': optional str,
        'segments': [(start, peak, end), ...],
        'segmentation_method': optional str,
        'segmentation_kwargs': optional dict
      }

    Output rows are one syllable/token per row, keyed by `file_id` and
    `segment_id`.
    """
    rows: List[Dict[str, Any]] = []
    for file_index, record in enumerate(file_segments):
        file_id = int(record.get("file_id", file_index))
        segments = record.get("segments") or record.get("syllables") or []
        segmentation_method = record.get("segmentation_method", "")
        segmentation_kwargs = record.get("segmentation_kwargs")
        segmentation_kwargs_ref = json.dumps(segmentation_kwargs, sort_keys=True) if isinstance(segmentation_kwargs, dict) else ""

        for segment_id, segment in enumerate(segments):
            if len(segment) != 3:
                raise ValueError("Each segment must be a (start, peak, end) triplet")
            start, peak, end = segment
            row = {
                "file_id": file_id,
                "segment_id": int(segment_id),
                "start": float(start),
                "peak": float(peak),
                "end": float(end),
                "segmentation_method": segmentation_method,
                "segmentation_kwargs_ref": segmentation_kwargs_ref,
                "num_segments": int(len(segments)),
            }
            if "audio_path" in record and record["audio_path"] is not None:
                row["audio_path"] = str(record["audio_path"])
            rows.append(row)

    frame = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame


def build_discovery_manifest(
    file_segments: Sequence[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build a normalized discovery manifest from clustered syllable rows.

    Expected input per file or per segment batch:
      {
        'file_id': int,
        'segment_id': int,
        'embedding_id': optional int,
        'start': float,
        'peak': float,
        'end': float,
        'cluster_label': int,
        'embedding_path': optional str,
        'discovery_method': optional str,
        'discovery_model_path': optional str,
        'discovery_kwargs': optional dict
      }

    The returned table is one discovered syllable/token per row and is joinable
    forward into the corpus manifest by `file_id` and `segment_id`.
    """
    rows: List[Dict[str, Any]] = []
    for file_index, record in enumerate(file_segments):
        file_id = int(record.get("file_id", file_index))
        segment_id = int(record.get("segment_id", 0))
        discovery_kwargs = record.get("discovery_kwargs")
        discovery_kwargs_ref = json.dumps(discovery_kwargs, sort_keys=True) if isinstance(discovery_kwargs, dict) else ""

        row = {
            "file_id": file_id,
            "segment_id": segment_id,
            "embedding_id": int(record.get("embedding_id", segment_id)),
            "start": float(record["start"]),
            "peak": float(record["peak"]),
            "end": float(record["end"]),
            "cluster_label": int(record["cluster_label"]),
            "embedding_path": str(record.get("embedding_path", "")),
            "discovery_method": record.get("discovery_method", ""),
            "discovery_model_path": str(record.get("discovery_model_path", "")),
            "discovery_kwargs_ref": discovery_kwargs_ref,
        }
        if "audio_path" in record and record["audio_path"] is not None:
            row["audio_path"] = str(record["audio_path"])
        if "cluster_score" in record and record["cluster_score"] is not None:
            row["cluster_score"] = float(record["cluster_score"])
        rows.append(row)

    frame = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame


def build_label_manifest(
    label_rows: ManifestLike,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build a normalized label manifest from attached corpus labels.

    This accepts the output of TextGrid attachment and preserves the standard
    label columns needed for downstream joins and evaluation.
    """
    frame = _coerce_frame(label_rows)
    normalized = frame.copy()

    for column in normalized.columns:
        normalized[column] = normalized[column].apply(
            lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) else json.dumps(value) if isinstance(value, (list, tuple)) else value
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(output_path, index=False)
    return normalized


def load_manifest(manifest: ManifestLike) -> pd.DataFrame:
    """Load a manifest from a DataFrame, list of dicts, or CSV/Parquet file."""
    return _coerce_frame(manifest)


def join_corpus_manifests(
    segmentation_manifest: ManifestLike,
    *,
    file_manifest: Optional[ManifestLike] = None,
    embedding_manifest: Optional[ManifestLike] = None,
    discovery_manifest: Optional[ManifestLike] = None,
    label_manifest: Optional[ManifestLike] = None,
) -> pd.DataFrame:
    """Join normalized corpus artifacts into one analysis-ready table.

    Joins are performed in the following order when inputs are provided:
    segmentation -> file lookup -> embedding -> discovery -> labels
    """
    frame = _coerce_frame(segmentation_manifest)

    if file_manifest is not None:
        file_df = _coerce_frame(file_manifest)
        if "file_id" not in file_df.columns or "audio_path" not in file_df.columns:
            raise ValueError("file_manifest must include file_id and audio_path")
        frame = frame.merge(file_df[["file_id", "audio_path"]], on="file_id", how="left", suffixes=("", "_file"))
        if "audio_path_file" in frame.columns:
            frame["audio_path"] = frame["audio_path"].fillna(frame["audio_path_file"]) if "audio_path" in frame.columns else frame["audio_path_file"]
            frame = frame.drop(columns=["audio_path_file"])

    def _merge_on_file_id(left: pd.DataFrame, right_like: ManifestLike, suffix: str) -> pd.DataFrame:
        right = _coerce_frame(right_like)
        if "file_id" not in right.columns:
            raise ValueError(f"{suffix} manifest must include file_id")
        return left.merge(right, on="file_id", how="left", suffixes=("", f"_{suffix}"))

    if embedding_manifest is not None:
        frame = _merge_on_file_id(frame, embedding_manifest, "embedding")

    if discovery_manifest is not None:
        disc = _coerce_frame(discovery_manifest)
        if "file_id" in disc.columns and "segment_id" in disc.columns:
            frame = frame.merge(disc, on=["file_id", "segment_id"], how="left", suffixes=("", "_discovery"))
        elif "file_id" in disc.columns:
            frame = frame.merge(disc, on="file_id", how="left", suffixes=("", "_discovery"))
        else:
            raise ValueError("discovery_manifest must include file_id or file_id+segment_id")

    if label_manifest is not None:
        labels = _coerce_frame(label_manifest)
        if "file_id" in labels.columns and "segment_id" in labels.columns:
            frame = frame.merge(labels, on=["file_id", "segment_id"], how="left", suffixes=("", "_label"))
        elif "audio_path" in labels.columns and "start" in labels.columns and "end" in labels.columns:
            # Allow direct join on spans when a post-hoc label manifest is denormalized.
            join_cols = ["audio_path", "start", "peak", "end"] if "peak" in labels.columns and "peak" in frame.columns else ["audio_path", "start", "end"]
            frame = frame.merge(labels, on=join_cols, how="left", suffixes=("", "_label"))
        else:
            raise ValueError("label_manifest must include file_id+segment_id or audio_path+span columns")

    return frame


__all__ = [
    "build_discovery_manifest",
    "build_file_manifest",
    "build_segmentation_manifest",
    "build_label_manifest",
    "join_corpus_manifests",
    "load_manifest",
]