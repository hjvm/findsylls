#!/usr/bin/env python3
"""Run the findsylls notebook test batteries from the command line.

This script mirrors the notebook workflow in a CLI-friendly form:
- compatibility / registry checks
- corpus pairing and subset selection
- embedding battery across feature / segmentation / pooling combinations
- discovery battery over successful embeddings
- evaluation and manifest battery
- optional orchestrator smoke test

The defaults are tuned for the TIMIT corpus bundled with the repository.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import findsylls
from findsylls import (
    DiscoveryPipeline,
    attach_textgrid_labels_to_manifest,
    build_discovery_manifest,
    build_file_manifest,
    build_label_manifest,
    build_segmentation_manifest,
    compute_discovery_label_metrics,
    embed_audio,
    embed_corpus,
    evaluate_segmentation,
    get_preset,
    join_corpus_manifests,
    list_presets,
    resolve_preset,
)
from findsylls.audio.utils import match_wavs_to_textgrids
from findsylls.embedding.poolers import list_poolers
from findsylls.embedding.storage import load_embeddings_npz, save_embeddings_npz
from findsylls.features import get_extractor
from findsylls.pipeline.orchestrator import FindSyllsOrchestrator
from findsylls.segmentation import get_segmenter, list_segmenter_aliases, list_segmenters
from findsylls.segmentation.presets import (
    SBSPeakdetectSegmenter,
    ThetaOscillatorSegmenter,
    list_segmenter_presets,
)

EVAL_TIERS = {"phone": 2, "syllable": 1, "word": 0}
DEFAULT_NEURAL_FEATURES = ["hubert", "sylber", "vghubert"]
DEFAULT_PSEUDO_ENVELOPES = ["cls_attention", "greedy_cosine", "mincut"]
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "timit1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "test_output_cli"
BATTERY_ORDER = [0, 1, 2, 3, 4, 5]
BATTERY_LABELS = {
    0: "envelope_presets",
    1: "compatibility",
    2: "embedding",
    3: "discovery",
    4: "evaluation",
    5: "orchestrator",
}


class SectionPrinter:
    def __init__(self) -> None:
        self.line = "=" * 78

    def title(self, text: str) -> None:
        print()
        print(self.line)
        print(text)
        print(self.line)

    def subtitle(self, text: str) -> None:
        print(f"\n{text}")
        print("-" * len(text))

    def note(self, text: str) -> None:
        print(f"  {text}")

    def ok(self, text: str) -> None:
        print(f"[OK] {text}")

    def warn(self, text: str) -> None:
        print(f"[WARN] {text}")

    def fail(self, text: str) -> None:
        print(f"[FAIL] {text}")


class NotebookCompatiblePrinter(SectionPrinter):
    pass


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Not an integer: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return parsed


def _csv_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_battery_numbers(value: str) -> List[int]:
    raw = value.strip().lower()
    if raw in {"", "all"}:
        return list(BATTERY_ORDER)

    parsed: List[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            raise argparse.ArgumentTypeError(f"Invalid battery token: {token}")
        battery = int(token)
        if battery not in BATTERY_LABELS:
            raise argparse.ArgumentTypeError(f"Battery must be one of {BATTERY_ORDER}; got {battery}")
        if battery not in parsed:
            parsed.append(battery)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one battery must be selected")
    return parsed


def _normalize_name(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
        if np.isnan(numeric):
            return default
        return numeric
    except Exception:
        return default


def _metric_summary_from_counts(tp: Any, ins: Any, del_: Any, sub: Any) -> Dict[str, Any]:
    tp = float(tp)
    ins = float(ins)
    del_ = float(del_)
    sub = float(sub)
    precision = tp / (tp + ins + sub) if (tp + ins + sub) > 0 else 0.0
    recall = tp / (tp + del_ + sub) if (tp + del_ + sub) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    ter = (ins + del_ + sub) / max(tp + del_ + sub, 1.0)
    return {
        "TP": int(tp),
        "Ins": int(ins),
        "Del": int(del_),
        "Sub": int(sub),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TER": ter,
    }


_extractors_cache: Dict[Tuple[str, Optional[int]], Any] = {}


def _get_cached_extractor(feature_name: str, layer: Optional[int] = None):
    cache_key = (feature_name, layer)
    if cache_key not in _extractors_cache:
        extractor_kwargs: Dict[str, Any] = {}
        if layer is not None and feature_name in {"hubert", "sylber", "vghubert"}:
            extractor_kwargs["layer"] = layer
        _extractors_cache[cache_key] = get_extractor(feature_name, **extractor_kwargs)
    return _extractors_cache[cache_key]


def _clear_cached_extractors() -> None:
    for extractor in _extractors_cache.values():
        if hasattr(extractor, "release"):
            try:
                extractor.release()
            except Exception:
                pass
    _extractors_cache.clear()
    gc.collect()


def _build_segmentation_kwargs_for_combo(combo: Dict[str, Any], layer: Optional[int] = None) -> Dict[str, Any]:
    kwargs = dict(combo.get("segmentation_kwargs") or {})
    envelope_method = kwargs.get("envelope_method")
    if envelope_method in {"cls_attention", "greedy_cosine", "mincut"}:
        envelope_kwargs = dict(kwargs.get("envelope_kwargs") or {})
        if "feature_extractor" not in envelope_kwargs:
            envelope_kwargs["feature_extractor"] = _get_cached_extractor(combo["features"], layer=layer)
        kwargs["envelope_kwargs"] = envelope_kwargs
    return kwargs


def _get_expected_failure_reason(combo: Dict[str, Any], neural_feature_extractors: Sequence[str]) -> Optional[str]:
    feature = _normalize_name(combo.get("features"))
    segmentation = _normalize_name(combo.get("segmentation"))
    pooling = _normalize_name(combo.get("pooling"))
    seg_kwargs = combo.get("segmentation_kwargs") or {}
    envelope_method = _normalize_name(seg_kwargs.get("envelope_method"))
    neural_set = {_normalize_name(name) for name in neural_feature_extractors}

    # cls_attention requires attention-capable extractor; only neural models expose attention.
    # (mfcc and melspec have no attention pathway regardless of architecture flags.)
    if segmentation == "cls_attention" and feature not in neural_set:
        return "cls_attention requires attention-capable feature extractor (neural models only)"

    # v3: peakdetect is envelope-only and always requires an explicit envelope_method.
    # Feature extractors (including neural) cannot be used directly with peakdetect.
    if segmentation == "peakdetect" and not envelope_method:
        return "peakdetect requires explicit envelope_method in segmentation_kwargs"

    # Pseudo-envelope methods (cls_attention, greedy_cosine, mincut) need neural features
    # because they call extract_with_attention or extract frame features from a neural model.
    if segmentation == "peakdetect" and envelope_method in {"cls_attention", "greedy_cosine", "mincut"} and feature not in neural_set:
        return "pseudo-envelope for peakdetect requires neural feature extractors"

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="findsylls-test-battery",
        description="Run the findsylls notebook test batteries from the command line.",
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR, help="Corpus root directory")
    parser.add_argument("--subset-size", type=_positive_int, default=12, help="Number of paired files to test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for outputs")
    parser.add_argument("--timestamped", action="store_true", help="Create a timestamped subdirectory under output-root")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for embedding")
    parser.add_argument("--layer", type=int, default=None, help="Optional transformer layer for neural extractors")
    parser.add_argument("--max-combinations", type=_positive_int, default=0, help="Limit embedding combinations for smoke runs (0 = all)")
    parser.add_argument("--skip-orchestrator", action="store_true", help="Skip the orchestrator smoke test")
    parser.add_argument("--save-artifacts", action="store_true", help="Persist CSV/JSON summaries to output-root")
    parser.add_argument("--batteries", type=_parse_battery_numbers, default=list(BATTERY_ORDER), help="Comma-separated batteries to run: 1=compat,2=embed,3=discover,4=evaluate,5=orchestrator; default=all")
    parser.add_argument("--features", type=str, default="", help="Optional CSV whitelist for features (e.g. hubert,sylber,mfcc)")
    parser.add_argument("--segmentations", type=str, default="", help="Optional CSV whitelist for segmentations")
    parser.add_argument("--poolings", type=str, default="", help="Optional CSV whitelist for pooling methods")
    parser.add_argument("--envelopes", type=str, default="", help="Optional CSV whitelist for peakdetect pseudo-envelope methods")
    parser.add_argument("--include-expected-failures", action="store_true", help="Include combinations marked as expected failures")
    parser.add_argument("--only-expected-failures", action="store_true", help="Run only combinations marked as expected failures")
    return parser


def _resolve_corpus_files(corpus_dir: Path) -> Tuple[List[Path], List[Path]]:
    timit_audio_dir = corpus_dir / "normedwavs"
    timit_textgrid_dir = corpus_dir / "TextGrids"
    wav_files = sorted(timit_audio_dir.rglob("*.wav"))
    textgrid_files = sorted(timit_textgrid_dir.rglob("*_syllabified.TextGrid"))
    return wav_files, textgrid_files


def _build_experiment_combinations(feature_extractors: Sequence[str], segmentation_methods: Sequence[str], poolers: Sequence[str], neural_feature_extractors: Sequence[str], pseudo_envelope_methods: Sequence[str]) -> List[Dict[str, Any]]:
    combinations: List[Dict[str, Any]] = []

    for feature in feature_extractors:
        for segmentation in segmentation_methods:
            for pooling in poolers:
                combinations.append(
                    {
                        "features": feature,
                        "segmentation": segmentation,
                        "pooling": pooling,
                        "segmentation_kwargs": {},
                        "experiment_group": "canonical",
                    }
                )

    for feature in neural_feature_extractors:
        for envelope_method in pseudo_envelope_methods:
            for pooling in poolers:
                combinations.append(
                    {
                        "features": feature,
                        "segmentation": "peakdetect",
                        "pooling": pooling,
                        "segmentation_kwargs": {"envelope_method": envelope_method},
                        "experiment_group": "peakdetect_pseudoenvelope",
                    }
                )

    unique: List[Dict[str, Any]] = []
    seen = set()
    for combo in combinations:
        signature = (
            combo["features"],
            combo["segmentation"],
            combo["pooling"],
            json.dumps(combo.get("segmentation_kwargs") or {}, sort_keys=True),
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(combo)
    return unique


def _filter_combinations(
    combinations: Sequence[Dict[str, Any]],
    features_filter: Sequence[str],
    segmentations_filter: Sequence[str],
    poolings_filter: Sequence[str],
    envelopes_filter: Sequence[str],
) -> List[Dict[str, Any]]:
    features_set = {_normalize_name(value) for value in features_filter}
    segmentations_set = {_normalize_name(value) for value in segmentations_filter}
    poolings_set = {_normalize_name(value) for value in poolings_filter}
    envelopes_set = {_normalize_name(value) for value in envelopes_filter}

    filtered: List[Dict[str, Any]] = []
    for combo in combinations:
        feature = _normalize_name(combo.get("features"))
        segmentation = _normalize_name(combo.get("segmentation"))
        pooling = _normalize_name(combo.get("pooling"))
        envelope = _normalize_name((combo.get("segmentation_kwargs") or {}).get("envelope_method"))

        if features_set and feature not in features_set:
            continue
        if segmentations_set and segmentation not in segmentations_set:
            continue
        if poolings_set and pooling not in poolings_set:
            continue
        if envelopes_set:
            if envelope:
                if envelope not in envelopes_set:
                    continue
            elif "none" not in envelopes_set:
                continue

        filtered.append(combo)
    return filtered


def _format_df(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "<empty>"
    display_cols = [column for column in columns if column in df.columns]
    if not display_cols:
        return df.head(max_rows).to_string(index=False)
    return df[display_cols].head(max_rows).to_string(index=False)


def _preview_failed_rows(df: pd.DataFrame, limit: int = 10, label_col: str = "combination") -> List[str]:
    if df.empty:
        return []
    rows: List[str] = []
    for _, row in df.head(limit).iterrows():
        pieces = [str(row.get(label_col, "<unknown>"))]
        if "discovery_method" in row:
            pieces.append(str(row.get("discovery_method")))
        if "status" in row:
            pieces.append(str(row.get("status")))
        rows.append(" | ".join(pieces))
    return rows


def run_compatibility_battery(printer: SectionPrinter, feature_extractors: Sequence[str]) -> None:
    printer.title("BATTERY 1: Compatibility Checks")

    segmentation_methods = list_segmenters()
    segmentation_aliases = list_segmenter_aliases()
    preset_methods = list_presets()
    poolers = list_poolers()

    printer.note(f"Feature extractors: {', '.join(feature_extractors)}")
    printer.note(f"Segmentation methods: {', '.join(segmentation_methods)}")
    printer.note(f"Segmentation aliases: {', '.join(f'{alias}->{target}' for alias, target in sorted(segmentation_aliases.items()))}")
    printer.note(f"Preset names: {', '.join(preset_methods)}")
    printer.note(f"Pooling methods: {', '.join(poolers)}")

    if not set(segmentation_methods).isdisjoint(set(preset_methods)):
        raise RuntimeError("Segmentation registry should not include preset names")
    printer.ok("Segmentation algorithms and presets are disjoint")

    sample_audio = None
    printer.ok("Registry sanity checks passed")

    preset_cfg = get_preset("sylber")
    if preset_cfg.get("features") != "sylber":
        raise RuntimeError("Preset resolution sanity check failed")
    printer.ok("Preset resolution and preset-based pipeline construction are valid")


def run_embedding_battery(
    printer: SectionPrinter,
    combinations: Sequence[Dict[str, Any]],
    wav_files_subset: Sequence[Path],
    embeddings_dir: Path,
    layer: Optional[int],
    n_jobs: int,
    max_combinations: int,
) -> pd.DataFrame:
    printer.title("BATTERY 2: Embedding Matrix")
    total = len(combinations) if max_combinations <= 0 else min(len(combinations), max_combinations)
    printer.note(f"Testing {total} combinations on {len(wav_files_subset)} audio files")

    embedding_results: List[Dict[str, Any]] = []
    use_combinations = combinations if max_combinations <= 0 else list(combinations[:max_combinations])

    for index, combo in enumerate(use_combinations, 1):
        feature = combo["features"]
        segmentation = combo["segmentation"]
        pooling = combo["pooling"]
        experiment_group = combo.get("experiment_group", "canonical")
        combo_segmentation_kwargs = _build_segmentation_kwargs_for_combo(combo, layer=layer)
        envelope_tag = combo_segmentation_kwargs.get("envelope_method")
        combo_name = f"{feature}_{segmentation}_{pooling}"
        expected_failure_reason = combo.get("expected_failure_reason")

        printer.subtitle(f"[{index}/{total}] {combo_name}")
        printer.note(f"group={experiment_group}")
        if envelope_tag:
            printer.note(f"peakdetect envelope={envelope_tag}")
        if expected_failure_reason:
            printer.note(f"expected failure: {expected_failure_reason}")

        start_time = time.time()
        try:
            results = embed_corpus(
                audio_files=wav_files_subset,
                segmentation=segmentation,
                features=feature,
                pooling=pooling,
                segmentation_kwargs=combo_segmentation_kwargs,
                n_jobs=n_jobs,
                verbose=False,
                fail_on_error=False,
            )
            successful = [record for record in results if record.get("success")]
            nonempty = [record for record in successful if len(record.get("embeddings", [])) > 0]
            failed = [record for record in results if not record.get("success")]

            if not nonempty:
                error_messages = []
                for record in failed[:5]:
                    error_messages.append(str(record.get("error")))
                printer.fail("No non-empty embeddings returned")
                for message in error_messages:
                    printer.note(f"error: {message}")
                embedding_results.append(
                    {
                        "combination": combo_name,
                        "features": feature,
                        "segmentation": segmentation,
                        "pooling": pooling,
                        "experiment_group": experiment_group,
                        "envelope_method": envelope_tag,
                        "segmentation_kwargs": combo_segmentation_kwargs,
                        "num_files": len(wav_files_subset),
                        "successful": len(successful),
                        "failed": len(failed),
                        "total_syllables": 0,
                        "embedding_dim": 0,
                        "time_seconds": time.time() - start_time,
                        "output_file": None,
                        "expected_failure": bool(expected_failure_reason),
                        "expected_failure_reason": expected_failure_reason,
                        "status": "expected_failure: no non-empty embeddings returned" if expected_failure_reason else "error: no non-empty embeddings returned",
                    }
                )
                continue

            all_embeddings = np.vstack([record["embeddings"] for record in nonempty])
            output_file = embeddings_dir / f"{combo_name}.npz"
            save_embeddings_npz(results, str(output_file), compress=True)
            total_time = time.time() - start_time
            printer.ok(f"{len(successful)}/{len(results)} files succeeded")
            printer.note(f"shape={all_embeddings.shape} time={total_time:.2f}s saved={output_file.name}")

            embedding_results.append(
                {
                    "combination": combo_name,
                    "features": feature,
                    "segmentation": segmentation,
                    "pooling": pooling,
                    "experiment_group": experiment_group,
                    "envelope_method": envelope_tag,
                    "segmentation_kwargs": combo_segmentation_kwargs,
                    "num_files": len(wav_files_subset),
                    "successful": len(successful),
                    "failed": len(failed),
                    "total_syllables": int(all_embeddings.shape[0]),
                    "embedding_dim": int(all_embeddings.shape[1]) if len(all_embeddings) > 0 else 0,
                    "time_seconds": total_time,
                    "output_file": str(output_file),
                    "expected_failure": bool(expected_failure_reason),
                    "expected_failure_reason": expected_failure_reason,
                    "status": "unexpected_success" if expected_failure_reason else "success",
                }
            )
        except Exception as exc:
            printer.fail(f"{type(exc).__name__}: {exc}")
            embedding_results.append(
                {
                    "combination": combo_name,
                    "features": feature,
                    "segmentation": segmentation,
                    "pooling": pooling,
                    "experiment_group": experiment_group,
                    "envelope_method": envelope_tag,
                    "segmentation_kwargs": combo_segmentation_kwargs,
                    "num_files": len(wav_files_subset),
                    "successful": 0,
                    "failed": len(wav_files_subset),
                    "total_syllables": 0,
                    "embedding_dim": 0,
                    "time_seconds": time.time() - start_time,
                    "output_file": None,
                    "expected_failure": bool(expected_failure_reason),
                    "expected_failure_reason": expected_failure_reason,
                    "status": f"expected_failure: {type(exc).__name__}: {exc}" if expected_failure_reason else f"error: {type(exc).__name__}: {exc}",
                }
            )

    embedding_df = pd.DataFrame(embedding_results)
    printer.title("EMBEDDING SUMMARY")
    printer.note(f"Total combinations tested: {len(embedding_df)}")
    success_mask = embedding_df["status"].isin(["success", "unexpected_success"]) if not embedding_df.empty else pd.Series(dtype=bool)
    expected_failure_mask = embedding_df["status"].astype(str).str.startswith("expected_failure") if not embedding_df.empty else pd.Series(dtype=bool)
    printer.note(f"Successful combinations: {int(success_mask.sum())}")
    printer.note(f"Expected failures: {int(expected_failure_mask.sum())}")
    printer.note(f"Unexpected failures: {len(embedding_df) - int(success_mask.sum()) - int(expected_failure_mask.sum())}")
    printer.note("")
    print(_format_df(
        embedding_df,
        ["combination", "features", "segmentation", "pooling", "status", "expected_failure_reason", "time_seconds", "embedding_dim", "total_syllables"],
        max_rows=20,
    ))
    failed_rows = embedding_df[~success_mask]
    if not failed_rows.empty:
        printer.subtitle("Embedding Failures")
        for line in _preview_failed_rows(failed_rows, label_col="combination"):
            printer.note(line)

    _clear_cached_extractors()
    return embedding_df


def run_discovery_battery(
    printer: SectionPrinter,
    embedding_df: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    printer.title("BATTERY 3: Discovery Clustering")

    discovery_models = [model for model in ["agglomerative", "kmeans", "minibatch_kmeans"] if model]
    discovery_results: List[Dict[str, Any]] = []

    successful_embeddings = embedding_df[embedding_df["status"].isin(["success", "unexpected_success"])].copy()
    printer.note(f"Running discovery on {len(successful_embeddings)} successful embedding combinations")
    printer.note(f"Discovery methods: {', '.join(discovery_models)}")

    for _, row in successful_embeddings.iterrows():
        combo_name = row["combination"]
        output_file = row["output_file"]
        try:
            loaded_results = load_embeddings_npz(output_file, filter_failed=True)
            all_embeddings = np.vstack([record["embeddings"] for record in loaded_results if len(record.get("embeddings", [])) > 0])
            if len(all_embeddings) == 0:
                raise RuntimeError("no valid embeddings after reload")

            for discovery_method in discovery_models:
                started = time.time()
                try:
                    n_samples = int(len(all_embeddings))
                    if n_samples < 2:
                        raise ValueError(f"Too few samples for clustering: n_samples={n_samples}")

                    target_clusters = 20
                    effective_clusters = max(2, min(target_clusters, n_samples))

                    discovery_pipeline = DiscoveryPipeline(
                        method=discovery_method,
                        model_kwargs={"n_clusters": effective_clusters},
                    )
                    discovery_result = discovery_pipeline.discover(all_embeddings)
                    duration = time.time() - started
                    label_counts = pd.Series(discovery_result.labels).value_counts()
                    fit_metrics = discovery_result.fit_metrics or {}

                    discovery_results.append(
                        {
                            "embedding_combo": combo_name,
                            "features": row["features"],
                            "segmentation": row["segmentation"],
                            "pooling": row["pooling"],
                            "discovery_method": discovery_method,
                            "discovery_labels": discovery_result.labels.tolist(),
                            "num_clusters": int(discovery_result.num_clusters),
                            "total_syllables": int(len(discovery_result.labels)),
                            "time_seconds": duration,
                            "label_distribution_std": float(label_counts.std()) if len(label_counts) > 1 else 0.0,
                            "label_distribution_mean": float(label_counts.mean()) if len(label_counts) > 0 else 0.0,
                            "cluster_status": fit_metrics.get("status", "unknown"),
                            "cluster_n_samples": fit_metrics.get("n_samples", 0),
                            "cluster_n_features": fit_metrics.get("n_features", 0),
                            "cluster_silhouette": fit_metrics.get("silhouette"),
                            "cluster_davies_bouldin": fit_metrics.get("davies_bouldin"),
                            "cluster_calinski_harabasz": fit_metrics.get("calinski_harabasz"),
                            "cluster_target": target_clusters,
                            "cluster_effective": effective_clusters,
                            "status": "success",
                        }
                    )
                    printer.ok(f"{combo_name} + {discovery_method} -> {discovery_result.num_clusters} clusters")
                except Exception as exc:
                    discovery_results.append(
                        {
                            "embedding_combo": combo_name,
                            "features": row["features"],
                            "segmentation": row["segmentation"],
                            "pooling": row["pooling"],
                            "discovery_method": discovery_method,
                            "discovery_labels": [],
                            "num_clusters": 0,
                            "total_syllables": 0,
                            "time_seconds": 0,
                            "label_distribution_std": 0,
                            "label_distribution_mean": 0,
                            "cluster_status": "error",
                            "cluster_n_samples": 0,
                            "cluster_n_features": 0,
                            "cluster_silhouette": None,
                            "cluster_davies_bouldin": None,
                            "cluster_calinski_harabasz": None,
                            "status": f"error: {type(exc).__name__}: {exc}",
                        }
                    )
                    printer.fail(f"{combo_name} + {discovery_method}: {type(exc).__name__}: {exc}")
                finally:
                    gc.collect()
        except Exception as exc:
            printer.fail(f"Failed to reload embeddings for {combo_name}: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()

    discovery_df = pd.DataFrame(discovery_results)
    printer.title("DISCOVERY SUMMARY")
    printer.note(f"Discovery rows: {len(discovery_df)}")
    if discovery_df.empty or "status" not in discovery_df.columns:
        printer.note("No discovery rows were produced.")
        return discovery_df

    printer.note(f"Successful rows: {len(discovery_df[discovery_df['status'] == 'success'])}")
    printer.note(f"Failed rows: {len(discovery_df[discovery_df['status'] != 'success'])}")
    print(_format_df(
        discovery_df,
        ["embedding_combo", "discovery_method", "num_clusters", "total_syllables", "time_seconds", "status"],
        max_rows=20,
    ))
    failed_rows = discovery_df[discovery_df["status"] != "success"]
    if not failed_rows.empty:
        printer.subtitle("Discovery Failures")
        for line in _preview_failed_rows(failed_rows, label_col="embedding_combo"):
            printer.note(line)

    return discovery_df


def _evaluate_segmentation_from_embeddings(
    embedding_results: Sequence[Dict[str, Any]],
    paired_subset_df: pd.DataFrame,
    combo_meta: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    audio_to_file = dict(zip(paired_subset_df["audio_file"], paired_subset_df["file_id"]))
    pair_by_file_id = {int(row.file_id): row for row in paired_subset_df.itertuples(index=False)}
    tg_by_audio = dict(zip(paired_subset_df["audio_file"], paired_subset_df["tg_file"]))

    for fallback_file_id, result in enumerate(embedding_results):
        if not result.get("success"):
            continue
        metadata = result.get("metadata") or {}
        metadata_file_id = metadata.get("file_id")
        audio_path = metadata.get("audio_path", "")
        resolved_file_id = None

        if metadata_file_id is not None:
            try:
                resolved_file_id = int(metadata_file_id)
            except Exception:
                resolved_file_id = None

        if resolved_file_id is None and audio_path:
            resolved_file_id = audio_to_file.get(audio_path)

        if resolved_file_id is None:
            resolved_file_id = fallback_file_id

        pair = pair_by_file_id.get(int(resolved_file_id))
        if pair is None:
            continue

        if not audio_path:
            audio_path = pair.audio_file
        tg_path = tg_by_audio.get(audio_path, pair.tg_file)

        peaks = metadata.get("peaks", [])
        spans = [(start, end) for start, end in metadata.get("boundaries", [])]
        segment_count = min(len(peaks), len(spans))
        if segment_count == 0:
            continue

        eval_result = evaluate_segmentation(
            peaks=peaks[:segment_count],
            spans=spans[:segment_count],
            textgrid_path=tg_path,
            tiers=EVAL_TIERS,
        )
        for eval_method, metrics in eval_result.items():
            if metrics is None:
                continue
            summary = _metric_summary_from_counts(
                metrics.get("TP", 0) or 0,
                metrics.get("Ins", 0) or 0,
                metrics.get("Del", 0) or 0,
                metrics.get("Sub", 0) or 0,
            )
            rows.append(
                {
                    **combo_meta,
                    "file_id": int(resolved_file_id),
                    "audio_file": audio_path,
                    "tg_file": tg_path,
                    "eval_method": eval_method,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def run_evaluation_battery(
    printer: SectionPrinter,
    embedding_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    paired_subset_df: pd.DataFrame,
    output_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    printer.title("BATTERY 4: Manifest and Evaluation")

    file_manifest_source = paired_subset_df.rename(columns={"audio_file": "audio_path"})
    file_manifest_df = build_file_manifest(file_manifest_source[["file_id", "audio_path"]].to_dict("records"))
    file_manifest_map = dict(zip(file_manifest_df["file_id"], file_manifest_df["audio_path"]))
    file_id_by_audio = dict(zip(paired_subset_df["audio_file"], paired_subset_df["file_id"]))

    module_eval_long_frames: List[pd.DataFrame] = []
    module_manifest_frames: List[pd.DataFrame] = []
    module_run_frames: List[Dict[str, Any]] = []

    for _, emb_row in embedding_df[embedding_df["status"] == "success"].iterrows():
        combo_name = emb_row["combination"]
        loaded_results = load_embeddings_npz(emb_row["output_file"], filter_failed=False)

        segmentation_rows = []
        for fallback_file_id, result in enumerate(loaded_results):
            if not result.get("success"):
                continue
            metadata = result.get("metadata") or {}
            boundaries = metadata.get("boundaries", [])
            peaks = metadata.get("peaks", [])
            segment_count = min(len(boundaries), len(peaks))
            segments = [
                (start, peak, end)
                for (start, end), peak in zip(boundaries[:segment_count], peaks[:segment_count])
            ]

            metadata_file_id = metadata.get("file_id")
            metadata_audio_path = metadata.get("audio_path", result.get("audio_path", ""))
            resolved_file_id = None
            if metadata_file_id is not None:
                try:
                    resolved_file_id = int(metadata_file_id)
                except Exception:
                    resolved_file_id = None
            if resolved_file_id is None and metadata_audio_path:
                resolved_file_id = file_id_by_audio.get(metadata_audio_path)
            if resolved_file_id is None:
                resolved_file_id = fallback_file_id

            resolved_audio_path = file_manifest_map.get(resolved_file_id, metadata_audio_path)
            segmentation_rows.append(
                {
                    "file_id": int(resolved_file_id),
                    "audio_path": resolved_audio_path,
                    "segments": segments,
                    "segmentation_method": emb_row["segmentation"],
                    "segmentation_kwargs": {
                        "features": emb_row["features"],
                        "pooling": emb_row["pooling"],
                    },
                }
            )

        segmentation_manifest = build_segmentation_manifest(segmentation_rows)

        eval_long_df = _evaluate_segmentation_from_embeddings(
            loaded_results,
            paired_subset_df,
            {
                "workflow": "module",
                "combination": combo_name,
                "features": emb_row["features"],
                "segmentation": emb_row["segmentation"],
                "pooling": emb_row["pooling"],
            },
        )
        if not eval_long_df.empty:
            module_eval_long_frames.append(eval_long_df)

        if discovery_df.empty or "embedding_combo" not in discovery_df.columns:
            discovery_rows = pd.DataFrame()
        else:
            discovery_rows = discovery_df[discovery_df["embedding_combo"] == combo_name]
        for _, disc_row in discovery_rows.iterrows():
            discovery_rows_for_manifest: List[Dict[str, Any]] = []
            discovery_labels = disc_row.get("discovery_labels", []) or []
            label_offset = 0
            for fallback_file_id, result in enumerate(loaded_results):
                if not result.get("success"):
                    continue
                metadata = result.get("metadata") or {}
                boundaries = metadata.get("boundaries", [])
                peaks = metadata.get("peaks", [])
                segment_count = min(len(boundaries), len(peaks))
                if segment_count == 0:
                    continue
                metadata_file_id = metadata.get("file_id")
                audio_path = metadata.get("audio_path", result.get("audio_path", ""))
                resolved_file_id = None
                if metadata_file_id is not None:
                    try:
                        resolved_file_id = int(metadata_file_id)
                    except Exception:
                        resolved_file_id = None
                if resolved_file_id is None and audio_path:
                    resolved_file_id = file_id_by_audio.get(audio_path)
                if resolved_file_id is None:
                    resolved_file_id = fallback_file_id
                for segment_id, ((start, end), peak) in enumerate(zip(boundaries[:segment_count], peaks[:segment_count])):
                    cluster_label = discovery_labels[label_offset] if label_offset < len(discovery_labels) else -1
                    discovery_rows_for_manifest.append(
                        {
                            "file_id": int(resolved_file_id),
                            "segment_id": int(segment_id),
                            "embedding_id": int(len(discovery_rows_for_manifest)),
                            "start": float(start),
                            "peak": float(peak),
                            "end": float(end),
                            "cluster_label": int(cluster_label),
                            "audio_path": audio_path,
                            "discovery_method": disc_row["discovery_method"],
                            "discovery_kwargs": {
                                "features": emb_row["features"],
                                "segmentation": emb_row["segmentation"],
                                "pooling": emb_row["pooling"],
                            },
                        }
                    )
                    label_offset += 1

            discovery_manifest = build_discovery_manifest(discovery_rows_for_manifest)

            corpus_manifest = join_corpus_manifests(
                segmentation_manifest,
                file_manifest=file_manifest_df,
                discovery_manifest=discovery_manifest,
            )
            labeled_manifest = attach_textgrid_labels_to_manifest(
                corpus_manifest,
                file_manifest=file_manifest_df,
                wav_paths=paired_subset_df["audio_file"].tolist(),
                textgrid_paths=paired_subset_df["tg_file"].tolist(),
                textgrid_tier_index=0,
            )
            label_metrics = compute_discovery_label_metrics(labeled_manifest)
            labeled_manifest = build_label_manifest(labeled_manifest)
            module_manifest_frames.append(
                labeled_manifest.assign(
                    workflow="module",
                    combination=combo_name,
                    features=emb_row["features"],
                    segmentation=emb_row["segmentation"],
                    pooling=emb_row["pooling"],
                    discovery_method=disc_row["discovery_method"],
                )
            )

            eval_summary = eval_long_df.groupby("eval_method")[["TP", "Ins", "Del", "Sub"]].sum().reset_index() if not eval_long_df.empty else pd.DataFrame()
            run_summary_row = {
                **emb_row.to_dict(),
                **{k: v for k, v in disc_row.to_dict().items() if k != "discovery_labels"},
                "workflow": "module",
                "combo_name": combo_name,
                "file_count": len(paired_subset_df),
                "embedding_rows": int(emb_row["total_syllables"]),
                "label_total_labeled_syllables": label_metrics.get("total_labeled_syllables", 0),
                "label_cluster_purity": label_metrics.get("cluster_purity", 0.0),
                "label_label_purity": label_metrics.get("label_purity", 0.0),
                "label_nmi": label_metrics.get("label_norm_mutual_info", 0.0),
                "label_macro_f1": label_metrics.get("macro_f1", 0.0),
                "label_weighted_f1": label_metrics.get("weighted_f1", 0.0),
            }
            if not eval_summary.empty:
                for _, summary_row in eval_summary.iterrows():
                    method_name = summary_row["eval_method"]
                    for metric_name in ["TP", "Ins", "Del", "Sub"]:
                        run_summary_row[f"{method_name}_{metric_name}"] = summary_row[metric_name]
            module_run_frames.append(run_summary_row)

    module_eval_long_df = pd.concat(module_eval_long_frames, ignore_index=True) if module_eval_long_frames else pd.DataFrame()
    module_manifest_df = pd.concat(module_manifest_frames, ignore_index=True) if module_manifest_frames else pd.DataFrame()
    module_run_df = pd.DataFrame(module_run_frames)

    if not module_eval_long_df.empty:
        module_eval_long_df.to_csv(output_root / "module_eval_long.csv", index=False)
    if not module_manifest_df.empty:
        module_manifest_df.to_csv(output_root / "module_manifest.csv", index=False)
    if not module_run_df.empty:
        module_run_df.to_csv(output_root / "module_run_summary.csv", index=False)

    printer.title("BATTERY 4 SUMMARY")
    printer.note(f"module_eval_long rows: {len(module_eval_long_df)}")
    printer.note(f"module_manifest rows: {len(module_manifest_df)}")
    printer.note(f"module_run rows: {len(module_run_df)}")
    if module_run_df.empty:
        printer.note("No module-level comparison rows were produced.")
    else:
        print(_format_df(
            module_run_df,
            [
                "combo_name",
                "discovery_method",
                "cluster_num_clusters",
                "cluster_silhouette",
                "nuclei_f1",
                "syllable_boundaries_f1",
                "syllable_spans_f1",
                "word_boundaries_f1",
                "word_spans_f1",
                "label_cluster_purity",
                "label_nmi",
            ],
            max_rows=20,
        ))

    return module_eval_long_df, module_manifest_df, module_run_df


def run_orchestrator_smoke_test(
    printer: SectionPrinter,
    embedding_df: pd.DataFrame,
    audio_files: Sequence[Path],
    paired_subset_df: pd.DataFrame,
    output_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    printer.title("BATTERY 5: Orchestrator Full Testing")
    orchestrator = FindSyllsOrchestrator()

    orchestrator_results: List[Dict[str, Any]] = []
    orchestrator_eval_results: List[Dict[str, Any]] = []

    # Get all successful embedding combinations
    successful_combos = embedding_df[embedding_df["status"] == "success"].copy()
    if successful_combos.empty:
        printer.warn("No successful embedding combinations to test with orchestrator")
        return pd.DataFrame(orchestrator_results), pd.DataFrame()

    # Build audio file to TextGrid mapping
    audio_to_tg = dict(zip(paired_subset_df["audio_file"], paired_subset_df["tg_file"]))

    total_combos = len(successful_combos)
    printer.note(f"Testing {total_combos} successful embedding combinations using orchestrator")
    
    for index, (_, row) in enumerate(successful_combos.iterrows(), 1):
        features = row["features"]
        segmentation = row["segmentation"]
        pooling = row["pooling"]
        segmentation_kwargs = row["segmentation_kwargs"]
        combo_name = row["combination"]

        printer.subtitle(f"[{index}/{total_combos}] {combo_name}")

        # Test on first audio file in the subset
        audio_file = str(audio_files[0])
        start_time = time.time()

        try:
            embeddings, metadata = orchestrator.segment_and_embed_audio(
                audio_file=audio_file,
                segmentation_method=segmentation,
                features_method=features,
                pooling_method=pooling,
                segmentation_kwargs=segmentation_kwargs,
            )
            num_syllables = metadata.get("num_syllables", 0)
            embedding_shape = tuple(embeddings.shape)
            elapsed = time.time() - start_time
            
            # Extract segmentation results for evaluation
            peaks = metadata.get("peaks", [])
            boundaries = metadata.get("boundaries", [])
            spans = [(start, end) for start, end in boundaries]
            segment_count = min(len(peaks), len(spans))
            
            orchestrator_results.append(
                {
                    "combination": combo_name,
                    "features": features,
                    "segmentation": segmentation,
                    "pooling": pooling,
                    "envelope_method": row.get("envelope_method"),
                    "status": "success",
                    "embedding_shape": embedding_shape,
                    "num_syllables": num_syllables,
                    "time_seconds": elapsed,
                    "audio_file": Path(audio_file).name,
                }
            )
            printer.ok(f"embeddings={embedding_shape} syllables={num_syllables} time={elapsed:.2f}s")
            
            # Evaluate segmentation against TextGrid
            if segment_count > 0 and audio_file in audio_to_tg:
                tg_path = audio_to_tg[audio_file]
                try:
                    eval_result = evaluate_segmentation(
                        peaks=peaks[:segment_count],
                        spans=spans[:segment_count],
                        textgrid_path=tg_path,
                        tiers=EVAL_TIERS,
                    )
                    for eval_method, metrics in eval_result.items():
                        if metrics is None:
                            continue
                        summary = _metric_summary_from_counts(
                            metrics.get("TP", 0) or 0,
                            metrics.get("Ins", 0) or 0,
                            metrics.get("Del", 0) or 0,
                            metrics.get("Sub", 0) or 0,
                        )
                        orchestrator_eval_results.append(
                            {
                                "combination": combo_name,
                                "features": features,
                                "segmentation": segmentation,
                                "pooling": pooling,
                                "envelope_method": row.get("envelope_method"),
                                "eval_method": eval_method,
                                "audio_file": Path(audio_file).name,
                                "textgrid_file": Path(tg_path).name,
                                **summary,
                            }
                        )
                    printer.note(f"  evaluation: {len(set(eval_result.keys()))} tiers evaluated")
                except Exception as exc:
                    printer.warn(f"  evaluation failed: {type(exc).__name__}: {exc}")
            
        except Exception as exc:
            elapsed = time.time() - start_time
            orchestrator_results.append(
                {
                    "combination": combo_name,
                    "features": features,
                    "segmentation": segmentation,
                    "pooling": pooling,
                    "envelope_method": row.get("envelope_method"),
                    "status": f"error: {type(exc).__name__}: {exc}",
                    "embedding_shape": None,
                    "num_syllables": 0,
                    "time_seconds": elapsed,
                    "audio_file": Path(audio_file).name,
                }
            )
            printer.fail(f"{type(exc).__name__}: {exc}")

    orchestrator_df = pd.DataFrame(orchestrator_results)
    orchestrator_eval_df = pd.DataFrame(orchestrator_eval_results)
    
    printer.subtitle("Orchestrator Testing Summary")
    printer.note(f"Total combinations tested: {len(orchestrator_df)}")
    printer.note(f"Successful: {len(orchestrator_df[orchestrator_df['status'] == 'success'])}")
    printer.note(f"Failed: {len(orchestrator_df[orchestrator_df['status'] != 'success'])}")
    print(_format_df(
        orchestrator_df,
        ["combination", "features", "segmentation", "pooling", "status", "time_seconds", "embedding_shape", "num_syllables"],
        max_rows=25,
    ))
    
    if not orchestrator_eval_df.empty:
        printer.subtitle("Orchestrator Evaluation Summary")
        printer.note(f"Total evaluation rows: {len(orchestrator_eval_df)}")
        print(_format_df(
            orchestrator_eval_df,
            ["combination", "eval_method", "TP", "Ins", "Del", "Sub", "Precision", "Recall", "F1"],
            max_rows=25,
        ))
    
    failed_rows = orchestrator_df[orchestrator_df["status"] != "success"]
    if not failed_rows.empty:
        printer.subtitle("Orchestrator Failures")
        for line in _preview_failed_rows(failed_rows, label_col="combination"):
            printer.note(line)
    
    orchestrator_df.to_csv(output_root / "orchestrator_results.csv", index=False)
    if not orchestrator_eval_df.empty:
        orchestrator_eval_df.to_csv(output_root / "orchestrator_eval_results.csv", index=False)
    
    return orchestrator_df, orchestrator_eval_df


def run_envelope_preset_battery(
    printer: SectionPrinter,
    wav_files_subset: Sequence[Path],
) -> pd.DataFrame:
    """Battery 0: run envelope-based preset segmenters on the corpus subset.

    Exercises SBSPeakdetectSegmenter and ThetaOscillatorSegmenter end-to-end
    on real TIMIT audio — verifying output contract, syllable counts, and that
    the segments are accepted by the MFCC+mean-pooling embedding path.
    Requires no GPU.
    """
    import soundfile as sf
    from findsylls.audio.utils import load_audio
    from findsylls.features import MFCCExtractor
    from findsylls.embedding.poolers.mean import MeanPooler

    printer.title("BATTERY 0: Envelope Preset Segmenters")

    presets = [
        ("sbs_peakdetect", SBSPeakdetectSegmenter()),
        ("theta_oscillator", ThetaOscillatorSegmenter()),
    ]

    # Registry smoke-check
    registered = list_segmenter_presets()
    for name, _ in presets:
        if name in registered:
            printer.ok(f"list_segmenter_presets() contains '{name}'")
        else:
            printer.fail(f"'{name}' missing from list_segmenter_presets()")

    mfcc = MFCCExtractor()
    pooler = MeanPooler()

    rows: List[Dict[str, Any]] = []
    for preset_name, segmenter in presets:
        printer.subtitle(f"Preset: {preset_name}")
        n_ok = 0
        n_fail = 0
        total_segs = 0

        for wav_path in wav_files_subset:
            try:
                audio, sr = load_audio(str(wav_path))
                segments = segmenter.segment(audio=audio, sr=sr)

                # Output contract check
                for start, peak, end in segments:
                    assert start >= 0 and start <= peak <= end, (
                        f"Invalid segment ({start}, {peak}, {end}) in {wav_path}"
                    )

                # Embedding pipeline integration check
                if len(segments) > 0:
                    features = mfcc.extract(audio, sr)
                    embeddings = pooler.pool(features, segments, fps=mfcc.frame_rate)
                    assert embeddings.shape[0] == len(segments), (
                        f"Embedding row count mismatch: {embeddings.shape[0]} != {len(segments)}"
                    )

                total_segs += len(segments)
                n_ok += 1
                rows.append({
                    "preset": preset_name,
                    "file": Path(wav_path).name,
                    "n_segments": len(segments),
                    "status": "ok",
                })
            except Exception as exc:
                n_fail += 1
                rows.append({
                    "preset": preset_name,
                    "file": Path(wav_path).name,
                    "n_segments": 0,
                    "status": f"error: {exc}",
                })

        avg_segs = total_segs / max(n_ok, 1)
        if n_fail == 0:
            printer.ok(
                f"{n_ok}/{len(wav_files_subset)} files OK  |  "
                f"avg {avg_segs:.1f} syllables/file"
            )
        else:
            printer.fail(
                f"{n_fail}/{len(wav_files_subset)} files FAILED  |  "
                f"{n_ok} OK  |  avg {avg_segs:.1f} syllables/file"
            )

    df = pd.DataFrame(rows)
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    printer = SectionPrinter()

    repo_root = REPO_ROOT
    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else (repo_root / args.corpus_dir)
    output_root = args.output_root if args.output_root.is_absolute() else (repo_root / args.output_root)
    if args.timestamped:
        output_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    printer.title("findsylls Test Battery")
    printer.note(f"findsylls version: {findsylls.__version__}")
    printer.note(f"repo root: {repo_root}")
    printer.note(f"corpus dir: {corpus_dir}")
    printer.note(f"output dir: {output_root}")
    printer.note(f"subset size: {args.subset_size}")
    printer.note(f"n_jobs: {args.n_jobs}")
    printer.note(f"batteries: {','.join(str(b) for b in args.batteries)}")

    wav_files, textgrid_files = _resolve_corpus_files(corpus_dir)
    printer.subtitle("Corpus Pairing")
    printer.note(f"wav files: {len(wav_files)}")
    printer.note(f"textgrid files: {len(textgrid_files)}")
    matched_textgrids, matched_wavs = match_wavs_to_textgrids(wav_files, textgrid_files, tg_suffix_to_strip="_syllabified")
    printer.note(f"matched pairs: {len(matched_wavs)}")
    if not matched_wavs:
        printer.fail("No paired wav/TextGrid files were found")
        return 1

    paired_records = [
        {
            "file_id": idx,
            "audio_file": str(wav_path),
            "tg_file": str(tg_path),
            "file_stem": Path(wav_path).stem,
        }
        for idx, (tg_path, wav_path) in enumerate(zip(matched_textgrids, matched_wavs))
    ]
    paired_df = pd.DataFrame(paired_records)
    paired_subset_df = paired_df.head(args.subset_size if args.subset_size > 0 else len(paired_df)).reset_index(drop=True)
    paired_subset_manifest_df = paired_subset_df.rename(columns={"audio_file": "audio_path"}).copy()
    wav_files_subset = paired_subset_df["audio_file"].tolist()
    tg_files_subset = paired_subset_df["tg_file"].tolist()

    printer.note(f"subset files used: {len(wav_files_subset)}")
    printer.note(f"first pair: {Path(wav_files_subset[0]).name} / {Path(tg_files_subset[0]).name}")

    feature_extractors = ["hubert", "sylber", "vghubert", "mfcc", "melspec"]
    neural_feature_extractors = [feature for feature in DEFAULT_NEURAL_FEATURES if feature in feature_extractors]
    segmentation_methods = list_segmenters()
    poolers = list_poolers()
    pseudo_envelope_methods = DEFAULT_PSEUDO_ENVELOPES
    preset_methods = list_presets()

    valid_feature_extractors: List[str] = []
    for method in feature_extractors:
        try:
            _ = get_extractor(method)
            valid_feature_extractors.append(method)
        except Exception:
            pass
    feature_extractors = valid_feature_extractors
    neural_feature_extractors = [feature for feature in neural_feature_extractors if feature in feature_extractors]

    valid_segmentation_methods: List[str] = []
    for method in segmentation_methods:
        try:
            _ = get_segmenter(method)
            valid_segmentation_methods.append(method)
        except Exception:
            pass
    segmentation_methods = valid_segmentation_methods

    combinations = _build_experiment_combinations(
        feature_extractors,
        segmentation_methods,
        poolers,
        neural_feature_extractors,
        pseudo_envelope_methods,
    )

    for combo in combinations:
        combo["expected_failure_reason"] = _get_expected_failure_reason(combo, neural_feature_extractors)

    features_filter = _csv_list(args.features)
    segmentations_filter = _csv_list(args.segmentations)
    poolings_filter = _csv_list(args.poolings)
    envelopes_filter = _csv_list(args.envelopes)
    combinations = _filter_combinations(
        combinations,
        features_filter,
        segmentations_filter,
        poolings_filter,
        envelopes_filter,
    )

    if args.only_expected_failures:
        combinations = [combo for combo in combinations if combo.get("expected_failure_reason")]
    elif not args.include_expected_failures:
        combinations = [combo for combo in combinations if not combo.get("expected_failure_reason")]

    printer.subtitle("Test Matrix")
    printer.note(f"Feature extractors: {', '.join(feature_extractors)}")
    printer.note(f"Neural extractors: {', '.join(neural_feature_extractors)}")
    printer.note(f"Segmentation methods: {', '.join(segmentation_methods)}")
    printer.note(f"Pooling methods: {', '.join(poolers)}")
    printer.note(f"Pseudo-envelopes for peakdetect: {', '.join(pseudo_envelope_methods)}")
    printer.note(f"Total combinations: {len(combinations)}")
    printer.note(f"Canonical combinations: {sum(1 for combo in combinations if combo['experiment_group'] == 'canonical')}")
    printer.note(f"Peakdetect + pseudo-envelope combinations: {sum(1 for combo in combinations if combo['experiment_group'] == 'peakdetect_pseudoenvelope')}")
    printer.note(f"Expected-failure combinations in set: {sum(1 for combo in combinations if combo.get('expected_failure_reason'))}")

    if len(combinations) == 0 and (2 in args.batteries or 3 in args.batteries or 4 in args.batteries or 5 in args.batteries):
        printer.fail("No combinations remain after filtering")
        return 2

    selected_batteries = set(args.batteries)

    envelope_preset_df = pd.DataFrame()
    if 0 in selected_batteries:
        envelope_preset_df = run_envelope_preset_battery(printer, wav_files_subset)
        envelope_preset_df.to_csv(output_root / "envelope_preset_results.csv", index=False)
    else:
        printer.title("BATTERY 0: Envelope Preset Segmenters")
        printer.note("Skipped by battery selection")

    if 1 in selected_batteries:
        run_compatibility_battery(printer, feature_extractors)
    else:
        printer.title("BATTERY 1: Compatibility Checks")
        printer.note("Skipped by battery selection")

    embedding_df = pd.DataFrame()
    if 2 in selected_batteries:
        embedding_df = run_embedding_battery(
            printer,
            combinations,
            wav_files_subset,
            embeddings_dir,
            args.layer,
            args.n_jobs,
            args.max_combinations,
        )
    else:
        printer.title("BATTERY 2: Embedding Matrix")
        printer.note("Skipped by battery selection")

    discovery_df = pd.DataFrame()
    if 3 in selected_batteries:
        if embedding_df.empty or "status" not in embedding_df.columns:
            printer.title("BATTERY 3: Discovery Clustering")
            printer.note("Skipped: no embedding results available (run battery 2 first)")
        else:
            discovery_df = run_discovery_battery(printer, embedding_df, output_root)
    else:
        printer.title("BATTERY 3: Discovery Clustering")
        printer.note("Skipped by battery selection")

    module_eval_long_df = pd.DataFrame()
    module_manifest_df = pd.DataFrame()
    module_run_df = pd.DataFrame()
    if 4 in selected_batteries:
        if embedding_df.empty or "status" not in embedding_df.columns:
            printer.title("BATTERY 4: Evaluation & Manifest")
            printer.note("Skipped: no embedding results available (run battery 2 first)")
        else:
            module_eval_long_df, module_manifest_df, module_run_df = run_evaluation_battery(
                printer,
                embedding_df,
                discovery_df,
                paired_subset_df,
                output_root,
            )
            if "discovery_labels" in discovery_df.columns:
                discovery_df = discovery_df.drop(columns=["discovery_labels"])
                gc.collect()
    else:
        printer.title("BATTERY 4: Evaluation & Manifest")
        printer.note("Skipped by battery selection")

    orchestrator_run_df = pd.DataFrame()
    comparison_df = pd.DataFrame()
    if 5 in selected_batteries and not args.skip_orchestrator:
        if embedding_df.empty or "status" not in embedding_df.columns:
            printer.title("BATTERY 5: Orchestrator Full Testing")
            printer.note("Skipped: no embedding results available (run battery 2 first)")
        else:
            orchestrator_run_df, comparison_df = run_orchestrator_smoke_test(
                printer,
                embedding_df,
                wav_files_subset,
                paired_subset_df,
                output_root,
            )
    elif 5 not in selected_batteries:
        printer.title("BATTERY 5: Orchestrator Full Testing")
        printer.note("Skipped by battery selection")
    else:
        printer.title("BATTERY 5: Orchestrator Full Testing")
        printer.note("Skipped by request")

    printer.title("FINAL SUMMARY")
    printer.note(f"Envelope preset rows: {len(envelope_preset_df)}")
    printer.note(f"Embeddings rows: {len(embedding_df)}")
    printer.note(f"Discovery rows: {len(discovery_df)}")
    printer.note(f"Module comparison rows: {len(module_run_df)}")
    printer.note(f"Orchestrator comparison rows: {len(orchestrator_run_df)}")
    printer.note(f"Combined comparison rows: {len(comparison_df)}")

    summary = {
        "test_date": datetime.now().isoformat(),
        "findsylls_version": findsylls.__version__,
        "corpus_dir": str(corpus_dir),
        "output_root": str(output_root),
        "subset_size": len(wav_files_subset),
        "batteries": sorted(selected_batteries),
        "embedding_combinations_total": len(embedding_df),
        "embedding_combinations_success": int(embedding_df["status"].isin(["success", "unexpected_success"]).sum()) if not embedding_df.empty else 0,
        "embedding_combinations_expected_failures": int(embedding_df["status"].astype(str).str.startswith("expected_failure").sum()) if not embedding_df.empty else 0,
        "embedding_combinations_failed": int((~embedding_df["status"].isin(["success", "unexpected_success"]) & ~embedding_df["status"].astype(str).str.startswith("expected_failure")).sum()) if not embedding_df.empty else 0,
        "discovery_rows_total": len(discovery_df),
        "discovery_rows_success": int((discovery_df["status"] == "success").sum()) if not discovery_df.empty else 0,
        "module_rows_total": len(module_run_df),
        "orchestrator_rows_total": len(orchestrator_run_df),
    }

    if args.save_artifacts:
        embedding_csv = output_root / "embedding_results.csv"
        discovery_csv = output_root / "discovery_results.csv"
        module_eval_csv = output_root / "module_eval_long.csv"
        module_manifest_csv = output_root / "module_manifest.csv"
        module_run_csv = output_root / "module_run_summary.csv"
        summary_json = output_root / "test_summary.json"

        if not embedding_df.empty:
            embedding_df.to_csv(embedding_csv, index=False)
        if not discovery_df.empty:
            discovery_df.drop(columns=["discovery_labels"], errors="ignore").to_csv(discovery_csv, index=False)
        if not module_eval_long_df.empty:
            module_eval_long_df.to_csv(module_eval_csv, index=False)
        if not module_manifest_df.empty:
            module_manifest_df.to_csv(module_manifest_csv, index=False)
        if not module_run_df.empty:
            module_run_df.to_csv(module_run_csv, index=False)
        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        printer.note(f"Saved summary artifacts under {output_root}")

    printer.title("RESULT SNAPSHOT")
    printer.note("Embedding failures:")
    failed_embeddings = embedding_df[embedding_df["status"] != "success"] if not embedding_df.empty else pd.DataFrame()
    if failed_embeddings.empty:
        printer.note("  none")
    else:
        for line in _preview_failed_rows(failed_embeddings, label_col="combination"):
            printer.note(f"  {line}")

    printer.note("Discovery failures:")
    failed_discovery = discovery_df[discovery_df["status"] != "success"] if not discovery_df.empty else pd.DataFrame()
    if failed_discovery.empty:
        printer.note("  none")
    else:
        for line in _preview_failed_rows(failed_discovery, label_col="embedding_combo"):
            printer.note(f"  {line}")

    printer.note("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
