"""Package-level orchestrator for end-to-end workflows."""

import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..embedding import embed_audio as embed_audio_wrapper
from ..embedding.pipeline import EmbeddingPipeline
from ..embedding.storage import load_embedding_manifest
from ..evaluation.evaluator import evaluate_segmentation
from ..pipeline.results import flatten_results
from ..audio.utils import match_wavs_to_textgrids
from ..pipeline.manifests import build_discovery_manifest, build_file_manifest, build_segmentation_manifest, join_corpus_manifests
from . import pipeline as pipeline_module


_ENVELOPE_FEATURE_METHODS = {
    "rms",
    "hilbert",
    "lowpass",
    "sbs",
    "theta",
    "cls_attention",
    "greedy_cosine",
    "mincut",
}


def _normalize_method_name(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _has_explicit_envelope_configuration(segmentation_kwargs: Optional[Dict[str, Any]]) -> bool:
    if not segmentation_kwargs:
        return False

    for key in ("envelope_method", "envelope_computer"):
        if segmentation_kwargs.get(key) is not None:
            return True
    return False


def _validate_peakdetect_envelope_policy(
    *,
    segmentation_method: Optional[str],
    features_method: Optional[str],
    segmentation_kwargs: Optional[Dict[str, Any]],
) -> None:
    segmentation_name = _normalize_method_name(segmentation_method)
    feature_name = _normalize_method_name(features_method)

    if segmentation_name != "peakdetect":
        return

    if _has_explicit_envelope_configuration(segmentation_kwargs):
        return

    if feature_name in _ENVELOPE_FEATURE_METHODS:
        return

    raise ValueError(
        "Invalid configuration: segmentation_method='peakdetect' requires a 1-D envelope method. "
        f"features_method='{features_method}' is not an envelope method. "
        "Set segmentation_kwargs['envelope_method'] or provide 'envelope_computer'."
    )


class FindSyllsOrchestrator:
    """Class-first orchestrator for segmentation, embedding, and discovery flows."""

    @staticmethod
    def _resolve_audio_inputs(audio_files: Union[Sequence[str], str]) -> List[str]:
        if isinstance(audio_files, (str, Path)):
            return sorted(glob(str(audio_files), recursive=True))
        return [str(Path(audio_file)) for audio_file in audio_files]

    def segment_audio(self, *args, **kwargs):
        return pipeline_module.segment_audio(*args, **kwargs)

    def run_evaluation(self, *args, **kwargs):
        return pipeline_module.run_evaluation(*args, **kwargs)

    def segment_and_embed_audio(
        self,
        audio_file: str,
        segmentation_method: str = "peakdetect",
        features_method: str = "mfcc",
        pooling_method: str = "mean",
        segmentation_kwargs: Optional[Dict[str, Any]] = None,
        feature_kwargs: Optional[Dict[str, Any]] = None,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        _validate_peakdetect_envelope_policy(
            segmentation_method=segmentation_method,
            features_method=features_method,
            segmentation_kwargs=segmentation_kwargs,
        )

        embeddings, metadata = embed_audio_wrapper(
            audio_path=audio_file,
            segmentation=segmentation_method,
            features=features_method,
            pooling=pooling_method,
            segmentation_kwargs=segmentation_kwargs,
            feature_kwargs=feature_kwargs,
            pooling_kwargs=pooling_kwargs,
            return_metadata=True,
        )
        return embeddings, metadata

    def segment_embed_and_discover(
        self,
        audio_file: str,
        discovery_method: str = "kmeans",
        discovery_kwargs: Optional[Dict[str, Any]] = None,
        **embed_kwargs,
    ) -> Dict[str, Any]:
        from ..discovery import DiscoveryPipeline

        embeddings, metadata = self.segment_and_embed_audio(audio_file, **embed_kwargs)
        pipeline = DiscoveryPipeline(method=discovery_method, model_kwargs=discovery_kwargs)
        discovery_result = pipeline.discover(embeddings)
        return {
            "embeddings": embeddings,
            "metadata": metadata,
            "discovery": discovery_result,
        }

    def discover_corpus(
        self,
        audio_files: Union[Sequence[str], str],
        output_dir: Union[str, Path],
        *,
        segmentation_method: str = "sylber",
        features_method: str = "sylber",
        pooling_method: str = "mean",
        discovery_method: str = "kmeans",
        segmentation_kwargs: Optional[Dict[str, Any]] = None,
        feature_kwargs: Optional[Dict[str, Any]] = None,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
        discovery_kwargs: Optional[Dict[str, Any]] = None,
        sr: int = 16000,
        layer: Optional[int] = None,
        device: str = "auto",
        n_jobs: int = 1,
        verbose: bool = True,
        fail_on_error: bool = False,
        persist: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Run corpus embedding and discovery from the top-level orchestrator."""
        from ..discovery import DiscoveryPipeline
        from ..embedding.storage import iter_embeddings_from_manifest

        _validate_peakdetect_envelope_policy(
            segmentation_method=segmentation_method,
            features_method=features_method,
            segmentation_kwargs=segmentation_kwargs,
        )

        resolved_audio_files = self._resolve_audio_inputs(audio_files)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        embedding_pipeline = EmbeddingPipeline(
            segmentation=segmentation_method,
            features=features_method,
            pooling=pooling_method,
            sr=sr,
            layer=layer,
            device=device,
            segmentation_kwargs=segmentation_kwargs,
            feature_kwargs=feature_kwargs,
            pooling_kwargs=pooling_kwargs,
        )

        embedding_bundle = embedding_pipeline.embed_corpus_to_storage(
            audio_files=resolved_audio_files,
            output_dir=output_dir / "embeddings",
            verbose=verbose,
            fail_on_error=fail_on_error,
        )

        embedding_manifest_path = Path(embedding_bundle["manifest_path"])
        embedding_rows = load_embedding_manifest(embedding_manifest_path)
        successful_rows = [row for row in embedding_rows if row.get("success")]

        file_manifest = build_file_manifest(successful_rows, output_path=output_dir / "file_manifest.csv")

        segmentation_rows: List[Dict[str, Any]] = []
        for row in successful_rows:
            embedding_path = Path(row["embedding_path"])
            with np.load(embedding_path, allow_pickle=True) as data:
                metadata = json.loads(str(data["metadata"]))
            boundaries = metadata.get("boundaries", [])
            peaks = metadata.get("peaks", [])
            segments = [(start, peak, end) for (start, end), peak in zip(boundaries, peaks)]
            segmentation_rows.append(
                {
                    "file_id": row["file_id"],
                    "audio_path": row["audio_path"],
                    "segments": segments,
                    "segmentation_method": row.get("segmentation", segmentation_method),
                    "segmentation_kwargs": {
                        "features": features_method,
                        "pooling": pooling_method,
                    },
                }
            )

        segmentation_manifest = build_segmentation_manifest(
            segmentation_rows,
            output_path=output_dir / "segmentation_manifest.csv",
        )

        discovery_pipeline = DiscoveryPipeline(
            method=discovery_method,
            model_kwargs=discovery_kwargs,
        )

        if discovery_pipeline.model.supports_streaming:
            discovery_result = discovery_pipeline.discover_from_storage(
                manifest_path=str(embedding_manifest_path),
            )
        else:
            embedding_chunks = list(iter_embeddings_from_manifest(embedding_manifest_path))
            embeddings = np.vstack(embedding_chunks) if embedding_chunks else np.empty((0, 0))
            discovery_result = discovery_pipeline.discover(embeddings)

        discovery_rows: List[Dict[str, Any]] = []
        label_offset = 0
        embedding_offset = 0
        for row in successful_rows:
            embedding_path = Path(row["embedding_path"])
            with np.load(embedding_path, allow_pickle=True) as data:
                metadata = json.loads(str(data["metadata"]))
            boundaries = metadata.get("boundaries", [])
            peaks = metadata.get("peaks", [])
            segment_count = min(len(boundaries), len(peaks))
            cluster_labels = discovery_result.labels[label_offset: label_offset + segment_count]
            label_offset += segment_count

            for segment_id, ((start, end), peak, cluster_label) in enumerate(
                zip(boundaries[:segment_count], peaks[:segment_count], cluster_labels)
            ):
                discovery_rows.append(
                    {
                        "file_id": row["file_id"],
                        "segment_id": int(segment_id),
                        "embedding_id": int(embedding_offset + segment_id),
                        "embedding_path": row["embedding_path"],
                        "audio_path": row["audio_path"],
                        "start": float(start),
                        "peak": float(peak),
                        "end": float(end),
                        "cluster_label": int(cluster_label),
                        "discovery_method": discovery_method,
                        "discovery_model_path": "",
                        "segmentation_method": row.get("segmentation", segmentation_method),
                        "features": row.get("features", features_method),
                        "pooling": row.get("pooling", pooling_method),
                    }
                )
            embedding_offset += segment_count

        discovery_manifest = build_discovery_manifest(
            discovery_rows,
            output_path=output_dir / "discovery_manifest.csv",
        )
        discovery_manifest_path = output_dir / "discovery_manifest.csv"

        corpus_manifest = join_corpus_manifests(
            segmentation_manifest,
            file_manifest=file_manifest,
            discovery_manifest=discovery_manifest,
        )
        corpus_manifest_path = output_dir / "corpus_manifest.csv"
        corpus_manifest.to_csv(corpus_manifest_path, index=False)

        model_path = None
        if persist:
            model_dir = output_dir / "discovery_model"
            discovery_pipeline.save(
                model_dir,
                metadata={
                    "embedding_manifest_path": str(embedding_manifest_path),
                    "segmentation_manifest_path": str(output_dir / "segmentation_manifest.csv"),
                    "discovery_manifest_path": str(discovery_manifest_path),
                    "corpus_manifest_path": str(corpus_manifest_path),
                },
                overwrite=overwrite,
            )
            model_path = str(model_dir)

        return {
            "output_dir": str(output_dir),
            "embedding_bundle": embedding_bundle,
            "embedding_manifest_path": str(embedding_manifest_path),
            "file_manifest_path": str(output_dir / "file_manifest.csv"),
            "segmentation_manifest_path": str(output_dir / "segmentation_manifest.csv"),
            "discovery_manifest_path": str(discovery_manifest_path),
            "corpus_manifest_path": str(corpus_manifest_path),
            "discovery_model_path": model_path,
            "discovery_result": discovery_result,
            "discovery_metrics": discovery_result.fit_metrics,
            "discovery_pipeline": discovery_pipeline,
            "corpus_manifest": corpus_manifest,
        }
