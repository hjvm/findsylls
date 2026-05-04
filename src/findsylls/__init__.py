"""findsylls: Unsupervised syllable-like segmentation & evaluation toolkit.

Public API:
  Segmentation: segment_audio, run_evaluation
    Envelope: get_amplitude_envelope
        Evaluation: evaluate_segmentation,
                  attach_textgrid_labels_to_manifest, compute_discovery_label_metrics,
                  export_discovery_label_artifacts, load_discovery_label_artifacts
    Manifests: build_file_manifest, build_segmentation_manifest, join_corpus_manifests, load_manifest
  Results: flatten_results, aggregate_results
  Plotting: plot_segmentation_result
  Embedding: embed_audio, embed_corpus, save_embeddings, load_embeddings
    Discovery: DiscoveryPipeline, save_discovery_pipeline, load_discovery_pipeline
"""
from .pipeline import (
        segment_audio,
        run_evaluation,
        flatten_results,
        aggregate_results,
        build_file_manifest,
        build_segmentation_manifest,
        build_discovery_manifest,
        join_corpus_manifests,
        load_manifest,
        discover_corpus,
        build_label_manifest,
)
from .envelope import get_amplitude_envelope
from .segmentation import list_segmenters, list_segmenter_aliases
from .presets import list_presets, get_preset, resolve_preset
from .evaluation import (
    evaluate_segmentation,
    attach_textgrid_labels_to_manifest,
    compute_discovery_label_metrics,
    export_discovery_label_artifacts,
    load_discovery_label_artifacts,
)
from .plotting import plot_segmentation_result

from .embedding import embed_audio, embed_corpus
from .embedding.storage import save_embeddings, load_embeddings

try:
    from .discovery import DiscoveryPipeline, save_discovery_pipeline, load_discovery_pipeline
except ImportError:
    DiscoveryPipeline = None
    save_discovery_pipeline = None
    load_discovery_pipeline = None

__all__ = [
    "__version__",
    "segment_audio",
    "run_evaluation",
    "get_amplitude_envelope",
    "list_segmenters",
    "list_segmenter_aliases",
    "list_presets",
    "get_preset",
    "resolve_preset",
    "evaluate_segmentation",
    "attach_textgrid_labels_to_manifest",
    "compute_discovery_label_metrics",
    "export_discovery_label_artifacts",
    "load_discovery_label_artifacts",
    "build_file_manifest",
    "build_segmentation_manifest",
    "build_discovery_manifest",
    "build_label_manifest",
    "join_corpus_manifests",
    "load_manifest",
    "discover_corpus",
    "flatten_results",
    "aggregate_results",
    "plot_segmentation_result",
    "embed_audio",
    "embed_corpus",
    "save_embeddings",
    "load_embeddings",
    "DiscoveryPipeline",
    "save_discovery_pipeline",
    "load_discovery_pipeline",
]

__version__ = "3.0.1"
