"""findsylls: Unsupervised syllable-like segmentation & evaluation toolkit.

Public API:
  Segmentation: segment_audio, run_evaluation
  Envelope: get_amplitude_envelope, segment_envelope
  Evaluation: evaluate_syllable_segmentation, evaluate_segmentation
  Results: flatten_results, aggregate_results
  Plotting: plot_segmentation_result
  Embedding: embed_audio, embed_corpus, save_embeddings, load_embeddings
"""
from .pipeline import segment_audio, run_evaluation, flatten_results, aggregate_results
from .envelope import get_amplitude_envelope
from .segmentation import segment_envelope
from .evaluation import evaluate_syllable_segmentation, evaluate_segmentation
from .plotting import plot_segmentation_result

# Embedding pipeline (Phase 1-3)
try:
    from .embedding import embed_audio, embed_corpus
    from .embedding.storage import save_embeddings, load_embeddings
    _has_embedding = True
except ImportError:
    _has_embedding = False
    # Create placeholder functions that raise informative errors
    def _embedding_not_available(*args, **kwargs):
        raise ImportError(
            "Embedding features require additional dependencies. "
            "Install with: pip install 'findsylls[embedding]' or pip install 'findsylls[all]'"
        )
    embed_audio = _embedding_not_available
    embed_corpus = _embedding_not_available
    save_embeddings = _embedding_not_available
    load_embeddings = _embedding_not_available

__all__ = [
    "__version__",
    "segment_audio",
    "run_evaluation",
    "get_amplitude_envelope",
    "segment_envelope",
    "evaluate_syllable_segmentation",
    "evaluate_segmentation",
    "flatten_results",
    "aggregate_results",
    "plot_segmentation_result",
    "embed_audio",
    "embed_corpus",
    "save_embeddings",
    "load_embeddings",
]

__version__ = "1.0.1"
