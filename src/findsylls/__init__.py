"""findsylls: Unsupervised syllable-like segmentation & evaluation toolkit.

Public API:
  segment_audio, run_evaluation
  get_amplitude_envelope, segment_envelope
  evaluate_syllable_segmentation, evaluate_segmentation
  flatten_results, aggregate_results
  plot_segmentation_result
"""
from .pipeline import segment_audio, run_evaluation, flatten_results, aggregate_results
from .envelope import get_amplitude_envelope
from .segmentation import segment_envelope
from .evaluation import evaluate_syllable_segmentation, evaluate_segmentation
from .plotting import plot_segmentation_result

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
]

__version__ = "0.1.1"
