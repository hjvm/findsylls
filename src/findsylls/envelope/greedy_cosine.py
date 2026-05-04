"""
Greedy Cosine Pseudo-Envelope Computer

Computes 1-D pseudo-envelope traces from frame features using the canonical
GreedyCosine phase-1 merge-similarity signal.

Critical Architecture Principle:
- Feature extractors provide ONLY frame features/states
- Segmenter and pseudo-envelope independently consume that same feature sequence
- Canonical similarity logic lives in `segmentation/greedy_cosine.py`
- This module imports canonical helpers and exposes a 1-D trace for plotting

This keeps segmentation behavior authoritative while enabling:
- plotting/diagnostics,
- envelope-based experiments (e.g., peakdetect on merge-similarity trace),
- direct consistency checks between thresholded trace and split/merge behavior.
"""

import numpy as np
from typing import Tuple

from .base import PseudoEnvelope
from ..features.base import FeatureExtractor
from ..segmentation.greedy_cosine import (
    compute_greedy_cosine_merge_similarity_trace,
)


class GreedyCosineEnvelope(PseudoEnvelope):
    """
    Compute a 1-D pseudo-envelope from canonical GreedyCosine merge similarity.

    Args:
        feature_extractor: FeatureExtractor that outputs frame features/states.
        norm_threshold: Norm threshold for canonical segmentation (default: 2.6).
        merge_threshold: Merge threshold for canonical segmentation (default: 0.8).
        aggregation_method: One of:
            - 'merge_similarity': raw phase-1 cosine similarity trace
            - 'merge_similarity_smoothed': smoothed similarity trace
        normalize: Whether to normalize to [0, 1].
        smooth_frames: Smoothing half-width for 'merge_similarity_smoothed'.

    Notes:
        - This is a pseudo-envelope view of canonical phase-1 internals.
        - Default thresholds align with original Sylber implementation.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        norm_threshold: float = 2.6,
        merge_threshold: float = 0.8,
        aggregation_method: str = 'merge_similarity',
        normalize: bool = True,
        smooth_frames: int = 1,
    ):
        super().__init__(feature_extractor=feature_extractor, normalize=normalize)
        self.norm_threshold = norm_threshold
        self.merge_threshold = merge_threshold
        self.aggregation_method = aggregation_method
        self.smooth_frames = smooth_frames

        if aggregation_method not in {'merge_similarity', 'merge_similarity_smoothed'}:
            raise ValueError(
                "aggregation_method must be one of {'merge_similarity', 'merge_similarity_smoothed'}"
            )

    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pseudo-envelope from canonical phase-1 merge-similarity trace.

        Returns:
            envelope: Shape [num_frames]
            times: Shape [num_frames], in seconds
        """
        features = self.feature_extractor.extract(audio, sr)
        num_frames = features.shape[0]

        envelope = compute_greedy_cosine_merge_similarity_trace(
            features,
            norm_threshold=self.norm_threshold,
            merge_threshold=self.merge_threshold,
        )

        if self.aggregation_method == 'merge_similarity_smoothed' and self.smooth_frames > 0:
            radius = int(self.smooth_frames)
            x = np.arange(-radius, radius + 1, dtype=np.float32)
            sigma = max(1.0, float(self.smooth_frames))
            kernel = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
            kernel /= kernel.sum() + 1e-8
            envelope = np.convolve(envelope, kernel, mode='same').astype(np.float32)

        envelope = self._normalize_envelope(envelope)

        times = self._frame_times(num_frames)

        return envelope.astype(np.float32), times.astype(np.float32)

    def __repr__(self):
        return (
            f"GreedyCosineEnvelope("
            f"feature_extractor={self.feature_extractor.__class__.__name__}, "
            f"norm_threshold={self.norm_threshold}, "
            f"merge_threshold={self.merge_threshold}, "
            f"aggregation_method={self.aggregation_method}, "
            f"normalize={self.normalize}, "
            f"smooth_frames={self.smooth_frames})"
        )


__all__ = ['GreedyCosineEnvelope']
