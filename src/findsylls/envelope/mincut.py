"""MinCut pseudo-envelope computer.

Produces a 1-D pre-segmentation trace from canonical MinCut DP costs.
The trace is derived from MinCut's boundary scoring tables before final
quantile-based border decoding.
"""

import numpy as np
from typing import Tuple

from .base import PseudoEnvelope
from ..features.base import FeatureExtractor
from ..segmentation.mincut import MINCUT_THRESHOLD, extract_mincut_boundaries


class MinCutEnvelope(PseudoEnvelope):
    """Compute a 1-D pseudo-envelope from canonical MinCut DP costs.

    Args:
        feature_extractor: Feature extractor that outputs frame states/features.
        threshold: Canonical MinCut threshold used for max-unit computation.
        s: Maximum chunk size in frames.
        min_hop: Minimum chunk size in frames.
        aggregation_method: One of {'min_cost', 'mean_cost'}.
        normalize: Whether to min-max normalize to [0, 1].
        invert: If True, return 1 - normalized_trace (useful when plotting
            boundary salience as peaks rather than low-cost valleys).
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        threshold: float = MINCUT_THRESHOLD,
        s: int = 35,
        min_hop: int = 3,
        aggregation_method: str = 'min_cost',
        normalize: bool = True,
        invert: bool = False,
    ):
        super().__init__(feature_extractor=feature_extractor, normalize=normalize)
        self.threshold = threshold
        self.s = s
        self.min_hop = min_hop
        self.aggregation_method = aggregation_method
        self.invert = invert

        if aggregation_method not in {'min_cost', 'mean_cost'}:
            raise ValueError("aggregation_method must be one of {'min_cost', 'mean_cost'}")

    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a peak-friendly MinCut boundary trace and corresponding frame times."""
        features = self.feature_extractor.extract(audio, sr)
        num_frames = int(features.shape[0])

        boundaries = extract_mincut_boundaries(
            features,
            threshold=self.threshold,
            s=self.s,
            min_hop=self.min_hop,
        )

        envelope = np.zeros(num_frames, dtype=np.float32)
        if len(boundaries) > 2:
            impulse = np.zeros(num_frames, dtype=np.float32)
            for boundary in boundaries[1:-1]:
                if 0 <= boundary < num_frames:
                    impulse[boundary] = 1.0

            # Convert sparse border impulses into a peak-detectable boundary trace.
            kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float32)
            kernel /= kernel.sum() + 1e-8
            envelope = np.convolve(impulse, kernel, mode='same').astype(np.float32)

        envelope = self._normalize_envelope(envelope)
        if self.invert:
            envelope = 1.0 - envelope

        times = self._frame_times(num_frames)
        return envelope.astype(np.float32), times.astype(np.float32)

    def __repr__(self):
        return (
            f"MinCutEnvelope("
            f"feature_extractor={self.feature_extractor.__class__.__name__}, "
            f"threshold={self.threshold}, "
            f"s={self.s}, "
            f"min_hop={self.min_hop}, "
            f"aggregation_method={self.aggregation_method}, "
            f"normalize={self.normalize}, "
            f"invert={self.invert})"
        )


__all__ = ['MinCutEnvelope']
