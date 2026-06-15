"""
Local Cosine Coherence Envelope (pseudo-envelope)

Computes a 1-D envelope as each frame's cosine similarity to the mean of its
local temporal neighborhood (a fixed ±window prototype, excluding the center
frame). The result is a smooth, segmentation-agnostic local-coherence cue --
high within steady regions (syllable nuclei), dipping at acoustic transitions
-- intended to be fed to ``peakdetect`` in place of a classical amplitude
envelope.

This is the *local* counterpart of ``SSMEnvelopeComputer``'s *global*
coherence (each frame's mean similarity to all frames). Like SSM, it derives
an INDEPENDENT reduction of the shared feature substrate; it deliberately does
NOT reuse the GreedyCosine segmenter's internal merge-similarity signal. That
segmenter-side signal lives in
``segmentation/greedy_cosine.compute_greedy_cosine_merge_similarity_trace``
and remains available there for diagnostics.

The name ``GreedyCosineEnvelope`` is retained as a backward-compatible alias.
"""

import numpy as np
from typing import Tuple

from .base import EnvelopeComputer
from ..features.base import FeatureExtractor


class LocalCosineEnvelope(EnvelopeComputer):
    """Local-window cosine-coherence pseudo-envelope.

    For each frame, build a local prototype as the mean of the surrounding
    ``±window_size`` frames (excluding the frame itself) and score the frame by
    its cosine similarity to that prototype. High where a frame matches its
    neighborhood (stable region); low at transitions.

    Args:
        feature_extractor: FeatureExtractor producing (N, D) frame features.
        window_size: Half-width, in frames, of the local prototype window
            (default: 5). The prototype excludes the center frame.
        normalize: Whether to min-max normalize the trace to [0, 1] (default: True).
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        window_size: int = 5,
        normalize: bool = True,
    ):
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.normalize = normalize

    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (envelope, times) for the local cosine-coherence trace.

        Vectorized over frames (prefix-sum sliding-window mean in float64);
        bit-equivalent to the per-frame reference pinned in
        ``tests/test_local_cosine_envelope.py``.
        """
        features = self.feature_extractor.extract(audio, sr)
        N = features.shape[0]
        eps = 1e-8
        w = self.window_size

        if N == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if N == 1:
            return np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

        feats = features.astype(np.float64)
        idx = np.arange(N)
        starts = np.maximum(0, idx - w)
        ends = np.minimum(N, idx + w + 1)

        # Prefix sums -> windowed sum; drop the center frame to form each frame's
        # local prototype (mean of its ±window_size neighbors, center excluded).
        cs = np.zeros((N + 1, feats.shape[1]), dtype=np.float64)
        np.cumsum(feats, axis=0, out=cs[1:])
        window_sum = cs[ends] - cs[starts]
        proto_count = ends - starts - 1  # frames in the prototype (center excluded)
        denom = np.maximum(proto_count, 1).astype(np.float64)
        proto = (window_sum - feats) / denom[:, None]

        dot = np.einsum("ij,ij->i", feats, proto)
        fn = np.sqrt((feats ** 2).sum(axis=1) + eps)
        pn = np.sqrt((proto ** 2).sum(axis=1) + eps)
        envelope = dot / (fn * pn)
        envelope[proto_count <= 0] = 1.0  # frames with no neighbors (e.g. window_size=0)

        if self.normalize:
            lo, hi = envelope.min(), envelope.max()
            if hi > lo:
                envelope = (envelope - lo) / (hi - lo)

        duration = len(audio) / sr
        times = np.linspace(0, duration, N)
        return envelope.astype(np.float32), times.astype(np.float32)

    def __repr__(self):
        return (
            f"LocalCosineEnvelope("
            f"feature_extractor={self.feature_extractor.__class__.__name__}, "
            f"window_size={self.window_size}, "
            f"normalize={self.normalize})"
        )


# Backward-compatible alias (the cue was historically exposed under this name).
GreedyCosineEnvelope = LocalCosineEnvelope


__all__ = ['LocalCosineEnvelope', 'GreedyCosineEnvelope']
