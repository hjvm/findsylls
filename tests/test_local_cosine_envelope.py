"""Regression tests for the local-window cosine-coherence LocalCosineEnvelope.

Pins the published modular-recombination cue (the "Cos. Sim. -> peakdetect"
rows of Table 2) so it cannot silently drift back to the segmenter's internal
merge-similarity signal — the regression introduced in the v3.0.0 rebuild.

Uses real audio (test_samples/SP20_117.wav) and MFCC features (no model
download); the envelope algorithm is feature-agnostic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import LocalCosineEnvelope
from findsylls.features import MFCCExtractor

_AUDIO_PATH = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(
    not _AUDIO_PATH.exists(), reason="test_samples/SP20_117.wav not found"
)


@pytest.fixture(scope="module")
def features():
    audio, sr = load_audio(str(_AUDIO_PATH), samplerate=16000)
    feats = MFCCExtractor(n_mfcc=13).extract(audio, sr)
    return audio, sr, feats


def _reference_local_window(features: np.ndarray, w: int = 5) -> np.ndarray:
    """Verbatim definition of the local-window cosine-coherence trace."""
    N = features.shape[0]
    env = np.zeros(N, dtype=np.float64)
    eps = 1e-8
    for i in range(N):
        start, end = max(0, i - w), min(N, i + w + 1)
        local = features[start:end]
        mask = np.ones(len(local), dtype=bool)
        mask[i - start] = False
        if mask.sum() > 0:
            proto = local[mask].mean(axis=0)
            frame = features[i]
            dot = float((frame * proto).sum())
            fn = (float((frame ** 2).sum()) + eps) ** 0.5
            pn = (float((proto ** 2).sum()) + eps) ** 0.5
            env[i] = dot / (fn * pn)
        else:
            env[i] = 1.0
    lo, hi = env.min(), env.max()
    if hi > lo:
        env = (env - lo) / (hi - lo)
    return env


def test_envelope_matches_local_window_reference(features):
    """The envelope IS the local-window cosine reduction (algorithm locked)."""
    audio, sr, _ = features
    env, times = LocalCosineEnvelope(MFCCExtractor(n_mfcc=13), window_size=5).compute(audio, sr)
    ref = _reference_local_window(MFCCExtractor(n_mfcc=13).extract(audio, sr), w=5)
    assert env.shape == times.shape == ref.shape
    np.testing.assert_allclose(env, ref, atol=1e-5)


def test_envelope_is_independent_of_segmenter_merge_similarity(features):
    """The cue must NOT be the segmenter's internal merge-similarity signal.

    This is the exact property that regressed in v3.0.0: the envelope delegated
    to compute_greedy_cosine_merge_similarity_trace, collapsing two distinct
    signals into one and degrading the recombination's boundary/span F1.
    """
    audio, sr, feats = features
    from findsylls.segmentation.greedy_cosine import (
        compute_greedy_cosine_merge_similarity_trace,
    )

    env, _ = LocalCosineEnvelope(MFCCExtractor(n_mfcc=13), window_size=5).compute(audio, sr)
    merge_sim = compute_greedy_cosine_merge_similarity_trace(
        feats, norm_threshold=2.6, merge_threshold=0.8
    )
    assert env.shape == merge_sim.shape
    # Same length, fundamentally different trace.
    assert not np.allclose(env, merge_sim, atol=1e-3)


def test_window_size_changes_the_trace(features):
    """Larger neighborhoods smooth the cue — window_size is a live parameter."""
    audio, sr, _ = features
    e1, _ = LocalCosineEnvelope(MFCCExtractor(n_mfcc=13), window_size=2).compute(audio, sr)
    e2, _ = LocalCosineEnvelope(MFCCExtractor(n_mfcc=13), window_size=15).compute(audio, sr)
    assert not np.allclose(e1, e2, atol=1e-3)


def test_greedycosine_alias_is_local_cosine():
    """The legacy GreedyCosineEnvelope name stays importable and identical."""
    from findsylls.envelope import GreedyCosineEnvelope as PkgAlias
    from findsylls.envelope.greedy_cosine import GreedyCosineEnvelope as ShimAlias
    assert PkgAlias is LocalCosineEnvelope
    assert ShimAlias is LocalCosineEnvelope
