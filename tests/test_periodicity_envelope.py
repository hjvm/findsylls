"""Regression tests for PeriodicityEnvelope (Xie & Niyogi 2006, Eq. 3).

Real audio (test_samples/SP20_117.wav). Pins the normalized-autocorrelation
formula, the "largest value among the peaks" selection, and the absolute
(unnormalized) default so the measure cannot silently drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import PeriodicityEnvelope, get_envelope_computer

_AUDIO = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(
    not _AUDIO.exists(), reason="test_samples/SP20_117.wav not found"
)


@pytest.fixture(scope="module")
def audio():
    a, sr = load_audio(str(_AUDIO), samplerate=16000)
    return np.asarray(a, dtype=np.float64), sr


def _ref_frame(frame: np.ndarray, h_min: int, h_max: int) -> float:
    """Independent per-lag reference: Eq. (3) + largest local maximum."""
    n = len(frame)
    r0 = float(np.dot(frame, frame))
    if r0 <= 0.0:
        return 0.0
    h_hi = min(h_max, n - 1)
    if h_min >= h_hi:
        return 0.0
    p = np.array(
        [float(np.dot(frame[h:], frame[: n - h])) * n / ((n - h) * r0)
         for h in range(h_min, h_hi + 1)]
    )
    if p.size >= 3:
        pk = np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
        return float(p[pk].max()) if pk.size else float(p.max())
    return float(p.max())


def test_matches_eq3_reference(audio):
    """compute() equals an independent per-lag Eq. (3) reference (formula locked)."""
    a, sr = audio
    env = PeriodicityEnvelope()
    e, times = env.compute(a, sr)
    assert e.shape == times.shape

    h_min = max(1, int(round(2.5 * sr / 1000)))
    h_max = int(round(15.0 * sr / 1000))
    n = env.frame_size
    starts = list(range(0, len(a) - n + 1, env.frame_shift))
    ref = np.array([_ref_frame(a[s:s + n], h_min, h_max) for s in starts], dtype=np.float32)

    assert e.shape == ref.shape
    np.testing.assert_allclose(e, ref, atol=1e-5)


def test_absolute_by_default_and_finite(audio):
    """Default is unnormalized (absolute); values finite and non-negative."""
    a, sr = audio
    assert PeriodicityEnvelope().normalize is False
    e, _ = PeriodicityEnvelope().compute(a, sr)
    assert np.all(np.isfinite(e)) and float(e.min()) >= 0.0
    # a real utterance has strongly-periodic voiced frames
    assert float(e.max()) > 0.8


def test_discriminates_voiced_from_unvoiced(audio):
    """Periodicity separates frame types: wide spread across the utterance."""
    a, sr = audio
    e, _ = PeriodicityEnvelope().compute(a, sr)
    spread = float(np.percentile(e, 90)) - float(np.percentile(e, 10))
    assert spread > 0.3


def test_normalize_rescales_to_unit_max(audio):
    a, sr = audio
    e, _ = PeriodicityEnvelope(normalize=True).compute(a, sr)
    assert np.isclose(float(e.max()), 1.0, atol=1e-6)


def test_dispatch_and_defaults():
    """Reachable via the dispatch string with paper-Table-1 defaults."""
    seg = get_envelope_computer("periodicity")
    assert type(seg).__name__ == "PeriodicityEnvelope"
    p = PeriodicityEnvelope()
    assert p.frame_size == 400 and p.frame_shift == 160
