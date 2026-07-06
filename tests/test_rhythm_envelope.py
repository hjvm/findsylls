"""Tests for RhythmEnvelope / fit_rhythm_sinusoid (Zhang & Glass 2009).

Real audio only (test_samples/): the rhythm sinusoid is fitted to first-pass
peaks detected on actual speech.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import GammatoneEnvelope, RhythmEnvelope, fit_rhythm_sinusoid
from findsylls.segmentation import segment_peakdetect

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(SAMPLE))


@pytest.fixture(scope="module")
def speech_peaks(audio):
    """First-pass peak train from the real ERB Hilbert-sum envelope."""
    a, sr = audio
    env, times = GammatoneEnvelope(reduction="normalized_sum").compute(a, sr)
    env = np.asarray(env, float)
    norm = (env - env.min()) / (env.max() - env.min())
    segs = segment_peakdetect(norm, times, delta=0.05, add_boundary_valleys=True)
    return np.array([p for _, p, _ in segs])


def test_fit_recovers_speech_rate_from_real_peaks(speech_peaks):
    assert speech_peaks.size >= 2
    k1, k2 = fit_rhythm_sinusoid(speech_peaks)
    period = 2 * np.pi / k1
    assert 0.1 <= period <= 0.5  # within the syllable-rate band
    assert 0.0 <= k2 < 1.0
    # crests must align with the peak train clearly better than chance: for a
    # random phase P(sin > 0.8) ~= 0.2, and real speech rhythm is irregular
    # (Zhang Fig. 3), so demand strong locking on a sizeable subset.
    crest = np.sin(k1 * speech_peaks + 2 * np.pi * k2)
    assert (crest > 0.8).mean() > 0.35
    assert np.median(crest) > 0.2


def test_fit_prefers_slow_rhythm_over_harmonic(speech_peaks):
    """The k1 tie-break must not jump to a harmonic: doubling grid resolution
    or widening the band must not halve the period."""
    k1_narrow, _ = fit_rhythm_sinusoid(speech_peaks, period_range=(0.1, 0.5))
    k1_wide, _ = fit_rhythm_sinusoid(speech_peaks, period_range=(0.05, 0.5), n_freq=400)
    assert 2 * np.pi / k1_wide >= 0.5 * (2 * np.pi / k1_narrow)


def test_fit_needs_two_peaks(speech_peaks):
    with pytest.raises(ValueError):
        fit_rhythm_sinusoid(speech_peaks[:1])


def test_weight_output_bounds(audio):
    a, sr = audio
    w, t = RhythmEnvelope(output="weight", floor=0.3).compute(a, sr)
    assert w.shape == t.shape
    assert w.min() >= 0.3 - 1e-6 and w.max() <= 1.0 + 1e-6
    assert w.max() - w.min() > 0.1  # a real fit modulates, not flat


def test_weighted_output_is_normalized(audio):
    a, sr = audio
    env, t = RhythmEnvelope(output="weighted", normalize=True).compute(a, sr)
    assert env.shape == t.shape
    assert env.min() == pytest.approx(0.0, abs=1e-6)
    assert env.max() == pytest.approx(1.0, abs=1e-6)


def test_bad_output_raises():
    with pytest.raises(ValueError):
        RhythmEnvelope(output="bogus")
