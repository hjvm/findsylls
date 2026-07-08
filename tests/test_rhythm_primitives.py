"""Tests for the rhythm segmentation primitives (Zhang & Glass 2009):
fit_rhythm_sinusoid (seeded LMS) and rhythm_crests. Real audio for the fit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import GammatoneEnvelope
from findsylls.segmentation import segment_peakdetect, fit_rhythm_sinusoid, rhythm_crests

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def speech_peaks():
    """Real first-pass peaks from the ERB Hilbert-sum envelope."""
    audio, sr = load_audio(str(SAMPLE))
    env, t = GammatoneEnvelope(reduction="normalized_sum").compute(audio, sr)
    env = np.asarray(env, float)
    norm = (env - env.min()) / (env.max() - env.min())
    return np.array([p for _, p, _ in segment_peakdetect(norm, t, delta=0.15,
                                                         add_boundary_valleys=True)])


def test_fit_from_two_peaks_seeds_at_default():
    """Seeded LMS must fit from just TWO peaks (no 'wait for N') and stay near a
    sensible period, not rail to the bounds."""
    k1, k2 = fit_rhythm_sinusoid([0.2, 0.4], seed_period=0.20)
    period = 2 * np.pi / k1
    assert 0.06 <= period <= 0.6            # inside bounds, not railed
    assert 0.0 <= k2 < 1.0


def test_fit_recovers_regular_rhythm():
    """Peaks at a fixed 220 ms spacing -> period ~220 ms."""
    peaks = 0.11 + 0.22 * np.arange(8)
    k1, _ = fit_rhythm_sinusoid(peaks, seed_period=0.20)
    assert abs(2 * np.pi / k1 - 0.22) < 0.03


def test_fit_on_real_speech_is_in_syllable_range(speech_peaks):
    assert speech_peaks.size >= 2
    k1, k2 = fit_rhythm_sinusoid(speech_peaks, seed_period=0.20)
    assert 0.08 <= 2 * np.pi / k1 <= 0.5
    # crests should align with the peak train better than chance
    crest = np.sin(k1 * speech_peaks + 2 * np.pi * k2)
    assert np.median(crest) > 0.0


def test_fit_needs_two_peaks():
    with pytest.raises(ValueError):
        fit_rhythm_sinusoid([0.2])


def test_crests_are_spaced_by_one_period():
    k1, k2 = fit_rhythm_sinusoid([0.2, 0.4, 0.6, 0.8], seed_period=0.20)
    c = rhythm_crests(k1, k2, 0.0, 1.0)
    assert c.size >= 3
    gaps = np.diff(c)
    np.testing.assert_allclose(gaps, 2 * np.pi / k1, atol=1e-6)
    # each crest is actually a crest (sin ~ 1)
    assert np.allclose(np.sin(k1 * c + 2 * np.pi * k2), 1.0, atol=1e-6)
