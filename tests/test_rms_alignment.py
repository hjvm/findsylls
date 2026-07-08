"""RMSEnvelope(center=False) must share PeriodicityEnvelope's frame grid.

Real audio only (test_samples/). This is the alignment the Xie detector relies
on: its energy cue (log gamma(0)) lives on the same frames as periodicity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import PeriodicityEnvelope, RMSEnvelope

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(SAMPLE))


def test_center_false_matches_periodicity_grid(audio):
    a, sr = audio
    _, per_t = PeriodicityEnvelope(frame_size=400, frame_shift=160).compute(a, sr)
    rms, rms_t = RMSEnvelope(frame_length=400, hop_length=160, center=False).compute(a, sr)
    assert len(rms) == len(per_t)
    np.testing.assert_allclose(rms_t, per_t, atol=1e-4)


def test_center_true_still_default_librosa_grid(audio):
    """center=True (default) keeps librosa's padded/centered grid, unchanged."""
    a, sr = audio
    _, per_t = PeriodicityEnvelope(frame_size=400, frame_shift=160).compute(a, sr)
    rms, rms_t = RMSEnvelope(frame_length=400, hop_length=160).compute(a, sr)
    # librosa center=True reflect-pads -> a different (longer, offset) grid
    assert len(rms) != len(per_t) or not np.allclose(rms_t, per_t, atol=1e-4)


def test_center_false_values_are_true_rms(audio):
    a, sr = audio
    rms, _ = RMSEnvelope(frame_length=400, hop_length=160, center=False).compute(a, sr)
    starts = range(0, len(a) - 400 + 1, 160)
    own = np.array([np.sqrt(np.mean(a[s:s + 400] ** 2)) for s in starts])
    np.testing.assert_allclose(rms, own, atol=1e-6)
