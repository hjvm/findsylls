"""Tests for the RMS dB/relative-energy option and the promoted gammatone
filterbank envelope. Real audio (test_samples/SP20_117.wav)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import RMSEnvelope, GammatoneEnvelope, get_envelope_computer
from findsylls.envelope.gammatone import gammatone_filterbank

_AUDIO = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"
pytestmark = pytest.mark.skipif(not _AUDIO.exists(), reason="test_samples/SP20_117.wav not found")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(_AUDIO), samplerate=16000)


# ── RMS relevant-energy (dB below max) ──────────────────────────────────────

def test_rms_linear_default_unchanged(audio):
    a, sr = audio
    assert RMSEnvelope().db is False
    e, _ = RMSEnvelope().compute(a, sr)
    assert np.all(e >= 0.0)  # linear RMS is non-negative


def test_rms_db_is_relative_to_max(audio):
    a, sr = audio
    lin, _ = RMSEnvelope(db=False).compute(a, sr)
    db, _ = RMSEnvelope(db=True, reference="max").compute(a, sr)
    assert np.isclose(db.max(), 0.0, atol=1e-6)   # loudest frame == 0 dB
    assert np.all(db <= 1e-6)                       # everything else below it
    # matches 20*log10(rms / rms.max()) on the non-silent frames
    ref = max(float(lin.max()), 1e-10)
    expected = 20.0 * np.log10(np.maximum(lin, 1e-10) / ref)
    np.testing.assert_allclose(db, expected, atol=1e-5)


# ── Gammatone filterbank + reductions ───────────────────────────────────────

def test_filterbank_is_multiband(audio):
    a, sr = audio
    fb, times = gammatone_filterbank(a, sr, bands=20)
    assert fb.ndim == 2 and fb.shape[0] == 20
    assert times.shape[0] == fb.shape[1]
    assert fb.min() > -0.01  # Hilbert magnitude (tiny resample overshoot allowed)


def test_reductions_differ_and_are_1d(audio):
    a, sr = audio
    e_sum, t = GammatoneEnvelope(reduction="sum").compute(a, sr)
    e_mean, _ = GammatoneEnvelope(reduction="mean").compute(a, sr)
    e_norm, _ = GammatoneEnvelope(reduction="normalized_sum").compute(a, sr)
    for e in (e_sum, e_mean, e_norm):
        assert e.ndim == 1 and e.shape[0] == t.shape[0]
    assert not np.allclose(e_sum, e_norm)  # Zhang's normalized_sum != plain sum
    with pytest.raises(ValueError):
        GammatoneEnvelope(reduction="bogus").compute(a, sr)


def test_gammatone_is_a_dispatch_method(audio):
    a, sr = audio
    seg = get_envelope_computer("gammatone")
    assert type(seg).__name__ == "GammatoneEnvelope"
    env, times = seg.compute(a, sr)
    assert env.ndim == 1 and env.shape == times.shape


# (filterbank↔Python-port parity is guarded by
#  tests/test_theta_parity_regression.py::TestGammatoneParity, which uses the
#  actual reference implementation — no need to duplicate it here.)
