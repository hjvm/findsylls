"""Tests for multiplicative envelope gating (ProductEnvelope, ThresholdGate).

Real audio only (test_samples/): all properties are checked on envelopes
computed from actual speech.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import (
    PeriodicityEnvelope,
    ProductEnvelope,
    RMSEnvelope,
    ThresholdGate,
)

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(SAMPLE))


def test_threshold_gate_is_exact_mask(audio):
    a, sr = audio
    per = PeriodicityEnvelope()
    env, times = per.compute(a, sr)
    mask, mtimes = ThresholdGate(per, 0.7).compute(a, sr)
    assert np.array_equal(mtimes, times)
    assert np.array_equal(mask, (np.asarray(env, float) >= 0.7).astype(np.float32))
    assert 0 < mask.sum() < mask.size  # real speech has voiced AND unvoiced frames


def test_product_gates_primary_same_grid(audio):
    """RMS(400/160) and periodicity(400/160) share a hop; gating must zero
    exactly the sub-threshold frames and preserve the rest."""
    a, sr = audio
    rms = RMSEnvelope(frame_length=400, hop_length=160)
    gate = ThresholdGate(PeriodicityEnvelope(), 0.7)
    prim, ptimes = rms.compute(a, sr)
    mask, mtimes = gate.compute(a, sr)
    out, otimes = ProductEnvelope([rms, gate]).compute(a, sr)

    assert np.array_equal(otimes, ptimes)  # primary defines the grid
    aligned = np.interp(ptimes, mtimes, mask)
    expected = np.asarray(prim, float) * aligned
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)
    assert (out == 0).any() and (out > 0).any()


def test_product_resamples_onto_primary_grid(audio):
    """Components on different grids (RMS hop 256 vs periodicity hop 160) must
    be interpolated onto the primary's grid."""
    a, sr = audio
    rms = RMSEnvelope(frame_length=1024, hop_length=256)
    gate = ThresholdGate(PeriodicityEnvelope(), 0.7)
    prim, ptimes = rms.compute(a, sr)
    out, otimes = ProductEnvelope([rms, gate]).compute(a, sr)
    assert out.shape == prim.shape
    assert np.array_equal(otimes, ptimes)
    assert np.all(out <= np.asarray(prim, float) + 1e-7)  # 0/1 gate only attenuates


def test_weight_zero_disables_signal(audio):
    a, sr = audio
    rms = RMSEnvelope(frame_length=400, hop_length=160)
    gate = ThresholdGate(PeriodicityEnvelope(), 0.7)
    prim, _ = rms.compute(a, sr)
    out, _ = ProductEnvelope([rms, gate], weights=[1.0, 0.0]).compute(a, sr)
    np.testing.assert_allclose(out, np.asarray(prim, np.float32), rtol=1e-6)


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        ProductEnvelope([RMSEnvelope()], weights=[1.0, 2.0])
