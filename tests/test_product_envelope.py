"""Tests for multiplicative envelope gating (ProductEnvelope, ThresholdGate)."""
from __future__ import annotations

import numpy as np

from findsylls.envelope import ProductEnvelope, ThresholdGate
from findsylls.envelope.base import EnvelopeComputer


class _Const(EnvelopeComputer):
    """Fixed (envelope, times) for deterministic testing on a chosen grid."""

    def __init__(self, env, times):
        self.env = np.asarray(env, float)
        self.times = np.asarray(times, float)

    def compute(self, audio, sr):
        return self.env, self.times


def test_threshold_gate_masks():
    inner = _Const([0.2, 0.8, 0.9, 0.1], [0, 1, 2, 3])
    mask, t = ThresholdGate(inner, 0.5).compute(None, 16000)
    assert list(mask) == [0.0, 1.0, 1.0, 0.0]
    assert list(t) == [0, 1, 2, 3]


def test_product_gates_primary():
    energy = _Const([1.0, 2.0, 3.0, 4.0], [0, 1, 2, 3])
    gate = ThresholdGate(_Const([0.0, 1.0, 1.0, 0.0], [0, 1, 2, 3]), 0.5)
    out, t = ProductEnvelope([energy, gate]).compute(None, 16000)
    assert list(out) == [0.0, 2.0, 3.0, 0.0]
    assert list(t) == [0, 1, 2, 3]


def test_weight_zero_disables_signal():
    energy = _Const([1.0, 2.0, 3.0], [0, 1, 2])
    gate = ThresholdGate(_Const([0.0, 0.0, 1.0], [0, 1, 2]), 0.5)
    # weight 0 on the gate => env ** 0 == 1 => gate ignored, energy passes through
    out, _ = ProductEnvelope([energy, gate], weights=[1.0, 0.0]).compute(None, 16000)
    assert list(out) == [1.0, 2.0, 3.0]


def test_weight_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        ProductEnvelope([_Const([1.0], [0])], weights=[1.0, 2.0])


def test_product_resamples_onto_primary_grid():
    # gate lives on a coarser grid; must be interpolated onto energy's grid
    energy = _Const([1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.5, 2.0])
    gate = _Const([0.0, 1.0, 0.0], [0.0, 1.0, 2.0])
    out, t = ProductEnvelope([energy, gate]).compute(None, 16000)
    assert len(out) == 5  # primary grid preserved
    assert out[2] == 1.0  # gate peak at t=1.0
    assert out[0] == 0.0 and out[4] == 0.0
