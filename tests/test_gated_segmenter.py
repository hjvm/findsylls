"""Tests for two-stage gated segmentation (RegionGatedSegmenter,
EnergyPeriodicitySegmenter — Xie & Niyogi 2006)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.envelope.base import EnvelopeComputer
from findsylls.segmentation import RegionGatedSegmenter, EnergyPeriodicitySegmenter

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"


class _Const(EnvelopeComputer):
    def __init__(self, env, times):
        self.env = np.asarray(env, float)
        self.times = np.asarray(times, float)

    def compute(self, audio, sr):
        return self.env, self.times


def test_region_gated_picks_energy_peak_inside_gate_only():
    # energy has a big peak at t=4 (index 4) but the gate is only ON for t in [1,3].
    t = np.arange(8, dtype=float)
    energy = _Const([0, 1, 5, 1, 9, 1, 0, 0], t)   # global max at index 4
    gate = _Const([0, 1, 1, 1, 0, 0, 0, 0], t)     # ON over indices 1..3
    seg = RegionGatedSegmenter(energy, gate, gate_on=0.5,
                               peak_to_dip=0.0, min_syllable_dur=0.0)
    out = seg._segment(None, 1)
    assert len(out) == 1
    # nucleus is the in-gate peak (index 2, value 5), NOT the bigger out-of-gate one
    assert abs(out[0][1] - 2.0) < 1e-6


def test_no_gate_no_segments():
    t = np.arange(5, dtype=float)
    energy = _Const([1, 2, 3, 2, 1], t)
    gate = _Const([0, 0, 0, 0, 0], t)
    seg = RegionGatedSegmenter(energy, gate, gate_on=0.5, min_syllable_dur=0.0)
    assert seg._segment(None, 1) == []


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")
@pytest.mark.parametrize("composition,ptd", [("slice", 4.5), ("product", 0.05)])
def test_energy_periodicity_real_audio(composition, ptd):
    from findsylls.audio.utils import load_audio

    audio, sr = load_audio(str(SAMPLE))
    seg = EnergyPeriodicitySegmenter(composition=composition, energy_peak_to_dip=ptd)
    out = seg.segment(audio, sr)
    assert len(out) > 0
    for s, p, e in out:
        assert s <= p <= e


def test_bad_composition_raises():
    with pytest.raises(ValueError):
        EnergyPeriodicitySegmenter(composition="bogus")
