"""Tests for two-stage gated segmentation (RegionGatedSegmenter,
EnergyPeriodicitySegmenter — Xie & Niyogi 2006).

Real audio only (test_samples/).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import PeriodicityEnvelope, RMSEnvelope, ThresholdGate
from findsylls.segmentation import RegionGatedSegmenter, EnergyPeriodicitySegmenter

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(SAMPLE))


def test_nuclei_only_inside_gate_on_regions(audio):
    """Every picked nucleus must lie inside a voiced (gate-on) region — the
    defining property of slice-and-pick."""
    a, sr = audio
    per = PeriodicityEnvelope()
    energy = RMSEnvelope(frame_length=400, hop_length=160, db=True, reference="max")
    seg = RegionGatedSegmenter(primary_env=energy,
                               gate_env=ThresholdGate(per, 0.5),
                               peak_to_dip=4.5, min_syllable_dur=0.05)
    out = seg._segment(a, sr)
    assert len(out) > 0

    penv, ptimes = per.compute(a, sr)
    penv = np.asarray(penv, float)
    for _, peak, _ in out:
        # periodicity at the nucleus must be at/above the gate threshold
        assert penv[np.argmin(np.abs(ptimes - peak))] >= 0.5


def test_impossible_gate_yields_no_segments(audio):
    a, sr = audio
    energy = RMSEnvelope(frame_length=400, hop_length=160, db=True, reference="max")
    # periodicity can never exceed ~1, so a gate at 5.0 is never on
    seg = RegionGatedSegmenter(primary_env=energy,
                               gate_env=ThresholdGate(PeriodicityEnvelope(), 5.0),
                               min_syllable_dur=0.0)
    assert seg._segment(a, sr) == []


def test_gating_restricts_relative_to_ungated(audio):
    """Slice-and-pick must produce nuclei only where the gate allows — i.e., a
    subset of the utterance, unlike an ungated convex-hull over everything."""
    a, sr = audio
    per = PeriodicityEnvelope()
    energy = RMSEnvelope(frame_length=400, hop_length=160, db=True, reference="max")
    strict = RegionGatedSegmenter(primary_env=energy,
                                  gate_env=ThresholdGate(per, 0.8),
                                  peak_to_dip=4.5, min_syllable_dur=0.05)
    loose = RegionGatedSegmenter(primary_env=energy,
                                 gate_env=ThresholdGate(per, 0.2),
                                 peak_to_dip=4.5, min_syllable_dur=0.05)
    assert len(strict._segment(a, sr)) <= len(loose._segment(a, sr))


@pytest.mark.parametrize("composition,ptd", [("slice", 4.5), ("product", 0.05)])
def test_energy_periodicity_real_audio(audio, composition, ptd):
    a, sr = audio
    seg = EnergyPeriodicitySegmenter(composition=composition, energy_peak_to_dip=ptd)
    out = seg.segment(a, sr)
    assert len(out) > 0
    for s, p, e in out:
        assert s <= p <= e


def test_tuned_defaults_give_plausible_syllable_rate(audio):
    """Pins the TIMIT-tuned defaults (pt=0.3, floor=-30, ptd=4.5): the detected
    nucleus rate on real speech must stay in the plausible syllable range."""
    a, sr = audio
    out = EnergyPeriodicitySegmenter().segment(a, sr)
    rate = len(out) / (len(a) / sr)
    assert 1.5 <= rate <= 8.0, f"nucleus rate {rate:.1f}/s outside plausible range"


def test_bad_composition_raises():
    with pytest.raises(ValueError):
        EnergyPeriodicitySegmenter(composition="bogus")
