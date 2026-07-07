"""Tests for the two-stage EnergyPeriodicitySegmenter (Xie & Niyogi 2006).

Real audio only (test_samples/). The preset owns its two-convex-hull logic
directly (GreedyCosine-style); there is no separate gating orchestrator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import PeriodicityEnvelope
from findsylls.segmentation import EnergyPeriodicitySegmenter

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(SAMPLE))


def test_nuclei_only_in_periodic_regions(audio):
    """Every nucleus must land where periodicity clears the region threshold —
    the defining property of stage-1 gating."""
    a, sr = audio
    thr = 0.5
    seg = EnergyPeriodicitySegmenter(periodicity_threshold=thr, energy_floor_db=None)
    out = seg.segment(a, sr)
    assert len(out) > 0

    penv, ptimes = PeriodicityEnvelope().compute(a, sr)
    penv = np.asarray(penv, float)
    for _, peak, _ in out:
        # the nucleus sits inside a region whose periodicity peak >= threshold;
        # periodicity at the nucleus itself is a lower bound on that peak
        # (energy peak need not coincide with periodicity peak), so assert the
        # region-level property: some nearby frame is voiced.
        near = np.abs(ptimes - peak) < 0.05
        assert penv[near].max() >= thr


def test_impossible_periodicity_threshold_yields_nothing(audio):
    a, sr = audio
    # periodicity never exceeds ~1, so a threshold of 5 admits no region
    seg = EnergyPeriodicitySegmenter(periodicity_threshold=5.0)
    assert seg._segment(a, sr) == []


def test_stricter_threshold_is_subset_like(audio):
    a, sr = audio
    loose = EnergyPeriodicitySegmenter(periodicity_threshold=0.3).segment(a, sr)
    strict = EnergyPeriodicitySegmenter(periodicity_threshold=0.8).segment(a, sr)
    assert len(strict) <= len(loose)


@pytest.mark.parametrize("composition,ptd", [("slice", 4.5), ("product", 0.05)])
def test_both_compositions_run(audio, composition, ptd):
    a, sr = audio
    seg = EnergyPeriodicitySegmenter(composition=composition, energy_peak_to_dip=ptd)
    out = seg.segment(a, sr)
    assert len(out) > 0
    for s, p, e in out:
        assert s <= p <= e


def test_tuned_defaults_give_plausible_syllable_rate(audio):
    a, sr = audio
    out = EnergyPeriodicitySegmenter().segment(a, sr)
    rate = len(out) / (len(a) / sr)
    assert 1.5 <= rate <= 8.0, f"nucleus rate {rate:.1f}/s outside plausible range"


def test_bad_composition_raises():
    with pytest.raises(ValueError):
        EnergyPeriodicitySegmenter(composition="bogus")
