"""Tests for RhythmGuidedSegmenter (Zhang & Glass 2009). Real audio only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.segmentation import RhythmGuidedSegmenter

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(SAMPLE))


def test_tuned_defaults_plausible_rate_and_valid(audio):
    a, sr = audio
    out = RhythmGuidedSegmenter().segment(a, sr)
    assert len(out) > 0
    rate = len(out) / (len(a) / sr)
    assert 1.5 <= rate <= 8.0, f"nucleus rate {rate:.1f}/s implausible"
    for s, p, e in out:
        assert s <= p <= e


def test_voicing_gate_suppresses_unvoiced_nuclei(audio):
    """Every nucleus must sit in a voiced frame (periodicity >= threshold)."""
    from findsylls.envelope import PeriodicityEnvelope

    a, sr = audio
    seg = RhythmGuidedSegmenter(voicing_threshold=0.4)
    out = seg.segment(a, sr)
    penv, ptimes = PeriodicityEnvelope().compute(a, sr)
    penv = np.asarray(penv, float)
    for _, p, _ in out:
        assert penv[np.argmin(np.abs(ptimes - p))] >= 0.4


def test_rhythm_ablation_changes_output(audio):
    """rhythm_floor=1.0 (the paper's nRG) must differ from the rhythm-guided
    run — proving the rhythm weight actually participates."""
    a, sr = audio
    rg = RhythmGuidedSegmenter(rhythm_floor=0.3).segment(a, sr)
    nrg = RhythmGuidedSegmenter(rhythm_floor=1.0).segment(a, sr)
    rg_peaks = np.array([p for _, p, _ in rg])
    nrg_peaks = np.array([p for _, p, _ in nrg])
    assert rg_peaks.size != nrg_peaks.size or not np.allclose(
        np.sort(rg_peaks), np.sort(nrg_peaks)
    )


def test_reference_and_cite(capsys):
    seg = RhythmGuidedSegmenter()
    assert "Zhang" in seg.REFERENCE and "2009" in seg.REFERENCE
    seg.cite()
    assert "Zhang" in capsys.readouterr().out
