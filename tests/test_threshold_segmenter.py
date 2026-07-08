"""Tests for the ThresholdSegmenter primitive. Real audio only (test_samples/)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import PeriodicityEnvelope
from findsylls.segmentation import segment_threshold, ThresholdSegmenter, get_segmenter

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def audio():
    return load_audio(str(SAMPLE))


def test_mask_and_regions_are_equivalent(audio):
    """The dense mask and the sparse regions carry the same information: rebuild
    the mask from the regions and it must match."""
    a, sr = audio
    ts = ThresholdSegmenter(PeriodicityEnvelope(400, 160), threshold=0.4)
    mask, t = ts.mask(a, sr)
    regions = ts.segment(audio=a, sr=sr)

    rebuilt = np.zeros_like(mask, dtype=bool)
    for start, _, end in regions:
        rebuilt[(t >= start) & (t <= end)] = True
    # regions cover exactly the on-frames (endpoints inclusive)
    assert rebuilt.sum() == int(mask.sum())


def test_regions_are_above_threshold(audio):
    a, sr = audio
    per = PeriodicityEnvelope(400, 160)
    env, t = per.compute(a, sr)
    env = np.asarray(env, float)
    for start, peak, end in segment_threshold(env, t, threshold=0.5):
        sel = (t >= start) & (t <= end)
        assert env[sel].max() >= 0.5            # region actually clears the bar
        assert env[np.argmin(np.abs(t - peak))] == pytest.approx(env[sel].max())


def test_higher_threshold_is_subset(audio):
    a, sr = audio
    env, t = PeriodicityEnvelope(400, 160).compute(a, sr)
    env = np.asarray(env, float)
    n_lo = sum(1 for _ in segment_threshold(env, t, threshold=0.3))
    n_hi = sum(1 for _ in segment_threshold(env, t, threshold=0.8))
    on_lo = (env >= 0.3).sum()
    on_hi = (env >= 0.8).sum()
    assert on_hi <= on_lo   # stricter threshold admits no more frames


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        segment_threshold(np.zeros(10), np.zeros(9))


def test_dispatch(audio):
    a, sr = audio
    seg = get_segmenter("threshold", cache=False, envelope_method="rms",
                        envelope_kwargs={"db": True, "reference": "max"}, threshold=-40.0)
    out = seg.segment(audio=a, sr=sr)
    assert isinstance(out, list) and len(out) > 0
    for s, p, e in out:
        assert s <= p <= e
