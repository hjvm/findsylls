"""Tests for the Mermelstein convex-hull segmentation primitive.

Real audio only (test_samples/): properties are asserted on envelopes computed
from actual speech, not synthetic traces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.audio.utils import load_audio
from findsylls.envelope import SBSEnvelope
from findsylls.segmentation import segment_convexhull, get_segmenter

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")


@pytest.fixture(scope="module")
def sbs_trace():
    audio, sr = load_audio(str(SAMPLE))
    env, times = SBSEnvelope().compute(audio, sr)
    return np.asarray(env, float), np.asarray(times, float)


def test_segments_real_speech_into_syllable_range(sbs_trace):
    env, times = sbs_trace
    segs = segment_convexhull(env, times, peak_to_dip=0.05)
    dur = times[-1] - times[0]
    assert 1.5 <= len(segs) / dur <= 10.0  # plausible syllable rate on speech
    for s, p, e in segs:
        assert s <= p <= e


def test_peak_is_segment_argmax(sbs_trace):
    env, times = sbs_trace
    for s, p, e in segment_convexhull(env, times, peak_to_dip=0.05):
        sel = (times >= s) & (times <= e)
        assert p == pytest.approx(times[sel][np.argmax(env[sel])])


def test_threshold_monotonicity(sbs_trace):
    """A deeper required dip can only merge segments, never create more."""
    env, times = sbs_trace
    counts = [len(segment_convexhull(env, times, peak_to_dip=d))
              for d in (0.01, 0.05, 0.2, 1.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] <= 2  # near-impossible dip -> at most the whole utterance


def test_min_duration_drops_short_segments(sbs_trace):
    env, times = sbs_trace
    segs = segment_convexhull(env, times, peak_to_dip=0.05, min_syllable_dur=10.0)
    assert segs == []  # utterance is ~4 s; nothing can last 10 s


def test_length_mismatch_raises(sbs_trace):
    env, times = sbs_trace
    with pytest.raises(ValueError):
        segment_convexhull(env[:-1], times)


def test_dispatch_real_audio():
    audio, sr = load_audio(str(SAMPLE))
    seg = get_segmenter("convexhull", envelope_method="sbs", cache=False)
    out = seg.segment(audio=audio, sr=sr)
    assert isinstance(out, list) and len(out) > 0
    for s, p, e in out:
        assert s <= p <= e
