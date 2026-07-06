"""Tests for the Mermelstein convex-hull segmentation primitive."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findsylls.segmentation import segment_convexhull, ConvexHullSegmenter, get_segmenter

SAMPLE = Path(__file__).resolve().parent.parent / "test_samples" / "SP20_117.wav"


def _two_bump_trace(dip_value):
    """Two Gaussian bumps (peaks) separated by a valley of depth 1 - dip_value."""
    t = np.linspace(0, 1, 200)
    trace = np.exp(-((t - 0.25) ** 2) / 0.005) + np.exp(-((t - 0.75) ** 2) / 0.005)
    # valley floor sits near dip_value at t=0.5
    trace = trace + dip_value
    return trace, t


def test_splits_deep_dip_into_two():
    trace, t = _two_bump_trace(0.1)
    segs = segment_convexhull(trace, t, peak_to_dip=0.5, min_syllable_dur=0.0)
    assert len(segs) == 2
    # peaks land near the two bump centres
    peaks = sorted(p for _, p, _ in segs)
    assert abs(peaks[0] - 0.25) < 0.05
    assert abs(peaks[1] - 0.75) < 0.05


def test_threshold_gates_splitting():
    trace, t = _two_bump_trace(0.1)
    # valley is ~1.0 deep; a threshold above that must NOT split
    segs = segment_convexhull(trace, t, peak_to_dip=5.0, min_syllable_dur=0.0)
    assert len(segs) == 1


def test_min_duration_drops_short_segments():
    trace, t = _two_bump_trace(0.1)
    segs = segment_convexhull(trace, t, peak_to_dip=0.5, min_syllable_dur=10.0)
    assert segs == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        segment_convexhull(np.zeros(10), np.zeros(9))


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav missing")
def test_real_audio_smoke():
    from findsylls.audio.utils import load_audio

    audio, sr = load_audio(str(SAMPLE))
    seg = get_segmenter("convexhull", envelope_method="sbs", cache=False)
    out = seg.segment(audio=audio, sr=sr)
    assert isinstance(out, list) and len(out) > 0
    for s, p, e in out:
        assert s <= p <= e
