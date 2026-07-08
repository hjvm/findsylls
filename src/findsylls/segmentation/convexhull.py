"""Mermelstein (1975) convex-hull segmentation.

The recursive convex-hull dip procedure that underlies the classic
energy/loudness syllabification tradition (Mermelstein 1975; and the
segmentation stage of Xie & Niyogi 2006, Howitt, SBS):

    Arch the *upper convex hull* over a segment. The point of maximum
    distance *below* the hull is the deepest dip. If that dip exceeds a
    peak-to-dip threshold, place a boundary there and recurse on the two
    halves. Otherwise the segment is one unit, whose nucleus is its maximum.
    (The hull, not an endpoint chord: with silence at the edges the endpoints
    sit low, so a chord would run along the valley floor and never split.)

    Mermelstein, P. (1975). Automatic segmentation of speech into syllabic
    units. JASA 58(4). https://doi.org/10.1121/1.380630

This is a peer of :func:`segment_peakdetect`: a pure ``(trace, times) ->
List[(start, peak, end)]`` primitive that knows nothing about envelopes. It
returns full ``(start, peak, end)`` triples so one primitive serves both roles
a two-stage detector needs -- region finding (use the bounds) and nucleus
picking (use the peak).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .base import EnvelopeBasedSegmenter
from ..envelope.base import EnvelopeComputer


def _hull_dip(trace: np.ndarray, lo: int, hi: int) -> Tuple[int, float]:
    """Deepest point of ``trace[lo:hi]`` below its upper convex hull.

    Returns ``(index, dip)`` where ``dip`` is the vertical gap hull - trace at
    that index. x is the integer sample position (uniformly spaced), so an
    ``argmax`` cross-product monotone chain gives the upper hull.
    """
    x = np.arange(lo, hi + 1, dtype=float)
    y = trace[lo:hi + 1]
    hull: List[int] = []  # positions into x/y
    for i in range(len(x)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
            if cross >= 0:  # left turn / collinear -> not on the upper hull
                hull.pop()
            else:
                break
        hull.append(i)
    hull_line = np.interp(x, x[hull], y[hull])
    dip = hull_line - y
    k = int(np.argmax(dip))
    return lo + k, float(dip[k])


def _boundaries(trace: np.ndarray, threshold: float) -> List[int]:
    """Interior split indices, sorted by position.

    Each segment is split at its deepest below-hull dip (the recursive hull
    rule), then its two halves are split in turn; the collected boundary indices
    are returned in ascending position order. Uses an explicit stack (not Python
    recursion) so ``peak_to_dip=0`` on a noisy trace can't blow the interpreter's
    recursion limit.
    """
    out: List[int] = []
    stack: List[Tuple[int, int]] = [(0, trace.shape[0] - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        idx, dip = _hull_dip(trace, lo, hi)
        if dip > threshold and lo < idx < hi:
            out.append(idx)
            stack.append((lo, idx))
            stack.append((idx, hi))
    out.sort()
    return out


def segment_convexhull(trace: np.ndarray, times: np.ndarray,
                       **kwargs) -> List[Tuple[float, float, float]]:
    """Segment a 1-D trace via the Mermelstein recursive convex-hull procedure.

    Args:
        trace: 1-D signal (energy, periodicity, loudness, ...).
        times: matching frame times in seconds (same length as ``trace``).
        **kwargs:
            peak_to_dip: minimum dip depth (in ``trace`` units) below the upper
                convex hull for a valley to split a segment (default 0.0 = split
                at every interior minimum). Xie uses 0.7 on periodicity, 4.5 dB
                on energy.
            min_syllable_dur: drop segments shorter than this many seconds
                (default 0.05).

    Returns:
        List of ``(start, peak, end)`` tuples in seconds.
    """
    trace = np.asarray(trace, dtype=float)
    times = np.asarray(times, dtype=float)
    if trace.ndim != 1 or times.ndim != 1:
        raise ValueError("segment_convexhull expects 1-D trace and times.")
    if trace.shape[0] != times.shape[0]:
        raise ValueError(
            "segment_convexhull expects trace and times of equal length; "
            f"got {trace.shape[0]} and {times.shape[0]}."
        )
    if trace.shape[0] < 2:
        return []

    peak_to_dip = float(kwargs.get("peak_to_dip", 0.0))
    min_syllable_dur = float(kwargs.get("min_syllable_dur", 0.05))

    bounds = [0, *_boundaries(trace, peak_to_dip), trace.shape[0] - 1]

    segments: List[Tuple[float, float, float]] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        start, end = float(times[lo]), float(times[hi])
        if end - start < min_syllable_dur:
            continue
        peak = lo + int(np.argmax(trace[lo:hi + 1]))
        segments.append((start, float(times[peak]), end))
    return segments


class ConvexHullSegmenter(EnvelopeBasedSegmenter):
    """Convex-hull (Mermelstein) segmenter over any envelope.

    Mirror of :class:`PeakdetectSegmenter`: composes a swappable envelope with
    the ``segment_convexhull`` primitive.

    Args:
        envelope_computer: EnvelopeComputer instance or callable(audio, sr) ->
            (envelope, times).
        peak_to_dip: minimum dip depth to split a segment (default 0.0).
        min_syllable_dur: minimum segment duration in seconds (default 0.05).
        sample_rate, sad, add_utterance_boundaries: as in BaseSegmenter.
    """

    def __init__(
        self,
        envelope_computer: Optional[Union[EnvelopeComputer, Callable]] = None,
        peak_to_dip: float = 0.0,
        min_syllable_dur: float = 0.05,
        sample_rate: int = 16000,
        sad=None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, sad=sad,
                         add_utterance_boundaries=add_utterance_boundaries)
        self.envelope_computer = envelope_computer
        self.peak_to_dip = peak_to_dip
        self.min_syllable_dur = min_syllable_dur

    def _segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        if self.envelope_computer is None:
            raise ValueError("Must provide envelope_computer or use segment(envelope=..., times=...)")
        if hasattr(self.envelope_computer, "compute"):
            envelope, times = self.envelope_computer.compute(audio, sr)
        else:
            envelope, times = self.envelope_computer(audio, sr)
        return self._segment_from_envelope(np.asarray(envelope), np.asarray(times))

    def _segment_from_envelope(self, envelope: np.ndarray,
                               times: np.ndarray) -> List[Tuple[float, float, float]]:
        return segment_convexhull(
            envelope, times,
            peak_to_dip=self.peak_to_dip,
            min_syllable_dur=self.min_syllable_dur,
        )


__all__ = ["segment_convexhull", "ConvexHullSegmenter"]
