"""Threshold segmentation: contiguous regions of a 1-D trace above a threshold.

A 0/1 mask and its list of contiguous regions are the same information in two
representations (dense per-frame vs sparse spans) -- a lossless bijection. So
thresholding an envelope and extracting its regions is ONE operation with two
output views:

- ``segment()`` / ``segment_threshold`` -> regions as ``(start, peak, end)``
- ``mask()`` -> the dense 0/1 view

This is the reusable "regions above a threshold" primitive (the region logic in
``vad/energy.py`` is a specialization of it). Composing conditions is then just
``&`` on the dense masks (equivalently, intersecting the region sets).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .base import EnvelopeBasedSegmenter
from ..envelope.base import EnvelopeComputer

_Env = Union[EnvelopeComputer, Callable]


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs of a boolean mask as (start_idx, end_idx) inclusive."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    ends = np.concatenate((idx[brk], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def segment_threshold(trace: np.ndarray, times: np.ndarray,
                      **kwargs) -> List[Tuple[float, float, float]]:
    """Segment a 1-D trace into contiguous regions where ``trace >= threshold``.

    Args:
        trace: 1-D signal (or a 0/1 mask).
        times: matching frame times in seconds (same length as ``trace``).
        **kwargs:
            threshold: cutoff (default 0.5, i.e. a 0/1 mask).
            min_syllable_dur: drop regions shorter than this many seconds
                (default 0.0).

    Returns:
        List of ``(start, peak, end)`` tuples in seconds; ``peak`` is the argmax
        of the trace within the region (arbitrary-but-defined for a flat mask).
    """
    trace = np.asarray(trace, dtype=float)
    times = np.asarray(times, dtype=float)
    if trace.ndim != 1 or times.ndim != 1:
        raise ValueError("segment_threshold expects 1-D trace and times.")
    if trace.shape[0] != times.shape[0]:
        raise ValueError(
            "segment_threshold expects trace and times of equal length; "
            f"got {trace.shape[0]} and {times.shape[0]}."
        )

    threshold = float(kwargs.get("threshold", 0.5))
    min_syllable_dur = float(kwargs.get("min_syllable_dur", 0.0))

    out: List[Tuple[float, float, float]] = []
    for lo, hi in _runs(trace >= threshold):
        start, end = float(times[lo]), float(times[hi])
        if end - start < min_syllable_dur:
            continue
        peak = lo + int(np.argmax(trace[lo:hi + 1]))
        out.append((start, float(times[peak]), end))
    return out


class ThresholdSegmenter(EnvelopeBasedSegmenter):
    """Regions of an envelope above a threshold (peer of PeakdetectSegmenter).

    ``segment()`` returns the regions; ``mask()`` returns the dense 0/1 view of
    the same thresholding (the two are equivalent representations).

    Args:
        envelope_computer: EnvelopeComputer or callable(audio, sr) -> (env, times).
        threshold: cutoff on the envelope (default 0.5).
        min_syllable_dur: minimum region duration in seconds (default 0.0).
        sample_rate, sad, add_utterance_boundaries: as in BaseSegmenter.
    """

    def __init__(
        self,
        envelope_computer: Optional[_Env] = None,
        threshold: float = 0.5,
        min_syllable_dur: float = 0.0,
        sample_rate: int = 16000,
        sad=None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, sad=sad,
                         add_utterance_boundaries=add_utterance_boundaries)
        self.envelope_computer = envelope_computer
        self.threshold = threshold
        self.min_syllable_dur = min_syllable_dur

    def _compute_env(self, audio: np.ndarray, sr: int):
        if self.envelope_computer is None:
            raise ValueError("Must provide envelope_computer or use segment(envelope=..., times=...)")
        if hasattr(self.envelope_computer, "compute"):
            return self.envelope_computer.compute(audio, sr)
        return self.envelope_computer(audio, sr)

    def mask(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Dense 0/1 view: 1 where the envelope is at/above ``threshold``."""
        env, times = self._compute_env(audio, sr)
        mask = (np.asarray(env, dtype=float) >= self.threshold).astype(np.float32)
        return mask, np.asarray(times, dtype=float)

    def _segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        env, times = self._compute_env(audio, sr)
        return self._segment_from_envelope(np.asarray(env), np.asarray(times))

    def _segment_from_envelope(self, envelope: np.ndarray,
                               times: np.ndarray) -> List[Tuple[float, float, float]]:
        return segment_threshold(
            envelope, times,
            threshold=self.threshold, min_syllable_dur=self.min_syllable_dur,
        )


__all__ = ["segment_threshold", "ThresholdSegmenter"]
