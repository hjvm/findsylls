"""Two-stage gated segmentation (region-restrict, then pick).

The slice-and-pick engine behind Xie & Niyogi (2006): a *gate* envelope
proposes regions (periodic stretches), and a *primary* envelope's nucleus is
picked inside each region independently. This is the faithful alternative to
multiplicative gating (``ConvexHullSegmenter(ProductEnvelope([primary, gate]))``):

- Slice-and-pick keeps each region's picker on that region's own trace, so it
  works on a dB primary (where multiplying by a 0/1 mask would invert), and
  reproduces Xie's per-region independent convex hulls.
- Regions are contiguous runs where the gate exceeds ``gate_on`` -- the same
  gate a ThresholdGate feeds the multiplicative path, so the two strategies are
  directly comparable (only multiply-vs-restrict differs).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from .base import BaseSegmenter
from .convexhull import segment_convexhull
from ..envelope.base import EnvelopeComputer

_Env = Union[EnvelopeComputer, callable]


def _compute(env: _Env, audio, sr):
    if hasattr(env, "compute"):
        return env.compute(audio, sr)
    return env(audio, sr)


def _on_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs of a boolean mask as (start_idx, end_idx) inclusive."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


class RegionGatedSegmenter(BaseSegmenter):
    """Gate proposes regions; primary's nucleus is picked inside each.

    Args:
        primary_env: envelope whose peaks are the nuclei (e.g. dB energy).
        gate_env: envelope defining where to look (e.g. periodicity or a
            ThresholdGate). Regions are contiguous frames where it exceeds
            ``gate_on``.
        gate_on: "on" threshold for the gate (default 0.5, i.e. a 0/1 mask).
        peak_to_dip: convex-hull dip threshold for picking nuclei within a
            region (primary's units; Xie uses 4.5 dB).
        min_syllable_dur: minimum nucleus-segment duration in seconds.
        sample_rate, sad, add_utterance_boundaries: as in BaseSegmenter.
    """

    def __init__(
        self,
        primary_env: _Env,
        gate_env: _Env,
        gate_on: float = 0.5,
        peak_to_dip: float = 0.0,
        min_syllable_dur: float = 0.05,
        sample_rate: int = 16000,
        sad=None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, sad=sad,
                         add_utterance_boundaries=add_utterance_boundaries)
        self.primary_env = primary_env
        self.gate_env = gate_env
        self.gate_on = gate_on
        self.peak_to_dip = peak_to_dip
        self.min_syllable_dur = min_syllable_dur

    def _segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        primary, ptimes = _compute(self.primary_env, audio, sr)
        gate, gtimes = _compute(self.gate_env, audio, sr)
        primary = np.asarray(primary, dtype=float)
        ptimes = np.asarray(ptimes, dtype=float)
        gate = np.asarray(gate, dtype=float)
        gtimes = np.asarray(gtimes, dtype=float)

        nuclei: List[Tuple[float, float, float]] = []
        for lo, hi in _on_runs(gate > self.gate_on):
            t_lo, t_hi = gtimes[lo], gtimes[hi]
            sel = (ptimes >= t_lo) & (ptimes <= t_hi)
            if sel.sum() < 2:
                continue
            nuclei.extend(segment_convexhull(
                primary[sel], ptimes[sel],
                peak_to_dip=self.peak_to_dip,
                min_syllable_dur=self.min_syllable_dur,
            ))
        return nuclei


__all__ = ["RegionGatedSegmenter"]
