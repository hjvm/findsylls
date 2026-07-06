"""Multiplicative envelope composition (gating).

Two-stage syllable detectors gate a primary envelope by other signals: e.g.
Xie & Niyogi (2006) picks nuclei on energy but only inside periodic regions,
i.e. ``energy * 1[periodicity > 0.7]``. Gating is just element-wise
multiplication once the components share a time grid.

- ``ThresholdGate`` turns any envelope into a 0/1 mask (component > threshold).
- ``ProductEnvelope`` multiplies several component envelopes, resampling each
  onto the first component's time grid (that primary component defines the
  output resolution; gates are interpolated onto it).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import EnvelopeComputer


class ThresholdGate(EnvelopeComputer):
    """Binary mask: 1 where ``inner`` envelope exceeds ``threshold``, else 0.

    Args:
        inner: the envelope to threshold.
        threshold: absolute cutoff (the inner envelope's own units).
        ge: if True (default) use ``>= threshold``, else strict ``>``.
    """

    def __init__(self, inner: EnvelopeComputer, threshold: float, ge: bool = True):
        self.inner = inner
        self.threshold = threshold
        self.ge = ge

    def compute(self, audio: np.ndarray, sr: int):
        env, times = self.inner.compute(audio, sr)
        env = np.asarray(env, dtype=float)
        mask = env >= self.threshold if self.ge else env > self.threshold
        return mask.astype(np.float32), np.asarray(times, dtype=float)


class ProductEnvelope(EnvelopeComputer):
    """Element-wise product of several envelopes on a common time grid.

    The first component is the primary: its ``times`` define the output grid,
    and every other component is linearly interpolated onto it before
    multiplying. (np.interp clamps outside each component's range, so a gate
    that ends slightly early just holds its edge value.)

    Args:
        components: envelope computers; component[0] is the primary.

    Note: multiplicative gating assumes a NON-NEGATIVE primary (linear energy,
    Hilbert-sum). On a dB/log primary (values <= 0) a 0/1 gate inverts -- the
    suppressed frames become 0, i.e. the maximum. For a dB primary, restrict the
    picker to the gated region instead of multiplying (slice-and-pick).
    """

    def __init__(self, components: List[EnvelopeComputer]):
        if not components:
            raise ValueError("ProductEnvelope needs at least one component.")
        self.components = components

    def compute(self, audio: np.ndarray, sr: int):
        primary, times = self.components[0].compute(audio, sr)
        times = np.asarray(times, dtype=float)
        out = np.asarray(primary, dtype=float).copy()
        for comp in self.components[1:]:
            env, ctimes = comp.compute(audio, sr)
            out *= np.interp(times, np.asarray(ctimes, dtype=float),
                             np.asarray(env, dtype=float))
        return out.astype(np.float32), times


__all__ = ["ThresholdGate", "ProductEnvelope"]
