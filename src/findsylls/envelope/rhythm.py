"""Speech-rhythm weight envelope (Zhang & Glass, ICASSP 2009).

Two-pass derived envelope: pick first-pass peaks on a base envelope, fit the
paper's rhythm sinusoid to the peak train, and return a phase-locked weight
that boosts on-beat instants and damps off-beat ones.

    Zhang, Y. & Glass, J. R. (2009). Speech rhythm guided syllable nuclei
    detection. ICASSP 2009. https://doi.org/10.1109/ICASSP.2009.4960454

The fit is the paper's Eq. (4): ``{k1,k2} = argmin (1/|P|) sum_i
(1 - sin(k1*p_i + 2*pi*k2))^2`` with ``k2 in [0,1)`` -- i.e. place the
sinusoid's crests on the observed peaks. The paper estimates it iteratively
left-to-right, but reports (Section 3.1) that a whole-utterance batch fit gives
"very similar results", so this class fits once over the utterance.

The weight is ``(1 + sin(k1*t + 2*pi*k2)) / 2`` in [0,1], lifted by ``floor``
so off-beat peaks are damped rather than killed (the paper's search window --
1.5 rhythm cycles past the last aligned crest -- is likewise permissive rather
than a hard gate). Multiply it into the base envelope via ``ProductEnvelope``
or use ``output="weighted"`` to get ``base * weight`` in one pass.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import EnvelopeComputer


def fit_rhythm_sinusoid(
    peak_times: np.ndarray,
    period_range: Tuple[float, float] = (0.1, 0.5),
    n_freq: int = 200,
    n_phase: int = 64,
) -> Tuple[float, float]:
    """Fit sin(k1*x + 2*pi*k2) to a peak train by the paper's LMS objective.

    Exhaustive vectorized grid over (k1, k2) evaluating Eq. (4) exactly.
    ponytail: |P| is tens of peaks, so the full grid is ~1e5 sin evals --
    cheaper and more robust than an iterative optimizer on a multimodal
    objective.

    Args:
        peak_times: first-pass peak locations in seconds (>= 2).
        period_range: rhythm period bounds 2*pi/k1 in seconds
            (default 0.1-0.5 s = 2-10 syllables/s).
        n_freq, n_phase: grid resolution.

    Returns:
        (k1, k2): angular frequency (rad/s) and phase offset in [0, 1).
    """
    p = np.asarray(peak_times, dtype=float)
    if p.size < 2:
        raise ValueError("Need at least two peaks to estimate rhythm.")
    k1 = 2 * np.pi / np.linspace(period_range[1], period_range[0], n_freq)
    k2 = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    # err[i,j] = mean_p (1 - sin(k1_i * p + 2 pi k2_j))^2
    phase = k1[:, None, None] * p[None, None, :] + 2 * np.pi * k2[None, :, None]
    err = ((1.0 - np.sin(phase)) ** 2).mean(axis=2)
    # Harmonic degeneracy: a k1 multiple also puts crests on every peak and ties
    # the objective. Among near-ties, prefer the smallest k1 (slowest rhythm) --
    # the paper's "avoid the uninteresting solution of large k1".
    best_per_k1 = err.min(axis=1)
    tol = 0.05 * (1.0 + float(best_per_k1.min()))
    i = int(np.flatnonzero(best_per_k1 <= best_per_k1.min() + tol)[0])  # k1 ascending
    j = int(np.argmin(err[i]))
    return float(k1[i]), float(k2[j])


class RhythmEnvelope(EnvelopeComputer):
    """Rhythm weight from a batch fit of Zhang & Glass's sinusoid.

    Args:
        base: envelope the first-pass peaks are picked on (default: the paper's
            ERB Hilbert-sum, ``GammatoneEnvelope(reduction="normalized_sum")``).
        delta: first-pass peakdetect delta on the [0,1]-normalized base
            envelope (default 0.05).
        period_range: rhythm period bounds in seconds (default (0.1, 0.5)).
        floor: minimum weight in [0,1] (default 0.3). 0 kills off-beat peaks
            outright; the paper's interval search is more permissive.
        output: "weight" (bare rhythm weight) or "weighted" (base * weight,
            default -- one base pass, directly usable as a primary envelope).
        normalize: min-max scale the "weighted" output to [0,1] (default True)
            so downstream thresholds are portable across utterances. Ignored
            for output="weight" (already in [floor, 1]).

    Falls back to a flat weight of 1 when fewer than two first-pass peaks are
    found (no rhythm evidence).
    """

    def __init__(
        self,
        base: Optional[EnvelopeComputer] = None,
        delta: float = 0.05,
        period_range: Tuple[float, float] = (0.1, 0.5),
        floor: float = 0.3,
        output: str = "weighted",
        normalize: bool = True,
    ):
        if output not in ("weight", "weighted"):
            raise ValueError(f"output must be 'weight' or 'weighted', got {output!r}")
        if base is None:
            from .gammatone import GammatoneEnvelope
            base = GammatoneEnvelope(reduction="normalized_sum")
        self.base = base
        self.delta = delta
        self.period_range = period_range
        self.floor = floor
        self.output = output
        self.normalize = normalize

    def compute(self, audio: np.ndarray, sr: int):
        from ..segmentation.peakdetect_segmenter import segment_peakdetect

        env, times = self.base.compute(audio, sr)
        env = np.asarray(env, dtype=float)
        times = np.asarray(times, dtype=float)

        # first pass: peaks on the [0,1]-normalized base envelope
        span = env.max() - env.min()
        norm = (env - env.min()) / span if span > 0 else np.zeros_like(env)
        segs = segment_peakdetect(norm, times, delta=self.delta,
                                  add_boundary_valleys=True)
        peaks = np.array([pk for _, pk, _ in segs])

        if peaks.size >= 2:
            k1, k2 = fit_rhythm_sinusoid(peaks, period_range=self.period_range)
            weight = (1.0 + np.sin(k1 * times + 2 * np.pi * k2)) / 2.0
            weight = self.floor + (1.0 - self.floor) * weight
        else:
            weight = np.ones_like(times)  # no rhythm evidence

        if self.output == "weight":
            return weight.astype(np.float32), times
        out = env * weight
        if self.normalize:
            span = out.max() - out.min()
            out = (out - out.min()) / span if span > 0 else np.zeros_like(out)
        return out.astype(np.float32), times


__all__ = ["fit_rhythm_sinusoid", "RhythmEnvelope"]
