"""Periodicity-based envelope (Xie & Niyogi, Interspeech 2006).

Frame-level periodicity is the largest peak of the normalized autocorrelation
in the pitch-period range (2.5-15 ms), per Eq. (3) of:

    Xie, Z. & Niyogi, P. (2006). Robust Acoustic-Based Syllable Detection.
    Interspeech 2006. https://doi.org/10.21437/Interspeech.2006-440

        P(h) = [gamma(h) / (n - h)] / [gamma(0) / n],   gamma(h) = (1/n) Sum_t x_{t+h} x_t

(the sample mean is dropped, as the paper does in its Eq. (2), since the mean of
a speech frame is ~0). gamma(0) is the frame energy.

Vowels/sonorants score high (~0.92), obstruents low (~0.45), so periodicity marks
*where* voiced nuclei live. It is an ABSOLUTE measure: the paper thresholds it
directly (e.g. 0.5, 0.7), so ``normalize`` defaults to ``False`` -- do not
per-utterance rescale it if you rely on those thresholds.

Role in the paper's detector (for context; this class is only the signal): in the
two-stage algorithm periodicity SEGMENTS the utterance into periodic vs aperiodic
regions, and relevant *energy* then picks the nucleus (the energy peak) within
each periodic region. A segmenter composes this envelope with an energy envelope;
this class does not do the picking.
"""

from __future__ import annotations

import numpy as np

from .base import EnvelopeComputer


class PeriodicityEnvelope(EnvelopeComputer):
    """Frame-level periodicity via normalized autocorrelation (Xie & Niyogi 2006).

    Parameters
    ----------
    frame_size : int
        Analysis window in samples (default 400 = 25 ms at 16 kHz; paper Table 1).
    frame_shift : int
        Hop between frames in samples (default 160 = 10 ms at 16 kHz; Table 1).
    min_period_ms : float
        Lower pitch-period bound in ms (default 2.5 ms; lag 40 at 16 kHz).
    max_period_ms : float
        Upper pitch-period bound in ms (default 15 ms; lag 240 at 16 kHz).
    normalize : bool
        If True, min-max scale the trace to [0, 1] across the utterance. Default
        False: periodicity is an absolute measure the paper thresholds directly.
    """

    def __init__(
        self,
        frame_size: int = 400,
        frame_shift: int = 160,
        min_period_ms: float = 2.5,
        max_period_ms: float = 15.0,
        normalize: bool = False,
    ):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.min_period_ms = min_period_ms
        self.max_period_ms = max_period_ms
        self.normalize = normalize

    def _frame_periodicity(self, frame: np.ndarray, h_min: int, h_max: int) -> float:
        """Largest peak of the normalized autocorrelation P(h) over [h_min, h_max].

        Implements Eq. (3), then takes the largest *local maximum* of P over the
        pitch-period range -- "the largest value among the peaks" per the paper --
        falling back to the range maximum if P is monotonic there.
        """
        n = len(frame)
        r = np.correlate(frame, frame, mode="full")[n - 1:]  # r[h] = Sum_t x_{t+h} x_t
        r0 = float(r[0])
        if r0 <= 0.0:
            return 0.0

        h_hi = min(h_max, n - 1)
        if h_min >= h_hi:
            return 0.0

        lags = np.arange(h_min, h_hi + 1)
        # P(h) = [gamma(h)/(n-h)] / [gamma(0)/n] = r[h] * n / ((n-h) * r[0])
        p = r[lags] * n / ((n - lags) * r0)

        if p.size >= 3:
            # interior local maxima; '>=' on one side keeps flat-topped peaks
            peaks = np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
            return float(p[peaks].max()) if peaks.size else float(p.max())
        return float(p.max())

    def compute(self, audio: np.ndarray, sr: int):
        """Compute the per-frame periodicity envelope.

        Returns
        -------
        envelope : np.ndarray, shape (n_frames,)
            Periodicity value per frame (absolute unless ``normalize=True``).
        times : np.ndarray, shape (n_frames,)
            Centre time of each frame in seconds.
        """
        audio = np.asarray(audio, dtype=np.float64)

        h_min = max(1, int(round(self.min_period_ms * sr / 1000)))
        h_max = int(round(self.max_period_ms * sr / 1000))

        n = self.frame_size
        starts = range(0, len(audio) - n + 1, self.frame_shift)  # include the last full frame
        envelope = np.zeros(len(starts), dtype=np.float32)
        times = np.zeros(len(starts), dtype=np.float32)

        for i, s in enumerate(starts):
            envelope[i] = self._frame_periodicity(audio[s:s + n], h_min, h_max)
            times[i] = (s + n / 2) / sr

        if self.normalize and envelope.size and envelope.max() > 0:
            envelope = envelope / envelope.max()

        return envelope, times
