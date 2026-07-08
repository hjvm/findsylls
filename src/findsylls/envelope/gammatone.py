"""Gammatone / ERB auditory filterbank envelope.

Provides a multi-band ERB (gammatone) filterbank with per-channel Hilbert
envelopes, and 1-D reductions of it. Shared front-end for auditory syllabic
methods:

- ``ThetaEnvelope`` calls ``gammatone_filterbank()`` and applies its oscillator
  + top-N reduction.
- Zhang & Glass (2009) ERB envelope = per-channel normalize + sum, i.e.
  ``GammatoneEnvelope(reduction="normalized_sum")``.
- Standalone: ``GammatoneEnvelope()`` (1-D sum) or ``.filterbank()`` (multi-band).

``make_erb_filters`` implements Slaney's gammatone filterbank on the ERB scale,
so ERB and gammatone refer to the same filters here.
"""
import numpy as np, librosa
from gammatone.filters import make_erb_filters, erb_filterbank
from scipy.signal import hilbert

from .base import EnvelopeComputer


def gammatone_filterbank(waveform, sr, bands=20, minfreq=50, maxfreq=7500, resample_rate=1000):
    """Multi-band ERB/gammatone filterbank with per-channel Hilbert envelopes.

    Returns:
        envelope: [bands, time] per-channel Hilbert envelopes (resampled to
            ``resample_rate``), low band first.
        times: (time,) frame centre times in seconds.
    """
    if bands < 2:
        raise ValueError(f"bands must be >= 2 for a log-spaced filterbank, got {bands}.")
    cfs = np.zeros((bands, 1))
    const = (maxfreq / minfreq) ** (1 / (bands - 1))
    cfs[0] = minfreq
    for k in range(1, bands):
        cfs[k] = cfs[k - 1] * const
    coefs = make_erb_filters(sr, cfs, width=1.0)
    filtered = erb_filterbank(waveform, coefs)
    hilbert_env = np.abs(hilbert(filtered))
    envelope = librosa.resample(hilbert_env, orig_sr=sr, target_sr=resample_rate)
    times = np.linspace(0, len(waveform) / sr, num=envelope.shape[1])
    return envelope, times


def _reduce_bands(bands_env: np.ndarray, reduction: str) -> np.ndarray:
    """Collapse a [bands, time] filterbank to a 1-D envelope."""
    if reduction == "sum":
        return bands_env.sum(axis=0)
    if reduction == "mean":
        return bands_env.mean(axis=0)
    if reduction == "normalized_sum":
        # per-band peak normalisation, then sum ("reinforce energy agreement of
        # each channel", Zhang & Glass 2009)
        peak = bands_env.max(axis=1, keepdims=True)
        peak = np.where(peak > 0, peak, 1.0)
        return (bands_env / peak).sum(axis=0)
    raise ValueError(
        f"reduction must be one of {{'sum','mean','normalized_sum'}}, got {reduction!r}"
    )


class GammatoneEnvelope(EnvelopeComputer):
    """ERB/gammatone auditory envelope.

    ``compute()`` returns a 1-D reduction of the filterbank (the ``EnvelopeComputer``
    contract); ``filterbank()`` exposes the underlying multi-band ``[bands, time]``
    representation for consumers that need it (e.g. the theta oscillator).

    Args:
        bands, minfreq, maxfreq, resample_rate: filterbank geometry.
        reduction: how ``compute()`` collapses bands to 1-D --
            'sum' (default), 'mean', or 'normalized_sum' (per-band peak-normalise
            then sum; the Zhang & Glass 2009 total envelope).
    """

    def __init__(self, bands=20, minfreq=50, maxfreq=7500, resample_rate=1000, reduction="sum"):
        self.bands = bands
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.resample_rate = resample_rate
        self.reduction = reduction

    def filterbank(self, audio: np.ndarray, sr: int):
        """Return the multi-band [bands, time] filterbank envelope and times."""
        return gammatone_filterbank(
            audio, sr,
            bands=self.bands, minfreq=self.minfreq,
            maxfreq=self.maxfreq, resample_rate=self.resample_rate,
        )

    def compute(self, audio: np.ndarray, sr: int):
        bands_env, times = self.filterbank(audio, sr)
        return _reduce_bands(bands_env, self.reduction), times
