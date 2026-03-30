"""Internal preprocessing: Multi-band gammatone filterbank for theta oscillator.

This module is NOT a standalone envelope method. It returns a multi-band [bands, time]
array used internally by theta_oscillator_envelope() which reduces it to a single envelope.
"""
import numpy as np, librosa
from gammatone.filters import make_erb_filters, erb_filterbank
from scipy.signal import hilbert

def _gammatone_filterbank_envelope(waveform, sr, **kwargs):
    """Compute multi-band gammatone filterbank envelope (internal preprocessing only).
    
    Returns:
        envelope: [bands, time] array of envelopes per frequency band
        times: Time points in seconds
    """
    bands = kwargs.get("bands", 20)
    minfreq = kwargs.get("minfreq", 50)
    maxfreq = kwargs.get("maxfreq", 7500)
    resample_rate = kwargs.get("resample_rate", 1000)
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
