import numpy as np, librosa
from gammatone.filters import make_erb_filters, erb_filterbank
from scipy.signal import hilbert

def gammatone_filterbank_envelope(waveform, sr, **kwargs):
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
