import numpy as np
from scipy.signal import hilbert

def compute_hilbert_envelope(waveform, sr, **kwargs):
    analytic = hilbert(waveform)
    envelope = np.abs(analytic)
    times = np.linspace(0, len(waveform) / sr, len(waveform))
    return envelope, times
