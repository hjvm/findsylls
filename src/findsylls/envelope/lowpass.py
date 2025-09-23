import numpy as np
from scipy.signal import butter, filtfilt

def compute_lowpass_envelope(waveform, sr, **kwargs):
    cutoff = kwargs.get("cutoff", 5)
    order = kwargs.get("order", 4)
    b, a = butter(N=order, Wn=cutoff / (sr / 2), btype='low')
    rectified = np.abs(waveform)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if rectified.size <= padlen:
        win_len = min(max(5, int(0.01 * sr)), rectified.size)
        if win_len <= 1:
            envelope = rectified.copy()
        else:
            kernel = np.ones(win_len, dtype=rectified.dtype) / win_len
            envelope = np.convolve(rectified, kernel, mode='same')
    else:
        envelope = filtfilt(b, a, rectified)
    times = np.linspace(0, len(waveform) / sr, len(waveform))
    return envelope, times
