import numpy as np
from scipy.signal import spectrogram, filtfilt
from scipy.signal.windows import hamming

def spectral_band_subtraction(waveform, sr, **kwargs):
    nfft = kwargs.get("nfft", 256)
    window_length = kwargs.get("window_length", 256)
    step = kwargs.get("step", 160)
    pivot_freq = kwargs.get("pivot_freq", 3000)
    smoothing_window_samples = kwargs.get("smoothing_window_samples", 7)
    noverlap = window_length - step
    f, t, Sxx = spectrogram(waveform, fs=sr, nfft=nfft, window='hamming', nperseg=window_length, noverlap=noverlap, mode='magnitude')
    t = t + (nfft / sr) / 2.0
    freq_res = sr / nfft
    pivot_bin = int(np.ceil(pivot_freq / freq_res))
    low_energy = np.sum(Sxx[1:pivot_bin, :], axis=0)
    high_energy = np.sum(Sxx[pivot_bin:(nfft // 2), :], axis=0)
    diff_signal = np.clip(low_energy - high_energy, a_min=0, a_max=None)
    win = hamming(smoothing_window_samples)
    padlen = 3 * (len(win) - 1)
    if diff_signal.size <= padlen:
        k = win / np.sum(win) if np.sum(win) != 0 else win
        envelope = np.convolve(diff_signal, k, mode='same')
    else:
        envelope = filtfilt(win, [1.0], diff_signal)
    if envelope.max() > 0:
        envelope /= envelope.max()
    return envelope, t
