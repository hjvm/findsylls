import numpy as np
from scipy.signal import hilbert
from .base import EnvelopeComputer

def compute_hilbert_envelope(waveform, sr, **kwargs):
    analytic = hilbert(waveform)
    envelope = np.abs(analytic)
    times = np.linspace(0, len(waveform) / sr, len(waveform))
    return envelope, times


class HilbertEnvelope(EnvelopeComputer):
    """Compute Hilbert envelope using analytic signal."""
    
    def compute(self, audio: np.ndarray, sr: int):
        return compute_hilbert_envelope(audio, sr)
