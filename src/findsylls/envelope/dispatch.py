from .rms import compute_rms_envelope
from .hilbert import compute_hilbert_envelope
from .lowpass import compute_lowpass_envelope
from .sbs import spectral_band_subtraction
from .gammatone import gammatone_filterbank_envelope
from .theta import theta_oscillator_envelope
import numpy as np


def get_amplitude_envelope(waveform: np.ndarray, sr: int, method: str = "sbs", **kwargs) -> tuple:
    if method == "rms":
        return compute_rms_envelope(waveform, sr, **kwargs)
    elif method == "hilbert":
        return compute_hilbert_envelope(waveform, sr, **kwargs)
    elif method == "lowpass":
        return compute_lowpass_envelope(waveform, sr, **kwargs)
    elif method == "sbs":
        return spectral_band_subtraction(waveform, sr, **kwargs)
    elif method == "gammatone":
        return gammatone_filterbank_envelope(waveform, sr, **kwargs)
    elif method == "theta":
        return theta_oscillator_envelope(waveform, sr, **kwargs)
    else:
        raise ValueError(f"Unsupported envelope method: {method}")
