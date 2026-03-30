"""Factory function for creating EnvelopeComputer instances.

This module provides get_envelope_computer() for backward compatibility
and convenience. The functional API (get_amplitude_envelope) is deprecated
in favor of using EnvelopeComputer classes directly.
"""
from .base import EnvelopeComputer
from .rms import RMSEnvelope, compute_rms_envelope
from .hilbert import HilbertEnvelope, compute_hilbert_envelope
from .lowpass import LowpassEnvelope, compute_lowpass_envelope
from .sbs import SBSEnvelope, spectral_band_subtraction
from .theta import ThetaEnvelope, theta_oscillator_envelope
import numpy as np


def get_envelope_computer(method: str = "sbs", **kwargs) -> EnvelopeComputer:
    """Factory function to create EnvelopeComputer instances.
    
    Args:
        method: Envelope computation method. Options:
            - 'rms': RMS envelope
            - 'hilbert': Hilbert transform envelope
            - 'lowpass': Lowpass filtered envelope
            - 'sbs': Spectral band subtraction
            - 'theta': Theta oscillator (uses gammatone filterbank internally)
        **kwargs: Method-specific parameters passed to the constructor
        
    Returns:
        EnvelopeComputer instance
        
    Example:
        envelope_computer = get_envelope_computer('hilbert')
        segmenter = PeakdetectSegmenter(envelope_computer)
    """
    if method == "rms":
        return RMSEnvelope(**kwargs)
    elif method == "hilbert":
        return HilbertEnvelope(**kwargs)
    elif method == "lowpass":
        return LowpassEnvelope(**kwargs)
    elif method == "sbs":
        return SBSEnvelope(**kwargs)
    elif method == "theta":
        return ThetaEnvelope(**kwargs)
    elif method == "gammatone":
        raise ValueError(
            "'gammatone' is not a standalone envelope method. "
            "It's a preprocessing step used internally by the theta oscillator. "
            "Use method='theta' instead."
        )
    else:
        raise ValueError(
            f"Unsupported envelope method: {method}. "
            f"Available: 'rms', 'hilbert', 'lowpass', 'sbs', 'theta'"
        )


def get_amplitude_envelope(waveform: np.ndarray, sr: int, method: str = "sbs", **kwargs) -> tuple:
    """[DEPRECATED] Compute amplitude envelope using various methods.
    
    Deprecated: Use get_envelope_computer() with EnvelopeComputer classes instead.
    This functional API is maintained for backward compatibility only.
    
    Args:
        waveform: Audio signal
        sr: Sample rate
        method: Envelope computation method
        **kwargs: Method-specific parameters
        
    Returns:
        (envelope, times) tuple
    """
    if method == "rms":
        return compute_rms_envelope(waveform, sr, **kwargs)
    elif method == "hilbert":
        return compute_hilbert_envelope(waveform, sr, **kwargs)
    elif method == "lowpass":
        return compute_lowpass_envelope(waveform, sr, **kwargs)
    elif method == "sbs":
        return spectral_band_subtraction(waveform, sr, **kwargs)
    elif method == "theta":
        return theta_oscillator_envelope(waveform, sr, **kwargs)
    elif method == "gammatone":
        raise ValueError(
            "'gammatone' is not a standalone envelope method. "
            "Use method='theta' instead."
        )
    else:
        raise ValueError(f"Unsupported envelope method: {method}")
