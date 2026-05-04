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
from .cls_attention import CLSAttentionEnvelope
from .greedy_cosine import GreedyCosineEnvelope
from .mincut import MinCutEnvelope
import numpy as np
from ..features import get_extractor


def _resolve_feature_extractor(kwargs):
    feature_extractor = kwargs.pop("feature_extractor", None)
    if feature_extractor is not None:
        return feature_extractor

    feature_type = kwargs.pop("feature_type", None)
    if feature_type is None:
        raise ValueError(
            "Pseudo-envelope methods require either feature_extractor or feature_type + feature_kwargs."
        )

    feature_kwargs = dict(kwargs.pop("feature_kwargs", {}) or {})
    layer = kwargs.pop("layer", None)
    device = kwargs.pop("device", None)

    if device is not None:
        feature_kwargs.setdefault("device", device)

    normalized_feature = str(feature_type).lower().replace("-", "_").strip()
    if layer is not None:
        if normalized_feature == "sylber":
            feature_kwargs.setdefault("encoding_layer", layer)
        else:
            feature_kwargs.setdefault("layer", layer)

    return get_extractor(feature_type, **feature_kwargs)


def _pop_pseudo_envelope_kwargs(kwargs, allowed_keys):
    return {key: kwargs[key] for key in allowed_keys if key in kwargs}


def get_envelope_computer(method: str = "sbs", **kwargs) -> EnvelopeComputer:
    """Factory function to create EnvelopeComputer instances.
    
    Args:
        method: Envelope computation method. Options:
            - 'rms': RMS envelope
            - 'hilbert': Hilbert transform envelope
            - 'lowpass': Lowpass filtered envelope
            - 'sbs': Spectral band subtraction
            - 'theta': Theta oscillator (uses gammatone filterbank internally)
            - 'cls_attention': CLS-attention pseudo-envelope
            - 'greedy_cosine': Greedy-cosine pseudo-envelope
            - 'mincut': MinCut pseudo-envelope
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
    elif method == "cls_attention":
        feature_extractor = _resolve_feature_extractor(kwargs)
        env_kwargs = _pop_pseudo_envelope_kwargs(
            kwargs,
            {"layer", "quantile", "aggregation_method", "normalize"},
        )
        return CLSAttentionEnvelope(feature_extractor=feature_extractor, **env_kwargs)
    elif method == "greedy_cosine":
        feature_extractor = _resolve_feature_extractor(kwargs)
        env_kwargs = _pop_pseudo_envelope_kwargs(
            kwargs,
            {
                "norm_threshold",
                "merge_threshold",
                "aggregation_method",
                "normalize",
                "smooth_frames",
            },
        )
        return GreedyCosineEnvelope(feature_extractor=feature_extractor, **env_kwargs)
    elif method == "mincut":
        feature_extractor = _resolve_feature_extractor(kwargs)
        env_kwargs = _pop_pseudo_envelope_kwargs(
            kwargs,
            {"threshold", "s", "min_hop", "aggregation_method", "normalize", "invert"},
        )
        return MinCutEnvelope(feature_extractor=feature_extractor, **env_kwargs)
    elif method == "gammatone":
        raise ValueError(
            "'gammatone' is not a standalone envelope method. "
            "It's a preprocessing step used internally by the theta oscillator. "
            "Use method='theta' instead."
        )
    else:
        raise ValueError(
            f"Unsupported envelope method: {method}. "
            f"Available: 'rms', 'hilbert', 'lowpass', 'sbs', 'theta', "
            f"'cls_attention', 'greedy_cosine', 'mincut'"
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
    elif method == "cls_attention":
        feature_extractor = _resolve_feature_extractor(kwargs)
        env_kwargs = _pop_pseudo_envelope_kwargs(
            kwargs,
            {"layer", "quantile", "aggregation_method", "normalize"},
        )
        return CLSAttentionEnvelope(feature_extractor=feature_extractor, **env_kwargs).compute(waveform, sr)
    elif method == "greedy_cosine":
        feature_extractor = _resolve_feature_extractor(kwargs)
        env_kwargs = _pop_pseudo_envelope_kwargs(
            kwargs,
            {
                "norm_threshold",
                "merge_threshold",
                "aggregation_method",
                "normalize",
                "smooth_frames",
            },
        )
        return GreedyCosineEnvelope(feature_extractor=feature_extractor, **env_kwargs).compute(waveform, sr)
    elif method == "mincut":
        feature_extractor = _resolve_feature_extractor(kwargs)
        env_kwargs = _pop_pseudo_envelope_kwargs(
            kwargs,
            {"threshold", "s", "min_hop", "aggregation_method", "normalize", "invert"},
        )
        return MinCutEnvelope(feature_extractor=feature_extractor, **env_kwargs).compute(waveform, sr)
    elif method == "gammatone":
        raise ValueError(
            "'gammatone' is not a standalone envelope method. "
            "Use method='theta' instead."
        )
    else:
        raise ValueError(f"Unsupported envelope method: {method}")
