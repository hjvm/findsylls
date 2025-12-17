"""
Dispatch system for segmentation methods.

Provides:
- Registration system for extensible method discovery
- Backward-compatible functional API (segment_envelope)
- New unified API (get_segmenter, segment_audio)
- Lazy loading for end-to-end models
"""

from typing import Dict, Type, List, Tuple, Optional
import numpy as np

from .base import BaseSegmenter, EnvelopeBasedSegmenter, End2EndSegmenter
from .peakdetect_segmenter import segment_peakdetect


# Registry for segmentation methods
_SEGMENTERS: Dict[str, Type[BaseSegmenter]] = {}
_ENVELOPE_METHODS_REGISTERED = False
_END2END_METHODS_REGISTERED = False


def register_segmenter(name: str, segmenter_class: Type[BaseSegmenter]) -> None:
    """
    Register a segmentation method.
    
    Args:
        name: Method name (used in method= parameter)
        segmenter_class: Segmenter class (must inherit from BaseSegmenter)
    """
    if not issubclass(segmenter_class, BaseSegmenter):
        raise TypeError(f"{segmenter_class} must inherit from BaseSegmenter")
    _SEGMENTERS[name] = segmenter_class


def get_segmenter(method: str, **kwargs) -> BaseSegmenter:
    """
    Get a segmenter instance for the specified method.
    
    Args:
        method: Method name
        **kwargs: Parameters to pass to segmenter constructor
    
    Returns:
        Segmenter instance
    
    Raises:
        ValueError: If method not found
    
    Example:
        >>> segmenter = get_segmenter('peaks_and_valleys', delta=0.02)
        >>> segments = segmenter.segment(audio=audio, sr=sr)
    """
    # Ensure methods are registered
    _register_envelope_methods()
    _register_end2end_methods()
    
    if method not in _SEGMENTERS:
        available = ', '.join(sorted(_SEGMENTERS.keys()))
        raise ValueError(
            f"Unknown segmentation method '{method}'. "
            f"Available methods: {available}"
        )
    
    segmenter_class = _SEGMENTERS[method]
    return segmenter_class(**kwargs)


def _register_envelope_methods():
    """Register all envelope-based methods."""
    global _ENVELOPE_METHODS_REGISTERED
    if not _ENVELOPE_METHODS_REGISTERED:
        from .peaks_and_valleys import PeaksAndValleysSegmenter
        register_segmenter('peaks_and_valleys', PeaksAndValleysSegmenter)  # Backward compatibility
        register_segmenter('billauer', PeaksAndValleysSegmenter)  # Explicit Billauer naming
        _ENVELOPE_METHODS_REGISTERED = True


def _register_end2end_methods() -> None:
    """Register end-to-end neural segmentation methods (lazy loading)."""
    global _END2END_METHODS_REGISTERED
    if _END2END_METHODS_REGISTERED:
        return
    
    # Lazy import - only load if packages are installed
    try:
        from .end2end.sylber_segmenter import SylberSegmenter
        register_segmenter('sylber', SylberSegmenter)
    except ImportError:
        pass  # sylber package not installed
    
    try:
        from .end2end.vg_hubert_segmenter import VGHubertSegmenter
        register_segmenter('vg_hubert', VGHubertSegmenter)
        register_segmenter('vg_hubert_mincut', VGHubertSegmenter)
    except ImportError:
        pass  # VG-HuBERT dependencies not installed
    
    _END2END_METHODS_REGISTERED = True


# Backward-compatible functional API
def segment_envelope(
    envelope: np.ndarray, 
    times: np.ndarray, 
    method: str = "peaks_and_valleys", 
    **kwargs
) -> List[Tuple[float, float, float]]:
    """
    Segment from pre-computed envelope (backward compatible).
    
    This is the original functional API. For new code, consider using
    get_segmenter() for more flexibility.
    
    Args:
        envelope: Amplitude envelope
        times: Time array (seconds)
        method: Segmentation method name
        **kwargs: Method-specific parameters
    
    Returns:
        List of (start, nucleus, end) tuples
    
    Raises:
        ValueError: If method requires raw audio (end-to-end methods)
    """
    if method is None:
        method = "peaks_and_valleys"
    
    # For backward compatibility, call original function directly if peaks_and_valleys or billauer
    if method in ("peaks_and_valleys", "billauer"):
        return segment_peakdetect(envelope=envelope, times=times, **kwargs)
    
    # Otherwise use new system
    try:
        segmenter = get_segmenter(method, **kwargs)
    except ValueError:
        raise ValueError(f"Unsupported segmentation method: {method}")
    
    # Check if segmenter can accept envelope
    if isinstance(segmenter, EnvelopeBasedSegmenter):
        return segmenter.segment(envelope=envelope, times=times, **kwargs)
    elif isinstance(segmenter, End2EndSegmenter):
        raise ValueError(
            f"Method '{method}' is an end-to-end neural method that requires raw audio. "
            f"Use segment_audio() from pipeline module instead."
        )
    else:
        raise ValueError(f"Unknown segmenter type: {type(segmenter)}")
