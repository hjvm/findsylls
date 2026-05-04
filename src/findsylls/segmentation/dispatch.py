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
_FEATURE_METHODS_REGISTERED = False

# Canonical methods are the public names we want to present to callers.
# Backward-compatible aliases are normalized before registry lookup.
_CANONICAL_SEGMENTERS: List[str] = [
    'peakdetect',
    'cls_attention',
    'mincut',
    'greedy_cosine',
]

_SEGMENTER_ALIASES: Dict[str, str] = {
    'greedycosine': 'greedy_cosine',
}

# Global cache for segmenter instances (keyed by method + kwargs hash)
# This allows model reuse across multiple files, critical for neural segmenters like Sylber
_SEGMENTER_CACHE: Dict[str, BaseSegmenter] = {}


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


def normalize_segmenter_name(method: str) -> str:
    """Resolve aliases to canonical segmentation method names."""
    key = method.lower().replace('-', '_')
    return _SEGMENTER_ALIASES.get(key, key)


def list_segmenters() -> List[str]:
    """List canonical registered segmentation methods."""
    _register_envelope_methods()
    _register_feature_methods()
    return [name for name in _CANONICAL_SEGMENTERS if name in _SEGMENTERS]


def list_segmenter_aliases() -> Dict[str, str]:
    """List supported alias -> canonical method mappings."""
    return dict(_SEGMENTER_ALIASES)


def get_segmenter(method: str, cache: bool = True, **kwargs) -> BaseSegmenter:
    """
    Get a segmenter instance for the specified method.
    
    Args:
        method: Method name
        cache: If True, cache and reuse segmenter instances (default: True)
               This significantly improves performance when processing multiple files
               with the same segmentation parameters, especially for neural models
               like Sylber that have expensive initialization.
        **kwargs: Parameters to pass to segmenter constructor
    
    Returns:
        Segmenter instance (cached if cache=True)
    
    Raises:
        ValueError: If method not found
    
    Example:
        >>> # First call loads model
        >>> segmenter = get_segmenter('sylber', cache=True)
        >>> segments1 = segmenter.segment(audio=audio1, sr=sr)
        >>> 
        >>> # Second call reuses cached model (no reload!)
        >>> segmenter = get_segmenter('sylber', cache=True)
        >>> segments2 = segmenter.segment(audio=audio2, sr=sr)
    """
    # Ensure methods are registered
    _register_envelope_methods()
    _register_feature_methods()

    method = normalize_segmenter_name(method)
    
    if method not in _SEGMENTERS:
        available = ', '.join(sorted(_SEGMENTERS.keys()))
        raise ValueError(
            f"Unknown segmentation method '{method}'. "
            f"Available methods: {available}"
        )
    
    segmenter_class = _SEGMENTERS[method]
    
    # If caching disabled, create new instance
    if not cache:
        return segmenter_class(**kwargs)
    
    # Create cache key from method + sorted kwargs
    # This ensures same parameters = same cached instance
    cache_key_parts = [method]
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        # Handle unhashable types by converting to string
        try:
            cache_key_parts.append(f"{k}={hash(v)}")
        except TypeError:
            # Fallback for unhashable types (dicts, lists, etc.)
            cache_key_parts.append(f"{k}={str(v)}")
    
    cache_key = "|".join(cache_key_parts)
    
    # Return cached instance if available
    if cache_key in _SEGMENTER_CACHE:
        return _SEGMENTER_CACHE[cache_key]
    
    # Create new instance and cache it
    instance = segmenter_class(**kwargs)
    _SEGMENTER_CACHE[cache_key] = instance
    
    return instance


def _register_envelope_methods():
    """Register all envelope-based methods."""
    global _ENVELOPE_METHODS_REGISTERED
    if not _ENVELOPE_METHODS_REGISTERED:
        from .peakdetect_segmenter import PeakdetectSegmenter
        from ..envelope.base import EnvelopeComputer
        
        # Create a default envelope computer for backward compatibility
        class DefaultHilbertEnvelope(EnvelopeComputer):
            def compute(self, audio, sr):
                from ..envelope.dispatch import get_amplitude_envelope
                return get_amplitude_envelope(audio, sr, method='hilbert')
        
        # Register with default envelope for dispatch compatibility
        class DefaultPeakdetectSegmenter(PeakdetectSegmenter):
            def __init__(self, envelope_method='hilbert', envelope_kwargs=None, **kwargs):
                # Create appropriate envelope computer based on method
                class ConfigurableEnvelope(EnvelopeComputer):
                    def __init__(self, method, env_kwargs):
                        self.method = method
                        self.env_kwargs = env_kwargs or {}
                    def compute(self, audio, sr):
                        from ..envelope.dispatch import get_amplitude_envelope
                        return get_amplitude_envelope(audio, sr, method=self.method, **self.env_kwargs)
                
                envelope_computer = ConfigurableEnvelope(envelope_method, envelope_kwargs)
                super().__init__(envelope_computer, **kwargs)
        
        register_segmenter('peakdetect', DefaultPeakdetectSegmenter)

        from .cls_attention import CLSAttentionSegmenter
        register_segmenter('cls_attention', CLSAttentionSegmenter)
        _ENVELOPE_METHODS_REGISTERED = True


def _register_feature_methods() -> None:
    """Register feature-based segmentation algorithms with default extractors."""
    global _FEATURE_METHODS_REGISTERED
    if _FEATURE_METHODS_REGISTERED:
        return

    from .mincut import MinCutSegmenter
    from .greedy_cosine import GreedyCosineSegmenter
    from ..features import get_extractor

    class DefaultMinCutSegmenter(MinCutSegmenter):
        def __init__(self, feature_type='hubert', feature_kwargs=None, feature_extractor=None, **kwargs):
            extractor = feature_extractor or get_extractor(feature_type, **(feature_kwargs or {}))
            super().__init__(feature_extractor=extractor, **kwargs)

    class DefaultGreedyCosineSegmenter(GreedyCosineSegmenter):
        def __init__(self, feature_type='hubert', feature_kwargs=None, feature_extractor=None, **kwargs):
            extractor = feature_extractor or get_extractor(feature_type, **(feature_kwargs or {}))
            super().__init__(feature_extractor=extractor, **kwargs)

    register_segmenter('mincut', DefaultMinCutSegmenter)
    register_segmenter('greedy_cosine', DefaultGreedyCosineSegmenter)

    _FEATURE_METHODS_REGISTERED = True


# Backward-compatible functional API
def segment_envelope(
    envelope: np.ndarray, 
    times: np.ndarray, 
    method: str = "peakdetect", 
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
        method = "peakdetect"

    method = normalize_segmenter_name(method)
    
    # For backward compatibility, call original function directly if peakdetect
    if method == "peakdetect":
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


def clear_segmenter_cache():
    """
    Clear the global segmenter cache.
    
    Useful for freeing memory or forcing model reloads.
    In most cases, you don't need to call this - cached models
    are automatically reused efficiently.
    
    Example:
        >>> # Process many files with cached models
        >>> for file in files:
        ...     segmenter = get_segmenter('sylber', cache=True)
        ...     segments = segmenter.segment(audio, sr)
        >>> 
        >>> # Optionally clear cache when done
        >>> clear_segmenter_cache()
    """
    global _SEGMENTER_CACHE
    _SEGMENTER_CACHE.clear()


def get_cache_info():
    """
    Get information about cached segmenters.
    
    Returns:
        dict with cache statistics
    
    Example:
        >>> info = get_cache_info()
        >>> print(f"Cached models: {info['num_cached']}")
        >>> print(f"Methods: {info['methods']}")
    """
    methods = set()
    for key in _SEGMENTER_CACHE.keys():
        method = key.split('|')[0]
        methods.add(method)
    
    return {
        'num_cached': len(_SEGMENTER_CACHE),
        'methods': sorted(methods),
        'cache_keys': list(_SEGMENTER_CACHE.keys())
    }
