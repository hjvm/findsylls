"""
Dispatch system for segmentation methods.

Provides:
- Registration system for extensible method discovery
- Unified API (get_segmenter)
- Lazy loading for end-to-end models
"""

from typing import Dict, Type, List

from .base import BaseSegmenter
# Light imports (no torch / no import cycle): used by the module-level peakdetect
# registry default below. Feature-based factories keep their imports lazy inside
# _register_feature_methods because they pull heavy/optional extractor deps.
from .peakdetect_segmenter import PeakdetectSegmenter
from .convexhull import ConvexHullSegmenter
from ..envelope.base import EnvelopeComputer


# Registry for segmentation methods
_SEGMENTERS: Dict[str, Type[BaseSegmenter]] = {}
_ENVELOPE_METHODS_REGISTERED = False
_FEATURE_METHODS_REGISTERED = False

# Canonical methods are the public names we want to present to callers.
# Backward-compatible aliases are normalized before registry lookup.
_CANONICAL_SEGMENTERS: List[str] = [
    'peakdetect',
    'convexhull',
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


def _kwarg_key(value) -> str:
    """Deterministic cache-key fragment for a get_segmenter kwarg value.

    Value types (str/int/float/bool/None) and nested dict/list/tuple containers
    use a content repr so equal configs share a cache entry reproducibly. Object
    instances (feature extractors, SAD backends, envelope computers) key by
    identity, so a reused instance hits the cache while distinct instances do not
    collide.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: repr(kv[0]))
        return "{" + ",".join(f"{k!r}:{_kwarg_key(v)}" for k, v in items) + "}"
    if isinstance(value, tuple):
        return "(" + ",".join(_kwarg_key(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ",".join(_kwarg_key(v) for v in value) + "]"
    return f"<{type(value).__name__}@{id(value)}>"


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
    
    # Create cache key from method + sorted kwargs. Value types use a
    # deterministic content repr; object instances (extractors, SAD backends,
    # envelope computers) key by identity so a *reused* instance hits the cache
    # while distinct instances stay separate. The cache is per-process and is
    # meant to be released at workflow boundaries (see clear_segmenter_cache).
    cache_key = "|".join(
        [method] + [f"{k}={_kwarg_key(kwargs[k])}" for k in sorted(kwargs.keys())]
    )
    
    # Return cached instance if available
    if cache_key in _SEGMENTER_CACHE:
        return _SEGMENTER_CACHE[cache_key]
    
    # Create new instance and cache it
    instance = segmenter_class(**kwargs)
    _SEGMENTER_CACHE[cache_key] = instance
    
    return instance


class _ConfigurableEnvelope(EnvelopeComputer):
    """Envelope computer that defers to a named ``get_amplitude_envelope`` method.

    Used by the ``peakdetect`` registry default so the dispatch caller can pick an
    envelope method (``hilbert``, ``sbs``, ``theta``, ...) by name. The
    ``get_amplitude_envelope`` import stays lazy to avoid eagerly pulling envelope
    backends at module load.
    """

    def __init__(self, method: str = "hilbert", env_kwargs=None):
        self.method = method
        self.env_kwargs = env_kwargs or {}

    def compute(self, audio, sr):
        from ..envelope.dispatch import get_amplitude_envelope
        return get_amplitude_envelope(audio, sr, method=self.method, **self.env_kwargs)


class DefaultPeakdetectSegmenter(PeakdetectSegmenter):
    """``peakdetect`` registry default: a PeakdetectSegmenter whose envelope is
    selected by name via ``envelope_method`` / ``envelope_kwargs``."""

    def __init__(self, envelope_method: str = "hilbert", envelope_kwargs=None, **kwargs):
        super().__init__(_ConfigurableEnvelope(envelope_method, envelope_kwargs), **kwargs)


class DefaultConvexHullSegmenter(ConvexHullSegmenter):
    """``convexhull`` registry default: a ConvexHullSegmenter whose envelope is
    selected by name via ``envelope_method`` / ``envelope_kwargs``."""

    def __init__(self, envelope_method: str = "sbs", envelope_kwargs=None, **kwargs):
        super().__init__(_ConfigurableEnvelope(envelope_method, envelope_kwargs), **kwargs)


def _register_envelope_methods():
    """Register all envelope-based methods."""
    global _ENVELOPE_METHODS_REGISTERED
    if not _ENVELOPE_METHODS_REGISTERED:
        register_segmenter('peakdetect', DefaultPeakdetectSegmenter)
        register_segmenter('convexhull', DefaultConvexHullSegmenter)

        # cls_attention pulls neural feature deps; import lazily.
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



def clear_segmenter_cache():
    """
    Release and clear the global segmenter cache.

    Calls ``.release()`` on each cached segmenter that provides one (freeing
    neural model memory) before dropping references. The cache is per-process and
    reuses model instances across files for efficiency; call this at workflow
    boundaries (end of a corpus pass / batch stage) to bound its lifetime, per
    the project's memory-safety policy.

    Example:
        >>> # Process many files with cached models
        >>> for file in files:
        ...     segmenter = get_segmenter('sylber', cache=True)
        ...     segments = segmenter.segment(audio, sr)
        >>>
        >>> # Release models when the workflow is done
        >>> clear_segmenter_cache()
    """
    global _SEGMENTER_CACHE
    for segmenter in _SEGMENTER_CACHE.values():
        release = getattr(segmenter, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
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
