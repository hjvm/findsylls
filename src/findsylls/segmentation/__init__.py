from .dispatch import segment_envelope, get_segmenter, register_segmenter
from .base import (
    BaseSegmenter,
    EnvelopeBasedSegmenter,
    End2EndSegmenter,
    SegmenterProtocol,
)
from .peakdetect_segmenter import (
    segment_peakdetect, 
    segment_billauer, 
    PeakdetectSegmenter,
)
from .mincut import MinCutSegmenter, min_cut, min_cut_optimized
from .greedy_cosine import GreedyCosineSegmenter, greedy_cosine_segment
from .cls_attention import segment_by_cls_attention, compute_cls_attention_envelope
from ..features import (
    FeatureExtractor,
    HuBERTExtractor,
    MFCCExtractor,
    MelSpectrogramExtractor,
    CustomCallableExtractor,
    get_extractor,
)

# Preset configurations (lazy import to avoid dependency errors)
try:
    from .presets import (
        SylberSegmenter,
        VGHubertMinCutSegmenter,
        VGHubertCLSSegmenter,
        SyllableLMSegmenter,
    )
    _PRESET_SEGMENTERS_AVAILABLE = True
except ImportError:
    _PRESET_SEGMENTERS_AVAILABLE = False

__all__ = [
    "segment_envelope",
    "segment_peakdetect",
    "segment_billauer",  # Backward compatibility
    "get_segmenter",
    "register_segmenter",
    "BaseSegmenter",
    "EnvelopeBasedSegmenter",
    "End2EndSegmenter",
    "SegmenterProtocol",
    # Envelope-based (Phase 5)
    "PeakdetectSegmenter",
    # Feature-based (Phase 5)
    "MinCutSegmenter",
    "GreedyCosineSegmenter",
    "min_cut",
    "min_cut_optimized",
    "greedy_cosine_segment",
    "segment_by_cls_attention",
    "compute_cls_attention_envelope",
    # Phase 5: Feature extractors
    "FeatureExtractor",
    "HuBERTExtractor",
    "MFCCExtractor",
    "MelSpectrogramExtractor",
    "CustomCallableExtractor",
    "get_extractor",
]

# Add preset segmenters if available
if _PRESET_SEGMENTERS_AVAILABLE:
    __all__.extend([
        "SylberSegmenter",
        "VGHubertMinCutSegmenter",
        "VGHubertCLSSegmenter",
        "SyllableLMSegmenter",
    ])