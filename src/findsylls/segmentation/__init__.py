from .dispatch import (
    get_segmenter,
    register_segmenter,
    normalize_segmenter_name,
    list_segmenters,
    list_segmenter_aliases,
)
from .base import (
    BaseSegmenter,
    EnvelopeBasedSegmenter,
    End2EndSegmenter,
    SegmenterProtocol,
)
from .peakdetect_segmenter import (
    segment_peakdetect,
    PeakdetectSegmenter,
)
from .mincut import (
    MinCutSegmenter,
    min_cut,
    min_cut_optimized,
    efficient_extraction_dp_helper,
    get_quantile_borders_helper,
    efficient_extraction,
    extract_mincut_boundaries,
    compute_mincut_presegmentation_trace,
)
from .greedy_cosine import GreedyCosineSegmenter, greedy_cosine_segment
from .cls_attention import CLSAttentionSegmenter
from .presets import ThetaRasanenSegmenter, SylberSegmenter, VGHubertMinCutSegmenter, VGHubertCLSSegmenter
from ..features import (
    FeatureExtractor,
    HuBERTExtractor,
    MFCCExtractor,
    MelSpectrogramExtractor,
    CustomCallableExtractor,
    get_extractor,
)

__all__ = [
    "segment_peakdetect",
    "get_segmenter",
    "register_segmenter",
    "normalize_segmenter_name",
    "list_segmenters",
    "list_segmenter_aliases",
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
    "efficient_extraction_dp_helper",
    "get_quantile_borders_helper",
    "efficient_extraction",
    "extract_mincut_boundaries",
    "compute_mincut_presegmentation_trace",
    "greedy_cosine_segment",
    "CLSAttentionSegmenter",
    # Preset configurations from published papers
    "ThetaRasanenSegmenter",
    "SylberSegmenter",
    "VGHubertMinCutSegmenter",
    "VGHubertCLSSegmenter",
    # Phase 5: Feature extractors
    "FeatureExtractor",
    "HuBERTExtractor",
    "MFCCExtractor",
    "MelSpectrogramExtractor",
    "CustomCallableExtractor",
    "get_extractor",
]