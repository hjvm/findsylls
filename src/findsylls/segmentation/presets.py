"""
Preset segmenter configurations from published papers.

These classes provide pre-configured segmenters that replicate the exact
configurations reported in the original papers. Each class composes a
feature extractor with an algorithm using the paper's hyperparameters.

For flexibility and custom configurations, use the generic wrappers:
- MinCutSegmenter(feature_extractor, **params)
- GreedyCosineSegmenter(feature_extractor, **params)

Available presets:
- SylberSegmenter: Sylber with greedy cosine (Park et al. 2024)
- VGHubertMinCutSegmenter: VG-HuBERT with SSM + MinCut (Peng et al. 2023)
- VGHubertCLSSegmenter: VG-HuBERT with CLS attention (Peng et al. 2023)
- SyllableLMSegmenter: Optimized MinCut on HuBERT (Baade et al. 2024)
"""

from typing import List, Tuple, Optional
import numpy as np

from .base import End2EndSegmenter
from .cls_attention import CLSAttentionSegmenter
from .greedy_cosine import GreedyCosineSegmenter
from .mincut import MinCutSegmenter
from ..features import SylberFeatureExtractor, VGHuBERTFeatureExtractor


class SylberSegmenter(End2EndSegmenter):
    """
    Sylber syllable segmentation (Park et al. 2024).
    
    Replicates the paper's default configuration:
    - Feature extractor: Sylber's fine-tuned HuBERT (layer 9, 768-dim)
    - Algorithm: Greedy cosine similarity with boundary refinement
    - Hyperparameters: norm_threshold=2.6, merge_threshold=0.8
    
    This is a convenience wrapper equivalent to:
        GreedyCosineSegmenter(SylberFeatureExtractor(), norm_threshold=2.6, merge_threshold=0.8)
    
    Reference:
        Park, C. J., Lai, P. C., & Dupoux, E. (2024). "Sylber: Syllabic Embedding 
        Representation of Speech from Raw Audio." arXiv:2410.14336.
    
    Args:
        norm_threshold: Energy threshold for silence detection (default: 2.6)
        merge_threshold: Cosine similarity threshold for merging (default: 0.8)
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)
    
    Example:
        >>> segmenter = SylberSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> # Returns [(start, nucleus, end), ...]
    """
    
    def __init__(
        self,
        norm_threshold: float = 2.6,
        merge_threshold: float = 0.8,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)
        
        # Create feature extractor
        self.feature_extractor = SylberFeatureExtractor(device=device)
        
        # Create algorithm wrapper with paper defaults
        self._segmenter = GreedyCosineSegmenter(
            feature_extractor=self.feature_extractor,
            norm_threshold=norm_threshold,
            merge_threshold=merge_threshold
        )
        
        self.norm_threshold = norm_threshold
        self.merge_threshold = merge_threshold
        self.device = device
    
    def segment(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio into syllables using Sylber method.
        
        Args:
            audio: Audio waveform (mono)
            sr: Sample rate
            **kwargs: Override default parameters (norm_threshold, merge_threshold)
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        return self._segmenter.segment(audio, sr, **kwargs)


class VGHubertMinCutSegmenter(End2EndSegmenter):
    """
    VG-HuBERT with MinCut segmentation (Peng et al. 2023).

    Uses VG-HuBERT features with graph-based MinCut segmentation:
    - Feature extractor: VG-HuBERT (layer 8 for syllables, 9 for words)
    - Algorithm: SSM-based MinCut with K = ceil(duration / sec_per_syllable)
    - Post-processing: greedy cosine merge (minCutMerge-0.3, Peng et al. 2023)
    - Hyperparameters: sec_per_syllable=0.20, merge_threshold=0.3

    Note: Uses use_reference=False (SSM path) to match the original paper, which
    predates SyllableLM's DP algorithm. The DP's delta/quantile parameters are
    calibrated for Data2Vec2 features and over-segment VGHuBERT features.
    The merge_threshold=0.3 replicates the minCutMerge-0.3 configuration used in
    the paper's main results (save_seg_feats_mincut.py). SyllableLM's DP does not
    use this post-processing step.

    Algorithm note: defaults to use_optimized=False (the reference Cython min_cut
    algorithm) for exact segment-count parity with the paper. min_cut_optimized
    backtracks from frame N rather than N-1, which can shift the second-to-last
    boundary by 1 frame and alter the final merge decision. Set use_optimized=True
    for ~50x speedup with near-identical (but not bit-exact) boundaries.

    This is equivalent to:
        MinCutSegmenter(VGHuBERTFeatureExtractor(mode='syllable'),
                        sec_per_syllable=0.20, use_reference=False,
                        use_optimized=False, merge_threshold=0.3)

    Reference:
        Peng, P., et al. (2023). "Syllable Discovery and Cross-Lingual Generalization
        in a Visually Grounded, Self-Supervised Speech Model." Interspeech 2023.

    Args:
        layer: VG-HuBERT layer (default: None = auto-select from mode)
        mode: Granularity - 'syllable' or 'word' (default: 'syllable')
        sec_per_syllable: Target syllable duration (default: 0.20)
        merge_threshold: Cosine similarity threshold for post-merge (default: 0.3).
                         Set to None to disable merging.
        use_optimized: Use optimized MinCut (default: False for reference parity;
                       set True for ~50x speedup with near-identical results).
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)

    Example:
        >>> # Syllable segmentation (auto layer=8, with merge)
        >>> segmenter = VGHubertMinCutSegmenter(mode='syllable')
        >>> segments = segmenter.segment(audio, sr=16000)
        >>>
        >>> # Without post-merge (plain MinCut)
        >>> segmenter_plain = VGHubertMinCutSegmenter(merge_threshold=None)
        >>>
        >>> # Word segmentation (auto layer=9)
        >>> word_segmenter = VGHubertMinCutSegmenter(mode='word', sec_per_syllable=0.4)
    """

    def __init__(
        self,
        layer: Optional[int] = None,
        mode: str = 'syllable',
        sec_per_syllable: float = 0.20,
        merge_threshold: Optional[float] = 0.3,
        use_optimized: bool = False,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)

        # Create feature extractor (layer auto-selected from mode)
        self.feature_extractor = VGHuBERTFeatureExtractor(
            layer=layer,
            mode=mode,
            device=device
        )

        # Create MinCut wrapper using the SSM path (use_reference=False) to match
        # Peng et al. 2023, which predates SyllableLM's DP. The DP's delta/quantile
        # are calibrated for Data2Vec2 features and saturate on VGHuBERT features.
        self._segmenter = MinCutSegmenter(
            feature_extractor=self.feature_extractor,
            sec_per_syllable=sec_per_syllable,
            use_reference=False,
            use_optimized=use_optimized,
            merge_threshold=merge_threshold,
        )
        
        self.layer = self.feature_extractor.layer
        self.mode = mode
        self.sec_per_syllable = sec_per_syllable
        self.merge_threshold = merge_threshold
        self.use_optimized = use_optimized
        self.device = device

    def segment(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio using VG-HuBERT + MinCut.

        Args:
            audio: Audio waveform (mono)
            sr: Sample rate
            **kwargs: Override default parameters

        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        return self._segmenter.segment(audio, sr, **kwargs)


class VGHubertCLSSegmenter(End2EndSegmenter):
    """
    VG-HuBERT with CLS attention segmentation (Peng & Harwath 2022).

    Replicates the word-discovery paper's canonical configuration:
    - Feature extractor: VG-HuBERT word checkpoint (layer 9)
    - Algorithm: CLS-token attention thresholding with per-head quantile union

    This is the configuration described in Peng & Harwath (Interspeech 2022), which
    introduced CLS attention segmentation using the word-level VG-HuBERT checkpoint.
    Use mode='syllable' only if you want layer-8 syllable-checkpoint features with
    CLS attention — that is not the canonical published configuration.

    Reference:
        Peng, P., & Harwath, D. (2022). "Self-Supervised Representation Learning for
        Speech Using Visual Grounding and Masked Language Modeling."
        Interspeech 2022. (word-discovery repo)

    Args:
        layer: VG-HuBERT layer (default: None = auto-select from mode)
        mode: Checkpoint and layer selection — 'word' (default, layer 9, word checkpoint)
              or 'syllable' (layer 8, syllable checkpoint).
        quantile: Per-head importance threshold (default: 0.9, matching the reference
                  save_seg_feats.py default of threshold=0.90 — keeps top 10% per head).
        min_distance: Optional gap tolerance in seconds for merging adjacent segments
                      (default: 0.0 = disabled, matching the reference which has no merging).
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)

    Example:
        >>> # Canonical word-discovery CLS segmentation (default)
        >>> segmenter = VGHubertCLSSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
        >>>
        >>> # Syllable-checkpoint variant (non-canonical)
        >>> syl_segmenter = VGHubertCLSSegmenter(mode='syllable')
    """

    def __init__(
        self,
        layer: Optional[int] = None,
        mode: str = 'word',
        quantile: float = 0.9,
        min_distance: float = 0.0,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)
        self._segmenter = CLSAttentionSegmenter(
            feature_extractor=VGHuBERTFeatureExtractor(
                layer=layer,
                mode=mode,
                device=device,
            ),
            layer=layer,
            mode=mode,
            quantile=quantile,
            min_distance=min_distance,
            device=device,
            sample_rate=sample_rate,
        )

        self.layer = self._segmenter.layer
        self.mode = mode
        self.quantile = quantile
        self.min_distance = min_distance
        self.device = device

    def segment(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio using VG-HuBERT CLS attention.

        Args:
            audio: Audio waveform (mono)
            sr: Sample rate
            **kwargs: Override parameters (quantile, min_distance)
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        return self._segmenter.segment(audio, sr, **kwargs)


class SyllableLMSegmenter(End2EndSegmenter):
    """
    SyllableLM optimized segmentation (Baade et al. 2024).

    Uses the SyllableLM DP MinCut (use_reference=True), which provides
    a 5-6× speedup over the original O(n²) MinCut via dynamic programming.

    Replicates the paper's configuration:
    - Feature extractor: HuBERT base (layer 9, 768-dim)
    - Algorithm: SyllableLM DP MinCut (use_reference=True)
    - Hyperparameters: delta=0.0033, quantile=0.75, min_hop=3, max_hop=50

    Warning: delta=0.0033 and quantile=0.75 were calibrated against Data2Vec2
    features (8.33 Hz). With standard HuBERT features (50 Hz), the DP may
    over-segment because the 0.75-quantile segment costs are well above delta.
    The original SyllableLM paper used Data2Vec2, not vanilla HuBERT; using
    HuBERT here matches the paper's intent but may require re-tuning delta/quantile
    for production use.

    Reference:
        Baade, A., et al. (2024). "SyllableLM: Learning Coarse Semantic Units
        for Speech Language Models." arXiv:2410.04029.

    Args:
        sec_per_syllable: Target syllable duration (default: 0.22)
        min_hop: Minimum segment length in frames (default: 3)
        max_hop: Maximum segment length in frames (default: 50)
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)

    Example:
        >>> segmenter = SyllableLMSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
    """

    def __init__(
        self,
        sec_per_syllable: float = 0.20,
        min_hop: int = 3,
        max_hop: int = 50,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)

        # Use vanilla HuBERT (not fine-tuned like Sylber)
        from ..features import HuBERTExtractor
        self.feature_extractor = HuBERTExtractor(device=device)

        # Explicitly opt in to the SyllableLM DP path (use_reference=True).
        # The MinCutSegmenter default is now use_reference=False (SSM path),
        # so this must be explicit or SyllableLMSegmenter silently changes behavior.
        self._segmenter = MinCutSegmenter(
            feature_extractor=self.feature_extractor,
            sec_per_syllable=sec_per_syllable,
            use_reference=True,
            use_optimized=True,
            min_hop=min_hop,
            max_hop=max_hop,
        )
        
        self.sec_per_syllable = sec_per_syllable
        self.min_hop = min_hop
        self.max_hop = max_hop
        self.device = device
    
    def segment(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio into syllables using SyllableLM optimized method.
        
        Args:
            audio: Audio waveform (mono)
            sr: Sample rate
            **kwargs: Override default parameters
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        return self._segmenter.segment(audio, sr, **kwargs)
