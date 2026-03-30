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
    - Algorithm: MinCut on self-similarity matrix (SSM) with dynamic programming
    - Hyperparameters: sec_per_syllable=0.22
    
    This is equivalent to:
        MinCutSegmenter(VGHuBERTFeatureExtractor(mode='syllable'), sec_per_syllable=0.22)
    
    Reference:
        Peng, P., et al. (2023). "Syllable Discovery and Cross-Lingual Generalization 
        in a Visually Grounded, Self-Supervised Speech Model." Interspeech 2023.
    
    Args:
        layer: VG-HuBERT layer (default: None = auto-select from mode)
        mode: Granularity - 'syllable' or 'word' (default: 'syllable')
        sec_per_syllable: Target syllable duration (default: 0.22)
        use_optimized: Use optimized MinCut (20-50× faster, default: True)
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)
    
    Example:
        >>> # Syllable segmentation (auto layer=8)
        >>> segmenter = VGHubertMinCutSegmenter(mode='syllable')
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> 
        >>> # Word segmentation (auto layer=9)
        >>> word_segmenter = VGHubertMinCutSegmenter(mode='word', sec_per_syllable=0.4)
    """
    
    def __init__(
        self,
        layer: Optional[int] = None,
        mode: str = 'syllable',
        sec_per_syllable: float = 0.22,
        use_optimized: bool = True,
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
        
        # Create MinCut wrapper
        self._segmenter = MinCutSegmenter(
            feature_extractor=self.feature_extractor,
            sec_per_syllable=sec_per_syllable,
            use_optimized=use_optimized
        )
        
        self.layer = self.feature_extractor.layer
        self.mode = mode
        self.sec_per_syllable = sec_per_syllable
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
    VG-HuBERT with CLS attention segmentation (Peng et al. 2023).
    
    Uses VG-HuBERT CLS token attention for boundary detection:
    - Feature extractor: VG-HuBERT (layer 8 for syllables, 9 for words)
    - Algorithm: Peak detection in CLS attention scores
    - Default thresholds tuned for syllable segmentation
    
    The CLS token (first token) in transformers attends to different sequence positions.
    Peaks in CLS attention correspond to salient positions (syllable/word onsets).
    
    Reference:
        Peng, P., et al. (2023). "Syllable Discovery and Cross-Lingual Generalization 
        in a Visually Grounded, Self-Supervised Speech Model." Interspeech 2023.
    
    Args:
        layer: VG-HuBERT layer (default: None = auto-select from mode)
        mode: Granularity - 'syllable' or 'word' (default: 'syllable')
        attn_threshold: Attention peak height threshold (default: 0.1)
        min_distance: Minimum distance between peaks in seconds (default: 0.05 for syllables)
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)
    
    Example:
        >>> # Syllable segmentation (auto layer=8, spacing ~50ms)
        >>> segmenter = VGHubertCLSSegmenter(mode='syllable')
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> 
        >>> # Word segmentation (auto layer=9, spacing ~200ms)
        >>> word_segmenter = VGHubertCLSSegmenter(mode='word', min_distance=0.2)
    """
    
    def __init__(
        self,
        layer: Optional[int] = None,
        mode: str = 'syllable',
        attn_threshold: float = 0.1,
        min_distance: float = 0.05,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)
        
        # Create feature extractor (will auto-load eager attention when needed)
        self.feature_extractor = VGHuBERTFeatureExtractor(
            layer=layer,
            mode=mode,
            device=device
        )
        
        self.layer = self.feature_extractor.layer
        self.mode = mode
        self.attn_threshold = attn_threshold
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
            **kwargs: Override parameters (attn_threshold, min_distance)
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
        """
        from ..segmentation.cls_attention import segment_by_cls_attention
        
        # Extract features and CLS attention scores
        features, cls_attention = self.feature_extractor.extract(
            audio, sr, return_attention=True
        )
        
        # Compute times array
        frame_rate = self.feature_extractor.frame_rate
        times = np.arange(len(cls_attention)) / frame_rate
        
        # Override defaults with kwargs
        threshold = kwargs.get('attn_threshold', self.attn_threshold)
        min_dist = kwargs.get('min_distance', self.min_distance)
        
        # Segment using CLS attention peaks
        segments = segment_by_cls_attention(
            attention_scores=cls_attention,
            times=times,
            threshold=threshold,
            min_distance=min_dist,
            frame_rate=frame_rate
        )
        
        return segments


class SyllableLMSegmenter(End2EndSegmenter):
    """
    SyllableLM optimized segmentation (Baade et al. 2024).
    
    Uses the optimized MinCut implementation from SyllableLM which provides
    20-50× speedup over the original MinCut while producing identical results.
    
    Replicates the paper's configuration:
    - Feature extractor: HuBERT base (layer 9, 768-dim)
    - Algorithm: Optimized MinCut with segment length constraints
    - Hyperparameters: sec_per_syllable=0.22, min_hop=3, max_hop=50
    
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
        sec_per_syllable: float = 0.22,
        min_hop: int = 3,
        max_hop: int = 50,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)
        
        # Use vanilla HuBERT (not fine-tuned like Sylber)
        from ..features import HuBERTExtractor
        self.feature_extractor = HuBERTExtractor(device=device)
        
        # Create MinCut wrapper with optimized algorithm
        self._segmenter = MinCutSegmenter(
            feature_extractor=self.feature_extractor,
            sec_per_syllable=sec_per_syllable,
            use_optimized=True,  # Always use optimized version
            min_hop=min_hop,
            max_hop=max_hop
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
