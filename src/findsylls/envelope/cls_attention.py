"""
CLS Attention Envelope Computer

Computes 1-D envelope traces from transformer CLS token attention using the EXACT SAME
aggregation logic as the canonical CLSAttentionSegmenter.

Critical Architecture Principle:
- Feature extractors provide ONLY raw data (`[n_heads, tgt_len, src_len]` attention tensors)
- Both segmenter AND envelope computer independently consume that raw data
- Both apply identical aggregation transformations to remain semantically aligned
- The envelope 1-D trace represents the SAME reduction policy as the segmenter

This ensures that:
- Envelope peaks align with segmenter boundaries
- Both use per-head quantile thresholding
- Both union important indices across heads
- The 1-D trace is a faithful representation of the segmenter's internal reduction
"""

import numpy as np
from typing import Tuple, Optional

from .base import PseudoEnvelope
from ..features.base import FeatureExtractor
from ..segmentation.cls_attention import (
    compute_cls_attention_importance_masks,
    compute_cls_attention_importance_union,
)


class CLSAttentionEnvelope(PseudoEnvelope):
    """
    Compute 1-D envelope from raw multi-head CLS attention.
    
    Applies IDENTICAL aggregation logic to CLSAttentionSegmenter so that:
    - Peaks in envelope correspond precisely to segment boundaries from segmenter
    - Both use identical per-head quantile thresholding + union strategy
    
    This envelope is primarily useful for:
    1. Visualization of segmenter internal reduction policy
    2. Diagnostics to verify aggregation logic
    3. Experiments comparing with envelope-only segmentation
    4. It is NOT a separate algorithm; it's a visualization of the segmenter's aggregation
    
    Args:
        feature_extractor: FeatureExtractor (must support attention with return_raw=True)
        layer: Which transformer layer to extract attention from (default: None = auto)
        quantile: Per-head importance threshold (default: 0.9, matching the reference
                  and CLSAttentionSegmenter — keeps the top 10% of frames per head)
        aggregation_method: How to convert multi-head binary union to 1-D signal:
                          'max' (default): per-frame maximum attention where important
                          'union_strength': per-frame count of heads marking as important
                          'mean_important': per-frame mean attention in important regions
        normalize: Whether to normalize envelope to [0, 1] (default: True)
    
    Example:
        >>> from findsylls.features import VGHuBERTFeatureExtractor
        >>> from findsylls.envelope import CLSAttentionEnvelope
        >>> 
        >>> # Create envelope computer (same extractor as segmenter for parity)
        >>> extractor = VGHuBERTFeatureExtractor(layer=9, mode='word')
        >>> envelope_computer = CLSAttentionEnvelope(
        ...     extractor,
        ...     layer=9,
        ...     quantile=0.9,
        ...     aggregation_method='max'
        ... )
        >>> 
        >>> # Compute envelope
        >>> envelope, times = envelope_computer.compute(audio, sr=16000)
        >>> 
        >>> # Should align with segments from CLSAttentionSegmenter
        >>> # (peaks in envelope = segment boundaries)
    
    Notes:
        - Requires raw attention via `extract_with_attention(..., return_raw=True)`
        - Default quantile and aggregation MUST match CLSAttentionSegmenter for alignment
        - Time complexity: Feature extraction + O(n_heads * seq_len^2) for attention reduction
        - Memory: O(n_heads * seq_len * src_len) during computation
    
    Reference:
        See CLSAttentionSegmenter.segment_by_cls_attention_raw_matrix for canonical algorithm.
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        layer: Optional[int] = None,
        quantile: float = 0.9,
        aggregation_method: str = 'max',
        normalize: bool = True,
    ):
        super().__init__(feature_extractor=feature_extractor, normalize=normalize)
        self.layer = layer
        self.quantile = quantile
        self.aggregation_method = aggregation_method
        
        # Validate aggregation method
        if aggregation_method not in ['max', 'union_strength', 'mean_important']:
            raise ValueError(
                f"aggregation_method must be 'max', 'union_strength', or 'mean_important', "
                f"got {aggregation_method}"
            )
        
        # Verify extractor supports attention
        if not getattr(feature_extractor, 'supports_attention', False):
            raise RuntimeError(
                f"{feature_extractor.__class__.__name__} does not support attention extraction. "
                f"CLSAttentionEnvelope requires an extractor that can provide raw attention via "
                f"extract_with_attention(..., return_raw=True)"
            )
    
    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CLS attention envelope applying identical aggregation as CLSAttentionSegmenter.
        
        Args:
            audio: Audio waveform (mono, float32)
            sr: Sample rate in Hz
        
        Returns:
            Tuple (envelope, times) where:
                - envelope is 1-D trace of shape (seq_len,) in range [0, 1]
                - times is time positions in seconds of shape (seq_len,)
        
        Raises:
            RuntimeError: If feature extractor cannot provide raw attention
        """
        # Extract raw multi-head attention.
        # Shape is [n_heads, T+1, T+1] for models with CLS token, [n_heads, T, T] otherwise.
        features, raw_attention = self.feature_extractor.extract_with_attention(
            audio,
            sr,
            layer=self.layer,
            return_raw=True,
        )

        has_cls = getattr(self.feature_extractor, 'has_cls_token', False)
        if has_cls:
            # CLS token at position 0; audio frames at positions 1..T.
            cls_attn = raw_attention[:, 0, 1:]   # [n_heads, T]
        else:
            # No CLS token: incoming attention from other frames as saliency proxy.
            # Matches save_seg_feats.py no_cls branch (Peng & Harwath 2022):
            #   attn_weights.sum(1) - diagonal  →  [n_heads, T]
            T_full = raw_attention.shape[1]
            incoming = raw_attention.sum(axis=1)                                          # [n_heads, T]
            self_attn = raw_attention[:, np.arange(T_full), np.arange(T_full)]           # [n_heads, T]
            cls_attn = incoming - self_attn                                               # [n_heads, T]

        n_heads, seq_len = cls_attn.shape    # seq_len = T (audio frames only)

        # Apply canonical aggregation helpers from segmenter module.
        important_masks = compute_cls_attention_importance_masks(
            cls_attn,
            quantile=self.quantile,
        )
        important_union = compute_cls_attention_importance_union(
            cls_attn,
            quantile=self.quantile,
        )

        # Aggregate to 1-D envelope using specified method
        if self.aggregation_method == 'max':
            envelope = np.zeros(seq_len, dtype=np.float32)
            for i in range(seq_len):
                if important_union[i]:
                    envelope[i] = cls_attn[:, i].max()

        elif self.aggregation_method == 'union_strength':
            envelope = np.zeros(seq_len, dtype=np.float32)
            for i in range(seq_len):
                envelope[i] = important_masks[:, i].sum() / n_heads

        elif self.aggregation_method == 'mean_important':
            envelope = np.zeros(seq_len, dtype=np.float32)
            for i in range(seq_len):
                if important_union[i]:
                    envelope[i] = cls_attn[:, i].mean()

        envelope = self._normalize_envelope(envelope)
        times = self._frame_times(seq_len)

        return envelope.astype(np.float32), times.astype(np.float32)
    
    def __repr__(self):
        return (
            f"CLSAttentionEnvelope("
            f"feature_extractor={self.feature_extractor.__class__.__name__}, "
            f"layer={self.layer}, "
            f"quantile={self.quantile}, "
            f"aggregation_method={self.aggregation_method}, "
            f"normalize={self.normalize})"
        )


__all__ = ['CLSAttentionEnvelope']
