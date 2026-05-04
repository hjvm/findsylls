"""
CLS Attention Segmentation Algorithm (Auxiliary/Legacy).

Uses CLS token attention weights from transformer models to identify syllable/word boundaries.
Based on VG-HuBERT word discovery paper (Peng & Harwath, Interspeech 2022).

The CLS token (first token) in transformer models attends to different positions in the sequence.
Peaks in CLS attention scores correspond to salient positions (word/syllable onsets).

The canonical path is the raw-matrix algorithm, which preserves the original
multi-head quantile/union semantics. A separate envelope helper remains only
for visualization or diagnostics and is not the CLSAttentionSegmenter
implementation.

Reference algorithm (save_seg_feats.py, Peng & Harwath 2022):
    cls_attn_weights = attn_weights[:, 0, 1:]          # [n_heads, T]
    threshold_value = quantile(cls_attn_weights, q, dim=-1, keepdim=True)  # [n_heads, 1]
    important_idx = where((cls_attn_weights >= threshold_value).sum(0) > 0)
"""

import numpy as np
from typing import List, Tuple, Optional

from .base import End2EndSegmenter
from ..features import FeatureExtractor, get_extractor


def compute_cls_attention_importance_masks(
    cls_attention: np.ndarray,
    quantile: float = 0.5,
) -> np.ndarray:
    """
    Compute per-head binary importance masks from CLS-token attention.

    Args:
        cls_attention: CLS-token attention weights [n_heads, T].
                       Extract from raw attention via raw_attention[:, 0, 1:].
        quantile: Per-head importance threshold in [0, 1].

    Returns:
        Boolean masks with shape [n_heads, T].
    """
    n_heads = cls_attention.shape[0]
    important_masks = []
    for head_idx in range(n_heads):
        head_attn = cls_attention[head_idx]         # [T]
        thresh = np.quantile(head_attn, quantile)   # scalar, per-head over T values
        head_important = head_attn >= thresh         # [T]
        important_masks.append(head_important)
    return np.stack(important_masks, axis=0)         # [n_heads, T]


def compute_cls_attention_importance_union(
    cls_attention: np.ndarray,
    quantile: float = 0.5,
) -> np.ndarray:
    """
    Compute union mask over heads from CLS-token attention (canonical reduction).

    A frame is important if ANY head's CLS attention exceeds its per-head threshold.

    Args:
        cls_attention: CLS-token attention weights [n_heads, T].
        quantile: Per-head importance threshold in [0, 1].

    Returns:
        Boolean union mask with shape [T].
    """
    important_masks = compute_cls_attention_importance_masks(cls_attention, quantile=quantile)
    return np.logical_or.reduce(important_masks, axis=0)


def segment_by_cls_attention_raw_matrix(
    attention: np.ndarray,
    times: np.ndarray,
    quantile: float = 0.9,
    merge_valley_tol: float = 0.0,
    has_cls_token: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Canonical CLS attention segmentation using raw multi-head attention.

    Implements the original algorithm from VG-HuBERT (Peng & Harwath 2022):
    1. Extract per-frame importance: [n_heads, T]
       - With CLS token (has_cls_token=True): attention[:, 0, 1:] — CLS row
       - Without CLS token (has_cls_token=False): attention.sum(1) - diagonal
         i.e. total incoming attention from other frames (Peng & Harwath 2022,
         save_seg_feats.py no_cls branch, used for plain HuBERT baseline)
    2. For each head: compute per-head quantile threshold over T values
    3. Binarize: frame is important if head_attn >= threshold
    4. Union across heads: frame is important if ANY head marks it
    5. Group contiguous important frames into segments
    6. Filter single-frame segments (unless all segments are single-frame)
    7. Return (start, peak, end) with peak = argmax of sum-of-head-attention

    Args:
        attention: Raw multi-head attention [n_heads, T+1, T+1] if has_cls_token,
                   or [n_heads, T, T] if not.
        times: Time positions in seconds for audio frames, shape [T].
        quantile: Per-head importance threshold (default 0.9, matching reference).
        merge_valley_tol: Optional gap tolerance in seconds for merging adjacent
                          segments (0.0 = disabled, matching reference behavior).
        has_cls_token: Whether the model has an injected CLS token at position 0.
                       True (default): VG-HuBERT — use attention[:, 0, 1:].
                       False: plain HuBERT — use incoming-attention formula.

    Returns:
        List of (start_time, peak_time, end_time) tuples in seconds.
    """
    if has_cls_token:
        # CLS token at position 0; audio frames at positions 1..T.
        cls_attn = attention[:, 0, 1:]   # [n_heads, T]
    else:
        # No CLS token: use incoming attention from other frames as saliency proxy.
        # Matches save_seg_feats.py no_cls branch (Peng & Harwath 2022):
        #   attn_weights.sum(1) - diagonal  →  [n_heads, T]
        T_full = attention.shape[1]
        incoming = attention.sum(axis=1)                                    # [n_heads, T]
        self_attn = attention[:, np.arange(T_full), np.arange(T_full)]     # [n_heads, T]
        cls_attn = incoming - self_attn                                     # [n_heads, T]

    T = cls_attn.shape[1]

    if len(times) != T:
        raise ValueError(f"times length {len(times)} != audio frame count {T}")

    important_union = compute_cls_attention_importance_union(cls_attn, quantile=quantile)

    diff = np.diff(important_union.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if important_union[0]:
        starts = np.concatenate([[0], starts])
    if important_union[-1]:
        ends = np.concatenate([ends, [T]])

    if len(starts) == 0:
        cls_attn_sum = cls_attn.sum(axis=0)
        peak_time = times[int(np.argmax(cls_attn_sum))]
        return [(times[0], peak_time, times[-1])]

    if merge_valley_tol > 0 and len(starts) > 1:
        spf = times[1] - times[0] if len(times) > 1 else 1.0 / 50.0
        merge_tol_frames = int(merge_valley_tol / spf)

        merged_starts = [starts[0]]
        merged_ends = [ends[0]]
        for i in range(1, len(starts)):
            gap = starts[i] - merged_ends[-1]
            if gap <= merge_tol_frames:
                merged_ends[-1] = ends[i]
            else:
                merged_starts.append(starts[i])
                merged_ends.append(ends[i])
        starts = np.array(merged_starts)
        ends = np.array(merged_ends)

    # Sum CLS attention across heads for peak detection.
    # Matches reference pool='max': argmax(cls_attn_weights.sum(0)[t_s:t_e]).
    cls_attn_sum = cls_attn.sum(axis=0)   # [T]

    boundaries_all = list(zip(starts.tolist(), ends.tolist()))
    # Filter single-frame segments unless all are single-frame (reference boundaries_ex1).
    boundaries_ex1 = [(s, e) for s, e in boundaries_all if e - s > 1]
    segments_src = boundaries_ex1 if len(boundaries_ex1) > 0 else boundaries_all

    segments = []
    for start_idx, end_idx in segments_src:
        start_time = times[start_idx]
        end_time = times[min(end_idx, T) - 1]
        peak_local = int(np.argmax(cls_attn_sum[start_idx:end_idx]))
        peak_idx = start_idx + peak_local
        peak_time = times[peak_idx]
        segments.append((start_time, peak_time, end_time))

    return segments


class CLSAttentionSegmenter(End2EndSegmenter):
    """
    First-class class wrapper for CLS attention segmentation.

    This aligns CLS attention with other class-based segmenters
    (e.g., MinCutSegmenter, GreedyCosineSegmenter).
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        feature_type: str = 'vg_hubert',
        feature_kwargs: Optional[dict] = None,
        layer: Optional[int] = None,
        mode: str = 'syllable',
        attention_aggregate: str = 'max',
        quantile: float = 0.9,
        min_distance: float = 0.0,
        device: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        super().__init__(sample_rate=sample_rate)
        if feature_extractor is None:
            extractor_kwargs = dict(feature_kwargs or {})

            feature_key = str(feature_type).lower().replace('-', '_')
            if feature_key in {'vg_hubert', 'vghubert'}:
                extractor_kwargs.setdefault('layer', layer)
                extractor_kwargs.setdefault('mode', mode)
                extractor_kwargs.setdefault('device', device)
            elif feature_key in {'hubert', 'sylber'}:
                extractor_kwargs.setdefault('device', device)

            self.feature_extractor = get_extractor(feature_type, **extractor_kwargs)
        else:
            self.feature_extractor = feature_extractor
        self.layer = layer
        self.attention_aggregate = attention_aggregate
        self.quantile = quantile
        self.min_distance = min_distance

    def segment(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        **kwargs,
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio using CLS attention with canonical raw-matrix algorithm.

        Applies per-head quantile/union thresholding matching Peng & Harwath 2022
        (threshold=0.9 in the reference = keep the top 10% of frames per head).
        Pass merge_valley_tol>0 to optionally merge segments separated by short gaps.
        """
        if not getattr(self.feature_extractor, 'supports_attention', False):
            raise RuntimeError(
                "CLS attention segmentation requires a feature extractor that supports "
                "attention extraction"
            )

        attention_layer = kwargs.get('layer', self.layer)

        # Returns [n_heads, T+1, T+1] (with CLS) or [n_heads, T, T] (no CLS).
        _, raw_attention = self.feature_extractor.extract_with_attention(
            audio,
            sr,
            layer=attention_layer,
            return_raw=True,
        )

        has_cls = getattr(self.feature_extractor, 'has_cls_token', False)
        frame_rate = self.feature_extractor.frame_rate
        # T is the number of audio frames, excluding any CLS token position.
        T = raw_attention.shape[1] - (1 if has_cls else 0)
        times = np.arange(T) / frame_rate

        return segment_by_cls_attention_raw_matrix(
            attention=raw_attention,
            times=times,
            quantile=kwargs.get('quantile', self.quantile),
            merge_valley_tol=kwargs.get('merge_valley_tol', self.min_distance),
            has_cls_token=has_cls,
        )


__all__ = [
    'compute_cls_attention_importance_masks',
    'compute_cls_attention_importance_union',
    'segment_by_cls_attention_raw_matrix',
    'CLSAttentionSegmenter',
]
