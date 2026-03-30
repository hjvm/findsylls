"""
CLS Attention Segmentation Algorithm.

Uses CLS token attention weights from transformer models to identify syllable/word boundaries.
Based on VG-HuBERT word discovery paper (Peng & Harwath, Interspeech 2022).

The CLS token (first token) in transformer models attends to different positions in the sequence.
Peaks in CLS attention scores correspond to salient positions (word/syllable onsets).

This is genuinely equivalent to an envelope + peak-finding approach.
"""

import numpy as np
from typing import List, Tuple, Optional


def segment_by_cls_attention(
    attention_scores: np.ndarray,
    times: Optional[np.ndarray] = None,
    threshold: float = 0.1,
    min_distance: float = 0.1,
    frame_rate: float = 50.0,
) -> List[Tuple[float, float, float]]:
    """
    Segment audio using CLS token attention scores.
    
    Finds peaks in CLS attention (indicating segment onsets) and creates segments
    between consecutive peaks. Each segment is represented as (start, peak, end)
    where peak is the nucleus/onset position.
    
    Args:
        attention_scores: CLS attention scores, shape (N,)
        times: Time stamps for each frame in seconds, shape (N,). If None, computed from frame_rate
        threshold: Minimum attention value for peak detection (default: 0.1)
        min_distance: Minimum time distance between peaks in seconds (default: 0.1s)
        frame_rate: Frame rate in Hz if times not provided (default: 50.0)
    
    Returns:
        List of (start_time, peak_time, end_time) tuples in seconds
    
    Example:
        >>> # Extract attention from VG-HuBERT
        >>> from findsylls.features import VGHuBERTFeatureExtractor
        >>> extractor = VGHuBERTFeatureExtractor(layer=9, mode='word')
        >>> features, attn = extractor.extract(audio, sr=16000, return_attention=True)
        >>> 
        >>> # Segment by attention peaks
        >>> segments = segment_by_cls_attention(
        ...     attn, 
        ...     threshold=0.15,
        ...     min_distance=0.2  # Words at least 200ms apart
        ... )
    
    Notes:
        - Attention scores are normalized to [0, 1] before peak finding
        - Uses scipy.signal.find_peaks for robust peak detection
        - Segments are created between consecutive peaks
        - If no peaks found, returns single segment for entire signal
        - This approach is equivalent to envelope-based segmentation
          (envelope = attention, threshold = peak height)
    """
    from scipy.signal import find_peaks
    
    # Normalize attention scores to [0, 1]
    attn_norm = (attention_scores - attention_scores.min()) / (
        attention_scores.max() - attention_scores.min() + 1e-8
    )
    
    # Compute times if not provided
    if times is None:
        times = np.arange(len(attn_norm)) / frame_rate
    
    # Convert min_distance from seconds to frames
    spf = times[1] - times[0] if len(times) > 1 else 1.0 / frame_rate
    min_distance_frames = int(min_distance / spf)
    
    # Find peaks in attention scores
    peaks, _ = find_peaks(
        attn_norm,
        height=threshold,
        distance=min_distance_frames
    )
    
    # Create segments between peaks
    if len(peaks) == 0:
        # No peaks found - return entire signal as one segment
        # Peak at midpoint (though not ideal, matches fallback behavior)
        start_time = times[0]
        end_time = times[-1]
        peak_time = (start_time + end_time) / 2.0
        return [(start_time, peak_time, end_time)]
    
    # Add boundaries at start and end
    boundaries = [0] + peaks.tolist() + [len(attn_norm) - 1]
    
    # Create segments with actual peak positions
    segments = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Peak is the boundary between segments (onset position)
        peak_idx = boundaries[i + 1] if i < len(peaks) else boundaries[i]
        
        # Convert to times
        start_time = times[start_idx]
        end_time = times[end_idx]
        peak_time = times[peak_idx]
        
        # Filter out very short segments (< 2 frames)
        if end_idx - start_idx >= 2:
            segments.append((start_time, peak_time, end_time))
    
    return segments


def compute_cls_attention_envelope(
    attention_scores: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert CLS attention scores to an envelope signal.
    
    This is a convenience function that just normalizes attention scores.
    The result can be used with any envelope-based segmentation method.
    
    Args:
        attention_scores: Raw CLS attention scores, shape (N,)
        normalize: Whether to normalize to [0, 1] range
    
    Returns:
        Envelope signal (same shape as input)
    
    Example:
        >>> envelope = compute_cls_attention_envelope(attn_scores)
        >>> # Can now use with any envelope segmentation method
        >>> from findsylls.segmentation import segment_envelope
        >>> segments = segment_envelope(envelope, times, method='peakdetect')
    """
    if normalize:
        envelope = (attention_scores - attention_scores.min()) / (
            attention_scores.max() - attention_scores.min() + 1e-8
        )
    else:
        envelope = attention_scores.copy()
    
    return envelope


__all__ = [
    'segment_by_cls_attention',
    'compute_cls_attention_envelope',
]
