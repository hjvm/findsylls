"""
Syllable pooling methods.

Aggregate frame-level features into syllable-level embeddings.

All methods accept syllable boundaries as (start, peak, end) tuples
and return NumPy arrays of shape (num_syllables, embedding_dim).
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


def pool_syllables(
    features: np.ndarray,
    syllables: List[Tuple[float, float, float]],
    sr: int,
    method: str = 'mean',
    hop_length: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Pool frame-level features into syllable-level embeddings.
    
    Args:
        features: Frame-level features, shape (num_frames, feature_dim)
        syllables: List of (start, peak, end) tuples in seconds
        sr: Sample rate in Hz
        method: Pooling method
            - 'mean': Average frames in syllable span (default)
            - 'onc': Onset-Nucleus-Coda template (3× feature_dim)
            - 'max': Max pooling across time
            - 'median': Median pooling
        hop_length: Hop size in samples (for frame timing).
                   If None, computed from features and audio duration.
        **kwargs: Additional method-specific parameters
        
    Returns:
        embeddings: np.ndarray, shape (num_syllables, embedding_dim)
            For 'onc' method: embedding_dim = 3 * feature_dim
            For other methods: embedding_dim = feature_dim
            
    Raises:
        ValueError: If method is unknown or syllables list is invalid
    """
    if len(syllables) == 0:
        # Return empty array with correct shape
        feature_dim = features.shape[1]
        if method == 'onc':
            return np.empty((0, 3 * feature_dim))
        else:
            return np.empty((0, feature_dim))
    
    # Calculate frames per second
    if hop_length is None:
        # Estimate from features
        # Assume features span roughly same duration as audio
        # This is approximate; for exact timing, pass hop_length explicitly
        warnings.warn(
            "hop_length not provided; using estimated frame timing. "
            "For accurate frame alignment, pass hop_length explicitly.",
            UserWarning
        )
        # Estimate: assume last syllable ends near last frame
        last_syll_end = max(end for _, _, end in syllables)
        fps = features.shape[0] / last_syll_end if last_syll_end > 0 else 50.0
    else:
        fps = sr / hop_length
    
    # Dispatch to pooling method
    if method == 'mean':
        return _pool_mean(features, syllables, fps)
    elif method == 'onc':
        return _pool_onc(features, syllables, fps)
    elif method == 'max':
        return _pool_max(features, syllables, fps)
    elif method == 'median':
        return _pool_median(features, syllables, fps)
    else:
        raise ValueError(
            f"Unknown pooling method: '{method}'. "
            f"Supported methods: 'mean', 'onc', 'max', 'median'"
        )


def _pool_mean(
    features: np.ndarray,
    syllables: List[Tuple[float, float, float]],
    fps: float
) -> np.ndarray:
    """
    Mean pooling: average all frames within syllable boundaries.
    
    Args:
        features: (num_frames, feature_dim)
        syllables: List of (start, peak, end) in seconds
        fps: Frames per second
        
    Returns:
        embeddings: (num_syllables, feature_dim)
    """
    embeddings = []
    
    for start, peak, end in syllables:
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(features.shape[0], end_frame)
        
        # Handle edge cases
        if start_frame >= end_frame:
            # Single frame or empty - use nearest frame
            frame_idx = min(start_frame, features.shape[0] - 1)
            embedding = features[frame_idx]
        else:
            # Average frames in range
            embedding = features[start_frame:end_frame].mean(axis=0)
        
        embeddings.append(embedding)
    
    return np.array(embeddings)


def _pool_onc(
    features: np.ndarray,
    syllables: List[Tuple[float, float, float]],
    fps: float
) -> np.ndarray:
    """
    Onset-Nucleus-Coda (ONC) template pooling.
    
    Extracts single frames at specific timepoints within each syllable:
    - Onset: 30% from start to peak
    - Nucleus: peak frame
    - Coda: 70% from peak to end
    
    NOTE: Peaks must be provided by the feature extractor or segmentation method.
    For methods that don't natively provide peaks, they should detect them internally
    (e.g., Sylber uses cosine similarity between frames).
    
    This is a preliminary implementation using fixed proportions. Future enhancement:
    select onset/coda points based on maximal velocity in a frequency band before/after peak.
    
    Args:
        features: (num_frames, feature_dim)
        syllables: List of (start, peak, end) in seconds
        fps: Frames per second
        
    Returns:
        embeddings: (num_syllables, 3 * feature_dim)
    """
    embeddings = []
    frame_hop = 1.0 / fps  # seconds per frame
    max_idx = features.shape[0] - 1
    
    for start, peak, end in syllables:
        # Compute timepoints: onset at 30%, nucleus at peak, coda at 70%
        t_on = start + 0.3 * max(0.0, (peak - start))
        t_pk = peak
        t_cd = peak + 0.7 * max(0.0, (end - peak))
        
        # Convert to frame indices
        idx_on = int(np.clip(round(t_on / frame_hop), 0, max_idx))
        idx_pk = int(np.clip(round(t_pk / frame_hop), 0, max_idx))
        idx_cd = int(np.clip(round(t_cd / frame_hop), 0, max_idx))
        
        # Extract single frames and concatenate
        embedding = np.concatenate([
            features[idx_on],
            features[idx_pk],
            features[idx_cd]
        ])
        embeddings.append(embedding)
    
    return np.array(embeddings)


def _pool_max(
    features: np.ndarray,
    syllables: List[Tuple[float, float, float]],
    fps: float
) -> np.ndarray:
    """
    Max pooling: take maximum activation in each dimension.
    
    Args:
        features: (num_frames, feature_dim)
        syllables: List of (start, peak, end) in seconds
        fps: Frames per second
        
    Returns:
        embeddings: (num_syllables, feature_dim)
    """
    embeddings = []
    
    for start, peak, end in syllables:
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(features.shape[0], end_frame)
        
        # Handle edge cases
        if start_frame >= end_frame:
            # Single frame - use it directly
            frame_idx = min(start_frame, features.shape[0] - 1)
            embedding = features[frame_idx]
        else:
            # Max pooling
            embedding = features[start_frame:end_frame].max(axis=0)
        
        embeddings.append(embedding)
    
    return np.array(embeddings)


def _pool_median(
    features: np.ndarray,
    syllables: List[Tuple[float, float, float]],
    fps: float
) -> np.ndarray:
    """
    Median pooling: take median value in each dimension.
    
    Args:
        features: (num_frames, feature_dim)
        syllables: List of (start, peak, end) in seconds
        fps: Frames per second
        
    Returns:
        embeddings: (num_syllables, feature_dim)
    """
    embeddings = []
    
    for start, peak, end in syllables:
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(features.shape[0], end_frame)
        
        # Handle edge cases
        if start_frame >= end_frame:
            # Single frame - use it directly
            frame_idx = min(start_frame, features.shape[0] - 1)
            embedding = features[frame_idx]
        else:
            # Median pooling
            embedding = np.median(features[start_frame:end_frame], axis=0)
        
        embeddings.append(embedding)
    
    return np.array(embeddings)
