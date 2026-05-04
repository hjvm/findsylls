"""
Greedy Cosine Similarity Segmentation Algorithm

This module provides both the functional greedy cosine algorithm and a Phase 5
wrapper class for mixing the algorithm with any feature extractor.

The algorithm has two phases:
1. Greedy merging: Start with high-energy frames (norm >= threshold), merge adjacent
   frames if cosine similarity >= merge_threshold
2. Boundary refinement: For adjacent segments, sweep the boundary region to find
   optimal split point that maximizes within-segment similarity

Functional API:
    >>> features = np.random.randn(100, 768)
    >>> segments = greedy_cosine_segment(features, merge_threshold=0.8)

Object-Oriented API (Phase 5):
    >>> from findsylls.segmentation.extractors import MFCCExtractor
    >>> extractor = MFCCExtractor()
    >>> segmenter = GreedyCosineSegmenter(extractor, merge_threshold=0.9)
    >>> segments = segmenter.segment(audio, sr=16000)

Original implementation:
https://github.com/Berkeley-Speech-Group/sylber/blob/main/sylber/utils/segment_utils.py

Reference:
Park, C. J., Lai, P. C., & Dupoux, E. (2024). "Sylber: Syllabic Embedding Representation
of Speech from Raw Audio." arXiv preprint arXiv:2410.14336.
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Callable

from .base import End2EndSegmenter, extract_frame_features
from ..features import FeatureExtractor


# =============================================================================
# Core Algorithm (Functional API)
# =============================================================================

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vectors.
    
    Handles both cases:
    - x: (T, D), y: (D,) -> output: (T,)
    - x: (D,), y: (D,) -> output: scalar
    
    Parameters
    ----------
    x : np.ndarray
        First vector or matrix of vectors (T, D) or (D,)
    y : np.ndarray
        Second vector (D,)
        
    Returns
    -------
    np.ndarray
        Cosine similarity values (T,) or scalar
        
    Notes
    -----
    This is equivalent to Sylber's cossim() function:
    ```python
    def cossim(x,y):
        return (x*y).sum(-1)/(((x**2).sum(-1)+1e-8)**.5)/(((y**2).sum(-1)+1e-8)**.5)
    ```
    """
    epsilon = 1e-8
    
    # Compute dot product
    dot_product = (x * y).sum(-1)
    
    # Compute norms
    x_norm = ((x**2).sum(-1) + epsilon) ** 0.5
    y_norm = ((y**2).sum(-1) + epsilon) ** 0.5
    
    return dot_product / (x_norm * y_norm)


def compute_greedy_cosine_norms(features: np.ndarray) -> np.ndarray:
    """Compute frame-wise L2 norms with Sylber-compatible epsilon stabilization."""
    return np.sqrt((features ** 2).sum(-1) + 1e-8)


def compute_greedy_cosine_mask(
    features: np.ndarray,
    norm_threshold: float = 2.6,
    norms: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute non-silence mask used by canonical GreedyCosine segmentation."""
    if norms is None:
        norms = compute_greedy_cosine_norms(features)
    return norms >= norm_threshold


def compute_greedy_cosine_merge_similarity_trace(
    features: np.ndarray,
    norm_threshold: float = 2.6,
    merge_threshold: float = 0.8,
    norms: Optional[np.ndarray] = None,
    start_value: float = 1.0,
    silence_value: float = 0.0,
) -> np.ndarray:
    """
    Compute the canonical phase-1 merge-similarity trace.

    This replays the phase-1 greedy loop from `greedy_cosine_segment` and records,
    for each non-silence frame, the cosine similarity between that frame and the
    current running segment prototype BEFORE thresholding by `merge_threshold`.

    Args:
        features: Feature matrix [T, D].
        norm_threshold: Silence threshold used by canonical algorithm.
        merge_threshold: Merge threshold used by canonical algorithm.
        norms: Optional precomputed norms [T].
        start_value: Value used at the first frame of each active region.
        silence_value: Value used for silence frames.

    Returns:
        Trace [T] where values near/above `merge_threshold` indicate merge-likely
        transitions and low values indicate split-likely transitions.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D array (T, D), got shape {features.shape}")

    t = features.shape[0]
    mask = compute_greedy_cosine_mask(features, norm_threshold=norm_threshold, norms=norms)
    trace = np.full(t, silence_value, dtype=np.float32)

    curr_avg = 0
    seg_cnt = 0
    for i in range(t):
        if not mask[i]:
            curr_avg = 0
            seg_cnt = 0
            trace[i] = silence_value
            continue

        if seg_cnt == 0:
            curr_avg = features[i]
            seg_cnt += 1
            trace[i] = start_value
            continue

        sim = float(cosine_similarity(curr_avg, features[i]))
        trace[i] = sim

        # Replay canonical branch behavior exactly so the next-step prototype
        # matches what segmentation would use.
        if sim >= merge_threshold:
            curr_avg = (curr_avg * seg_cnt + features[i]) / (seg_cnt + 1)
            seg_cnt += 1
        else:
            curr_avg = features[i]
            seg_cnt += 1

    return trace.astype(np.float32)


def greedy_cosine_segments_to_envelope(
    segments: np.ndarray,
    num_frames: int,
    aggregation_method: str = 'segment_mask',
    normalize: bool = True,
    boundary_smooth_frames: int = 1,
) -> np.ndarray:
    """
    Convert canonical GreedyCosine segments into a 1-D pseudo-envelope.

    Args:
        segments: Frame segments [N, 2] with end-exclusive boundaries.
        num_frames: Sequence length.
        aggregation_method: One of {'segment_mask', 'boundary_impulse', 'boundary_density'}.
        normalize: Whether to normalize envelope to [0, 1].
        boundary_smooth_frames: Smoothing half-width for 'boundary_density'.

    Returns:
        Envelope array of shape [num_frames].
    """
    envelope = np.zeros(num_frames, dtype=np.float32)

    if aggregation_method == 'segment_mask':
        for start, end in segments:
            s = max(0, int(start))
            e = min(num_frames, int(end))
            if e > s:
                envelope[s:e] = 1.0

    elif aggregation_method in {'boundary_impulse', 'boundary_density'}:
        for start, _ in segments:
            s = int(start)
            if 0 <= s < num_frames:
                envelope[s] = 1.0

        if aggregation_method == 'boundary_density' and boundary_smooth_frames > 0:
            radius = int(boundary_smooth_frames)
            x = np.arange(-radius, radius + 1, dtype=np.float32)
            sigma = max(1.0, float(boundary_smooth_frames))
            kernel = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
            kernel /= kernel.sum() + 1e-8
            envelope = np.convolve(envelope, kernel, mode='same').astype(np.float32)
    else:
        raise ValueError(
            "aggregation_method must be one of {'segment_mask', 'boundary_impulse', 'boundary_density'}"
        )

    if normalize:
        env_min = envelope.min()
        env_max = envelope.max()
        if env_max > env_min:
            envelope = (envelope - env_min) / (env_max - env_min)

    return envelope.astype(np.float32)


def greedy_cosine_segment(
    features: np.ndarray,
    norm_threshold: float = 2.6,
    merge_threshold: float = 0.8,
    norms: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Segment feature sequence using greedy cosine similarity merging.
    
    This implements Sylber's two-phase greedy segmentation algorithm:
    
    Phase 1: Greedy Merging
    ------------------------
    1. Compute frame-wise norms and mask frames below norm_threshold (silence)
    2. For each high-energy frame:
       - If starting new segment: initialize running average
       - If cosine similarity with running average >= merge_threshold:
         * Merge into current segment, update running average
       - Else:
         * Start new segment
    3. Split segments at silence frames (norm < threshold)
    
    Phase 2: Boundary Refinement
    -----------------------------
    For each pair of adjacent segments:
    1. Merge if cosine similarity between segment means >= merge_threshold
    2. Otherwise, refine boundary:
       - Define search region around boundary (±half segment duration)
       - For each candidate boundary position in search region:
         * Compute sum of cosine similarities of frames to their segment centers
       - Select position that maximizes total similarity
    
    Parameters
    ----------
    features : np.ndarray, shape (T, D)
        Feature sequence where T is number of frames and D is feature dimension
    norm_threshold : float, default=2.6
        Minimum L2 norm for a frame to be considered non-silence.
        Frames with norm < norm_threshold split segments.
        Sylber default: 2.6
    merge_threshold : float, default=0.8
        Minimum cosine similarity for merging:
        - Phase 1: Merge frame into current segment if similarity >= threshold
        - Phase 2: Merge adjacent segments if mean similarity >= threshold
        Sylber default: 0.8
    norms : np.ndarray, optional, shape (T,)
        Pre-computed L2 norms of features. If None, will be computed.
        
    Returns
    -------
    segments : np.ndarray, shape (N, 2)
        Array of [start, end] indices for N segments.
        Indices are frame indices (0-based, end-exclusive).
        
    Examples
    --------
    >>> features = np.random.randn(100, 768)  # 100 frames, 768-dim features
    >>> segments = greedy_cosine_segment(features, norm_threshold=2.6, merge_threshold=0.8)
    >>> segments.shape
    (15, 2)  # Found 15 segments
    >>> segments[0]
    array([5, 12])  # First segment: frames 5-11 (end is exclusive)
    
    Notes
    -----
    This is a direct port of Sylber's get_segment() function:
    https://github.com/Berkeley-Speech-Group/sylber/blob/main/sylber/utils/segment_utils.py
    
    The norm threshold is used WITHIN this algorithm (embedded check), not as
    separate preprocessing. This differs from SD-HuBERT which applies norm masking
    as a preprocessing step before MinCut.
    
    References
    ----------
    .. [1] Park, C. J., Lai, P. C., & Dupoux, E. (2024). "Sylber: Syllabic 
       Embedding Representation of Speech from Raw Audio." arXiv:2410.14336.
    """
    # Input validation
    if features.ndim != 2:
        raise ValueError(f"features must be 2D array (T, D), got shape {features.shape}")
    
    T, D = features.shape
    
    # Compute norms if not provided
    if norms is None:
        norms = compute_greedy_cosine_norms(features)
    
    # Phase 1: Greedy merging with silence detection
    # -----------------------------------------------
    
    # Mask: True for high-energy frames (potential syllable nuclei/segments)
    mask = compute_greedy_cosine_mask(features, norm_threshold=norm_threshold, norms=norms)
    
    # Initialize state
    curr_avg = 0  # Running average of current segment
    seg_cnt = 0   # Frame count in current segment
    segments = []
    s = -1  # Start index of current segment (-1 = no active segment)
    midboundaries = []  # List of (boundary_idx, segment_idx) for refinement
    
    for i in range(T):
        if not mask[i]:
            # Silence frame: close current segment if active
            seg_cnt = 0
            if s > -1:
                segments.append([s, i])
            s = -1
            curr_avg = 0
            seg_cnt = 0
        else:
            # High-energy frame
            if seg_cnt == 0:
                # Start new segment
                curr_avg = features[i]
                seg_cnt += 1
                s = i
            else:
                # Check if frame should merge into current segment
                sim = cosine_similarity(curr_avg, features[i])
                
                if sim >= merge_threshold:
                    # Merge into current segment, update running average
                    curr_avg = (curr_avg * seg_cnt + features[i]) / (seg_cnt + 1)
                    seg_cnt += 1
                else:
                    # Start new segment (greedy split)
                    curr_avg = features[i]
                    seg_cnt += 1
                    segments.append([s, i])
                    midboundaries.append([i, len(segments) - 1])
                    s = i
    
    # Close final segment if active
    if s > -1:
        segments.append([s, T])
    
    # Phase 2: Boundary Refinement
    # -----------------------------
    
    merged = []  # Track which segments to merge
    
    for bd, segi in midboundaries:
        # Check if segment index is valid (might have been merged)
        if segi >= len(segments) - 1:
            continue
        
        # Compute segment means
        prev_seg_features = features[segments[segi][0]:segments[segi][1]]
        next_seg_features = features[segments[segi + 1][0]:segments[segi + 1][1]]
        
        prev_center = prev_seg_features.mean(0)
        next_center = next_seg_features.mean(0)
        
        # Check if segments should be merged
        merge_sim = cosine_similarity(prev_center, next_center)
        
        if merge_sim >= merge_threshold:
            # Merge segments
            segments[segi + 1] = [segments[segi][0], segments[segi + 1][1]]
            merged.append(segi)
            continue
        
        # Refine boundary: search region around original boundary
        # Search range: ±half of segment duration (at least 1 frame)
        prev_half_dur = max(1, (segments[segi][1] - segments[segi][0]) // 2)
        next_half_dur = max(1, (segments[segi + 1][1] - segments[segi + 1][0]) // 2)
        
        # Boundary search region: [s, bd)
        s = max(segments[segi][0], bd - prev_half_dur)
        bd_end = min(segments[segi + 1][1], bd + next_half_dur)
        
        # For each candidate boundary position, compute total similarity
        # sim_sweep[i] = sum of (similarity to prev_center for frames before i)
        #                + (similarity to next_center for frames after i)
        search_region = features[s:bd_end]
        
        sim_prev = cosine_similarity(search_region, prev_center)
        sim_next = cosine_similarity(search_region, next_center)
        
        # Compute sweep: for each split point i, sum similarities
        sim_sweep = [
            sim_prev[:i].sum() + sim_next[i:].sum()
            for i in range(bd_end - s)
        ]
        
        # Find optimal boundary
        opt_b = np.arange(s, bd_end)[np.argmax(sim_sweep)]
        
        # Update segment boundaries
        segments[segi] = [segments[segi][0], opt_b]
        segments[segi + 1] = [opt_b, segments[segi + 1][1]]
    
    # Remove merged segments
    segments = [
        segment for segi, segment in enumerate(segments)
        if segi not in merged
    ]
    
    return np.array(segments, dtype=np.int64)


def greedy_cosine_segment_to_times(
    segments: np.ndarray,
    frame_rate: float = 50.0
) -> np.ndarray:
    """
    Convert frame-based segments to time-based segments.
    
    Parameters
    ----------
    segments : np.ndarray, shape (N, 2)
        Array of [start, end] frame indices
    frame_rate : float, default=50.0
        Frame rate in Hz (frames per second)
        Sylber uses 50 Hz (20ms frames)
        
    Returns
    -------
    time_segments : np.ndarray, shape (N, 2)
        Array of [start_time, end_time] in seconds
        
    Examples
    --------
    >>> segments = np.array([[0, 10], [15, 25]])
    >>> greedy_cosine_segment_to_times(segments, frame_rate=50.0)
    array([[0.  , 0.2 ],
           [0.3 , 0.5 ]])
    """
    return segments / frame_rate


def greedy_cosine_segment_with_features(
    features: np.ndarray,
    norm_threshold: float = 2.6,
    merge_threshold: float = 0.8,
    frame_rate: float = 50.0,
    return_segment_features: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Segment features and optionally extract segment-level features.
    
    This is a convenience function that mimics Sylber's output format.
    
    Parameters
    ----------
    features : np.ndarray, shape (T, D)
        Feature sequence
    norm_threshold : float, default=2.6
        Minimum norm for non-silence frames
    merge_threshold : float, default=0.8
        Minimum cosine similarity for merging
    frame_rate : float, default=50.0
        Frame rate in Hz for time conversion
    return_segment_features : bool, default=False
        If True, also return mean features for each segment
        
    Returns
    -------
    segments : np.ndarray, shape (N, 2)
        Segment boundaries in seconds
    segment_features : np.ndarray, shape (N, D) or None
        Mean features for each segment (only if return_segment_features=True)
        
    Examples
    --------
    >>> features = np.random.randn(100, 768)
    >>> segments, seg_feats = greedy_cosine_segment_with_features(
    ...     features, return_segment_features=True
    ... )
    >>> segments.shape
    (15, 2)  # 15 segments with start/end times
    >>> seg_feats.shape
    (15, 768)  # Mean feature for each segment
    """
    # Get frame-based segments
    frame_segments = greedy_cosine_segment(
        features,
        norm_threshold=norm_threshold,
        merge_threshold=merge_threshold
    )
    
    # Convert to time
    time_segments = greedy_cosine_segment_to_times(frame_segments, frame_rate)
    
    if not return_segment_features:
        return time_segments, None
    
    # Compute segment features (mean of frames in each segment)
    if len(frame_segments) == 0:
        segment_features = np.array([])
    else:
        segment_features = np.stack([
            features[s:e].mean(0) for s, e in frame_segments
        ])
    
    return time_segments, segment_features


# =============================================================================
# Phase 5 Wrapper Class (Object-Oriented API)
# =============================================================================

class GreedyCosineSegmenter(End2EndSegmenter):
    """
    Apply Greedy Cosine algorithm to any feature extractor.
    
    This wrapper allows you to use the greedy cosine similarity merging
    algorithm (from Sylber) with any features.
    
    Args:
        feature_extractor: Feature extractor (FeatureExtractor instance or callable)
        norm_threshold: Minimum feature norm to consider (default: 2.6)
        merge_threshold: Cosine similarity threshold for merging (default: 0.8)
    
    Example:
        >>> from findsylls.segmentation.extractors import MFCCExtractor
        >>> extractor = MFCCExtractor()
        >>> segmenter = GreedyCosineSegmenter(extractor, merge_threshold=0.9)
        >>> segments = segmenter.segment(audio, sr=16000)
    """
    
    def __init__(
        self,
        feature_extractor: Union[FeatureExtractor, Callable],
        norm_threshold: float = 2.6,
        merge_threshold: float = 0.8,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.norm_threshold = norm_threshold
        self.merge_threshold = merge_threshold
        
        # Determine frame rate
        if isinstance(feature_extractor, FeatureExtractor):
            self.frame_rate = feature_extractor.frame_rate
        elif hasattr(feature_extractor, 'frame_rate'):
            self.frame_rate = feature_extractor.frame_rate
        else:
            self.frame_rate = 50.0
    
    def segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        """Segment audio using Greedy Cosine on extracted features."""
        # Extract features through capability-based contract.
        features = extract_frame_features(self.feature_extractor, audio, sr)
        
        # Apply canonical greedy cosine algorithm on extractor states/features.
        frame_segments = greedy_cosine_segment(
            features,
            norm_threshold=self.norm_threshold,
            merge_threshold=self.merge_threshold,
            norms=None  # Computed internally
        )
        
        # Convert frame segments to time segments with feature-based nuclei
        time_segments = []
        for start_frame, end_frame in frame_segments:
            # Nucleus: frame with highest cosine similarity to the segment centroid.
            # The centroid (mean of final [start, end]) is the quantity the greedy
            # loop converges to; argmax cos(f_i, centroid) is the most representative
            # frame by the algorithm's own measure.
            if end_frame - start_frame > 1:
                segment_features = features[start_frame:end_frame]
                centroid = segment_features.mean(axis=0)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                feat_norm = segment_features / (np.linalg.norm(segment_features, axis=1, keepdims=True) + 1e-8)
                nucleus_frame = start_frame + int(np.argmax(feat_norm @ centroid_norm))
            else:
                nucleus_frame = start_frame

            start_sec = start_frame / self.frame_rate
            end_sec = end_frame / self.frame_rate
            nucleus_sec = nucleus_frame / self.frame_rate
            time_segments.append((start_sec, nucleus_sec, end_sec))

        return time_segments
    
    def get_embeddings(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
        """Get segments and their feature embeddings."""
        features = extract_frame_features(self.feature_extractor, audio, sr)
        
        frame_segments = greedy_cosine_segment(
            features,
            norm_threshold=self.norm_threshold,
            merge_threshold=self.merge_threshold,
            norms=None
        )
        
        # Compute embeddings and time segments with feature-based nuclei
        embeddings = []
        time_segments = []
        for start_frame, end_frame in frame_segments:
            segment_features = features[start_frame:end_frame]
            embeddings.append(np.mean(segment_features, axis=0))
            
            # Nucleus: frame with highest cosine similarity to the segment centroid.
            if end_frame - start_frame > 1:
                centroid = segment_features.mean(axis=0)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                feat_norm = segment_features / (np.linalg.norm(segment_features, axis=1, keepdims=True) + 1e-8)
                nucleus_frame = start_frame + int(np.argmax(feat_norm @ centroid_norm))
            else:
                nucleus_frame = start_frame
            
            start_sec = start_frame / self.frame_rate
            end_sec = end_frame / self.frame_rate
            nucleus_sec = nucleus_frame / self.frame_rate
            time_segments.append((start_sec, nucleus_sec, end_sec))
        
        return time_segments, np.array(embeddings)


__all__ = [
    'cosine_similarity',
    'compute_greedy_cosine_norms',
    'compute_greedy_cosine_mask',
    'compute_greedy_cosine_merge_similarity_trace',
    'greedy_cosine_segments_to_envelope',
    'greedy_cosine_segment',
    'greedy_cosine_segment_to_times',
    'greedy_cosine_segment_with_features',
    'GreedyCosineSegmenter'
]
