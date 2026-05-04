"""MinCut segmentation algorithms and wrappers.

This module now exposes two MinCut families:
1. Legacy fixed-K MinCut (`min_cut`, `min_cut_optimized`) retained for compatibility.
2. Canonical dynamic MinCut parity path based on the latest SyllableLM DP +
   quantile-border search semantics.

The dynamic path is the canonical implementation used by `MinCutSegmenter` and
by the MinCut pseudo-envelope to derive pre-segmentation boundary traces.
"""

import warnings
import numpy as np
import torch
from typing import List, Tuple, Union, Callable, Optional

from .base import End2EndSegmenter, extract_frame_features
from ..features import FeatureExtractor


MINCUT_THRESHOLD = 1.0 / 0.10 / 50.0
MINCUT_MODEL_PRESETS = {
    '8.33hz': {'delta': 0.0033, 'quantile': 0.75},
    '6.25hz': {'delta': 0.0028, 'quantile': 0.75},
    '5.0hz': {'delta': 0.0019, 'quantile': 0.75},
}

# Default sec_per_syllable matches the syllable-discovery reference (Peng et al. 2023,
# save_seg_feats_mincut.py: secPerSyllable=0.2).
_DEFAULT_SEC_PER_SYLLABLE = 0.20


# =============================================================================
# Core Algorithm (Functional API)
# =============================================================================

def min_cut(ssm: np.ndarray, K: int) -> List[int]:
    """
    Partition a self-similarity matrix into K segments using min-cut.
    
    This implementation uses dynamic programming to find boundary points that
    minimize inter-segment similarity while maximizing intra-segment similarity.
    
    Args:
        ssm: Self-similarity matrix of shape (N, N) where ssm[i,j] represents
             similarity between frame i and frame j. Should be non-negative.
        K: Number of boundary points to find (returns K boundaries, creating K-1 segments)
           
    Returns:
        List of boundary frame indices (length K), including start (0) and end (N).
        These define K-1 segments: [bound[0]:bound[1]], [bound[1]:bound[2]], ...
        
    Example:
        >>> features = np.random.randn(100, 768)  # 100 frames, 768-dim features
        >>> ssm = features @ features.T
        >>> ssm = ssm - np.min(ssm) + 1e-7  # make non-negative
        >>> boundaries = min_cut(ssm, K=11)  # Get 11 boundaries (10 segments)
        >>> boundaries
        [0, 8, 19, 31, ..., 100]
    """
    N = ssm.shape[0]
    
    # C[i, k] = minimum cost to partition ssm[0:i] into k segments
    # B[i, k] = best split point for partition ending at i with k segments
    C = np.ones((N, K), dtype=np.float64) * np.inf
    B = np.zeros((N, K), dtype=np.int32)
    
    # Base case: 0 segments up to frame 0
    C[0, 0] = 0.0
    
    # Dynamic programming: for each position i and number of segments k
    for i in range(1, N):
        # Precompute costs for all possible segment starts j to current position i
        # For segment [j:i]:
        #   - intra_sim = sum of similarities within [j:i]
        #   - inter_sim = sum of similarities between [j:i] and rest of frames
        temp = []
        for j in range(i):
            # Intra-segment similarity (within [j:i])
            intra_sim = ssm[j:i, j:i].sum() / 2.0
            
            # Inter-segment similarity (between [j:i] and everything else)
            inter_sim = ssm[j:i, :j].sum() + ssm[j:i, i:].sum()
            
            temp.append((intra_sim, inter_sim))
        
        # Try adding segments
        for k in range(1, K):
            # For each possible split point j, compute total cost
            obj = []
            for j, (intra_sim, inter_sim) in enumerate(temp):
                # Cost = previous cost + ratio of inter-segment to total similarity
                # We want to minimize inter-segment connections (cuts)
                total_sim = intra_sim + inter_sim
                if total_sim > 0:
                    cut_cost = inter_sim / total_sim
                else:
                    cut_cost = 0.0
                obj.append(C[j, k - 1] + cut_cost)
            
            # Choose best split point
            ind = np.argmin(obj)
            B[i, k] = ind
            C[i, k] = obj[ind]
    
    # Backtrack to find boundaries
    boundary = []
    prev_b = N - 1
    boundary.append(prev_b)
    
    for k in range(K - 1, 0, -1):
        prev_b = B[prev_b, k]
        boundary.append(prev_b)
    
    boundary = boundary[::-1]  # Reverse to get chronological order
    
    return boundary


def min_cut_optimized(ssm: np.ndarray, K: int, min_hop: int = 3, max_hop: int = 50) -> List[int]:
    """
    Optimized MinCut implementation from Baade et al. (2024) SyllableLM.
    
    Provides ~20-50x speedup over original with identical segmentation quality:
    - Cumulative sum preprocessing for O(1) range queries
    - Segment length constraints (min_hop, max_hop) for speech
    - 5-component cost calculation (interior, left, top, bottom, right)
    
    Args:
        ssm: Self-similarity matrix of shape (N, N) where ssm[i,j] represents
             similarity between frame i and frame j. Should be non-negative.
        K: Number of boundary points to find (returns K boundaries, creating K-1 segments)
        min_hop: Minimum segment length in frames (default: 3 frames ~ 60ms at 50fps)
        max_hop: Maximum segment length in frames (default: 50 frames ~ 1s at 50fps)
           
    Returns:
        List of boundary frame indices (length K), including start (0) and end (N).
        These define K-1 segments: [bound[0]:bound[1]], [bound[1]:bound[2]], ...
        
    Example:
        >>> features = np.random.randn(100, 768)  # 100 frames, 768-dim features
        >>> ssm = features @ features.T
        >>> ssm = ssm - np.min(ssm) + 1e-7  # make non-negative
        >>> boundaries = min_cut_optimized(ssm, K=11)  # Get 11 boundaries (10 segments)
        >>> boundaries
        [0, 8, 19, 31, ..., 100]
        
    Reference:
        Baade et al. (2024). SyllableLM: Learning Coarse Semantic Units for 
        Speech Language Models. arXiv:2410.04029
        https://github.com/AlanBaade/SyllableLM
    """
    N = ssm.shape[0]
    
    # Preprocess: compute cumulative sum for O(1) range queries
    # This is the key optimization that provides ~20-50x speedup
    dp = np.cumsum(np.cumsum(ssm, axis=0), axis=1)
    dp = np.pad(dp, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    
    # Dynamic programming with segment length constraints
    # C[i, k] = minimum cost to partition dp[0:i] into k segments
    # B[i, k] = best split point for partition ending at i with k segments
    C = np.ones((N + 1, K), dtype=np.float32, order="C") * 100000
    B = np.ones((N + 1, K), dtype=np.int32)
    C[0, 0] = 0.0
    
    # For each position i (enforce min_hop constraint)
    for i in range(min_hop, N + 1):
        # Precompute costs for all valid segment starts j
        # Only consider segments of length [min_hop, max_hop]
        temp = []
        for j in range(max(0, i - max_hop + 1), i - min_hop + 1):
            # 5-component cost calculation using cumulative sums (O(1) per segment)
            interior = dp[i, i] - dp[j, i] - dp[i, j] + dp[j, j]
            left = dp[i, j] - dp[j, j]
            top = dp[j, i] - dp[j, j]
            bottom = dp[N, i] - dp[i, i] - dp[N, j] + dp[i, j]
            right = dp[i, N] - dp[i, i] - dp[j, N] + dp[j, i]
            
            # Cost = ratio of external connections to total connections
            # We want to minimize cuts between segments
            total = left + top + bottom + right + interior
            cost = (left + top + bottom + right) / (total + 1e-5)
            temp.append((j, cost))
        
        # Try extending to k segments
        for k in range(1, K):
            # Find best previous split point
            obj = [C[j, k - 1] + item for (j, item) in temp]
            ind = np.argmin(obj)
            B[i, k] = temp[ind][0]
            C[i, k] = obj[ind]
    
    # Backtrack to find boundaries
    boundary = []
    prev_b = N
    boundary.append(prev_b)
    
    for k in range(K - 1, 0, -1):
        prev_b = B[prev_b, k]
        boundary.append(prev_b)
    
    boundary = boundary[::-1]  # Reverse to get chronological order
    
    return boundary


def efficient_extraction_dp_helper(
    embeddings: np.ndarray,
    threshold: float = MINCUT_THRESHOLD,
    s: int = 35,
    min_hop: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical MinCut DP helper matching latest SyllableLM semantics.

    Args:
        embeddings: Feature sequence with shape [N, D].
        threshold: Maximum unit-rate constraint (used to set max units m).
        s: Maximum chunk size in frames.
        min_hop: Minimum chunk size in frames.

    Returns:
        dists: DP chunk-cost tensor [s+1, N+s].
        back: Backpointer tensor [N+1, m+1].
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N, D], got shape {embeddings.shape}")

    x = torch.as_tensor(embeddings, dtype=torch.float32).unsqueeze(0)  # [1, N, D]
    b, n, _ = x.shape

    if n == 0:
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 1), dtype=np.int64)

    s = int(min(n, max(1, s)))
    min_hop = int(min(s, max(1, min_hop)))

    dists = x.new_full((b, s + 1, n + s), 16384.0)

    rolled = torch.stack([torch.roll(x, shifts=-i, dims=-2) for i in range(s)]).transpose(0, 1)
    if s > 1:
        rolled_prepend = x[:, :s].unsqueeze(2).repeat(1, 1, s - 1, 1)
        arranged = torch.cat([rolled_prepend, rolled], dim=2)
    else:
        arranged = rolled

    len_indices = torch.arange(s, device=x.device, dtype=x.dtype) + 1
    dots = arranged.pow(2).mean(dim=-1).cumsum(dim=-2)
    middle = -1.0 / len_indices.view(1, -1, 1) * arranged.cumsum(dim=-3).pow(2).mean(dim=-1)
    outs = dots + middle
    outs = torch.cat([outs[:, i:i + 1].roll(shifts=-(s - i - 1), dims=2) for i in range(s)], dim=1)

    if s > 1:
        dists[:, 1:, s:] = outs[:, :, :-(s - 1)]
    else:
        dists[:, 1:, s:] = outs

    dists += dists.new_full(dists.shape, 16384.0).tril(s - 2)
    dists = dists.clamp(max=16384.0)

    m = max(1, int(threshold * n))
    total_dists = x.new_full((b, n + 2), 16384.0)
    total_dists[:, 0] = 0.0
    back = x.new_zeros((b, n + 1, m + 1), dtype=torch.long)

    magic_mask = torch.tensor(
        [[(j + 1 - k if j + 1 >= k else n + 1) for j in range(n)] for k in range(min_hop, s + 1)],
        device=x.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(b, s + 1 - min_hop, n)

    for j in range(1, m + 1):
        prev = total_dists.unsqueeze(1).expand(b, s + 1 - min_hop, n + 2).gather(2, magic_mask)
        cur_min = torch.min(prev + dists[:, min_hop:, s:n + s], dim=1)
        total_dists[:, 1:-1] = cur_min.values
        back[:, 1:1 + n, j] = cur_min.indices + min_hop

    return dists[0].cpu().numpy(), back[0].cpu().numpy()


def get_quantile_borders_helper(
    dists: np.ndarray,
    back: np.ndarray,
    n: int,
    s: int,
    num_units: int,
    delta: float,
    quantile: float,
) -> List[int]:
    """Canonical quantile-border search over DP tables (latest SyllableLM semantics)."""
    num_units = max(1, int(num_units))
    min_, max_ = max(1, num_units // 3), max(1, num_units)
    best_m = min_

    while min_ <= max_:
        mid_ = (min_ + max_) // 2
        q = n
        j = mid_
        costs = []

        while q > 0 and j > 0:
            step = int(back[q, j])
            if step <= 0 or step > q:
                break
            costs.append(float(dists[step, q - 1 + s]) / float(step))
            q = q - step
            j = j - 1

        quantile_cost = float(np.quantile(costs, quantile)) if costs else np.inf

        if quantile_cost > delta:
            min_ = mid_ + 1
            best_m = mid_
        else:
            max_ = mid_ - 1

    q = n
    j = best_m
    borders = [q]
    while q > 0 and j > 0:
        step = int(back[q, j])
        if step <= 0 or step > q:
            break
        q = q - step
        borders.append(q)
        j = j - 1

    if borders[-1] != 0:
        borders.append(0)

    borders = sorted(set(int(b) for b in borders))
    if borders[0] != 0:
        borders = [0] + borders
    if borders[-1] != n:
        borders = borders + [n]
    return borders


def efficient_extraction(
    embeddings: np.ndarray,
    threshold: float = MINCUT_THRESHOLD,
    s: int = 35,
    min_hop: int = 3,
    deltas: Optional[List[float]] = None,
    quantiles: Optional[List[float]] = None,
) -> List[List[List[int]]]:
    """Canonical dynamic MinCut extraction.

    Returns a list with one entry per (delta, quantile) pair, each containing a
    batch list of boundary lists. For 2D input, batch size is 1.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"embeddings must be [N, D] or [B, N, D], got shape {embeddings.shape}")

    deltas = list(deltas or [0.0033])
    quantiles = list(quantiles or [0.75])
    if len(deltas) != len(quantiles):
        raise ValueError("deltas and quantiles must have identical lengths")

    b, n, _ = x.shape
    s_eff = min(n, max(1, int(s))) if n > 0 else 1
    m = max(1, int(threshold * max(1, n)))

    dists_batch = []
    back_batch = []
    for idx in range(b):
        dists_i, back_i = efficient_extraction_dp_helper(
            x[idx],
            threshold=threshold,
            s=s,
            min_hop=min_hop,
        )
        dists_batch.append(dists_i)
        back_batch.append(back_i)

    batch_outs: List[List[List[int]]] = []
    for delta, quantile in zip(deltas, quantiles):
        one_setting = []
        for dists_i, back_i in zip(dists_batch, back_batch):
            one_setting.append(
                get_quantile_borders_helper(
                    dists_i,
                    back_i,
                    n=n,
                    s=s_eff,
                    num_units=m,
                    delta=float(delta),
                    quantile=float(quantile),
                )
            )
        batch_outs.append(one_setting)

    return batch_outs


def extract_mincut_boundaries(
    features: np.ndarray,
    threshold: float = MINCUT_THRESHOLD,
    s: int = 35,
    min_hop: int = 3,
    delta: float = 0.0033,
    quantile: float = 0.75,
) -> List[int]:
    """Extract canonical MinCut boundaries from frame features."""
    boundaries = efficient_extraction(
        features,
        threshold=threshold,
        s=s,
        min_hop=min_hop,
        deltas=[delta],
        quantiles=[quantile],
    )[0][0]
    if len(boundaries) < 2:
        n = int(features.shape[0])
        return [0, n]
    return boundaries


def compute_mincut_presegmentation_trace(
    features: np.ndarray,
    threshold: float = MINCUT_THRESHOLD,
    s: int = 35,
    min_hop: int = 3,
    aggregation: str = 'min_cost',
) -> np.ndarray:
    """Compute a 1-D pre-segmentation trace from canonical MinCut DP costs.

    The trace is derived from the same chunk cost table used by MinCut boundary
    search, before delta/quantile border decoding.
    """
    dists, _ = efficient_extraction_dp_helper(
        features,
        threshold=threshold,
        s=s,
        min_hop=min_hop,
    )
    n = int(features.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    s_eff = dists.shape[0] - 1
    min_hop_eff = int(min(max(1, min_hop), s_eff))
    hop_lengths = np.arange(min_hop_eff, s_eff + 1, dtype=np.float32)
    trace = np.zeros(n, dtype=np.float32)

    for q in range(1, n + 1):
        col = q - 1 + s_eff
        vals = dists[min_hop_eff:, col] / hop_lengths
        if aggregation == 'min_cost':
            trace[q - 1] = float(np.min(vals))
        elif aggregation == 'mean_cost':
            trace[q - 1] = float(np.mean(vals))
        else:
            raise ValueError("aggregation must be one of {'min_cost', 'mean_cost'}")

    return trace.astype(np.float32)


def _apply_cosine_merge(
    boundary_pairs: List[Tuple[int, int]],
    features: np.ndarray,
    merge_threshold: float,
) -> List[Tuple[int, int]]:
    """
    Greedy cosine-similarity segment merging post-processing (Peng et al. 2023).

    Replicates the minCutMerge-X algorithm from syllable-discovery/save_seg_feats_mincut.py.
    Repeatedly merges the most similar adjacent segment pair (by cosine similarity
    of mean features) until the maximum pairwise similarity drops below
    merge_threshold or only 2 segments remain.

    This is the SSM-path post-processing step; SyllableLM DP does not use it.
    Requires at least 3 segments to operate (merging 2 into 1 would discard all
    boundaries and leave a single segment spanning the whole utterance).

    Args:
        boundary_pairs: List of (start_frame, end_frame) pairs (ALL pairs,
                        including single-frame ones).
        features: Raw (unnormalised) feature matrix [n_frames, d].
        merge_threshold: Cosine similarity threshold; adjacent pairs with
                         similarity >= threshold are merged.

    Returns:
        Merged list of (start_frame, end_frame) pairs.
    """
    if len(boundary_pairs) < 3:
        return boundary_pairs

    pairs: List[Tuple[int, int]] = list(boundary_pairs)

    def _mean(l: int, r: int) -> np.ndarray:
        seg = features[l:r]
        return seg.mean(axis=0) if len(seg) > 0 else np.zeros(features.shape[1], dtype=features.dtype)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    seg_means = [_mean(l, r) for l, r in pairs]
    sims = [_cosine(seg_means[i], seg_means[i + 1]) for i in range(len(seg_means) - 1)]

    while len(pairs) >= 3 and sims:
        best = int(np.argmax(sims))
        if sims[best] < merge_threshold:
            break
        new_pair = (pairs[best][0], pairs[best + 1][1])
        pairs = pairs[:best] + [new_pair] + pairs[best + 2:]
        seg_means = [_mean(l, r) for l, r in pairs]
        sims = [_cosine(seg_means[i], seg_means[i + 1]) for i in range(len(seg_means) - 1)]

    return pairs


# =============================================================================
# Phase 5 Wrapper Class (Object-Oriented API)
# =============================================================================

class MinCutSegmenter(End2EndSegmenter):
    """
    Apply MinCut algorithm to any feature extractor.
    
    This wrapper allows you to use the MinCut graph-based segmentation
    algorithm with any features (MFCC, HuBERT, custom, etc.).
    
    Args:
        feature_extractor: Feature extractor (FeatureExtractor instance or callable)
        sec_per_syllable: Target duration per syllable (default: 0.22s)
        use_optimized: Use optimized MinCut for 6-50× speedup (default: True)
        min_hop: Minimum segment length in frames (for optimized only)
        max_hop: Maximum segment length in frames (for optimized only)
    
    Example:
        >>> from findsylls.segmentation.extractors import HuBERTExtractor
        >>> extractor = HuBERTExtractor()
        >>> segmenter = MinCutSegmenter(extractor, sec_per_syllable=0.2)
        >>> segments = segmenter.segment(audio, sr=16000)
    """
    
    def __init__(
        self,
        feature_extractor: Union[FeatureExtractor, Callable],
        threshold: float = MINCUT_THRESHOLD,
        s: int = 35,
        min_hop: int = 3,
        delta: Optional[float] = None,
        quantile: Optional[float] = None,
        model_key: Optional[str] = None,
        use_reference: bool = False,
        sec_per_syllable: float = _DEFAULT_SEC_PER_SYLLABLE,
        use_optimized: bool = True,
        max_hop: int = 50,
        merge_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.s = s
        self.sec_per_syllable = sec_per_syllable
        self.use_reference = use_reference
        self.use_optimized = use_optimized
        self.min_hop = min_hop
        self.max_hop = max_hop
        self.merge_threshold = merge_threshold

        preset = MINCUT_MODEL_PRESETS.get(str(model_key).lower()) if model_key is not None else None
        self.delta = float(delta if delta is not None else (preset['delta'] if preset else 0.0033))
        self.quantile = float(quantile if quantile is not None else (preset['quantile'] if preset else 0.75))

        # Warn when use_reference=True unless the extractor is explicitly marked as
        # calibrated for the SyllableLM DP parameters (is_dp_mincut_calibrated=True).
        if use_reference and not getattr(feature_extractor, 'is_dp_mincut_calibrated', False):
            warnings.warn(
                f"MinCutSegmenter(use_reference=True) uses the SyllableLM DP with "
                f"delta={self.delta}, quantile={self.quantile}. These parameters were "
                f"calibrated for Data2Vec2 features and will over-segment with other "
                f"feature types (e.g. HuBERT, VGHuBERT). Use use_reference=False for "
                f"the fixed-K SSM path, which works with any feature extractor. "
                f"Silence this warning by setting is_dp_mincut_calibrated=True on "
                f"your feature extractor after re-tuning delta/quantile.",
                UserWarning,
                stacklevel=2,
            )

        # Determine frame rate
        if isinstance(feature_extractor, FeatureExtractor):
            self.frame_rate = feature_extractor.frame_rate
        elif hasattr(feature_extractor, 'frame_rate'):
            self.frame_rate = feature_extractor.frame_rate
        else:
            self.frame_rate = 50.0  # Default assumption
    
    def segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        """Segment audio using MinCut on extracted features."""
        features = extract_frame_features(self.feature_extractor, audio, sr)
        n = int(features.shape[0])
        if n == 0:
            return []

        # Cosine SSM for peak selection (within-segment nucleus detection).
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        ssm_cosine = features_norm @ features_norm.T
        ssm_cosine = ssm_cosine - np.min(ssm_cosine) + 1e-7

        if self.use_reference:
            boundaries = extract_mincut_boundaries(
                features,
                threshold=self.threshold,
                s=self.s,
                min_hop=self.min_hop,
                delta=self.delta,
                quantile=self.quantile,
            )
            all_pairs = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
            filtered = [(s, e) for s, e in all_pairs if e - s > 1]
            boundary_pairs = filtered if filtered else all_pairs
        else:
            duration = n / self.frame_rate
            # ceil matches save_seg_feats_mincut.py: int(np.ceil(audio_len_sec / secPerSyllable))
            k = max(2, int(np.ceil(duration / self.sec_per_syllable)) + 1)
            # Raw dot-product SSM for MinCut — matches reference (feat @ feat.T, no L2 norm).
            ssm_raw = features @ features.T
            ssm_raw = ssm_raw - np.min(ssm_raw) + 1e-7
            if self.use_optimized:
                boundaries = min_cut_optimized(ssm_raw, k, self.min_hop, self.max_hop)
            else:
                boundaries = min_cut(ssm_raw, k)

            all_pairs = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
            if self.merge_threshold is not None:
                # Merge uses ALL pairs including single-frame ones (ref: save_seg_feats_mincut.py).
                boundary_pairs = _apply_cosine_merge(all_pairs, features, self.merge_threshold)
            else:
                filtered = [(s, e) for s, e in all_pairs if e - s > 1]
                boundary_pairs = filtered if filtered else all_pairs

        segments = []
        for start_frame, end_frame in boundary_pairs:
            if end_frame - start_frame > 1:
                segment_ssm = ssm_cosine[start_frame:end_frame, start_frame:end_frame]
                avg_similarities = segment_ssm.mean(axis=1)
                nucleus_frame = start_frame + int(np.argmax(avg_similarities))
            else:
                nucleus_frame = start_frame

            start = start_frame / self.frame_rate
            end = end_frame / self.frame_rate
            nucleus = nucleus_frame / self.frame_rate
            segments.append((start, nucleus, end))

        return segments
    
    def get_embeddings(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
        """Get segments and their feature embeddings."""
        features = extract_frame_features(self.feature_extractor, audio, sr)
        n = int(features.shape[0])
        if n == 0:
            return [], np.zeros((0, 0), dtype=np.float32)

        # Cosine SSM for peak selection (identical to segment() for API parity).
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        ssm_cosine = features_norm @ features_norm.T
        ssm_cosine = ssm_cosine - np.min(ssm_cosine) + 1e-7

        if self.use_reference:
            boundaries = extract_mincut_boundaries(
                features,
                threshold=self.threshold,
                s=self.s,
                min_hop=self.min_hop,
                delta=self.delta,
                quantile=self.quantile,
            )
            all_pairs = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
            filtered = [(s, e) for s, e in all_pairs if e - s > 1]
            boundary_pairs = filtered if filtered else all_pairs
        else:
            duration = n / self.frame_rate
            # ceil matches save_seg_feats_mincut.py: int(np.ceil(audio_len_sec / secPerSyllable))
            k = max(2, int(np.ceil(duration / self.sec_per_syllable)) + 1)
            # Raw dot-product SSM for MinCut — matches reference (feat @ feat.T, no L2 norm).
            ssm_raw = features @ features.T
            ssm_raw = ssm_raw - np.min(ssm_raw) + 1e-7
            if self.use_optimized:
                boundaries = min_cut_optimized(ssm_raw, k, self.min_hop, self.max_hop)
            else:
                boundaries = min_cut(ssm_raw, k)

            all_pairs = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
            if self.merge_threshold is not None:
                boundary_pairs = _apply_cosine_merge(all_pairs, features, self.merge_threshold)
            else:
                filtered = [(s, e) for s, e in all_pairs if e - s > 1]
                boundary_pairs = filtered if filtered else all_pairs

        embeddings = []
        time_segments = []
        for start_frame, end_frame in boundary_pairs:
            segment_features = features[start_frame:end_frame]
            if segment_features.shape[0] == 0:
                continue
            embeddings.append(np.mean(segment_features, axis=0))

            if end_frame - start_frame > 1:
                segment_ssm = ssm_cosine[start_frame:end_frame, start_frame:end_frame]
                avg_similarities = segment_ssm.mean(axis=1)
                nucleus_frame = start_frame + int(np.argmax(avg_similarities))
            else:
                nucleus_frame = start_frame

            start_sec = start_frame / self.frame_rate
            end_sec = end_frame / self.frame_rate
            nucleus_sec = nucleus_frame / self.frame_rate
            time_segments.append((start_sec, nucleus_sec, end_sec))

        return time_segments, np.array(embeddings)


__all__ = [
    'MINCUT_THRESHOLD',
    'MINCUT_MODEL_PRESETS',
    '_DEFAULT_SEC_PER_SYLLABLE',
    'min_cut',
    'min_cut_optimized',
    'efficient_extraction_dp_helper',
    'get_quantile_borders_helper',
    'efficient_extraction',
    'extract_mincut_boundaries',
    'compute_mincut_presegmentation_trace',
    '_apply_cosine_merge',
    'MinCutSegmenter',
]
