"""
Pure Python implementation of the MinCut algorithm for speech segmentation.

This is a Python port of the Cython implementation from:
https://github.com/jasonppy/syllable-discovery

The algorithm partitions a self-similarity matrix into K segments by
minimizing inter-segment similarity using dynamic programming.

Reference:
    Peng et al. (2023). Syllable Discovery and Cross-Lingual Generalization 
    in a Visually Grounded, Self-Supervised Speech Model. Interspeech 2023.
"""

import numpy as np
from typing import List


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
