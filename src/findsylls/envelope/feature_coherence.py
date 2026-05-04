"""
Feature-Based Envelope Computers

Computes amplitude envelopes based on feature similarities.
This bridges feature-based and envelope-based segmentation approaches.

Two envelope types are provided:
1. SSMEnvelopeComputer: Global coherence from full self-similarity matrix
2. (Removed) GreedyCosineEnvelope legacy local-prototype path

Note: GreedyCosineEnvelope and CLSAttentionEnvelope now live in dedicated modules
(`envelope/greedy_cosine.py`, `envelope/cls_attention.py`) to enforce canonical
algorithm-derived aggregation policies.
"""

import numpy as np
from typing import Tuple, Optional

from .base import EnvelopeComputer
from ..features.base import FeatureExtractor


class SSMEnvelopeComputer(EnvelopeComputer):
    """
    Compute envelope from full self-similarity matrix (SSM).
    
    This uses the SAME SSM computation as MinCut segmentation:
    - Computes full N×N cosine similarity matrix
    - Envelope = row-wise average (global coherence)
    - High values = frame similar to most other frames (stable region)
    - Low values = frame dissimilar to others (transition)
    
    This is the most principled feature-based envelope, matching what
    MinCut algorithm already computes internally.
    
    Args:
        feature_extractor: FeatureExtractor to use for computing features
        normalize: Whether to normalize the envelope to [0, 1] (default: True)
        cache_ssm: Whether to cache SSM for potential reuse (default: False)
    
    Example:
        >>> from findsylls.features import MFCCExtractor
        >>> from findsylls.envelope import SSMEnvelopeComputer
        >>> from findsylls.segmentation import PeakdetectSegmenter
        >>> 
        >>> # Create SSM-based envelope
        >>> extractor = MFCCExtractor(n_mfcc=13)
        >>> envelope_computer = SSMEnvelopeComputer(extractor)
        >>> 
        >>> # Use with peak detection (hybrid: SSM envelope + peak finding)
        >>> segmenter = PeakdetectSegmenter(envelope_computer, delta=0.05)
        >>> segments = segmenter.segment(audio, sr=16000)
    
    Notes:
        - Consistent with MinCut algorithm (same SSM computation)
        - More efficient than windowed coherence (no loop over frames)
        - Time complexity: O(N² × D) for SSM computation
        - Space complexity: O(N²) for storing SSM
        - Global coherence (all frames) vs local coherence (neighbors)
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        normalize: bool = True,
        cache_ssm: bool = False
    ):
        self.feature_extractor = feature_extractor
        self.normalize = normalize
        self.cache_ssm = cache_ssm
        self._cached_ssm = None
        self._cached_audio_hash = None
    
    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SSM-based envelope.
        
        Args:
            audio: Audio signal (mono, float32)
            sr: Sample rate
        
        Returns:
            envelope: (N,) array of global coherence scores
            times: (N,) array of time points in seconds
        """
        # Extract features
        features = self.feature_extractor.extract(audio, sr)
        N = features.shape[0]
        
        # Normalize features for cosine similarity (same as MinCut)
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Compute full self-similarity matrix (same as MinCut)
        ssm = features_norm @ features_norm.T
        ssm = ssm - np.min(ssm) + 1e-7  # Non-negative + stability
        
        # Cache SSM if requested
        if self.cache_ssm:
            audio_hash = hash(audio.tobytes())
            self._cached_ssm = ssm
            self._cached_audio_hash = audio_hash
        
        # Envelope = row-wise average (how similar each frame is to ALL frames)
        envelope = ssm.mean(axis=1)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            env_min = envelope.min()
            env_max = envelope.max()
            if env_max > env_min:
                envelope = (envelope - env_min) / (env_max - env_min)
        
        # Create time array based on feature frames
        duration = len(audio) / sr
        times = np.linspace(0, duration, N)
        
        return envelope, times
    
    def get_cached_ssm(self) -> Optional[np.ndarray]:
        """Get cached SSM if available (for MinCut reuse)."""
        return self._cached_ssm
    
    def __repr__(self):
        return (f"SSMEnvelopeComputer("
                f"feature_extractor={self.feature_extractor.__class__.__name__}, "
                f"normalize={self.normalize}, "
                f"cache_ssm={self.cache_ssm})")


# GreedyCosineEnvelope and CLSAttentionEnvelope moved to dedicated modules.


__all__ = [
    'SSMEnvelopeComputer',
]
