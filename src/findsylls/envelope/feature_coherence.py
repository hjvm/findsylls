"""
Feature-Based Envelope Computers

Computes amplitude envelopes based on feature similarities.
This bridges feature-based and envelope-based segmentation approaches.

Three envelope types are provided:
1. SSMEnvelopeComputer: Global coherence from full self-similarity matrix
2. GreedyCosineEnvelope: Frame-to-prototype similarity (matches GreedyCosine algorithm)
3. CLSAttentionEnvelope: Attention scores from transformer CLS tokens
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


class GreedyCosineEnvelope(EnvelopeComputer):
    """
    Compute envelope from frame-to-local-prototype cosine similarity.
    
    This mimics the GreedyCosine segmentation algorithm's similarity computation:
    - For each frame, compute local prototype (mean of surrounding frames)
    - Compute cosine similarity of frame to its local prototype
    - High values = frame similar to local context (stable region)
    - Low values = frame dissimilar from context (transition)
    
    This is conceptually aligned with GreedyCosine's phase 1, where frames
    are compared to running segment averages.
    
    Args:
        feature_extractor: FeatureExtractor to use for computing features
        window_size: Number of frames in each direction for local prototype (default: 5)
        normalize: Whether to normalize the envelope to [0, 1] (default: True)
    
    Example:
        >>> from findsylls.features import MFCCExtractor
        >>> from findsylls.envelope import GreedyCosineEnvelope
        >>> from findsylls.segmentation import PeakdetectSegmenter
        >>> 
        >>> # Create GreedyCosine-style envelope
        >>> extractor = MFCCExtractor(n_mfcc=13)
        >>> envelope_computer = GreedyCosineEnvelope(extractor, window_size=5)
        >>> 
        >>> # Use with peak detection (hybrid approach)
        >>> segmenter = PeakdetectSegmenter(envelope_computer, delta=0.05)
        >>> segments = segmenter.segment(audio, sr=16000)
    
    Notes:
        - Measures frame-to-prototype similarity (like GreedyCosine)
        - Different from SSM (frame-to-all-frames)
        - Different from Pairwise (frame-to-neighbor-frames)
        - Local prototype = mean of frames in window
        - Time complexity: O(N × window_size × D)
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        window_size: int = 5,
        normalize: bool = True
    ):
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.normalize = normalize
    
    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GreedyCosine-style envelope.
        
        Args:
            audio: Audio signal (mono, float32)
            sr: Sample rate
        
        Returns:
            envelope: (N,) array of frame-to-prototype similarity scores
            times: (N,) array of time points in seconds
        """
        # Extract features
        features = self.feature_extractor.extract(audio, sr)
        N = features.shape[0]
        
        # Compute envelope: similarity to local prototype
        envelope = np.zeros(N)
        epsilon = 1e-8
        
        for i in range(N):
            # Define local window (excluding current frame for prototype)
            start = max(0, i - self.window_size)
            end = min(N, i + self.window_size + 1)
            
            # Compute local prototype (mean of window)
            # Exclude current frame to avoid trivial similarity
            local_features = features[start:end]
            mask = np.ones(len(local_features), dtype=bool)
            mask[i - start] = False
            
            if mask.sum() > 0:
                local_prototype = local_features[mask].mean(axis=0)
                
                # Compute cosine similarity (same as GreedyCosine)
                frame = features[i]
                dot_product = (frame * local_prototype).sum()
                frame_norm = ((frame**2).sum() + epsilon) ** 0.5
                proto_norm = ((local_prototype**2).sum() + epsilon) ** 0.5
                
                envelope[i] = dot_product / (frame_norm * proto_norm)
            else:
                envelope[i] = 1.0  # Edge case
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            env_min = envelope.min()
            env_max = envelope.max()
            if env_max > env_min:
                envelope = (envelope - env_min) / (env_max - env_min)
        
        # Create time array
        duration = len(audio) / sr
        times = np.linspace(0, duration, N)
        
        return envelope, times
    
    def __repr__(self):
        return (f"GreedyCosineEnvelope("
                f"feature_extractor={self.feature_extractor.__class__.__name__}, "
                f"window_size={self.window_size}, "
                f"normalize={self.normalize})")


class CLSAttentionEnvelope(EnvelopeComputer):
    """
    Compute envelope from transformer CLS token attention scores.
    
    This extracts attention weights from a transformer model's CLS token,
    following VG-HuBERT word-discovery approach:
    - CLS token attends to all sequence positions
    - Peaks in CLS attention = salient positions (word onsets)
    - Can be used as envelope for segmentation
    
    This is specific to transformer-based feature extractors that:
    1. Have a CLS token (or similar special token)
    2. Provide attention weights via output_attentions=True
    
    Args:
        feature_extractor: FeatureExtractor (must be transformer-based)
        layer: Which transformer layer to extract attention from (default: -1 = last)
        aggregate: How to aggregate across attention heads:
                  'max' (default), 'mean', or specific head index
        normalize: Whether to normalize the envelope to [0, 1] (default: True)
    
    Example:
        >>> from findsylls.features import HuBERTExtractor
        >>> from findsylls.envelope import CLSAttentionEnvelope
        >>> from findsylls.segmentation import PeakdetectSegmenter
        >>> 
        >>> # Create CLS attention envelope
        >>> extractor = HuBERTExtractor(layer=9)
        >>> envelope_computer = CLSAttentionEnvelope(extractor, layer=9)
        >>> 
        >>> # Use with peak detection
        >>> segmenter = PeakdetectSegmenter(envelope_computer, delta=0.05)
        >>> segments = segmenter.segment(audio, sr=16000)
    
    Notes:
        - Only works with transformer-based extractors (HuBERT, VG-HuBERT, etc.)
        - Requires PyTorch and Transformers
        - Different from feature similarity - uses learned attention
        - VG-HuBERT paper uses this for word segmentation
        - Time complexity: Same as feature extraction + O(N) for attention
    
    Reference:
        Peng et al. (2023). Syllable Discovery and Cross-Lingual Generalization
        in a Visually Grounded, Self-Supervised Speech Model. Interspeech 2023.
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        layer: int = -1,
        aggregate: str = 'max',
        normalize: bool = True
    ):
        self.feature_extractor = feature_extractor
        self.layer = layer
        self.aggregate = aggregate
        self.normalize = normalize
        
        # Validate aggregate mode
        if aggregate not in ['max', 'mean'] and not isinstance(aggregate, int):
            raise ValueError(f"aggregate must be 'max', 'mean', or head index, got {aggregate}")
    
    def compute(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CLS attention envelope.
        
        Args:
            audio: Audio signal (mono, float32)
            sr: Sample rate
        
        Returns:
            envelope: (N,) array of CLS attention scores
            times: (N,) array of time points in seconds
        
        Raises:
            RuntimeError: If feature extractor doesn't support attention extraction
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("CLSAttentionEnvelope requires PyTorch")
        
        extractor_class = self.feature_extractor.__class__.__name__
        
        # VG-HuBERT has native return_attention support
        if extractor_class == 'VGHuBERTFeatureExtractor':
            features, cls_attention = self.feature_extractor.extract(audio, sr, return_attention=True)
            envelope = cls_attention
        
        # Sylber: has limited layers (0 to encoding_layer-1)
        elif extractor_class == 'SylberFeatureExtractor':
            if self.layer >= self.feature_extractor.encoding_layer:
                raise RuntimeError(
                    f"Sylber model with encoding_layer={self.feature_extractor.encoding_layer} "
                    f"only has layers 0-{self.feature_extractor.encoding_layer-1}, "
                    f"cannot access layer {self.layer}"
                )
            
            self.feature_extractor._lazy_load()
            
            import librosa
            audio_proc = audio
            if sr != 16000:
                audio_proc = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if audio_proc.dtype != np.float32:
                audio_proc = audio_proc.astype(np.float32)
            audio_proc = (audio_proc - audio_proc.mean()) / (audio_proc.std() + 1e-8)
            
            audio_tensor = torch.from_numpy(audio_proc).unsqueeze(0)
            if hasattr(self.feature_extractor, '_device'):
                audio_tensor = audio_tensor.to(self.feature_extractor._device)
            
            with torch.no_grad():
                outputs = self.feature_extractor._model(
                    audio_tensor,
                    output_attentions=True,
                    return_dict=True
                )
            
            attn = outputs.attentions[self.layer]
            cls_attn = attn[0, :, 0, :]
            if cls_attn.shape[1] > 1:
                cls_attn = cls_attn[:, 1:]
            
            if self.aggregate == 'max':
                envelope = cls_attn.max(dim=0)[0].cpu().numpy()
            elif self.aggregate == 'mean':
                envelope = cls_attn.mean(dim=0).cpu().numpy()
            else:
                envelope = cls_attn[self.aggregate].cpu().numpy()
        
        # Standard HuBERT or other transformers
        else:
            if not hasattr(self.feature_extractor, '_model'):
                raise RuntimeError(
                    f"{extractor_class} doesn't support attention extraction"
                )
            
            if hasattr(self.feature_extractor, '_lazy_load'):
                self.feature_extractor._lazy_load()
            
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            if hasattr(self.feature_extractor, '_device'):
                audio_tensor = audio_tensor.to(self.feature_extractor._device)
            
            with torch.no_grad():
                outputs = self.feature_extractor._model(
                    audio_tensor,
                    output_attentions=True,
                    return_dict=True
                )
            
            attn = outputs.attentions[self.layer]
            cls_attn = attn[0, :, 0, :]
            if cls_attn.shape[1] > 1:
                cls_attn = cls_attn[:, 1:]
            
            if self.aggregate == 'max':
                envelope = cls_attn.max(dim=0)[0].cpu().numpy()
            elif self.aggregate == 'mean':
                envelope = cls_attn.mean(dim=0).cpu().numpy()
            else:
                envelope = cls_attn[self.aggregate].cpu().numpy()
        
        # Align with feature dimensions
        features = self.feature_extractor.extract(audio, sr)
        N = features.shape[0]
        if len(envelope) > N:
            envelope = envelope[:N]
        elif len(envelope) < N:
            from scipy.interpolate import interp1d
            old_times = np.linspace(0, 1, len(envelope))
            new_times = np.linspace(0, 1, N)
            f = interp1d(old_times, envelope, kind='linear', fill_value='extrapolate')
            envelope = f(new_times)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            env_min = envelope.min()
            env_max = envelope.max()
            if env_max > env_min:
                envelope = (envelope - env_min) / (env_max - env_min)
        
        # Create time array
        duration = len(audio) / sr
        times = np.linspace(0, duration, N)
        
        return envelope, times
    
    def __repr__(self):
        return (f"CLSAttentionEnvelope("
                f"feature_extractor={self.feature_extractor.__class__.__name__}, "
                f"layer={self.layer}, "
                f"aggregate={self.aggregate}, "
                f"normalize={self.normalize})")


__all__ = [
    'SSMEnvelopeComputer',
    'GreedyCosineEnvelope',
    'CLSAttentionEnvelope'
]
