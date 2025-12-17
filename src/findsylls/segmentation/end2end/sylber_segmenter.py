"""
Sylber segmentation wrapper.

Wraps the sylber PyPI package for syllable segmentation using self-supervised
syllabic distillation (Cho et al. ICLR 2025).

Paper: https://arxiv.org/abs/2410.07168
GitHub: https://github.com/Berkeley-Speech-Group/sylber
PyPI: https://pypi.org/project/sylber/

Key Features:
- Achieves 4.27 tokens/second (user's target!)
- Self-supervised learning (no phonetic annotations needed)
- Pre-trained on HuggingFace Hub (cheoljun95/sylber)
- Provides both segmentation AND embeddings (768-dim per syllable)
- Dynamic segmentation via norm-based thresholding + similarity merging

Installation:
    pip install sylber
"""

from typing import List, Tuple, Optional
import numpy as np
import librosa

from ..base import End2EndSegmenter


class SylberSegmenter(End2EndSegmenter):
    """
    Sylber end-to-end syllable segmenter.
    
    Uses self-supervised syllabic distillation to learn syllable-level
    representations. Applies dynamic segmentation based on feature norms
    and similarity-based merging.
    
    Args:
        sample_rate: Target sample rate (default: 16000)
        model_ckpt: HuggingFace model checkpoint (default: "cheoljun95/sylber")
        encoding_layer: Which layer to use for segmentation (default: 9)
        merge_threshold: Similarity threshold for merging segments (default: 0.8)
        norm_threshold: Norm threshold for boundary detection (default: 2.6)
        device: Device for inference ('cpu', 'cuda', 'mps')
        cache: Whether to cache model after first load (default: True)
    
    Example:
        >>> segmenter = SylberSegmenter(device='cuda')
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> # Get embeddings too
        >>> segments, embeddings = segmenter.get_embeddings(audio, sr=16000)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        model_ckpt: str = "sylber",  # Just the model name, not full repo path
        encoding_layer: int = 9,
        merge_threshold: float = 0.8,
        norm_threshold: float = 2.6,
        device: str = 'cpu',
        cache: bool = True,
        detect_peaks: bool = False
    ):
        super().__init__(sample_rate=sample_rate, device=device, cache=cache)
        self.model_ckpt = model_ckpt
        self.encoding_layer = encoding_layer
        self.merge_threshold = merge_threshold
        self.norm_threshold = norm_threshold
        self.detect_peaks = detect_peaks
        self._segmenter = None  # Lazy loading
    
    def _lazy_load_model(self):
        """Lazy load Sylber model on first use."""
        if self._segmenter is None or not self.cache:
            try:
                from sylber import Segmenter
            except ImportError as e:
                raise ImportError(
                    "sylber package not installed. Install with: pip install sylber"
                ) from e
            
            # Determine device
            device = self._get_device()
            
            # Load model
            # Force CPU if CUDA not available
            if device == 'cuda':
                import torch
                if not torch.cuda.is_available():
                    device = 'cpu'
            
            self._segmenter = Segmenter(
                model_ckpt=self.model_ckpt,
                encoding_layer=self.encoding_layer,
                merge_threshold=self.merge_threshold,
                norm_threshold=self.norm_threshold,
                device=device
            )
        
        return self._segmenter
    
    def _detect_peaks_cosine_similarity(
        self,
        features: np.ndarray,
        segments: List[Tuple[float, float]],
        fps: float
    ) -> List[Tuple[float, float, float]]:
        """
        Detect acoustic peaks within segments using cosine similarity.
        
        For each segment, finds the frame with maximum similarity to its neighbors.
        This is native to Sylber's learned representation space.
        
        Args:
            features: (num_frames, feature_dim) frame-level features
            segments: List of (start, end) tuples in seconds
            fps: Frames per second
            
        Returns:
            segments_with_peaks: List of (start, peak, end) tuples in seconds
        """
        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)
        
        segments_with_peaks = []
        frame_hop = 1.0 / fps
        
        for start, end in segments:
            start_frame = int(round(start / frame_hop))
            end_frame = int(round(end / frame_hop))
            
            start_frame = max(0, min(start_frame, features.shape[0] - 1))
            end_frame = max(0, min(end_frame, features.shape[0]))
            
            if end_frame - start_frame <= 1:
                peak_frame = start_frame
            else:
                similarities = []
                for i in range(start_frame, end_frame):
                    sim_sum = 0.0
                    count = 0
                    
                    if i > start_frame:
                        sim_sum += cosine_sim(features[i], features[i-1])
                        count += 1
                    
                    if i < end_frame - 1:
                        sim_sum += cosine_sim(features[i], features[i+1])
                        count += 1
                    
                    avg_sim = sim_sum / count if count > 0 else 0.0
                    similarities.append(avg_sim)
                
                peak_idx = np.argmax(similarities)
                peak_frame = start_frame + peak_idx
            
            peak = peak_frame * frame_hop
            segments_with_peaks.append((start, peak, end))
        
        return segments_with_peaks
    
    def segment(
        self, 
        audio: np.ndarray, 
        sr: int = 16000,
        return_features: bool = False,
        **kwargs
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio using Sylber.
        
        Args:
            audio: Raw audio waveform (mono, float32)
            sr: Sample rate of audio
            return_features: If True, also return segment embeddings
            **kwargs: Additional parameters (overrides instance defaults)
        
        Returns:
            List of (start, nucleus, end) tuples in seconds
            If return_features=True, returns (segments, features) tuple
        """
        # Get segmenter (lazy load)
        segmenter = self._lazy_load_model()
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # WORKAROUND: sylber has a bug with "assert wav != None" that fails with numpy arrays
        # Use temporary file instead
        import tempfile
        import soundfile as sf
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name
        
        try:
            # Write audio to temp file
            sf.write(temp_file, audio, self.sample_rate)
            
            # Run segmentation using file
            outputs = segmenter(wav_file=temp_file, in_second=True)
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Convert to (start, nucleus, end) format
        if self.detect_peaks:
            # Detect peaks using cosine similarity in Sylber's feature space
            features = outputs['hidden_states']
            num_frames = features.shape[0]
            duration = len(audio) / self.sample_rate
            fps = num_frames / duration if duration > 0 else 50.0
            
            raw_segments = [(s, e) for s, e in outputs['segments']]
            segments = self._detect_peaks_cosine_similarity(features, raw_segments, fps)
        else:
            # Use midpoint as nucleus approximation
            segments = []
            for start, end in outputs['segments']:
                nucleus = (start + end) / 2
                segments.append((start, nucleus, end))
        
        # Validate output
        self._validate_output(segments)
        
        if return_features:
            return segments, outputs['segment_features']
        
        return segments
    
    def get_embeddings(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
        """
        Get both segments and embeddings.
        
        Args:
            audio: Raw audio waveform
            sr: Sample rate
        
        Returns:
            (segments, embeddings) tuple
            - segments: List of (start, nucleus, end) tuples
            - embeddings: (N, 768) array of syllable embeddings
        """
        return self.segment(audio, sr, return_features=True)
    
    def __repr__(self) -> str:
        return (
            f"SylberSegmenter("
            f"model='{self.model_ckpt}', "
            f"layer={self.encoding_layer}, "
            f"merge_threshold={self.merge_threshold}, "
            f"norm_threshold={self.norm_threshold}, "
            f"device='{self.device}')"
        )
