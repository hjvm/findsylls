"""
Sylber feature extractor.

Extracts features from Sylber's fine-tuned HuBERT model (Cho et al., ICLR 2025).
Sylber uses self-supervised syllabic distillation to learn syllable-level representations.

Paper: https://arxiv.org/abs/2410.07168
PyPI: https://pypi.org/project/sylber/
Model: cheoljun95/sylber on HuggingFace Hub

Key characteristics:
- Fine-tuned on facebook/hubert-base-ls960
- Layer 9 features (768-dim, 50 Hz)
- Optimized for syllable segmentation with greedy cosine similarity
"""

import numpy as np
from typing import Optional
import warnings

from .base import FeatureExtractor


class SylberFeatureExtractor(FeatureExtractor):
    """
    Sylber feature extractor.
    
    Extracts syllable-optimized features from Sylber's fine-tuned HuBERT model.
    
    Args:
        model_ckpt: HuggingFace checkpoint (default: "cheoljun95/sylber")
        encoding_layer: Which layer to extract (default: 9)
        device: Compute device ('cpu', 'cuda', 'mps', or None for auto)
    
    Example:
        >>> extractor = SylberFeatureExtractor()
        >>> features = extractor.extract(audio, sr=16000)
        >>> print(features.shape)  # (N, 768) at 50 Hz
        
        >>> # Use with segmentation algorithm
        >>> from findsylls.segmentation import GreedyCosineSegmenter
        >>> segmenter = GreedyCosineSegmenter(
        ...     extractor, 
        ...     merge_threshold=0.8,
        ...     norm_threshold=2.6
        ... )
        >>> segments = segmenter.segment(audio, sr=16000)
    """
    
    def __init__(
        self,
        model_ckpt: str = "cheoljun95/sylber",
        encoding_layer: int = 9,
        device: Optional[str] = None,
    ):
        self.model_ckpt = model_ckpt
        self.encoding_layer = encoding_layer
        self._model = None
        self._device = device
    
    def _lazy_load(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return
        
        try:
            from transformers import HubertModel, HubertConfig
            from huggingface_hub import hf_hub_download
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required for SylberFeatureExtractor. "
                "Install with: pip install transformers torch huggingface-hub"
            )
        
        # Determine device
        if self._device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self._device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, using CPU")
            self._device = 'cpu'
        
        # Load HuBERT base architecture
        speech_upstream = "facebook/hubert-base-ls960"
        config = HubertConfig.from_pretrained(
            speech_upstream,
            num_hidden_layers=self.encoding_layer
        )
        self._model = HubertModel(config)
        
        # Load Sylber's fine-tuned weights
        model_ckpt_path = hf_hub_download(
            repo_id=self.model_ckpt,
            filename="sylber.ckpt"
        )
        state_dict = torch.load(model_ckpt_path, map_location=self._device)
        self._model.load_state_dict(state_dict, strict=False)
        
        self._model = self._model.to(self._device)
        self._model.eval()
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract Sylber features from audio."""
        self._lazy_load()
        
        import torch
        import librosa
        
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio (Sylber does this)
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        
        # Extract features
        with torch.no_grad():
            outputs = self._model(audio_tensor)
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        return features
    
    @property
    def frame_rate(self) -> float:
        """Sylber produces features at 50 Hz (20ms frames)."""
        return 50.0


__all__ = ['SylberFeatureExtractor']
