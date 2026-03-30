"""
HuBERT feature extractor.

Extracts contextualized representations from Hugging Face HuBERT transformer models.
Default configuration: facebook/hubert-base-ls960, layer 9, 768-dim at 50 Hz.

Reference:
    Hsu, W. N., Bolte, B., Tsai, Y. H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021).
    HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.
    IEEE/ACM Transactions on Audio, Speech, and Language Processing.
"""

import numpy as np
from typing import Optional
import warnings

from .base import FeatureExtractor


class HuBERTExtractor(FeatureExtractor):
    """
    HuBERT feature extractor.
    
    Extracts contextualized representations from HuBERT transformer.
    Default configuration: facebook/hubert-base-ls960, layer 9, 768-dim at 50 Hz.
    
    Args:
        model_name: HuBERT model from Hugging Face (default: facebook/hubert-base-ls960)
        layer: Which layer to extract from (default: 9, range 0-11)
        device: Compute device ('cpu', 'cuda', or None for auto)
    
    Example:
        >>> extractor = HuBERTExtractor(layer=9)
        >>> features = extractor.extract(audio, sr=16000)
        >>> print(features.shape)  # (N, 768) where N depends on audio length
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        layer: int = 9,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.layer = layer
        self._model = None
        self._processor = None
        self._device = device
    
    def _lazy_load(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return
        
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required for HuBERTExtractor. "
                "Install with: pip install transformers torch"
            )
        
        self._model = HubertModel.from_pretrained(self.model_name)
        self._model.eval()
        
        # Move to device
        if self._device is None:
            import torch
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self._device == 'cuda':
            import torch
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            else:
                warnings.warn("CUDA requested but not available, using CPU")
                self._device = 'cpu'
        
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract HuBERT features."""
        self._lazy_load()
        
        import torch
        
        # Resample if needed
        if sr != self._processor.sampling_rate:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self._processor.sampling_rate
            )
        
        # Process
        inputs = self._processor(
            audio,
            sampling_rate=self._processor.sampling_rate,
            return_tensors="pt",
            padding=False
        )
        
        if self._device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self._model(
                inputs["input_values"],
                output_hidden_states=True
            )
            features = outputs.hidden_states[self.layer]
        
        return features[0].cpu().numpy()
    
    @property
    def frame_rate(self) -> float:
        """HuBERT produces features at 50 Hz."""
        return 50.0


__all__ = ['HuBERTExtractor']
