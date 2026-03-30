"""
VG-HuBERT feature extractor.

Extracts features from VG-HuBERT (Visually Grounded HuBERT) trained on SpokenCOCO
(Peng et al., Interspeech 2023).

Papers:
- Word Discovery in Visually Grounded, Self-Supervised Speech Models
  Peng & Harwath, Interspeech 2022
- Syllable Segmentation and Cross-Lingual Generalization in a Visually
  Grounded, Self-Supervised Speech Model
  Peng et al., Interspeech 2023

PyPI: https://pypi.org/project/vg-hubert/
Model: hjvm/VG-HuBERT on HuggingFace Hub

Key characteristics:
- Trained on SpokenCOCO with visual grounding
- Layer 8 best for syllables, layer 9 for words
- 768-dim features at 50 Hz
- Optimized for MinCut segmentation
"""

import numpy as np
from typing import Optional
import warnings

from .base import FeatureExtractor


class VGHuBERTFeatureExtractor(FeatureExtractor):
    """
    VG-HuBERT (Visually Grounded HuBERT) feature extractor.
    
    Extracts features from VG-HuBERT model trained with visual grounding.
    Layer is automatically chosen based on mode if not explicitly provided:
    - mode='syllable' → layer=8 (best for syllable boundaries)
    - mode='word' → layer=9 (best for word boundaries)
    
    The extractor automatically switches to eager attention when attention weights
    are requested via extract(return_attention=True), so you don't need to know
    upfront whether you'll use CLS or MinCut segmentation.
    
    Args:
        model_ckpt: HuggingFace checkpoint (default: "hjvm/VG-HuBERT")
        layer: Which layer to extract (default: None = auto-select based on mode)
        mode: Granularity mode - "syllable" or "word" (default: "syllable")
              Controls which checkpoint is loaded (syllable vs word trained weights)
        device: Compute device ('cpu', 'cuda', 'mps', or None for auto)
    
    Example:
        >>> # Create once, use for both MinCut (fast) and CLS (attention)
        >>> extractor = VGHuBERTFeatureExtractor(mode="syllable")
        >>> 
        >>> # MinCut: fast SDPA attention (no weights needed)
        >>> features = extractor.extract(audio, sr=16000)
        >>> 
        >>> # CLS: automatically switches to eager attention for weights
        >>> features, attn = extractor.extract(audio, sr=16000, return_attention=True)
        >>> 
        >>> # Word-level (auto layer=9)
        >>> extractor_word = VGHuBERTFeatureExtractor(mode="word")
    """
    
    def __init__(
        self,
        model_ckpt: str = "hjvm/VG-HuBERT",
        layer: Optional[int] = None,
        mode: str = "syllable",
        device: Optional[str] = None,
    ):
        self.model_ckpt = model_ckpt
        self.mode = mode
        
        # Auto-select layer based on mode if not provided
        if layer is None:
            self.layer = 8 if mode == "syllable" else 9
        else:
            self.layer = layer
        
        self._model = None  # SDPA model (fast, no attention)
        self._model_with_attention = None  # Eager model (slower, has attention)
        self._device = None
    
    def _lazy_load(self, needs_attention: bool = False):
        """
        Lazy load model on first use.
        
        Args:
            needs_attention: If True, load with eager attention for weight extraction.
                           If False, load with fast SDPA attention (default).
        """
        # Check if we already have the right model loaded
        if needs_attention:
            if self._model_with_attention is not None:
                return  # Already have attention-enabled model
        else:
            if self._model is not None:
                return  # Already have fast SDPA model
        
        try:
            from vg_hubert import Segmenter
            import torch
        except ImportError:
            raise ImportError(
                "vg-hubert and torch required for VGHuBERTFeatureExtractor. "
                "Install with: pip install vg-hubert torch"
            )
        
        # Determine device (only once)
        if self._device is None:
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                self._device = 'cpu'
        
        # Load VG-HuBERT using Segmenter wrapper
        # segmentation_method controls attention: 'CLS' → eager, None → SDPA
        segmentation_method = 'CLS' if needs_attention else None
        
        model = Segmenter(
            model_ckpt=self.model_ckpt,
            mode=self.mode,
            layer=self.layer,
            segmentation_method=segmentation_method,
            device=self._device
        )
        
        # Cache in appropriate slot
        if needs_attention:
            self._model_with_attention = model
        else:
            self._model = model
    
    def extract(self, audio: np.ndarray, sr: int, return_attention: bool = False):
        """
        Extract VG-HuBERT features from audio.
        
        Automatically uses the appropriate attention implementation:
        - return_attention=False: Fast SDPA attention (no weights)
        - return_attention=True: Eager attention (extracts weights)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            return_attention: If True, returns (features, cls_attention_scores)
                            If False, returns features only
        
        Returns:
            If return_attention=False: np.ndarray of shape (N, 768)
            If return_attention=True: tuple of (features, cls_attention) where
                                     cls_attention is np.ndarray of shape (N,)
        """
        # Load appropriate model based on whether attention is needed
        self._lazy_load(needs_attention=return_attention)
        
        # Get the right model
        model = self._model_with_attention if return_attention else self._model
        
        import librosa
        import torch
        import numpy as np
        
        # Resample if needed (VG-HuBERT expects 16kHz)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if not return_attention:
            # Use Segmenter to extract features
            # The Segmenter.__call__ returns a dict with 'hidden_states' 
            # which contains the features from the specified layer
            outputs = model(wav=audio, in_second=True)
            
            # Extract features (already numpy array from specified layer)
            features = outputs['hidden_states']
            
            return features
        else:
            # Need to extract both features and attention
            # Access the underlying model directly
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            if self._device in ['cuda', 'mps']:
                audio_tensor = audio_tensor.to(self._device)
            
            with torch.no_grad():
                outputs = model.model(
                    input_values=audio_tensor,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
                
                # Extract features from specified layer
                features = outputs.hidden_states[self.layer][0].cpu().numpy()
                
                # Extract CLS token attention from specified layer
                # Shape: (num_heads, seq_len, seq_len)
                attn = outputs.attentions[self.layer][0]
                
                # CLS token attention: first token attends to all positions
                # Shape: (num_heads, seq_len)
                cls_attn = attn[:, 0, :]
                
                # Remove CLS position itself (first position)
                if cls_attn.shape[1] > features.shape[0]:
                    cls_attn = cls_attn[:, 1:]
                
                # Take max across heads (following VG-HuBERT paper)
                cls_attn_score = cls_attn.max(dim=0)[0].cpu().numpy()
                
                # Ensure alignment with features
                if len(cls_attn_score) > features.shape[0]:
                    cls_attn_score = cls_attn_score[:features.shape[0]]
                elif len(cls_attn_score) < features.shape[0]:
                    # Pad if needed
                    import numpy as np
                    padding = features.shape[0] - len(cls_attn_score)
                    cls_attn_score = np.pad(cls_attn_score, (0, padding), mode='edge')
            
            return features, cls_attn_score
    
    @property
    def frame_rate(self) -> float:
        """VG-HuBERT produces features at 50 Hz."""
        return 50.0


__all__ = ['VGHuBERTFeatureExtractor']
