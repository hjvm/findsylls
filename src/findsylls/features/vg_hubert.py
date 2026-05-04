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
from typing import Optional, Tuple, Union
import warnings

from .base import FeatureExtractor


class VGHuBERTFeatureExtractor(FeatureExtractor):
    """
    VG-HuBERT (Visually Grounded HuBERT) feature extractor.
    
    Extracts features from VG-HuBERT model trained with visual grounding.
    Layer is automatically chosen based on mode if not explicitly provided:
    - mode='syllable' → layer=8 (fairseq tgt_layer=8, best for syllable boundaries)
    - mode='word' → layer=9 (fairseq tgt_layer=9, best for word boundaries)

    Layer numbers follow the fairseq / paper convention (0-indexed transformer layer).
    Internally, HuggingFace hidden_states[layer+1] is used so that the output of
    transformer layer `layer` is extracted (matching the reference implementations).
    
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

    @property
    def supports_attention(self) -> bool:
        return True

    @staticmethod
    def _aggregate_attention(cls_attn, aggregate: Union[str, int]) -> np.ndarray:
        """Aggregate CLS attention across heads into a 1-D trace."""
        if aggregate == 'max':
            return cls_attn.max(dim=0)[0].cpu().numpy()
        if aggregate == 'mean':
            return cls_attn.mean(dim=0).cpu().numpy()

        if isinstance(aggregate, int):
            if aggregate < 0 or aggregate >= cls_attn.shape[0]:
                raise ValueError(
                    f"Attention head index {aggregate} is out of range "
                    f"(num_heads={cls_attn.shape[0]})"
                )
            return cls_attn[aggregate].cpu().numpy()

        raise ValueError("aggregate must be one of {'max', 'mean'} or an int head index")

    @staticmethod
    def _align_raw_attention(attention: np.ndarray, n_frames: int) -> np.ndarray:
        """Align raw attention tensor to feature frame count as [H, N, N]."""
        if attention.ndim != 3:
            raise ValueError(f"Expected raw attention with 3 dims [H, T, S], got shape {attention.shape}")

        h, t, s = attention.shape
        if t > n_frames:
            attention = attention[:, :n_frames, :]
            t = n_frames
        elif t < n_frames and t > 0:
            attention = np.pad(attention, ((0, 0), (0, n_frames - t), (0, 0)), mode='edge')
            t = n_frames

        if s > n_frames:
            attention = attention[:, :, :n_frames]
        elif s < n_frames and s > 0:
            attention = np.pad(attention, ((0, 0), (0, 0), (0, n_frames - s)), mode='edge')

        if attention.shape[1] == 0 or attention.shape[2] == 0:
            return np.zeros((h, n_frames, n_frames), dtype=np.float32)

        return attention
    
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
                "Install with: pip install 'findsylls[end2end]'"
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
        if not return_attention:
            # Load appropriate model based on whether attention is needed
            self._lazy_load(needs_attention=False)

            # Get the right model
            model = self._model

            import librosa

            # Resample if needed (VG-HuBERT expects 16kHz)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Use Segmenter to extract features
            # The Segmenter.__call__ returns a dict with 'hidden_states' 
            # which contains the features from the specified layer
            outputs = model(wav=audio, in_second=True)
            
            # Extract features (already numpy array from specified layer)
            features = outputs['hidden_states']
            
            return features

        features, cls_attn_score = self.extract_with_attention(audio, sr)
        return features, cls_attn_score

    def extract_with_attention(
        self,
        audio: np.ndarray,
        sr: int,
        layer: Optional[int] = None,
        aggregate: Union[str, int] = 'max',
        return_raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract VG-HuBERT features and CLS-token attention.

        Args:
            audio: Audio signal.
            sr: Sample rate.
            layer: Optional hidden-state layer override.
            aggregate: How to aggregate heads ('max', 'mean', or int) when return_raw=False.
            return_raw: If True, return raw multi-head attention [n_heads, seq_len, src_len].
                       If False (default), return aggregated 1-D attention [seq_len].
        """
        if layer is None:
            requested_layer = self.layer
        else:
            layer_int = int(layer)
            requested_layer = self.layer if layer_int < 0 else layer_int

        # Current Segmenter wrapper is configured for one layer at init-time.
        if requested_layer != self.layer:
            raise ValueError(
                f"VGHuBERTFeatureExtractor is configured for layer={self.layer}; "
                f"requested layer={requested_layer}. Instantiate a new extractor for a different layer."
            )

        self._lazy_load(needs_attention=True)
        model = self._model_with_attention

        import librosa
        import torch

        # Resample if needed (VG-HuBERT expects 16kHz)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

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

        # hidden_states[layer+1] = output of transformer layer `layer` (0-indexed).
        # Matches fairseq tgt_layer convention: layer 8 → hidden_states[9].
        features = outputs.hidden_states[self.layer + 1][0].cpu().numpy()

        # Strip CLS token from features when the encoder was augmented with CLS injection.
        # CLS occupies position 0 and is not a real audio frame.
        if getattr(model, 'has_cls_token', False):
            features = features[1:]

        # attentions[layer] = attention FROM transformer layer `layer` (0-indexed).
        # Same layer as features above, so features and attention are aligned.
        attn = outputs.attentions[self.layer][0]
        cls_attn = attn[:, 0, :]  # [n_heads, T+1]

        # Remove CLS self-attention column (position 0) — only present after CLS injection.
        if cls_attn.shape[1] > features.shape[0]:
            cls_attn = cls_attn[:, 1:]

        # Return raw multi-head attention if requested.
        # Preserve the natural shape [n_heads, T+1, T+1] including the CLS token
        # row/column so callers can extract attention[:, 0, 1:] → [n_heads, T].
        if return_raw:
            attn_raw = attn.cpu().numpy() if hasattr(attn, 'cpu') else attn
            return features, attn_raw.astype(np.float32, copy=False)

        # Aggregate attention to 1-D for default behavior
        cls_attn_score = self._aggregate_attention(cls_attn, aggregate)

        # Ensure alignment with features
        if len(cls_attn_score) > features.shape[0]:
            cls_attn_score = cls_attn_score[:features.shape[0]]
        elif len(cls_attn_score) < features.shape[0] and len(cls_attn_score) > 0:
            padding = features.shape[0] - len(cls_attn_score)
            cls_attn_score = np.pad(cls_attn_score, (0, padding), mode='edge')
        elif len(cls_attn_score) == 0:
            cls_attn_score = np.zeros(features.shape[0], dtype=np.float32)

        return features, cls_attn_score.astype(np.float32, copy=False)
    
    @property
    def frame_rate(self) -> float:
        """VG-HuBERT produces features at 50 Hz."""
        return 50.0

    @property
    def has_cls_token(self) -> bool:
        """VG-HuBERT injects a CLS token; raw attention is [n_heads, T+1, T+1]."""
        return True

    def release(self) -> None:
        """Release loaded model state so memory can be reclaimed."""
        self._model = None
        self._model_with_attention = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ['VGHuBERTFeatureExtractor']
