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
from typing import Optional, Tuple, Union
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
        self._model_with_attention = None
        self._processor = None
        self._device = device

    @property
    def supports_attention(self) -> bool:
        return True

    def _prepare_input_values(self, audio: np.ndarray, sr: int, needs_attention: bool = False):
        """Return model-ready input_values tensor for HuBERT."""
        self._lazy_load(needs_attention=needs_attention)

        import torch

        # Resample if needed
        if sr != self._processor.sampling_rate:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self._processor.sampling_rate
            )

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        inputs = self._processor(
            audio,
            sampling_rate=self._processor.sampling_rate,
            return_tensors="pt",
            padding=False
        )

        input_values = inputs["input_values"]
        if self._device == 'cuda':
            input_values = input_values.cuda()
        return input_values

    def _resolve_attention_layer(self, requested_layer: Optional[int]) -> int:
        """Map optional/negative layer index to a valid hidden-state layer index."""
        layer = self.layer if requested_layer is None else int(requested_layer)

        model_ref = self._model_with_attention if self._model_with_attention is not None else self._model
        if model_ref is None:
            raise RuntimeError("HuBERT model is not loaded")

        # hidden_states includes embedding output at index 0, then transformer layers.
        num_hidden_state_slots = model_ref.config.num_hidden_layers + 1
        if layer < 0:
            layer = num_hidden_state_slots + layer
        if not (0 <= layer < num_hidden_state_slots):
            raise ValueError(
                f"Requested layer {requested_layer} is out of range for HuBERT "
                f"(valid hidden-state indices: 0-{num_hidden_state_slots - 1})"
            )
        return layer

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
        """Lazy load model on first use.

        Args:
            needs_attention: If True, load an eager-attention model used for
                `output_attentions=True` calls.
        """
        if needs_attention:
            if self._model_with_attention is not None:
                return
        else:
            if self._model is not None:
                return
        
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required for HuBERTExtractor. "
                "Install with: pip install 'findsylls[embedding]'"
            )
        
        model_kwargs = {}
        if needs_attention:
            model_kwargs["attn_implementation"] = "eager"

        try:
            model = HubertModel.from_pretrained(self.model_name, **model_kwargs)
        except TypeError:
            # Older transformers may not expose attn_implementation.
            model = HubertModel.from_pretrained(self.model_name)

        model.eval()
        
        # Move to device
        if self._device is None:
            import torch
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self._device == 'cuda':
            import torch
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                warnings.warn("CUDA requested but not available, using CPU")
                self._device = 'cpu'

        if needs_attention:
            self._model_with_attention = model
        else:
            self._model = model
        
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract HuBERT features."""
        import torch
        input_values = self._prepare_input_values(audio, sr, needs_attention=False)
        layer_idx = self._resolve_attention_layer(self.layer)
        
        # Extract features
        with torch.no_grad():
            outputs = self._model(
                input_values,
                output_hidden_states=True
            )
            features = outputs.hidden_states[layer_idx]
        
        return features[0].cpu().numpy()

    def extract_with_attention(
        self,
        audio: np.ndarray,
        sr: int,
        layer: Optional[int] = None,
        aggregate: Union[str, int] = 'max',
        return_raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract HuBERT frame features and CLS-token attention.

        Args:
            audio: Audio waveform.
            sr: Audio sample rate in Hz.
            layer: Hidden-state layer index to expose as features.
            aggregate: How to aggregate attention heads ('max', 'mean', or int) when return_raw=False.
            return_raw: If True, return raw multi-head attention [n_heads, seq_len, src_len].
                       If False (default), return aggregated 1-D attention [seq_len].

        Returns:
            Tuple (features, attention) where attention is aligned to feature frames.
        """
        self._lazy_load(needs_attention=True)
        import torch

        input_values = self._prepare_input_values(audio, sr, needs_attention=True)
        feature_layer_idx = self._resolve_attention_layer(layer)
        model = self._model_with_attention if self._model_with_attention is not None else self._model
        if model is None:
            raise RuntimeError("HuBERT model is not loaded")

        with torch.no_grad():
            outputs = model(
                input_values,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        features = outputs.hidden_states[feature_layer_idx][0].cpu().numpy()

        # attentions are transformer-layer only (no embedding slot).
        attn_layer_idx = min(feature_layer_idx, len(outputs.attentions) - 1)
        attn = outputs.attentions[attn_layer_idx][0]  # (heads, seq_len, seq_len)
        cls_attn = attn[:, 0, :]

        # Drop CLS self-position if still present.
        if cls_attn.shape[1] > features.shape[0]:
            cls_attn = cls_attn[:, 1:]

        # Return raw multi-head attention if requested.
        # Preserve the natural shape [n_heads, T+1, T+1] including the CLS token
        # row/column so callers can extract attention[:, 0, 1:] → [n_heads, T].
        if return_raw:
            attn_raw = attn.cpu().numpy() if hasattr(attn, 'cpu') else attn
            return features, attn_raw.astype(np.float32, copy=False)

        cls_trace = self._aggregate_attention(cls_attn, aggregate)

        if len(cls_trace) > features.shape[0]:
            cls_trace = cls_trace[:features.shape[0]]
        elif len(cls_trace) < features.shape[0] and len(cls_trace) > 0:
            pad = features.shape[0] - len(cls_trace)
            cls_trace = np.pad(cls_trace, (0, pad), mode='edge')
        elif len(cls_trace) == 0:
            cls_trace = np.zeros(features.shape[0], dtype=np.float32)

        return features, cls_trace.astype(np.float32, copy=False)
    
    @property
    def frame_rate(self) -> float:
        """HuBERT produces features at 50 Hz."""
        return 50.0

    @property
    def has_cls_token(self) -> bool:
        """Standard HuBERT has no CLS token; raw attention is [n_heads, T, T]."""
        return False

    def release(self) -> None:
        """Release loaded model state so memory can be reclaimed."""
        self._model = None
        self._model_with_attention = None
        self._processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ['HuBERTExtractor']
