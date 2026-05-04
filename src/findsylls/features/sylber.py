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
from typing import Optional, Tuple, Union
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

    @property
    def supports_attention(self) -> bool:
        return True

    def _prepare_audio_tensor(self, audio: np.ndarray, sr: int):
        """Prepare normalized audio tensor for the Sylber model."""
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

        return torch.from_numpy(audio).unsqueeze(0).to(self._device)

    def _resolve_hidden_layer(self, requested_layer: Optional[int]) -> int:
        """Resolve/validate hidden-state layer index for Sylber model outputs."""
        layer = self.encoding_layer if requested_layer is None else int(requested_layer)
        num_hidden_state_slots = self._model.config.num_hidden_layers + 1

        if layer < 0:
            layer = num_hidden_state_slots + layer
        if not (0 <= layer < num_hidden_state_slots):
            raise ValueError(
                f"Requested layer {requested_layer} is out of range for Sylber "
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
                "sylber, transformers, and torch required for SylberFeatureExtractor. "
                "Install with: pip install 'findsylls[end2end]'"
            )
        
        # Determine device
        if self._device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self._device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, using CPU")
            self._device = 'cpu'
        
        # Load HuBERT base architecture.
        # Force eager attention so output_attentions=True doesn't trigger
        # SDPA fallback warnings and remains compatible with transformers v5.
        speech_upstream = "facebook/hubert-base-ls960"
        config = HubertConfig.from_pretrained(
            speech_upstream,
            num_hidden_layers=self.encoding_layer
        )

        # Newer transformers honor this private config knob.
        # Keep a fallback for versions that require constructor args.
        try:
            config._attn_implementation = "eager"
        except Exception:
            pass

        try:
            self._model = HubertModel(config, attn_implementation="eager")
        except TypeError:
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
        import torch
        audio_tensor = self._prepare_audio_tensor(audio, sr)
        
        # Extract features
        with torch.no_grad():
            outputs = self._model(audio_tensor)
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        return features

    def extract_with_attention(
        self,
        audio: np.ndarray,
        sr: int,
        layer: Optional[int] = None,
        aggregate: Union[str, int] = 'max',
        return_raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Sylber features and CLS-token attention.

        Args:
            layer: Hidden-state layer index to expose as features.
            aggregate: How to aggregate attention heads ('max', 'mean', or int) when return_raw=False.
            return_raw: If True, return raw multi-head attention [n_heads, seq_len, src_len].
                       If False (default), return aggregated 1-D attention [seq_len].

        Raises:
            RuntimeError: When underlying model/config does not expose attention tensors.
        """
        self._lazy_load()

        import torch

        audio_tensor = self._prepare_audio_tensor(audio, sr)
        feature_layer_idx = self._resolve_hidden_layer(layer)

        with torch.no_grad():
            outputs = self._model(
                audio_tensor,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        if outputs.attentions is None:
            raise RuntimeError(
                "SylberFeatureExtractor attention is unavailable for this model/config"
            )

        features = outputs.hidden_states[feature_layer_idx][0].cpu().numpy()
        attn_layer_idx = min(feature_layer_idx, len(outputs.attentions) - 1)
        attn = outputs.attentions[attn_layer_idx][0]
        cls_attn = attn[:, 0, :]

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
        """Sylber produces features at 50 Hz (20ms frames)."""
        return 50.0

    @property
    def has_cls_token(self) -> bool:
        """Sylber is fine-tuned HuBERT — no CLS token; raw attention is [n_heads, T, T]."""
        return False

    def release(self) -> None:
        """Release loaded model state so memory can be reclaimed."""
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ['SylberFeatureExtractor']
