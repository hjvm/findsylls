"""
VG-HuBERT syllable segmentation using MinCut algorithm.

This implementation is based on:
    Peng et al. (2023). Syllable Discovery and Cross-Lingual Generalization 
    in a Visually Grounded, Self-Supervised Speech Model. Interspeech 2023.
    https://github.com/jasonppy/syllable-discovery

The method uses:
1. VG-HuBERT features (visually-grounded HuBERT trained on SpokenCOCO)
2. Self-similarity matrix computed from features
3. MinCut algorithm to partition into syllable-like segments
4. Optional merging of adjacent segments based on feature similarity
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
import os
import warnings
import logging

from ..base import End2EndSegmenter
from ..algorithms import min_cut

logger = logging.getLogger(__name__)


class VGHubertSegmenter(End2EndSegmenter):
    """
    VG-HuBERT-based syllable segmenter using MinCut algorithm.
    
    This segmenter uses pre-trained VG-HuBERT (Visually Grounded HuBERT) to extract
    features, then applies a graph-based MinCut algorithm to partition the audio
    into syllable-like segments.
    
    Parameters:
        model_path: Path to directory containing VG-HuBERT checkpoint files:
                   - best_bundle.pth (or snapshot_*.pth)
                   - args.pkl (model configuration)
        layer: Which HuBERT layer to extract features from (0-11, default: 8)
        sec_per_syllable: Target syllable duration in seconds (default: 0.2)
                         Used to estimate number of segments: K = ceil(audio_len / sec_per_syllable)
        merge_threshold: Optional cosine similarity threshold for merging adjacent segments.
                        If None, no merging. Typical values: 0.3-0.4 (default: None)
        reduce_method: How to pool features within a segment (default: 'mean')
                      Options: 'mean', 'max', 'median'
        device: torch device ('cuda' or 'cpu', default: 'cuda' if available)
        snapshot: Which checkpoint to load (default: 'best')
                 - 'best': load best_bundle.pth (better for syllables)
                 - int: load snapshot_N.pth (e.g., 20 is better for words)
    
    Example:
        >>> segmenter = VGHubertSegmenter(
        ...     model_path='/path/to/vg-hubert_3',
        ...     layer=8,
        ...     sec_per_syllable=0.2,
        ...     merge_threshold=0.3
        ... )
        >>> syllables = segmenter.segment_audio('/path/to/audio.wav')
        >>> # Returns list of (start, peak, end) tuples in seconds
    """
    
    def __init__(
        self,
        model_path: str,
        layer: int = 8,
        sec_per_syllable: float = 0.2,
        merge_threshold: Optional[float] = None,
        reduce_method: str = 'mean',
        device: str = 'cpu',
        snapshot: str = 'best',
        sample_rate: int = 16000,
        cache: bool = True
    ):
        super().__init__(sample_rate=sample_rate, device=device, cache=cache)
        
        self.model_path = model_path
        self.layer = layer
        self.sec_per_syllable = sec_per_syllable
        self.merge_threshold = merge_threshold
        self.reduce_method = reduce_method
        self.snapshot = snapshot
        
        # Lazy load model
        self._model = None
        
    def _lazy_load_model(self):
        """Lazy load VG-HuBERT model on first use."""
        if self._model is None or not self.cache:
            import pickle
            from pathlib import Path
            
            # Check if model path exists
            if not os.path.isdir(self.model_path):
                # Provide helpful error message with suggestions
                model_path = Path(self.model_path)
                parent_exists = model_path.parent.exists()
                
                error_msg = f"VG-HuBERT model path does not exist: {self.model_path}\n\n"
                
                if parent_exists:
                    # Parent directory exists, so likely a typo or wrong subdirectory
                    error_msg += "The parent directory exists, but this specific path was not found.\n"
                    error_msg += "Common issues:\n"
                    error_msg += "  • Typo in directory name (check spelling)\n"
                    error_msg += "  • Missing subdirectory (e.g., 'vg-hubert_3' vs 'vg-hubert-3')\n"
                    error_msg += f"  • Check what's in parent: ls {model_path.parent}\n\n"
                else:
                    # Parent doesn't exist, so likely model not downloaded
                    error_msg += "The directory structure doesn't exist. Model may not be downloaded.\n\n"
                
                error_msg += "To download VG-HuBERT:\n"
                error_msg += "  wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar\n"
                error_msg += f"  tar -xf vg-hubert_3.tar -C {model_path.parent}/\n\n"
                error_msg += "Expected directory structure:\n"
                error_msg += f"  {self.model_path}/\n"
                error_msg += "    ├── best_bundle.pth\n"
                error_msg += "    ├── snapshot_20.pth\n"
                error_msg += "    └── args.pkl"
                
                raise FileNotFoundError(error_msg)
            
            # Check for required files
            args_path = os.path.join(self.model_path, "args.pkl")
            if not os.path.exists(args_path):
                # Directory exists but missing required files
                actual_files = os.listdir(self.model_path)
                error_msg = (
                    f"VG-HuBERT model directory exists but is missing required files.\n\n"
                    f"Path: {self.model_path}\n"
                    f"Missing: args.pkl\n\n"
                    f"Found files: {', '.join(actual_files)}\n\n"
                    f"This suggests an incomplete download or extraction.\n"
                    f"Please re-download and extract:\n"
                    f"  wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar\n"
                    f"  tar -xf vg-hubert_3.tar\n\n"
                    f"Expected files:\n"
                    f"  - best_bundle.pth (or snapshot_*.pth)\n"
                    f"  - args.pkl"
                )
                raise FileNotFoundError(error_msg)
            
            with open(args_path, "rb") as f:
                model_args = pickle.load(f)
            
            # Get device
            device = self._get_device()
            device_obj = torch.device(device)
            
            # Load VG-HuBERT using transformers HuBERT with trained weights
            # This avoids fairseq dependency issues with Python 3.11+
            checkpoint_path = None
            if self.snapshot == 'best' or self.snapshot == 'best_bundle':
                checkpoint_path = os.path.join(self.model_path, "best_bundle.pth")
            else:
                checkpoint_path = os.path.join(self.model_path, f"snapshot_{self.snapshot}.pth")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Available files in {self.model_path}:\n" +
                    "\n".join(os.listdir(self.model_path))
                )
            
            try:
                from .vg_hubert_models import load_vg_hubert_into_transformers
                model = load_vg_hubert_into_transformers(checkpoint_path, device=device)
                logger.info("Successfully loaded VG-HuBERT model with trained weights")
            except Exception as e:
                # Fallback to base transformers HuBERT
                warnings.warn(
                    f"Failed to load VG-HuBERT weights ({e}). "
                    "Falling back to transformers HuBERT (facebook/hubert-base-ls960). "
                    "Performance will be significantly worse than actual VG-HuBERT.",
                    UserWarning,
                    stacklevel=2
                )
                logger.warning(f"Using transformers HuBERT as fallback: {e}")
                from transformers import HubertModel
                model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
                model.eval()
                model = model.to(device_obj)
            
            self._model = model
        
        return self._model

    
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, float]:
        """
        Extract features from audio using VG-HuBERT.
        
        Args:
            audio: Audio waveform (mono, float32)
            sr: Sample rate (must be 16000)
            
        Returns:
            Tuple of:
                - features: numpy array of shape (T, D) where T is time frames, D is feature dim
                - spf: seconds per frame (for converting frame indices to time)
        """
        if sr != 16000:
            raise ValueError(f"VG-HuBERT requires 16kHz audio, got {sr}Hz")
        
        if len(audio.shape) != 1:
            raise ValueError(f"Audio must be mono (1D), got shape {audio.shape}")
        
        # Lazy load model
        model = self._lazy_load_model()
        device = self._get_device()
        device_obj = torch.device(device)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device_obj)
        
        with torch.no_grad():
            # Check if using transformers HuBERT or custom model
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                # Transformers HuBERT
                out = model(
                    input_values=audio_tensor,
                    output_hidden_states=True,
                    return_dict=True
                )
                feat = out['hidden_states'][self.layer][0].cpu().float().numpy()
            else:
                # Custom VG-HuBERT model
                out = model(
                    audio_tensor,
                    padding_mask=None,
                    mask=False,
                    tgt_layer=self.layer,
                    need_attention_weights=False,
                    pre_feats=False
                )
                # Remove CLS token if present
                feat = out['features'].squeeze(0)
                if hasattr(model, 'use_audio_cls_token') and model.use_audio_cls_token:
                    feat = feat[1:]  # Remove CLS token
                feat = feat.cpu().float().numpy()
        
        # Calculate seconds per frame
        spf = len(audio) / sr / feat.shape[0]
        
        return feat, spf
    
    def _merge_segments(
        self,
        seg_boundary_pairs: List[List[int]],
        feat: np.ndarray,
        merge_threshold: float
    ) -> List[List[int]]:
        """
        Merge adjacent segments if their feature similarity exceeds threshold.
        
        Args:
            seg_boundary_pairs: List of [start_frame, end_frame] pairs
            feat: Feature array of shape (T, D)
            merge_threshold: Cosine similarity threshold (0-1)
            
        Returns:
            Merged list of [start_frame, end_frame] pairs
        """
        if len(seg_boundary_pairs) < 3:
            return seg_boundary_pairs
        
        # Compute mean features for each segment
        all_feat = [
            feat[round(l):round(r)].mean(0) 
            for l, r in seg_boundary_pairs
        ]
        
        # Compute cosine similarities between adjacent segments
        all_sim = [
            np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            for f1, f2 in zip(all_feat[:-1], all_feat[1:])
        ]
        
        # Iteratively merge most similar adjacent segments above threshold
        while len(seg_boundary_pairs) >= 3:
            max_sim_idx = np.argmax(all_sim)
            
            if all_sim[max_sim_idx] < merge_threshold:
                break  # No more segments to merge
            
            # Merge segments at max_sim_idx and max_sim_idx+1
            l_merge = seg_boundary_pairs[max_sim_idx]
            r_merge = seg_boundary_pairs[max_sim_idx + 1]
            
            # Remove old segments and insert merged one
            seg_boundary_pairs = [
                pair for i, pair in enumerate(seg_boundary_pairs)
                if i != max_sim_idx and i != max_sim_idx + 1
            ]
            seg_boundary_pairs.insert(max_sim_idx, [l_merge[0], r_merge[1]])
            
            # Recompute features and similarities
            all_feat = [
                feat[round(l):round(r)].mean(0)
                for l, r in seg_boundary_pairs
            ]
            all_sim = [
                np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
                for f1, f2 in zip(all_feat[:-1], all_feat[1:])
            ]
        
        return seg_boundary_pairs
    
    def segment(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> List[Tuple[float, float, float]]:
        """
        Segment audio into syllable-like units.
        
        Args:
            audio: Audio waveform (mono, float32), shape (N,)
            sr: Sample rate (must be 16000)
            
        Returns:
            List of (start, peak, end) tuples in seconds, where:
                - start: segment start time
                - peak: approximate syllable nucleus time (segment midpoint)
                - end: segment end time
        """
        # Extract features
        feat, spf = self.extract_features(audio, sr)
        
        # Estimate number of syllables
        audio_len_sec = len(audio) / sr
        num_syllable = int(np.ceil(audio_len_sec / self.sec_per_syllable))
        
        if num_syllable <= 0:
            num_syllable = 1
        
        # Compute self-similarity matrix
        ssm = feat @ feat.T
        ssm = ssm - np.min(ssm) + 1e-7  # Make non-negative
        
        # Run MinCut to get boundary frames
        # K+1 boundaries create K segments
        seg_boundary_frame = min_cut(ssm, num_syllable + 1)
        
        # Convert to pairs of [start, end] frames
        seg_boundary_frame_pairs = [
            [l, r]
            for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])
        ]
        
        # Filter out very short segments (< 2 frames)
        seg_boundary_frame_pairs = [
            item for item in seg_boundary_frame_pairs
            if item[1] - item[0] > 2
        ]
        
        # If all segments filtered out, use original
        if len(seg_boundary_frame_pairs) == 0:
            seg_boundary_frame_pairs = [
                [l, r]
                for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])
            ]
        
        # Optional merging step
        if self.merge_threshold is not None and len(seg_boundary_frame_pairs) >= 3:
            seg_boundary_frame_pairs = self._merge_segments(
                seg_boundary_frame_pairs,
                feat,
                self.merge_threshold
            )
        
        # Convert frame boundaries to time boundaries
        # Return as (start, peak, end) tuples
        syllables = []
        for start_frame, end_frame in seg_boundary_frame_pairs:
            start_time = start_frame * spf
            end_time = end_frame * spf
            peak_time = (start_time + end_time) / 2.0  # Midpoint as nucleus proxy
            syllables.append((start_time, peak_time, end_time))
        
        return syllables
