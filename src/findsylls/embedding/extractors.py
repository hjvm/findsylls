"""
Feature extraction for syllable embeddings.

Extracts frame-level features from audio waveforms using various methods:
- Neural: Sylber, VG-HuBERT, HuBERT, Wav2Vec2
- Classical: MFCC, Mel-spectrogram

All functions return NumPy arrays, even if computed with PyTorch internally.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import warnings

from ..envelope.dispatch import get_amplitude_envelope

from ..features import get_extractor


# Global model cache for lazy loading
_SYLBER_SEGMENTER = None
_VG_HUBERT_SEGMENTER = None


def _detect_peaks_cosine_similarity(
    features: np.ndarray,
    segments: List[Tuple[float, float]],
    fps: float
) -> List[Tuple[float, float, float]]:
    """
    Detect acoustic peaks within segments using cosine similarity.
    
    For each segment, computes frame-to-frame cosine similarity and finds
    the frame with maximum similarity to its neighbors. This is native to
    Sylber's learned representation space.
    
    Args:
        features: (num_frames, feature_dim) frame-level features
        segments: List of (start, end) tuples in seconds
        fps: Frames per second
        
    Returns:
        segments_with_peaks: List of (start, peak, end) tuples in seconds
    """
    def cosine_sim(a, b):
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    segments_with_peaks = []
    frame_hop = 1.0 / fps
    
    for start, end in segments:
        # Convert to frame indices
        start_frame = int(round(start / frame_hop))
        end_frame = int(round(end / frame_hop))
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, features.shape[0] - 1))
        end_frame = max(0, min(end_frame, features.shape[0]))
        
        # Handle edge cases
        if end_frame - start_frame <= 1:
            # Very short segment - peak is the midpoint
            peak_frame = start_frame
        else:
            # Compute frame-to-frame similarities within segment
            # For each frame, compute similarity to previous and next frames
            similarities = []
            for i in range(start_frame, end_frame):
                sim_sum = 0.0
                count = 0
                
                # Similarity to previous frame
                if i > start_frame:
                    sim_sum += cosine_sim(features[i], features[i-1])
                    count += 1
                
                # Similarity to next frame
                if i < end_frame - 1:
                    sim_sum += cosine_sim(features[i], features[i+1])
                    count += 1
                
                # Average similarity
                avg_sim = sim_sum / count if count > 0 else 0.0
                similarities.append(avg_sim)
            
            # Find frame with maximum similarity (most stable/prototypical)
            peak_idx = np.argmax(similarities)
            peak_frame = start_frame + peak_idx
        
        # Convert back to seconds
        peak = peak_frame * frame_hop
        segments_with_peaks.append((start, peak, end))
    
    return segments_with_peaks


def extract_features(
    audio: np.ndarray,
    sr: int,
    method: str = 'sylber',
    layer: Optional[int] = None,
    device: str = 'auto',
    return_times: bool = False,
    return_segments: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List]]:
    """
    Extract frame-level features from audio.
    
    Args:
        audio: Audio waveform (mono, float32)
        sr: Sample rate in Hz
        method: Feature extraction method
            - 'sylber': Sylber model (768-dim, ~50 fps)
            - 'vg_hubert': VG-HuBERT model (768-dim, ~50 fps)
            - 'mfcc': Mel-frequency cepstral coefficients
            - 'melspec': Mel-spectrogram
            - 'rms'/'hilbert'/'lowpass'/'sbs'/'theta': Envelope-derived 1-D features
        layer: Layer index for neural models (model-specific defaults if None)
        device: 'auto', 'cuda', 'cpu' (for neural models)
        return_times: If True, return time points for each frame
        return_segments: If True and method supports it, return detected segments with peaks
                        (e.g., Sylber returns [(start, peak, end), ...])
        **kwargs: Additional method-specific parameters
        
    Returns:
        If return_times=False and return_segments=False:
            features: np.ndarray, shape (num_frames, feature_dim)
        If return_times=True and return_segments=False:
            (features, times): features and corresponding time points in seconds
        If return_times=True and return_segments=True:
            (features, times, segments): features, times, and segments with peaks
            
    Raises:
        ValueError: If method is unknown
        ImportError: If required dependencies are missing
    """
    if method == 'sylber':
        result = _extract_sylber_features(
            audio, sr, layer=layer, device=device, 
            return_segments=return_segments, **kwargs
        )
    elif method == 'vg_hubert':
        features, times = _extract_vg_hubert_features(audio, sr, layer=layer, device=device, **kwargs)
        result = (features, times) if not return_segments else (features, times, None)
    elif method == 'mfcc':
        features, times = _extract_mfcc_features(audio, sr, **kwargs)
        result = (features, times) if not return_segments else (features, times, None)
    elif method == 'melspec':
        features, times = _extract_melspec_features(audio, sr, **kwargs)
        result = (features, times) if not return_segments else (features, times, None)
    elif method in {'rms', 'hilbert', 'lowpass', 'sbs', 'theta'}:
        features, times = _extract_envelope_features(audio, sr, method=method, **kwargs)
        result = (features, times) if not return_segments else (features, times, None)
    else:
        raise ValueError(
            f"Unknown feature extraction method: '{method}'. "
            f"Supported methods: 'sylber', 'vg_hubert', 'mfcc', 'melspec', 'rms', 'hilbert', 'lowpass', 'sbs', 'theta'"
        )
    
    # Handle return format
    if not return_times and not return_segments:
        return result[0]  # Just features
    elif return_times and not return_segments:
        return result[:2]  # Features and times
    else:
        return result  # All three


def _extract_sylber_features(
    audio: np.ndarray,
    sr: int,
    layer: Optional[int] = None,
    device: str = 'auto',
    return_segments: bool = False,
    **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List]]:
    """
    Extract Sylber features from audio.
    
    Reuses the existing SylberSegmenter to avoid duplication. Extracts frame-level
    hidden states (768-dim at ~50 fps) for subsequent pooling. Optionally returns
    detected syllable segments with peaks identified using cosine similarity.
    
    Args:
        audio: Audio waveform (mono, float32)
        sr: Sample rate
        layer: Which transformer layer to use (ignored - Sylber uses layer 9)
        device: Device for inference ('cpu', 'cuda', 'mps', 'auto')
        return_segments: If True, return segments with peaks detected via cosine similarity
        **kwargs: Additional parameters (ignored)
    
    Returns:
        If return_segments=False:
            (features, times): (num_frames, 768) array and time points
        If return_segments=True:
            (features, times, segments): features, times, and list of (start, peak, end) tuples
    """
    global _SYLBER_SEGMENTER
    
    # Lazy load segmenter (reuses existing SylberSegmenter preset)
    if _SYLBER_SEGMENTER is None:
        try:
            from ..segmentation.presets import SylberSegmenter
        except ImportError:
            raise ImportError(
                "Sylber feature extraction requires the sylber package. "
                "Install with: pip install sylber"
            )
        
        _SYLBER_SEGMENTER = SylberSegmenter(sample_rate=16000, device=device)
    
    # Use segmenter's internal Sylber model to get raw outputs
    # This reuses all the device handling, temp file workaround, etc.
    segmenter = _SYLBER_SEGMENTER._lazy_load_model()
    
    # Resample if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Use the same temp file approach as SylberSegmenter
    import tempfile
    import soundfile as sf
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    try:
        sf.write(temp_file, audio, 16000)
        outputs = segmenter(wav_file=temp_file, in_second=True)
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    # Extract hidden_states (raw frame-level features)
    # outputs = {"segments": array of [start, end],
    #            "segment_features": segment-averaged features (pre-pooled),
    #            "hidden_states": raw frame-level features}
    features = outputs['hidden_states']  # Shape: (num_frames, 768)
    
    # Calculate time points (Sylber outputs ~50 fps)
    num_frames = features.shape[0]
    duration = len(audio) / sr
    times = np.linspace(0, duration, num_frames)
    fps = num_frames / duration if duration > 0 else 50.0
    
    if return_segments:
        # Extract segments and detect peaks using cosine similarity
        raw_segments = outputs['segments']  # (N, 2) array of [start, end]
        segments_list = [(s, e) for s, e in raw_segments]
        segments_with_peaks = _detect_peaks_cosine_similarity(features, segments_list, fps)
        return features, times, segments_with_peaks
    else:
        return features, times


def _extract_vg_hubert_features(
    audio: np.ndarray,
    sr: int,
    layer: Optional[int] = None,
    device: str = 'auto',
    model_ckpt: Optional[str] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features using VG-HuBERT model from vg-hubert PyPI package.
    
    Args:
        audio: Audio waveform (mono)
        sr: Sample rate (must be 16000 for VG-HuBERT)
        layer: Layer to extract from (default: 8 for syllables)
        device: Device placement ('auto', 'cuda', 'cpu', 'mps')
        model_ckpt: HuggingFace model checkpoint or local path (default: "hjvm/VG-HuBERT")
                   If None, uses default from HuggingFace Hub with automatic download
        **kwargs: Additional arguments (e.g., mode='syllable')
        
    Returns:
        (features, times): (num_frames, feature_dim) array and time points
        
    Raises:
        ValueError: If sr != 16000
        ImportError: If vg-hubert package not installed
    """
    global _VG_HUBERT_SEGMENTER
    
    if sr != 16000:
        raise ValueError(f"VG-HuBERT requires 16kHz audio, got {sr}Hz. Please resample.")
    
    # Handle device
    if device == 'auto':
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Default model checkpoint (auto-downloads from HuggingFace)
    if model_ckpt is None:
        model_ckpt = "hjvm/VG-HuBERT"
    
    # Default layer and mode
    if layer is None:
        layer = 8
    mode = kwargs.get('mode', 'syllable')
    
    # Lazy load VG-HuBERT segmenter (we'll reuse its feature extraction)
    if _VG_HUBERT_SEGMENTER is None:
        try:
            from ..segmentation.end2end.vg_hubert_segmenter import VGHubertSegmenter
        except ImportError:
            raise ImportError(
                "VG-HuBERT feature extraction requires the VG-HuBERT segmenter. "
                "Make sure findsylls[embedding] is properly installed with: pip install 'findsylls[embedding]' vg-hubert"
            )
        
        _VG_HUBERT_SEGMENTER = VGHubertSegmenter(
            model_ckpt=model_ckpt,
            mode=mode,
            layer=layer,
            device=device,
            cache=True  # Keep model loaded
        )
    
    # Extract features using VG-HuBERT
    features, spf = _VG_HUBERT_SEGMENTER.extract_features(audio, sr=sr)
    
    # Calculate time points
    num_frames = features.shape[0]
    times = np.arange(num_frames) * spf
    
    return features, times


def _extract_envelope_features(
    audio: np.ndarray,
    sr: int,
    method: str,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract envelope-based 1-D features for pooling/segmentation."""
    envelope, times = get_amplitude_envelope(audio, sr, method=method, **kwargs)
    envelope = np.asarray(envelope, dtype=np.float32)
    if envelope.ndim == 1:
        features = envelope[:, None]
    else:
        features = envelope
    return features, np.asarray(times, dtype=np.float32)


def _extract_mfcc_features(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    include_delta: bool = False,
    include_delta_delta: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MFCC features with optional delta and delta-delta coefficients.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCCs (default: 13)
        n_fft: FFT window size (default: 400)
        hop_length: Hop size in samples (default: 160 = 10ms at 16kHz)
        n_mels: Number of mel bands (default: 40)
        include_delta: If True, append first-order derivatives (deltas) to features
        include_delta_delta: If True, append second-order derivatives (delta-deltas)
                             Note: include_delta must be True for this to take effect
        
    Returns:
        (features, times): (num_frames, feature_dim) array and time points
                          feature_dim = n_mfcc * (1 + include_delta + include_delta_delta)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "MFCC feature extraction requires librosa. "
            "Install with: pip install librosa"
        )
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        **kwargs
    )
    
    # Build feature array: [mfcc, delta, delta_delta]
    feature_list = [mfccs]
    
    if include_delta:
        # Compute first-order derivatives (delta)
        delta = librosa.feature.delta(mfccs)
        feature_list.append(delta)
        
        if include_delta_delta:
            # Compute second-order derivatives (delta-delta)
            delta_delta = librosa.feature.delta(mfccs, order=2)
            feature_list.append(delta_delta)
    
    # Stack features: (feature_dim, num_frames) where feature_dim = n_mfcc * num_derivatives
    features = np.vstack(feature_list)
    
    # Transpose to (time, feature_dim)
    features = features.T
    
    # Calculate time points
    num_frames = features.shape[0]
    times = librosa.frames_to_time(
        np.arange(num_frames),
        sr=sr,
        hop_length=hop_length
    )
    
    return features, times


def _extract_melspec_features(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mel-spectrogram features.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands (default: 80)
        n_fft: FFT window size (default: 400)
        hop_length: Hop size in samples (default: 160)
        
    Returns:
        (features, times): (num_frames, n_mels) array and time points
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "Mel-spectrogram extraction requires librosa. "
            "Install with: pip install librosa"
        )
    
    # Extract mel-spectrogram
    melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        **kwargs
    )
    
    # Convert to log scale
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # Transpose to (time, feature_dim)
    features = melspec_db.T
    
    # Calculate time points
    num_frames = features.shape[0]
    times = librosa.frames_to_time(
        np.arange(num_frames),
        sr=sr,
        hop_length=hop_length
    )
    
    return features, times


def _extract_features_v3(
    audio: np.ndarray,
    sr: int,
    method: str = 'sylber',
    layer: Optional[int] = None,
    device: str = 'auto',
    return_times: bool = False,
    return_segments: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """Thin adapter to the shared features module.

    This override keeps the old function signature but routes extraction through
    `findsylls.features` to avoid duplicated feature implementations.
    """
    feature_type = method.lower().replace('-', '').replace('_', '')
    aliases = {
        'melspec': 'mel',
        'melspectrogram': 'mel',
        'vghubert': 'vghubert',
        'spectralbandsubtraction': 'sbs',
    }
    feature_type = aliases.get(feature_type, feature_type)

    envelope_methods = {'rms', 'hilbert', 'lowpass', 'sbs', 'theta'}
    if feature_type in envelope_methods:
        features, times = _extract_envelope_features(audio, sr, method=feature_type, **kwargs)
        if not return_times and not return_segments:
            return features
        if return_segments:
            return features, times, None
        return features, times

    extractor_override = kwargs.pop("feature_extractor", None)
    extractor_kwargs = dict(kwargs)
    if layer is not None:
        extractor_kwargs.setdefault('layer', layer)
        extractor_kwargs.setdefault('encoding_layer', layer)
    
    # Only pass device to neural feature extractors that accept it
    # (mfcc, mel, spectrogram are CPU-only and don't accept device)
    neural_extractors = {'sylber', 'vghubert', 'hubert'}
    if device is not None and feature_type in neural_extractors:
        extractor_kwargs.setdefault('device', device)

    extractor = extractor_override if extractor_override is not None else get_extractor(feature_type, **extractor_kwargs)
    features = extractor.extract(audio, sr)

    if not return_times and not return_segments:
        return features

    times = np.arange(features.shape[0], dtype=np.float64) / float(extractor.frame_rate)
    if return_segments:
        return features, times, None
    return features, times


# Route the public API through the shared-features adapter.
extract_features = _extract_features_v3
