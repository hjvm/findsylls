"""
Mel-spectrogram feature extractor.

Log-scale mel-spectrogram features provide time-frequency representations
on a perceptually-motivated frequency scale.
"""

import numpy as np

from .base import FeatureExtractor


class MelSpectrogramExtractor(FeatureExtractor):
    """
    Log Mel-spectrogram feature extractor.
    
    Time-frequency representation on mel scale.
    
    Args:
        n_mels: Number of mel bands (default: 80)
        n_fft: FFT window size (default: 400)
        hop_length: Hop between frames (default: 320 → 50 Hz at 16kHz)
        fmin: Minimum frequency (default: 0 Hz)
        fmax: Maximum frequency (default: 8000 Hz)
    
    Example:
        >>> extractor = MelSpectrogramExtractor(n_mels=80)
        >>> features = extractor.extract(audio, sr=16000)
        >>> print(features.shape)  # (N, 80)
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 320,
        fmin: float = 0.0,
        fmax: float = 8000.0,
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract log mel-spectrogram features."""
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa required for MelSpectrogramExtractor. "
                "Install with: pip install librosa"
            )
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Shape: (n_mels, n_frames) -> transpose to (n_frames, n_mels)
        return log_mel.T
    
    @property
    def frame_rate(self) -> float:
        """Mel-spectrogram frame rate depends on hop_length."""
        return 16000.0 / self.hop_length


__all__ = ['MelSpectrogramExtractor']
