"""
MFCC (Mel-Frequency Cepstral Coefficients) feature extractor.

Classical speech features based on mel-scale frequency analysis.
Commonly used baseline for speech processing tasks.
"""

import numpy as np

from .base import FeatureExtractor


class MFCCExtractor(FeatureExtractor):
    """
    MFCC (Mel-Frequency Cepstral Coefficients) feature extractor.
    
    Classical speech features based on mel-scale frequency analysis.
    
    Args:
        n_mfcc: Number of MFCC coefficients (default: 13)
        n_fft: FFT window size (default: 400, ~25ms at 16kHz)
        hop_length: Hop between frames (default: 320, ~20ms at 16kHz → 50 Hz)
        n_mels: Number of mel bands (default: 40)
        include_deltas: Include delta and delta-delta features (default: False)
    
    Example:
        >>> extractor = MFCCExtractor(n_mfcc=13, include_deltas=True)
        >>> features = extractor.extract(audio, sr=16000)
        >>> print(features.shape)  # (N, 39) if include_deltas=True (13 + 13 + 13)
    """
    
    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 400,
        hop_length: int = 320,
        n_mels: int = 40,
        include_deltas: bool = False,
    ):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.include_deltas = include_deltas
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features."""
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa required for MFCCExtractor. "
                "Install with: pip install librosa"
            )
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Shape: (n_mfcc, n_frames) -> transpose to (n_frames, n_mfcc)
        mfccs = mfccs.T
        
        if self.include_deltas:
            # Compute delta and delta-delta
            delta = librosa.feature.delta(mfccs.T).T
            delta2 = librosa.feature.delta(mfccs.T, order=2).T
            
            # Concatenate
            mfccs = np.concatenate([mfccs, delta, delta2], axis=1)
        
        return mfccs
    
    @property
    def frame_rate(self) -> float:
        """MFCC frame rate depends on hop_length."""
        # Assuming 16kHz default sr for rate calculation
        # hop_length=320 at 16kHz → 50 Hz
        return 16000.0 / self.hop_length


__all__ = ['MFCCExtractor']
