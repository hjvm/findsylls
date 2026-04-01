"""
Tests for feature extractors (extractors.py).
"""

import pytest
import numpy as np
from pathlib import Path

from findsylls.segmentation.extractors import (
    FeatureExtractor,
    HuBERTExtractor,
    MFCCExtractor,
    MelSpectrogramExtractor,
    CustomCallableExtractor,
    get_extractor,
)


class TestMFCCExtractor:
    """Test MFCC feature extractor."""
    
    def test_basic_extraction(self):
        """Test basic MFCC extraction."""
        extractor = MFCCExtractor(n_mfcc=13)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        assert features.shape[1] == 13
        assert features.shape[0] > 0
    
    def test_with_deltas(self):
        """Test MFCC with delta features."""
        extractor = MFCCExtractor(n_mfcc=13, include_deltas=True)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        # 13 MFCCs + 13 deltas + 13 delta-deltas = 39
        assert features.shape[1] == 39
    
    def test_frame_rate(self):
        """Test frame rate property."""
        extractor = MFCCExtractor()
        
        # Should be approximately 50 Hz (20ms hop)
        assert extractor.frame_rate == pytest.approx(50.0, rel=0.1)
    
    def test_resampling(self):
        """Test automatic resampling."""
        extractor = MFCCExtractor()
        
        # Audio at different sample rates
        audio_8k = np.random.randn(8000 * 2).astype(np.float32)
        audio_22k = np.random.randn(22050 * 2).astype(np.float32)
        
        features_8k = extractor.extract(audio_8k, sr=8000)
        features_22k = extractor.extract(audio_22k, sr=22050)
        
        # Both should work
        assert features_8k.shape[1] == 13
        assert features_22k.shape[1] == 13


class TestMelSpectrogramExtractor:
    """Test mel-spectrogram feature extractor."""
    
    def test_basic_extraction(self):
        """Test basic mel-spectrogram extraction."""
        extractor = MelSpectrogramExtractor(n_mels=80)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        assert features.shape[1] == 80
        assert features.shape[0] > 0
    
    def test_custom_parameters(self):
        """Test with custom parameters."""
        extractor = MelSpectrogramExtractor(
            n_mels=128,
            n_fft=1024,
            hop_length=160
        )
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        assert features.shape[1] == 128
    
    def test_frame_rate(self):
        """Test frame rate property."""
        extractor = MelSpectrogramExtractor()
        
        # Should be approximately 50 Hz
        assert extractor.frame_rate == pytest.approx(50.0, rel=0.1)


class TestHuBERTExtractor:
    """Test HuBERT feature extractor."""
    
    @pytest.mark.skipif(True, reason="Skip HuBERT tests (requires model download)")
    def test_basic_extraction(self):
        """Test basic HuBERT extraction."""
        extractor = HuBERTExtractor()
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        # HuBERT produces 768-dim features
        assert features.shape[1] == 768
        assert features.shape[0] > 0
    
    @pytest.mark.skipif(True, reason="Skip HuBERT tests")
    def test_lazy_loading(self):
        """Test that model is loaded lazily."""
        extractor = HuBERTExtractor()
        
        # Model should not be loaded yet
        assert extractor._model is None
        
        # Extract features
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        # Now model should be loaded
        assert extractor._model is not None
    
    def test_frame_rate(self):
        """Test frame rate property."""
        extractor = HuBERTExtractor()
        
        # HuBERT produces 50 Hz features
        assert extractor.frame_rate == 50.0


class TestCustomCallableExtractor:
    """Test custom callable extractor wrapper."""
    
    def test_basic_callable(self):
        """Test wrapping a simple callable."""
        def simple_extractor(audio, sr):
            n_frames = len(audio) // (sr // 50)
            return np.random.randn(n_frames, 16).astype(np.float32)
        
        extractor = CustomCallableExtractor(simple_extractor, frame_rate=50.0)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        assert features.shape[1] == 16
        assert features.shape[0] > 0
    
    def test_frame_rate_property(self):
        """Test frame rate property."""
        def dummy(audio, sr):
            return np.zeros((10, 5))
        
        extractor = CustomCallableExtractor(dummy, frame_rate=100.0)
        
        assert extractor.frame_rate == 100.0
    
    def test_lambda_function(self):
        """Test with lambda function."""
        extractor = CustomCallableExtractor(
            lambda audio, sr: np.random.randn(len(audio) // 320, 32).astype(np.float32),
            frame_rate=50.0
        )
        
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        assert features.shape[1] == 32


class TestGetExtractor:
    """Test get_extractor factory function."""
    
    def test_get_mfcc(self):
        """Test getting MFCC extractor."""
        extractor = get_extractor('mfcc', n_mfcc=20)
        
        assert isinstance(extractor, MFCCExtractor)
        
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        assert features.shape[1] == 20
    
    def test_get_mel(self):
        """Test getting mel-spectrogram extractor."""
        extractor = get_extractor('mel', n_mels=64)
        
        assert isinstance(extractor, MelSpectrogramExtractor)
        
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        assert features.shape[1] == 64
    
    @pytest.mark.skipif(True, reason="Skip HuBERT tests")
    def test_get_hubert(self):
        """Test getting HuBERT extractor."""
        extractor = get_extractor('hubert', layer=6)
        
        assert isinstance(extractor, HuBERTExtractor)
        assert extractor.frame_rate == 50.0
    
    def test_invalid_type(self):
        """Test error handling for invalid type."""
        with pytest.raises(ValueError, match="Unknown feature_type"):
            get_extractor('invalid')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_short_audio_mfcc(self):
        """Test MFCC with very short audio."""
        extractor = MFCCExtractor()
        
        # 0.1 seconds
        audio = np.random.randn(1600).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        # Should still produce features
        assert features.shape[0] > 0
        assert features.shape[1] == 13
    
    def test_silent_audio(self):
        """Test with silent audio."""
        extractor = MFCCExtractor()
        
        audio = np.zeros(16000, dtype=np.float32)
        features = extractor.extract(audio, sr=16000)
        
        # Should handle gracefully
        assert features.shape[0] > 0
    
    def test_very_loud_audio(self):
        """Test with loud audio."""
        extractor = MelSpectrogramExtractor()
        
        audio = (np.random.randn(16000) * 100).astype(np.float32)
        features = extractor.extract(audio, sr=16000)
        
        # Should not crash
        assert features.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
