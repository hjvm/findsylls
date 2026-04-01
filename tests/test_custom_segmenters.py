"""
Tests for convenience wrappers (custom_segmenters.py).
"""

import pytest
import numpy as np
from pathlib import Path

from findsylls.segmentation.custom_segmenters import (
    MinCutSegmenter,
    GreedyCosineSegmenter,
    quick_segment,
    compare_methods,
    compare_features,
)
from findsylls.segmentation.extractors import (
    MFCCExtractor,
    CustomCallableExtractor,
)


class TestMinCutSegmenter:
    """Test MinCutSegmenter wrapper."""
    
    def test_with_mfcc_extractor(self):
        """Test MinCut with MFCC features."""
        extractor = MFCCExtractor(n_mfcc=13)
        segmenter = MinCutSegmenter(extractor, sec_per_syllable=0.2)
        
        # Create dummy audio (2 seconds at 16kHz)
        audio = np.random.randn(16000 * 2).astype(np.float32)
        segments = segmenter.segment(audio, sr=16000)
        
        assert len(segments) > 0
        for start, nucleus, end in segments:
            assert start < nucleus < end
    
    def test_optimized_vs_original(self):
        """Test that optimized flag works."""
        extractor = MFCCExtractor()
        
        seg_opt = MinCutSegmenter(extractor, use_optimized=True)
        seg_orig = MinCutSegmenter(extractor, use_optimized=False)
        
        audio = np.random.randn(16000).astype(np.float32)
        
        segs_opt = seg_opt.segment(audio, sr=16000)
        segs_orig = seg_orig.segment(audio, sr=16000)
        
        # Both should produce segments
        assert len(segs_opt) > 0
        assert len(segs_orig) > 0
    
    def test_with_callable_extractor(self):
        """Test with simple callable."""
        def simple_extractor(audio, sr):
            # Return dummy features
            n_frames = len(audio) // (sr // 50)
            return np.random.randn(n_frames, 32).astype(np.float32)
        
        wrapped = CustomCallableExtractor(simple_extractor, frame_rate=50.0)
        segmenter = MinCutSegmenter(wrapped)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        segments = segmenter.segment(audio, sr=16000)
        
        assert len(segments) > 0
    
    def test_get_embeddings(self):
        """Test embeddings extraction."""
        extractor = MFCCExtractor(n_mfcc=13)
        segmenter = MinCutSegmenter(extractor)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        segments, embeddings = segmenter.get_embeddings(audio, sr=16000)
        
        assert len(segments) == embeddings.shape[0]
        assert embeddings.shape[1] == 13  # MFCC dimension


class TestGreedyCosineSegmenter:
    """Test GreedyCosineSegmenter wrapper."""
    
    def test_basic_segmentation(self):
        """Test basic greedy cosine segmentation."""
        extractor = MFCCExtractor()
        segmenter = GreedyCosineSegmenter(extractor, merge_threshold=0.85)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        segments = segmenter.segment(audio, sr=16000)
        
        assert len(segments) > 0
        for start, nucleus, end in segments:
            assert start < nucleus < end
    
    def test_parameter_sensitivity(self):
        """Test that merge_threshold affects output."""
        extractor = MFCCExtractor()
        
        seg_low = GreedyCosineSegmenter(extractor, merge_threshold=0.7)
        seg_high = GreedyCosineSegmenter(extractor, merge_threshold=0.95)
        
        # Use fixed seed for reproducibility
        np.random.seed(42)
        audio = np.random.randn(16000 * 3).astype(np.float32)
        
        segs_low = seg_low.segment(audio, sr=16000)
        segs_high = seg_high.segment(audio, sr=16000)
        
        # Lower threshold → more merging → fewer segments
        # Higher threshold → less merging → more segments
        # (This relationship should generally hold)
        assert len(segs_low) != len(segs_high)
    
    def test_get_embeddings(self):
        """Test embeddings extraction."""
        extractor = MFCCExtractor(n_mfcc=20)
        segmenter = GreedyCosineSegmenter(extractor)
        
        audio = np.random.randn(16000 * 2).astype(np.float32)
        segments, embeddings = segmenter.get_embeddings(audio, sr=16000)
        
        assert len(segments) == embeddings.shape[0]
        assert embeddings.shape[1] == 20


class TestConvenienceFunctions:
    """Test quick_segment and comparison functions."""
    
    def test_quick_segment_mincut(self):
        """Test quick_segment with mincut."""
        audio = np.random.randn(16000 * 2).astype(np.float32)
        
        segments = quick_segment(audio, sr=16000, method='mincut', feature_type='mfcc')
        
        assert len(segments) > 0
        assert isinstance(segments, list)
    
    def test_quick_segment_greedy(self):
        """Test quick_segment with greedy."""
        audio = np.random.randn(16000 * 2).astype(np.float32)
        
        segments = quick_segment(audio, sr=16000, method='greedy', feature_type='mfcc')
        
        assert len(segments) > 0
    
    def test_quick_segment_with_kwargs(self):
        """Test passing kwargs to quick_segment."""
        audio = np.random.randn(16000 * 2).astype(np.float32)
        
        segments = quick_segment(
            audio, sr=16000,
            method='mincut',
            feature_type='mfcc',
            sec_per_syllable=0.3  # Custom parameter
        )
        
        assert len(segments) > 0
    
    def test_quick_segment_invalid_method(self):
        """Test error handling for invalid method."""
        audio = np.random.randn(16000).astype(np.float32)
        
        with pytest.raises(ValueError, match="Unknown method"):
            quick_segment(audio, sr=16000, method='invalid')
    
    def test_compare_methods(self):
        """Test compare_methods function."""
        audio = np.random.randn(16000 * 2).astype(np.float32)
        
        results = compare_methods(
            audio, sr=16000,
            methods=['mincut', 'greedy'],
            feature_type='mfcc'
        )
        
        assert 'mincut' in results
        assert 'greedy' in results
        assert len(results['mincut']) > 0
        assert len(results['greedy']) > 0
    
    def test_compare_features(self):
        """Test compare_features function."""
        audio = np.random.randn(16000 * 2).astype(np.float32)
        
        results = compare_features(
            audio, sr=16000,
            feature_types=['mfcc', 'mel'],
            method='mincut'
        )
        
        assert 'mfcc' in results
        assert 'mel' in results
        assert len(results['mfcc']) > 0
        assert len(results['mel']) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_short_audio(self):
        """Test with very short audio."""
        extractor = MFCCExtractor()
        segmenter = MinCutSegmenter(extractor)
        
        # 0.5 seconds
        audio = np.random.randn(8000).astype(np.float32)
        segments = segmenter.segment(audio, sr=16000)
        
        # Should still produce at least 1 segment
        assert len(segments) >= 1
    
    def test_silence_audio(self):
        """Test with silent audio."""
        extractor = MFCCExtractor()
        segmenter = MinCutSegmenter(extractor)
        
        # All zeros
        audio = np.zeros(16000 * 2, dtype=np.float32)
        segments = segmenter.segment(audio, sr=16000)
        
        # Should handle gracefully
        assert len(segments) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
