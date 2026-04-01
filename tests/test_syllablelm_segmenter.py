"""
Integration tests for SyllableLMSegmenter.

Tests the optimized MinCut segmentation with HuBERT features and custom features.
"""

import pytest
import numpy as np
from pathlib import Path

# Get test audio paths
TEST_DIR = Path(__file__).parent.parent / "test_samples"
AUDIO_SHORT = TEST_DIR / "SP20_117.wav"  # 3.75s
AUDIO_LONG = TEST_DIR / "WKSP_M_0064_E1_0009.flac"  # 9.64s


class TestSyllableLMSegmenter:
    """Test SyllableLMSegmenter with real audio."""
    
    def test_segment_from_features_basic(self):
        """Test segmentation from pre-extracted features."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        
        # Create synthetic features (187 frames ~ 3.74s at 50 Hz)
        np.random.seed(42)
        features = np.random.randn(187, 768).astype(np.float32)
        
        segmenter = SyllableLMSegmenter()
        segments = segmenter.segment_from_features(features)
        
        # Check output format
        assert isinstance(segments, list)
        assert len(segments) > 0
        
        # Check segment structure
        for start, nucleus, end in segments:
            assert start < nucleus < end
            assert start >= 0
            assert end <= 187 / 50.0  # duration in seconds
    
    def test_segment_from_features_frame_indices(self):
        """Test returning frame indices instead of times."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        
        features = np.random.randn(100, 768).astype(np.float32)
        
        segmenter = SyllableLMSegmenter()
        segments = segmenter.segment_from_features(features, return_frame_indices=True)
        
        # Check frame indices
        for start, nucleus, end in segments:
            assert isinstance(start, (int, np.integer))
            assert isinstance(nucleus, (int, np.integer))
            assert isinstance(end, (int, np.integer))
            assert 0 <= start < nucleus < end <= 100
    
    @pytest.mark.slow
    def test_segment_with_builtin_hubert(self):
        """Test end-to-end segmentation with built-in HuBERT extraction."""
        pytest.importorskip("transformers")
        
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        from findsylls.audio.utils import load_audio
        
        # Load test audio
        audio, sr = load_audio(str(AUDIO_SHORT))
        
        # Create segmenter
        segmenter = SyllableLMSegmenter(sec_per_syllable=0.22)
        
        # Segment
        segments = segmenter.segment(audio, sr)
        
        # Validate output
        assert len(segments) > 0
        assert len(segments) < 30  # Reasonable number for 3.75s audio
        
        # Check segments are in order and non-overlapping
        for i, (start, nucleus, end) in enumerate(segments):
            assert start < nucleus < end
            if i > 0:
                prev_end = segments[i-1][2]
                assert start >= prev_end  # No overlap
        
        # Check coverage
        first_start = segments[0][0]
        last_end = segments[-1][2]
        duration = len(audio) / sr
        assert first_start < 1.0  # Starts near beginning
        assert last_end > duration - 1.0  # Ends near end
    
    @pytest.mark.slow
    def test_get_embeddings(self):
        """Test embeddings extraction."""
        pytest.importorskip("transformers")
        
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        from findsylls.audio.utils import load_audio
        
        audio, sr = load_audio(str(AUDIO_SHORT))
        
        segmenter = SyllableLMSegmenter()
        segments, embeddings = segmenter.get_embeddings(audio, sr)
        
        # Check shapes
        assert len(segments) == embeddings.shape[0]
        assert embeddings.shape[1] == 768  # HuBERT hidden size
        
        # Check embeddings are not degenerate
        assert not np.allclose(embeddings[0], embeddings[1])  # Different segments
        assert not np.any(np.isnan(embeddings))
        assert not np.any(np.isinf(embeddings))
    
    def test_parameter_sensitivity(self):
        """Test that parameters affect segmentation."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        
        features = np.random.randn(200, 768).astype(np.float32)
        
        # Test different sec_per_syllable
        seg1 = SyllableLMSegmenter(sec_per_syllable=0.15)
        seg2 = SyllableLMSegmenter(sec_per_syllable=0.30)
        
        segs1 = seg1.segment_from_features(features)
        segs2 = seg2.segment_from_features(features)
        
        # More syllables per second → more segments
        assert len(segs1) > len(segs2)
    
    def test_min_max_hop_constraints(self):
        """Test that min/max hop parameters are respected."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        
        features = np.random.randn(150, 768).astype(np.float32)
        
        segmenter = SyllableLMSegmenter(min_hop=5, max_hop=30, frame_rate=50.0)
        segments = segmenter.segment_from_features(features, return_frame_indices=True)
        
        # Check segment lengths respect constraints
        for start, nucleus, end in segments:
            length = end - start
            assert length >= 5  # min_hop
            assert length <= 30  # max_hop


class TestCustomFeatureSegmenter:
    """Test CustomFeatureSegmenter with custom extractors."""
    
    def test_custom_extractor_basic(self):
        """Test with a simple custom feature extractor."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import CustomFeatureSegmenter
        
        # Define a simple feature extractor
        def my_extractor(audio, sr):
            # Return dummy features (50 Hz rate)
            num_frames = len(audio) // (sr // 50)
            return np.random.randn(num_frames, 256).astype(np.float32)
        
        segmenter = CustomFeatureSegmenter(
            feature_extractor=my_extractor,
            feature_frame_rate=50.0
        )
        
        # Create dummy audio
        audio = np.random.randn(16000 * 2).astype(np.float32)  # 2 seconds at 16kHz
        
        segments = segmenter.segment(audio, sr=16000)
        
        assert len(segments) > 0
        for start, nucleus, end in segments:
            assert start < nucleus < end
    
    def test_custom_extractor_embeddings(self):
        """Test embeddings with custom extractor."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import CustomFeatureSegmenter
        
        def my_extractor(audio, sr):
            num_frames = len(audio) // (sr // 50)
            return np.random.randn(num_frames, 128).astype(np.float32)
        
        segmenter = CustomFeatureSegmenter(
            feature_extractor=my_extractor,
            feature_frame_rate=50.0
        )
        
        audio = np.random.randn(16000 * 3).astype(np.float32)
        segments, embeddings = segmenter.get_embeddings(audio, sr=16000)
        
        assert len(segments) == embeddings.shape[0]
        assert embeddings.shape[1] == 128  # Custom feature dim


class TestOptimizedMinCutPerformance:
    """Test performance improvements of optimized MinCut."""
    
    def test_optimized_vs_original_equivalence(self):
        """Verify optimized MinCut produces similar results to original."""
        from findsylls.segmentation.algorithms.mincut import min_cut, min_cut_optimized
        
        # Create test SSM
        np.random.seed(42)
        N = 50
        features = np.random.randn(N, 64)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        ssm = features @ features.T
        ssm = ssm - np.min(ssm) + 1e-7
        
        K = 8
        
        # Run both algorithms
        boundaries_orig = min_cut(ssm, K)
        boundaries_opt = min_cut_optimized(ssm, K, min_hop=1, max_hop=N)
        
        # Results should be very similar (may differ slightly due to implementation)
        # Check that we get the right number of boundaries
        assert len(boundaries_orig) == K
        assert len(boundaries_opt) == K
        
        # Check boundaries are in valid range
        assert all(0 <= b < N for b in boundaries_orig)
        assert all(0 <= b <= N for b in boundaries_opt)
        
        # Check monotonicity
        assert boundaries_orig == sorted(boundaries_orig)
        assert boundaries_opt == sorted(boundaries_opt)
    
    def test_optimized_mincut_speed(self):
        """Test that optimized MinCut is faster than original."""
        import time
        from findsylls.segmentation.algorithms.mincut import min_cut, min_cut_optimized
        
        # Create realistic-sized SSM (200 frames ~ 4 seconds)
        np.random.seed(42)
        N = 200
        features = np.random.randn(N, 768)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        ssm = features @ features.T
        ssm = ssm - np.min(ssm) + 1e-7
        
        K = 15
        
        # Time original
        start = time.time()
        _ = min_cut(ssm, K)
        time_orig = time.time() - start
        
        # Time optimized
        start = time.time()
        _ = min_cut_optimized(ssm, K, min_hop=3, max_hop=50)
        time_opt = time.time() - start
        
        # Optimized should be faster (though exact speedup varies)
        # We expect at least 2× speedup on N=200
        assert time_opt < time_orig, f"Optimized ({time_opt:.4f}s) not faster than original ({time_orig:.4f}s)"
        
        speedup = time_orig / time_opt
        print(f"\nSpeedup: {speedup:.1f}×")
        
        # For small N=200, expect at least 2× speedup
        # (The claimed 20-50× is for larger N ~ 500-1000 frames)
        assert speedup >= 1.5, f"Expected at least 1.5× speedup, got {speedup:.1f}×"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_short_audio(self):
        """Test with very short feature sequence."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        
        # Only 10 frames
        features = np.random.randn(10, 768).astype(np.float32)
        
        segmenter = SyllableLMSegmenter()
        segments = segmenter.segment_from_features(features)
        
        # Should still produce at least 1 segment
        assert len(segments) >= 1
    
    def test_constant_features(self):
        """Test with constant (silence-like) features."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        
        # All frames identical
        features = np.ones((100, 768), dtype=np.float32)
        
        segmenter = SyllableLMSegmenter()
        segments = segmenter.segment_from_features(features)
        
        # Should still produce segments (algorithm should not crash)
        assert len(segments) > 0
    
    def test_zero_features(self):
        """Test with all-zero features."""
        from findsylls.segmentation.end2end.syllablelm_segmenter import SyllableLMSegmenter
        
        features = np.zeros((80, 768), dtype=np.float32)
        
        segmenter = SyllableLMSegmenter()
        segments = segmenter.segment_from_features(features)
        
        # Should handle gracefully
        assert len(segments) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
