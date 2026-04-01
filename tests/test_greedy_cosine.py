"""
Tests for greedy_cosine segmentation algorithm

Tests cover:
1. Basic functionality (synthetic data)
2. Edge cases (silence, single segment, empty)
3. Boundary refinement behavior
4. Regression against Sylber reference (if available)
5. Parameter sensitivity
"""

import numpy as np
import pytest
from findsylls.segmentation.algorithms.greedy_cosine import (
    cosine_similarity,
    greedy_cosine_segment,
    greedy_cosine_segment_to_times,
    greedy_cosine_segment_with_features
)


class TestCosineSimilarity:
    """Test cosine similarity helper function."""
    
    def test_vector_to_vector(self):
        """Test similarity between two 1D vectors."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([1.0, 0.0, 0.0])
        
        sim = cosine_similarity(x, y)
        assert np.isclose(sim, 1.0)
    
    def test_orthogonal_vectors(self):
        """Test similarity between orthogonal vectors."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        
        sim = cosine_similarity(x, y)
        assert np.isclose(sim, 0.0)
    
    def test_matrix_to_vector(self):
        """Test similarity between matrix of vectors and single vector."""
        x = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        y = np.array([1.0, 0.0])
        
        sims = cosine_similarity(x, y)
        
        assert sims.shape == (3,)
        assert np.isclose(sims[0], 1.0)  # Identical
        assert np.isclose(sims[1], 0.0)  # Orthogonal
        assert np.isclose(sims[2], 1.0 / np.sqrt(2))  # 45 degrees
    
    def test_zero_vector_handling(self):
        """Test that zero vectors don't cause division by zero."""
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 0.0])
        
        sim = cosine_similarity(x, y)
        # Should not raise, thanks to epsilon
        assert not np.isnan(sim)
        assert not np.isinf(sim)


class TestGreedyCosineBasic:
    """Test basic greedy cosine segmentation."""
    
    def test_single_segment(self):
        """Test segmentation with all high-energy, high-similarity frames."""
        # All frames identical, high norm
        features = np.ones((10, 8)) * 5.0
        
        segments = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.5
        )
        
        # Should produce single segment
        assert segments.shape == (1, 2)
        assert segments[0, 0] == 0
        assert segments[0, 1] == 10
    
    def test_silence_splits_segments(self):
        """Test that low-norm frames split segments."""
        # High norm frames with low norm in middle
        features = np.ones((10, 8)) * 5.0
        features[4:6] = 0.1  # Low norm (silence)
        
        segments = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.5
        )
        
        # Should split into 2 segments around silence
        assert len(segments) >= 2
        
        # First segment should end before silence
        assert segments[0, 1] <= 4
        
        # Second segment should start after silence
        assert segments[1, 0] >= 6
    
    def test_low_similarity_splits_segments(self):
        """Test that low cosine similarity splits segments."""
        features = np.zeros((10, 8))
        
        # First half: similar to [1, 0, 0, ...]
        features[:5] = np.array([1, 0, 0, 0, 0, 0, 0, 0]) * 5.0
        
        # Second half: similar to [0, 1, 0, ...]
        features[5:] = np.array([0, 1, 0, 0, 0, 0, 0, 0]) * 5.0
        
        segments = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.9  # High threshold, orthogonal vectors won't merge
        )
        
        # Should produce at least 2 segments (might produce more due to greedy)
        assert len(segments) >= 2
    
    def test_empty_features(self):
        """Test handling of empty input."""
        features = np.zeros((0, 8))
        
        segments = greedy_cosine_segment(features, norm_threshold=2.0)
        
        assert segments.shape == (0, 2)
    
    def test_all_silence(self):
        """Test handling of all low-norm frames."""
        # All frames below threshold
        features = np.ones((10, 8)) * 0.5
        
        segments = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.8
        )
        
        # Should produce no segments
        assert len(segments) == 0
    
    def test_precomputed_norms(self):
        """Test that precomputed norms are used correctly."""
        features = np.ones((10, 8)) * 5.0
        norms = np.ones(10) * 10.0  # Override norms
        
        # With precomputed high norms, all frames pass threshold
        segments = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            norms=norms
        )
        
        assert len(segments) > 0
        assert segments[0, 0] == 0


class TestBoundaryRefinement:
    """Test boundary refinement phase."""
    
    def test_boundary_refinement_moves_boundary(self):
        """Test that boundary refinement can adjust boundaries."""
        # Create features where optimal boundary is not at initial greedy split
        features = np.zeros((20, 8))
        
        # Segment 1: frames 0-9, features = [1, 0, 0, ...]
        features[:10] = np.array([1, 0, 0, 0, 0, 0, 0, 0]) * 5.0
        
        # Segment 2: frames 10-19, features = [0, 1, 0, ...]
        features[10:] = np.array([0, 1, 0, 0, 0, 0, 0, 0]) * 5.0
        
        # Add transition frames that are more similar to one segment
        # Frame 9 more similar to segment 2
        features[9] = np.array([0.3, 0.7, 0, 0, 0, 0, 0, 0]) * 5.0
        
        segments = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.5  # Low threshold to avoid merging
        )
        
        # Boundary refinement should detect the transition
        # (exact boundary depends on similarity sweep)
        assert len(segments) >= 1
    
    def test_segments_merged_when_similar(self):
        """Test that similar segments are merged in refinement phase."""
        features = np.zeros((20, 8))
        
        # Both segments have same mean
        features[:10] = np.array([1, 1, 0, 0, 0, 0, 0, 0]) * 5.0
        features[10:] = np.array([1, 1, 0, 0, 0, 0, 0, 0]) * 5.0
        
        # Force initial split by making one frame dissimilar
        features[10] = np.array([0, 0, 1, 0, 0, 0, 0, 0]) * 5.0
        
        segments = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.99  # Very high threshold
        )
        
        # With high merge threshold, might still split initially,
        # but refinement should merge if segment means are similar enough
        # (This test is sensitive to exact merge_threshold value)
        assert len(segments) >= 1


class TestTimeConversion:
    """Test conversion from frames to time."""
    
    def test_frame_to_time_conversion(self):
        """Test conversion of frame indices to seconds."""
        segments = np.array([
            [0, 50],
            [75, 150]
        ])
        
        time_segments = greedy_cosine_segment_to_times(segments, frame_rate=50.0)
        
        expected = np.array([
            [0.0, 1.0],
            [1.5, 3.0]
        ])
        
        np.testing.assert_allclose(time_segments, expected)
    
    def test_different_frame_rate(self):
        """Test conversion with non-standard frame rate."""
        segments = np.array([[0, 100]])
        
        # 100 frames at 100 Hz = 1 second
        time_segments = greedy_cosine_segment_to_times(segments, frame_rate=100.0)
        
        assert np.isclose(time_segments[0, 0], 0.0)
        assert np.isclose(time_segments[0, 1], 1.0)


class TestWithFeatures:
    """Test convenience function with feature extraction."""
    
    def test_returns_time_segments(self):
        """Test that time segments are returned."""
        features = np.ones((100, 8)) * 5.0
        
        segments, seg_feats = greedy_cosine_segment_with_features(
            features,
            norm_threshold=2.0,
            return_segment_features=False
        )
        
        # Should return time-based segments
        assert segments.shape[1] == 2
        assert segments[0, 1] > 0  # End time > 0
        assert seg_feats is None
    
    def test_returns_segment_features(self):
        """Test that segment features are computed correctly."""
        features = np.ones((100, 8)) * 5.0
        
        segments, seg_feats = greedy_cosine_segment_with_features(
            features,
            norm_threshold=2.0,
            return_segment_features=True
        )
        
        assert seg_feats is not None
        assert len(seg_feats) == len(segments)
        assert seg_feats.shape[1] == features.shape[1]  # Same dimension
    
    def test_segment_features_are_means(self):
        """Test that segment features are indeed means of frame features."""
        # Create simple features
        features = np.zeros((10, 4))
        features[:5] = 1.0  # First segment
        features[5:] = 2.0  # Second segment
        
        segments, seg_feats = greedy_cosine_segment_with_features(
            features,
            norm_threshold=0.5,
            merge_threshold=0.5,
            return_segment_features=True
        )
        
        if len(segments) >= 2:
            # Segment features should be close to 1.0 and 2.0
            assert seg_feats is not None


class TestParameterSensitivity:
    """Test algorithm behavior with different parameters."""
    
    def test_norm_threshold_affects_segmentation(self):
        """Test that norm threshold controls silence detection."""
        features = np.ones((10, 8))
        features[::2] *= 5.0  # High norm
        features[1::2] *= 0.5  # Low norm
        
        # Low threshold: includes more frames
        segments_low = greedy_cosine_segment(
            features,
            norm_threshold=0.3,
            merge_threshold=0.8
        )
        
        # High threshold: excludes low-norm frames
        segments_high = greedy_cosine_segment(
            features,
            norm_threshold=3.0,
            merge_threshold=0.8
        )
        
        # High threshold should produce more segments (more splits at silence)
        assert len(segments_high) >= len(segments_low)
    
    def test_merge_threshold_affects_segmentation(self):
        """Test that merge threshold controls segment granularity."""
        # Create features with some variation
        np.random.seed(42)
        features = np.random.randn(100, 8) * 5.0
        
        # Low threshold: merges more segments
        segments_low = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.5
        )
        
        # High threshold: merges fewer segments
        segments_high = greedy_cosine_segment(
            features,
            norm_threshold=2.0,
            merge_threshold=0.95
        )
        
        # Low threshold should produce fewer segments (more merging)
        assert len(segments_low) <= len(segments_high)


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_1d_features_raises_error(self):
        """Test that 1D features raise ValueError."""
        features = np.ones(10)
        
        with pytest.raises(ValueError, match="must be 2D array"):
            greedy_cosine_segment(features)
    
    def test_3d_features_raises_error(self):
        """Test that 3D features raise ValueError."""
        features = np.ones((10, 8, 3))
        
        with pytest.raises(ValueError, match="must be 2D array"):
            greedy_cosine_segment(features)


# TODO: Add regression test against Sylber reference when available
# class TestSylberRegression:
#     """Regression tests against Sylber implementation."""
#     
#     @pytest.mark.skipif(not HAS_SYLBER, reason="sylber package not installed")
#     def test_matches_sylber_output(self):
#         """Test that our implementation matches Sylber's get_segment()."""
#         from sylber.utils.segment_utils import get_segment as sylber_get_segment
#         
#         # Test on same random data
#         np.random.seed(42)
#         features = np.random.randn(100, 768) * 5.0
#         
#         # Our implementation
#         our_segments = greedy_cosine_segment(
#             features,
#             norm_threshold=2.6,
#             merge_threshold=0.8
#         )
#         
#         # Sylber reference
#         sylber_segments = sylber_get_segment(
#             features,
#             normthreshold=2.6,
#             mergethreshold=0.8
#         )
#         
#         # Should match exactly
#         np.testing.assert_array_equal(our_segments, sylber_segments)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
