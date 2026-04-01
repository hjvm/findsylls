"""
Example: Method Comparison - Compare Algorithms with Same Features

This script demonstrates how to compare different segmentation algorithms
using the same features.
"""

from findsylls.audio import load_audio
from findsylls.segmentation.extractors import HuBERTExtractor
from findsylls.segmentation.custom_segmenters import MinCutSegmenter, GreedyCosineSegmenter
import numpy as np


def compare_methods_example(audio_path: str):
    """Compare MinCut and Greedy Cosine with same features."""
    print("=" * 60)
    print("Method Comparison: Same Features, Different Algorithms")
    print("=" * 60)
    
    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f}s")
    
    # Use HuBERT features for all methods
    extractor = HuBERTExtractor(layer=9)
    
    # Define segmenters
    segmenters = {
        'MinCut (original)': MinCutSegmenter(
            extractor,
            sec_per_syllable=0.22,
            use_optimized=False
        ),
        'MinCut (optimized)': MinCutSegmenter(
            extractor,
            sec_per_syllable=0.22,
            use_optimized=True
        ),
        'Greedy Cosine (0.85)': GreedyCosineSegmenter(
            extractor,
            merge_threshold=0.85
        ),
        'Greedy Cosine (0.90)': GreedyCosineSegmenter(
            extractor,
            merge_threshold=0.90
        ),
    }
    
    # Run segmentation
    results = {}
    print("\n" + "-" * 60)
    print("Running different algorithms on HuBERT features:")
    print("-" * 60)
    
    import time
    for name, segmenter in segmenters.items():
        print(f"\n{name}:")
        
        start = time.time()
        segments = segmenter.segment(audio, sr)
        elapsed = time.time() - start
        
        results[name] = segments
        
        # Statistics
        num_segments = len(segments)
        seg_rate = num_segments / duration
        avg_duration = np.mean([end - start for start, _, end in segments])
        
        print(f"  Segments: {num_segments}")
        print(f"  Segments/sec: {seg_rate:.2f}")
        print(f"  Avg duration: {avg_duration:.3f}s")
        print(f"  Time: {elapsed*1000:.1f}ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"{'Method':<25} {'Segments':<12} {'Segs/sec':<12} {'Time (ms)'}")
    print("-" * 60)
    
    for name, segments in results.items():
        num_segments = len(segments)
        seg_rate = num_segments / duration
        print(f"{name:<25} {num_segments:<12} {seg_rate:<12.2f} -")
    
    return results


def parameter_sensitivity_example(audio_path: str):
    """Show parameter sensitivity for greedy cosine."""
    print("\n" + "=" * 60)
    print("Parameter Sensitivity: Greedy Cosine Merge Threshold")
    print("=" * 60)
    
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    extractor = HuBERTExtractor()
    
    # Test different merge thresholds
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    print("\nMerge Threshold vs Number of Segments:")
    print("-" * 40)
    
    for threshold in thresholds:
        segmenter = GreedyCosineSegmenter(extractor, merge_threshold=threshold)
        segments = segmenter.segment(audio, sr)
        print(f"  {threshold:.2f}: {len(segments):3d} segments ({len(segments)/duration:.2f} segs/sec)")
    
    print("\nNote: Higher threshold → less merging → more segments")


def quick_comparison_example(audio_path: str):
    """Use convenience function."""
    print("\n" + "=" * 60)
    print("Quick Method Comparison")
    print("=" * 60)
    
    from findsylls.segmentation.custom_segmenters import compare_methods
    
    audio, sr = load_audio(audio_path)
    
    results = compare_methods(
        audio, sr,
        methods=['mincut', 'greedy'],
        feature_type='hubert'
    )
    
    print("\nResults:")
    for method, segments in results.items():
        print(f"  {method}: {len(segments)} segments")


if __name__ == "__main__":
    import sys
    
    audio_path = "test_samples/SP20_117.wav"
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    
    try:
        # Detailed comparison
        compare_methods_example(audio_path)
        
        # Parameter sensitivity
        parameter_sensitivity_example(audio_path)
        
        # Quick comparison
        quick_comparison_example(audio_path)
        
    except FileNotFoundError:
        print(f"\nError: Audio file not found: {audio_path}")
        print("Usage: python ablation_methods.py <audio_file>")
