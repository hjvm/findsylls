"""
Example: Ablation Study - Compare Features with Same Algorithm

This script demonstrates how to compare different feature types
using the same segmentation algorithm.
"""

from findsylls.audio import load_audio
from findsylls.segmentation.extractors import HuBERTExtractor, MFCCExtractor, MelSpectrogramExtractor
from findsylls.segmentation.custom_segmenters import MinCutSegmenter
import numpy as np


def compare_features_example(audio_path: str):
    """Compare HuBERT, MFCC, and Mel features with MinCut."""
    print("=" * 60)
    print("Ablation Study: Feature Comparison")
    print("=" * 60)
    
    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f}s, Sample rate: {sr} Hz")
    
    # Define feature extractors
    extractors = {
        'HuBERT (layer 9)': HuBERTExtractor(layer=9),
        'MFCC (13 coef)': MFCCExtractor(n_mfcc=13, include_deltas=False),
        'MFCC (13 + deltas)': MFCCExtractor(n_mfcc=13, include_deltas=True),
        'Mel Spectrogram': MelSpectrogramExtractor(n_mels=80),
    }
    
    # Run segmentation with each feature type
    results = {}
    print("\n" + "-" * 60)
    print("Running MinCut segmentation with different features:")
    print("-" * 60)
    
    for name, extractor in extractors.items():
        print(f"\n{name}:")
        segmenter = MinCutSegmenter(extractor, sec_per_syllable=0.22, use_optimized=True)
        segments = segmenter.segment(audio, sr)
        results[name] = segments
        
        # Compute statistics
        num_segments = len(segments)
        seg_rate = num_segments / duration
        avg_duration = np.mean([end - start for start, _, end in segments])
        
        print(f"  Segments: {num_segments}")
        print(f"  Segments/sec: {seg_rate:.2f}")
        print(f"  Avg segment duration: {avg_duration:.3f}s")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary Comparison:")
    print("=" * 60)
    print(f"{'Feature Type':<25} {'Segments':<12} {'Segs/sec':<12} {'Avg Duration'}")
    print("-" * 60)
    
    for name, segments in results.items():
        num_segments = len(segments)
        seg_rate = num_segments / duration
        avg_duration = np.mean([end - start for start, _, end in segments])
        print(f"{name:<25} {num_segments:<12} {seg_rate:<12.2f} {avg_duration:.3f}s")
    
    return results


def quick_comparison_example(audio_path: str):
    """Use convenience function for quick comparison."""
    print("\n" + "=" * 60)
    print("Quick Comparison Using Convenience Function")
    print("=" * 60)
    
    from findsylls.segmentation.custom_segmenters import compare_features
    
    audio, sr = load_audio(audio_path)
    
    # Compare features with one line
    results = compare_features(
        audio, sr,
        feature_types=['hubert', 'mfcc'],
        method='mincut',
        sec_per_syllable=0.22
    )
    
    print("\nResults:")
    for feature_type, segments in results.items():
        print(f"  {feature_type}: {len(segments)} segments")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Use test audio if available
    audio_path = "test_samples/SP20_117.wav"
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    
    try:
        # Detailed comparison
        results = compare_features_example(audio_path)
        
        # Quick comparison
        quick_results = quick_comparison_example(audio_path)
        
    except FileNotFoundError:
        print(f"\nError: Audio file not found: {audio_path}")
        print("Usage: python ablation_features.py <audio_file>")
