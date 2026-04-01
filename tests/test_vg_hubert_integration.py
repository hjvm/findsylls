#!/usr/bin/env python3
"""
Test script for VG-HuBERT integration with vg-hubert package.

This script verifies that:
1. VGHubertSegmenter can be imported
2. Segmentation works with vg-hubert package
3. Feature extraction works
4. Embedding pipeline works
"""

import sys
from pathlib import Path
import numpy as np

# Add findsylls to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_import():
    """Test that VGHubertSegmenter can be imported."""
    print("=" * 70)
    print("TEST 1: Import VGHubertSegmenter")
    print("=" * 70)
    
    try:
        from findsylls.segmentation.end2end import VGHubertSegmenter
        print("✓ VGHubertSegmenter imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import VGHubertSegmenter: {e}")
        return False


def test_segmenter_init():
    """Test that VGHubertSegmenter can be initialized."""
    print("\n" + "=" * 70)
    print("TEST 2: Initialize VGHubertSegmenter")
    print("=" * 70)
    
    try:
        from findsylls.segmentation.end2end import VGHubertSegmenter
        
        # Test with default parameters (should try to download)
        print("Creating segmenter with default parameters...")
        print("  model_ckpt='hjvm/VG-HuBERT' (will attempt HuggingFace download)")
        
        segmenter = VGHubertSegmenter(
            model_ckpt='hjvm/VG-HuBERT',
            mode='syllable',
            device='cpu'  # Use CPU for testing
        )
        
        print(f"✓ Segmenter initialized successfully")
        print(f"  Mode: {segmenter.mode}")
        print(f"  Layer: {segmenter.layer}")
        print(f"  Merge threshold: {segmenter.merge_threshold}")
        return True
        
    except ImportError as e:
        print(f"✗ vg-hubert package not installed: {e}")
        print("  Install with: pip install vg-hubert")
        return False
    except Exception as e:
        print(f"✗ Failed to initialize segmenter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_segmentation():
    """Test segmentation with dummy audio."""
    print("\n" + "=" * 70)
    print("TEST 3: Segmentation with dummy audio")
    print("=" * 70)
    
    try:
        from findsylls.segmentation.end2end import VGHubertSegmenter
        
        # Create dummy audio (1 second of noise at 16kHz)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        print("Creating segmenter...")
        segmenter = VGHubertSegmenter(
            model_ckpt='hjvm/VG-HuBERT',
            mode='syllable',
            device='cpu'
        )
        
        print("Running segmentation on 1 second of audio...")
        syllables = segmenter.segment(audio, sr=16000)
        
        print(f"✓ Segmentation successful")
        print(f"  Found {len(syllables)} segments")
        if len(syllables) > 0:
            start, peak, end = syllables[0]
            print(f"  First segment: {start:.3f}s - {end:.3f}s (peak at {peak:.3f}s)")
        
        return True
        
    except ImportError as e:
        print(f"✗ vg-hubert package not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\n" + "=" * 70)
    print("TEST 4: Feature extraction")
    print("=" * 70)
    
    try:
        from findsylls.embedding.extractors import extract_features
        
        # Create dummy audio
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        print("Extracting VG-HuBERT features...")
        features, times = extract_features(
            audio,
            sr=16000,
            method='vg_hubert',
            model_ckpt='hjvm/VG-HuBERT',
            device='cpu'
        )
        
        print(f"✓ Feature extraction successful")
        print(f"  Features shape: {features.shape}")
        print(f"  Time points: {len(times)}")
        print(f"  Feature dimension: {features.shape[1]}")
        
        return True
        
    except ImportError as e:
        print(f"✗ vg-hubert package not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("VG-HUBERT INTEGRATION TEST SUITE")
    print("=" * 70)
    print()
    print("This test suite verifies the vg-hubert PyPI package integration.")
    print()
    
    results = []
    
    # Test 1: Import
    results.append(("Import", test_import()))
    
    if not results[0][1]:
        print("\n⚠ Cannot proceed with remaining tests (import failed)")
        return
    
    # Test 2: Initialization
    results.append(("Initialization", test_segmenter_init()))
    
    if not results[1][1]:
        print("\n⚠ Cannot proceed with remaining tests (initialization failed)")
        print("  Make sure to install: pip install vg-hubert")
        return
    
    # Test 3: Segmentation
    results.append(("Segmentation", test_segmentation()))
    
    # Test 4: Feature extraction
    results.append(("Feature extraction", test_feature_extraction()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        print("  Make sure vg-hubert package is installed: pip install vg-hubert")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
