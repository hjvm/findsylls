"""
Test Phase 1 embedding pipeline implementation.

Tests:
1. MFCC + mean pooling (no neural dependencies)
2. Sylber + mean pooling (requires transformers)
3. Sylber + ONC pooling (tests 3× dimensions)
4. Edge cases (empty syllables, single syllable)
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np


def test_mfcc_mean():
    """Test classical MFCC with mean pooling."""
    print("\n" + "="*70)
    print("TEST 1: MFCC + Mean Pooling")
    print("="*70)
    
    from findsylls.embedding import embed_audio
    
    audio_path = 'test_samples/SP20_117.wav'
    
    embeddings, metadata = embed_audio(
        audio_path,
        segmentation='peaks_and_valleys',
        embedder='mfcc',
        pooling='mean',
        embedder_kwargs={'n_mfcc': 13}
    )
    
    print(f"✓ Audio loaded: {metadata['audio_path']}")
    print(f"✓ Duration: {metadata['duration']:.2f}s")
    print(f"✓ Segmentation: {metadata['segmentation_method']}")
    print(f"✓ Embedder: {metadata['embedder']}")
    print(f"✓ Pooling: {metadata['pooling']}")
    print(f"✓ Syllables found: {metadata['num_syllables']}")
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Expected: ({metadata['num_syllables']}, 13)")
    
    assert embeddings.shape == (metadata['num_syllables'], 13), \
        f"Expected shape ({metadata['num_syllables']}, 13), got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
    assert not np.isinf(embeddings).any(), "Embeddings contain Inf"
    
    print("✅ TEST 1 PASSED")
    return True


def test_sylber_mean():
    """Test Sylber with mean pooling."""
    print("\n" + "="*70)
    print("TEST 2: Sylber + Mean Pooling")
    print("="*70)
    
    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        print("⚠️  Skipping: PyTorch/transformers not available")
        return True
    
    from findsylls.embedding import embed_audio
    
    audio_path = 'test_samples/SP20_117.wav'
    
    embeddings, metadata = embed_audio(
        audio_path,
        segmentation='sylber',
        embedder='sylber',
        pooling='mean',
        layer=8
    )
    
    print(f"✓ Audio loaded: {metadata['audio_path']}")
    print(f"✓ Duration: {metadata['duration']:.2f}s")
    print(f"✓ Segmentation: {metadata['segmentation_method']}")
    print(f"✓ Embedder: {metadata['embedder']} (layer {metadata['layer']})")
    print(f"✓ Pooling: {metadata['pooling']}")
    print(f"✓ Syllables found: {metadata['num_syllables']}")
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Expected: ({metadata['num_syllables']}, 768)")
    
    assert embeddings.shape == (metadata['num_syllables'], 768), \
        f"Expected shape ({metadata['num_syllables']}, 768), got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
    assert not np.isinf(embeddings).any(), "Embeddings contain Inf"
    
    print("✅ TEST 2 PASSED")
    return True


def test_sylber_onc():
    """Test Sylber with ONC pooling (3× dimensions)."""
    print("\n" + "="*70)
    print("TEST 3: Sylber + ONC Pooling")
    print("="*70)
    
    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        print("⚠️  Skipping: PyTorch/transformers not available")
        return True
    
    from findsylls.embedding import embed_audio
    
    audio_path = 'test_samples/SP20_117.wav'
    
    embeddings, metadata = embed_audio(
        audio_path,
        segmentation='sylber',
        embedder='sylber',
        pooling='onc',
        layer=8
    )
    
    print(f"✓ Audio loaded: {metadata['audio_path']}")
    print(f"✓ Duration: {metadata['duration']:.2f}s")
    print(f"✓ Segmentation: {metadata['segmentation_method']}")
    print(f"✓ Embedder: {metadata['embedder']} (layer {metadata['layer']})")
    print(f"✓ Pooling: {metadata['pooling']}")
    print(f"✓ Syllables found: {metadata['num_syllables']}")
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Expected: ({metadata['num_syllables']}, 2304)  [768 × 3 for ONC]")
    
    assert embeddings.shape == (metadata['num_syllables'], 2304), \
        f"Expected shape ({metadata['num_syllables']}, 2304), got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
    assert not np.isinf(embeddings).any(), "Embeddings contain Inf"
    
    # Verify structure: can extract onset, nucleus, coda
    onset = embeddings[:, :768]
    nucleus = embeddings[:, 768:1536]
    coda = embeddings[:, 1536:]
    
    print(f"✓ Onset shape: {onset.shape}")
    print(f"✓ Nucleus shape: {nucleus.shape}")
    print(f"✓ Coda shape: {coda.shape}")
    
    print("✅ TEST 3 PASSED")
    return True


def test_step_by_step():
    """Test step-by-step API (Level 3)."""
    print("\n" + "="*70)
    print("TEST 4: Step-by-Step API")
    print("="*70)
    
    from findsylls.audio import load_audio
    from findsylls.pipeline.pipeline import segment_audio
    from findsylls.embedding import extract_features, pool_syllables
    
    audio_path = 'test_samples/SP20_117.wav'
    
    # Step 1: Load audio
    audio, sr = load_audio(audio_path, samplerate=16000)
    print(f"✓ Audio loaded: {len(audio)} samples at {sr} Hz")
    
    # Step 2: Segment
    syllables, _, _ = segment_audio(audio_path, method='peaks_and_valleys', samplerate=sr)
    print(f"✓ Segmented into {len(syllables)} syllables")
    
    # Step 3: Extract features
    features = extract_features(audio, sr, method='mfcc', n_mfcc=20)
    print(f"✓ Features extracted: {features.shape}")
    
    # Step 4: Pool
    embeddings = pool_syllables(
        features,
        syllables,
        sr=sr,
        method='mean',
        hop_length=160
    )
    print(f"✓ Embeddings pooled: {embeddings.shape}")
    
    assert embeddings.shape == (len(syllables), 20), \
        f"Expected shape ({len(syllables)}, 20), got {embeddings.shape}"
    
    print("✅ TEST 4 PASSED")
    return True


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70)
    
    from findsylls.embedding import pool_syllables
    import numpy as np
    
    # Create dummy features
    features = np.random.randn(100, 13)
    
    # Test 1: Empty syllables
    print("Testing empty syllables list...")
    embeddings = pool_syllables(features, [], sr=16000, method='mean', hop_length=160)
    assert embeddings.shape == (0, 13), f"Expected (0, 13), got {embeddings.shape}"
    print("✓ Empty syllables handled correctly")
    
    # Test 2: Single syllable
    print("Testing single syllable...")
    syllables = [(0.0, 0.5, 1.0)]
    embeddings = pool_syllables(features, syllables, sr=16000, method='mean', hop_length=160)
    assert embeddings.shape == (1, 13), f"Expected (1, 13), got {embeddings.shape}"
    print("✓ Single syllable handled correctly")
    
    # Test 3: ONC with empty syllables
    print("Testing ONC with empty syllables...")
    embeddings = pool_syllables(features, [], sr=16000, method='onc', hop_length=160)
    assert embeddings.shape == (0, 39), f"Expected (0, 39), got {embeddings.shape}"
    print("✓ ONC empty syllables handled correctly")
    
    print("✅ TEST 5 PASSED")
    return True


def test_mix_methods():
    """Test mixing segmentation and embedding methods."""
    print("\n" + "="*70)
    print("TEST 6: Mix Segmentation & Embedding Methods")
    print("="*70)
    
    from findsylls.embedding import embed_audio
    
    audio_path = 'test_samples/SP20_117.wav'
    
    # Classical segmentation + classical embedding
    print("\n1. peaks_and_valleys + MFCC...")
    emb1, _ = embed_audio(
        audio_path,
        segmentation='peaks_and_valleys',
        embedder='mfcc',
        pooling='mean'
    )
    print(f"   ✓ Shape: {emb1.shape}")
    
    # Neural segmentation + classical embedding
    try:
        import torch
        print("\n2. Sylber segmentation + MFCC embedding...")
        emb2, _ = embed_audio(
            audio_path,
            segmentation='sylber',
            embedder='mfcc',
            pooling='mean'
        )
        print(f"   ✓ Shape: {emb2.shape}")
        
        # Classical segmentation + neural embedding
        print("\n3. peaks_and_valleys segmentation + Sylber embedding...")
        emb3, _ = embed_audio(
            audio_path,
            segmentation='peaks_and_valleys',
            embedder='sylber',
            pooling='mean'
        )
        print(f"   ✓ Shape: {emb3.shape}")
    except ImportError:
        print("   ⚠️  Skipping neural tests: PyTorch not available")
    
    print("\n✅ TEST 6 PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE 1 EMBEDDING PIPELINE TESTS")
    print("="*70)
    
    tests = [
        ("MFCC + Mean", test_mfcc_mean),
        ("Sylber + Mean", test_sylber_mean),
        ("Sylber + ONC", test_sylber_onc),
        ("Step-by-Step API", test_step_by_step),
        ("Edge Cases", test_edge_cases),
        ("Mix Methods", test_mix_methods),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed:  {passed}/{len(tests)}")
    print(f"Failed:  {failed}/{len(tests)}")
    print(f"Skipped: {skipped}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
