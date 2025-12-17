"""
Tests for Phase 3: Corpus Processing

Tests:
1. embed_corpus() basic functionality
2. Parallel processing
3. Error handling
4. NPZ save/load
5. HDF5 save/load (if available)
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from findsylls.embedding.pipeline import embed_corpus
from findsylls.embedding.storage import (
    save_embeddings_npz, load_embeddings_npz,
    save_embeddings, load_embeddings
)

try:
    from findsylls.embedding.storage import save_embeddings_hdf5, load_embeddings_hdf5
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def print_test_header(test_num, test_name):
    """Print formatted test header."""
    print("\n" + "="*70)
    print(f"TEST {test_num}: {test_name}")
    print("="*70)


def test_embed_corpus_basic():
    """Test basic corpus processing."""
    print_test_header(1, "embed_corpus() Basic Functionality")
    
    # Get test files
    test_dir = Path(__file__).parent.parent / 'test_samples'
    audio_files = list(test_dir.glob('*.wav'))[:2]  # Use first 2 files
    
    print(f"\nProcessing {len(audio_files)} files...")
    
    results = embed_corpus(
        audio_files=audio_files,
        embedder='mfcc',
        pooling='mean',
        n_jobs=1,
        verbose=False
    )
    
    # Verify results structure
    assert len(results) == len(audio_files), "Wrong number of results"
    
    for result in results:
        assert 'audio_path' in result
        assert 'embeddings' in result
        assert 'metadata' in result
        assert 'success' in result
        assert 'error' in result
        
        if result['success']:
            assert result['embeddings'] is not None
            assert result['embeddings'].shape[1] == 13  # MFCC dimension
            assert result['metadata'] is not None
            assert result['error'] is None
    
    print(f"✓ Processed {len(results)} files")
    print(f"✓ All results have correct structure")
    print(f"✓ Embeddings shape: {results[0]['embeddings'].shape}")
    print("✅ TEST 1 PASSED")


def test_embed_corpus_parallel():
    """Test parallel processing."""
    print_test_header(2, "Parallel Processing")
    
    # Get test files
    test_dir = Path(__file__).parent.parent / 'test_samples'
    audio_files = list(test_dir.glob('*.wav'))[:2]
    
    print(f"\nProcessing with n_jobs=2...")
    
    results = embed_corpus(
        audio_files=audio_files,
        embedder='mfcc',
        pooling='mean',
        n_jobs=2,
        verbose=False
    )
    
    assert len(results) == len(audio_files)
    assert all(r['success'] for r in results)
    
    print(f"✓ Parallel processing completed")
    print(f"✓ All {len(results)} files processed successfully")
    print("✅ TEST 2 PASSED")


def test_error_handling():
    """Test error handling with invalid files."""
    print_test_header(3, "Error Handling")
    
    # Mix valid and invalid files
    test_dir = Path(__file__).parent.parent / 'test_samples'
    valid_file = list(test_dir.glob('*.wav'))[0]
    invalid_file = 'nonexistent.wav'
    
    audio_files = [str(valid_file), invalid_file]
    
    print(f"\nProcessing with fail_on_error=False...")
    print(f"  Valid file: {valid_file.name}")
    print(f"  Invalid file: {invalid_file}")
    
    results = embed_corpus(
        audio_files=audio_files,
        embedder='mfcc',
        n_jobs=1,
        verbose=False,
        fail_on_error=False  # Should skip invalid file
    )
    
    assert len(results) == 2
    assert results[0]['success'] == True
    assert results[1]['success'] == False
    assert results[1]['error'] is not None
    
    print(f"\n✓ Valid file processed successfully")
    print(f"✓ Invalid file failed gracefully")
    print(f"✓ Error message: {results[1]['error'][:50]}...")
    print("✅ TEST 3 PASSED")


def test_npz_save_load():
    """Test NPZ save and load."""
    print_test_header(4, "NPZ Save/Load")
    
    # Create test results
    test_dir = Path(__file__).parent.parent / 'test_samples'
    audio_files = list(test_dir.glob('*.wav'))[:2]
    
    results = embed_corpus(
        audio_files=audio_files,
        embedder='mfcc',
        n_jobs=1,
        verbose=False
    )
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / 'test_embeddings.npz'
        
        print(f"\nSaving to: {npz_path}")
        save_embeddings_npz(results, npz_path)
        
        print(f"Loading from: {npz_path}")
        loaded_results = load_embeddings_npz(npz_path)
        
        # Verify
        assert len(loaded_results) == len(results)
        
        for orig, loaded in zip(results, loaded_results):
            if orig['success']:
                assert np.allclose(orig['embeddings'], loaded['embeddings'])
                assert orig['metadata']['num_syllables'] == loaded['metadata']['num_syllables']
                assert orig['audio_path'] == loaded['audio_path']
        
        file_size = npz_path.stat().st_size / 1024
        print(f"\n✓ Saved {len(results)} files")
        print(f"✓ File size: {file_size:.1f} KB")
        print(f"✓ Loaded and verified all data")
        print("✅ TEST 4 PASSED")


def test_hdf5_save_load():
    """Test HDF5 save and load (if h5py available)."""
    print_test_header(5, "HDF5 Save/Load")
    
    if not HAS_H5PY:
        print("\nh5py not installed. Skipping HDF5 tests.")
        print("Install with: pip install h5py")
        print("⚠️  TEST 5 SKIPPED")
        return "SKIPPED"
    
    # Create test results
    test_dir = Path(__file__).parent.parent / 'test_samples'
    audio_files = list(test_dir.glob('*.wav'))[:2]
    
    results = embed_corpus(
        audio_files=audio_files,
        embedder='mfcc',
        n_jobs=1,
        verbose=False
    )
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / 'test_embeddings.h5'
        
        print(f"\nSaving to: {h5_path}")
        save_embeddings_hdf5(results, h5_path)
        
        print(f"Loading from: {h5_path}")
        loaded_results = load_embeddings_hdf5(h5_path)
        
        # Verify
        assert len(loaded_results) == len(results)
        
        for orig, loaded in zip(results, loaded_results):
            if orig['success']:
                assert np.allclose(orig['embeddings'], loaded['embeddings'])
                assert orig['audio_path'] == loaded['audio_path']
        
        # Test partial loading
        print(f"Testing partial loading...")
        partial = load_embeddings_hdf5(h5_path, file_indices=[0])
        assert len(partial) == 1
        assert partial[0]['audio_path'] == results[0]['audio_path']
        
        file_size = h5_path.stat().st_size / 1024
        print(f"\n✓ Saved {len(results)} files")
        print(f"✓ File size: {file_size:.1f} KB")
        print(f"✓ Loaded and verified all data")
        print(f"✓ Partial loading works")
        print("✅ TEST 5 PASSED")


def test_auto_format_detection():
    """Test automatic format detection."""
    print_test_header(6, "Auto Format Detection")
    
    # Create test results
    test_dir = Path(__file__).parent.parent / 'test_samples'
    audio_files = list(test_dir.glob('*.wav'))[:1]
    
    results = embed_corpus(
        audio_files=audio_files,
        embedder='mfcc',
        n_jobs=1,
        verbose=False
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test NPZ auto-detection
        npz_path = Path(tmpdir) / 'test.npz'
        save_embeddings(results, npz_path)  # Auto-detects NPZ
        loaded = load_embeddings(npz_path)  # Auto-detects NPZ
        assert len(loaded) == len(results)
        print(f"\n✓ NPZ auto-detection works")
        
        # Test HDF5 auto-detection (if available)
        if HAS_H5PY:
            try:
                h5_path = Path(tmpdir) / 'test.h5'
                save_embeddings(results, h5_path)  # Auto-detects HDF5
                loaded = load_embeddings(h5_path)  # Auto-detects HDF5
                assert len(loaded) == len(results)
                print(f"✓ HDF5 auto-detection works")
            except ImportError:
                print(f"⚠️  HDF5 auto-detection skipped (h5py error)")
        else:
            print(f"⚠️  HDF5 auto-detection skipped (h5py not installed)")
        
        print("✅ TEST 6 PASSED")


def main():
    """Run all tests."""
    print("="*70)
    print("PHASE 3 CORPUS PROCESSING TESTS")
    print("="*70)
    
    tests = [
        ("embed_corpus() Basic", test_embed_corpus_basic),
        ("Parallel Processing", test_embed_corpus_parallel),
        ("Error Handling", test_error_handling),
        ("NPZ Save/Load", test_npz_save_load),
        ("HDF5 Save/Load", test_hdf5_save_load),
        ("Auto Format Detection", test_auto_format_detection)
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result == "SKIPPED":
                skipped += 1
            else:
                passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
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
    else:
        print(f"\n❌ {failed} test(s) failed")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
