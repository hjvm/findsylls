"""
Phase 3 Example: Corpus Processing

Demonstrates:
1. Batch processing multiple audio files
2. Parallel execution
3. Saving and loading embeddings
4. Error handling
5. Working with results
"""

from pathlib import Path
import numpy as np

from findsylls.embedding.pipeline import embed_audio, embed_corpus
from findsylls.embedding.storage import save_embeddings, load_embeddings


def main():
    """Complete corpus processing workflow."""
    
    # ================================================================
    # PART 1: Process corpus in parallel
    # ================================================================
    
    print("="*70)
    print("PART 1: Processing Audio Corpus")
    print("="*70)
    
    # Get audio files from test_samples
    test_dir = Path(__file__).parent.parent / 'test_samples'
    audio_files = list(test_dir.glob('*.wav')) + list(test_dir.glob('*.flac'))
    
    print(f"\nFound {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"  - {f.name}")
    
    # Process corpus with MFCC features
    print("\nProcessing corpus with MFCC + mean pooling...")
    print("Using 4 parallel jobs for CPU-based features\n")
    
    results = embed_corpus(
        audio_files=audio_files,
        segmentation='peaks_and_valleys',  # Classical segmentation
        embedder='mfcc',                   # Fast CPU-based features
        pooling='mean',
        n_jobs=4,                          # Parallel processing
        verbose=True,                      # Show progress
        fail_on_error=False                # Skip failed files
    )
    
    # ================================================================
    # PART 2: Inspect results
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 2: Inspecting Results")
    print("="*70)
    
    for result in results:
        if result['success']:
            n_sylls = result['metadata']['num_syllables']
            shape = result['embeddings'].shape
            duration = result['metadata']['duration']
            filename = Path(result['audio_path']).name
            
            print(f"\n{filename}:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Syllables: {n_sylls}")
            print(f"  Embeddings: {shape}")
            print(f"  Segmentation: {result['metadata']['segmentation']}")
            print(f"  Embedder: {result['metadata']['embedder']}")
        else:
            print(f"\n{Path(result['audio_path']).name}: FAILED")
            print(f"  Error: {result['error']}")
    
    # ================================================================
    # PART 3: Save embeddings to NPZ format
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 3: Saving Embeddings (NPZ Format)")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'output' / 'embeddings'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = output_dir / 'corpus_mfcc_embeddings.npz'
    save_embeddings(results, npz_path, format='npz')
    
    print(f"\nSaved to: {npz_path}")
    print(f"File size: {npz_path.stat().st_size / 1024:.1f} KB")
    
    # ================================================================
    # PART 4: Load embeddings from NPZ
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 4: Loading Embeddings from NPZ")
    print("="*70)
    
    loaded_results = load_embeddings(npz_path)
    
    print(f"\nLoaded {len(loaded_results)} files")
    print("\nVerifying data integrity...")
    
    for orig, loaded in zip(results, loaded_results):
        if orig['success']:
            assert np.allclose(orig['embeddings'], loaded['embeddings'])
            assert orig['metadata']['num_syllables'] == loaded['metadata']['num_syllables']
    
    print("✓ All data verified successfully!")
    
    # ================================================================
    # PART 5: Process with different methods
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 5: Comparing Different Methods")
    print("="*70)
    
    # Try different embedder configurations
    configs = [
        {
            'name': 'MFCC (13-dim)',
            'embedder': 'mfcc',
            'embedder_kwargs': {}
        },
        {
            'name': 'MFCC + Delta (26-dim)',
            'embedder': 'mfcc',
            'embedder_kwargs': {'include_delta': True}
        },
        {
            'name': 'MFCC + Delta + Delta-Delta (39-dim)',
            'embedder': 'mfcc',
            'embedder_kwargs': {'include_delta': True, 'include_delta_delta': True}
        },
        {
            'name': 'Mel-spectrogram (80-dim)',
            'embedder': 'melspec',
            'embedder_kwargs': {}
        }
    ]
    
    comparison_results = {}
    
    # Use single file for quick comparison
    test_file = audio_files[0]
    print(f"\nTesting on: {test_file.name}\n")
    
    for config in configs:
        print(f"Processing with {config['name']}...")
        
        embeddings, meta = embed_audio(
            audio_path=str(test_file),
            segmentation='peaks_and_valleys',
            embedder=config['embedder'],
            pooling='mean',
            embedder_kwargs=config['embedder_kwargs']
        )
        
        comparison_results[config['name']] = {
            'embeddings': embeddings,
            'metadata': meta
        }
        
        print(f"  ✓ Shape: {embeddings.shape}")
        print(f"  ✓ Syllables: {meta['num_syllables']}")
    
    # ================================================================
    # PART 6: HDF5 storage (if h5py available)
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 6: HDF5 Storage (Optional)")
    print("="*70)
    
    try:
        import h5py
        
        h5_path = output_dir / 'corpus_mfcc_embeddings.h5'
        save_embeddings(results, h5_path, format='hdf5')
        
        print(f"\nSaved to: {h5_path}")
        print(f"File size: {h5_path.stat().st_size / 1024:.1f} KB")
        
        # Demonstrate partial loading
        print("\nDemonstrating partial loading...")
        partial_results = load_embeddings(h5_path, file_indices=[0])
        print(f"Loaded only file 0: {Path(partial_results[0]['audio_path']).name}")
        
    except ImportError:
        print("\nh5py not installed. Skipping HDF5 storage demo.")
        print("Install with: pip install h5py")
    
    # ================================================================
    # PART 7: Working with Sylber (if available)
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 7: Sylber Embeddings (Sequential Processing)")
    print("="*70)
    
    print("\nNote: For GPU models like Sylber, use n_jobs=1 to avoid")
    print("memory issues. Parallel processing helps CPU-based features.\n")
    
    try:
        # Process first file only to test Sylber
        sylber_result = embed_audio(
            audio_path=str(audio_files[0]),
            segmentation='sylber',
            embedder='sylber',
            pooling='mean'
        )
        
        embeddings, meta = sylber_result
        print(f"✓ Sylber processing successful")
        print(f"  File: {Path(audio_files[0]).name}")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Syllables: {meta['num_syllables']}")
        
        # For full corpus with Sylber, use n_jobs=1
        print("\nFor full corpus with Sylber:")
        print("  results = embed_corpus(")
        print("      audio_files,")
        print("      embedder='sylber',")
        print("      n_jobs=1,  # Sequential for GPU models")
        print("      verbose=True")
        print("  )")
        
    except Exception as e:
        print(f"Sylber not available: {e}")
        print("Sylber requires model download from HuggingFace Hub")
    
    # ================================================================
    # Summary
    # ================================================================
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nPhase 3 Features Demonstrated:")
    print("  ✓ Batch corpus processing with embed_corpus()")
    print("  ✓ Parallel execution (n_jobs parameter)")
    print("  ✓ Progress tracking with tqdm")
    print("  ✓ Error handling (fail_on_error=False)")
    print("  ✓ NPZ storage format")
    print("  ✓ HDF5 storage format (if h5py installed)")
    print("  ✓ Method comparison workflow")
    print("  ✓ Metadata preservation")
    
    print("\nBest Practices:")
    print("  • Use n_jobs=-1 for CPU features (MFCC, melspec)")
    print("  • Use n_jobs=1 for GPU models (Sylber, VG-HuBERT)")
    print("  • NPZ for small-medium corpora (< 1000 files)")
    print("  • HDF5 for large corpora (partial loading support)")
    print("  • Set fail_on_error=False to skip problematic files")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
