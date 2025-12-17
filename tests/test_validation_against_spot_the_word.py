"""
Validation script: Compare findsylls implementation against spot_the_word legacy code.

This ensures our new implementation produces consistent results with the original
spot_the_word syllable representation extraction.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from findsylls.embedding.pipeline import embed_audio
from findsylls.audio.utils import load_audio


def load_spot_the_word_embeddings(pkl_path: str) -> Dict:
    """Load embeddings from spot_the_word format."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nLoaded {pkl_path}")
    print(f"  Format: {data['feat']} + {data['syl_repr']}")
    print(f"  Syllables: {data['embeddings'].shape[0]}")
    print(f"  Dimensions: {data['embeddings'].shape[1]}")
    print(f"  Utterances: {len(data['per_utt_meta_idx'])}")
    print(f"  Audio files: {len(data['wav_paths'])}")
    
    return data


def extract_with_findsylls(
    audio_path: str, 
    embedder: str, 
    pooling: str,
    syllable_boundaries: Optional[List[Tuple[float, float, float]]] = None
) -> Dict:
    """
    Extract embeddings using new findsylls API.
    
    Args:
        audio_path: Path to audio file
        embedder: Feature extractor ('mfcc', 'sylber', etc.)
        pooling: Pooling method ('mean', 'onc', etc.)
        syllable_boundaries: Optional pre-defined boundaries (s, p, e) to use
                           If None, will segment automatically
    """
    from findsylls.audio.utils import load_audio
    from findsylls.embedding.extractors import extract_features
    from findsylls.embedding.pooling import pool_syllables
    
    # Load audio
    audio, sr = load_audio(audio_path, samplerate=16000)
    
    # Use provided boundaries if given
    if syllable_boundaries is None:
        from findsylls.pipeline.pipeline import segment_audio as segment_audio_pipeline
        # Use OLD spot_the_word defaults: envelope_fn='sbs', segment_fn='peaks_and_valleys'
        syllables, _, _ = segment_audio_pipeline(
            audio_file=audio_path,
            samplerate=16000,
            envelope_fn='sbs',
            segment_fn='peaks_and_valleys'
        )
    else:
        syllables = syllable_boundaries
    
    # Extract frame-level features
    features, times = extract_features(
        audio,
        sr=sr,
        method=embedder,
        return_times=True
    )
    
    # Calculate hop_length
    num_frames = features.shape[0]
    if num_frames > 1:
        avg_frame_time = times[-1] / (num_frames - 1)
        hop_length = int(avg_frame_time * sr)
    else:
        hop_length = 160
    
    # Pool frames into syllable embeddings
    embeddings = pool_syllables(
        features,
        syllables,
        sr=sr,
        method=pooling,
        hop_length=hop_length
    )
    
    return {
        'embeddings': embeddings,
        'num_syllables': len(syllables),
        'boundaries': [(s, e) for s, _, e in syllables],
        'peaks': [p for _, p, _ in syllables]
    }


def compare_single_file(
    audio_path: str,
    old_data: Dict,
    old_indices: List[int],
    embedder: str,
    pooling: str,
    use_old_boundaries: bool = True
) -> Dict:
    """
    Compare embeddings for a single file.
    
    Args:
        audio_path: Path to audio file
        old_data: Full old dataset
        old_indices: Syllable indices for this file in old data
        embedder: Embedder type ('mfcc', 'sylber', etc.)
        pooling: Pooling method ('mean', 'onc', etc.)
        use_old_boundaries: If True, use boundaries from old data for fair comparison
    
    Returns:
        Dict with comparison statistics
    """
    # Get old embeddings and metadata for this file
    old_embeddings = old_data['embeddings'][old_indices]
    old_meta = [old_data['meta'][i] for i in old_indices]
    
    # Prepare old boundaries (s, p, e) if using them
    old_boundaries = None
    if use_old_boundaries:
        old_boundaries = [(m['s'], m['p'], m['e']) for m in old_meta]
    
    # Extract with new code
    try:
        new_result = extract_with_findsylls(
            audio_path, 
            embedder, 
            pooling,
            syllable_boundaries=old_boundaries
        )
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'audio_path': audio_path
        }
    
    new_embeddings = new_result['embeddings']
    
    # Compare dimensions
    if old_embeddings.shape[1] != new_embeddings.shape[1]:
        return {
            'success': False,
            'error': f'Dimension mismatch: old={old_embeddings.shape[1]}, new={new_embeddings.shape[1]}',
            'audio_path': audio_path
        }
    
    # Number of syllables
    num_sylls_old = len(old_indices)
    num_sylls_new = new_result['num_syllables']
    
    # If using old boundaries, syllable counts MUST match
    if use_old_boundaries and num_sylls_old != num_sylls_new:
        return {
            'success': False,
            'error': f'Syllable count mismatch despite using old boundaries: old={num_sylls_old}, new={num_sylls_new}',
            'audio_path': audio_path
        }
    
    # Compare embeddings
    if num_sylls_old == num_sylls_new:
        # Compute correlation and RMSE
        correlation = np.corrcoef(
            old_embeddings.flatten(),
            new_embeddings.flatten()
        )[0, 1]
        
        rmse = np.sqrt(np.mean((old_embeddings - new_embeddings) ** 2))
        max_diff = np.max(np.abs(old_embeddings - new_embeddings))
        
        # Also compute per-syllable correlations
        per_syll_correlations = []
        for i in range(num_sylls_old):
            if old_embeddings[i].std() > 0 and new_embeddings[i].std() > 0:
                corr = np.corrcoef(old_embeddings[i], new_embeddings[i])[0, 1]
                per_syll_correlations.append(corr)
        
        return {
            'success': True,
            'audio_path': audio_path,
            'num_syllables_old': num_sylls_old,
            'num_syllables_new': num_sylls_new,
            'syllables_match': True,
            'used_old_boundaries': use_old_boundaries,
            'correlation': correlation,
            'rmse': rmse,
            'max_diff': max_diff,
            'per_syllable_correlation_mean': np.mean(per_syll_correlations) if per_syll_correlations else 0.0,
            'old_boundaries': [(m['s'], m['e']) for m in old_meta],
            'new_boundaries': new_result['boundaries']
        }
    else:
        # Different number of syllables
        return {
            'success': True,
            'audio_path': audio_path,
            'num_syllables_old': num_sylls_old,
            'num_syllables_new': num_sylls_new,
            'syllables_match': False,
            'used_old_boundaries': use_old_boundaries,
            'note': 'Segmentation differs - expected if using different methods'
        }


def validate_brent_mfcc():
    """Validate MFCC extraction on Brent corpus."""
    print("\n" + "="*70)
    print("VALIDATION: Brent MFCC Mean Embeddings")
    print("="*70)
    
    # Path to old embeddings
    old_pkl = '/Users/hjvm/Documents/UPenn/unsupervised_speech_segmentation/spot_the_word/syllable_reprs/brent_mfcc_mean.pkl'
    
    if not Path(old_pkl).exists():
        print(f"\n⚠️  Old embeddings not found: {old_pkl}")
        print("Skipping validation.")
        return
    
    # Load old embeddings
    old_data = load_spot_the_word_embeddings(old_pkl)
    
    # Test on first few files
    test_files_idx = list(range(min(10, len(old_data['per_utt_meta_idx']))))  # Test first 10 files
    
    print(f"\nTesting on {len(test_files_idx)} files from Brent corpus...")
    
    results = []
    for file_idx in test_files_idx:
        # Get syllable indices for this file
        syll_indices = old_data['per_utt_meta_idx'][file_idx]
        
        # Get audio path from first syllable's metadata
        first_syll_meta = old_data['meta'][syll_indices[0]]
        audio_path = first_syll_meta['wav']
        
        # Convert relative path to absolute
        spot_the_word_root = '/Users/hjvm/Documents/UPenn/unsupervised_speech_segmentation/spot_the_word'
        audio_path_abs = str(Path(spot_the_word_root) / audio_path.lstrip('./'))
        
        if not Path(audio_path_abs).exists():
            print(f"\n⚠️  Audio file not found: {audio_path_abs}")
            continue
        
        print(f"\nProcessing file {file_idx + 1}/{len(test_files_idx)}: {Path(audio_path).name}")
        
        # Compare
        result = compare_single_file(
            audio_path=audio_path_abs,
            old_data=old_data,
            old_indices=syll_indices,
            embedder='mfcc',
            pooling='mean'
        )
        
        results.append(result)
        
        # Print result
        if result['success']:
            if result['syllables_match']:
                print(f"  ✓ Syllable count matches: {result['num_syllables_old']}")
                print(f"  ✓ Correlation: {result['correlation']:.4f}")
                print(f"  ✓ RMSE: {result['rmse']:.6f}")
                print(f"  ✓ Max diff: {result['max_diff']:.6f}")
            else:
                print(f"  ⚠️  Syllable count differs: old={result['num_syllables_old']}, new={result['num_syllables_new']}")
                print(f"  Note: {result.get('note', '')}")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r['success']]
    matching = [r for r in successful if r.get('syllables_match', False)]
    
    print(f"Files processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Matching syllable counts: {len(matching)}")
    
    if matching:
        avg_corr = np.mean([r['correlation'] for r in matching])
        avg_rmse = np.mean([r['rmse'] for r in matching])
        print(f"\nAverage correlation: {avg_corr:.4f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        
        if avg_corr > 0.95:
            print("\n✅ HIGH CORRELATION - Implementation matches well!")
        elif avg_corr > 0.8:
            print("\n⚠️  MODERATE CORRELATION - Some differences present")
        else:
            print("\n❌ LOW CORRELATION - Significant differences")
    
    return results


def main():
    """Run all validation tests."""
    print("="*70)
    print("FINDSYLLS VALIDATION AGAINST SPOT_THE_WORD")
    print("="*70)
    
    print("\nThis script validates the new findsylls implementation")
    print("against legacy embeddings from spot_the_word repository.")
    print("\nNote: Exact matches are not expected due to:")
    print("  - Different segmentation algorithms")
    print("  - Different frame alignment")
    print("  - Different pooling implementations")
    print("\nWe're looking for HIGH CORRELATION (>0.95) to confirm consistency.")
    
    # Test MFCC on Brent
    results = validate_brent_mfcc()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    
    return results


if __name__ == '__main__':
    results = main()
