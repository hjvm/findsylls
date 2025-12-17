"""
High-level embedding pipeline API.

Provides user-facing functions for extracting syllable embeddings from audio.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

from ..audio.utils import load_audio
from ..pipeline.pipeline import segment_audio as segment_audio_pipeline
from .extractors import extract_features
from .pooling import pool_syllables


def embed_audio(
    audio_path: str,
    segmentation: str = 'sylber',
    embedder: str = 'sylber',
    pooling: str = 'mean',
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    embedder_kwargs: Optional[Dict[str, Any]] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Extract syllable embeddings from audio file.
    
    Complete pipeline: load audio → segment → extract features → pool embeddings
    
    Args:
        audio_path: Path to audio file
        segmentation: Segmentation method (any from Methods 1-11)
            - 'peaks_and_valleys': Classical envelope-based (default for envelope methods)
            - 'sylber': Sylber end-to-end model
            - 'vg_hubert': VG-HuBERT model (future)
            - etc.
        embedder: Feature extraction method
            - 'sylber': Sylber model (768-dim, ~50 fps)
            - 'vg_hubert': VG-HuBERT model (768-dim, ~50 fps)
                          Requires embedder_kwargs={'model_path': '/path/to/vg-hubert_3'}
            - 'mfcc': Mel-frequency cepstral coefficients (13-dim, ~100 fps)
                      Use embedder_kwargs={'include_delta': True, 'include_delta_delta': True}
                      for delta features (39-dim = 13 + 13 + 13)
            - 'melspec': Mel-spectrogram (80-dim, ~100 fps)
        pooling: Syllable pooling method
            - 'mean': Average frames (default)
            - 'onc': Onset-Nucleus-Coda template (3× dimensions)
            - 'max': Max pooling
            - 'median': Median pooling
        sr: Target sample rate (default: 16000)
        layer: Layer index for neural models (model-specific defaults if None)
        device: Device for neural models: 'auto', 'cuda', 'cpu'
        segmentation_kwargs: Additional arguments for segmentation
        embedder_kwargs: Additional arguments for feature extraction
        pooling_kwargs: Additional arguments for pooling
        return_metadata: If True, return (embeddings, metadata) tuple
                        If False, return only embeddings
        
    Returns:
        If return_metadata=True:
            (embeddings, metadata): 
                - embeddings: np.ndarray, shape (num_syllables, embedding_dim)
                - metadata: dict with boundaries, methods, parameters, etc.
        If return_metadata=False:
            embeddings: np.ndarray, shape (num_syllables, embedding_dim)
            
    Example:
        >>> embeddings, meta = embed_audio(
        ...     'audio.wav',
        ...     segmentation='sylber',
        ...     embedder='sylber',
        ...     pooling='mean'
        ... )
        >>> print(embeddings.shape)  # (num_syllables, 768)
        >>> print(meta['num_syllables'])  # e.g., 15
    """
    # Handle None kwargs
    segmentation_kwargs = segmentation_kwargs or {}
    embedder_kwargs = embedder_kwargs or {}
    pooling_kwargs = pooling_kwargs or {}
    
    # Step 1: Load audio
    audio, actual_sr = load_audio(audio_path, samplerate=sr)
    duration = len(audio) / actual_sr
    
    # Check if we can use embedder-native segmentation (e.g., Sylber provides peaks)
    use_embedder_segments = (embedder == 'sylber' and pooling == 'onc')
    
    if use_embedder_segments:
        # Step 2+3 combined: Extract features AND get segments with peaks from embedder
        # Warn user that peak detection is additional computation
        import warnings
        warnings.warn(
            "ONC pooling with embedder='sylber' detects peaks using cosine similarity "
            "between frames. While this is native to Sylber's representation space, "
            "Sylber was not explicitly trained for peak detection. Peaks are inferred "
            "as the frames with maximum similarity to neighbors within each segment.",
            UserWarning
        )
        
        features, times, syllables = extract_features(
            audio,
            sr=actual_sr,
            method=embedder,
            layer=layer,
            device=device,
            return_times=True,
            return_segments=True,
            **embedder_kwargs
        )
    else:
        # Step 2: Segment into syllables using separate segmentation method
        # Enable peak detection for Sylber segmentation when ONC is requested
        if segmentation == 'sylber' and pooling == 'onc':
            segmentation_kwargs = segmentation_kwargs.copy()
            segmentation_kwargs['detect_peaks'] = True
            import warnings
            warnings.warn(
                "ONC pooling with segmentation='sylber' detects peaks using cosine similarity "
                "between frames in Sylber's representation space. While this uses data already "
                "computed during segmentation (minimal overhead), Sylber was not explicitly "
                "trained for peak detection. Peaks are inferred as the frames with maximum "
                "similarity to neighbors within each segment. This is more informed than using "
                "temporal midpoints.",
                UserWarning
            )
        
        # Use pipeline.segment_audio which handles both envelope-based and end2end methods
        syllables, _, _ = segment_audio_pipeline(
            audio_file=audio_path,
            samplerate=actual_sr,
            method=segmentation,
            segmentation_kwargs=segmentation_kwargs
        )
        
        # Warn if ONC pooling is requested with other segmenters that don't provide peaks
        if pooling == 'onc' and segmentation not in ['sylber', 'peaks_and_valleys']:
            import warnings
            warnings.warn(
                f"ONC pooling requested with segmentation='{segmentation}' + embedder='{embedder}'. "
                f"Note: segmentation='{segmentation}' does not detect acoustic peaks (uses midpoint "
                f"as proxy), which may not align with phonetic nuclei. For peak-based segmentation, "
                f"use segmentation='peaks_and_valleys' (envelope peaks) or segmentation='sylber' "
                f"(cosine similarity peaks).",
                UserWarning
            )
        
        # Step 3: Extract frame-level features
        features, times = extract_features(
            audio,
            sr=actual_sr,
            method=embedder,
            layer=layer,
            device=device,
            return_times=True,
            **embedder_kwargs
        )
    
    # Step 4: Pool frames into syllable embeddings
    # Calculate hop_length for accurate frame timing
    num_frames = features.shape[0]
    if num_frames > 1:
        # Estimate hop_length from times
        avg_frame_time = times[-1] / (num_frames - 1)
        hop_length = int(avg_frame_time * actual_sr)
    else:
        # Fallback for single frame
        hop_length = 160  # 10ms at 16kHz
    
    embeddings = pool_syllables(
        features,
        syllables,
        sr=actual_sr,
        method=pooling,
        hop_length=hop_length,
        **pooling_kwargs
    )
    
    # Prepare metadata
    if return_metadata:
        metadata = {
            # Syllable information
            'boundaries': [(start, end) for start, _, end in syllables],
            'peaks': [peak for _, peak, _ in syllables],
            'num_syllables': len(syllables),
            
            # Audio information
            'audio_path': str(Path(audio_path).resolve()),
            'duration': duration,
            'sample_rate': actual_sr,
            
            # Method information
            'segmentation_method': segmentation,
            'embedder': embedder,
            'pooling': pooling,
            'layer': layer,
            
            # Embedding information
            'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'fps': num_frames / duration if duration > 0 else 0,
            'hop_length': hop_length,
            
            # Timestamps
            'created_at': datetime.now().isoformat(),
            'findsylls_version': '0.1.0',  # TODO: get from package metadata
        }
        
        return embeddings, metadata
    else:
        return embeddings, None


# Convenience wrapper for backward compatibility
def embed_audio_simple(
    audio_path: str,
    segmentation: str = 'sylber',
    embedder: str = 'sylber',
    pooling: str = 'mean',
    **kwargs
) -> np.ndarray:
    """
    Simplified version of embed_audio that returns only embeddings.
    
    Args:
        audio_path: Path to audio file
        segmentation: Segmentation method
        embedder: Feature extraction method
        pooling: Pooling method
        **kwargs: Additional arguments passed to embed_audio
        
    Returns:
        embeddings: np.ndarray, shape (num_syllables, embedding_dim)
    """
    embeddings, _ = embed_audio(
        audio_path,
        segmentation=segmentation,
        embedder=embedder,
        pooling=pooling,
        return_metadata=False,
        **kwargs
    )
    return embeddings


def embed_corpus(
    audio_files: Union[List[str], List[Path]],
    segmentation: str = 'sylber',
    embedder: str = 'sylber',
    pooling: str = 'mean',
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    embedder_kwargs: Optional[Dict[str, Any]] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    verbose: bool = True,
    fail_on_error: bool = False
) -> List[Dict[str, Any]]:
    """
    Process multiple audio files in parallel and extract syllable embeddings.
    
    This function provides batch processing with:
    - Parallel execution using joblib
    - Progress tracking with tqdm
    - Error handling (skip or fail)
    - Consistent metadata for all files
    
    Args:
        audio_files: List of paths to audio files
        segmentation: Segmentation method (see embed_audio)
        embedder: Feature extraction method (see embed_audio)
        pooling: Pooling method (see embed_audio)
        sr: Target sample rate (default: 16000)
        layer: Layer index for neural models
        device: Device for neural models: 'auto', 'cuda', 'cpu'
        segmentation_kwargs: Additional arguments for segmentation
        embedder_kwargs: Additional arguments for feature extraction
        pooling_kwargs: Additional arguments for pooling
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
        verbose: Show progress bar
        fail_on_error: If True, raise exception on error. If False, skip failed files.
        
    Returns:
        results: List of dicts, one per file, containing:
            - 'audio_path': str, path to audio file
            - 'embeddings': np.ndarray, shape (num_syllables, embedding_dim)
            - 'metadata': dict with all embedding metadata
            - 'success': bool, whether processing succeeded
            - 'error': str or None, error message if failed
            
    Example:
        >>> audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
        >>> results = embed_corpus(
        ...     audio_files,
        ...     embedder='mfcc',
        ...     pooling='mean',
        ...     n_jobs=4
        ... )
        >>> # Access individual file results
        >>> for result in results:
        ...     if result['success']:
        ...         print(f"{result['audio_path']}: {result['embeddings'].shape}")
        
    Note:
        - For neural models (Sylber, VG-HuBERT), parallel processing may not speed up
          if GPU is the bottleneck. Use n_jobs=1 for GPU processing.
        - For CPU-based features (MFCC, melspec), n_jobs=-1 can provide significant speedup.
        - Models are loaded once per worker, so memory usage scales with n_jobs.
    """
    audio_files = [str(Path(f)) for f in audio_files]
    
    segmentation_kwargs = segmentation_kwargs or {}
    embedder_kwargs = embedder_kwargs or {}
    pooling_kwargs = pooling_kwargs or {}
    
    def process_single_file(audio_path: str) -> Dict[str, Any]:
        """Process a single audio file and return result dict."""
        result = {
            'audio_path': audio_path,
            'embeddings': None,
            'metadata': None,
            'success': False,
            'error': None
        }
        
        try:
            embeddings, metadata = embed_audio(
                audio_path=audio_path,
                segmentation=segmentation,
                embedder=embedder,
                pooling=pooling,
                sr=sr,
                layer=layer,
                device=device,
                segmentation_kwargs=segmentation_kwargs,
                embedder_kwargs=embedder_kwargs,
                pooling_kwargs=pooling_kwargs,
                return_metadata=True
            )
            
            result['embeddings'] = embeddings
            result['metadata'] = metadata
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            if fail_on_error:
                raise
            else:
                if verbose:
                    warnings.warn(f"Failed to process {audio_path}: {e}")
        
        return result
    
    # Process files
    if n_jobs == 1:
        # Sequential processing with progress bar
        if verbose:
            results = [
                process_single_file(f) 
                for f in tqdm(audio_files, desc="Processing audio files")
            ]
        else:
            results = [process_single_file(f) for f in audio_files]
    else:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(process_single_file)(f) for f in audio_files
        )
    
    # Summary statistics
    if verbose:
        n_success = sum(r['success'] for r in results)
        n_total = len(results)
        print(f"\nProcessed {n_success}/{n_total} files successfully")
        if n_success < n_total:
            print(f"Failed: {n_total - n_success} files")
    
    return results
