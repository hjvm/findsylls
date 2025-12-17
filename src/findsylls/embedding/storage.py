"""
Storage utilities for saving and loading embeddings.

Supports multiple formats:
- NPZ: NumPy compressed format (fast, simple)
- HDF5: Hierarchical format (flexible, large datasets)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import warnings

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def save_embeddings_npz(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    compress: bool = True
) -> None:
    """
    Save corpus embeddings to NPZ format.
    
    NPZ format stores:
    - embeddings_N: Embedding array for file N
    - metadata_N: JSON string of metadata for file N
    - audio_paths: List of audio file paths
    - success_flags: Boolean array indicating success/failure
    
    Args:
        results: List of result dicts from embed_corpus()
        output_path: Path to save NPZ file
        compress: Use compression (default: True)
        
    Example:
        >>> results = embed_corpus(audio_files)
        >>> save_embeddings_npz(results, 'corpus_embeddings.npz')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_dict = {}
    audio_paths = []
    success_flags = []
    
    for i, result in enumerate(results):
        audio_paths.append(result['audio_path'])
        success_flags.append(result['success'])
        
        if result['success']:
            save_dict[f'embeddings_{i}'] = result['embeddings']
            save_dict[f'metadata_{i}'] = json.dumps(result['metadata'])
        else:
            # Save empty array and error info
            save_dict[f'embeddings_{i}'] = np.array([])
            save_dict[f'metadata_{i}'] = json.dumps({
                'error': result['error'],
                'audio_path': result['audio_path']
            })
    
    # Add global info
    save_dict['audio_paths'] = np.array(audio_paths, dtype=object)
    save_dict['success_flags'] = np.array(success_flags, dtype=bool)
    save_dict['num_files'] = len(results)
    
    # Save
    if compress:
        np.savez_compressed(output_path, **save_dict)
    else:
        np.savez(output_path, **save_dict)
    
    print(f"Saved embeddings to {output_path}")


def load_embeddings_npz(
    input_path: Union[str, Path],
    filter_failed: bool = True
) -> List[Dict[str, Any]]:
    """
    Load corpus embeddings from NPZ format.
    
    Args:
        input_path: Path to NPZ file
        filter_failed: If True, exclude failed files from results
        
    Returns:
        results: List of result dicts (same format as embed_corpus output)
        
    Example:
        >>> results = load_embeddings_npz('corpus_embeddings.npz')
        >>> print(f"Loaded {len(results)} files")
    """
    input_path = Path(input_path)
    
    with np.load(input_path, allow_pickle=True) as data:
        audio_paths = data['audio_paths'].tolist()
        success_flags = data['success_flags']
        num_files = int(data['num_files'])
        
        results = []
        for i in range(num_files):
            embeddings = data[f'embeddings_{i}']
            metadata = json.loads(str(data[f'metadata_{i}']))
            
            result = {
                'audio_path': audio_paths[i],
                'embeddings': embeddings if len(embeddings) > 0 else None,
                'metadata': metadata if 'error' not in metadata else None,
                'success': bool(success_flags[i]),
                'error': metadata.get('error', None) if 'error' in metadata else None
            }
            
            if not filter_failed or result['success']:
                results.append(result)
    
    print(f"Loaded {len(results)} files from {input_path}")
    return results


def save_embeddings_hdf5(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    compression: str = 'gzip'
) -> None:
    """
    Save corpus embeddings to HDF5 format.
    
    HDF5 format provides:
    - Hierarchical organization (one group per file)
    - Compression
    - Efficient partial loading
    - Metadata storage as attributes
    
    Structure:
        /file_0/
            embeddings (dataset)
            metadata (attributes)
        /file_1/
            embeddings (dataset)
            metadata (attributes)
        ...
        /corpus_info (attributes: audio_paths, success_flags, etc.)
    
    Args:
        results: List of result dicts from embed_corpus()
        output_path: Path to save HDF5 file
        compression: Compression algorithm ('gzip', 'lzf', None)
        
    Example:
        >>> results = embed_corpus(audio_files)
        >>> save_embeddings_hdf5(results, 'corpus_embeddings.h5')
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for HDF5 storage. "
            "Install with: pip install h5py"
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create group for each file
        for i, result in enumerate(results):
            group = f.create_group(f'file_{i}')
            
            # Store embeddings
            if result['success']:
                group.create_dataset(
                    'embeddings',
                    data=result['embeddings'],
                    compression=compression
                )
                
                # Store metadata as attributes
                for key, value in result['metadata'].items():
                    # Convert non-serializable types
                    if isinstance(value, (np.ndarray, list)):
                        value = json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
                    elif not isinstance(value, (str, int, float, bool)):
                        value = json.dumps(value)
                    group.attrs[key] = value
            else:
                # Store error info
                group.attrs['error'] = result['error']
            
            # Store file-level info
            group.attrs['audio_path'] = result['audio_path']
            group.attrs['success'] = result['success']
        
        # Store corpus-level info
        f.attrs['num_files'] = len(results)
        f.attrs['num_success'] = sum(r['success'] for r in results)
    
    print(f"Saved embeddings to {output_path}")


def load_embeddings_hdf5(
    input_path: Union[str, Path],
    filter_failed: bool = True,
    file_indices: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Load corpus embeddings from HDF5 format.
    
    Args:
        input_path: Path to HDF5 file
        filter_failed: If True, exclude failed files from results
        file_indices: If provided, load only specific file indices
        
    Returns:
        results: List of result dicts (same format as embed_corpus output)
        
    Example:
        >>> # Load all files
        >>> results = load_embeddings_hdf5('corpus_embeddings.h5')
        >>> 
        >>> # Load only specific files
        >>> results = load_embeddings_hdf5('corpus_embeddings.h5', file_indices=[0, 5, 10])
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for HDF5 storage. "
            "Install with: pip install h5py"
        )
    
    input_path = Path(input_path)
    
    with h5py.File(input_path, 'r') as f:
        num_files = f.attrs['num_files']
        
        if file_indices is None:
            file_indices = range(num_files)
        
        results = []
        for i in file_indices:
            group = f[f'file_{i}']
            
            success = bool(group.attrs['success'])
            audio_path = str(group.attrs['audio_path'])
            
            if success:
                embeddings = group['embeddings'][:]
                
                # Reconstruct metadata from attributes
                metadata = {}
                for key in group.attrs.keys():
                    if key not in ['audio_path', 'success']:
                        value = group.attrs[key]
                        # Try to parse JSON strings back
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        metadata[key] = value
                
                result = {
                    'audio_path': audio_path,
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'success': True,
                    'error': None
                }
            else:
                error = str(group.attrs.get('error', 'Unknown error'))
                result = {
                    'audio_path': audio_path,
                    'embeddings': None,
                    'metadata': None,
                    'success': False,
                    'error': error
                }
            
            if not filter_failed or result['success']:
                results.append(result)
    
    print(f"Loaded {len(results)} files from {input_path}")
    return results


def save_embeddings(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = 'auto'
) -> None:
    """
    Save embeddings in the appropriate format based on file extension.
    
    Args:
        results: List of result dicts from embed_corpus()
        output_path: Path to save file (.npz or .h5/.hdf5)
        format: 'auto', 'npz', or 'hdf5' (auto detects from extension)
        
    Example:
        >>> results = embed_corpus(audio_files)
        >>> save_embeddings(results, 'corpus.npz')  # Auto-detects NPZ
        >>> save_embeddings(results, 'corpus.h5')   # Auto-detects HDF5
    """
    output_path = Path(output_path)
    
    if format == 'auto':
        suffix = output_path.suffix.lower()
        if suffix == '.npz':
            format = 'npz'
        elif suffix in ['.h5', '.hdf5']:
            format = 'hdf5'
        else:
            warnings.warn(
                f"Unknown extension '{suffix}', defaulting to NPZ format. "
                "Use format='npz' or format='hdf5' to specify explicitly."
            )
            format = 'npz'
    
    if format == 'npz':
        save_embeddings_npz(results, output_path)
    elif format == 'hdf5':
        save_embeddings_hdf5(results, output_path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz' or 'hdf5'.")


def load_embeddings(
    input_path: Union[str, Path],
    format: str = 'auto',
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Load embeddings in the appropriate format based on file extension.
    
    Args:
        input_path: Path to load file (.npz or .h5/.hdf5)
        format: 'auto', 'npz', or 'hdf5' (auto detects from extension)
        **kwargs: Additional arguments for format-specific loaders
        
    Returns:
        results: List of result dicts
        
    Example:
        >>> results = load_embeddings('corpus.npz')
        >>> results = load_embeddings('corpus.h5', file_indices=[0, 5, 10])
    """
    input_path = Path(input_path)
    
    if format == 'auto':
        suffix = input_path.suffix.lower()
        if suffix == '.npz':
            format = 'npz'
        elif suffix in ['.h5', '.hdf5']:
            format = 'hdf5'
        else:
            raise ValueError(
                f"Cannot auto-detect format from extension '{suffix}'. "
                "Use format='npz' or format='hdf5' to specify explicitly."
            )
    
    if format == 'npz':
        return load_embeddings_npz(input_path, **kwargs)
    elif format == 'hdf5':
        return load_embeddings_hdf5(input_path, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz' or 'hdf5'.")
