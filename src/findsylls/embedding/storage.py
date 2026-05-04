"""
Storage utilities for saving and loading embeddings.

Supports multiple formats:
- NPZ: NumPy compressed format (fast, simple)
- HDF5: Hierarchical format (flexible, large datasets)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union
import json
import csv
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


def write_embedding_manifest(
    rows: List[Dict[str, Any]],
    manifest_path: Union[str, Path],
) -> None:
    """Write embedding manifest rows to CSV."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fieldnames = [
            'file_id', 'audio_path', 'embedding_path', 'num_rows', 'embedding_dim',
            'success', 'error', 'segmentation', 'features', 'pooling'
        ]
    else:
        fieldnames = list(rows[0].keys())

    with manifest_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_embedding_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load embedding manifest CSV rows."""
    manifest_path = Path(manifest_path)
    rows: List[Dict[str, Any]] = []
    with manifest_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['file_id'] = int(row.get('file_id', 0))
            row['num_rows'] = int(row.get('num_rows', 0))
            row['embedding_dim'] = int(row.get('embedding_dim', 0))
            row['success'] = str(row.get('success', 'False')).lower() == 'true'
            rows.append(row)
    return rows


def iter_embeddings_from_manifest(
    manifest_path: Union[str, Path],
    chunk_size: int = 10000,
):
    """
    Yield embedding chunks from per-file `.npz` files listed in a manifest.

    Args:
        manifest_path: CSV produced by embed_corpus_to_storage
        chunk_size: Max number of rows per yielded chunk

    Yields:
        np.ndarray chunks of shape (N, D)
    """
    rows = load_embedding_manifest(manifest_path)
    pending = []
    pending_rows = 0

    for row in rows:
        if not row['success']:
            continue
        embedding_path = row.get('embedding_path')
        if not embedding_path:
            continue
        ep = Path(embedding_path)
        if not ep.exists():
            continue

        with np.load(ep, allow_pickle=True) as data:
            emb = data['embeddings']

        if emb.size == 0:
            continue

        pending.append(emb)
        pending_rows += emb.shape[0]

        if pending_rows >= chunk_size:
            yield np.vstack(pending)
            pending = []
            pending_rows = 0

    if pending:
        yield np.vstack(pending)


def load_embedding_rows(
    manifest_path: Union[str, Path],
    requests: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Load specific embedding rows by (file_id, segment_id) keys.

    This helper enforces key-based retrieval using persisted identity arrays
    stored in each per-file shard:
      - file_id (scalar)
      - segment_ids (vector, length N)

    Args:
        manifest_path: Path to embedding manifest CSV produced by embed_corpus_to_storage.
        requests: Sequence of (file_id, segment_id) pairs to retrieve.

    Returns:
        A list of dicts with keys: file_id, segment_id, embedding_path, embedding.

    Raises:
        KeyError: Requested file_id or segment_id is missing.
        ValueError: Shard lacks required identity arrays.
    """
    if not requests:
        return []

    rows = load_embedding_manifest(manifest_path)
    file_row_by_id = {int(row["file_id"]): row for row in rows if row.get("success")}

    grouped_requests: Dict[int, List[int]] = {}
    for file_id, segment_id in requests:
        grouped_requests.setdefault(int(file_id), []).append(int(segment_id))

    results: List[Dict[str, Any]] = []
    missing_requests: List[Tuple[int, int]] = []

    for file_id, segment_ids_needed in grouped_requests.items():
        file_row = file_row_by_id.get(file_id)
        if file_row is None:
            missing_requests.extend((file_id, sid) for sid in segment_ids_needed)
            continue

        embedding_path = Path(str(file_row["embedding_path"]))
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding shard does not exist: {embedding_path}")

        with np.load(embedding_path, allow_pickle=True) as data:
            embeddings = data["embeddings"]
            has_segment_ids = "segment_ids" in data
            has_file_id = "file_id" in data

            if has_segment_ids and has_file_id:
                shard_file_id = int(np.array(data["file_id"]).item())
                if shard_file_id != file_id:
                    raise ValueError(
                        f"Shard file_id mismatch for {embedding_path}: "
                        f"manifest={file_id}, shard={shard_file_id}"
                    )
                segment_ids = np.asarray(data["segment_ids"], dtype=np.int64)
                id_to_row = {int(seg_id): idx for idx, seg_id in enumerate(segment_ids.tolist())}
            else:
                raise ValueError(
                    "Embedding shard is missing persisted identity arrays "
                    f"(file_id/segment_ids): {embedding_path}. "
                    "Re-embed with embed_corpus_to_storage."
                )

            for segment_id in segment_ids_needed:
                row_idx = id_to_row.get(segment_id)
                if row_idx is None:
                    missing_requests.append((file_id, segment_id))
                    continue
                results.append(
                    {
                        "file_id": file_id,
                        "segment_id": segment_id,
                        "embedding_path": str(embedding_path),
                        "embedding": np.asarray(embeddings[row_idx]),
                    }
                )

    if missing_requests:
        missing_str = ", ".join([f"({fid}, {sid})" for fid, sid in missing_requests])
        raise KeyError(f"Missing requested embedding rows for keys: {missing_str}")

    request_order = {(int(fid), int(sid)): i for i, (fid, sid) in enumerate(requests)}
    results.sort(key=lambda row: request_order[(row["file_id"], row["segment_id"])])
    return results
