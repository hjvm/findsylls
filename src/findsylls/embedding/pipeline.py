"""
High-level embedding pipeline API.

Provides user-facing functions for extracting syllable embeddings from audio.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import gc

from ..audio.utils import load_audio
from ..pipeline.pipeline import segment_audio as segment_audio_pipeline
from ..segmentation import list_segmenters
from ..presets import list_presets, resolve_preset
from ..features import get_extractor
from .extractors import extract_features
from .pooling import pool_syllables
from .storage import write_embedding_manifest


def _normalize_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).lower().replace('-', '_').strip()


def _canonical_feature_name(value: Optional[str]) -> Optional[str]:
    """Normalize known feature aliases to canonical extractor names."""
    name = _normalize_name(value)
    if name == 'vghubert':
        return 'vg_hubert'
    if name in {'mel', 'melspectrogram'}:
        return 'melspec'
    if name in {'spec_band_subtraction', 'spectral_band_subtraction'}:
        return 'sbs'
    return name


def _normalize_syllable_tuples(
    syllables: List[Union[Tuple[float, float], Tuple[float, float, float]]],
    *,
    pooling: Optional[str],
) -> List[Tuple[float, float, float]]:
    """Validate and normalize segmentation tuples before pooling.

    Pooling-dependent policy:
    - `onc` and `max` require explicit peak information.
        - `mean` and `median` accept both (start, end) and (start, peak, end).
            For (start, end), midpoint peak is inferred.
    """
    if syllables is None:
        raise ValueError("Segmentation returned None; expected a list of (start, peak, end) tuples.")

    pooling_name = _normalize_name(pooling)
    peak_required = pooling_name in {"onc", "max"}

    normalized: List[Tuple[float, float, float]] = []

    for idx, segment in enumerate(syllables):
        if not isinstance(segment, (tuple, list)):
            raise ValueError(
                f"Invalid segment at index {idx}: expected tuple/list, got {segment!r}."
            )

        if len(segment) == 3:
            start, peak, end = segment
        elif len(segment) == 2:
            if peak_required:
                raise ValueError(
                    f"Invalid segment at index {idx}: pooling='{pooling_name}' requires (start, peak, end), "
                    f"got span-only tuple {segment!r}."
                )
            start, end = segment
            peak = start + (end - start) / 2.0
        else:
            raise ValueError(
                f"Invalid segment at index {idx}: expected (start, end) or (start, peak, end), got {segment!r}."
            )

        if not (np.isfinite(start) and np.isfinite(peak) and np.isfinite(end)):
            raise ValueError(
                f"Invalid segment at index {idx}: start/peak/end must be finite, got {segment!r}."
            )

        if start < 0 or end < 0:
            raise ValueError(
                f"Invalid segment at index {idx}: segment times must be non-negative, got {segment!r}."
            )

        if start > end:
            raise ValueError(
                f"Invalid segment at index {idx}: expected start <= end, got {segment!r}."
            )

        if not (start <= peak <= end):
            raise ValueError(
                f"Invalid segment at index {idx}: expected start <= peak <= end, got {segment!r}."
            )

        normalized.append((float(start), float(peak), float(end)))

    return normalized


def _bind_segmentation_to_selected_features(
    *,
    segmentation: str,
    features: str,
    layer: Optional[int],
    device: Optional[str],
    feature_kwargs: Optional[Dict[str, Any]],
    segmentation_kwargs: Optional[Dict[str, Any]],
    feature_extractor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ensure feature-based segmenters operate on the selected feature representation.

    For feature-driven segmentation algorithms, the segmentation stage must consume
    the same feature family chosen for embedding extraction.
    """
    seg = _normalize_name(segmentation)
    bound_kwargs: Dict[str, Any] = dict(segmentation_kwargs or {})

    if seg in {"mincut", "greedy_cosine", "cls_attention"}:
        bound_kwargs.setdefault("feature_type", features)
        caller_feature_kwargs = dict(feature_kwargs or {})
        if device is not None:
            caller_feature_kwargs.setdefault("device", device)
        if layer is not None:
            normalized_features = _canonical_feature_name(features)
            if normalized_features == "sylber":
                caller_feature_kwargs.setdefault("encoding_layer", layer)
            elif normalized_features in {"hubert", "vg_hubert"}:
                caller_feature_kwargs.setdefault("layer", layer)
        existing_feature_kwargs = bound_kwargs.get("feature_kwargs") or {}
        # Allow explicit segmentation_kwargs.feature_kwargs to override shared defaults.
        merged_feature_kwargs = {**caller_feature_kwargs, **dict(existing_feature_kwargs)}
        if merged_feature_kwargs:
            bound_kwargs["feature_kwargs"] = merged_feature_kwargs
        if feature_extractor is not None:
            bound_kwargs.setdefault("feature_extractor", feature_extractor)

    return bound_kwargs


def _resolve_peakdetect_envelope_method(
    *,
    features: str,
    segmentation_kwargs: Optional[Dict[str, Any]],
) -> str:
    """Resolve envelope method for peakdetect while enforcing envelope-only policy."""
    kwargs = dict(segmentation_kwargs or {})
    envelope_method = kwargs.get("envelope_method")

    if envelope_method:
        return str(envelope_method)

    envelope_feature_methods = {
        "rms",
        "hilbert",
        "lowpass",
        "sbs",
        "theta",
        "cls_attention",
        "greedy_cosine",
        "mincut",
    }
    if features in envelope_feature_methods:
        return features

    raise ValueError(
        "Invalid configuration: segmentation='peakdetect' requires a 1-D envelope method. "
        f"features='{features}' is not an envelope method. "
        "Set segmentation_kwargs['envelope_method'] to one of: "
        "'hilbert', 'rms', 'lowpass', 'sbs', 'theta', 'cls_attention', 'greedy_cosine', 'mincut', "
        "or use a canonical neural segmenter ('cls_attention', 'mincut', 'greedy_cosine')."
    )


def _flatten_peakdetect_envelope_kwargs(bound_segmentation_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten envelope kwargs for envelope dispatch.

    Pseudo-envelope methods resolve their feature context from top-level kwargs,
    so nested segmentation_kwargs['envelope_kwargs'] must be merged.
    """
    flattened = dict(bound_segmentation_kwargs)
    nested = flattened.pop("envelope_kwargs", None)
    if isinstance(nested, dict):
        flattened.update(nested)
    return flattened


def _resolve_embedding_config(
    *,
    preset: Optional[str],
    segmentation: Optional[str],
    features: Optional[str],
    pooling: Optional[str],
    segmentation_kwargs: Optional[Dict[str, Any]],
    feature_kwargs: Optional[Dict[str, Any]],
    pooling_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved = resolve_preset(
        preset,
        segmentation=segmentation,
        features=features,
        pooling=pooling,
        segmentation_kwargs=segmentation_kwargs,
        feature_kwargs=feature_kwargs,
        pooling_kwargs=pooling_kwargs,
    )

    seg = _normalize_name(resolved['segmentation']) or 'greedy_cosine'
    feat = _canonical_feature_name(resolved['features']) or 'sylber'
    pool = _normalize_name(resolved['pooling']) or 'mean'

    available_segmenters = set(list_segmenters())
    available_presets = set(list_presets())

    if seg in available_presets:
        raise ValueError(
            f"'{seg}' is a preset name, not a segmentation method. "
            f"Use preset='{seg}' and set segmentation to one of: {', '.join(sorted(available_segmenters))}."
        )

    if seg not in available_segmenters:
        raise ValueError(
            f"Unknown segmentation method '{seg}'. "
            f"Available segmentation methods: {', '.join(sorted(available_segmenters))}. "
            f"Available presets: {', '.join(sorted(available_presets))}."
        )

    canonical_features = {'hubert', 'sylber', 'vg_hubert', 'mfcc', 'melspec', 'rms', 'hilbert', 'lowpass', 'sbs', 'theta'}
    if feat not in canonical_features:
        raise ValueError(
            f"Unknown feature extractor '{feat}'. "
            f"Supported features: hubert, sylber, vg_hubert, mfcc, melspec, rms, hilbert, lowpass, sbs, theta."
        )

    return {
        'preset': preset,
        'segmentation': seg,
        'features': feat,
        'pooling': pool,
        'segmentation_kwargs': resolved['segmentation_kwargs'],
        'feature_kwargs': resolved['feature_kwargs'],
        'pooling_kwargs': resolved['pooling_kwargs'],
    }


class EmbeddingPipeline:
    """Class-first orchestrator for embedding workflows.

    The functional API below delegates to this class.
    """

    def __init__(
        self,
        segmentation: Optional[str] = None,
        features: Optional[str] = None,
        pooling: Optional[str] = None,
        preset: Optional[str] = None,
        sr: int = 16000,
        layer: Optional[int] = None,
        device: str = 'auto',
        segmentation_kwargs: Optional[Dict[str, Any]] = None,
        feature_kwargs: Optional[Dict[str, Any]] = None,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.segmentation = segmentation
        self.features = features
        self.pooling = pooling
        self.preset = preset
        self.sr = sr
        self.layer = layer
        self.device = device
        self.segmentation_kwargs = segmentation_kwargs or {}
        self.feature_kwargs = feature_kwargs or {}
        self.pooling_kwargs = pooling_kwargs or {}

    def embed_audio(self, audio_path: str, return_metadata: bool = True):
        return _embed_audio_impl(
            audio_path=audio_path,
            segmentation=self.segmentation,
            features=self.features,
            pooling=self.pooling,
            preset=self.preset,
            sr=self.sr,
            layer=self.layer,
            device=self.device,
            segmentation_kwargs=self.segmentation_kwargs,
            feature_kwargs=self.feature_kwargs,
            pooling_kwargs=self.pooling_kwargs,
            return_metadata=return_metadata,
        )

    def embed_corpus(
        self,
        audio_files: Union[List[str], List[Path]],
        n_jobs: int = 1,
        verbose: bool = True,
        fail_on_error: bool = False,
    ):
        return _embed_corpus_impl(
            audio_files=audio_files,
            segmentation=self.segmentation,
            features=self.features,
            pooling=self.pooling,
            preset=self.preset,
            sr=self.sr,
            layer=self.layer,
            device=self.device,
            segmentation_kwargs=self.segmentation_kwargs,
            feature_kwargs=self.feature_kwargs,
            pooling_kwargs=self.pooling_kwargs,
            n_jobs=n_jobs,
            verbose=verbose,
            fail_on_error=fail_on_error,
        )

    def embed_corpus_to_storage(
        self,
        audio_files: Union[List[str], List[Path]],
        output_dir: Union[str, Path],
        manifest_name: str = 'embedding_manifest.csv',
        verbose: bool = True,
        fail_on_error: bool = False,
    ) -> Dict[str, Any]:
        return _embed_corpus_to_storage_impl(
            audio_files=audio_files,
            output_dir=output_dir,
            manifest_name=manifest_name,
            segmentation=self.segmentation,
            features=self.features,
            pooling=self.pooling,
            preset=self.preset,
            sr=self.sr,
            layer=self.layer,
            device=self.device,
            segmentation_kwargs=self.segmentation_kwargs,
            feature_kwargs=self.feature_kwargs,
            pooling_kwargs=self.pooling_kwargs,
            verbose=verbose,
            fail_on_error=fail_on_error,
        )


def _embed_audio_impl(
    audio_path: str,
    segmentation: Optional[str] = None,
    features: Optional[str] = None,
    pooling: Optional[str] = None,
    preset: Optional[str] = None,
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    feature_kwargs: Optional[Dict[str, Any]] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
    feature_extractor: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    cfg = _resolve_embedding_config(
        preset=preset,
        segmentation=segmentation,
        features=features,
        pooling=pooling,
        segmentation_kwargs=segmentation_kwargs,
        feature_kwargs=feature_kwargs,
        pooling_kwargs=pooling_kwargs,
    )
    segmentation = cfg['segmentation']
    features = cfg['features']
    pooling = cfg['pooling']
    segmentation_kwargs = cfg['segmentation_kwargs']
    feature_kwargs = cfg['feature_kwargs']
    pooling_kwargs = cfg['pooling_kwargs']

    extractor_device = None if device == 'auto' else device

    # Step 1: Load audio
    audio, actual_sr = load_audio(audio_path, samplerate=sr)
    duration = len(audio) / actual_sr

    bound_segmentation_kwargs = _bind_segmentation_to_selected_features(
        segmentation=segmentation,
        features=features,
        layer=layer,
        device=extractor_device,
        feature_kwargs=feature_kwargs,
        segmentation_kwargs=segmentation_kwargs,
        feature_extractor=feature_extractor,
    )

    if segmentation == 'peakdetect':
        envelope_method = _resolve_peakdetect_envelope_method(
            features=features,
            segmentation_kwargs=bound_segmentation_kwargs,
        )
        peakdetect_envelope_kwargs = _flatten_peakdetect_envelope_kwargs(bound_segmentation_kwargs)
        peakdetect_envelope_kwargs.setdefault("feature_type", features)
        if extractor_device is not None:
            peakdetect_envelope_kwargs.setdefault("device", extractor_device)
        if layer is not None:
            peakdetect_envelope_kwargs.setdefault("layer", layer)
        if feature_kwargs:
            peakdetect_envelope_kwargs.setdefault("feature_kwargs", dict(feature_kwargs))
        if feature_extractor is not None:
            peakdetect_envelope_kwargs.setdefault("feature_extractor", feature_extractor)
        peakdetect_kwargs = {
            k: v
            for k, v in bound_segmentation_kwargs.items()
            if k not in {
                "envelope_method",
                "envelope_kwargs",
                "envelope_computer",
                "feature_type",
                "feature_kwargs",
                "feature_extractor",
                "layer",
                "device",
            }
        }
        peakdetect_kwargs["envelope_method"] = envelope_method
        peakdetect_kwargs["envelope_kwargs"] = peakdetect_envelope_kwargs
        syllables, _, _ = segment_audio_pipeline(
            audio_file=audio_path,
            samplerate=actual_sr,
            method='peakdetect',
            segmentation_kwargs=peakdetect_kwargs,
            return_envelope=False,
        )
    else:
        syllables, _, _ = segment_audio_pipeline(
            audio_file=audio_path,
            samplerate=actual_sr,
            method=segmentation,
            segmentation_kwargs=bound_segmentation_kwargs,
        )

    frame_features, times = extract_features(
        audio,
        sr=actual_sr,
        method=features,
        layer=layer,
        device=extractor_device,
        return_times=True,
        feature_extractor=feature_extractor,
        **feature_kwargs
    )

    syllables = _normalize_syllable_tuples(syllables, pooling=pooling)

    # Step 4: Pool frames into syllable embeddings
    num_frames = frame_features.shape[0]
    if num_frames > 1:
        avg_frame_time = times[-1] / (num_frames - 1)
        hop_length = int(avg_frame_time * actual_sr)
    else:
        hop_length = 160

    embeddings = pool_syllables(
        frame_features,
        syllables,
        sr=actual_sr,
        method=pooling,
        hop_length=hop_length,
        **pooling_kwargs
    )

    if return_metadata:
        metadata = {
            'boundaries': [(start, end) for start, _, end in syllables],
            'peaks': [peak for _, peak, _ in syllables],
            'num_syllables': len(syllables),
            'audio_path': str(Path(audio_path).resolve()),
            'duration': duration,
            'sample_rate': actual_sr,
            'segmentation_method': segmentation,
            'features': features,
            'pooling': pooling,
            'preset': preset,
            'layer': layer,
            'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'fps': num_frames / duration if duration > 0 else 0,
            'hop_length': hop_length,
            'created_at': datetime.now().isoformat(),
        }
        return embeddings, metadata

    return embeddings, None


def embed_audio(
    audio_path: str,
    segmentation: Optional[str] = None,
    features: Optional[str] = None,
    pooling: Optional[str] = None,
    preset: Optional[str] = None,
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    feature_kwargs: Optional[Dict[str, Any]] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Extract syllable embeddings from audio file.
    
    Complete pipeline: load audio → segment → extract features → pool embeddings
    
    Args:
        audio_path: Path to audio file
        segmentation: Segmentation method (any from Methods 1-11)
            - 'peakdetect': Classical envelope-based (default for envelope methods)
            - 'sylber': Sylber end-to-end model
            - 'vg_hubert': VG-HuBERT model (future)
            - etc.
        features: Feature extraction method
            - 'sylber': Sylber model (768-dim, ~50 fps)
            - 'vg_hubert': VG-HuBERT model (768-dim, ~50 fps)
                          Requires feature_kwargs={'model_path': '/path/to/vg-hubert_3'}
            - 'mfcc': Mel-frequency cepstral coefficients (13-dim, ~100 fps)
                      Use feature_kwargs={'include_delta': True, 'include_delta_delta': True}
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
        feature_kwargs: Additional arguments for feature extraction
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
        ...     features='sylber',
        ...     pooling='mean'
        ... )
        >>> print(embeddings.shape)  # (num_syllables, 768)
        >>> print(meta['num_syllables'])  # e.g., 15
    """
    return _embed_audio_impl(
        audio_path=audio_path,
        segmentation=segmentation,
        features=features,
        pooling=pooling,
        preset=preset,
        sr=sr,
        layer=layer,
        device=device,
        segmentation_kwargs=segmentation_kwargs,
        feature_kwargs=feature_kwargs,
        pooling_kwargs=pooling_kwargs,
        return_metadata=return_metadata,
    )


def _embed_corpus_impl(
    audio_files: Union[List[str], List[Path]],
    segmentation: Optional[str] = None,
    features: Optional[str] = None,
    pooling: Optional[str] = None,
    preset: Optional[str] = None,
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    feature_kwargs: Optional[Dict[str, Any]] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    verbose: bool = True,
    fail_on_error: bool = False
) -> List[Dict[str, Any]]:
    audio_files = [str(Path(f)) for f in audio_files]

    cfg = _resolve_embedding_config(
        preset=preset,
        segmentation=segmentation,
        features=features,
        pooling=pooling,
        segmentation_kwargs=segmentation_kwargs,
        feature_kwargs=feature_kwargs,
        pooling_kwargs=pooling_kwargs,
    )
    segmentation = cfg['segmentation']
    features = cfg['features']
    pooling = cfg['pooling']
    segmentation_kwargs = cfg['segmentation_kwargs']
    feature_kwargs = cfg['feature_kwargs']
    pooling_kwargs = cfg['pooling_kwargs']

    shared_extractor = None
    if n_jobs == 1:
        extractor_kwargs: Dict[str, Any] = dict(feature_kwargs or {})
        if layer is not None:
            if features == 'sylber':
                extractor_kwargs.setdefault('encoding_layer', layer)
            elif features in {'hubert', 'vg_hubert'}:
                extractor_kwargs.setdefault('layer', layer)
        extractor_device = None if device == 'auto' else device
        if extractor_device is not None and features in {'hubert', 'sylber', 'vg_hubert'}:
            extractor_kwargs.setdefault('device', extractor_device)
        try:
            shared_extractor = get_extractor(features, **extractor_kwargs)
        except Exception:
            shared_extractor = None

    def process_single_file(audio_path: str) -> Dict[str, Any]:
        result = {
            'audio_path': audio_path,
            'embeddings': None,
            'metadata': None,
            'success': False,
            'error': None,
        }

        try:
            embeddings, metadata = _embed_audio_impl(
                audio_path=audio_path,
                segmentation=segmentation,
                features=features,
                pooling=pooling,
                preset=preset,
                sr=sr,
                layer=layer,
                device=device,
                segmentation_kwargs=segmentation_kwargs,
                feature_kwargs=feature_kwargs,
                pooling_kwargs=pooling_kwargs,
                return_metadata=True,
                feature_extractor=shared_extractor,
            )
            result['embeddings'] = embeddings
            result['metadata'] = metadata
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)
            if fail_on_error:
                raise
            if verbose:
                warnings.warn(f"Failed to process {audio_path}: {e}")

        return result

    try:
        if n_jobs == 1:
            if verbose:
                results = [process_single_file(f) for f in tqdm(audio_files, desc="Processing audio files")]
            else:
                results = [process_single_file(f) for f in audio_files]
        else:
            results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(process_single_file)(f) for f in audio_files
            )
    finally:
        if shared_extractor is not None and hasattr(shared_extractor, 'release'):
            shared_extractor.release()
        gc.collect()

    if verbose:
        n_success = sum(r['success'] for r in results)
        n_total = len(results)
        print(f"\nProcessed {n_success}/{n_total} files successfully")
        if n_success < n_total:
            print(f"Failed: {n_total - n_success} files")
            if n_success == 0:
                first_errors = [r.get('error') for r in results if r.get('error')]
                if first_errors:
                    unique_errors = []
                    for msg in first_errors:
                        if msg not in unique_errors:
                            unique_errors.append(msg)
                        if len(unique_errors) >= 3:
                            break
                    print("No files produced embeddings. Example errors:")
                    for i, msg in enumerate(unique_errors, 1):
                        print(f"  {i}. {msg}")

    return results


def embed_corpus(
    audio_files: Union[List[str], List[Path]],
    segmentation: Optional[str] = None,
    features: Optional[str] = None,
    pooling: Optional[str] = None,
    preset: Optional[str] = None,
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    feature_kwargs: Optional[Dict[str, Any]] = None,
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
        features: Feature extraction method (see embed_audio)
        pooling: Pooling method (see embed_audio)
        sr: Target sample rate (default: 16000)
        layer: Layer index for neural models
        device: Device for neural models: 'auto', 'cuda', 'cpu'
        segmentation_kwargs: Additional arguments for segmentation
        feature_kwargs: Additional arguments for feature extraction
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
        ...     features='mfcc',
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
    return _embed_corpus_impl(
        audio_files=audio_files,
        segmentation=segmentation,
        features=features,
        pooling=pooling,
        preset=preset,
        sr=sr,
        layer=layer,
        device=device,
        segmentation_kwargs=segmentation_kwargs,
        feature_kwargs=feature_kwargs,
        pooling_kwargs=pooling_kwargs,
        n_jobs=n_jobs,
        verbose=verbose,
        fail_on_error=fail_on_error,
    )


def _embed_corpus_to_storage_impl(
    audio_files: Union[List[str], List[Path]],
    output_dir: Union[str, Path],
    manifest_name: str,
    segmentation: Optional[str] = None,
    features: Optional[str] = None,
    pooling: Optional[str] = None,
    preset: Optional[str] = None,
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    feature_kwargs: Optional[Dict[str, Any]] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    fail_on_error: bool = False,
) -> Dict[str, Any]:
    """Embed corpus file-by-file and persist embeddings incrementally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [str(Path(f)) for f in audio_files]

    cfg = _resolve_embedding_config(
        preset=preset,
        segmentation=segmentation,
        features=features,
        pooling=pooling,
        segmentation_kwargs=segmentation_kwargs,
        feature_kwargs=feature_kwargs,
        pooling_kwargs=pooling_kwargs,
    )
    segmentation = cfg['segmentation']
    features = cfg['features']
    pooling = cfg['pooling']
    segmentation_kwargs = cfg['segmentation_kwargs']
    feature_kwargs = cfg['feature_kwargs']
    pooling_kwargs = cfg['pooling_kwargs']

    rows: List[Dict[str, Any]] = []
    iterator = tqdm(audio_files, desc='Embedding to storage') if verbose else audio_files

    shared_extractor = None
    extractor_kwargs: Dict[str, Any] = dict(feature_kwargs or {})
    if layer is not None:
        if features == 'sylber':
            extractor_kwargs.setdefault('encoding_layer', layer)
        elif features in {'hubert', 'vg_hubert'}:
            extractor_kwargs.setdefault('layer', layer)
    extractor_device = None if device == 'auto' else device
    if extractor_device is not None and features in {'hubert', 'sylber', 'vg_hubert'}:
        extractor_kwargs.setdefault('device', extractor_device)
    try:
        shared_extractor = get_extractor(features, **extractor_kwargs)
    except Exception:
        shared_extractor = None

    try:
        for idx, audio_path in enumerate(iterator):
            row: Dict[str, Any] = {
            'file_id': idx,
            'audio_path': audio_path,
            'embedding_path': '',
            'num_rows': 0,
            'embedding_dim': 0,
            'success': False,
            'error': '',
            'segmentation': segmentation,
            'features': features,
            'pooling': pooling,
            'preset': preset or '',
            }
            try:
                embeddings, metadata = _embed_audio_impl(
                audio_path=audio_path,
                segmentation=segmentation,
                features=features,
                pooling=pooling,
                preset=preset,
                sr=sr,
                layer=layer,
                device=device,
                segmentation_kwargs=segmentation_kwargs,
                feature_kwargs=feature_kwargs,
                pooling_kwargs=pooling_kwargs,
                return_metadata=True,
                feature_extractor=shared_extractor,
                )

                out_path = embeddings_dir / f"{idx:07d}.npz"
                segment_ids = np.arange(int(embeddings.shape[0]), dtype=np.int64)
                file_ids = np.full(int(embeddings.shape[0]), idx, dtype=np.int64)
                np.savez_compressed(
                    out_path,
                    embeddings=embeddings,
                    metadata=json.dumps(metadata),
                    segment_ids=segment_ids,
                    file_id=np.array(idx, dtype=np.int64),
                    file_ids=file_ids,
                )

                row['embedding_path'] = str(out_path)
                row['num_rows'] = int(embeddings.shape[0])
                row['embedding_dim'] = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
                row['success'] = True
            except Exception as e:
                row['error'] = str(e)
                if fail_on_error:
                    raise
                if verbose:
                    warnings.warn(f"Failed to process {audio_path}: {e}")

            rows.append(row)
            gc.collect()
    finally:
        if shared_extractor is not None and hasattr(shared_extractor, 'release'):
            shared_extractor.release()
        gc.collect()

    manifest_path = output_dir / manifest_name
    write_embedding_manifest(rows, manifest_path)

    if verbose:
        n_success = sum(1 for r in rows if r['success'])
        print(f"\nEmbedded {n_success}/{len(rows)} files to {output_dir}")

    return {
        'manifest_path': str(manifest_path),
        'output_dir': str(output_dir),
        'num_files': len(rows),
        'num_success': sum(1 for r in rows if r['success']),
        'num_failed': sum(1 for r in rows if not r['success']),
    }


def embed_corpus_to_storage(
    audio_files: Union[List[str], List[Path]],
    output_dir: Union[str, Path],
    manifest_name: str = 'embedding_manifest.csv',
    segmentation: Optional[str] = None,
    features: Optional[str] = None,
    pooling: Optional[str] = None,
    preset: Optional[str] = None,
    sr: int = 16000,
    layer: Optional[int] = None,
    device: str = 'auto',
    segmentation_kwargs: Optional[Dict[str, Any]] = None,
    feature_kwargs: Optional[Dict[str, Any]] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    fail_on_error: bool = False,
) -> Dict[str, Any]:
    """Thin wrapper around EmbeddingPipeline.embed_corpus_to_storage()."""
    pipeline = EmbeddingPipeline(
        segmentation=segmentation,
        features=features,
        pooling=pooling,
        preset=preset,
        sr=sr,
        layer=layer,
        device=device,
        segmentation_kwargs=segmentation_kwargs,
        feature_kwargs=feature_kwargs,
        pooling_kwargs=pooling_kwargs,
    )
    return pipeline.embed_corpus_to_storage(
        audio_files=audio_files,
        output_dir=output_dir,
        manifest_name=manifest_name,
        verbose=verbose,
        fail_on_error=fail_on_error,
    )
