"""
Pipeline functions for syllable segmentation and evaluation.

Provides high-level APIs for:
- Segmenting audio files
- Running batch evaluation on datasets
- Aggregating results

Supports both envelope-based (classical) and end-to-end (neural) methods.
"""

import pandas as pd
from typing import Optional, Union, List, Tuple
import numpy as np

from ..audio.utils import load_audio, match_wavs_to_textgrids
from ..segmentation import get_segmenter
from ..segmentation.base import EnvelopeBasedSegmenter, End2EndSegmenter
from ..evaluation.evaluator import evaluate_segmentation
from .results import flatten_results


def segment_audio(
    audio_file: str,
    samplerate: int = 16000,
    method: str = "peakdetect",
    segmentation_kwargs: Optional[dict] = None,
    return_envelope: bool = True
) -> Tuple[List[Tuple[float, float, float]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segment audio file into syllables.
    
    Supports both envelope-based (classical) and end-to-end (neural) methods.
    
    Args:
        audio_file: Path to audio file
        samplerate: Target sample rate
        method: Segmentation method name
        segmentation_kwargs: Parameters for segmentation
        return_envelope: If True, compute and return envelope for visualization
                         (only for envelope-based methods)
    
    Returns:
        (syllables, envelope, times) tuple
        - syllables: List of (start, nucleus, end) tuples
        - envelope: Computed envelope (None for end-to-end methods)
        - times: Time array (None for end-to-end methods)
    
    Examples:
        # Canonical API
        >>> syllables, _, _ = segment_audio('test.wav', method='sylber')
        >>> syllables, env, times = segment_audio('test.wav',
        ...                                        method='peakdetect',
        ...                                        segmentation_kwargs={'envelope_method': 'hilbert'})
    """
    if segmentation_kwargs is None:
        segmentation_kwargs = {}
    
    # Load audio
    audio, sr = load_audio(audio_file, samplerate=samplerate)
    
    # Get segmenter
    segmenter_kwargs = {**segmentation_kwargs}
    if method == "peakdetect":
        envelope_method = segmenter_kwargs.get("envelope_method", "hilbert")
        segmenter_kwargs["envelope_method"] = envelope_method
    segmenter = get_segmenter(method, **segmenter_kwargs)
    
    # Route based on segmenter type
    if isinstance(segmenter, End2EndSegmenter):
        # End-to-end method: process audio directly (Sylber, VG-HuBERT with native segmentation)
        syllables = segmenter.segment(audio=audio, sr=sr)
        return syllables, None, None
    
    elif isinstance(segmenter, EnvelopeBasedSegmenter):
        # Envelope-based method (classical signal processing or peak detection)
        if return_envelope:
            # Compute envelope once for both visualization AND segmentation
            from ..envelope.dispatch import get_amplitude_envelope

            envelope, times = get_amplitude_envelope(audio, sr, method=segmenter_kwargs.get("envelope_method", "hilbert"), **(segmenter_kwargs.get("envelope_kwargs") or {}))
            # Pass pre-computed envelope to avoid recomputation
            syllables = segmenter.segment(envelope=envelope, times=times)
            return syllables, envelope, times
        else:
            # Segment from audio (envelope computed internally)
            syllables = segmenter.segment(audio=audio, sr=sr)
            return syllables, None, None
    
    else:
        raise ValueError(f"Unknown segmenter type: {type(segmenter)}")

def run_evaluation(
    textgrid_paths: Union[List[str], str],
    wav_paths: Union[List[str], str],
    tiers: Optional[dict] = None,
    tolerance: float = 0.05,
    method: str = "peakdetect",
    segmentation_kwargs: Optional[dict] = None,
    tg_suffix_to_strip: Optional[str] = None
) -> pd.DataFrame:
    """
    Run batch evaluation on matched TextGrid and audio files.
    
    Args:
        textgrid_paths: Path(s) to TextGrid files (glob pattern or list)
        wav_paths: Path(s) to audio files (glob pattern or list)
        tiers: Mapping of tier names to indices, e.g. {'phone': 2, 'syllable': 1, 'word': 0}
        tolerance: Time tolerance for boundary matching in seconds (default: 0.05)
        method: Segmentation method name
        segmentation_kwargs: Parameters for segmentation
        tg_suffix_to_strip: Suffix to strip from TextGrid filenames for matching
    
    Returns:
        DataFrame with flattened evaluation results
    
    Examples:
        # Canonical API
        >>> results = run_evaluation(
        ...     'data/**/*.TextGrid', 'data/**/*.wav',
        ...     method='sylber'
        ... )
    """
    if segmentation_kwargs is None:
        segmentation_kwargs = {}
    
    # Match TextGrids with audio files
    matched_tg, matched_wav = match_wavs_to_textgrids(
        wav_paths, textgrid_paths, 
        tg_suffix_to_strip=tg_suffix_to_strip
    )
    
    method_name = method
    
    results = []
    for tg_file, wav_file in zip(matched_tg, matched_wav):
        try:
            # Segment audio
            syllables, _, _ = segment_audio(
                str(wav_file),  # Convert Path to string
                method=method,
                segmentation_kwargs=segmentation_kwargs,
                return_envelope=False  # Don't need envelope for evaluation
            )
            
            # Extract peaks and spans
            peaks = [p for (_, p, _) in syllables]
            spans = [(s, e) for (s, _, e) in syllables]
            
            # Evaluate
            eval_result = evaluate_segmentation(
                peaks=peaks,
                spans=spans,
                textgrid_path=str(tg_file),  # Convert Path to string
                tiers=tiers,
                tolerance=tolerance
            )
            
            # Add metadata
            eval_result["method"] = method_name
            eval_result["segmentation"] = method_name
            eval_result["tg_file"] = str(tg_file)
            eval_result["audio_file"] = str(wav_file)
            
            results.append(eval_result)
            
        except Exception as e:
            print(f"Error processing {tg_file}: {e}")
            continue
    
    if results:
        return flatten_results(results)
    
    print("No valid results found. Check your input files and parameters.")
    return pd.DataFrame()


def segment_and_embed_audio(*args, **kwargs):
    """Thin wrapper delegating to package orchestrator."""
    from .orchestrator import FindSyllsOrchestrator

    return FindSyllsOrchestrator().segment_and_embed_audio(*args, **kwargs)


def segment_embed_and_discover(*args, **kwargs):
    """Thin wrapper delegating to package orchestrator."""
    from .orchestrator import FindSyllsOrchestrator

    return FindSyllsOrchestrator().segment_embed_and_discover(*args, **kwargs)


def discover_corpus(*args, **kwargs):
    """Thin wrapper delegating to package orchestrator."""
    from .orchestrator import FindSyllsOrchestrator

    return FindSyllsOrchestrator().discover_corpus(*args, **kwargs)
