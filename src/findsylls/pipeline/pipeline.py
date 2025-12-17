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
from ..segmentation import get_segmenter, segment_envelope
from ..segmentation.base import EnvelopeBasedSegmenter, End2EndSegmenter
from ..envelope.dispatch import get_amplitude_envelope
from ..evaluation.evaluator import evaluate_segmentation
from .results import flatten_results


def segment_audio(
    audio_file: str,
    samplerate: int = 16000,
    method: Optional[str] = None,
    envelope_fn: Optional[str] = None,
    segment_fn: Optional[str] = None,
    envelope_kwargs: Optional[dict] = None,
    segmentation_kwargs: Optional[dict] = None,
    return_envelope: bool = True
) -> Tuple[List[Tuple[float, float, float]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segment audio file into syllables.
    
    Supports both envelope-based (classical) and end-to-end (neural) methods.
    
    Args:
        audio_file: Path to audio file
        samplerate: Target sample rate
        method: Segmentation method name (new unified API)
                If provided, overrides segment_fn
        envelope_fn: Envelope method (for envelope-based methods, backward compatible)
        segment_fn: Segmentation method (backward compatible, use 'method' instead)
        envelope_kwargs: Parameters for envelope computation
        segmentation_kwargs: Parameters for segmentation
        return_envelope: If True, compute and return envelope for visualization
                         (only for envelope-based methods)
    
    Returns:
        (syllables, envelope, times) tuple
        - syllables: List of (start, nucleus, end) tuples
        - envelope: Computed envelope (None for end-to-end methods)
        - times: Time array (None for end-to-end methods)
    
    Examples:
        # Old API (backward compatible)
        >>> syllables, env, times = segment_audio('test.wav', 
        ...                                        envelope_fn='hilbert',
        ...                                        segment_fn='peaks_and_valleys')
        
        # New API (recommended)
        >>> syllables, _, _ = segment_audio('test.wav', method='sylber')
        >>> syllables, env, times = segment_audio('test.wav', 
        ...                                        method='peaks_and_valleys',
        ...                                        envelope_fn='hilbert')
    """
    if envelope_kwargs is None:
        envelope_kwargs = {}
    if segmentation_kwargs is None:
        segmentation_kwargs = {}
    
    # Load audio
    audio, sr = load_audio(audio_file, samplerate=samplerate)
    
    # Determine which method to use
    if method is None:
        # Backward compatibility: use segment_fn if provided
        method = segment_fn or "peaks_and_valleys"
    
    # Set default envelope method
    if envelope_fn is None:
        envelope_fn = "hilbert"  # Changed default from "sbs" to more common "hilbert"
    
    # Get segmenter
    try:
        segmenter = get_segmenter(method, **segmentation_kwargs)
    except ValueError as e:
        # Fall back to old functional API if method not found in registry
        # This handles the case where user provides a method string before
        # the segmenter class is implemented
        envelope, times = get_amplitude_envelope(audio, sr, method=envelope_fn, **envelope_kwargs)
        syllables = segment_envelope(
            envelope=envelope, times=times, 
            method=method, **segmentation_kwargs
        )
        return syllables, envelope, times
    
    # Route based on segmenter type
    if isinstance(segmenter, End2EndSegmenter):
        # End-to-end method: process audio directly
        syllables = segmenter.segment(audio=audio, sr=sr)
        return syllables, None, None
    
    elif isinstance(segmenter, EnvelopeBasedSegmenter):
        # Envelope-based method
        if return_envelope:
            # Compute envelope for visualization
            envelope, times = get_amplitude_envelope(
                audio, sr, method=envelope_fn, **envelope_kwargs
            )
            # Segment (method will compute its own envelope internally if needed)
            syllables = segmenter.segment(
                audio=audio, sr=sr, 
                envelope_method=envelope_fn
            )
            return syllables, envelope, times
        else:
            # Just segment without returning envelope
            syllables = segmenter.segment(audio=audio, sr=sr)
            return syllables, None, None
    
    else:
        raise ValueError(f"Unknown segmenter type: {type(segmenter)}")

def run_evaluation(
    textgrid_paths: Union[List[str], str],
    wav_paths: Union[List[str], str],
    phone_tier: Optional[int] = 1,
    syllable_tier: Optional[int] = None,
    word_tier: Optional[int] = None,
    tolerance: float = 0.05,
    method: Optional[str] = None,
    envelope_fn: Optional[str] = None,
    segmentation_fn: Optional[str] = None,
    envelope_kwargs: Optional[dict] = None,
    segmentation_kwargs: Optional[dict] = None,
    tg_suffix_to_strip: Optional[str] = None
) -> pd.DataFrame:
    """
    Run batch evaluation on matched TextGrid and audio files.
    
    Args:
        textgrid_paths: Path(s) to TextGrid files (glob pattern or list)
        wav_paths: Path(s) to audio files (glob pattern or list)
        phone_tier: TextGrid tier index for phones (default: 1)
        syllable_tier: TextGrid tier index for syllables (optional)
        word_tier: TextGrid tier index for words (optional)
        tolerance: Time tolerance for boundary matching in seconds (default: 0.05)
        method: Segmentation method name (new unified API)
        envelope_fn: Envelope method (backward compatible)
        segmentation_fn: Segmentation method (backward compatible, use 'method' instead)
        envelope_kwargs: Parameters for envelope computation
        segmentation_kwargs: Parameters for segmentation
        tg_suffix_to_strip: Suffix to strip from TextGrid filenames for matching
    
    Returns:
        DataFrame with flattened evaluation results
    
    Examples:
        # Old API (backward compatible)
        >>> results = run_evaluation(
        ...     'data/**/*.TextGrid', 'data/**/*.wav',
        ...     envelope_fn='hilbert', segmentation_fn='peaks_and_valleys'
        ... )
        
        # New API with Sylber
        >>> results = run_evaluation(
        ...     'data/**/*.TextGrid', 'data/**/*.wav',
        ...     method='sylber'
        ... )
    """
    if envelope_kwargs is None:
        envelope_kwargs = {}
    if segmentation_kwargs is None:
        segmentation_kwargs = {}
    
    # Match TextGrids with audio files
    matched_tg, matched_wav = match_wavs_to_textgrids(
        wav_paths, textgrid_paths, 
        tg_suffix_to_strip=tg_suffix_to_strip
    )
    
    # Determine method name for result tracking
    method_name = method or segmentation_fn or "peaks_and_valleys"
    envelope_name = envelope_fn or "hilbert"
    
    results = []
    for tg_file, wav_file in zip(matched_tg, matched_wav):
        try:
            # Segment audio
            syllables, _, _ = segment_audio(
                str(wav_file),  # Convert Path to string
                method=method,
                envelope_fn=envelope_fn,
                segment_fn=segmentation_fn,
                envelope_kwargs=envelope_kwargs,
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
                phone_tier=phone_tier,
                syllable_tier=syllable_tier,
                word_tier=word_tier,
                tolerance=tolerance
            )
            
            # Add metadata
            eval_result["method"] = method_name
            eval_result["envelope"] = envelope_name
            eval_result["segmentation"] = method_name  # For backward compatibility
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
