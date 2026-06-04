"""
Pipeline functions for syllable segmentation and evaluation.

Provides high-level APIs for:
- Segmenting audio files
- Running batch evaluation on datasets
- Aggregating results

Supports both envelope-based (classical) and end-to-end (neural) methods.
"""

import pandas as pd
from typing import Optional, Union, List, Tuple, TYPE_CHECKING
import numpy as np

from ..audio.utils import load_audio, match_wavs_to_textgrids
from ..segmentation import get_segmenter
from ..segmentation.base import EnvelopeBasedSegmenter, End2EndSegmenter
from ..evaluation.evaluator import evaluate_segmentation
from .results import flatten_results

if TYPE_CHECKING:
    from ..vad.base import BaseSAD


def segment_audio(
    audio_file: str,
    samplerate: int = 16000,
    method: str = "peakdetect",
    segmentation_kwargs: Optional[dict] = None,
    return_envelope: bool = True,
    sad: Optional[Union[str, "BaseSAD"]] = None,
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
    # Load audio once, then delegate to the array-based core.
    audio, sr = load_audio(audio_file, samplerate=samplerate)
    return segment_loaded_audio(
        audio,
        sr,
        method=method,
        segmentation_kwargs=segmentation_kwargs,
        return_envelope=return_envelope,
        sad=sad,
    )


def segment_loaded_audio(
    audio: np.ndarray,
    sr: int,
    method: str = "peakdetect",
    segmentation_kwargs: Optional[dict] = None,
    return_envelope: bool = True,
    sad: Optional[Union[str, "BaseSAD"]] = None,
) -> Tuple[List[Tuple[float, float, float]], Optional[np.ndarray], Optional[np.ndarray]]:
    """Segment a pre-loaded audio array into syllables.

    Core of ``segment_audio`` operating on an in-memory ``(audio, sr)`` pair so
    callers that already hold the waveform (e.g. the embedding pipeline) do not
    reload it from disk.

    ``sad`` is a first-class convenience for restricting segmentation to detected
    speech regions: pass ``'energy'`` / ``'silero'`` or a ``BaseSAD`` instance. It
    is injected into the segmenter constructor (an explicit ``segmentation_kwargs``
    ``'sad'`` entry takes precedence).
    """
    if segmentation_kwargs is None:
        segmentation_kwargs = {}

    # Get segmenter
    segmenter_kwargs = {**segmentation_kwargs}
    if sad is not None:
        segmenter_kwargs.setdefault("sad", sad)
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
            # Compute envelope once for visualization.
            from ..envelope.dispatch import get_amplitude_envelope

            envelope, times = get_amplitude_envelope(audio, sr, method=segmenter_kwargs.get("envelope_method", "hilbert"), **(segmenter_kwargs.get("envelope_kwargs") or {}))
            if getattr(segmenter, "sad", None) is not None:
                # SAD chunking happens only on the audio entry point; the
                # envelope=/times= shortcut bypasses it. When a SAD is configured,
                # segment from audio so speech-region restriction is honored, and
                # still return the full-file envelope for visualization.
                syllables = segmenter.segment(audio=audio, sr=sr)
            else:
                # No SAD: reuse the precomputed envelope to avoid recomputation.
                syllables = segmenter.segment(envelope=envelope, times=times)
            return syllables, envelope, times
        else:
            # Segment from audio (envelope computed internally; honors SAD)
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
    tg_suffix_to_strip: Optional[str] = None,
    sad: Optional[Union[str, "BaseSAD"]] = None,
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
                return_envelope=False,  # Don't need envelope for evaluation
                sad=sad,
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
