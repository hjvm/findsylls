import pandas as pd
from typing import Optional, Union, List
from ..audio.utils import load_audio, match_wavs_to_textgrids
from ..segmentation.dispatch import segment_envelope
from ..envelope.dispatch import get_amplitude_envelope
from ..evaluation.evaluator import evaluate_segmentation
from .results import flatten_results

def segment_audio(audio_file: str, samplerate: int = 16000, envelope_fn: str = "sbs", envelope_kwargs: dict | None = None, segment_fn: str = "peaks_and_valleys", segmentation_kwargs: dict | None = None):
    if envelope_kwargs is None:
        envelope_kwargs = {}
    if segmentation_kwargs is None:
        segmentation_kwargs = {}
    audio, sr = load_audio(audio_file, samplerate=samplerate)
    envelope, times = get_amplitude_envelope(audio, sr, method=envelope_fn, **envelope_kwargs)
    syllables = segment_envelope(envelope=envelope, times=times, sr=sr, method=segment_fn, **segmentation_kwargs)
    return syllables, envelope, times

def run_evaluation(textgrid_paths: Union[List[str], str], wav_paths: Union[List[str], str], phone_tier: Optional[Union[int, None]] = 1, syllable_tier: Optional[Union[int, None]] = None, word_tier: Optional[int] = None, tolerance: float = 0.05, envelope_fn: str = "sbs", envelope_kwargs: dict | None = None, segmentation_fn: str = "peaks_and_valleys", segmentation_kwargs: dict | None = None, tg_suffix_to_strip=None):
    matched_tg, matched_wav = match_wavs_to_textgrids(wav_paths, textgrid_paths, tg_suffix_to_strip=tg_suffix_to_strip)
    results = []
    for tg_file, wav_file in zip(matched_tg, matched_wav):
        try:
            syllables, _, _ = segment_audio(wav_file, envelope_fn=envelope_fn, segment_fn=segmentation_fn, envelope_kwargs=envelope_kwargs, segmentation_kwargs=segmentation_kwargs)
            peaks = [p for (_, p, _) in syllables]
            spans = [(s, e) for (s, _, e) in syllables]
            eval_result = evaluate_segmentation(peaks=peaks, spans=spans, textgrid_path=tg_file, phone_tier=phone_tier, syllable_tier=syllable_tier, word_tier=word_tier, tolerance=tolerance)
            eval_result["envelope"] = envelope_fn
            eval_result["segmentation"] = segmentation_fn
        except Exception as e:
            print(f"Error processing {tg_file}: {e}")
            continue
        eval_result["tg_file"] = str(tg_file)
        eval_result["audio_file"] = str(wav_file)
        results.append(eval_result)
    if results:
        return flatten_results(results)
    print("No valid results found. Check your input files and parameters.")
    return pd.DataFrame()
