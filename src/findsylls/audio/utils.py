"""Audio loading and filename matching utilities.

These functions are used by the higher‑level pipeline:
 - load_audio: unified loading (torchaudio preferred; falls back to soundfile/librosa)
 - match_wavs_to_textgrids: fuzzy pair matching between wav/flac files and TextGrid annotations
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import numpy as np

try:
    import torchaudio  # type: ignore
    _HAVE_TORCHAUDIO = True
except Exception:  # pragma: no cover
    torchaudio = None  # type: ignore
    _HAVE_TORCHAUDIO = False

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def load_audio(audio_file: str | Path, samplerate: int = 16000):
    """Load audio as mono float32 waveform and return (audio, sr).

    Preference order:
      1. torchaudio (fast, handles many formats)
      2. soundfile
      3. librosa
    """
    path = Path(audio_file)
    if not path.exists():  # pragma: no cover
        raise FileNotFoundError(path)

    if _HAVE_TORCHAUDIO:
        try:
            wav, sr = torchaudio.load(str(path))  # type: ignore
            if wav.ndim > 1:
                wav = wav.mean(0, keepdim=False)
            wav = wav.numpy()
        except Exception:
            # Fall back to soundfile/librosa if torchaudio fails (e.g., missing torchcodec)
            wav = None
            sr = None
    else:
        wav = None
        sr = None
    
    if wav is None:
        if sf is not None:
            wav, sr = sf.read(str(path))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        elif librosa is not None:  # pragma: no cover
            wav, sr = librosa.load(str(path), sr=None, mono=True)
        if wav is None or sr is None:
            raise RuntimeError("No backend available to read audio (need torchaudio, soundfile, or librosa)")
    # Resample
    if sr != samplerate:
        if librosa is not None:
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=samplerate)
        else:  # linear
            x_old = np.linspace(0, 1, num=len(wav), endpoint=False)
            x_new = np.linspace(0, 1, num=int(len(wav) * samplerate / sr), endpoint=False)
            wav = np.interp(x_new, x_old, wav).astype(wav.dtype)
        sr = samplerate
    return wav.astype(np.float32), sr


def _collect_files(patterns: Sequence[str] | str, exts: set[str]) -> List[Path]:
    if isinstance(patterns, (str, Path)):
        patterns = [str(patterns)]
    files: List[Path] = []
    for p in patterns:
        for match in Path().glob(p):
            if match.suffix.lower() in exts:
                files.append(match)
    return sorted(files)


def match_wavs_to_textgrids(
    wav_patterns: Sequence[str] | str,
    textgrid_patterns: Sequence[str] | str,
    tg_suffix_to_strip: str | None = None,
) -> Tuple[List[Path], List[Path]]:
    """Attempt to pair wav/flac files with TextGrids using filename heuristics.

    Returns (textgrid_list, wav_list) aligned by index.
    """
    wavs = _collect_files(wav_patterns, AUDIO_EXTS)
    tgs = _collect_files(textgrid_patterns, {".TextGrid", ".textgrid"})

    if not wavs or not tgs:
        return [], []

    def _normalize(name: str) -> str:
        base = Path(name).stem
        if tg_suffix_to_strip and base.endswith(tg_suffix_to_strip):
            base = base[: -len(tg_suffix_to_strip)]
        return re.sub(r"[^A-Za-z0-9]+", "", base).lower()

    wav_map = { _normalize(w.name): w for w in wavs }
    tg_map = { _normalize(t.name): t for t in tgs }

    matched_tg: List[Path] = []
    matched_wav: List[Path] = []
    for key, w in wav_map.items():
        if key in tg_map:
            matched_wav.append(w)
            matched_tg.append(tg_map[key])
    return matched_tg, matched_wav

__all__ = ["load_audio", "match_wavs_to_textgrids"]
