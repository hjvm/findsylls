from typing import List, Optional, Tuple
import numpy as np
from .base import BaseSAD


class SileroVAD(BaseSAD):
    """
    Silero VAD — ~1 MB neural LSTM, language-agnostic, requires torch.

    Lazy-loads the model on first call.  Call `release()` to free GPU memory
    between pipeline phases.

    Args:
        threshold: Speech probability threshold (default: 0.5).
        min_speech_duration_ms: Minimum speech segment to keep, in ms (default: 250).
        min_silence_duration_ms: Minimum silence to split regions, in ms (default: 100).
        pad_duration: Symmetric padding around each kept region, in seconds (default: 0.0).
        device: Torch device string, or None for auto-detect (default: None).

    Reference:
        Silero VAD: https://github.com/snakers4/silero-vad
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        pad_duration: float = 0.0,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.pad_duration = pad_duration
        self.device = device
        self._model = None
        self._get_speech_timestamps = None
        self._torch_device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        dev = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = model.to(dev)
        self._get_speech_timestamps = utils[0]
        self._torch_device = dev

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        self._load()
        import torch

        wav = torch.from_numpy(audio.astype(np.float32))
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
            effective_sr = 16000
        else:
            effective_sr = sr

        wav = wav.to(self._torch_device)
        timestamps = self._get_speech_timestamps(
            wav,
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )

        regions = [(float(t['start']), float(t['end'])) for t in timestamps]

        if self.pad_duration > 0:
            total = len(audio) / sr
            regions = [
                (max(0.0, s - self.pad_duration), min(total, e + self.pad_duration))
                for s, e in regions
            ]

        return regions

    def release(self) -> None:
        """Free model memory."""
        self._model = None
        self._get_speech_timestamps = None
