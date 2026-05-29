from typing import List, Tuple
import numpy as np
from .base import BaseSAD


class EnergyVAD(BaseSAD):
    """
    Energy-threshold VAD using RMS amplitude. No dependencies beyond numpy.

    Computes RMS per frame, normalises to [0,1] relative to the loudest frame,
    then labels frames above `threshold` as speech.  Short speech bursts and
    short silences are filtered by `min_speech_duration` / `min_silence_duration`.

    Args:
        threshold: RMS threshold as a fraction of the file's peak RMS (default: 0.02).
        frame_length: Analysis frame length in seconds (default: 0.025).
        frame_shift: Frame shift in seconds (default: 0.010).
        min_speech_duration: Minimum speech segment to keep, in seconds (default: 0.10).
        min_silence_duration: Silence gaps shorter than this are bridged, in seconds (default: 0.30).
        pad_duration: Symmetric padding added around each kept region, in seconds (default: 0.0).
    """

    def __init__(
        self,
        threshold: float = 0.02,
        frame_length: float = 0.025,
        frame_shift: float = 0.010,
        min_speech_duration: float = 0.10,
        min_silence_duration: float = 0.30,
        pad_duration: float = 0.0,
    ):
        self.threshold = threshold
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.pad_duration = pad_duration

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        frame_len = max(1, int(self.frame_length * sr))
        frame_shift = max(1, int(self.frame_shift * sr))
        n_frames = max(1, (len(audio) - frame_len) // frame_shift + 1)

        rms = np.array([
            np.sqrt(np.mean(audio[i * frame_shift: i * frame_shift + frame_len] ** 2))
            for i in range(n_frames)
        ], dtype=np.float32)

        peak = rms.max()
        if peak > 0:
            rms = rms / peak
        is_speech = rms >= self.threshold

        # Frame mask → time regions
        regions: List[Tuple[float, float]] = []
        in_speech = False
        start_frame = 0
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                in_speech = False
                s = start_frame * frame_shift / sr
                e = i * frame_shift / sr
                if e - s >= self.min_speech_duration:
                    regions.append((s, e))
        if in_speech:
            s = start_frame * frame_shift / sr
            e = len(audio) / sr
            if e - s >= self.min_speech_duration:
                regions.append((s, e))

        # Merge regions whose gap is shorter than min_silence_duration
        if self.min_silence_duration > 0 and len(regions) > 1:
            merged = [regions[0]]
            for s, e in regions[1:]:
                prev_s, prev_e = merged[-1]
                if s - prev_e < self.min_silence_duration:
                    merged[-1] = (prev_s, e)
                else:
                    merged.append((s, e))
            regions = merged

        if self.pad_duration > 0:
            total = len(audio) / sr
            regions = [
                (max(0.0, s - self.pad_duration), min(total, e + self.pad_duration))
                for s, e in regions
            ]

        return regions
