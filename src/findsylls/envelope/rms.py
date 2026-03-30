import numpy as np, librosa
from .base import EnvelopeComputer

def compute_rms_envelope(waveform, sr, **kwargs):
    frame_length = kwargs.get("frame_length", 1024)
    hop_length = kwargs.get("hop_length", 256)
    envelope = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=hop_length)
    return envelope, times


class RMSEnvelope(EnvelopeComputer):
    """Compute RMS (Root Mean Square) envelope."""
    
    def __init__(self, frame_length=1024, hop_length=256):
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def compute(self, audio: np.ndarray, sr: int):
        return compute_rms_envelope(audio, sr, frame_length=self.frame_length, hop_length=self.hop_length)
