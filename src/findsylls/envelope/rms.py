import numpy as np, librosa

def compute_rms_envelope(waveform, sr, **kwargs):
    frame_length = kwargs.get("frame_length", 1024)
    hop_length = kwargs.get("hop_length", 256)
    envelope = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=hop_length)
    return envelope, times
