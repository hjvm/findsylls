import numpy as np, librosa
from .base import EnvelopeComputer

_DB_FLOOR = 1e-10


def compute_rms_envelope(waveform, sr, **kwargs):
    """RMS envelope, optionally log-compressed to dB relative to a reference.

    kwargs:
        frame_length, hop_length: librosa RMS framing.
        db (bool): if True, return 20*log10(rms / ref) in dB (default False, linear).
        reference ("max" | float): dB reference. "max" -> dB below the loudest
            frame (Xie & Niyogi 2006 "relevant energy": max frame = 0 dB, rest < 0).
    """
    frame_length = kwargs.get("frame_length", 1024)
    hop_length = kwargs.get("hop_length", 256)
    db = kwargs.get("db", False)
    reference = kwargs.get("reference", "max")

    envelope = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=hop_length)

    if db:
        if reference == "max":
            ref = float(envelope.max()) if envelope.size else 0.0
        else:
            ref = float(reference)
        ref = max(ref, _DB_FLOOR)
        envelope = 20.0 * np.log10(np.maximum(envelope, _DB_FLOOR) / ref)

    return envelope, times


class RMSEnvelope(EnvelopeComputer):
    """RMS (root-mean-square) envelope.

    With ``db=True`` and ``reference="max"`` this is Xie & Niyogi (2006)
    "relevant energy" -- log frame energy in dB below the loudest frame -- since
    ``20*log10(rms/rms.max()) == 10*log10(energy/energy.max())``. In that mode the
    output is utterance-dependent and <= 0 dB; the default (``db=False``) is the
    plain linear RMS envelope.
    """

    def __init__(self, frame_length=1024, hop_length=256, db=False, reference="max"):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.db = db
        self.reference = reference

    def compute(self, audio: np.ndarray, sr: int):
        return compute_rms_envelope(
            audio, sr,
            frame_length=self.frame_length, hop_length=self.hop_length,
            db=self.db, reference=self.reference,
        )
