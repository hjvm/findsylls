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
        center (bool): librosa framing. True (default) reflect-pads by
            frame_length//2 so frame i is centered at i*hop. False left-aligns
            frames to ``range(0, len-frame_length+1, hop)`` with center-of-frame
            times ``(i*hop + frame_length/2)/sr`` -- the same grid as
            PeriodicityEnvelope, so the two cues can be composed frame-for-frame.
    """
    frame_length = kwargs.get("frame_length", 1024)
    hop_length = kwargs.get("hop_length", 256)
    db = kwargs.get("db", False)
    reference = kwargs.get("reference", "max")
    center = kwargs.get("center", True)

    envelope = librosa.feature.rms(
        y=waveform, frame_length=frame_length, hop_length=hop_length, center=center
    )[0]
    if center:
        times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=hop_length)
    else:
        # left-aligned frames start at i*hop; report each frame's true centre
        times = (np.arange(len(envelope)) * hop_length + frame_length / 2.0) / sr

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

    ``center=False`` left-aligns frames onto the same grid as
    ``PeriodicityEnvelope`` (same ``frame_length``/``hop_length``), so the two
    can be composed frame-for-frame -- used by the Xie detector, whose energy
    (``log gamma(0)``) shares periodicity's frames.
    """

    def __init__(self, frame_length=1024, hop_length=256, db=False, reference="max", center=True):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.db = db
        self.reference = reference
        self.center = center

    def compute(self, audio: np.ndarray, sr: int):
        return compute_rms_envelope(
            audio, sr,
            frame_length=self.frame_length, hop_length=self.hop_length,
            db=self.db, reference=self.reference, center=self.center,
        )
