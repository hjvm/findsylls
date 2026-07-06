from .dispatch import get_amplitude_envelope, get_envelope_computer
from .base import EnvelopeComputer, PseudoEnvelope
from .rms import RMSEnvelope
from .hilbert import HilbertEnvelope
from .theta import ThetaEnvelope
from .gammatone import GammatoneEnvelope
from .sbs import SBSEnvelope
from .lowpass import LowpassEnvelope
from .feature_coherence import (
    SSMEnvelopeComputer,
)
from .cls_attention import CLSAttentionEnvelope
from .local_cosine import LocalCosineEnvelope, GreedyCosineEnvelope
from .mincut import MinCutEnvelope
from .periodicity import PeriodicityEnvelope
from .product import ProductEnvelope, ThresholdGate
from .rhythm import RhythmEnvelope, fit_rhythm_sinusoid

__all__ = [
    "get_amplitude_envelope",  # Functional wrapper over get_envelope_computer
    "get_envelope_computer",   # Factory function for EnvelopeComputer instances
    "EnvelopeComputer",
    "PseudoEnvelope",
    "RMSEnvelope",
    "HilbertEnvelope",
    "ThetaEnvelope",
    "GammatoneEnvelope",
    "SBSEnvelope",
    "LowpassEnvelope",
    "SSMEnvelopeComputer",
    "LocalCosineEnvelope",
    "GreedyCosineEnvelope",  # backward-compatible alias of LocalCosineEnvelope
    "MinCutEnvelope",
    "CLSAttentionEnvelope",
    "PeriodicityEnvelope",
    "ProductEnvelope",
    "ThresholdGate",
    "RhythmEnvelope",
    "fit_rhythm_sinusoid",
]
