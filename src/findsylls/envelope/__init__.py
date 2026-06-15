from .dispatch import get_amplitude_envelope, get_envelope_computer
from .base import EnvelopeComputer, PseudoEnvelope
from .rms import RMSEnvelope
from .hilbert import HilbertEnvelope
from .theta import ThetaEnvelope
from .sbs import SBSEnvelope
from .lowpass import LowpassEnvelope
from .feature_coherence import (
    SSMEnvelopeComputer,
)
from .cls_attention import CLSAttentionEnvelope
from .local_cosine import LocalCosineEnvelope, GreedyCosineEnvelope
from .mincut import MinCutEnvelope

__all__ = [
    "get_amplitude_envelope",  # Functional wrapper over get_envelope_computer
    "get_envelope_computer",   # Factory function for EnvelopeComputer instances
    "EnvelopeComputer",
    "PseudoEnvelope",
    "RMSEnvelope",
    "HilbertEnvelope",
    "ThetaEnvelope",
    "SBSEnvelope",
    "LowpassEnvelope",
    "SSMEnvelopeComputer",
    "LocalCosineEnvelope",
    "GreedyCosineEnvelope",  # backward-compatible alias of LocalCosineEnvelope
    "MinCutEnvelope",
    "CLSAttentionEnvelope",
]
