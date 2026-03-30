from .dispatch import get_amplitude_envelope, get_envelope_computer
from .base import EnvelopeComputer
from .rms import RMSEnvelope
from .hilbert import HilbertEnvelope
from .theta import ThetaEnvelope
from .sbs import SBSEnvelope
from .lowpass import LowpassEnvelope
from .feature_coherence import (
    SSMEnvelopeComputer,
    GreedyCosineEnvelope,
    CLSAttentionEnvelope
)

__all__ = [
    "get_amplitude_envelope",  # Deprecated functional API (backward compatibility)
    "get_envelope_computer",   # Factory function for EnvelopeComputer instances
    "EnvelopeComputer",
    "RMSEnvelope",
    "HilbertEnvelope",
    "ThetaEnvelope",
    "SBSEnvelope",
    "LowpassEnvelope",
    "SSMEnvelopeComputer",
    "GreedyCosineEnvelope",
    "CLSAttentionEnvelope",
]
