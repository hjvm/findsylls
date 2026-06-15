"""Backward-compatibility shim.

The local-window cosine-coherence envelope was renamed ``LocalCosineEnvelope``
and moved to ``envelope/local_cosine.py``. This module preserves the old import
path ``findsylls.envelope.greedy_cosine.GreedyCosineEnvelope``.
"""

from .local_cosine import GreedyCosineEnvelope, LocalCosineEnvelope

__all__ = ['LocalCosineEnvelope', 'GreedyCosineEnvelope']
