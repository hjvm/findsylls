"""Embedding pooler implementations and registry."""

from .base import BasePooler
from .dispatch import get_pooler, list_poolers, register_pooler
from .max import MaxPooler
from .mean import MeanPooler
from .median import MedianPooler
from .onc import ONCPooler

__all__ = [
    "BasePooler",
    "MeanPooler",
    "MaxPooler",
    "MedianPooler",
    "ONCPooler",
    "get_pooler",
    "list_poolers",
    "register_pooler",
]
