"""Dispatch registry for embedding poolers."""

from typing import Dict, List, Type

from .base import BasePooler
from .max import MaxPooler
from .mean import MeanPooler
from .median import MedianPooler
from .onc import ONCPooler


_POOLERS: Dict[str, Type[BasePooler]] = {
    "mean": MeanPooler,
    "max": MaxPooler,
    "median": MedianPooler,
    "onc": ONCPooler,
}


def register_pooler(name: str, pooler_class: Type[BasePooler]) -> None:
    """Register a custom pooling strategy."""
    if not issubclass(pooler_class, BasePooler):
        raise TypeError("pooler_class must inherit from BasePooler")
    _POOLERS[name] = pooler_class


def get_pooler(name: str, **kwargs) -> BasePooler:
    """Create a pooler by name."""
    key = name.lower()
    if key not in _POOLERS:
        available = ", ".join(sorted(_POOLERS))
        raise ValueError(f"Unknown pooler '{name}'. Available: {available}")
    return _POOLERS[key](**kwargs)


def list_poolers() -> List[str]:
    """List registered pooler names."""
    return sorted(_POOLERS)
