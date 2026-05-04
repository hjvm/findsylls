"""Dispatch registry for discovery models."""

from typing import Dict, List, Type

from .agglomerative import AgglomerativeDiscovery
from .base import BaseDiscoveryModel
from .kmeans import KMeansDiscovery
from .minibatch_kmeans import MiniBatchKMeansDiscovery


_MODELS: Dict[str, Type[BaseDiscoveryModel]] = {
    "kmeans": KMeansDiscovery,
    "minibatch_kmeans": MiniBatchKMeansDiscovery,
    "agglomerative": AgglomerativeDiscovery,
}


def register_discovery_model(name: str, model_class: Type[BaseDiscoveryModel]) -> None:
    """Register a custom discovery model class."""
    if not issubclass(model_class, BaseDiscoveryModel):
        raise TypeError("model_class must inherit from BaseDiscoveryModel")
    _MODELS[name] = model_class


def get_discovery_model(name: str, **kwargs) -> BaseDiscoveryModel:
    """Create discovery model by name."""
    key = name.lower()
    if key not in _MODELS:
        available = ", ".join(sorted(_MODELS))
        raise ValueError(f"Unknown discovery model '{name}'. Available: {available}")
    return _MODELS[key](**kwargs)


def list_discovery_models() -> List[str]:
    """List registered discovery model names."""
    return sorted(_MODELS)
