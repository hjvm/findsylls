"""Discovery module for clustering syllable embeddings."""

from .agglomerative import AgglomerativeDiscovery
from .base import BaseDiscoveryModel
from .dispatch import (
    get_discovery_model,
    list_discovery_models,
    register_discovery_model,
)
from .kmeans import KMeansDiscovery
from .minibatch_kmeans import MiniBatchKMeansDiscovery
from .pipeline import DiscoveryPipeline
from .storage import (
    compute_intrinsic_fit_metrics,
    load_discovery_pipeline,
    save_discovery_pipeline,
)
from .types import DiscoveryResult

__all__ = [
    "BaseDiscoveryModel",
    "KMeansDiscovery",
    "MiniBatchKMeansDiscovery",
    "AgglomerativeDiscovery",
    "DiscoveryPipeline",
    "DiscoveryResult",
    "compute_intrinsic_fit_metrics",
    "save_discovery_pipeline",
    "load_discovery_pipeline",
    "get_discovery_model",
    "list_discovery_models",
    "register_discovery_model",
]
