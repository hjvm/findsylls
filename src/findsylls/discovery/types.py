"""Typed containers for discovery outputs."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DiscoveryResult:
    """Result container for discovery outputs."""

    labels: np.ndarray
    num_clusters: int
    model_name: str
    fit_metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
