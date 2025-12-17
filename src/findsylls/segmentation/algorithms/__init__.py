"""Low-level segmentation algorithms.

This module contains reusable graph-based algorithms used by end-to-end segmenters:

- mincut: Graph-based partitioning (used by VG-HuBERT, SD-HuBERT)

Note: For envelope-based peak detection, use segment_peakdetect() from parent module.
"""

from .mincut import min_cut

__all__ = ["min_cut"]
