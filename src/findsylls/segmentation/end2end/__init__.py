"""
End-to-end neural segmentation methods.

This module contains wrappers for end-to-end neural syllable segmentation models.
These models:
- Take raw audio as input
- Process through learned representations (typically transformers)
- Produce syllable boundaries end-to-end
- Often use pre-trained weights

Available methods:
- SylberSegmenter: Self-supervised syllabic distillation (Cho et al. ICLR 2025)
- VGHubertSegmenter: VG-HuBERT + MinCut (Peng et al. 2023)
- SDHubertSegmenter: SD-HuBERT (Cho et al. ICASSP 2024) [TODO]
"""

from .sylber_segmenter import SylberSegmenter
from .vg_hubert_segmenter import VGHubertSegmenter

__all__ = ["SylberSegmenter", "VGHubertSegmenter"]
