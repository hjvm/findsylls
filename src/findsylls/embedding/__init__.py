"""
Syllable Embedding Pipeline

Extract per-syllable embeddings from audio for downstream tasks like
clustering, classification, and cross-lingual phonetic analysis.

Two orthogonal dimensions:
1. Embedder (feature extraction): Sylber, VG-HuBERT, MFCC, etc.
2. Pooling (frame→syllable): mean, ONC template, max, etc.

Usage:
    >>> from findsylls.embedding import embed_audio
    >>> embeddings, metadata = embed_audio(
    ...     'audio.wav',
    ...     segmentation='sylber',
    ...     embedder='sylber',
    ...     pooling='mean'
    ... )
"""

from .pipeline import embed_audio
from .extractors import extract_features
from .pooling import pool_syllables

__all__ = [
    'embed_audio',
    'extract_features',
    'pool_syllables',
]
