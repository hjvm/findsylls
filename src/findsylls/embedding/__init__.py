"""
Syllable Embedding Pipeline

Extract per-syllable embeddings from audio for downstream tasks like
clustering, classification, and cross-lingual phonetic analysis.

Two orthogonal dimensions:
1. Features (feature extraction): Sylber, VG-HuBERT, MFCC, etc.
2. Pooling (frame→syllable): mean, ONC template, max, etc.

Usage:
    >>> from findsylls.embedding import embed_audio, embed_corpus
    >>> # Single file
    >>> embeddings, metadata = embed_audio(
    ...     'audio.wav',
    ...     segmentation='sylber',
    ...     features='sylber',
    ...     pooling='mean'
    ... )
    >>> # Multiple files
    >>> results = embed_corpus(
    ...     ['audio1.wav', 'audio2.wav'],
    ...     features='mfcc',
    ...     n_jobs=4
    ... )
"""

from .pipeline import EmbeddingPipeline, embed_audio, embed_corpus, embed_corpus_to_storage
from .extractors import extract_features
from ..presets import list_presets, get_preset, resolve_preset
from .poolers import (
    BasePooler,
    MeanPooler,
    MaxPooler,
    MedianPooler,
    ONCPooler,
    get_pooler,
    list_poolers,
)

__all__ = [
    'EmbeddingPipeline',
    'embed_audio',
    'embed_corpus',
    'embed_corpus_to_storage',
    'extract_features',
    'list_presets',
    'get_preset',
    'resolve_preset',
    'BasePooler',
    'MeanPooler',
    'MaxPooler',
    'MedianPooler',
    'ONCPooler',
    'get_pooler',
    'list_poolers',
]
