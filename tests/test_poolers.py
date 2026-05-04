import numpy as np

from findsylls.embedding.poolers import get_pooler


def _toy_inputs():
    features = np.random.RandomState(0).randn(100, 8)
    syllables = [(0.00, 0.05, 0.10), (0.15, 0.20, 0.30), (0.35, 0.40, 0.50)]
    return features, syllables


def test_pooler_shapes():
    features, syllables = _toy_inputs()

    mean_emb = get_pooler("mean").pool(features, syllables, fps=100.0)
    max_emb = get_pooler("max").pool(features, syllables, fps=100.0)
    median_emb = get_pooler("median").pool(features, syllables, fps=100.0)
    onc_emb = get_pooler("onc").pool(features, syllables, fps=100.0)

    assert mean_emb.shape == (3, 8)
    assert max_emb.shape == (3, 8)
    assert median_emb.shape == (3, 8)
    assert onc_emb.shape == (3, 24)


def test_empty_pooling_returns_empty_rows():
    features = np.random.RandomState(1).randn(10, 4)
    syllables = []

    assert get_pooler("mean").pool(features, syllables, fps=100.0).shape == (0, 4)
    assert get_pooler("onc").pool(features, syllables, fps=100.0).shape == (0, 12)
