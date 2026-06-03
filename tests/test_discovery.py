import numpy as np
import pytest

from findsylls.discovery import DiscoveryPipeline


pytest.importorskip("sklearn")


def test_kmeans_discovery_shapes():
    x = np.random.RandomState(0).randn(30, 6)
    result = DiscoveryPipeline(method="kmeans", model_kwargs={"n_clusters": 3, "random_state": 0}).discover(x)

    assert result.labels.shape == (30,)
    assert result.num_clusters == 3
    assert result.model_name == "kmeans"
    assert result.fit_metrics is not None
    assert result.fit_metrics["status"] == "ok"
    assert result.metadata["fit_metrics"] == result.fit_metrics


def test_agglomerative_discovery_shapes():
    x = np.random.RandomState(1).randn(24, 5)
    result = DiscoveryPipeline(method="agglomerative", model_kwargs={"n_clusters": 4}).discover(x)

    assert result.labels.shape == (24,)
    assert result.num_clusters == 4
    assert result.model_name == "agglomerative"


def test_discovery_pipeline_save_and_load_roundtrip(tmp_path):
    x = np.random.RandomState(2).randn(32, 4)
    pipeline = DiscoveryPipeline(method="kmeans", model_kwargs={"n_clusters": 3, "random_state": 0})
    pipeline.fit(x)

    output_dir = tmp_path / "discovery_artifacts"
    pipeline.save(output_dir)

    loaded = DiscoveryPipeline.load(output_dir)

    assert loaded.method == "kmeans"
    assert loaded.model_kwargs == {"n_clusters": 3, "random_state": 0}
    np.testing.assert_array_equal(loaded.predict(x), pipeline.predict(x))
    assert loaded.fit_metrics == pipeline.fit_metrics


def _toy_two_stage(seed: int = 0):
    """60 samples, 8 dims, 10 well-separated fine clusters (6 samples each)."""
    rng = np.random.RandomState(seed)
    n_fine, per, dim = 10, 6, 8
    labels = np.repeat(np.arange(n_fine), per)
    centers = rng.randn(n_fine, dim) * 5.0
    embeddings = np.repeat(centers, per, axis=0) + rng.randn(n_fine * per, dim)
    return embeddings, labels


def test_collapse_clusters_shapes():
    from findsylls.discovery import collapse_clusters

    embeddings, labels = _toy_two_stage()
    new_labels, centroids, centroid_map = collapse_clusters(embeddings, labels, n_clusters=3)

    assert new_labels.shape == (60,)
    assert centroids.shape == (3, 8)
    assert centroid_map.shape == (10,)
    assert set(new_labels.tolist()) == {0, 1, 2}
    assert set(centroid_map.tolist()).issubset({0, 1, 2})


def test_collapse_clusters_centroid_map_consistency():
    from findsylls.discovery import collapse_clusters

    embeddings, labels = _toy_two_stage()
    new_labels, _, centroid_map = collapse_clusters(embeddings, labels, n_clusters=3)

    # The map is the authoritative source for test-time assignment.
    np.testing.assert_array_equal(centroid_map[labels], new_labels)


def test_collapse_clusters_rejects_too_many():
    from findsylls.discovery import collapse_clusters

    embeddings, labels = _toy_two_stage()
    with pytest.raises(ValueError):
        collapse_clusters(embeddings, labels, n_clusters=10)
