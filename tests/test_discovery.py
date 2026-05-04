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
