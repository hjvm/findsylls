import numpy as np
import pytest

from findsylls.discovery import DiscoveryPipeline
from findsylls.embedding import embed_corpus_to_storage
from findsylls.embedding.storage import load_embedding_manifest, write_embedding_manifest
from findsylls.segmentation import get_segmenter


pytest.importorskip("sklearn")


def test_billauer_alias_removed():
    with pytest.raises(ValueError):
        get_segmenter("billauer")


def test_cls_attention_registered():
    segmenter = get_segmenter("cls_attention")
    assert type(segmenter).__name__ == "CLSAttentionSegmenter"


def test_embed_corpus_to_storage_writes_manifest(tmp_path):
    audio_files = ["test_samples/SP20_117.wav"]

    info = embed_corpus_to_storage(
        audio_files=audio_files,
        output_dir=tmp_path,
        segmentation="peakdetect",
        features="mfcc",
        pooling="mean",
        segmentation_kwargs={"envelope_method": "hilbert"},
        verbose=False,
    )

    manifest_path = info["manifest_path"]
    rows = load_embedding_manifest(manifest_path)

    assert info["num_files"] == 1
    assert len(rows) == 1
    assert rows[0]["audio_path"].endswith("SP20_117.wav")
    assert rows[0]["embedding_path"]


def test_discover_from_storage_minibatch(tmp_path):
    rng = np.random.RandomState(0)
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    e1 = rng.randn(20, 8).astype(np.float32)
    e2 = rng.randn(30, 8).astype(np.float32)

    p1 = emb_dir / "0000000.npz"
    p2 = emb_dir / "0000001.npz"
    np.savez_compressed(p1, embeddings=e1)
    np.savez_compressed(p2, embeddings=e2)

    manifest_path = tmp_path / "embedding_manifest.csv"
    rows = [
        {
            "file_id": 0,
            "audio_path": "a.wav",
            "embedding_path": str(p1),
            "num_rows": 20,
            "embedding_dim": 8,
            "success": True,
            "error": "",
            "segmentation": "peakdetect",
            "features": "mfcc",
            "pooling": "mean",
        },
        {
            "file_id": 1,
            "audio_path": "b.wav",
            "embedding_path": str(p2),
            "num_rows": 30,
            "embedding_dim": 8,
            "success": True,
            "error": "",
            "segmentation": "peakdetect",
            "features": "mfcc",
            "pooling": "mean",
        },
    ]
    write_embedding_manifest(rows, manifest_path)

    pipeline = DiscoveryPipeline(
        method="minibatch_kmeans",
        model_kwargs={"n_clusters": 2, "random_state": 0, "batch_size": 16},
    )
    result = pipeline.discover_from_storage(str(manifest_path), chunk_size=16)

    assert result.num_clusters <= 2
    assert result.labels.shape[0] == 50


def test_discover_from_storage_non_streaming_raises(tmp_path):
    manifest_path = tmp_path / "empty_manifest.csv"
    write_embedding_manifest([], manifest_path)

    pipeline = DiscoveryPipeline(method="agglomerative", model_kwargs={"n_clusters": 2})
    with pytest.raises(NotImplementedError):
        pipeline.discover_from_storage(str(manifest_path), chunk_size=32)
