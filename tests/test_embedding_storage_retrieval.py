from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from findsylls.embedding.storage import load_embedding_rows


def _write_manifest(tmp_path: Path, rows):
    manifest_path = tmp_path / "embedding_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_load_embedding_rows_by_key(tmp_path):
    shard0 = tmp_path / "0000000.npz"
    shard1 = tmp_path / "0000001.npz"

    np.savez_compressed(
        shard0,
        embeddings=np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]),
        file_id=np.array(0, dtype=np.int64),
        segment_ids=np.array([0, 1, 2], dtype=np.int64),
    )
    np.savez_compressed(
        shard1,
        embeddings=np.array([[10.0, 10.1], [20.0, 20.1]]),
        file_id=np.array(1, dtype=np.int64),
        segment_ids=np.array([0, 1], dtype=np.int64),
    )

    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "file_id": 0,
                "audio_path": "a.wav",
                "embedding_path": str(shard0),
                "num_rows": 3,
                "embedding_dim": 2,
                "success": True,
                "error": "",
                "segmentation": "peakdetect",
                "features": "mfcc",
                "pooling": "mean",
            },
            {
                "file_id": 1,
                "audio_path": "b.wav",
                "embedding_path": str(shard1),
                "num_rows": 2,
                "embedding_dim": 2,
                "success": True,
                "error": "",
                "segmentation": "peakdetect",
                "features": "mfcc",
                "pooling": "mean",
            },
        ],
    )

    rows = load_embedding_rows(manifest_path, [(1, 1), (0, 2)])

    assert len(rows) == 2
    assert rows[0]["file_id"] == 1
    assert rows[0]["segment_id"] == 1
    assert np.allclose(rows[0]["embedding"], np.array([20.0, 20.1]))
    assert rows[1]["file_id"] == 0
    assert rows[1]["segment_id"] == 2
    assert np.allclose(rows[1]["embedding"], np.array([3.0, 3.1]))


def test_load_embedding_rows_errors_on_missing_key(tmp_path):
    shard0 = tmp_path / "0000000.npz"
    np.savez_compressed(
        shard0,
        embeddings=np.array([[1.0, 1.1]]),
        file_id=np.array(0, dtype=np.int64),
        segment_ids=np.array([0], dtype=np.int64),
    )

    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "file_id": 0,
                "audio_path": "a.wav",
                "embedding_path": str(shard0),
                "num_rows": 1,
                "embedding_dim": 2,
                "success": True,
                "error": "",
                "segmentation": "peakdetect",
                "features": "mfcc",
                "pooling": "mean",
            }
        ],
    )

    with pytest.raises(KeyError):
        load_embedding_rows(manifest_path, [(0, 99)])


def test_load_embedding_rows_legacy_shard_is_rejected(tmp_path):
    shard0 = tmp_path / "0000000.npz"
    np.savez_compressed(
        shard0,
        embeddings=np.array([[1.0, 1.1], [2.0, 2.1]]),
    )

    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "file_id": 0,
                "audio_path": "a.wav",
                "embedding_path": str(shard0),
                "num_rows": 2,
                "embedding_dim": 2,
                "success": True,
                "error": "",
                "segmentation": "peakdetect",
                "features": "mfcc",
                "pooling": "mean",
            }
        ],
    )

    with pytest.raises(ValueError):
        load_embedding_rows(manifest_path, [(0, 1)])
