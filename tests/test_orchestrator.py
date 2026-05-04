import json
from pathlib import Path

import numpy as np
import pandas as pd

from findsylls.pipeline.orchestrator import FindSyllsOrchestrator


def test_orchestrator_methods_exist():
    orch = FindSyllsOrchestrator()
    assert hasattr(orch, "segment_audio")
    assert hasattr(orch, "run_evaluation")
    assert hasattr(orch, "segment_and_embed_audio")
    assert hasattr(orch, "segment_embed_and_discover")


def test_orchestrator_discovery_path(monkeypatch):
    orch = FindSyllsOrchestrator()

    def fake_embed(*args, **kwargs):
        return np.random.randn(10, 4), {"ok": True}

    monkeypatch.setattr(orch, "segment_and_embed_audio", fake_embed)

    out = orch.segment_embed_and_discover("dummy.wav", discovery_method="kmeans", discovery_kwargs={"n_clusters": 2, "random_state": 0})
    assert "discovery" in out
    assert out["discovery"].labels.shape == (10,)


def test_orchestrator_corpus_discovery_path(monkeypatch, tmp_path):
    orch = FindSyllsOrchestrator()

    class FakeEmbeddingPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_corpus_to_storage(self, audio_files, output_dir, verbose=True, fail_on_error=False):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            embeddings_dir = output_dir / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            file_specs = [
                (0, "audio0.wav", np.array([[0.0, 0.0], [0.0, 1.0]]), [(0.0, 0.1), (0.1, 0.2)], [0.05, 0.15]),
                (1, "audio1.wav", np.array([[10.0, 10.0], [10.0, 11.0]]), [(0.0, 0.1), (0.1, 0.2)], [0.05, 0.15]),
            ]
            for file_id, audio_path, embeddings, boundaries, peaks in file_specs:
                embedding_path = embeddings_dir / f"{file_id:07d}.npz"
                metadata = {
                    "boundaries": boundaries,
                    "peaks": peaks,
                    "segmentation_method": "peakdetect",
                    "features": "mfcc",
                    "pooling": "mean",
                }
                np.savez_compressed(embedding_path, embeddings=embeddings, metadata=json.dumps(metadata))
                rows.append(
                    {
                        "file_id": file_id,
                        "audio_path": audio_path,
                        "embedding_path": str(embedding_path),
                        "num_rows": int(embeddings.shape[0]),
                        "embedding_dim": int(embeddings.shape[1]),
                        "success": True,
                        "error": "",
                        "segmentation": "peakdetect",
                        "features": "mfcc",
                        "pooling": "mean",
                    }
                )

            manifest_path = output_dir / "embedding_manifest.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            return {
                "manifest_path": str(manifest_path),
                "output_dir": str(output_dir),
                "num_files": len(rows),
                "num_success": len(rows),
                "num_failed": 0,
            }

    monkeypatch.setattr("findsylls.pipeline.orchestrator.EmbeddingPipeline", FakeEmbeddingPipeline)

    out_dir = tmp_path / "corpus_discovery"
    out = orch.discover_corpus(
        ["audio0.wav", "audio1.wav"],
        out_dir,
        segmentation_method="peakdetect",
        features_method="mfcc",
        pooling_method="mean",
        segmentation_kwargs={"envelope_method": "hilbert"},
        discovery_method="kmeans",
        discovery_kwargs={"n_clusters": 2, "random_state": 0},
        persist=True,
    )

    assert out["discovery_result"].labels.shape == (4,)
    assert Path(out["embedding_manifest_path"]).exists()
    assert Path(out["segmentation_manifest_path"]).exists()
    assert Path(out["discovery_manifest_path"]).exists()
    assert Path(out["corpus_manifest_path"]).exists()
    assert Path(out["discovery_model_path"]).exists()
    assert set(out["corpus_manifest"]["cluster_label"].dropna().astype(int).tolist()) <= {0, 1}
