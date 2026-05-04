from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from findsylls.evaluation import (
    attach_textgrid_labels_to_manifest,
    compute_discovery_label_metrics,
    export_discovery_label_artifacts,
)


pytest.importorskip("textgrid")
from textgrid import IntervalTier, TextGrid  # type: ignore


def _write_textgrid(path: Path) -> None:
    tg = TextGrid(minTime=0.0, maxTime=1.0)
    tier = IntervalTier(name="syllables", minTime=0.0, maxTime=1.0)
    tier.add(0.0, 0.45, "alpha")
    tier.add(0.45, 1.0, "beta")
    tg.append(tier)
    tg.write(str(path))


def test_attach_textgrid_labels_to_manifest_with_globs(tmp_path):
    audio_src = Path("/Users/hjvm/Documents/UPenn/unsupervised_speech_segmentation/findsylls/test_samples/SP20_117.wav")
    audio_path = tmp_path / "toy.wav"
    shutil.copy(audio_src, audio_path)

    tg_path = tmp_path / "toy.TextGrid"
    _write_textgrid(tg_path)

    manifest = pd.DataFrame(
        [
            {"audio_path": str(audio_path), "start": 0.05, "peak": 0.12, "end": 0.25, "cluster_label": 0},
            {"audio_path": str(audio_path), "start": 0.55, "peak": 0.65, "end": 0.85, "cluster_label": 1},
        ]
    )

    labeled = attach_textgrid_labels_to_manifest(
        manifest,
        wav_paths=str(tmp_path / "*.wav"),
        textgrid_paths=str(tmp_path / "*.TextGrid"),
        textgrid_tier_index=0,
        primary_label_mode="sequence",
        output_path=tmp_path / "labeled_manifest.csv",
    )

    assert (tmp_path / "labeled_manifest.csv").exists()
    assert list(labeled["primary_label"]) == ["alpha", "beta"]
    assert bool(labeled.loc[0, "label_attached"]) is True
    assert labeled.loc[0, "label_source"] == "textgrid"
    assert labeled.loc[0, "tg_labels"] == ["alpha"]
    assert labeled.loc[1, "tg_labels"] == ["beta"]


def test_compute_discovery_label_metrics():
    manifest = pd.DataFrame(
        [
            {"cluster_label": 0, "primary_label": "a"},
            {"cluster_label": 0, "primary_label": "a"},
            {"cluster_label": 1, "primary_label": "b"},
            {"cluster_label": 1, "primary_label": "c"},
        ]
    )

    metrics = compute_discovery_label_metrics(manifest)

    assert metrics["total_labeled_syllables"] == 4
    assert metrics["cluster_purity"] == pytest.approx(0.75)
    assert metrics["label_purity"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert metrics["weighted_f1"] == pytest.approx((1.0 * 2 + (2.0 / 3.0) * 2) / 4.0)
    assert metrics["label_norm_mutual_info"] > 0.0
    assert metrics["n_clusters"] == 2
    assert metrics["n_labels"] == 3


def test_export_discovery_label_artifacts(tmp_path):
    manifest = pd.DataFrame(
        [
            {"audio_path": "a.wav", "start": 0.0, "peak": 0.1, "end": 0.2, "cluster_label": 0, "primary_label": "a"},
            {"audio_path": "a.wav", "start": 0.2, "peak": 0.3, "end": 0.4, "cluster_label": 0, "primary_label": "a"},
            {"audio_path": "b.wav", "start": 0.0, "peak": 0.1, "end": 0.2, "cluster_label": 1, "primary_label": "b"},
        ]
    )

    outputs = export_discovery_label_artifacts(manifest, tmp_path / "demo")

    assert Path(outputs["syllables_csv"]).exists()
    assert Path(outputs["cluster_metrics_csv"]).exists()
    assert Path(outputs["label_metrics_csv"]).exists()
    assert Path(outputs["global_metrics_json"]).exists()

    cluster_df = pd.read_csv(outputs["cluster_metrics_csv"])
    label_df = pd.read_csv(outputs["label_metrics_csv"])
    assert "precision" in cluster_df.columns
    assert "recall" in cluster_df.columns
    assert "f1" in cluster_df.columns
    assert "purity" in label_df.columns
