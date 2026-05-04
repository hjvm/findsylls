from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from findsylls.pipeline import build_discovery_manifest, build_file_manifest, build_label_manifest, build_segmentation_manifest, join_corpus_manifests
from findsylls.evaluation import attach_textgrid_labels_to_manifest


pytest.importorskip("textgrid")
from textgrid import IntervalTier, TextGrid  # type: ignore


def _write_textgrid(path: Path) -> None:
    tg = TextGrid(minTime=0.0, maxTime=1.0)
    tier = IntervalTier(name="syllables", minTime=0.0, maxTime=1.0)
    tier.add(0.0, 0.5, "aa")
    tier.add(0.5, 1.0, "bb")
    tg.append(tier)
    tg.write(str(path))


def test_segmentation_manifest_and_join(tmp_path):
    file_manifest = build_file_manifest(
        [
            {"audio_path": "a.wav"},
            {"audio_path": "b.wav"},
        ]
    )
    segmentation_manifest = build_segmentation_manifest(
        [
            {"file_id": 0, "segments": [(0.0, 0.1, 0.2), (0.2, 0.3, 0.4)], "segmentation_method": "peakdetect"},
            {"file_id": 1, "segments": [(0.5, 0.6, 0.7)], "segmentation_method": "peakdetect"},
        ],
        output_path=tmp_path / "segmentation_manifest.csv",
    )

    assert list(segmentation_manifest.columns)[:5] == ["file_id", "segment_id", "start", "peak", "end"]
    assert (tmp_path / "segmentation_manifest.csv").exists()

    joined = join_corpus_manifests(segmentation_manifest, file_manifest=file_manifest)
    assert "audio_path" in joined.columns
    assert joined.loc[joined["file_id"] == 0, "audio_path"].iloc[0] == "a.wav"
    assert len(joined) == 3


def test_posthoc_labels_with_file_manifest(tmp_path):
    audio_src = Path("/Users/hjvm/Documents/UPenn/unsupervised_speech_segmentation/findsylls/test_samples/SP20_117.wav")
    audio_path = tmp_path / "toy.wav"
    audio_path.write_bytes(audio_src.read_bytes())

    file_manifest = build_file_manifest([
        {"file_id": 0, "audio_path": str(audio_path)},
    ])
    tg_path = tmp_path / "toy.TextGrid"
    _write_textgrid(tg_path)

    segmentation_manifest = build_segmentation_manifest([
        {"file_id": 0, "segments": [(0.05, 0.15, 0.25), (0.55, 0.65, 0.85)], "segmentation_method": "peakdetect"},
    ])

    labeled = attach_textgrid_labels_to_manifest(
        segmentation_manifest,
        file_manifest=file_manifest,
        wav_paths=str(tmp_path / "*.wav"),
        textgrid_paths=str(tmp_path / "*.TextGrid"),
        textgrid_tier_index=0,
    )

    assert list(labeled["primary_label"]) == ["aa", "bb"]
    assert labeled["audio_path"].notna().all()

    label_manifest = build_label_manifest(labeled, output_path=tmp_path / "label_manifest.csv")
    assert (tmp_path / "label_manifest.csv").exists()
    assert "primary_label" in label_manifest.columns

    joined = join_corpus_manifests(segmentation_manifest, file_manifest=file_manifest, label_manifest=label_manifest)
    assert joined["primary_label"].notna().all()


def test_discovery_manifest_and_join(tmp_path):
    file_manifest = build_file_manifest([
        {"file_id": 0, "audio_path": "a.wav"},
    ])
    segmentation_manifest = build_segmentation_manifest([
        {"file_id": 0, "segments": [(0.0, 0.1, 0.2), (0.2, 0.3, 0.4)], "segmentation_method": "peakdetect"},
    ])
    discovery_manifest = build_discovery_manifest(
        [
            {
                "file_id": 0,
                "segment_id": 0,
                "embedding_id": 10,
                "start": 0.0,
                "peak": 0.1,
                "end": 0.2,
                "cluster_label": 1,
                "embedding_path": "/tmp/0.npz",
                "discovery_method": "kmeans",
            },
            {
                "file_id": 0,
                "segment_id": 1,
                "embedding_id": 11,
                "start": 0.2,
                "peak": 0.3,
                "end": 0.4,
                "cluster_label": 0,
                "embedding_path": "/tmp/0.npz",
                "discovery_method": "kmeans",
            },
        ],
        output_path=tmp_path / "discovery_manifest.csv",
    )

    assert (tmp_path / "discovery_manifest.csv").exists()
    assert list(discovery_manifest.columns)[:6] == ["file_id", "segment_id", "embedding_id", "start", "peak", "end"]

    joined = join_corpus_manifests(
        segmentation_manifest,
        file_manifest=file_manifest,
        discovery_manifest=discovery_manifest,
    )

    assert "cluster_label" in joined.columns
    assert joined.loc[joined["segment_id"] == 0, "cluster_label"].iloc[0] == 1
    assert joined.loc[joined["segment_id"] == 1, "cluster_label"].iloc[0] == 0
