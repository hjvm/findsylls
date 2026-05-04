"""Compatibility contract tests for segmentation/preset/embedding wiring."""

import pytest

from findsylls import list_segmenters, list_presets, get_preset
from findsylls.embedding import embed_audio
from findsylls.presets import resolve_preset


def test_segmenters_are_algorithm_only():
    segmenters = set(list_segmenters())
    assert segmenters == {"peakdetect", "cls_attention", "mincut", "greedy_cosine"}


def test_preset_registry_is_separate():
    presets = set(list_presets())
    assert presets == {"sylber", "vg_hubert_mincut", "vg_hubert_cls", "syllablelm"}

    sylber = get_preset("sylber")
    assert sylber["segmentation"] == "greedy_cosine"
    assert sylber["features"] == "sylber"


def test_embedding_rejects_preset_name_as_segmentation():
    with pytest.raises(ValueError, match="preset name"):
        embed_audio(
            "test_samples/SP20_117.wav",
            segmentation="sylber",
            features="hubert",
            pooling="mean",
        )


def test_embedding_accepts_cls_attention_with_hubert_features():
    emb, meta = embed_audio(
        "test_samples/SP20_117.wav",
        segmentation="cls_attention",
        features="hubert",
        pooling="mean",
    )
    assert emb.shape[0] > 0
    assert meta["segmentation_method"] == "cls_attention"
    assert meta["features"] == "hubert"


def test_embedding_rejects_peakdetect_without_explicit_envelope_for_mfcc():
    with pytest.raises(ValueError, match="requires a 1-D envelope method"):
        embed_audio(
            "test_samples/SP20_117.wav",
            segmentation="peakdetect",
            features="mfcc",
            pooling="onc",
        )


def test_embedding_accepts_explicit_preset():
    emb, meta = embed_audio(
        "test_samples/SP20_117.wav",
        preset="syllablelm",
    )
    assert emb.shape[0] > 0
    assert meta["segmentation_method"] == "mincut"
    assert meta["features"] == "hubert"
    assert meta["preset"] == "syllablelm"


def test_resolve_preset_preserves_runtime_feature_extractor_identity():
    sentinel = object()
    resolved = resolve_preset(
        preset=None,
        segmentation="peakdetect",
        features="hubert",
        pooling="mean",
        segmentation_kwargs={
            "envelope_method": "mincut",
            "envelope_kwargs": {
                "feature_extractor": sentinel,
            },
        },
        feature_kwargs=None,
        pooling_kwargs=None,
    )

    assert (
        resolved["segmentation_kwargs"]["envelope_kwargs"]["feature_extractor"] is sentinel
    )
