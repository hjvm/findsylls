"""Tests for normalized segmentation naming."""

from findsylls import list_segmenters, list_segmenter_aliases
from findsylls.segmentation import normalize_segmenter_name


def test_segmenter_aliases_normalize_to_canonical_names():
    assert normalize_segmenter_name("greedycosine") == "greedy_cosine"


def test_segmenter_alias_map_is_exposed():
    aliases = list_segmenter_aliases()

    assert aliases["greedycosine"] == "greedy_cosine"


def test_canonical_segmenter_list_is_exported():
    segmenters = list_segmenters()

    assert "peakdetect" in segmenters
    assert "cls_attention" in segmenters
    assert "mincut" in segmenters
    assert "greedy_cosine" in segmenters
    assert "sylber" not in segmenters
    assert "vg_hubert_mincut" not in segmenters
    assert "syllablelm" not in segmenters