"""Preset recipe registry for end-to-end published configurations.

Presets are configuration bundles that resolve to concrete embedding pipeline
settings (segmentation + features + pooling + kwargs). They are intentionally
separate from `list_segmenters()` so segmentation method discovery remains
algorithmic.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


_RUNTIME_OBJECT_KEYS = {
    "feature_extractor",
    "envelope_computer",
}


def _copy_preserving_runtime_objects(value: Any) -> Any:
    """Deep-copy config data while preserving runtime object identity.

    Runtime objects (e.g. prebuilt extractors) must remain shared references,
    otherwise each copy starts with unloaded state and can trigger repeated
    model loads in per-file loops.
    """
    if isinstance(value, dict):
        copied: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _RUNTIME_OBJECT_KEYS:
                copied[key] = item
            else:
                copied[key] = _copy_preserving_runtime_objects(item)
        return copied
    if isinstance(value, list):
        return [_copy_preserving_runtime_objects(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_preserving_runtime_objects(item) for item in value)
    return deepcopy(value)


_PRESET_RECIPES: Dict[str, Dict[str, Any]] = {
    "sylber": {
        "segmentation": "greedy_cosine",
        "features": "sylber",
        "pooling": "mean",
        "segmentation_kwargs": {
            "feature_type": "sylber",
            "norm_threshold": 2.6,
            "merge_threshold": 0.8,
        },
        "feature_kwargs": {},
        "pooling_kwargs": {},
    },
    "vg_hubert_mincut": {
        "segmentation": "mincut",
        "features": "vghubert",
        "pooling": "mean",
        "segmentation_kwargs": {
            "feature_type": "vghubert",
            "sec_per_syllable": 0.22,
            "use_optimized": True,
            "min_hop": 3,
            "max_hop": 50,
        },
        "feature_kwargs": {},
        "pooling_kwargs": {},
    },
    "vg_hubert_cls": {
        "segmentation": "cls_attention",
        "features": "vghubert",
        "pooling": "mean",
        "segmentation_kwargs": {},
        "feature_kwargs": {"mode": "word"},
        "pooling_kwargs": {},
    },
    "syllablelm": {
        "segmentation": "mincut",
        "features": "hubert",
        "pooling": "mean",
        "segmentation_kwargs": {
            "feature_type": "hubert",
            "sec_per_syllable": 0.22,
            "use_optimized": True,
            "min_hop": 3,
            "max_hop": 50,
        },
        "feature_kwargs": {},
        "pooling_kwargs": {},
    },
}


def list_presets() -> list[str]:
    """Return available preset names."""
    return sorted(_PRESET_RECIPES.keys())


def get_preset(name: str) -> Dict[str, Any]:
    """Return a deep copy of a preset recipe."""
    key = name.strip().lower().replace("-", "_")
    if key not in _PRESET_RECIPES:
        available = ", ".join(list_presets())
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
    return deepcopy(_PRESET_RECIPES[key])


def resolve_preset(
    preset: Optional[str],
    *,
    segmentation: Optional[str],
    features: Optional[str],
    pooling: Optional[str],
    segmentation_kwargs: Optional[Dict[str, Any]],
    feature_kwargs: Optional[Dict[str, Any]],
    pooling_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve final embedding configuration with optional preset overrides."""
    resolved = {
        "preset": preset,
        "segmentation": segmentation,
        "features": features,
        "pooling": pooling,
        "segmentation_kwargs": _copy_preserving_runtime_objects(segmentation_kwargs or {}),
        "feature_kwargs": _copy_preserving_runtime_objects(feature_kwargs or {}),
        "pooling_kwargs": _copy_preserving_runtime_objects(pooling_kwargs or {}),
    }

    if not preset:
        return resolved

    recipe = get_preset(preset)

    for field in ("segmentation", "features", "pooling"):
        provided = resolved[field]
        expected = recipe[field]
        if provided is not None and str(provided).lower().replace("-", "_") != expected:
            raise ValueError(
                f"Preset '{preset}' requires {field}='{expected}', got '{provided}'. "
                f"Remove the explicit {field} override or choose a different preset."
            )
        resolved[field] = expected

    resolved["segmentation_kwargs"] = {
        **recipe.get("segmentation_kwargs", {}),
        **resolved["segmentation_kwargs"],
    }
    resolved["feature_kwargs"] = {
        **recipe.get("feature_kwargs", {}),
        **resolved["feature_kwargs"],
    }
    resolved["pooling_kwargs"] = {
        **recipe.get("pooling_kwargs", {}),
        **resolved["pooling_kwargs"],
    }

    return resolved
