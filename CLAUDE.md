# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**findsylls** (v3.0.0) is a language-agnostic toolkit for unsupervised syllable-level speech segmentation, embedding extraction, and evaluation. Python APIs are the primary surface; the CLI and notebooks are convenience wrappers.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e '.[dev]'

# Run all tests
pytest

# Run a single test file
pytest tests/test_smoke.py -v

# Run a single test
pytest tests/test_smoke.py::test_segment_audio_sample -v

# Lint and format
ruff check src/ tests/
black src/ tests/

# Type check
mypy src/findsylls/
```

Optional extras: `viz`, `embedding`, `end2end`, `storage`, `all`.

## Architecture

### Data Flow

```
wav → load_audio() → (audio, sr)
  ├─ Envelope-based: get_amplitude_envelope() → (envelope, times)
  │                  → segment_envelope() → [(start, peak, end), ...]
  └─ End-to-end:     get_segmenter('sylber'/'vg_hubert') → segment() → [(start, peak, end), ...]
                      ↓
             evaluate_segmentation() → metrics dict
                      ↓
             flatten_results() → pd.DataFrame
```

### Module Map

| Module | Role |
|--------|------|
| `audio/` | I/O and normalization. `load_audio()` prefers torchaudio, falls back to soundfile/librosa; always returns mono float32. `match_wavs_to_textgrids()` does multi-step fuzzy matching (exact base → stripped suffix → alt-index → prefix → substring). |
| `envelope/` | Envelope computation. `dispatch.get_amplitude_envelope(method)` returns `(env, times)`. Supported: `rms`, `hilbert`, `lowpass`, `sbs`, `gammatone`, `theta`, `cls_attention`, `greedy_cosine`, `mincut`. |
| `segmentation/` | `dispatch.get_segmenter(method)` returns a `BaseSegmenter`. Canonical methods: `peakdetect`, `cls_attention`, `mincut`, `greedy_cosine`. `presets.py` ships published configurations (Sylber, VG-HuBERT, SyllableLM). |
| `features/` | `FeatureExtractor` ABC — `extract(audio, sr)`, `frame_rate`, `supports_attention`. Implementations: `mfcc`, `melspectrogram`, `hubert`, `vg_hubert`, `sylber`. |
| `embedding/` | `embed_audio()` / `embed_corpus()`. Poolers (mean, max, median, onc) in `poolers/`. HDF5 corpus storage in `storage.py`. |
| `evaluation/` | Orchestrated by `evaluate_segmentation()`. Keys are dynamically generated (`{tier_name}_boundaries`, `{tier_name}_spans`, `nuclei`). Default boundary tolerance: 0.05 s. |
| `discovery/` | `DiscoveryPipeline` wraps clustering (k-means, mini-batch, agglomerative) over corpus embeddings. |
| `pipeline/` | `segment_audio()` (single file), `run_evaluation()` (batched), `FindSyllsOrchestrator` (full corpus workflows). |
| `presets.py` | Named recipe bundles: `sylber`, `vg_hubert_mincut`, `vg_hubert_cls`, `syllablelm`. Use `resolve_preset()` to merge with user overrides. |
| `parsing/` | TextGrid parsing — extracts phone/syllable/word intervals, filters vowels via `SYLLABIC` set. |
| `plotting/` | Visualization helpers expecting flattened DataFrames. |

### Core Abstractions

- **`BaseSegmenter`** (`segmentation/base.py`) — all segmenters implement `segment(audio, sr) → List[(start, peak, end)]`.
- **`EnvelopeComputer`** (`envelope/base.py`) — `compute(audio, sr) → (envelope, times)`.
- **`FeatureExtractor`** (`features/base.py`) — `extract(audio, sr) → np.ndarray`, `frame_rate`, optional `extract_with_attention()`.
- **`BasePooler`** (`embedding/base.py`) — `pool(features, spans) → np.ndarray`.
- **`BaseDiscoveryModel`** (`discovery/base.py`) — `fit()`, `predict()`, `save()`.

## Adding Methods

**New segmentation method:**
1. Implement a class inheriting `BaseSegmenter` (or `EnvelopeBasedSegmenter` / `End2EndSegmenter`).
2. The `segment()` method must return `List[(start, peak, end)]`.
3. Register in `segmentation/dispatch.py`.

**New envelope method:**
1. Implement a class inheriting `EnvelopeComputer`; `compute()` returns `(np.ndarray, np.ndarray)` where times and envelope have the same length.
2. Register in `envelope/dispatch.py`.

**New feature extractor:**
1. Subclass `FeatureExtractor` in `features/`.
2. Register in `features/dispatch.py` (or wherever features are resolved).

## Evaluation Conventions

- Tier specification via `tiers` dict: `tiers={'phone': 2, 'syllable': 1, 'word': 0}`. There are no legacy `phone_tier`/`syllable_tier`/`word_tier` kwargs — always use the `tiers` dict.
- The `phone` tier drives `nuclei` evaluation; all other tiers generate `{name}_boundaries` and `{name}_spans` keys.
- `flatten_results()` dynamically detects all evaluation keys — no constant maintenance needed when adding tiers.
- Do not hardcode the 0.05 s tolerance outside of `config/constants.py`; pass it via `tolerance` arg.

## Memory / Model Lifecycle

Neural segmenters and feature extractors hold GPU/CPU memory. Call `segmenter.release()` or `extractor.release()` at explicit lifecycle boundaries (between corpus phases, after batch jobs). The `_SEGMENTER_CACHE` in `segmentation/dispatch.py` reuses instances per-file by default (`cache=True`).

## Common Pitfalls

- `match_wavs_to_textgrids()` returns parallel ordered lists; downstream code assumes zipped alignment — preserve sorted deterministic order when modifying.
- `extract_syllable_intervals()` returns a dict with `intervals` and `deleted` keys, not a raw list.
- Envelope `times` array must be the same length as the `envelope` array or peak-picking will be misaligned.
- Substitution count is meaningful for span metrics but should be 0 for nuclei/boundary F1 — see note in `aggregate_results`.
