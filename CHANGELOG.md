# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `EnergyPeriodicitySegmenter` — Xie & Niyogi (2006) two-stage syllable-nucleus
  detector (periodicity gates voiced regions; energy convex-hull picks nuclei).
- `RhythmGuidedSegmenter` — Zhang & Glass (2009) rhythm-guided nucleus detection,
  rebuilt around rhythm-licensed dynamic peak sensitivity (recovers closely-spaced
  merged nuclei).
- `PeriodicityEnvelope` — normalized-autocorrelation periodicity (Xie & Niyogi),
  dispatch method `periodicity`.
- `GammatoneEnvelope` — the ERB/gammatone filterbank promoted from a theta-internal
  step to a first-class envelope (`.filterbank()` multi-band, `.compute()` 1-D
  reduction), dispatch method `gammatone`.
- `convexhull` and `threshold` segmentation methods: `segment_convexhull` /
  `ConvexHullSegmenter` (Mermelstein) and `segment_threshold` / `ThresholdSegmenter`
  (regions above a threshold, with a dense `.mask()` view).
- `fit_rhythm_sinusoid` and `rhythm_crests` rhythm primitives (`segmentation/rhythm.py`).
- `RMSEnvelope` gains `db` / `reference` (relevant energy, dB below max) and `center`
  (frame alignment) options.

### Fixed
- `VOWELS` constant was missing the TIMIT vowels `ux`, `axr`, `ax-h`, which
  undercounted the reference vowel set in nuclei evaluation.

### Removed (breaking)
- `ProductEnvelope`, `ThresholdGate` (multiplicative-gating helpers — no longer used;
  compositions use `ThresholdSegmenter` masks / numpy directly).
- `RhythmEnvelope` and the `rhythm` **envelope** dispatch method (rhythm operates on
  peak trains, not audio — its math moved to `segmentation/rhythm.py`).
- `RegionGatedSegmenter` (redundant orchestrator; two-stage logic lives in the presets).
- `RhythmGuidedSegmenter` constructor changed: `delta`/`rhythm_floor`/`first_pass_delta`/
  `period_range` replaced by `tight_delta`/`loose_delta`/`seed_period`.

## [3.2.0] - 2026-06-03

> Consolidates unreleased work since 3.0.1 (the 3.0.2 / 3.1.x series was
> published from `__version__` without CHANGELOG entries) and adds the changes
> below.

### Breaking Changes
- `attach_textgrid_labels_to_manifest()` now takes a required
  `textgrid_tiers: Dict[str, int]` (same shape as `evaluate_segmentation`'s
  `tiers`) instead of `textgrid_tier_index: int = 0`. Per-tier label columns are
  written with a `{tier_name}_` prefix (`{tier}_tg_labels`,
  `{tier}_labels_concat`, `{tier}_primary_label`, `{tier}_primary_label_peak`,
  `{tier}_primary_label_max_overlap`); `textgrid_path`, `label_attached`, and
  `label_source` remain shared and unprefixed. The private `_row_textgrid_labels`
  return key `tier_labels_concat` was renamed to `labels_concat`. Downstream
  callers must pass an explicit `label_column` (e.g. `syllable_primary_label`)
  to `compute_discovery_label_metrics` / `export_discovery_label_artifacts`.
- `evaluate_segmentation()` now always emits tier-prefixed boundary/span keys
  (`{tier}_boundaries`, `{tier}_spans`) even when a single non-phone tier is
  evaluated. Previously a single tier produced generic `boundaries`/`spans` keys
  plus a `tier_level` metadata field, so the same syllable metric changed name
  depending on whether other tiers were evaluated alongside it (splitting
  `groupby('eval_method')` in cross-corpus aggregation). The `tier_level` key is
  removed; `flatten_results`/`plot_segmentation` no longer reference it.

### Added
- `SBSPeakdetectSegmenter` now exposes the lower-level peakdetect controls that
  were previously only reachable via the generic `PeakdetectSegmenter`:
  `min_syllable_dur`, `merge_valley_tol`, `min_amplitude_threshold`, and
  `lookahead`. In particular `min_amplitude_threshold` (fraction of max envelope)
  suppresses low-amplitude spurious peaks such as sonorant-onset bumps (e.g. the
  [l] in "clever") without affecting real nuclei — ~0.08–0.1 is typical.
  `max_syllable_dur` and `amplitude_ratio_tol` are now typed `Optional` (pass
  `None` to disable the cap / shallow-valley merge). Defaults unchanged.
- First-class `sad=` parameter on `segment_audio`, `run_evaluation`, `embed_audio`,
  `embed_corpus`, and `FindSyllsOrchestrator.discover_corpus` (accepts `'energy'`,
  `'silero'`, or a `BaseSAD` instance), restricting segmentation to detected speech
  regions. Previously reachable only via `segmentation_kwargs={'sad': ...}`; an
  explicit `segmentation_kwargs['sad']` still takes precedence.
- `collapse_clusters(embeddings, labels, n_clusters)` in `discovery/collapse.py`:
  collapses K fine-grained cluster labels into n_clusters coarse labels via
  agglomerative clustering on per-cluster centroids. Returns `(new_labels,
  centroids, centroid_map)` for test-time nearest-centroid assignment. Exported
  from `findsylls.discovery` and the top-level `findsylls` package.

### Removed
- `segmentation.dispatch.segment_envelope()` — unused backward-compat functional
  API (no callers, not exported). Use `get_segmenter(...)` /
  `EnvelopeBasedSegmenter.segment(envelope=, times=)`.

### Fixed
- `clear_segmenter_cache()` now calls `.release()` on cached segmenters before
  clearing (frees neural model memory), and is invoked at `embed_corpus` /
  `embed_corpus_to_storage` workflow boundaries so the per-process segmenter
  cache no longer outlives a corpus pass. `get_segmenter`'s cache key is now
  deterministic for value/dict/list kwargs (object instances still key by
  identity, preserving shared-instance reuse).
- Corrected the false "[DEPRECATED]/backward compatibility" labels on
  `get_amplitude_envelope` (it is the supported functional wrapper over
  `get_envelope_computer`).
- `scripts/findsylls_test_battery.py`: corrected `EVAL_TIERS` to match the
  TIMIT `*_syllabified.TextGrid` tier order `{phone: 0, word: 1, syllable: 2}`
  (was inverted, silently scoring nuclei against the syllable tier), and added
  `tg_suffix_to_strip="_syllabified"` to the label-attachment call so syllable
  labels actually attach.

## [3.0.1] - 2026-05-04

### Removed
- `SyllableLMSegmenter` preset class and `"syllablelm"` preset: the class used
  vanilla HuBERT but the SyllableLM paper used Data2Vec2 features — substituting
  a different extractor under that name was incorrect. The DP algorithm
  (`MinCutSegmenter(use_reference=True)`) and its parity tests are unaffected.
- Stale repository artifacts: `examples/` (all scripts broken on v3 API),
  `docs/FINDSYLLS_USER_GUIDE.md`, `docs/VALIDATION_RESULTS.md`,
  `docs/VG_HUBERT_README.md` (all superseded by README), 8 tracked `data/`
  symlinks pointing to local machine paths, `.github/copilot-instructions.md`
  (v1.x guidance, entirely wrong for v3).
- `findsylls_demo.ipynb` renamed to `research_evaluation.ipynb` (Interspeech
  2026 research notebook, not a user demo).

### Fixed
- `gammatone` removed from all public envelope method lists in README and
  dispatch docs — it is internal preprocessing used only by the `theta`
  oscillator, never a standalone callable method.
- `pyproject.toml`: removed `vg-hubert` from the `embedding` extra (VG-HuBERT
  is an end-to-end segmenter, not an embedding dep); completed author name.
- `MANIFEST.in`: removed duplicate `include README.md` / `include LICENSE` lines.
- `requirements.txt`: `seaborn` moved to optional/commented section (it is in
  the `viz` extra, not a core runtime dependency).

## [3.0.0] - 2026-04-01

### BREAKING CHANGES

**Major Architectural Overhaul**:

1. **Embedding Layer Rebuilt**:
   - Fully refactored from functional to OOP-first design following segmentation module pattern.
   - New class-based `EmbeddingPipeline` replaces legacy `embed_audio`/`embed_corpus` signatures.
   - Modular pooler architecture: `BasePooler` + concrete instances (MeanPooler, ONCPooler, MaxPooler, MedianPooler).
   - Poolers now live in `src/findsylls/embedding/poolers/` with dedicated dispatch registry.
   - Removed embedding-internal feature extraction duplication; now uses `src/findsylls/features/` exclusively.
   - Fixed broken Sylber path (NotImplementedError) and removed stale VG-HuBERT references.

2. **Discovery Layer Added** (New First-Class Module):
   - New package: `src/findsylls/discovery/`.
   - Purpose: Cluster syllable embeddings into identity groups (distinct from syllable segmentation).
   - Modular design: `BaseDiscoveryModel` + concrete instances (KMeansDiscovery, AgglomerativeDiscovery).
   - Class-based `DiscoveryPipeline` for corpus-level orchestration.
   - Fully separated from segmentation layer (no cross-layer coupling).

3. **Removed Legacy Code**:
   - Deleted 8 stale development docs: `docs/dev/PHASE*.md`, `docs/dev/DEVELOPMENT_GUIDE.md`, `docs/dev/UNIFIED_ROADMAP.md`, `docs/dev/TODO_INTERSPEECH.md`.
   - Deleted `RELEASE_NOTES_v1.0.0.md`.
   - Removed embedding-internal feature extraction legacy paths.
   - Removed compatibility aliases that were no longer needed.

4. **API Simplifications**:
   - Embedding API now OOP-first (breaking change from v2.0.0 functional style).
   - Discovery API fully OOP as first-class module.

### Added

- `src/findsylls/embedding/poolers/` subpackage with modular pooler classes and dispatch.
- `src/findsylls/discovery/` subpackage with modular discovery models and pipeline.
- `BasePooler` abstract class for extensible pooling strategies.
- `BaseDiscoveryModel` abstract class for extensible clustering/discovery methods.

### Changed

- Embedding pipeline now uses `src/findsylls/features/` exclusively (no duplication).
- Consolidated and cleaned up documentation set; removed phase-based development docs.
- Updated version across all metadata (pyproject.toml, CITATION.cff, `__init__.py`).

### Packaging

- Bumped package version to `3.0.0` due to scope and breaking nature of changes.

## [2.0.0] - 2026-03-30

### BREAKING CHANGES

- Segmentation method naming is now standardized on `peakdetect`.
- Legacy user-facing references to `peaks_and_valleys` were removed from docs/examples/notebooks.

### Changed

- Unified terminology and examples across package modules, CLI docs, tests, and notebooks.
- Updated README to a lean, release-focused user guide.
- Refreshed package citation metadata and release metadata for the major version.

### Packaging

- Bumped package version to `2.0.0`.

## [1.0.3] - 2026-03-30

### Changed

- Updated project citation metadata to reference the published arXiv preprint:
  - Vázquez Martínez, Héctor Javier (2026), arXiv:2603.26292
- Updated `CITATION.cff` preferred citation from software-only metadata to article metadata.
- Updated README citation section with preprint plain-text and BibTeX entries.

### Packaging

- Bumped package version to `1.0.3` for PyPI release.

## [1.0.2] - 2024-12-18

### BREAKING CHANGES

**API Terminology Update**: Parameter names have been updated for technical accuracy:
- `embedder` → `features` (in `embed_audio()` and `embed_corpus()`)
- `embedder_kwargs` → `feature_kwargs`
- Metadata key: `'embedder'` → `'features'`

**Rationale**: These parameters specify feature extraction methods (MFCC, Sylber, etc.), not embedders. The embeddings are created by pooling the extracted features over syllable spans. This change makes the pipeline conceptually clearer: features → pooling → embeddings.

**Migration Guide**:
```python
# OLD (v1.0.1 and earlier)
embed_audio('audio.wav', embedder='mfcc', embedder_kwargs={'include_delta': True})
embed_corpus(files, embedder='sylber', embedder_kwargs={})

# NEW (v1.0.2+)
embed_audio('audio.wav', features='mfcc', feature_kwargs={'include_delta': True})
embed_corpus(files, features='sylber', feature_kwargs={})
```

**Note**: Saved embeddings from v1.0.1 will have `metadata['embedder']` while v1.0.2+ saves `metadata['features']`. Both keys contain the same information.

### Performance

- **CRITICAL FIX**: Added model caching in `get_segmenter()` to prevent reloading neural segmentation models (Sylber, VG-HuBERT) for every audio file during batch processing
- Models are now cached globally within each worker process and reused across all files
- **Speedup**: ~2.8x improvement for large corpus processing with neural segmentation
  - Before: ~1.4 sec/file (projected 11+ hours for 28K files)
  - After: ~0.5 sec/file (projected 4 hours for 28K files)
- New utility functions: `clear_segmenter_cache()`, `get_cache_info()`

### Changed

- `segmentation.dispatch.get_segmenter()` now accepts `cache=True` parameter (default: enabled)
- Cache key automatically generated from method name + kwargs
- `embed_corpus()` documentation updated to note automatic model caching
- All examples and documentation updated to use new `features` parameter naming

### Technical Details

- Global cache dictionary `_SEGMENTER_CACHE` stores segmenter instances per worker process
- Each unique configuration (method + parameters) cached separately
- In multiprocessing mode (joblib default), each worker maintains its own cache
- Result: Model loaded once per worker instead of once per file (massive speedup for batch processing)

## [1.0.1] - 2024-12-17

### Fixed
- **`embed_corpus` export**: Fixed missing export of `embed_corpus` function from `embedding.__init__.py`, which caused import failure in top-level `findsylls.__init__.py`. The function was fully implemented but not included in `__all__`, preventing batch corpus processing functionality from being accessible.

## [1.0.0] - 2024-12-17

**Major release with syllable embedding pipeline (Phases 1-3) and validation against legacy implementation.**

### Added

#### Embedding Pipeline (Phase 1: Core Infrastructure)
- **`embedding.pipeline` module**: High-level APIs for extracting syllable embeddings
  - `embed_audio()`: Extract embeddings from single audio file
  - `embed_corpus()`: Batch processing with parallel execution (joblib)
- **`embedding.extractors` module**: Feature extraction methods
  - Sylber: 768-dim self-supervised representations (~50 fps)
  - MFCC: 13/26/39-dim coefficients with delta/delta-delta support (~100 fps)
  - Mel-spectrogram: 80-dim filterbank features (~100 fps)
  - VG-HuBERT: 768-dim representations (requires manual model download)
- **`embedding.pooling` module**: Frame-to-syllable aggregation
  - Mean pooling (average frames within syllable)
  - ONC (Onset-Nucleus-Coda) template pooling (30%/peak/70%, 3× dimensions)
  - Max pooling and Median pooling
- **`embedding.storage` module**: Persistent storage utilities
  - NPZ format (NumPy, always available)
  - HDF5 format (optional, requires h5py, supports partial loading)
  - Auto-format detection from file extension

#### Phase 2: Enhancements
- VG-HuBERT feature extraction with manual model path support
- MFCC delta and delta-delta features (13→26→39 dimensions)
  - `include_delta=True`: adds Δ-MFCC (26-dim)
  - `include_delta_delta=True`: adds Δ²-MFCC (39-dim)
- Contextual error messages for missing dependencies

#### Phase 3: Corpus Processing
- Batch embedding extraction with progress tracking (tqdm)
- Parallel processing support (configurable n_jobs)
- Error handling and recovery (per-file success/error tracking)
- Storage format auto-detection (.npz vs .h5)

#### Documentation
- Complete embedding pipeline documentation (`docs/EMBEDDING_PIPELINE.md`)
- Phase summaries: `PHASE1_COMPLETE.md`, `PHASE2_SUMMARY.md`, `PHASE3_SUMMARY.md`
- VG-HuBERT setup guide (`docs/VG_HUBERT_README.md`)
- Validation report against legacy spot_the_word implementation (`docs/VALIDATION_RESULTS.md`)

#### Examples
- `examples/simple_embedding.py`: Basic single-file embedding
- `examples/mfcc_delta_features.py`: MFCC with delta/delta-delta
- `examples/vg_hubert_embedding.py`: VG-HuBERT extraction
- `examples/corpus_processing.py`: Batch processing with storage
- `examples/README.md`: Examples overview

#### Tests
- `tests/test_embedding.py`: Phase 1 & 2 tests (6 tests)
- `tests/test_corpus.py`: Phase 3 corpus processing tests (6 tests)
- `tests/test_validation_against_spot_the_word.py`: Legacy validation
- All tests passing (12/12)

#### Repository Organization
- Created `docs/` directory (moved 9 documentation files)
- Created `notebooks/` directory (research notebooks)
- Created `examples/` directory (usage examples)
- Removed `legacy/` directory (old code)

### Changed
- **Version**: Bumped to 1.0.0 (major release with validated embedding features)
- **Dependencies**: Added `joblib>=1.3` and `tqdm>=4.65` as core dependencies
- **Optional dependencies**: Added `storage` (h5py) and `embedding` (torch, transformers) groups
- **Public API**: Exported `embed_audio`, `embed_corpus`, `save_embeddings`, `load_embeddings`
- **Package description**: Updated to include embedding extraction capabilities

### Validated
- ✅ **High correlation (r=0.9990)** with legacy spot_the_word implementation
- ✅ **100% syllable count match** (10/10 test files)
- ✅ Segmentation produces identical boundaries (sbs + peakdetect)
- ✅ MFCC feature extraction matches legacy code
- ✅ Mean pooling is consistent
- Tested on Brent corpus (4,209 syllables across 862 utterances)

### Notes
- Our `onc` pooling = legacy `onc-strict` (30% onset, peak nucleus, 70% coda)
- Old findsylls defaults: `envelope_fn='sbs'`, `segment_fn='peakdetect'`
- New findsylls defaults: `envelope_fn='hilbert'`, `method='peakdetect'`

---

## [0.1.1] - 2024-09-23
### Added
- CLI (`findsylls` executable) with `segment` and `evaluate` subcommands.
- MANIFEST.in to control packaged data and exclude tests / samples.

### Changed
- Bumped version to 0.1.1.

## [0.1.0] - 2024-09-23
### Added
- Established `src/findsylls` package layout.
- Modular envelope, segmentation, evaluation, pipeline, plotting subpackages.
- Fuzzy WAV/TextGrid matching and evaluation aggregation.
- Legacy exploratory code quarantined under `legacy/` (excluded from distribution).


