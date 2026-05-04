# Changelog

All notable changes to this project will be documented in this file.

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


