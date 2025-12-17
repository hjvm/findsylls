# Changelog

All notable changes to this project will be documented in this file.

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
- ✅ Segmentation produces identical boundaries (sbs + peaks_and_valleys)
- ✅ MFCC feature extraction matches legacy code
- ✅ Mean pooling is consistent
- Tested on Brent corpus (4,209 syllables across 862 utterances)

### Notes
- Our `onc` pooling = legacy `onc-strict` (30% onset, peak nucleus, 70% coda)
- Old findsylls defaults: `envelope_fn='sbs'`, `segment_fn='peaks_and_valleys'`
- New findsylls defaults: `envelope_fn='hilbert'`, `method='peaks_and_valleys'`

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


