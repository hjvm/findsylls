# Release Notes: findsylls v1.0.0

**Release Date**: December 17, 2024  
**Type**: Major Release

## 🎉 Major Features: Syllable Embedding Pipeline

This release adds a complete **syllable embedding extraction pipeline** (Phases 1-3) validated against legacy implementation with **r=0.9990 correlation**.

### What's New

#### 🔬 Embedding Extraction
Extract per-syllable embeddings for downstream tasks (clustering, classification, cross-lingual analysis):

```python
from findsylls import embed_audio, embed_corpus

# Single file
embeddings, metadata = embed_audio(
    'audio.wav',
    segmentation='peaks_and_valleys',
    embedder='mfcc',
    pooling='mean'
)

# Batch processing
results = embed_corpus(
    audio_paths=['file1.wav', 'file2.wav'],
    embedder='mfcc',
    pooling='mean',
    n_jobs=4
)
```

#### 🎯 Feature Extractors
- **Sylber**: 768-dim self-supervised (GPU, ~50 fps)
- **VG-HuBERT**: 768-dim phonetically-informed (GPU, requires manual download)
- **MFCC**: 13/26/39-dim with delta/delta-delta support (CPU, ~100 fps)
- **Mel-spectrogram**: 80-dim filterbank (CPU, ~100 fps)

#### 🔄 Pooling Methods
- **Mean**: Average frames within syllable boundaries
- **ONC**: Onset-Nucleus-Coda template (30%/peak/70%, 3× dimensions)
- **Max/Median**: Alternative aggregations

#### 💾 Storage Utilities
- **NPZ format**: NumPy compressed (always available)
- **HDF5 format**: Memory-mapped for large corpora (optional, requires h5py)
- Auto-detection from file extension

#### ⚡ Batch Processing
- Parallel execution with `n_jobs` parameter (joblib)
- Progress tracking with tqdm
- Per-file error handling and recovery

### Validation ✅

**Tested against legacy spot_the_word implementation:**
- ✅ **Correlation: 0.9990** (near-perfect match)
- ✅ **Syllable count: 100% match** (10/10 files)
- ✅ Identical segmentation boundaries
- ✅ Consistent MFCC extraction and mean pooling
- Dataset: Brent corpus (4,209 syllables, 862 utterances)

See [docs/VALIDATION_RESULTS.md](docs/VALIDATION_RESULTS.md) for full report.

### Breaking Changes

None. This release is fully backward-compatible with v0.1.x. All existing segmentation and evaluation APIs remain unchanged.

### New Dependencies

**Core** (required):
- `joblib>=1.3` - Parallel processing
- `tqdm>=4.65` - Progress tracking

**Optional** (install extras):
- `h5py>=3.8` - HDF5 storage (`pip install 'findsylls[storage]'`)
- `torch>=2.0`, `transformers>=4.30` - Neural embedders (`pip install 'findsylls[embedding]'`)

### Documentation

New documentation:
- [EMBEDDING_PIPELINE.md](docs/EMBEDDING_PIPELINE.md) - Complete architecture guide
- [VALIDATION_RESULTS.md](docs/VALIDATION_RESULTS.md) - Validation report
- [PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md) - Phase 1 details
- [PHASE2_SUMMARY.md](docs/PHASE2_SUMMARY.md) - Phase 2 enhancements
- [PHASE3_SUMMARY.md](docs/PHASE3_SUMMARY.md) - Phase 3 corpus processing
- [VG_HUBERT_README.md](docs/VG_HUBERT_README.md) - VG-HuBERT setup

Updated:
- [README.md](README.md) - Complete rewrite with embedding examples
- [CHANGELOG.md](CHANGELOG.md) - Detailed changelog

### Examples

New example scripts in `examples/`:
- `simple_embedding.py` - Basic usage
- `mfcc_delta_features.py` - MFCC with deltas
- `vg_hubert_embedding.py` - VG-HuBERT extraction
- `corpus_processing.py` - Batch processing with storage

### Tests

New test suites:
- `tests/test_embedding.py` - Phase 1 & 2 tests (6 tests)
- `tests/test_corpus.py` - Phase 3 tests (6 tests)
- `tests/test_validation_against_spot_the_word.py` - Legacy validation

All tests passing: **12/12** ✅

### Installation

```bash
# Core (segmentation + MFCC/Mel embeddings)
pip install findsylls

# With HDF5 storage
pip install 'findsylls[storage]'

# With neural embedders
pip install 'findsylls[embedding]'

# All features
pip install 'findsylls[all]'
```

### Migration from v0.1.x

No changes required! All v0.1.x code continues to work. New embedding features are opt-in:

```python
# v0.1.x code still works
from findsylls import segment_audio
sylls, env, t = segment_audio('audio.wav')

# New in v1.0.0 - optional
from findsylls import embed_audio
embeddings, metadata = embed_audio('audio.wav')
```

### Known Issues

- VG-HuBERT requires manual model download (not on HuggingFace Hub)
- Streaming for large files not yet implemented (planned for future release)
- HDF5 requires separate `h5py` installation

### Contributors

- Héctor Javier Vázquez Martínez (@hjvm)

### What's Next

Planned for future releases:
- Additional neural embedders (HuBERT, Wav2Vec2, WavLM)
- Streaming support for large files
- Enhanced CLI with embedding subcommand
- Alternative segmentation algorithms

---

**Full Changelog**: https://github.com/hjvm/findsylls/blob/main/CHANGELOG.md  
**Documentation**: https://github.com/hjvm/findsylls/tree/main/docs  
**Issues**: https://github.com/hjvm/findsylls/issues
