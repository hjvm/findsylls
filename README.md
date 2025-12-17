# findsylls

[![PyPI version](https://img.shields.io/pypi/v/findsylls.svg)](https://pypi.org/project/findsylls/)
[![Python versions](https://img.shields.io/pypi/pyversions/findsylls.svg)](https://pypi.org/project/findsylls/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Unsupervised syllable segmentation, evaluation, and **embedding extraction** toolkit for speech audio. Extract amplitude/modulation envelopes, segment into syllables, extract per-syllable embeddings, and evaluate versus TextGrid annotations.

## Features

### Core Segmentation & Evaluation
- **Classical envelope-based methods**: RMS, Hilbert, low-pass, spectral band subtraction (SBS), gammatone, theta oscillator
- **End-to-end neural methods** (optional): Sylber (self-supervised syllabic distillation)
- Peak & valley segmentation using Billauer's peakdetect algorithm
- Robust TextGrid parsing for phones, syllables, words with vowel filtering
- Multi-level evaluation metrics (precision/recall/F1 at nuclei, boundaries, spans)
- Batch pipeline utilities + fuzzy filename matching

### Syllable Embedding Pipeline ✨ NEW in v1.0.0

Extract **per-syllable embeddings** for clustering, classification, and cross-lingual analysis:

**Embedder Methods:**
- `sylber` - 768-dim self-supervised (GPU, ~50 fps)
- `vg_hubert` - 768-dim VG-HuBERT (GPU, ~50 fps, requires manual download)
- `mfcc` - 13/26/39-dim with delta/delta-delta (CPU, ~100 fps)
- `melspec` - 80-dim mel-filterbank (CPU, ~100 fps)

**Pooling Methods:**
- `mean` - Average frames within syllable (default)
- `onc` - Onset-Nucleus-Coda template (3× dimensions, preserves structure)
- `max`, `median` - Alternative aggregations

**Storage Formats:**
- NPZ (NumPy, always available)
- HDF5 (optional, requires h5py, memory-mapped for large corpora)

## Install
```bash
# Core installation (envelope-based segmentation + MFCC/Mel embeddings)
pip install findsylls

# With end-to-end neural segmentation (Sylber)
pip install 'findsylls[end2end]'

# With HDF5 storage for large embedding datasets
pip install 'findsylls[storage]'

# With neural embedders (VG-HuBERT, requires PyTorch)
pip install 'findsylls[embedding]'

# With all optional features
pip install 'findsylls[all]'

# With visualization extras
pip install 'findsylls[viz]'

# Local development (editable)
pip install -e .[dev]
```

**Note:** Neural methods (Sylber, VG-HuBERT) require PyTorch and download pre-trained models on first use (~500-800MB).

## Quick Start

### Syllable Segmentation
```python
from findsylls import segment_audio

# Segment audio into syllables
sylls, env, t = segment_audio(
    "example.wav",
    envelope_fn="sbs",
    segment_fn="peaks_and_valleys"
)
# sylls: [(start, peak, end), ...]
print(f"Found {len(sylls)} syllables")
```

### Syllable Embeddings ✨
```python
from findsylls import embed_audio

# Extract per-syllable embeddings (single file)
embeddings, metadata = embed_audio(
    'audio.wav',
    segmentation='peaks_and_valleys',  # or 'sylber'
    embedder='mfcc',                   # or 'sylber', 'vg_hubert'
    pooling='mean'                     # or 'onc', 'max'
)
# embeddings: (num_syllables, embedding_dim) NumPy array
# metadata: dict with boundaries, methods, timestamps

print(f"Shape: {embeddings.shape}")
print(f"Syllables: {metadata['num_syllables']}")

# Process entire corpus (batch)
from findsylls import embed_corpus, save_embeddings

results = embed_corpus(
    audio_paths=['audio1.wav', 'audio2.wav', 'audio3.wav'],
    segmentation='peaks_and_valleys',
    embedder='mfcc',
    pooling='mean',
    n_jobs=4  # Parallel processing
)

# Save to disk
save_embeddings(results, 'embeddings.npz')  # or .h5 for HDF5
```

### MFCC with Delta Features
```python
from findsylls import embed_audio

# 13-dim MFCC
embeddings_13, _ = embed_audio('audio.wav', embedder='mfcc')

# 26-dim MFCC (13 + 13 delta)
embeddings_26, _ = embed_audio(
    'audio.wav',
    embedder='mfcc',
    embedder_kwargs={'include_delta': True}
)

# 39-dim MFCC (13 + 13 delta + 13 delta-delta)
embeddings_39, _ = embed_audio(
    'audio.wav',
    embedder='mfcc',
    embedder_kwargs={'include_delta': True, 'include_delta_delta': True}
)
```

### Batch Evaluation
```python
from findsylls import run_evaluation, aggregate_results

results = run_evaluation(
    textgrid_paths="data/**/*.TextGrid",
    wav_paths="data/**/*.wav",
    phone_tier=1,
    syllable_tier=2,
    word_tier=3,
    envelope_fn="hilbert",
)
print(results.head())

# Aggregate metrics
summary = aggregate_results(results, dataset_name="MyCorpus")
print(summary)
```

## CLI
After install:
```bash
# Segment audio
findsylls segment input.wav --envelope sbs --method peaks_and_valleys --out sylls.json

# Extract embeddings
findsylls embed input.wav --embedder mfcc --pooling mean --out embeddings.npz

# Evaluate against TextGrid annotations
findsylls evaluate "data/**/*.wav" "data/**/*.TextGrid" \
    --phone-tier 1 --syllable-tier 2 --word-tier 3 \
    --envelope hilbert --out results.csv

# Show help
findsylls --help
findsylls segment --help
findsylls embed --help
findsylls evaluate --help
```

---

## Documentation

- **[EMBEDDING_PIPELINE.md](docs/EMBEDDING_PIPELINE.md)** - Complete embedding guide with architecture
- **[VALIDATION_RESULTS.md](docs/VALIDATION_RESULTS.md)** - Validation against legacy implementation (r=0.9990)
- **[PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md)** - Phase 1 implementation details
- **[PHASE2_SUMMARY.md](docs/PHASE2_SUMMARY.md)** - Phase 2 enhancements (VG-HuBERT, MFCC deltas)
- **[PHASE3_SUMMARY.md](docs/PHASE3_SUMMARY.md)** - Phase 3 corpus processing
- **[VG_HUBERT_README.md](docs/VG_HUBERT_README.md)** - VG-HuBERT setup guide
- **[examples/](examples/)** - Usage examples for all features

---

## Syllable Embedding Pipeline

The embedding pipeline extends findsylls to extract **per-syllable embeddings** for downstream tasks.

### Key Concepts

**Two orthogonal dimensions**:
1. **Embedder** (feature extraction): Sylber, VG-HuBERT, MFCC, Mel-spectrogram
2. **Pooling** (frame → syllable aggregation): mean, ONC template, max, median

**Any segmentation method can feed any embedding method.**

### Complete Example

```python
from findsylls import embed_audio

# Single file with full control
embeddings, metadata = embed_audio(
    'audio.wav',
    segmentation='peaks_and_valleys',  # Segmentation method
    embedder='sylber',                 # Feature extractor
    pooling='mean',                    # Aggregation method
    sr=16000,                          # Sample rate
    return_metadata=True               # Include metadata
)

# embeddings: (num_syllables, 768) for Sylber
# metadata contains: boundaries, peaks, num_syllables, methods, etc.

# Corpus processing with storage
from findsylls import embed_corpus, save_embeddings

results = embed_corpus(
    audio_paths=['data/audio1.wav', 'data/audio2.wav'],
    segmentation='peaks_and_valleys',
    embedder='mfcc',
    pooling='mean',
    embedder_kwargs={'include_delta': True},  # 26-dim MFCC
    n_jobs=4  # Parallel processing
)

# Save to disk (auto-detects format from extension)
save_embeddings(results, 'embeddings.npz')   # NumPy format
save_embeddings(results, 'embeddings.h5')    # HDF5 format (requires h5py)
```

### Embedder Options

**Neural** (GPU-enabled, requires `[embedding]` or `[all]`):
- `sylber` - 768-dim, ~50 fps, self-supervised
- `vg_hubert` - 768-dim, ~50 fps, phonetically-informed (requires manual download)

**Classical** (CPU-optimized, always available):
- `mfcc` - 13/26/39-dim (with deltas), ~100 fps
- `melspec` - 80-dim mel-filterbank, ~100 fps

### Pooling Options

- `mean` - Average frames in syllable span (default)
- `onc` - Onset-Nucleus-Coda template (3× dimensions)
  - Onset: 30% from start to peak
  - Nucleus: peak frame  
  - Coda: 70% from peak to end
- `max` - Max pooling across time
- `median` - Median pooling

### Storage Options

```python
from findsylls import save_embeddings, load_embeddings

# NumPy format (always available, good for small-medium datasets)
save_embeddings(results, 'embeddings.npz')
loaded = load_embeddings('embeddings.npz')

# HDF5 format (requires h5py, best for large corpora)
# Supports memory-mapped partial loading
save_embeddings(results, 'embeddings.h5')
loaded = load_embeddings('embeddings.h5')  # Full load
loaded = load_embeddings('embeddings.h5', indices=[0, 5, 10])  # Partial load
```

For complete documentation, see [docs/EMBEDDING_PIPELINE.md](docs/EMBEDDING_PIPELINE.md).

---

## API Reference

### Segmentation & Evaluation
| Function | Purpose |
|----------|---------|
| `segment_audio` | One-file end-to-end (load → envelope → segment) |
| `run_evaluation` | Batch match WAV/TextGrid and compute metrics |
| `get_amplitude_envelope` | Compute envelope via registered method |
| `segment_envelope` | Dispatch segmentation algorithm |
| `flatten_results` / `aggregate_results` | Reshape & aggregate evaluation outputs |
| `plot_segmentation_result` | Multi-panel qualitative plot (optional) |

### Embedding Extraction ✨
| Function | Purpose |
|----------|---------|
| `embed_audio` | Extract syllable embeddings from single audio file |
| `embed_corpus` | Batch embedding extraction with parallel processing |
| `save_embeddings` | Save embeddings to NPZ or HDF5 format |
| `load_embeddings` | Load embeddings from NPZ or HDF5 format |

## Validation

**v1.0.0 validated against legacy spot_the_word implementation:**
- ✅ **r=0.9990 correlation** on MFCC embeddings
- ✅ **100% syllable count match** (10/10 test files)
- ✅ Identical segmentation boundaries
- Tested on Brent corpus (4,209 syllables, 862 utterances)

See [docs/VALIDATION_RESULTS.md](docs/VALIDATION_RESULTS.md) for full report.

## Adding Methods
1. Envelope: implement `compute_*` returning `(env, times)` in `envelope/` and register in `envelope/dispatch.py`.
2. Segmentation: implement `segment_<name>(envelope, times, **kwargs)` in `segmentation/` and add branch in `segmentation/dispatch.py`.

## TextGrid Tier Indexing
Indices are 0-based (as provided by the `textgrid` library). Pass `None` to skip a tier or `-1` for placeholder syllable generation (currently returns empty list).

## Evaluation Conventions
- Default tolerance = 0.05s.
- Evaluation keys are generated dynamically based on tier specifications (e.g., `syllable_boundaries`, `word_spans`).
- To evaluate against a new tier, pass it via the `tiers` parameter: `tiers={'my_tier': 3}`.
- Substitutions matter for span metrics; remain zero for nuclei/boundary F1 semantics.

## Roadmap

- [x] Classical envelope-based segmentation (v0.1.0)
- [x] Multi-level TextGrid evaluation (v0.1.0)
- [x] End-to-end neural segmentation (Sylber) (v0.1.1)
- [x] Syllable embedding pipeline (v1.0.0)
  - [x] Core extractors (Sylber, MFCC, Mel-spectrogram)
  - [x] Pooling methods (mean, ONC, max, median)
  - [x] Corpus processing with parallel execution
  - [x] Storage utilities (NPZ, HDF5)
  - [x] VG-HuBERT support with MFCC delta features
- [ ] Additional neural embedders (HuBERT, Wav2Vec2, WavLM)
- [ ] Streaming / large-file handling
- [ ] Alternative segmentation algorithms (Mermelstein, oscillator-based)
- [ ] Enhanced CLI with progress tracking

## Performance Notes
- Audio loading prefers `torchaudio` (install separately) else falls back to `soundfile`/`librosa`
- Envelope computation is vectorized (NumPy); SBS/Hilbert faster than theta/gammatone
- Embedding extraction: Neural methods (GPU) ~50 fps, Classical methods (CPU) ~100 fps
- Corpus processing supports parallel execution via `n_jobs` parameter

## FAQ

**Can I use my own segmentation boundaries?**  
Yes! Extract embeddings from pre-segmented syllables by calling the lower-level API (see [docs/EMBEDDING_PIPELINE.md](docs/EMBEDDING_PIPELINE.md)).

**Why are boundary/spans columns missing in evaluation?**  
If a tier index is `None` or produces no intervals, those metrics are skipped.

**How do I add a custom envelope method?**  
Implement a function returning `(envelope, times)` and register it in `envelope/dispatch.py`.

**Can I stream long recordings?**  
Not yet; current design assumes full in-memory arrays. Streaming is on the roadmap.

**Why do I get 0 TP for nuclei evaluation?**  
Likely vowel set mismatch; confirm phone tier labels match ARPABET or adjust `SYLLABIC` constant.

**What's the difference between ONC pooling methods?**  
Our `onc` pooling = legacy spot_the_word `onc-strict` (30% onset, peak nucleus, 70% coda).

## Roadmap / TODO
- Implement `generate_syllable_intervals` (placeholder now).
- Additional segmentation algorithms (Mermelstein, oscillator-based).
- More robust CLI progress + JSON schema for outputs.
- Optional streaming / large-file handling.

## Roadmap / TODO
- Implement `generate_syllable_intervals` (placeholder now).
- Additional segmentation algorithms (Mermelstein, oscillator-based).
- More robust CLI progress + JSON schema for outputs.
- Optional streaming / large-file handling.

## Legacy Code
The previous exploratory/monolithic implementations are retained under a `legacy/` folder (formerly `old/` and `findsylls_old/`) for reference only. They are excluded from distribution and not supported; prefer the public API described above.

## License
MIT. See `LICENSE`.

## Citation
If you use this software in academic work, please cite the repository until a formal paper/preprint is available. A `CITATION.cff` file will appear in a future release.

Plain text:
```
Vázquez Martínez, Héctor Javier. (2024). findsylls: Unsupervised syllable segmentation, evaluation, and embedding extraction toolkit (Version 1.0.0) [Computer software]. https://github.com/hjvm/findsylls
```

BibTeX:
```bibtex
@software{findsylls,
    author = {Vázquez Martínez, Héctor Javier},
    title = {findsylls: Unsupervised syllable segmentation, evaluation, and embedding extraction toolkit},
    year = {2024},
    version = {1.0.0},
    url = {https://github.com/hjvm/findsylls},
    license = {MIT}
}
```

---
For development guidelines see `.github/copilot-instructions.md`.
