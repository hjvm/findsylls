# Examples

This directory contains usage examples for the findsylls embedding pipeline.

## Available Examples

### MFCC Delta Features
**File**: [mfcc_deltas.py](mfcc_deltas.py)

Demonstrates how to extract MFCC features with first and second-order derivatives (delta and delta-delta features).

```python
from findsylls.embedding.pipeline import embed_audio

# Standard 13-dimensional MFCC
embeddings, meta = embed_audio('audio.wav', embedder='mfcc')

# 26-dimensional MFCC with deltas
embeddings, meta = embed_audio('audio.wav', embedder='mfcc', include_delta=True)

# 39-dimensional MFCC with deltas and delta-deltas
embeddings, meta = embed_audio('audio.wav', embedder='mfcc', 
                               include_delta=True, include_delta_delta=True)
```

**Output dimensions**:
- Standard: 13 → 39 with ONC pooling
- With delta: 26 → 78 with ONC pooling
- With delta+delta-delta: 39 → 117 with ONC pooling

---

### VG-HuBERT Embedding
**File**: [vg_hubert_embedding.py](vg_hubert_embedding.py)

Complete example of extracting VG-HuBERT embeddings from audio.

```python
from findsylls.embedding.pipeline import embed_audio

# Extract VG-HuBERT embeddings with mean pooling
embeddings, meta = embed_audio('audio.wav', embedder='vg-hubert', pool_method='mean')
```

**Requirements**:
- VG-HuBERT model must be downloaded manually (see [docs/VG_HUBERT_README.md](../docs/VG_HUBERT_README.md))
- Model path: `models/vg-hubert_3/`

**Output dimensions**:
- VG-HuBERT: 768 → 2304 with ONC pooling

---

### Corpus Processing (Phase 3)
**File**: [corpus_processing.py](corpus_processing.py) ⭐ **NEW**

Comprehensive example demonstrating batch processing of multiple audio files.

```python
from findsylls.embedding.pipeline import embed_corpus
from findsylls.embedding.storage import save_embeddings, load_embeddings

# Process multiple files in parallel
results = embed_corpus(
    audio_files=['file1.wav', 'file2.wav', 'file3.wav'],
    embedder='mfcc',
    pooling='mean',
    n_jobs=4,  # Parallel processing
    verbose=True
)

# Save to disk
save_embeddings(results, 'corpus_embeddings.npz')

# Load back
results = load_embeddings('corpus_embeddings.npz')
```

**Features demonstrated**:
- Batch processing with parallel execution
- Progress tracking
- Error handling
- NPZ and HDF5 storage formats
- Method comparison workflow
- Best practices for CPU vs GPU models

**Run**: `python examples/corpus_processing.py`

---

## Quick Reference

### All Embedders

| Embedder | Dimensions | Auto-download | Notes |
|----------|-----------|---------------|-------|
| `sylber` | 768 | ✅ Yes | HuggingFace Hub model |
| `vg-hubert` | 768 | ❌ Manual | See VG_HUBERT_README.md |
| `mfcc` | 13/26/39 | N/A | Delta features optional |
| `melspec` | 80 | N/A | Mel-spectrogram |

### All Pooling Methods

- `mean` - Average across time
- `onc` - Onset-Nucleus-Coda (3x dimensions)
- `max` - Maximum across time
- `median` - Median across time

### Parallel Processing Guidelines

**CPU Features** (MFCC, melspec):
```python
results = embed_corpus(audio_files, embedder='mfcc', n_jobs=-1)  # Use all CPUs
```

**GPU Models** (Sylber, VG-HuBERT):
```python
results = embed_corpus(audio_files, embedder='sylber', n_jobs=1)  # Sequential
```

---

## More Information

- **Full documentation**: [docs/EMBEDDING_PIPELINE.md](../docs/EMBEDDING_PIPELINE.md)
- **Quick reference**: [docs/EMBEDDING_QUICKREF.md](../docs/EMBEDDING_QUICKREF.md)
- **VG-HuBERT setup**: [docs/VG_HUBERT_README.md](../docs/VG_HUBERT_README.md)
- **Phase 3 summary**: [docs/PHASE3_SUMMARY.md](../docs/PHASE3_SUMMARY.md)
