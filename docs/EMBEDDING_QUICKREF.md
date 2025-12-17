# Embedding Pipeline - Quick Reference

## Available Embedders

| Embedder | Dimensions | FPS | Notes |
|----------|-----------|-----|-------|
| `'sylber'` | 768 | ~50 | Auto-download from HuggingFace |
| `'vg_hubert'` | 768 | ~50 | Requires `model_path` parameter |
| `'mfcc'` | 13, 26, 39 | ~100 | Use `include_delta` for 26/39-dim |
| `'melspec'` | 80 | ~100 | Mel-scale spectrogram |

## Pooling Methods

| Method | Output Dim | Description |
|--------|-----------|-------------|
| `'mean'` | 1× | Average all frames |
| `'onc'` | 3× | Onset + Nucleus + Coda |
| `'max'` | 1× | Max pooling |
| `'median'` | 1× | Median pooling |

## Quick Examples

### Sylber (Auto-download)
```python
from findsylls.embedding.pipeline import embed_audio

# Mean pooling (768-dim)
emb, meta = embed_audio('audio.wav', embedder='sylber', pooling='mean')

# ONC pooling (2304-dim = 768×3)
emb, meta = embed_audio('audio.wav', embedder='sylber', pooling='onc')
```

### VG-HuBERT (Requires model)
```python
# Download first:
# wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar

emb, meta = embed_audio(
    'audio.wav',
    embedder='vg_hubert',
    pooling='mean',
    embedder_kwargs={'model_path': '/path/to/vg-hubert_3'}
)
```

### MFCC with Deltas
```python
# Standard 13-dim
emb, meta = embed_audio('audio.wav', embedder='mfcc', pooling='mean')

# 26-dim (13 + 13 delta)
emb, meta = embed_audio(
    'audio.wav',
    embedder='mfcc',
    pooling='mean',
    embedder_kwargs={'include_delta': True}
)

# 39-dim (13 + 13 delta + 13 delta-delta)
emb, meta = embed_audio(
    'audio.wav',
    embedder='mfcc',
    pooling='mean',
    embedder_kwargs={
        'include_delta': True,
        'include_delta_delta': True
    }
)
```

### Mix Methods
```python
# Sylber segmentation + MFCC embeddings
emb, meta = embed_audio(
    'audio.wav',
    segmentation='sylber',
    embedder='mfcc',
    pooling='mean'
)

# VG-HuBERT segmentation + VG-HuBERT embeddings
emb, meta = embed_audio(
    'audio.wav',
    segmentation='vg_hubert',
    embedder='vg_hubert',
    pooling='onc',
    segmentation_kwargs={'model_path': '/path/to/vg-hubert_3'},
    embedder_kwargs={'model_path': '/path/to/vg-hubert_3'}
)
```

## Metadata

The `meta` dict contains:
```python
{
    'num_syllables': int,           # Number of syllables
    'boundaries': [(start, end)],   # Syllable boundaries in seconds
    'peaks': [peak_times],          # Nucleus positions in seconds
    'segmentation': 'method_name',  # Segmentation method used
    'embedder': 'method_name',      # Embedder method used
    'pooling': 'method_name',       # Pooling method used
    'embedding_dim': int,           # Final embedding dimension
    'fps': float,                   # Frames per second (features)
    'audio_file': str,              # Input audio path
    'duration': float,              # Audio duration in seconds
    'sample_rate': int,             # Sample rate used
    'created_at': str               # Timestamp
}
```

## Common Patterns

### Extract features only (no pooling)
```python
from findsylls.embedding.extractors import extract_features

features, times = extract_features(audio, sr, method='sylber')
# features: (num_frames, 768)
# times: (num_frames,)
```

### Step-by-step
```python
from findsylls.audio import load_audio
from findsylls.pipeline.pipeline import segment_audio as seg_audio
from findsylls.embedding import extract_features, pool_syllables

audio, sr = load_audio('audio.wav')
syllables, _, _ = seg_audio(audio, sr, method='sylber')
features, times = extract_features(audio, sr, method='mfcc')
embeddings = pool_syllables(features, times, syllables, method='mean', fps=100)
```

## Installation

```bash
# Core dependencies (already installed)
pip install numpy librosa

# For neural models
pip install torch transformers

# Optional: Download VG-HuBERT
wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar
tar -xf vg-hubert_3.tar -C ~/models/
```

## Examples

See `examples/` directory:
- `mfcc_deltas.py` - MFCC delta features demo
- `vg_hubert_embedding.py` - VG-HuBERT complete example

## Tests

```bash
python test_embedding_phase1.py  # All 6 tests should pass
```
