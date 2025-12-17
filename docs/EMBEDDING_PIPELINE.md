# Syllable Embedding Pipeline Architecture

**Status**: Phase 1, 2 & 3 COMPLETE ✅  
**Last Updated**: 2025-12-17

**Implemented Features:**
- ✅ Sylber embedding extraction
- ✅ VG-HuBERT embedding extraction (requires model download)
- ✅ MFCC with delta/delta-delta support
- ✅ Mel-spectrogram extraction
- ✅ Mean, ONC, Max, Median pooling
- ✅ Full pipeline integration
- ✅ **Phase 3**: Corpus batch processing
- ✅ **Phase 3**: Parallel execution (joblib)
- ✅ **Phase 3**: NPZ storage format
- ✅ **Phase 3**: HDF5 storage format (optional)

---

## Overview

The embedding pipeline extends `findsylls` to extract **syllable-level embeddings** from audio, enabling downstream tasks like:
- Syllable clustering and discovery
- Cross-lingual phonetic analysis
- Speech representation learning
- Syllable-based speech recognition

### Complete Pipeline Flow

```
Audio File(s)
    ↓
[Segmentation] ← Already implemented
    ↓
Syllable Boundaries: [(start₁, peak₁, end₁), (start₂, peak₂, end₂), ...]
    ↓
[Feature Extraction] ← NEW: Extract frame-level features
    ↓
Frame Features: (num_frames, feature_dim)
    ↓
[Syllable Pooling] ← NEW: Aggregate frames → syllables
    ↓
Syllable Embeddings: (num_syllables, embedding_dim)
    ↓
[Storage/Output] ← NEW: Save or return embeddings
```

---

## Key Design Decisions

### 1. Separation of Concerns

**Two orthogonal dimensions:**

1. **Embedder/Feature Extractor** (the "what")
   - Sylber - Pre-trained on 400+ languages
   - VG-HuBERT - Vision-grounded HuBERT
   - HuBERT - Base self-supervised model
   - Wav2Vec2 - Another SSL model
   - MFCC - Classical acoustic features
   - Mel-spectrogram - Time-frequency representation

2. **Pooling Method** (the "how")
   - `mean` - Average all frames in syllable span
   - `max` - Max pooling across time
   - `median` - Median pooling
   - `onc` - Onset-Nucleus-Coda template (3× embedding_dim)
   - `weighted_mean` - Attention-weighted averaging
   - `first` / `last` - Single frame selection

**Rationale**: Any embedder can be combined with any pooling method, providing maximum flexibility.

### 2. NumPy-Based Implementation

**Decision**: Use NumPy throughout, no PyTorch conversion needed.

**Reasoning**:
- GPU acceleration only benefits neural network forward passes (already using PyTorch internally)
- Syllable pooling is simple array operations (slicing, averaging) - no GPU benefit
- NumPy is simpler, works everywhere, no extra dependencies for classical methods
- Users can easily convert: `torch.from_numpy(embeddings)` if needed

**Where PyTorch IS used** (already):
- VG-HuBERT model forward pass (GPU-enabled)
- Sylber model forward pass (GPU-enabled)
- Other neural models

**Where PyTorch is NOT needed**:
- Audio I/O (CPU-bound)
- Frame slicing and pooling (trivial operations)
- MFCC/classical feature extraction (librosa is optimized C)

### 3. Storage Format Strategy

**Default: NumPy NPZ (.npz)**
- No extra dependencies (NumPy already required)
- Compressed by default
- Perfect for small-to-medium datasets (< 1GB)
- Simple API: `np.savez_compressed()` / `np.load()`
- Can store multiple arrays + metadata

**Optional: HDF5 (.h5 / .hdf5)**
- For large corpora (> 1GB)
- Memory-mapped access (load only what you need)
- Hierarchical organization
- Better for datasets that don't fit in RAM
- Requires extra dependency: `h5py`

**Not used**:
- Pickle (security risks, not portable)
- Individual .npy files (too many files for large corpora)
- PyTorch .pt files (unnecessary since we use NumPy)

---

## API Design

### Level 1: High-Level Pipeline (Convenience)

```python
from findsylls.embedding import embed_audio

# Single file - get embeddings + metadata
embeddings, metadata = embed_audio(
    audio_path='audio.wav',
    segmentation='sylber',          # or 'vg_hubert', 'peaks_and_valleys', etc.
    embedder='sylber',              # Feature extractor
    pooling='mean',                 # Syllable pooling method
    layer=8,                        # For neural models
    segmentation_kwargs={},         # Pass to segmenter
    embedder_kwargs={}              # Pass to feature extractor
)

# embeddings: np.ndarray, shape (num_syllables, embedding_dim)
# metadata: dict with 'boundaries', 'audio_path', 'duration', etc.

# Ignore metadata if not needed
embeddings, _ = embed_audio('audio.wav', segmentation='sylber', embedder='sylber')
```

### Level 2: Batch Corpus Processing

```python
from findsylls.embedding import embed_corpus

# Process entire corpus
results = embed_corpus(
    audio_paths='data/**/*.wav',     # Glob pattern or list of paths
    segmentation='vg_hubert',
    embedder='vg_hubert',
    pooling='onc',                   # Onset-Nucleus-Coda template
    output_file='embeddings.npz',    # Save to disk
    output_format='npz',             # 'npz' or 'hdf5'
    n_jobs=4,                        # Parallel processing
    return_results=True              # Also return in memory
)

# If output_file is None, only returns in-memory results
# If return_results=False, only saves to disk

# Results structure:
# [
#     {
#         'audio_path': 'file1.wav',
#         'embeddings': np.ndarray,  # (num_syllables, embedding_dim)
#         'metadata': {...}
#     },
#     ...
# ]
```

### Level 3: Step-by-Step (Maximum Control)

```python
from findsylls.audio import load_audio
from findsylls.segmentation import segment_audio
from findsylls.embedding import extract_features, pool_syllables

# Step 1: Load audio
audio, sr = load_audio('audio.wav', sr=16000)

# Step 2: Segment into syllables
syllables, _, _ = segment_audio(audio, method='sylber')
# syllables: [(start, peak, end), ...]

# Step 3: Extract frame-level features
features = extract_features(
    audio, 
    sr=sr,
    method='sylber',     # or 'vg_hubert', 'mfcc', etc.
    layer=8,             # For neural models
    return_times=False   # If True, returns (features, times)
)
# features: np.ndarray, shape (num_frames, feature_dim)

# Step 4: Pool frames into syllable embeddings
embeddings = pool_syllables(
    features,
    syllables=syllables,
    sr=sr,
    method='mean',        # or 'onc', 'max', etc.
    hop_length=160        # For frame timing (16000 sr / 100 fps = 160)
)
# embeddings: np.ndarray, shape (num_syllables, embedding_dim)
```

---

## Module Structure

```
src/findsylls/
├── embedding/                     # NEW MODULE
│   ├── __init__.py               # Public API exports
│   │   # from .pipeline import embed_audio, embed_corpus
│   │   # from .extractors import extract_features
│   │   # from .pooling import pool_syllables
│   │
│   ├── pipeline.py               # High-level APIs
│   │   # embed_audio() - single file convenience function
│   │   # embed_corpus() - batch processing with parallel support
│   │
│   ├── extractors.py             # Feature extraction (frame-level)
│   │   # extract_features() - dispatch function
│   │   # _extract_sylber_features()
│   │   # _extract_vg_hubert_features()
│   │   # _extract_hubert_features()
│   │   # _extract_wav2vec2_features()
│   │   # _extract_mfcc_features()
│   │   # _extract_melspec_features()
│   │
│   ├── pooling.py                # Syllable pooling methods
│   │   # pool_syllables() - dispatch function
│   │   # _pool_mean()
│   │   # _pool_max()
│   │   # _pool_onc() - Onset-Nucleus-Coda template
│   │   # _pool_weighted_mean()
│   │
│   └── storage.py                # Save/load functionality
│       # save_embeddings_npz()
│       # save_embeddings_hdf5()
│       # load_embeddings_npz()
│       # load_embeddings_hdf5()
```

---

## Embedder Methods (Detailed)

### Neural Feature Extractors

#### 1. Sylber
- **Source**: Pre-trained model from HuggingFace (`cheoljun95/sylber`)
- **Model**: Transformer-based encoder (12 layers)
- **Output**: Frame-level features, ~50 fps (frames per second)
- **Embedding dim**: 768 (default layer 8)
- **Typical usage**: Multi-lingual syllable discovery
- **GPU**: Supported (auto-detected)

```python
features = extract_features(audio, sr, method='sylber', layer=8)
# Returns: (num_frames, 768) where num_frames ≈ duration * 50
```

#### 2. VG-HuBERT (Vision-Grounded HuBERT)
- **Source**: Local checkpoint (downloaded separately)
- **Model**: HuBERT with vision grounding trained on SpokenCOCO
- **Output**: Frame-level features, ~50 fps
- **Embedding dim**: 768 (default layer 8)
- **Typical usage**: Cross-lingual syllable analysis
- **GPU**: Supported
- **Download**: https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar

```python
features = extract_features(
    audio, sr, 
    method='vg_hubert',
    model_path='/path/to/vg-hubert_3',  # Required
    layer=8,  # Layer 8 best for syllables
    device='cuda'
)
# Returns: (num_frames, 768) where num_frames ≈ duration * 50
```

#### 3. HuBERT (Base)
- **Source**: `facebook/hubert-base-ls960` from HuggingFace
- **Model**: Self-supervised speech representation
- **Output**: Frame-level features, ~50 fps
- **Embedding dim**: 768
- **Typical usage**: General speech representation
- **GPU**: Supported

```python
features = extract_features(audio, sr, method='hubert', layer=9)
```

#### 4. Wav2Vec2
- **Source**: Various models from HuggingFace
- **Model**: Self-supervised speech representation
- **Output**: Frame-level features, ~50 fps
- **Embedding dim**: 768 or 1024 (model-dependent)
- **GPU**: Supported

```python
features = extract_features(
    audio, sr,
    method='wav2vec2',
    model_name='facebook/wav2vec2-base',
    layer=11
)
```

### Classical Feature Extractors

#### 5. MFCC (Mel-Frequency Cepstral Coefficients)
- **Source**: librosa
- **Output**: Frame-level features, configurable fps (default ~100)
- **Embedding dim**: Configurable (default 13, or 26/39 with deltas)
- **Typical usage**: Classical baseline
- **GPU**: Not needed (already optimized)

```python
# Standard MFCCs (13-dim)
features = extract_features(
    audio, sr,
    method='mfcc',
    n_mfcc=13,
    hop_length=160,  # 100 fps at 16kHz
    n_fft=400
)
# Returns: (num_frames, 13)

# With delta (first-order derivatives) → 26-dim
features = extract_features(
    audio, sr,
    method='mfcc',
    n_mfcc=13,
    include_delta=True
)
# Returns: (num_frames, 26) = [mfcc, delta_mfcc]

# With delta + delta-delta (second-order derivatives) → 39-dim
features = extract_features(
    audio, sr,
    method='mfcc',
    n_mfcc=13,
    include_delta=True,
    include_delta_delta=True
)
# Returns: (num_frames, 39) = [mfcc, delta, delta_delta]
```

**Note**: `include_delta_delta=True` requires `include_delta=True` to take effect.

#### 6. Mel-Spectrogram
- **Source**: librosa
- **Output**: Frame-level mel-scale power spectrogram
- **Embedding dim**: n_mels (default 80)
- **Typical usage**: Time-frequency representation
- **GPU**: Not needed

```python
features = extract_features(
    audio, sr,
    method='melspec',
    n_mels=80,
    hop_length=160
)
# Returns: (num_frames, 80)
```

---

## Pooling Methods (Detailed)

### 1. Mean Pooling
**Most common approach** - Average all frames within syllable boundaries.

```python
embeddings = pool_syllables(features, syllables, sr, method='mean')
# Output shape: (num_syllables, feature_dim)
```

**Implementation**:
```python
for start, peak, end in syllables:
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    embedding = features[start_frame:end_frame].mean(axis=0)
```

**Best for**: General-purpose syllable representation

---

### 2. ONC (Onset-Nucleus-Coda) Template
**Linguistic template** - Captures syllable internal structure.

```python
embeddings = pool_syllables(features, syllables, sr, method='onc')
# Output shape: (num_syllables, 3 * feature_dim)
```

**Current Implementation**:
```python
# Extract single frames at specific timepoints:
for start, peak, end in syllables:
    # Onset: 30% from start to peak
    t_onset = start + 0.3 * (peak - start)
    idx_onset = int(round(t_onset / frame_hop))
    
    # Nucleus: peak frame
    idx_nucleus = int(round(peak / frame_hop))
    
    # Coda: 70% from peak to end
    t_coda = peak + 0.7 * (end - peak)
    idx_coda = int(round(t_coda / frame_hop))
    
    embedding = np.concatenate([
        features[idx_onset],    # single onset frame
        features[idx_nucleus],  # single nucleus frame
        features[idx_coda]      # single coda frame
    ])  # 3× feature_dim
```

**Peak Detection**:
ONC pooling requires syllable peaks to identify the nucleus. Peaks are provided by:

1. **Native peak detection** (recommended for Sylber):
   - **`embedder='sylber'`**: When using `embedder='sylber'` + `pooling='onc'`, peaks are detected using **cosine similarity** between consecutive frames within each segment (native to Sylber's learned representation)
   - **`segmentation='sylber'`**: When using `segmentation='sylber'` + `pooling='onc'`, peaks are ALSO detected via cosine similarity using the frame-level features already computed during segmentation (minimal overhead!)
   - Both approaches find the most stable/prototypical frame within each syllable
   - Example: `embed_audio('audio.wav', embedder='sylber', pooling='onc')` or `embed_audio('audio.wav', segmentation='sylber', embedder='mfcc', pooling='onc')`

2. **Envelope-based peaks**:
   - `segmentation='peaks_and_valleys'`: Provides real acoustic peaks from amplitude envelope
   - No warning shown (these are genuine acoustic peaks)

3. **Midpoint fallback**:
   - Other segmentation methods use midpoints as peak proxies (may not align with phonetic nuclei)
   - A warning is shown in these cases

**Warnings**:
```python
# Warning 1: embedder='sylber' uses cosine similarity peak detection
embed_audio('audio.wav', embedder='sylber', pooling='onc')
# UserWarning: ONC pooling with embedder='sylber' detects peaks using cosine 
# similarity between frames. While this is native to Sylber's representation 
# space, Sylber was not explicitly trained for peak detection...

# Warning 2: segmentation='sylber' also uses cosine similarity (minimal overhead!)
embed_audio('audio.wav', segmentation='sylber', embedder='mfcc', pooling='onc')
# UserWarning: ONC pooling with segmentation='sylber' detects peaks using 
# cosine similarity between frames in Sylber's representation space. While 
# this uses data already computed during segmentation (minimal overhead), 
# Sylber was not explicitly trained for peak detection...

# No warning: peaks_and_valleys provides real acoustic peaks
embed_audio('audio.wav', segmentation='peaks_and_valleys', embedder='mfcc', pooling='onc')
```

**Understanding the difference**:
- `segmentation='sylber'`: Uses Sylber to **find boundaries** → can detect peaks via cosine similarity → works with ANY embedder
- `embedder='sylber'`: Uses Sylber to **extract features** → can detect peaks via cosine similarity → segments + embeds in one step

**Future Enhancement** (Planned):
- Select onset/coda points based on **maximal velocity** in a frequency band
- Onset: point of maximal velocity before peak (better captures acoustic transition)
- Coda: point of maximal velocity after peak
- Will better represent actual phonetic boundaries

**Best for**: 
- Preserving syllable-internal phonetic structure
- Cross-linguistic analysis where onset/coda matter
- When syllable position information is important

**Note**: Output dimension is **3× the input feature dimension**
- Sylber (768) → ONC (2304)
- MFCC (13) → ONC (39)

---

### 3. Max Pooling
**Picks strongest activation** in each dimension.

```python
embeddings = pool_syllables(features, syllables, sr, method='max')
```

**Best for**: Capturing salient features, noise robustness

---

### 4. Median Pooling
**Robust to outliers**.

```python
embeddings = pool_syllables(features, syllables, sr, method='median')
```

**Best for**: Noisy audio, robustness to extreme values

---

### 5. Weighted Mean
**Attention-weighted averaging** (future implementation).

```python
embeddings = pool_syllables(
    features, syllables, sr,
    method='weighted_mean',
    attention_weights=attention_scores  # From model
)
```

**Best for**: When model provides attention scores (e.g., VG-HuBERT with attention)

---

## Metadata Structure

Each embedding comes with comprehensive metadata:

```python
metadata = {
    # Syllable information
    'boundaries': [(start1, end1), (start2, end2), ...],  # In seconds
    'peaks': [peak1, peak2, ...],                         # Nucleus proxies
    'num_syllables': 15,
    
    # Audio information
    'audio_path': '/path/to/audio.wav',
    'duration': 3.45,                                     # Total duration in seconds
    'sample_rate': 16000,
    
    # Method information
    'segmentation_method': 'sylber',
    'embedder': 'sylber',
    'pooling': 'mean',
    'layer': 8,                                           # For neural models
    
    # Embedding information
    'embedding_dim': 768,
    'fps': 50.0,                                          # Frames per second of features
    'hop_length': 160,                                    # Samples per frame
    
    # Timestamps
    'created_at': '2025-12-17T10:30:00',
    'findsylls_version': '0.1.0'
}
```

---

## Storage Format Specifications

### NPZ Format (Default)

**Structure**:
```python
# Save
np.savez_compressed(
    'embeddings.npz',
    embeddings_file1=emb1,
    embeddings_file2=emb2,
    metadata=np.array([{...}, {...}], dtype=object)
)

# Load
data = np.load('embeddings.npz', allow_pickle=True)
emb1 = data['embeddings_file1']
metadata = data['metadata'].item()  # Get dict from object array
```

**Naming convention**: `embeddings_{sanitized_filename}`
- Spaces → underscores
- Special chars removed
- Extension stripped

---

### HDF5 Format (Optional)

**Structure**:
```python
# Hierarchical organization
corpus_embeddings.h5
├── audio1.wav/
│   ├── embeddings       [dataset: (15, 768)]
│   └── attributes:
│       ├── duration: 3.45
│       ├── segmentation: 'sylber'
│       ├── embedder: 'sylber'
│       └── ...
├── audio2.wav/
│   ├── embeddings       [dataset: (20, 768)]
│   └── attributes: ...
└── _corpus_metadata/
    └── attributes:
        ├── created_at: '2025-12-17'
        ├── total_files: 100
        └── ...
```

**Usage**:
```python
import h5py

# Save
with h5py.File('embeddings.h5', 'w') as f:
    grp = f.create_group('audio1.wav')
    grp.create_dataset('embeddings', data=emb1)
    grp.attrs['duration'] = 3.45
    grp.attrs['segmentation'] = 'sylber'

# Load (memory-mapped)
with h5py.File('embeddings.h5', 'r') as f:
    emb1 = f['audio1.wav/embeddings'][:]  # Load into memory
    # Or access without loading: f['audio1.wav/embeddings'][0:10]
```

---

## Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETE
**Files created:**
1. ✅ `src/findsylls/embedding/__init__.py`
2. ✅ `src/findsylls/embedding/extractors.py` (Sylber + MFCC with deltas)
3. ✅ `src/findsylls/embedding/pooling.py` (mean + ONC + max + median)
4. ✅ `src/findsylls/embedding/pipeline.py` (embed_audio)

**Achieved**: Basic single-file embedding working
```python
embeddings, meta = embed_audio('audio.wav', segmentation='sylber', embedder='sylber', pooling='mean')
```

### Phase 2: Additional Extractors ✅ COMPLETE
**Added to extractors.py:**
- ✅ VG-HuBERT feature extraction (requires model download)
- ✅ Mel-spectrogram extraction
- ✅ MFCC delta and delta-delta support

**Usage:**
```python
# VG-HuBERT
embed_audio('audio.wav', embedder='vg_hubert', 
            embedder_kwargs={'model_path': '/path/to/vg-hubert_3'})

# MFCC with deltas (39-dim)
embed_audio('audio.wav', embedder='mfcc',
            embedder_kwargs={'include_delta': True, 'include_delta_delta': True})
```

### Phase 3: Corpus Processing (Priority 3) - TODO
**Add to pipeline.py:**
- `embed_corpus()` function
- Parallel processing support (joblib)
- Progress bars (tqdm)

**Create:**
- `src/findsylls/embedding/storage.py`

### Phase 4: Advanced Features (Priority 4) - TODO
**Add to extractors.py:**
- HuBERT feature extraction (transformers)
- Wav2Vec2 support
- Other SSL models

**Add to pooling.py:**
- Weighted pooling
- Attention-based pooling

### Stretch Goals (Future)
**VG-HuBERT HuggingFace Hub Integration:**
- Fork VG-HuBERT repository
- Convert to HuggingFace model format
- Upload to Hub (e.g., `harwath/vg-hubert-base`)
- Enable auto-download like Sylber
- Remove manual wget/tar requirement
- **Benefit**: Consistent API across all neural models

---

## Usage Examples

### Example 1: Quick Single-File Embedding

```python
from findsylls.embedding import embed_audio

# Simplest usage
embeddings, meta = embed_audio(
    'speech.wav',
    segmentation='sylber',
    embedder='sylber',
    pooling='mean'
)

print(f"Extracted {len(embeddings)} syllable embeddings")
print(f"Each embedding: {embeddings.shape[1]} dimensions")
print(f"Boundaries: {meta['boundaries']}")
```

### Example 2: ONC Template for Linguistic Analysis

```python
from findsylls.embedding import embed_audio

# Use ONC template to preserve syllable structure
embeddings, meta = embed_audio(
    'speech.wav',
    segmentation='vg_hubert',
    embedder='vg_hubert',
    pooling='onc',  # 3× dimensions
    layer=6
)

# embeddings.shape: (num_syllables, 2304) for 768-dim features
# Each row: [onset_768, nucleus_768, coda_768]
```

### Example 3: Process Entire Corpus

```python
from findsylls.embedding import embed_corpus

# Process all .wav files in directory
results = embed_corpus(
    audio_paths='data/**/*.wav',
    segmentation='sylber',
    embedder='sylber',
    pooling='mean',
    output_file='corpus_embeddings.npz',
    n_jobs=8,  # Parallel processing
    return_results=False  # Don't keep in memory
)

# Embeddings saved to corpus_embeddings.npz
```

### Example 4: Load Saved Embeddings

```python
import numpy as np

# Load from NPZ
data = np.load('corpus_embeddings.npz', allow_pickle=True)
print("Files in corpus:", [k for k in data.keys() if k.startswith('embeddings_')])

# Get specific file
emb1 = data['embeddings_file1']
metadata = data['metadata'].item()[0]  # First file's metadata

# Use embeddings
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=50)
labels = kmeans.fit_predict(emb1)
print(f"Clustered into {len(set(labels))} syllable types")
```

### Example 5: MFCC with Delta Features (39-dim)

```python
from findsylls.embedding import embed_audio

# Standard 39-dimensional MFCC features (13 + 13 deltas + 13 delta-deltas)
embeddings, meta = embed_audio(
    'speech.wav',
    segmentation='peaks_and_valleys',
    embedder='mfcc',
    pooling='mean',
    embedder_kwargs={
        'n_mfcc': 13,
        'include_delta': True,
        'include_delta_delta': True
    }
)

print(f"Shape: {embeddings.shape}")  # (num_syllables, 39)

# With ONC pooling: 39 × 3 = 117 dimensions
embeddings_onc, _ = embed_audio(
    'speech.wav',
    segmentation='peaks_and_valleys',
    embedder='mfcc',
    pooling='onc',
    embedder_kwargs={
        'n_mfcc': 13,
        'include_delta': True,
        'include_delta_delta': True
    }
)

print(f"ONC shape: {embeddings_onc.shape}")  # (num_syllables, 117)
```

### Example 6: Step-by-Step with Custom Processing

```python
from findsylls.audio import load_audio
from findsylls.segmentation import segment_audio
from findsylls.embedding import extract_features, pool_syllables
import numpy as np

# Load and segment
audio, sr = load_audio('speech.wav')
syllables, _, _ = segment_audio(audio, method='sylber')

# Extract features with custom settings
features = extract_features(
    audio, sr,
    method='mfcc',
    n_mfcc=20,  # More coefficients
    n_fft=512,  # Custom FFT size
    hop_length=160
)

# Filter short syllables (< 50ms)
long_syllables = [(s, p, e) for s, p, e in syllables if (e - s) > 0.05]

# Pool with custom method
embeddings = pool_syllables(features, long_syllables, sr, method='mean')

# Custom post-processing
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

---

## Testing Strategy

### Unit Tests

**extractors.py**:
- Test each extractor returns correct shape
- Test with various audio lengths
- Test error handling (invalid method, missing model)

**pooling.py**:
- Test each pooling method returns correct shape
- Test boundary edge cases (start=0, end=duration)
- Test empty syllable list
- Verify ONC returns 3× dimensions

**pipeline.py**:
- Test embed_audio with all method combinations
- Test metadata structure
- Test error handling

**storage.py**:
- Test NPZ save/load
- Test HDF5 save/load
- Test large corpus handling

### Integration Tests

- Test full pipeline: audio → embeddings
- Test corpus processing with multiple files
- Test parallel processing
- Test various segmentation + embedding combinations

### Performance Tests

- Benchmark extractors (features/second)
- Benchmark pooling methods
- Memory usage with large corpora
- Parallel speedup measurements

---

## Phase 3: Corpus Processing (IMPLEMENTED ✅)

### Overview

Phase 3 adds batch processing capabilities for handling multiple audio files efficiently:

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

### Features Implemented

#### 1. `embed_corpus()` Function

**Location**: `src/findsylls/embedding/pipeline.py`

Batch processes multiple audio files with:
- Parallel execution via `joblib` (configurable with `n_jobs`)
- Progress tracking with `tqdm`
- Error handling (skip or fail modes)
- Consistent metadata across files

**Parameters**:
- `audio_files`: List of paths
- `n_jobs`: Number of parallel workers (-1 = all CPUs, 1 = sequential)
- `verbose`: Show progress bar
- `fail_on_error`: True = raise on error, False = skip failed files
- All other params same as `embed_audio()`

**Returns**: List of result dicts:
```python
{
    'audio_path': str,
    'embeddings': np.ndarray,
    'metadata': dict,
    'success': bool,
    'error': str or None
}
```

#### 2. Storage Utilities

**Location**: `src/findsylls/embedding/storage.py`

Two storage formats supported:

**NPZ Format** (NumPy compressed):
- Simple, fast, portable
- Good for small-medium corpora (< 1000 files)
- Functions: `save_embeddings_npz()`, `load_embeddings_npz()`

**HDF5 Format** (optional, requires h5py):
- Hierarchical structure
- Efficient partial loading
- Good for large corpora (> 1000 files)
- Functions: `save_embeddings_hdf5()`, `load_embeddings_hdf5()`

**Auto-detection**: `save_embeddings()` and `load_embeddings()` automatically detect format from file extension (.npz or .h5/.hdf5)

#### 3. Best Practices

**Parallel Processing Guidelines**:
- **CPU features** (MFCC, melspec): Use `n_jobs=-1` for all CPUs
- **GPU models** (Sylber, VG-HuBERT): Use `n_jobs=1` to avoid memory issues

**Storage Format Selection**:
- **Small corpora** (< 100 files): NPZ is fine
- **Medium corpora** (100-1000 files): NPZ or HDF5
- **Large corpora** (> 1000 files): HDF5 with partial loading

**Error Handling**:
- Set `fail_on_error=False` for production pipelines to skip problematic files
- Check `result['success']` and `result['error']` fields

### Example Workflows

#### Basic Corpus Processing

```python
from pathlib import Path
from findsylls.embedding.pipeline import embed_corpus
from findsylls.embedding.storage import save_embeddings

# Get all WAV files in directory
audio_files = list(Path('corpus/').glob('*.wav'))

# Process with MFCC features
results = embed_corpus(
    audio_files,
    embedder='mfcc',
    pooling='mean',
    n_jobs=4,
    verbose=True,
    fail_on_error=False
)

# Save results
save_embeddings(results, 'corpus_mfcc.npz')
```

#### Method Comparison

```python
from findsylls.embedding.pipeline import embed_audio

methods = ['mfcc', 'melspec', 'sylber']
test_file = 'sample.wav'

for method in methods:
    embeddings, meta = embed_audio(
        test_file,
        embedder=method,
        pooling='mean'
    )
    print(f"{method}: {embeddings.shape}")
```

#### Large Corpus with HDF5

```python
# Process large corpus
results = embed_corpus(
    audio_files,  # 10,000+ files
    embedder='sylber',
    n_jobs=1,  # Sequential for GPU
    verbose=True
)

# Save to HDF5
save_embeddings(results, 'large_corpus.h5')

# Later: Load only specific files
from findsylls.embedding.storage import load_embeddings_hdf5
subset = load_embeddings_hdf5('large_corpus.h5', file_indices=[0, 100, 200])
```

### Testing

**Test File**: `tests/test_corpus.py`

Tests implemented:
1. ✅ Basic `embed_corpus()` functionality
2. ✅ Parallel processing with `n_jobs`
3. ✅ Error handling (skip failed files)
4. ✅ NPZ save/load with data integrity
5. ✅ HDF5 save/load (if h5py installed)
6. ✅ Auto format detection

All tests passing (5/6 passed, 1/6 skipped without h5py).

### Examples

**Complete example**: `examples/corpus_processing.py`

Demonstrates:
- Batch processing workflow
- Parallel execution
- NPZ and HDF5 storage
- Method comparison
- Best practices

Run with: `python examples/corpus_processing.py`

---

## Dependencies

### Required (Core)
- `numpy` - Core array operations
- `librosa` - Audio processing, MFCC extraction
- `soundfile` or `torchaudio` - Audio I/O

### Required (Phase 3 - Corpus Processing)
- `joblib` - Parallel processing
- `tqdm` - Progress bars

### Optional (Neural Extractors)
- `torch` - PyTorch models (Sylber, VG-HuBERT)
- `transformers` - HuggingFace models (Sylber)

### Optional (Large Corpus Storage)
- `h5py` - HDF5 format support

Install Phase 3 dependencies:
```bash
pip install joblib tqdm h5py
```

---

## Open Questions / Future Work

1. **Streaming**: Support for streaming large files (process in chunks)?
2. **Caching**: Cache extracted features to avoid re-extraction?
3. **Multi-modal**: Support vision features (for VG-HuBERT)?
4. **Frame alignment**: More sophisticated frame-to-time alignment methods?
5. **Syllable features**: Extract duration, F0, energy as additional features?
6. **Distributed processing**: Support for cluster computing (Dask, Ray)?

---

## References & Related Work

### Papers on Syllable Embeddings
- Peng et al. (2023) - VG-HuBERT syllable discovery
- Kreuk et al. (2020) - Self-supervised syllable discovery with audio-visual grounding

### Related Toolkits
- SpeechBrain - Feature extraction utilities
- ESPnet - End-to-end speech processing
- Fairseq - Self-supervised speech models

---

**Document Status**: PHASES 1, 2 & 3 COMPLETE ✅  
**Implementation Status**: Fully implemented and tested  
**Next Steps**: Deploy and gather user feedback
