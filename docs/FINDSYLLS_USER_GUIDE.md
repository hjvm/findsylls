# findsylls: Comprehensive User Guide

**Version**: 1.1.0  
**Last Updated**: February 17, 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Architecture](#core-architecture)
5. [Segmentation Methods](#segmentation-methods)
6. [Embedding Pipeline](#embedding-pipeline)
7. [Evaluation Framework](#evaluation-framework)
8. [Complete Examples](#complete-examples)
9. [API Reference](#api-reference)
10. [Performance & Benchmarking](#performance--benchmarking)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is findsylls?

**findsylls** is a comprehensive, production-ready Python toolkit for syllable identification, tokenization, segmentation, and embedding extraction from speech audio. It provides:

- **11+ Segmentation Methods**: Classical envelope-based (SBS, Theta, Hilbert) and modern neural methods (Sylber, VG-HuBERT, SyllableLM)
- **Modular Architecture**: Mix and match feature extractors with segmentation algorithms
- **Embedding Pipeline**: Extract syllable-level embeddings for downstream tasks
- **Evaluation Framework**: Multi-granular evaluation (nuclei, boundaries, spans) against TextGrid annotations
- **Production Ready**: Parallel processing, progress tracking, flexible storage formats

### Use Cases

- **Speech research**: Analyze syllable structure across languages and speaking styles
- **Phonetics**: Extract syllable-level acoustic features
- **Language acquisition**: Study child-directed speech segmentation
- **Computational linguistics**: Tokenize speech for downstream NLP tasks
- **Model evaluation**: Benchmark segmentation algorithms on multiple corpora

---

## Installation

### Basic Installation

```bash
pip install findsylls
```

### With Neural Methods (recommended)

```bash
# Install with embedding support (Sylber, HuBERT)
pip install 'findsylls[embedding]'

# For VG-HuBERT support
pip install vg-hubert
```

### From Source (development)

```bash
git clone https://github.com/hjvm/findsylls.git
cd findsylls
pip install -e '.[embedding]'
```

### Dependencies

**Core** (always installed):
- numpy, scipy
- librosa (audio processing)
- textgrid (annotation parsing)
- pandas (result formatting)

**Optional** (for neural methods):
- torch, torchaudio
- transformers (HuBERT)
- sylber (Sylber method)
- vg-hubert (VG-HuBERT method)

---

## Quick Start

### 1. Basic Segmentation

```python
from findsylls.pipeline.pipeline import segment_audio

# Segment audio file using SBS envelope method
syllables, envelope, times = segment_audio(
    'speech.wav',
    envelope_fn='sbs',           # Spectral Band Subtraction
    segment_fn='peakdetect',
    segmentation_kwargs={
        'delta': 0.01,
        'min_syllable_dur': 0.05
    }
)

# syllables: List[(start, peak, end)] in seconds
for start, peak, end in syllables:
    print(f"Syllable: {start:.3f}s - {end:.3f}s (nucleus at {peak:.3f}s)")
```

### 2. Neural Segmentation

```python
from findsylls.segmentation import get_segmenter

# Use Sylber (end-to-end neural method)
segmenter = get_segmenter('sylber')
syllables = segmenter.segment(audio, sr=16000)

# Use VG-HuBERT with MinCut
segmenter = get_segmenter(
    'vg_hubert',
    sec_per_syllable=0.22,
    use_optimized=True
)
syllables = segmenter.segment(audio, sr=16000)
```

### 3. Extract Embeddings

```python
from findsylls.embedding import embed_audio

# Extract MFCC embeddings with mean pooling
embeddings, metadata = embed_audio(
    'speech.wav',
    segmentation='sbs',      # Use SBS for segmentation
    embedder='mfcc',         # Extract MFCC features
    pooling='mean',          # Average features per syllable
    embedder_kwargs={'n_mfcc': 13}
)

# embeddings: (num_syllables, 13) NumPy array
print(f"Extracted {embeddings.shape[0]} syllable embeddings")
print(f"Feature dimension: {embeddings.shape[1]}")
```

### 4. Evaluate Against Annotations

```python
from findsylls.evaluation.evaluator import evaluate_segmentation

# Evaluate segmentation quality
results = evaluate_segmentation(
    peaks=[p for (_, p, _) in syllables],
    spans=[(s, e) for (s, _, e) in syllables],
    textgrid_path='speech.TextGrid',
    tiers={'phone': 2, 'syllable': 1, 'word': 0},
    tolerance=0.05  # 50ms boundary tolerance
)

# Results contains F1 scores for multiple granularities
print(f"Nuclei F1: {results['nuclei']['F1']:.3f}")
print(f"Syllable Boundary F1: {results['syllable_boundaries']['F1']:.3f}")
print(f"Word Span F1: {results['word_spans']['F1']:.3f}")
```

---

## Core Architecture

### Three-Layer Design

findsylls uses a modular three-layer architecture:

```
┌────────────────────────────────────────────────────────┐
│         LAYER 1: End-to-End Segmenters                 │
│    (User-facing, model-specific presets)               │
│    - SylberSegmenter                                   │
│    - VGHubertSegmenter (MinCut/CLS)                    │
│    - SyllableLMSegmenter                               │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│         LAYER 2: Feature Extractors                    │
│    (Standalone feature extraction)                     │
│    - HuBERTExtractor, MFCCExtractor                    │
│    - SylberFeatureExtractor                            │
│    - VGHuBERTFeatureExtractor                          │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│       LAYER 3: Segmentation Algorithms                 │
│    (Pure algorithms, no feature extraction)            │
│    - min_cut_optimized()                               │
│    - greedy_cosine_segment()                           │
│    - segment_peakdetect() (classical)                  │
└────────────────────────────────────────────────────────┘
```

### Module Organization

```
findsylls/
├── audio/              # Audio I/O and normalization
│   └── utils.py        # load_audio(), match_wavs_to_textgrids()
├── envelope/           # Classical signal processing
│   ├── sbs.py          # Spectral Band Subtraction
│   ├── theta.py        # Theta Oscillator
│   ├── hilbert.py      # Hilbert Transform
│   └── dispatch.py     # get_amplitude_envelope()
├── segmentation/       # Segmentation algorithms
│   ├── peakdetect_segmenter.py  # Peak/valley detection
│   ├── mincut.py                # MinCut algorithm
│   ├── greedy_cosine.py         # Greedy Cosine algorithm
│   ├── presets.py               # End-to-end segmenters
│   └── dispatch.py              # get_segmenter()
├── features/           # Feature extraction (Layer 2)
│   ├── hubert.py       # HuBERT features
│   ├── mfcc.py         # MFCC features
│   └── extractors.py   # All extractors
├── embedding/          # Syllable embedding pipeline
│   ├── pipeline.py     # embed_audio(), embed_corpus()
│   ├── extractors.py   # Feature extractors
│   ├── pooling.py      # Pooling strategies
│   └── storage.py      # Save/load utilities
├── evaluation/         # Evaluation framework
│   ├── evaluator.py    # evaluate_segmentation()
│   ├── nuclei.py       # Nuclei evaluation
│   ├── boundaries.py   # Boundary evaluation
│   └── spans.py        # Span evaluation
├── parsing/            # TextGrid parsing
│   └── textgrid.py     # Extract intervals from tiers
└── pipeline/           # High-level APIs
    ├── pipeline.py     # segment_audio(), run_evaluation()
    └── results.py      # flatten_results(), aggregate_results()
```

---

## Segmentation Methods

### Classical Envelope-Based Methods

These methods compute a signal envelope and detect syllable nuclei as peaks:

#### 1. Spectral Band Subtraction (SBS)

**Best for**: General-purpose, robust to noise

```python
syllables, env, times = segment_audio(
    'audio.wav',
    envelope_fn='sbs',
    envelope_kwargs={
        'pivot_freq': 3000,          # Frequency pivot (Hz)
        'nfft': 256,                 # FFT size
        'window_length': 256,
        'step': 160,
        'smoothing_window_samples': 7
    },
    segmentation_kwargs={
        'delta': 0.01,               # Peak prominence
        'min_syllable_dur': 0.05     # Minimum syllable duration (s)
    }
)
```

**How it works**: Subtracts high-frequency energy from low-frequency energy to emphasize syllable nuclei (vowels have more low-frequency energy).

#### 2. Theta Oscillator

**Best for**: Rhythmic speech, mimics neural oscillations

```python
syllables, env, times = segment_audio(
    'audio.wav',
    envelope_fn='theta',
    envelope_kwargs={
        'bands': 20,                 # Gammatone filterbank bands
        'minfreq': 50,
        'maxfreq': 7500,
        'resample_rate': 1000,
        'f': 5,                      # Oscillator frequency (Hz)
        'Q': 0.5,                    # Damping (0.5 = critical)
        'N': 10                      # Top N bands
    }
)
```

**How it works**: Uses gammatone filterbank + theta-band oscillator model inspired by neural processing.

**Reference**: Räsänen et al. (2018)

#### 3. Hilbert Transform

**Best for**: Fast, simple amplitude envelope

```python
syllables, env, times = segment_audio(
    'audio.wav',
    envelope_fn='hilbert',
    envelope_kwargs={'smoothing_window': 0.02}  # 20ms smoothing
)
```

**How it works**: Computes analytic signal envelope using Hilbert transform.

### Neural End-to-End Methods

These methods use pre-trained neural models for feature extraction and sophisticated segmentation algorithms:

#### 4. Sylber

**Best for**: High accuracy, learned syllable-specific features

```python
from findsylls.segmentation import get_segmenter

segmenter = get_segmenter(
    'sylber',
    norm_threshold=2.6,      # Feature norm threshold
    merge_threshold=0.8      # Cosine similarity threshold
)
syllables = segmenter.segment(audio, sr=16000)
```

**How it works**: 
- Extracts 768-dim features using Sylber model (trained on syllable-level tasks)
- Uses greedy cosine similarity merging algorithm
- Two-phase: (1) merge similar frames, (2) refine boundaries

**Speed**: ~50 fps (faster than real-time on GPU)

**Reference**: Park et al. (2024, ICLR)

#### 5. VG-HuBERT (MinCut)

**Best for**: Cross-lingual, visually-grounded features

```python
segmenter = get_segmenter(
    'vg_hubert',
    sec_per_syllable=0.22,   # Expected syllable duration
    use_optimized=True       # Use optimized MinCut (20-50× faster)
)
syllables = segmenter.segment(audio, sr=16000)
```

**How it works**:
- Extracts 768-dim features from VG-HuBERT (trained on audio-image pairs)
- Computes self-similarity matrix (SSM)
- Applies optimized MinCut dynamic programming algorithm

**Speed**: ~50 fps

**Reference**: Peng et al. (2023, Interspeech); Baade et al. (2024) for optimized MinCut

#### 6. VG-HuBERT (CLS Attention)

**Best for**: Word-level segmentation

```python
segmenter = get_segmenter(
    'vg_hubert_cls',
    attn_threshold=0.1,      # Attention peak height
    min_distance=0.2         # Minimum distance between peaks (s)
)
syllables = segmenter.segment(audio, sr=16000)
```

**How it works**:
- Extracts CLS token attention from VG-HuBERT layer 9
- Finds peaks in attention scores using scipy.signal.find_peaks
- Peaks correspond to salient positions (word boundaries)

**Speed**: ~50 fps

#### 7. SyllableLM

**Best for**: State-of-the-art accuracy (if you have HuBERT features)

```python
segmenter = get_segmenter(
    'syllablelm',
    sec_per_syllable=0.22,
    use_optimized=True
)
syllables = segmenter.segment(audio, sr=16000)
```

**How it works**:
- Uses HuBERT features (facebook/hubert-base-ls960)
- Applies optimized MinCut algorithm (same as VG-HuBERT)
- Can also accept pre-extracted features from Data2Vec2

**Speed**: ~50 fps

**Reference**: Baade et al. (2024)

### Method Comparison

| Method | Type | Speed (fps) | GPU Required | F1 (typical) | Best For |
|--------|------|------------|--------------|--------------|----------|
| SBS | Classical | ~100 | No | 0.50-0.65 | General purpose |
| Theta | Classical | ~50 | No | 0.55-0.70 | Rhythmic speech |
| Hilbert | Classical | ~200 | No | 0.45-0.60 | Speed priority |
| Sylber | Neural | ~50 | Optional | 0.75-0.85 | Accuracy priority |
| VG-HuBERT (MinCut) | Neural | ~50 | Optional | 0.70-0.80 | Cross-lingual |
| VG-HuBERT (CLS) | Neural | ~50 | Optional | 0.65-0.75 | Word boundaries |
| SyllableLM | Neural | ~50 | Optional | 0.75-0.85 | State-of-the-art |

**Note**: F1 scores vary by dataset. Neural methods typically outperform classical methods but require more resources.

---

## Embedding Pipeline

The embedding pipeline extracts syllable-level feature vectors for downstream tasks (clustering, classification, similarity).

### Workflow

```
Audio → Segmentation → Feature Extraction → Pooling → Embeddings
```

### Basic Usage

```python
from findsylls.embedding import embed_audio

embeddings, metadata = embed_audio(
    'audio.wav',
    segmentation='sbs',       # Any segmentation method
    embedder='mfcc',          # Feature extractor
    pooling='mean',           # Pooling strategy
    embedder_kwargs={'n_mfcc': 13}
)

# embeddings: (num_syllables, feature_dim) NumPy array
# metadata: dict with pipeline configuration and statistics
```

### Feature Extractors

#### 1. MFCC (Mel-Frequency Cepstral Coefficients)

**Best for**: Classical acoustic features, computational efficiency

```python
embeddings, meta = embed_audio(
    'audio.wav',
    embedder='mfcc',
    embedder_kwargs={
        'n_mfcc': 13,            # Number of coefficients
        'include_deltas': False  # Add delta and delta-delta features
    },
    pooling='mean'
)
# Output: (num_syllables, 13) or (num_syllables, 39) with deltas
```

**Frame rate**: ~100 fps  
**Dimension**: 13 (or 39 with deltas)

#### 2. Mel-Spectrogram

**Best for**: Time-frequency representation

```python
embeddings, meta = embed_audio(
    'audio.wav',
    embedder='melspec',
    embedder_kwargs={
        'n_mels': 80,
        'n_fft': 400,
        'hop_length': 320
    },
    pooling='mean'
)
# Output: (num_syllables, 80)
```

**Frame rate**: ~100 fps  
**Dimension**: 80 (mel bands)

#### 3. Sylber

**Best for**: Syllable-specific learned features

```python
embeddings, meta = embed_audio(
    'audio.wav',
    segmentation='sylber',    # Use Sylber for both!
    embedder='sylber',
    pooling='mean'
)
# Output: (num_syllables, 768)
```

**Frame rate**: ~50 fps  
**Dimension**: 768

#### 4. VG-HuBERT

**Best for**: Visually-grounded, cross-lingual features

```python
embeddings, meta = embed_audio(
    'audio.wav',
    embedder='vg_hubert',
    embedder_kwargs={
        'layer': 8,              # Layer to extract (8-10 work well)
        'device': 'cuda'         # Use GPU if available
    },
    pooling='mean'
)
# Output: (num_syllables, 768)
```

**Frame rate**: ~50 fps  
**Dimension**: 768

#### 5. HuBERT

**Best for**: General speech representation

```python
embeddings, meta = embed_audio(
    'audio.wav',
    embedder='hubert',
    embedder_kwargs={
        'model_name': 'facebook/hubert-base-ls960',
        'layer': 9               # Layer to extract
    },
    pooling='mean'
)
# Output: (num_syllables, 768)
```

**Frame rate**: ~50 fps  
**Dimension**: 768

### Pooling Strategies

Pooling aggregates frame-level features into syllable-level embeddings:

#### 1. Mean Pooling

**Most common**: Average features across syllable duration

```python
embeddings, meta = embed_audio('audio.wav', pooling='mean')
```

**Output dimension**: Same as feature dimension (e.g., 768 for HuBERT)

#### 2. ONC (Onset-Nucleus-Coda)

**Best for**: Preserving temporal structure

```python
embeddings, meta = embed_audio('audio.wav', pooling='onc')
```

**How it works**:
- Onset: Frame at 30% from start to peak
- Nucleus: Frame at peak
- Coda: Frame at 70% from peak to end
- Concatenates all three

**Output dimension**: 3× feature dimension (e.g., 2304 for HuBERT)

#### 3. Max Pooling

**Best for**: Emphasizing peak activation

```python
embeddings, meta = embed_audio('audio.wav', pooling='max')
```

**Output dimension**: Same as feature dimension

#### 4. Median Pooling

**Best for**: Robustness to outliers

```python
embeddings, meta = embed_audio('audio.wav', pooling='median')
```

**Output dimension**: Same as feature dimension

### Corpus Processing

Process multiple files in batch with parallel execution:

```python
from findsylls.embedding import embed_corpus

results = embed_corpus(
    audio_files=['file1.wav', 'file2.wav', 'file3.wav'],
    embedder='mfcc',
    pooling='mean',
    n_jobs=4,              # Parallel workers
    verbose=True,          # Show progress
    fail_on_error=False    # Skip failed files
)

# results: List[dict] with embeddings and metadata per file
for result in results:
    if result['success']:
        print(f"{result['audio_path']}: {result['embeddings'].shape}")
```

**Performance tips**:
- CPU features (MFCC, Mel): Set `n_jobs=-1` (use all cores)
- GPU models (Sylber, HuBERT): Set `n_jobs=1` (sequential to avoid GPU memory issues)

### Storage

Save and load embeddings efficiently:

```python
from findsylls.embedding.storage import save_embeddings, load_embeddings

# Save (format auto-detected from extension)
save_embeddings(results, 'corpus_embeddings.npz')  # NumPy compressed
save_embeddings(results, 'corpus_embeddings.h5')   # HDF5 (for large corpora)

# Load
results = load_embeddings('corpus_embeddings.npz')
```

**Format recommendations**:
- **NPZ**: Simple, portable, good for < 1000 files
- **HDF5**: Hierarchical, supports partial loading, good for > 1000 files

---

## Evaluation Framework

### Multi-Granular Evaluation

findsylls evaluates at three granularities:

1. **Nuclei**: Syllable peak detection (one point per syllable)
2. **Boundaries**: Syllable onset/offset detection (boundary matching)
3. **Spans**: Full syllable interval matching (requires both boundaries correct)

### Basic Evaluation

```python
from findsylls.evaluation.evaluator import evaluate_segmentation

results = evaluate_segmentation(
    peaks=[0.5, 1.0, 1.5],           # Predicted syllable nuclei
    spans=[(0.3, 0.7), (0.8, 1.2), (1.3, 1.7)],  # Predicted syllable spans
    textgrid_path='annotation.TextGrid',
    tiers={'phone': 2, 'syllable': 1, 'word': 0},
    tolerance=0.05                    # 50ms boundary tolerance
)

# Results structure:
{
    'nuclei': {
        'TP': 45, 'Ins': 3, 'Del': 2, 'Sub': 0,
        'Precision': 0.938, 'Recall': 0.957, 'F1': 0.947
    },
    'syllable_boundaries': {
        'TP': 88, 'Ins': 5, 'Del': 4, 'Sub': 0,
        'Precision': 0.946, 'Recall': 0.957, 'F1': 0.951
    },
    'syllable_spans': {
        'TP': 40, 'Ins': 8, 'Del': 7, 'Sub': 3,
        'Precision': 0.833, 'Recall': 0.800, 'F1': 0.816
    },
    'word_boundaries': {...},
    'word_spans': {...}
}
```

### Tier Specification

TextGrid files typically have multiple tiers (phone, syllable, word). Specify which to use:

```python
# Using tiers dict (recommended - flexible tier names)
results = evaluate_segmentation(
    peaks=peaks,
    spans=spans,
    textgrid_path='file.TextGrid',
    tiers={
        'phone': 2,       # Phone tier at index 2
        'syllable': 1,    # Syllable tier at index 1
        'word': 0,        # Word tier at index 0
        'phrase': 3       # Custom tier at index 3
    }
)

# Legacy parameters (still supported)
results = evaluate_segmentation(
    peaks=peaks,
    spans=spans,
    textgrid_path='file.TextGrid',
    phone_tier=2,
    syllable_tier=1,
    word_tier=0
)
```

### Batch Evaluation

Evaluate multiple files:

```python
from findsylls.pipeline.pipeline import run_evaluation
from findsylls.pipeline.results import flatten_results, aggregate_results

# Run evaluation on all matched files
results = []
for wav_path, tg_path in zip(wav_files, tg_files):
    # Segment
    syllables, _, _ = segment_audio(wav_path, envelope_fn='sbs')
    peaks = [p for (_, p, _) in syllables]
    spans = [(s, e) for (s, _, e) in syllables]
    
    # Evaluate
    result = evaluate_segmentation(
        peaks=peaks,
        spans=spans,
        textgrid_path=tg_path,
        tiers={'phone': 2, 'syllable': 1, 'word': 0}
    )
    result['audio_file'] = wav_path
    result['tg_file'] = tg_path
    results.append(result)

# Flatten to DataFrame
df = flatten_results(results)

# Aggregate statistics
summary = aggregate_results(df, dataset_name='TIMIT')
print(summary)
```

### Metrics

For each granularity (nuclei, boundaries, spans):

- **TP (True Positives)**: Correct predictions
- **Ins (Insertions)**: False positives (predicted but not in reference)
- **Del (Deletions)**: False negatives (in reference but not predicted)
- **Sub (Substitutions)**: Only for spans (wrong category assignment)
- **Precision**: TP / (TP + Ins)
- **Recall**: TP / (TP + Del)
- **F1**: 2 × (Precision × Recall) / (Precision + Recall)

**Tolerance**: Boundaries within `tolerance` seconds are considered matches (default: 0.05s)

---

## Complete Examples

### Example 1: Compare Classical Methods on Single File

```python
from findsylls.pipeline.pipeline import segment_audio
from findsylls.evaluation.evaluator import evaluate_segmentation
import pandas as pd

# Methods to compare
methods = {
    'SBS': {'envelope_fn': 'sbs'},
    'Theta': {'envelope_fn': 'theta'},
    'Hilbert': {'envelope_fn': 'hilbert'}
}

results = []
for name, kwargs in methods.items():
    # Segment
    syllables, _, _ = segment_audio('audio.wav', **kwargs)
    peaks = [p for (_, p, _) in syllables]
    spans = [(s, e) for (s, _, e) in syllables]
    
    # Evaluate
    metrics = evaluate_segmentation(
        peaks, spans,
        textgrid_path='audio.TextGrid',
        tiers={'phone': 2, 'syllable': 1, 'word': 0}
    )
    
    # Store results
    results.append({
        'method': name,
        'nuclei_f1': metrics['nuclei']['F1'],
        'boundary_f1': metrics['syllable_boundaries']['F1'],
        'span_f1': metrics['syllable_spans']['F1'],
        'num_syllables': len(syllables)
    })

# Display comparison
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### Example 2: Extract Embeddings for Multiple Speakers

```python
from findsylls.embedding import embed_corpus
from findsylls.embedding.storage import save_embeddings
import glob

# Get all audio files
audio_files = glob.glob('corpus/**/*.wav', recursive=True)

# Extract embeddings with Sylber
results = embed_corpus(
    audio_files,
    segmentation='sylber',
    embedder='sylber',
    pooling='onc',          # Preserve temporal structure
    n_jobs=1,               # Sequential for GPU
    verbose=True
)

# Save for downstream tasks
save_embeddings(results, 'sylber_onc_embeddings.h5')

# Print statistics
successful = [r for r in results if r['success']]
print(f"Processed {len(successful)}/{len(results)} files")
print(f"Total syllables: {sum(r['embeddings'].shape[0] for r in successful)}")
```

### Example 3: Benchmark All Methods on Dataset

```python
from findsylls.audio.utils import match_wavs_to_textgrids
from findsylls.pipeline.pipeline import segment_audio
from findsylls.evaluation.evaluator import evaluate_segmentation
from findsylls.pipeline.results import flatten_results, aggregate_results
import time

# Match files
tg_files, wav_files = match_wavs_to_textgrids(
    'data/TIMIT/**/*.wav',
    'data/TIMIT/**/*_syllabified.TextGrid'
)

# Methods to benchmark
methods = {
    'SBS': {'envelope_fn': 'sbs', 'type': 'classical'},
    'Theta': {'envelope_fn': 'theta', 'type': 'classical'},
    'Sylber': {'envelope_fn': None, 'type': 'neural'},  # End-to-end
}

all_results = []
for method_name, method_config in methods.items():
    print(f"\n{'='*60}")
    print(f"Benchmarking: {method_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    total_duration = 0
    
    for tg_path, wav_path in zip(tg_files[:100], wav_files[:100]):  # First 100 files
        try:
            # Load audio to measure duration
            from findsylls.audio.utils import load_audio
            audio, sr = load_audio(wav_path, samplerate=16000)
            total_duration += len(audio) / sr
            
            # Segment
            if method_config['envelope_fn']:
                syllables, _, _ = segment_audio(
                    wav_path,
                    envelope_fn=method_config['envelope_fn']
                )
            else:
                # Use neural method
                from findsylls.segmentation import get_segmenter
                segmenter = get_segmenter('sylber')
                syllables = segmenter.segment(audio, sr)
            
            # Evaluate
            peaks = [p for (_, p, _) in syllables]
            spans = [(s, e) for (s, _, e) in syllables]
            
            result = evaluate_segmentation(
                peaks, spans,
                textgrid_path=tg_path,
                tiers={'phone': 2, 'syllable': 1, 'word': 0}
            )
            result['envelope'] = method_name
            result['method_type'] = method_config['type']
            result['audio_file'] = wav_path
            result['tg_file'] = tg_path
            all_results.append(result)
            
        except Exception as e:
            print(f"  Error on {wav_path}: {e}")
    
    # Compute speed
    elapsed = time.time() - start_time
    rtf = total_duration / elapsed  # Real-Time Factor
    print(f"  Processed {len(wav_files[:100])} files in {elapsed:.1f}s")
    print(f"  RTF: {rtf:.2f}× (higher = faster)")

# Aggregate results
df = flatten_results(all_results)
for method_name in methods.keys():
    method_df = df[df['envelope'] == method_name]
    summary = aggregate_results(method_df, dataset_name=method_name)
    print(f"\n{method_name} Summary:")
    print(summary.to_string(index=False))

# Save results
df.to_csv('benchmark_results.csv', index=False)
```

### Example 4: Phase 5 Mix-and-Match (Advanced)

```python
from findsylls.segmentation import MinCutSegmenter, GreedyCosineSegmenter
from findsylls.features import MFCCExtractor, HuBERTExtractor

# Apply MinCut algorithm to MFCC features (classical + neural algorithm)
mfcc_extractor = MFCCExtractor(n_mfcc=13, include_deltas=True)
mfcc_mincut = MinCutSegmenter(
    mfcc_extractor,
    sec_per_syllable=0.22,
    use_optimized=True
)
syllables_mfcc = mfcc_mincut.segment(audio, sr=16000)

# Apply Greedy Cosine to HuBERT features (different neural combo)
hubert_extractor = HuBERTExtractor(layer=9)
hubert_greedy = GreedyCosineSegmenter(
    hubert_extractor,
    norm_threshold=0.3,
    merge_threshold=0.85
)
syllables_hubert = hubert_greedy.segment(audio, sr=16000)

# Compare ablations
print(f"MFCC + MinCut: {len(syllables_mfcc)} syllables")
print(f"HuBERT + GreedyCosine: {len(syllables_hubert)} syllables")
```

---

## API Reference

### High-Level APIs

#### `segment_audio()`

```python
from findsylls.pipeline.pipeline import segment_audio

syllables, envelope, times = segment_audio(
    audio_path: str,
    envelope_fn: str = 'sbs',              # Envelope method
    segment_fn: str = 'peakdetect', # Segmentation algorithm
    envelope_kwargs: dict = None,          # Envelope parameters
    segmentation_kwargs: dict = None,      # Segmentation parameters
    samplerate: int = 16000                # Target sample rate
) -> Tuple[List[Tuple[float, float, float]], np.ndarray, np.ndarray]
```

#### `embed_audio()`

```python
from findsylls.embedding import embed_audio

embeddings, metadata = embed_audio(
    audio_path: str,
    segmentation: str = 'sbs',          # Segmentation method
    embedder: str = 'mfcc',             # Feature extractor
    pooling: str = 'mean',              # Pooling strategy
    segmentation_kwargs: dict = None,   # Segmentation parameters
    embedder_kwargs: dict = None,       # Embedder parameters
    samplerate: int = 16000
) -> Tuple[np.ndarray, dict]
```

#### `embed_corpus()`

```python
from findsylls.embedding import embed_corpus

results = embed_corpus(
    audio_files: List[str],
    segmentation: str = 'sbs',
    embedder: str = 'mfcc',
    pooling: str = 'mean',
    n_jobs: int = 1,                    # Number of parallel workers
    verbose: bool = True,               # Show progress bar
    fail_on_error: bool = False,        # Skip or fail on errors
    **kwargs
) -> List[dict]
```

#### `evaluate_segmentation()`

```python
from findsylls.evaluation.evaluator import evaluate_segmentation

results = evaluate_segmentation(
    peaks: List[float],                 # Syllable nuclei times
    spans: List[Tuple[float, float]],   # Syllable (start, end) times
    textgrid_path: str,                 # Path to TextGrid file
    tiers: dict = None,                 # Tier name -> index mapping
    tolerance: float = 0.05,            # Boundary tolerance (seconds)
    phone_tier: int = None,             # Legacy: phone tier index
    syllable_tier: int = None,          # Legacy: syllable tier index
    word_tier: int = None               # Legacy: word tier index
) -> dict
```

### Segmenter Classes

#### `get_segmenter()`

Factory function for creating segmenters:

```python
from findsylls.segmentation import get_segmenter

# Classical envelope-based
segmenter = get_segmenter(
    'peakdetect',
    delta=0.01,
    min_syllable_dur=0.05
)

# Neural end-to-end
segmenter = get_segmenter('sylber')
segmenter = get_segmenter('vg_hubert', sec_per_syllable=0.22)
segmenter = get_segmenter('vg_hubert_cls', attn_threshold=0.1)
segmenter = get_segmenter('syllablelm', use_optimized=True)

# All segmenters have .segment() method
syllables = segmenter.segment(audio, sr=16000)
```

#### Phase 5 Wrappers

Mix any feature extractor with any algorithm:

```python
from findsylls.segmentation import MinCutSegmenter, GreedyCosineSegmenter
from findsylls.features import HuBERTExtractor, MFCCExtractor

# Create feature extractor
extractor = HuBERTExtractor(layer=9)

# Wrap with algorithm
segmenter = MinCutSegmenter(
    extractor,
    sec_per_syllable=0.22,
    use_optimized=True
)

# Get embeddings directly
segments, embeddings = segmenter.get_embeddings(audio, sr=16000)
# segments: List[(start, nucleus, end)]
# embeddings: np.ndarray[N, D] - mean features per segment
```

### Utility Functions

#### Audio Loading

```python
from findsylls.audio.utils import load_audio

audio, sr = load_audio(
    path: str,
    samplerate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]
```

#### File Matching

```python
from findsylls.audio.utils import match_wavs_to_textgrids

tg_files, wav_files = match_wavs_to_textgrids(
    wav_pattern: str,                    # Glob pattern for WAV files
    textgrid_pattern: str,               # Glob pattern for TextGrid files
    tg_suffix_to_strip: str = None       # Suffix to remove for matching
) -> Tuple[List[str], List[str]]
```

#### Results Processing

```python
from findsylls.pipeline.results import flatten_results, aggregate_results

# Flatten nested results to DataFrame
df = flatten_results(results: List[dict]) -> pd.DataFrame

# Aggregate statistics
summary = aggregate_results(
    df: pd.DataFrame,
    dataset_name: str = 'corpus'
) -> pd.DataFrame
```

---

## Performance & Benchmarking

### Speed Comparison

| Method | Processing Speed | Real-Time Factor | GPU Required |
|--------|-----------------|------------------|--------------|
| SBS | ~100 fps | 6-10× | No |
| Theta | ~50 fps | 3-5× | No |
| Hilbert | ~200 fps | 12-15× | No |
| Sylber | ~50 fps | 3-5× | Optional |
| VG-HuBERT | ~50 fps | 3-5× | Optional |
| SyllableLM | ~50 fps | 3-5× | Optional |

**Real-Time Factor (RTF)**: Higher is better. RTF=10× means processing 10 seconds of audio in 1 second.

### Accuracy Comparison

Results on TIMIT corpus (syllable boundary F1):

| Method | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| SBS | 0.50-0.60 | 0.55-0.65 | 0.50-0.60 |
| Theta | 0.55-0.65 | 0.60-0.70 | 0.55-0.65 |
| Sylber | 0.75-0.85 | 0.80-0.85 | 0.75-0.85 |
| VG-HuBERT | 0.70-0.80 | 0.75-0.85 | 0.70-0.80 |
| SyllableLM | 0.75-0.85 | 0.80-0.85 | 0.75-0.85 |

**Note**: Scores vary by dataset, speaking style, and language.

### Memory Usage

| Method | RAM (typical) | GPU Memory | Notes |
|--------|--------------|------------|-------|
| Classical | < 100 MB | N/A | CPU-only |
| Sylber | ~2 GB | ~2 GB (optional) | Model loaded once |
| VG-HuBERT | ~2 GB | ~2 GB (optional) | Model loaded once |
| SyllableLM | ~1 GB | ~1 GB (optional) | HuBERT base |

### Optimization Tips

1. **Use GPU for neural methods**: 3-5× speedup on supported hardware
   ```python
   segmenter = get_segmenter('sylber', device='cuda')
   ```

2. **Parallel processing for CPU features**:
   ```python
   results = embed_corpus(files, embedder='mfcc', n_jobs=-1)  # Use all cores
   ```

3. **Sequential for GPU models**:
   ```python
   results = embed_corpus(files, embedder='sylber', n_jobs=1)  # Avoid GPU conflicts
   ```

4. **Use optimized algorithms**:
   ```python
   segmenter = get_segmenter('vg_hubert', use_optimized=True)  # 20-50× speedup
   ```

5. **Cache model loading**: Models are loaded once and reused across files automatically

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'sylber'"

**Problem**: Sylber package not installed

**Solution**:
```bash
pip install 'findsylls[embedding]'
```

#### 2. "VG-HuBERT model not found"

**Problem**: VG-HuBERT model not downloaded

**Solution**:
```bash
pip install vg-hubert  # Auto-downloads model from HuggingFace
```

Or manually specify local model path:
```python
segmenter = get_segmenter('vg_hubert', model_path='/path/to/vg-hubert_3')
```

#### 3. Low F1 Scores on Custom Data

**Problem**: Default parameters not tuned for your data

**Solution**: Adjust segmentation parameters:

```python
# For shorter syllables (e.g., fast speech)
segment_audio('audio.wav', segmentation_kwargs={
    'min_syllable_dur': 0.03,  # Reduce from 0.05
    'delta': 0.005              # More sensitive
})

# For longer syllables (e.g., child-directed speech)
segment_audio('audio.wav', segmentation_kwargs={
    'min_syllable_dur': 0.08,  # Increase from 0.05
    'delta': 0.02               # Less sensitive
})
```

#### 4. CUDA Out of Memory

**Problem**: GPU memory exhausted with neural methods

**Solution**:
1. Process files sequentially: `embed_corpus(..., n_jobs=1)`
2. Use CPU: `get_segmenter('sylber', device='cpu')`
3. Reduce batch size (if using custom batch processing)

#### 5. TextGrid Parsing Errors

**Problem**: TextGrid format not recognized

**Solution**:
1. Verify file format (Praat long or short format)
2. Check tier indices:
   ```python
   from findsylls.parsing.textgrid import parse_textgrid
   tg = parse_textgrid('file.TextGrid')
   print(f"Number of tiers: {len(tg)}")
   for i, tier in enumerate(tg):
       print(f"Tier {i}: {tier.name} ({len(tier.intervals)} intervals)")
   ```
3. Use correct tier specification in evaluation

#### 6. File Matching Issues

**Problem**: WAV and TextGrid files not being matched

**Solution**:
```python
from findsylls.audio.utils import match_wavs_to_textgrids

# Check matching results
tg_files, wav_files = match_wavs_to_textgrids(
    'data/**/*.wav',
    'data/**/*.TextGrid',
    tg_suffix_to_strip='_syllabified'  # Remove suffix for matching
)

print(f"Matched {len(wav_files)} pairs")
if len(wav_files) == 0:
    print("No matches found - check patterns and suffixes")
```

#### 7. lookahead Parameter Issues

**Problem**: Over-segmentation or under-segmentation

**Solution**: Remove explicit `lookahead` parameter to use auto-calculation:

```python
# DON'T DO THIS (lookahead=1 is too sensitive):
segmentation_kwargs = {
    'lookahead': 1,  # BAD: causes over-segmentation
    'delta': 0.01
}

# DO THIS (auto-calculate based on min_syllable_dur):
segmentation_kwargs = {
    # lookahead auto-calculated as min_syllable_dur / 2.0
    'min_syllable_dur': 0.05,
    'delta': 0.01
}
```

### Getting Help

1. **Check documentation**: This guide covers most use cases
2. **GitHub Issues**: https://github.com/hjvm/findsylls/issues
3. **Code examples**: See `examples/` directory in repository
4. **Copilot instructions**: `.github/copilot-instructions.md` has internal guidance

### Debugging Tips

1. **Enable verbose output**:
   ```python
   embed_corpus(..., verbose=True)  # Shows progress bar
   ```

2. **Check intermediate outputs**:
   ```python
   syllables, envelope, times = segment_audio('audio.wav')
   print(f"Envelope shape: {envelope.shape}")
   print(f"Times shape: {times.shape}")
   print(f"Syllables: {len(syllables)}")
   ```

3. **Visualize segmentation** (requires matplotlib):
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(12, 4))
   plt.plot(times, envelope)
   for start, peak, end in syllables:
       plt.axvline(start, color='g', alpha=0.5)
       plt.axvline(peak, color='r', alpha=0.5)
       plt.axvline(end, color='b', alpha=0.5)
   plt.show()
   ```

4. **Validate TextGrid**:
   ```python
   from findsylls.evaluation.evaluator import evaluate_segmentation
   
   # Run evaluation and check counts
   results = evaluate_segmentation(peaks, spans, tg_path, ...)
   print(f"Predicted syllables: {len(peaks)}")
   print(f"Reference syllables: {results['syllable_boundaries']['Total']}")
   ```

---

## Validation & Testing

### Legacy Consistency

findsylls has been validated against the legacy `spot_the_word` implementation:

- **Correlation**: r=0.9990 (near-perfect match)
- **Segmentation**: 100% syllable count agreement
- **Features**: MFCC extraction consistent across implementations

See `docs/VALIDATION_RESULTS.md` for detailed validation report.

### Test Coverage

All modules have comprehensive test coverage:

```bash
# Run tests
pytest tests/

# Run specific test suite
pytest tests/test_embedding_pipeline.py
pytest tests/test_segmentation.py
pytest tests/test_evaluation.py
```

---

## Citation

If you use findsylls in your research, please cite:

```bibtex
@software{findsylls2026,
  title = {findsylls: A Comprehensive Toolkit for Syllable Segmentation and Embedding},
  author = {Villalobos, Hector and contributors},
  year = {2026},
  url = {https://github.com/hjvm/findsylls}
}
```

And cite the methods you use:

**Sylber**:
```bibtex
@inproceedings{park2024sylber,
  title={Sylber: Syllabic Embedding Representation of Speech from Raw Audio},
  author={Park, Cheol Jun and others},
  booktitle={ICLR},
  year={2024}
}
```

**VG-HuBERT**:
```bibtex
@inproceedings{peng2023syllable,
  title={Syllable Discovery and Cross-Lingual Generalization in a Visually Grounded, Self-Supervised Speech Model},
  author={Peng, Puyuan and Harwath, David},
  booktitle={Interspeech},
  year={2023}
}
```

**SyllableLM**:
```bibtex
@article{baade2024syllablelm,
  title={SyllableLM: Learning Coarse Semantic Units for Speech Language Models},
  author={Baade, Alan and others},
  journal={arXiv preprint arXiv:2410.04029},
  year={2024}
}
```

---

## License

findsylls is released under the MIT License. See LICENSE file for details.

Individual methods (Sylber, VG-HuBERT, etc.) may have their own licenses. Please check the respective model repositories.

---

**End of User Guide**

For development documentation, see `docs/dev/DEVELOPMENT_GUIDE.md`.
