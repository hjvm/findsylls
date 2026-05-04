# findsylls

[![PyPI version](https://img.shields.io/pypi/v/findsylls.svg)](https://pypi.org/project/findsylls/)
[![Python versions](https://img.shields.io/pypi/pyversions/findsylls.svg)](https://pypi.org/project/findsylls/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Language-agnostic toolkit for syllable-level speech tokenization and embedding extraction.

findsylls provides:
- Envelope computation from waveform (RMS, Hilbert, low-pass, SBS, gammatone, theta)
- Syllable segmentation (peak/valley and neural options)
- Evaluation against TextGrid annotations (nuclei, boundaries, spans)
- Per-syllable embedding extraction for downstream tasks

## Install

```bash
# Core package
pip install findsylls

# Optional extras
pip install 'findsylls[viz]'       # plotting helpers
pip install 'findsylls[embedding]' # neural feature extraction
pip install 'findsylls[end2end]'   # neural segmentation methods
pip install 'findsylls[storage]'   # HDF5 storage support
pip install 'findsylls[all]'       # all extras
```

## Quick Start

### 1) Segment a file into syllables

```python
from findsylls import segment_audio

sylls, envelope, times = segment_audio(
    "example.wav",
    envelope_fn="sbs",
  segment_fn="peakdetect",
)

print(f"Found {len(sylls)} syllables")
# sylls: [(start, peak, end), ...]
```

### 2) Evaluate against TextGrid annotations

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

summary = aggregate_results(results, dataset_name="MyCorpus")
print(summary)
```

### 3) Extract syllable embeddings

```python
from findsylls import embed_audio

embeddings, metadata = embed_audio(
    "example.wav",
  segmentation="peakdetect",
    features="mfcc",      # mfcc | melspec | sylber | vg_hubert
    pooling="mean",       # mean | onc | max | median
)

print(embeddings.shape)
print(metadata["num_syllables"])
```

### 4) Batch embedding extraction

```python
from findsylls import embed_corpus, save_embeddings

results = embed_corpus(
    audio_paths=["a.wav", "b.wav", "c.wav"],
  segmentation="peakdetect",
    features="mfcc",
    pooling="mean",
    n_jobs=4,
)

save_embeddings(results, "embeddings.npz")
```

## CLI

```bash
# Segment audio
findsylls segment input.wav --envelope sbs --method peakdetect --out sylls.json

# Extract embeddings
findsylls embed input.wav --features mfcc --pooling mean --out embeddings.npz

# Evaluate against TextGrid annotations
findsylls evaluate "data/**/*.wav" "data/**/*.TextGrid" \
  --phone-tier 1 --syllable-tier 2 --word-tier 3 \
  --envelope hilbert --out results.csv
```

## Methods Overview

### Envelope Methods
- `rms`
- `hilbert`
- `lowpass`
- `sbs`
- `gammatone`
- `theta`
- Feature-based envelopes (e.g., SSM / GreedyCosine / CLS-attention where available)

### Segmentation Methods
- `peakdetect`
- `cls_attention`
- `sylber`
- `greedy_cosine`
- `vg_hubert_mincut` (aliases: `vg_hubert`, `vg_hubert_ssm`, `featssm`)
- `vg_hubert_cls`
- `syllablelm`

Backward-compatibility aliases are still accepted by the dispatcher, but the canonical names above are what the notebook and API docs should use.

### Embedding Features
- `mfcc` (13/26/39 dims with deltas)
- `melspec` (mel-filterbank)
- `sylber`
- `vghubert` (also accepted as `vg-hubert` or `vg_hubert`)

## Examples and Notebook

- Interactive demo notebook: [findsylls_demo.ipynb](findsylls_demo.ipynb)
- Example scripts: [examples/](examples/)
- Streaming workflow tutorial: [notebooks/streaming_workflows.ipynb](notebooks/streaming_workflows.ipynb) *(coming soon)*

## Corpus-Scale Workflows

For large corpora, findsylls supports **storage-backed embedding extraction** and **streaming clustering** to avoid loading all embeddings into memory.

### Storage-First Embedding Extraction

Use `embed_corpus_to_storage()` to write embeddings directly to disk per-file, with a manifest CSV for indexing:

```python
from findsylls import embed_corpus_to_storage

info = embed_corpus_to_storage(
    audio_files=['a.wav', 'b.wav', 'c.wav', ...],
    output_dir='./embeddings',
    segmentation='peakdetect',
    features='mfcc',
    pooling='mean',
)

print(f"Embedded {info['num_success']}/{info['num_files']} files")
# Output: ./embeddings/embedding_manifest.csv + ./embeddings/000000*.npz
```

The manifest CSV contains:
- `file_id`: File index
- `audio_path`: Original audio file path
- `embedding_path`: Path to `.npz` file with embeddings
- `num_rows`: Number of syllables
- `embedding_dim`: Embedding dimensionality
- `success`: Whether embedding succeeded
- `error`: Error message if failed

### Streaming Clustering Discovery

Cluster large embeddings without loading all into memory using `MiniBatchKMeans`:

```python
from findsylls import DiscoveryPipeline
from findsylls.embedding.storage import load_embedding_manifest

# Load manifest from storage-backed embedding
manifest_path = './embeddings/embedding_manifest.csv'

pipeline = DiscoveryPipeline(
    discovery_method='minibatch_kmeans',
    n_clusters=50,
)

# Fit and predict in chunks (default: 10K embeddings per chunk)
labels_by_file = pipeline.discover_from_storage(
    manifest_path=manifest_path,
    chunk_size=10000,
)

print(f"Discovered clusters across all files")
for file_id, labels in labels_by_file.items():
    print(f"  File {file_id}: {len(labels)} syllables")
```

**Memory Profile:**
- **In-Memory Clustering** (`embed_corpus()` + `vstack()` + `KMeans`): ~10 GB for 500K syllables × 768D embeddings
- **Streaming Clustering** (`embed_corpus_to_storage()` + `discover_from_storage()` with MiniBatchKMeans): ~500 MB

This makes corpus-scale analysis practical on commodity hardware.

## Evaluation & Metrics

### Intrinsic Clustering Metrics (No Ground Truth Required)

When you run discovery, findsylls automatically computes:

- **Silhouette Score** (-1 to +1, higher is better): Measures how close samples are to their cluster vs other clusters
- **Davies-Bouldin Index** (lower is better): Ratio of within-cluster to between-cluster distances
- **Calinski-Harabasz Index** (higher is better): Ratio of between-cluster to within-cluster dispersion

Example:
```python
from findsylls import DiscoveryPipeline

pipeline = DiscoveryPipeline(method='kmeans', n_clusters=50)
result = pipeline.discover(embeddings)

print(f"Silhouette: {result.fit_metrics['silhouette']:.3f}")
print(f"Davies-Bouldin: {result.fit_metrics['davies_bouldin']:.3f}")
print(f"Calinski-Harabasz: {result.fit_metrics['calinski_harabasz']:.1f}")
```

### Evaluating Against TextGrid Annotations

Compare segmentation and discovered clusters against manual annotations:

```python
from findsylls import evaluate_segmentation, compute_discovery_label_metrics

# Evaluate segmentation at multiple granularities
eval_result = evaluate_segmentation(
    peaks=[0.15, 0.35, ...],      # syllable nuclei in seconds
    spans=[(0.1, 0.2), (0.3, 0.4), ...],  # syllable boundaries
    textgrid_path="annotations.TextGrid",
    tiers={'phone': 2, 'syllable': 1, 'word': 0}  # TextGrid tier indices
)

# Displays metrics like:
# - nuclei_f1: align detected nuclei with vowel intervals
# - syllable_boundaries_f1: align boundaries with syllable tiers
# - word_spans_f1: align with word-level boundaries
```

### Label-Aware Discovery Metrics

Connect discovered clusters to ground-truth labels:

```python
from findsylls.evaluation import attach_textgrid_labels_to_manifest, compute_discovery_label_metrics

# Add TextGrid labels to discovery results
labeled_manifest = attach_textgrid_labels_to_manifest(
    manifest=discovery_result_manifest,
    wav_paths=['audio.wav', ...],
    textgrid_paths=['annotations.TextGrid', ...],
    textgrid_tier_index=2,  # phone tier
)

# Compute metrics
metrics = compute_discovery_label_metrics(labeled_manifest)

print(f"Cluster Purity: {metrics['cluster_purity']:.3f}")
print(f"Label Purity: {metrics['label_purity']:.3f}")
print(f"Normalized MI: {metrics['label_norm_mutual_info']:.3f}")
print(f"Macro F1: {metrics['macro_f1']:.3f}")
```

**Metrics Glossary:**
- **Cluster Purity**: What fraction of each cluster's members share the most common label (0-1)
- **Label Purity**: What fraction of each label's instances fall in the most common cluster (0-1)
- **Normalized MI**: Mutual information between cluster and label assignments, normalized by entropy (0-1)
- **Macro F1**: Unweighted average F1 across clusters (treating each cluster's dominant label as class)

### Common Workflows

1. **Corpus discovery with evaluation:**
   - Run `embed_corpus_to_storage()` to extract syllable embeddings
   - Run `discover_from_storage()` to cluster them
   - Attach ground truth via `attach_textgrid_labels_to_manifest()`
   - Compute metrics with `compute_discovery_label_metrics()`

2. **Comparing segmentation methods:**
   - Run `evaluate_segmentation()` on each segmentation method
   - Compare F1 scores across methods

3. **Hyperparameter tuning:**
   - Extract embeddings with different pooling methods
   - Cluster with varying `n_clusters`
   - Compare intrinsic metrics (Silhouette, Davies-Bouldin) to choose best hyperparameters

## Citation

If you use findsylls in academic work, please cite:

- https://arxiv.org/abs/2603.26292

Plain text:

```
Vázquez Martínez, Héctor Javier. (2026). findsylls: A Language-Agnostic Toolkit for Syllable-Level Speech Tokenization and Embedding. arXiv:2603.26292. https://arxiv.org/abs/2603.26292
```

BibTeX:

```bibtex
@misc{martinez2026findsyllslanguageagnostictoolkitsyllablelevel,
  title={findsylls: A Language-Agnostic Toolkit for Syllable-Level Speech Tokenization and Embedding},
  author={Héctor Javier Vázquez Martínez},
  year={2026},
  eprint={2603.26292},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2603.26292},
}
```

## License

MIT. See [LICENSE](LICENSE).
