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
- Neural/custom segmenters exposed through the segmentation module

### Embedding Features
- `mfcc` (13/26/39 dims with deltas)
- `melspec` (mel-filterbank)
- `sylber`
- `vg_hubert`

## Examples and Notebook

- Interactive demo notebook: [findsylls_demo.ipynb](findsylls_demo.ipynb)
- Example scripts: [examples/](examples/)

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
