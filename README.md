# findsylls

Unsupervised syllable(-like) segmentation & evaluation toolkit for speech audio. Extract amplitude / modulation envelopes, segment into candidate syllables, and (optionally) evaluate versus Praat TextGrid annotations at nuclei, syllable boundary/span, and word boundary/span levels.

## Features
- Pluggable amplitude envelope front-ends: RMS, Hilbert, low-pass, spectral band subtraction (SBS), gammatone, theta oscillator.
- Peak & valley segmentation (extensible: hooks for future algorithms like Mermelstein, oscillator-based).
- Robust TextGrid parsing for phones, syllables, words with vowel / syllabic consonant filtering.
- Multi-level evaluation metrics (TP / Ins / Del / Sub; precision/recall/F1/TER aggregation helpers).
- Batch pipeline utilities + fuzzy filename matching (`.wav` ↔ `.TextGrid`).
- Optional plotting layer for qualitative inspection.

## Install (Local)
```bash
# Core install
pip install findsylls

# Or from a local clone (editable for development)
pip install -e .[dev]

# With plotting extras
pip install 'findsylls[viz]'
```

## Quick Start
```python
from findsylls import segment_audio
sylls, env, t = segment_audio("example.wav", envelope_fn="sbs", segment_fn="peaks_and_valleys")
print(sylls[:5])
```
Batch evaluation:
```python
from findsylls import run_evaluation
results = run_evaluation(
    textgrid_paths="data/**/*.TextGrid",
    wav_paths="data/**/*.wav",
    phone_tier=1,
    syllable_tier=2,
    word_tier=3,
    envelope_fn="hilbert",
)
print(results.head())
```
Aggregate:
```python
from findsylls import aggregate_results
summary = aggregate_results(results, dataset_name="MyCorpus")
print(summary)
```

## CLI
After install:
```bash
findsylls segment input.wav --envelope sbs --method peaks_and_valleys --out sylls.json
findsylls evaluate "data/**/*.wav" "data/**/*.TextGrid" --phone-tier 1 --syllable-tier 2 --word-tier 3 --envelope hilbert --out results.csv
```
Show help:
```bash
findsylls --help
findsylls segment --help
findsylls evaluate --help
```

## API Surface
| Function | Purpose |
|----------|---------|
| `segment_audio` | One-file end‑to‑end (load → envelope → segment). |
| `run_evaluation` | Batch match WAV/TextGrid and compute metrics. |
| `get_amplitude_envelope` | Compute envelope via a registered method. |
| `segment_envelope` | Dispatch segmentation algorithm. |
| `flatten_results` / `aggregate_results` | Reshape & aggregate evaluation outputs. |
| `plot_segmentation_result` | Multi-panel qualitative plot (optional). |

## Adding Methods
1. Envelope: implement `compute_*` returning `(env, times)` in `envelope/` and register in `envelope/dispatch.py`.
2. Segmentation: implement `segment_<name>(envelope, times, **kwargs)` in `segmentation/` and add branch in `segmentation/dispatch.py`.

## TextGrid Tier Indexing
Indices are 0-based (as provided by the `textgrid` library). Pass `None` to skip a tier or `-1` for placeholder syllable generation (currently returns empty list).

## Evaluation Conventions
- Default tolerance = 0.05s.
- `EVAL_METHODS` ordering drives flatten/aggregate loops; include new metric keys there if extending.
- Substitutions matter for span metrics; remain zero for nuclei/boundary F1 semantics.

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
(Provide a citation once there is a paper / preprint.)

---
For development guidelines see `.github/copilot-instructions.md`.
