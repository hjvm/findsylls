# VG-HuBERT Implementation

VG-HuBERT (Visually Grounded HuBERT) syllable segmentation has been successfully integrated into findsylls using the `vg-hubert` PyPI package!

## Quick Start

```python
from findsylls.segmentation.end2end import VGHubertSegmenter
from findsylls.audio.utils import load_audio

# Create segmenter - model downloads automatically from HuggingFace!
segmenter = VGHubertSegmenter(
    model_ckpt='hjvm/VG-HuBERT',  # Auto-downloads (default)
    mode='syllable',              # 'syllable' or 'word'
    layer=8,                      # HuBERT layer (default: 8 for syllables)
    merge_threshold=0.3,          # MinCutMerge post-processing (recommended)
    device='cuda'                 # Use 'cuda', 'cpu', or 'mps'
)

# Segment audio
audio, sr = load_audio('audio.wav', target_sr=16000)
syllables = segmenter.segment(audio, sr=sr)

# Result: list of (start, peak, end) tuples in seconds
for start, peak, end in syllables:
    print(f"Syllable: {start:.3f}s to {end:.3f}s (nucleus at {peak:.3f}s)")
```

## Installation

```bash
# Install findsylls with embedding support
pip install 'findsylls[embedding]'

# Install vg-hubert package
pip install vg-hubert
```

## Model Download

**No manual download required!** The vg-hubert package automatically downloads the model from HuggingFace Hub (hjvm/VG-HuBERT) on first use.

For offline use or local models:
```python
# Use local model directory
segmenter = VGHubertSegmenter(
    model_ckpt='/path/to/local/vg-hubert_3'
)
```

## How It Works

VG-HuBERT uses a two-stage segmentation approach:

### 1. Feature Extraction
- Audio → VG-HuBERT → Layer 8 features (768-dim)
- VG-HuBERT is trained on SpokenCOCO (audio-image pairs)
- Features capture visually-grounded speech representations

### 2. MinCut Segmentation
- Compute self-similarity matrix: `SSM = features @ features.T`
- Estimate number of syllables: `K = ceil(audio_duration / 0.2)`
- Run optimized MinCut algorithm (40x faster than original)
- Optional MinCutMerge post-processing to prevent over-segmentation

### 3. Optional Peak Detection
- Detect syllable nuclei using feature norm peaks
- Enable with `detect_peaks=True`

## Parameters

### Required
- **model_ckpt** (str, default='hjvm/VG-HuBERT'): HuggingFace model or local path
  - Default auto-downloads from HuggingFace Hub
  - Can specify local path for offline use

### Optional
- **mode** (str, default='syllable'): Segmentation mode
  - `'syllable'`: MinCut-based syllable segmentation
  - `'word'`: Attention-based word segmentation
  
- **layer** (int, optional): Which HuBERT layer to use
  - Default: 8 for syllables, 9 for words
  - Range: 0-11
  
- **sec_per_syllable** (float, default=0.2): Target syllable duration
  - 0.15-0.2 works well for English
  - Adjust for other languages
  
- **merge_threshold** (float, default=0.3): Cosine similarity threshold for merging
  - Recommended: 0.3 (matches original paper)
  - `None`: No merging (more segments)
  - `0.3-0.4`: Typical range
  - Higher = more merging = fewer segments
  
- **min_segment_frames** (int, default=2): Filter very short segments
  
- **device** (str, default='cpu'): Compute device
  - `'cuda'`: NVIDIA GPU (fastest)
  - `'mps'`: Apple Silicon GPU
  - `'cpu'`: CPU (slowest)
  
- **detect_peaks** (bool, default=False): Detect nucleus peaks
  - `False`: Use segment midpoint
  - `True`: Detect peak using feature norms

## Performance

From Peng et al. (2023) on SpokenCOCO test set:

| Metric | Value |
|--------|-------|
| Boundary Precision | 0.57 |
| Boundary Recall | 0.64 |
| Boundary F1 | 0.60 |
| Over-segmentation | 0.11 |

**Optimization**: This fork uses an optimized MinCut algorithm (~40x speedup) from [SyllableLM](https://github.com/AlanBaade/SyllableLM).

Works across multiple languages:
- English (SpokenCOCO)
- French (ZeroSpeech 2020)
- Mandarin (ZeroSpeech 2020)
- Estonian Conversational Speech

### Pure Python Implementation
Our implementation is pure Python (no Cython compilation needed):
- Slower than original Cython version (~2-3x)
- But more portable and easier to install
- For production, consider compiling the original Cython version

## Comparison with Other Methods

| Method | Approach | Training | Performance | Speed |
|--------|----------|----------|-------------|-------|
| **VG-HuBERT** | Graph-based | Pre-trained | F1 ~0.60 | Medium |
| **Sylber** | Norm + Merge | Pre-trained | F1 ~0.68 | Fast |
| **SD-HuBERT** | NormCut + MinCut | Pre-trained | F1 ~0.64-0.68 | Medium |

Advantages of VG-HuBERT:
- ✓ No training needed
- ✓ Works across languages
- ✓ Interpretable (graph-based)
- ✓ Adjustable granularity

Disadvantages:
- ✗ Requires model download (~500MB)
- ✗ Medium speed
- ✗ Fixed number of segments

## Example: Full Pipeline

```python
from findsylls.pipeline.pipeline import run_evaluation
from pathlib import Path

# Setup
wav_dir = Path("data/TIMIT/test")
tg_dir = Path("data/TIMIT/test")

# Run evaluation with VG-HuBERT
results = run_evaluation(
    wav_dir=wav_dir,
    tg_dir=tg_dir,
    envelope_fn=None,  # End-to-end, doesn't use envelope
    segmentation_fn='vg_hubert',
    segmentation_kwargs={
        'model_path': '/path/to/vg-hubert_3',
        'layer': 8,
        'sec_per_syllable': 0.2,
        'merge_threshold': 0.3,
        'device': 'cuda'
    },
    tiers={'phone': 2, 'syllable': 1, 'word': 0}
)

# Aggregate results
from findsylls.pipeline.results import aggregate_results
summary = aggregate_results(results, dataset_name='TIMIT')
print(summary)
```

## Troubleshooting

### Model Loading Errors

**Error:** `FileNotFoundError: Model path does not exist`
- Solution: Download VG-HuBERT checkpoint (see Download section)

**Error:** `ModuleNotFoundError: No module named 'models.audio_encoder'`
- Solution: VGHubertSegmenter will automatically fall back to transformers HuBERT
- Install: `pip install transformers`

### Performance Issues

**Slow inference:**
- Use GPU: Set `device='cuda'` (requires `pip install torch` with CUDA)
- Reduce audio length: Segment long files into chunks
- Use faster alternative: Try Sylber instead

**Too many/too few segments:**
- Adjust `sec_per_syllable`: Lower = more segments, higher = fewer segments
- Enable merging: Set `merge_threshold=0.3` to merge similar segments
- Try different layer: Lower layers = coarser, higher layers = finer

### Memory Issues

**CUDA out of memory:**
```python
# Process in smaller chunks
import numpy as np

def segment_in_chunks(audio, sr, segmenter, chunk_duration=10.0):
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    offset = 0.0
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i+chunk_samples]
        syllables = segmenter.segment(chunk, sr=sr)
        
        # Adjust timestamps
        adjusted = [(s+offset, p+offset, e+offset) for s, p, e in syllables]
        chunks.extend(adjusted)
        
        offset += len(chunk) / sr
    
    return chunks
```

## References

**Paper:**
```bibtex
@inproceedings{peng2023syllable,
  title={Syllable Discovery and Cross-Lingual Generalization in a Visually Grounded, Self-Supervised Speech Model},
  author={Peng, Puyuan and Harwath, David},
  booktitle={Interspeech 2023},
  year={2023}
}
```

**Code:** https://github.com/jasonppy/syllable-discovery

**Model:** https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/

## Next Steps

1. **Try VG-HuBERT on your data:**
   ```bash
   python test_vg_hubert.py
   ```

2. **Compare with Sylber:**
   See which works better for your use case

3. **Implement SD-HuBERT:**
   Next end-to-end method (better performance than VG-HuBERT)

4. **Run full benchmark:**
   Evaluate all methods across all datasets

---

## Future Enhancements (Stretch Goals)

### Convert VG-HuBERT to HuggingFace Hub Model

**Goal**: Upload VG-HuBERT as a proper HuggingFace model to enable auto-download like Sylber.

**Current situation:**
- Sylber: Auto-downloads from HuggingFace Hub (seamless user experience)
- VG-HuBERT: Requires manual wget + tar download (~500MB)

**Proposed improvement:**
1. Fork the original VG-HuBERT repository: https://github.com/jasonppy/syllable-discovery
2. Convert model to HuggingFace format (HubertModel with custom weights)
3. Upload to HuggingFace Hub (e.g., `harwath/vg-hubert-base`)
4. Update `VGHubertSegmenter` to use `transformers.AutoModel.from_pretrained()`
5. Remove manual download requirement

**Benefits:**
- ✅ Consistent API across all neural models
- ✅ Automatic model downloading
- ✅ Version control via HuggingFace Hub
- ✅ Better discoverability
- ✅ Simplified installation instructions

**Implementation steps:**
```python
# After conversion, usage would be:
from transformers import AutoModel

# Auto-download on first use (like Sylber)
model = AutoModel.from_pretrained('harwath/vg-hubert-base')

# Current code in VGHubertSegmenter would simplify to:
if self._model is None:
    from transformers import HubertModel
    self._model = HubertModel.from_pretrained('harwath/vg-hubert-base')
    self._model.eval()
    self._model = self._model.to(device)
```

**Considerations:**
- Need permission from original authors (Puyuan Peng, David Harwath)
- Preserve attribution and licensing
- Maintain compatibility with original checkpoint format
- Document conversion process for reproducibility

**Status**: Planned for future release (not currently prioritized)
