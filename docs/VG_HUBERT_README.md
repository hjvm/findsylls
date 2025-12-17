# VG-HuBERT Implementation

VG-HuBERT (Visually Grounded HuBERT) syllable segmentation has been successfully implemented in findsylls!

## Quick Start

```python
from findsylls.segmentation.end2end import VGHubertSegmenter
from findsylls.audio.utils import load_audio

# Create segmenter
segmenter = VGHubertSegmenter(
    model_path='/path/to/vg-hubert_3',  # Download from link below
    layer=8,                             # Which HuBERT layer (default: 8)
    sec_per_syllable=0.2,               # Target syllable duration (default: 0.2)
    merge_threshold=0.3,                 # Merge similar segments (optional)
    device='cuda'                        # Use 'cuda', 'cpu', or 'mps'
)

# Segment audio
audio, sr = load_audio('audio.wav', target_sr=16000)
syllables = segmenter.segment(audio, sr=sr)

# Result: list of (start, peak, end) tuples in seconds
for start, peak, end in syllables:
    print(f"Syllable: {start:.3f}s to {end:.3f}s (nucleus at {peak:.3f}s)")
```

## Model Download

Download the pre-trained VG-HuBERT model:

**Option 1: VG-HuBERT_3 (recommended for syllables)**
```bash
wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar
tar -xf vg-hubert_3.tar
```

**Option 2: Use transformers HuBERT (fallback)**
```bash
pip install transformers
# VGHubertSegmenter will automatically use facebook/hubert-base-ls960
```

The model directory should contain:
- `best_bundle.pth` - Model weights (best for syllables)
- `snapshot_20.pth` - Alternative checkpoint (better for words)
- `args.pkl` - Model configuration

## How It Works

VG-HuBERT uses a two-stage segmentation approach:

### 1. Feature Extraction
- Audio → VG-HuBERT → Layer 8 features (typically 768-dim)
- VG-HuBERT is trained on SpokenCOCO (audio-image pairs)
- Features capture visually-grounded speech representations

### 2. MinCut Segmentation
- Compute self-similarity matrix: `SSM = features @ features.T`
- Estimate number of syllables: `K = ceil(audio_duration / 0.2)`
- Run MinCut algorithm to partition SSM into K segments
- MinCut minimizes inter-segment similarity (finds natural boundaries)

### 3. Optional Merging
- Compute cosine similarity between adjacent segments
- Merge segments where `similarity >= merge_threshold`
- Typical threshold: 0.3-0.4

## Parameters

### Required
- `model_path` (str): Path to VG-HuBERT checkpoint directory

### Optional
- `layer` (int, default=8): Which HuBERT layer to use (0-11)
  - Layer 8 works best for syllables
  - Try different layers for different granularities
  
- `sec_per_syllable` (float, default=0.2): Target syllable duration
  - 0.15-0.2 works well for English
  - Adjust for other languages
  
- `merge_threshold` (float, optional): Cosine similarity threshold for merging
  - `None`: No merging (more segments)
  - `0.3-0.4`: Typical range (fewer segments)
  - Higher = more aggressive merging
  
- `reduce_method` (str, default='mean'): How to pool features within segments
  - `'mean'`: Average pooling (most common)
  - `'max'`: Max pooling
  - `'median'`: Median pooling
  
- `device` (str, default='cpu'): Compute device
  - `'cuda'`: NVIDIA GPU (fastest)
  - `'mps'`: Apple Silicon GPU
  - `'cpu'`: CPU (slowest)
  
- `snapshot` (str, default='best'): Which checkpoint to load
  - `'best'`: Load best_bundle.pth (better for syllables)
  - `20`: Load snapshot_20.pth (better for words)

## Performance

From Peng et al. (2023) on SpokenCOCO test set:

| Metric | Value |
|--------|-------|
| Boundary Precision | 0.57 |
| Boundary Recall | 0.64 |
| Boundary F1 | 0.60 |
| Over-segmentation | 0.11 |

Works across multiple languages:
- English (SpokenCOCO)
- French (ZeroSpeech 2020)
- Mandarin (ZeroSpeech 2020)
- Estonian Conversational Speech

## Implementation Details

### MinCut Algorithm
The MinCut algorithm uses dynamic programming to find optimal segment boundaries:

```python
from findsylls.segmentation.algorithms.mincut import min_cut
import numpy as np

# Create self-similarity matrix
features = model.extract_features(audio)  # Shape: (T, D)
ssm = features @ features.T               # Shape: (T, T)
ssm = ssm - np.min(ssm) + 1e-7           # Make non-negative

# Segment into K boundaries (K-1 segments)
K = 11  # For 10 segments
boundaries = min_cut(ssm, K)
# Returns: [0, 8, 19, 31, ..., T-1]
```

The algorithm minimizes:
```
cost = inter_segment_similarity / total_similarity
```

Where:
- `inter_segment_similarity`: Similarity between [j:i] and rest of frames
- `total_similarity`: Total similarity involving [j:i]

This effectively finds boundaries where feature similarity "cuts" are minimal.

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
