#!/usr/bin/env python3
"""
VG-HuBERT Feature Extraction Example

Shows how to use VG-HuBERT for syllable embedding extraction.

Requirements:
1. Download VG-HuBERT model:
   wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar
   tar -xf vg-hubert_3.tar -C ~/models/

2. Set model path below or pass as argument:
   python examples/vg_hubert_embedding.py --model-path /path/to/vg-hubert_3
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.findsylls.embedding.pipeline import embed_audio


def main():
    parser = argparse.ArgumentParser(description='VG-HuBERT embedding extraction example')
    parser.add_argument(
        '--model-path',
        type=str,
        default=str(Path.home() / 'models' / 'vg-hubert_3'),
        help='Path to VG-HuBERT model directory'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default='test_samples/SP20_117.wav',
        help='Path to audio file'
    )
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print("=" * 70)
        print("VG-HuBERT MODEL NOT FOUND")
        print("=" * 70)
        print(f"\nModel path: {model_path}")
        print("\nTo download VG-HuBERT:")
        print("  wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar")
        print(f"  tar -xf vg-hubert_3.tar -C {model_path.parent}/")
        print("\nOr specify custom path:")
        print("  python examples/vg_hubert_embedding.py --model-path /your/path")
        print("=" * 70)
        return 1
    
    print("VG-HuBERT Embedding Extraction")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Audio: {args.audio}")
    print()
    
    # Example 1: Mean pooling (768-dim per syllable)
    print("1. Mean Pooling (768-dim):")
    emb_mean, meta = embed_audio(
        args.audio,
        segmentation='peaks_and_valleys',
        embedder='vg_hubert',
        pooling='mean',
        embedder_kwargs={'model_path': str(model_path)}
    )
    print(f"   Shape: {emb_mean.shape}")
    print(f"   Syllables: {meta['num_syllables']}")
    
    # Example 2: ONC pooling (2304-dim = 768 × 3)
    print("\n2. ONC Pooling (2304-dim = 768 × 3):")
    emb_onc, _ = embed_audio(
        args.audio,
        segmentation='peaks_and_valleys',
        embedder='vg_hubert',
        pooling='onc',
        embedder_kwargs={'model_path': str(model_path)}
    )
    print(f"   Shape: {emb_onc.shape}")
    
    # Example 3: Different layer
    print("\n3. Layer 6 + Mean Pooling:")
    emb_l6, _ = embed_audio(
        args.audio,
        segmentation='peaks_and_valleys',
        embedder='vg_hubert',
        pooling='mean',
        layer=6,
        embedder_kwargs={'model_path': str(model_path)}
    )
    print(f"   Shape: {emb_l6.shape}")
    
    # Example 4: VG-HuBERT end-to-end (segmentation + embedding)
    print("\n4. VG-HuBERT Segmentation + Embedding:")
    emb_e2e, meta_e2e = embed_audio(
        args.audio,
        segmentation='vg_hubert',
        embedder='vg_hubert',
        pooling='mean',
        segmentation_kwargs={'model_path': str(model_path)},
        embedder_kwargs={'model_path': str(model_path)}
    )
    print(f"   Shape: {emb_e2e.shape}")
    print(f"   Syllables: {meta_e2e['num_syllables']} (VG-HuBERT segmentation)")
    
    print("\n" + "=" * 70)
    print("✅ VG-HuBERT embedding extraction successful!")
    print("=" * 70)
    
    # Show usage patterns
    print("\nUsage patterns:")
    print("  # Mean pooling")
    print("  embed_audio('audio.wav', embedder='vg_hubert', pooling='mean',")
    print(f"              embedder_kwargs={{'model_path': '{model_path}'}})")
    print()
    print("  # ONC pooling")
    print("  embed_audio('audio.wav', embedder='vg_hubert', pooling='onc',")
    print(f"              embedder_kwargs={{'model_path': '{model_path}'}})")
    print()
    print("  # Different layer")
    print("  embed_audio('audio.wav', embedder='vg_hubert', layer=6,")
    print(f"              embedder_kwargs={{'model_path': '{model_path}'}})")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
