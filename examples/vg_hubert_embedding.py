#!/usr/bin/env python3
"""
VG-HuBERT Feature Extraction Example

Shows how to use VG-HuBERT for syllable embedding extraction with
automatic model downloading from HuggingFace Hub.

Requirements:
    pip install 'findsylls[embedding]' vg-hubert

The vg-hubert package automatically downloads the model from HuggingFace
(hjvm/VG-HuBERT) on first use - no manual setup required!
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
        '--model-ckpt',
        type=str,
        default='hjvm/VG-HuBERT',
        help='HuggingFace model checkpoint or local path (default: hjvm/VG-HuBERT)'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default='test_samples/SP20_117.wav',
        help='Path to audio file'
    )
    args = parser.parse_args()
    
    print("VG-HuBERT Embedding Extraction")
    print("=" * 70)
    print(f"Model: {args.model_ckpt}")
    if args.model_ckpt == 'hjvm/VG-HuBERT':
        print("  (Auto-downloading from HuggingFace Hub on first use)")
    print(f"Audio: {args.audio}")
    print()
    
    # Example 1: Mean pooling (768-dim per syllable)
    print("1. Mean Pooling (768-dim):")
    emb_mean, meta = embed_audio(
        args.audio,
        segmentation='peakdetect',
        features='vg_hubert',
        pooling='mean',
        feature_kwargs={'model_ckpt': args.model_ckpt}
    )
    print(f"   Shape: {emb_mean.shape}")
    print(f"   Syllables: {meta['num_syllables']}")
    
    # Example 2: ONC pooling (2304-dim = 768 × 3)
    print("\n2. ONC Pooling (2304-dim = 768 × 3):")
    emb_onc, _ = embed_audio(
        args.audio,
        segmentation='peakdetect',
        features='vg_hubert',
        pooling='onc',
        feature_kwargs={'model_ckpt': args.model_ckpt}
    )
    print(f"   Shape: {emb_onc.shape}")
    
    # Example 3: Different layer
    print("\n3. Layer 6 + Mean Pooling:")
    emb_l6, _ = embed_audio(
        args.audio,
        segmentation='peakdetect',
        features='vg_hubert',
        pooling='mean',
        layer=6,
        feature_kwargs={'model_ckpt': args.model_ckpt}
    )
    print(f"   Shape: {emb_l6.shape}")
    
    # Example 4: VG-HuBERT end-to-end (segmentation + embedding)
    print("\n4. VG-HuBERT Segmentation + Embedding:")
    emb_e2e, meta_e2e = embed_audio(
        args.audio,
        segmentation='vg_hubert',
        features='vg_hubert',
        pooling='mean',
        segmentation_kwargs={'model_ckpt': args.model_ckpt},
        feature_kwargs={'model_ckpt': args.model_ckpt}
    )
    print(f"   Shape: {emb_e2e.shape}")
    print(f"   Syllables: {meta_e2e['num_syllables']} (VG-HuBERT segmentation)")
    
    print("\n" + "=" * 70)
    print("✅ VG-HuBERT embedding extraction successful!")
    print("=" * 70)
    
    # Show usage patterns
    print("\nUsage patterns:")
    print("  # Mean pooling (auto-download)")
    print("  embed_audio('audio.wav', features='vg_hubert', pooling='mean')")
    print()
    print("  # ONC pooling")
    print("  embed_audio('audio.wav', features='vg_hubert', pooling='onc')")
    print()
    print("  # Different layer")
    print("  embed_audio('audio.wav', features='vg_hubert', layer=6)")
    print()
    print("  # Use local model")
    print("  embed_audio('audio.wav', features='vg_hubert',")
    print("              feature_kwargs={'model_ckpt': '/path/to/model'})")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
