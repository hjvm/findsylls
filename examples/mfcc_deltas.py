#!/usr/bin/env python3
"""
Quick reference for MFCC delta features in findsylls.

Run this script to see all MFCC variants in action.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.findsylls.embedding.pipeline import embed_audio

# Example audio file
AUDIO = 'test_samples/SP20_117.wav'

print("MFCC Delta Features - Quick Reference")
print("=" * 60)

# 1. Standard MFCCs (13-dim)
emb_13, _ = embed_audio(
    AUDIO, 
    embedder='mfcc', 
    pooling='mean',
    embedder_kwargs={'n_mfcc': 13}
)
print(f"\n1. Standard:          {emb_13.shape[1]:3d}-dim")

# 2. MFCCs + Delta (26-dim)
emb_26, _ = embed_audio(
    AUDIO,
    embedder='mfcc',
    pooling='mean',
    embedder_kwargs={'n_mfcc': 13, 'include_delta': True}
)
print(f"2. + Delta:           {emb_26.shape[1]:3d}-dim")

# 3. MFCCs + Delta + Delta-Delta (39-dim)
emb_39, _ = embed_audio(
    AUDIO,
    embedder='mfcc',
    pooling='mean',
    embedder_kwargs={
        'n_mfcc': 13,
        'include_delta': True,
        'include_delta_delta': True
    }
)
print(f"3. + Delta-Delta:     {emb_39.shape[1]:3d}-dim")

# 4. With ONC pooling (117-dim = 39 × 3)
emb_onc, _ = embed_audio(
    AUDIO,
    segmentation='peaks_and_valleys',
    embedder='mfcc',
    pooling='onc',
    embedder_kwargs={
        'n_mfcc': 13,
        'include_delta': True,
        'include_delta_delta': True
    }
)
print(f"4. ONC (39 × 3):      {emb_onc.shape[1]:3d}-dim")

# 5. Custom: 20 MFCCs with deltas (60-dim)
emb_60, _ = embed_audio(
    AUDIO,
    embedder='mfcc',
    pooling='mean',
    embedder_kwargs={
        'n_mfcc': 20,
        'include_delta': True,
        'include_delta_delta': True
    }
)
print(f"5. Custom (20 × 3):   {emb_60.shape[1]:3d}-dim")

print("\n" + "=" * 60)
print("Usage pattern:")
print("  embedder_kwargs={")
print("      'n_mfcc': 13,               # Number of base coefficients")
print("      'include_delta': True,       # +1st derivatives")
print("      'include_delta_delta': True  # +2nd derivatives (needs delta=True)")
print("  }")
print("=" * 60)
