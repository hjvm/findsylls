"""
Before/after comparison: segmentation with and without SAD on Kono data.

Shows how SAD prevents boundaries from falling in silent regions between utterances.
Compares SBSPeakdetectSegmenter and ThetaOscillatorSegmenter.

Usage:
    python scripts/sad_before_after.py
    python scripts/sad_before_after.py --save
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from findsylls.audio.utils import load_audio
from findsylls.envelope.sbs import SBSEnvelope
from findsylls.envelope.theta import ThetaEnvelope
from findsylls.segmentation.presets import SBSPeakdetectSegmenter, ThetaOscillatorSegmenter
from findsylls.vad import EnergyVAD
from findsylls.plotting.plot_envelope_segmentation import plot_envelope_segmentation

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kono', 'audio')

FILES = [
    ("section4/IAteMeatYesterday_53840.wav", "Kono: 'I Ate Meat Yesterday' (6 utterances, 12.4s)"),
    ("section3/WhoWillCryTomorrow_19113.wav", "Kono: 'Who Will Cry Tomorrow' (3 utterances, 8.8s)"),
]


def shade_speech_regions(ax, regions, ymin=-1.15, ymax=1.15, alpha=0.12):
    """Shade detected speech regions in light blue."""
    for s, e in regions:
        ax.axvspan(s, e, ymin=0, ymax=1, color='steelblue', alpha=alpha, zorder=0)


def make_comparison_figure(audio_path: str, file_label: str) -> plt.Figure:
    audio, sr = load_audio(audio_path)

    sbs_env_computer = SBSEnvelope()
    theta_env_computer = ThetaEnvelope(f=5, Q=0.5, N=8)

    sbs_envelope, sbs_times = sbs_env_computer.compute(audio, sr)
    theta_envelope, theta_times = theta_env_computer.compute(audio, sr)

    vad = EnergyVAD()
    speech_regions = vad.get_speech_regions(audio, sr)

    # 3 rows × 2 cols: left col = SBS, right col = Theta
    # Row 0: no SAD, no boundaries
    # Row 1: no SAD, with boundaries
    # Row 2: with SAD
    configs = [
        # (row, col, label, segmenter, envelope, times, show_vad)
        (0, 0, "SBS — no SAD, no boundaries",
         SBSPeakdetectSegmenter(sad=None, add_utterance_boundaries=False),
         sbs_envelope, sbs_times, False),
        (1, 0, "SBS — no SAD, with boundaries",
         SBSPeakdetectSegmenter(sad=None, add_utterance_boundaries=True),
         sbs_envelope, sbs_times, False),
        (2, 0, "SBS — with SAD",
         SBSPeakdetectSegmenter(sad="energy"),
         sbs_envelope, sbs_times, True),
        (0, 1, "Theta — no SAD, no boundaries",
         ThetaOscillatorSegmenter(sad=None, add_utterance_boundaries=False),
         theta_envelope, theta_times, False),
        (1, 1, "Theta — no SAD, with boundaries",
         ThetaOscillatorSegmenter(sad=None, add_utterance_boundaries=True),
         theta_envelope, theta_times, False),
        (2, 1, "Theta — with SAD",
         ThetaOscillatorSegmenter(sad="energy"),
         theta_envelope, theta_times, True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(20, 12), sharex=True, sharey=False)
    fig.suptitle(file_label, fontsize=13, fontweight='bold')

    for row, col, label, segmenter, envelope, times, show_vad in configs:
        ax = axes[row, col]
        segments = segmenter.segment(audio, sr)

        plot_envelope_segmentation(
            audio, sr, envelope, times, segments,
            title=f"{label}  [{len(segments)} segments]",
            ax=ax,
        )

        shade_speech_regions(ax, speech_regions)

        if show_vad:
            for s, e in speech_regions:
                ax.axvline(s, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.7)
                ax.axvline(e, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.7)

        # Only bottom row gets x-label
        if row < 2:
            ax.set_xlabel('')

    # Legend on top-left panel only
    speech_patch = mpatches.Patch(color='steelblue', alpha=0.25, label='VAD speech region')
    vad_line = plt.Line2D([0], [0], color='darkorange', linestyle=':', linewidth=1.5, label='VAD boundary')
    axes[0, 0].legend(handles=[speech_patch, vad_line], loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Save figures to scripts/ instead of showing')
    args = parser.parse_args()

    for rel_path, label in FILES:
        audio_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue

        print(f"Processing: {rel_path}")
        fig = make_comparison_figure(audio_path, label)

        if args.save:
            out_name = os.path.splitext(os.path.basename(rel_path))[0] + '_sad_comparison.png'
            out_path = os.path.join(os.path.dirname(__file__), out_name)
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            print(f"  Saved → {out_path}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == '__main__':
    main()
