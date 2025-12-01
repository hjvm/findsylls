import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ast import literal_eval
from typing import Optional, Tuple
from .layers import plot_background
from .overlays import plot_nuclei_overlay, plot_boundary_overlay, plot_span_overlay
from ..audio.utils import load_audio
from ..envelope.dispatch import get_amplitude_envelope
from ..parsing.textgrid_parser import parse_textgrid_intervals

def safe_parse(val):
    """Parse stringified lists, handling numpy types."""
    if isinstance(val, str):
        # First, try direct literal_eval (fastest, works for clean data)
        try:
            return literal_eval(val)
        except (ValueError, SyntaxError):
            # If that fails, try cleaning numpy type wrappers
            try:
                import re
                # Match np.float64(number) or np.int64(number) etc and extract just the number
                val_cleaned = re.sub(r'np\.\w+\(([-\d.e]+)\)', r'\1', val)
                return literal_eval(val_cleaned)
            except Exception:
                # If all parsing fails, return empty list
                return []
    return val

def plot_segmentation_result(df: pd.DataFrame, file_id: str, envelope_fn: str = "sbs", envelope_kwargs: dict = None, figsize: Tuple[int, int] = (15, 9), show: bool = True, phone_tier: Optional[int] = None, syll_tier: Optional[int] = None, word_tier: Optional[int] = None):
    df = df[df["file_id"] == file_id]
    if df.empty:
        print(f"No data found for file_id: {file_id}"); return
    audio_file = df.iloc[0]["audio_file"]
    audio, sr = load_audio(audio_file)
    audio = audio.mean(0) if getattr(audio, 'ndim', 1) > 1 else audio
    t_audio = np.linspace(0, len(audio)/sr, len(audio))
    A, t_env = get_amplitude_envelope(audio, sr, method=envelope_fn, **(envelope_kwargs or {}))
    if A.max() > 0: A = A / A.max()
    fig, axes = plt.subplots(df.shape[0], 1, figsize=figsize, sharex=True)
    fig.suptitle(f"Segmentation Results for {file_id} using {envelope_fn} envelope.")
    if df.shape[0] == 1:
        axes = [axes]
    for i, method in enumerate(df.eval_method):
        ax = axes[i]; row = df[df["eval_method"] == method]
        if row.empty:
            ax.set_title(f"{method} (not available)"); continue
        ax.set_title(method)
        row = row.iloc[0]
        matches = safe_parse(row.get("matches", []))
        substitutions = safe_parse(row.get("substitutions", []))
        insertions = safe_parse(row.get("insertions", []))
        deletions = safe_parse(row.get("deletions", []))
        phone_intervals = parse_textgrid_intervals(row['tg_file'], phone_tier) if phone_tier is not None else None
        syll_intervals = parse_textgrid_intervals(row['tg_file'], syll_tier) if syll_tier is not None else None
        word_intervals = parse_textgrid_intervals(row['tg_file'], word_tier) if word_tier is not None else None
        plot_background(ax, t_audio, audio, t_env, A)
        if method == "nuclei":
            plot_nuclei_overlay(ax, t_env, A, matches, insertions, deletions, phone_intervals)
        elif method == "syll_boundaries":
            plot_boundary_overlay(ax, matches, insertions, deletions, syll_intervals)
        elif method == "syll_spans":
            plot_span_overlay(ax, matches, substitutions, insertions, deletions, syll_intervals)
        elif method == "word_boundaries":
            plot_boundary_overlay(ax, matches, insertions, deletions, word_intervals)
        elif method == "word_spans":
            plot_span_overlay(ax, matches, substitutions, insertions, deletions, word_intervals)
        ax.legend(loc='upper right')
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0,0,1,0.97])
    if show: plt.show()
    return fig, axes[-1]
