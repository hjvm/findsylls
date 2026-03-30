import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ast import literal_eval
from typing import Optional, Tuple, Union
from .layers import plot_background
from .overlays import plot_nuclei_overlay, plot_boundary_overlay, plot_span_overlay
from ..audio.utils import load_audio
from ..envelope.dispatch import get_amplitude_envelope
from ..envelope.base import EnvelopeComputer
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

def plot_segmentation_result(
    df: pd.DataFrame,
    file_id: str,
    envelope_fn: Optional[str] = "sbs",
    envelope_kwargs: Optional[dict] = None,
    envelope_computer: Optional[EnvelopeComputer] = None,
    figsize: Tuple[int, int] = (15, 9),
    show: bool = True,
    phone_tier: Optional[int] = None,
    syll_tier: Optional[int] = None,
    word_tier: Optional[int] = None
):
    """
    Plot segmentation results with envelope visualization.
    
    Automatically detects and handles both single-tier and multi-tier evaluation formats:
    - Single-tier: has 'tier_level' column, eval_method uses simple names ("nuclei", "boundaries", "spans")
    - Multi-tier: no 'tier_level' column, eval_method uses prefixed names ("syllable_boundaries", "word_spans")
    
    Args:
        df: Results dataframe (from run_evaluation or segment_audio)
        file_id: File identifier to plot
        envelope_fn: Envelope method name ('sbs', 'theta', etc.) if envelope_computer not provided
        envelope_kwargs: Parameters for envelope_fn
        envelope_computer: EnvelopeComputer instance (overrides envelope_fn if provided).
                          Use this for neural method pseudo-envelopes.
        figsize: Figure size (width, height)
        show: Whether to display the plot
        phone_tier: TextGrid phone tier index
        syll_tier: TextGrid syllable tier index
        word_tier: TextGrid word tier index
    
    Returns:
        (fig, last_axis) tuple
    
    Example:
        >>> # Classical method (uses string name)
        >>> plot_segmentation_result(df, "file1", envelope_fn="sbs")
        >>> 
        >>> # Neural method (uses pseudo-envelope via OOP)
        >>> from findsylls.features import SylberFeatureExtractor
        >>> from findsylls.envelope import GreedyCosineEnvelope
        >>> extractor = SylberFeatureExtractor(device='cuda')
        >>> envelope = GreedyCosineEnvelope(extractor, window_size=5)
        >>> plot_segmentation_result(df, "file1", envelope_computer=envelope)
    """
    df = df[df["file_id"] == file_id]
    if df.empty:
        print(f"No data found for file_id: {file_id}"); return
    audio_file = df.iloc[0]["audio_file"]
    audio, sr = load_audio(audio_file)
    audio = audio.mean(0) if getattr(audio, 'ndim', 1) > 1 else audio
    t_audio = np.linspace(0, len(audio)/sr, len(audio))
    
    # Compute envelope using either EnvelopeComputer or legacy string method
    if envelope_computer is not None:
        A, t_env = envelope_computer.compute(audio, sr)
        envelope_label = envelope_computer.__class__.__name__
    else:
        A, t_env = get_amplitude_envelope(audio, sr, method=envelope_fn, **(envelope_kwargs or {}))
        envelope_label = envelope_fn
    
    if A.max() > 0: A = A / A.max()
    fig, axes = plt.subplots(df.shape[0], 1, figsize=figsize, sharex=True)
    fig.suptitle(f"Segmentation Results for {file_id} using {envelope_label} envelope.")
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
        
        # Determine tier and evaluation type from either single-tier or multi-tier format
        # Single-tier format: has 'tier_level' column, method names are simple ("nuclei", "boundaries", "spans")
        # Multi-tier format: no 'tier_level' column, method names are prefixed ("syllable_boundaries", "word_spans")
        if 'tier_level' in row.index and pd.notna(row.get('tier_level')):
            # Single-tier format
            tier_name = row['tier_level']
            eval_type = method
        else:
            # Multi-tier format: extract tier from prefixed method name
            if '_' in method and method != 'nuclei':
                # Split method like "syllable_boundaries" -> ("syllable", "boundaries")
                # Use rsplit to handle method names with underscores correctly
                parts = method.rsplit('_', 1)
                tier_name = parts[0] if len(parts) == 2 else 'syllable'
                eval_type = parts[1] if len(parts) == 2 else method
            else:
                # Nuclei or unprefixed name
                tier_name = 'syllable'
                eval_type = method
        
        # Map tier name to appropriate intervals
        if tier_name == 'syllable':
            eval_intervals = syll_intervals
        elif tier_name == 'word':
            eval_intervals = word_intervals
        else:
            eval_intervals = syll_intervals  # Default fallback
        
        plot_background(ax, t_audio, audio, t_env, A)
        
        # Plot based on evaluation type (without tier prefix)
        if eval_type == "nuclei":
            plot_nuclei_overlay(ax, t_env, A, matches, insertions, deletions, phone_intervals)
        elif eval_type == "boundaries":
            if eval_intervals is not None:
                plot_boundary_overlay(ax, matches, insertions, deletions, eval_intervals)
        elif eval_type == "spans":
            if eval_intervals is not None:
                plot_span_overlay(ax, matches, substitutions, insertions, deletions, eval_intervals)
        ax.legend(loc='upper right')
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0,0,1,0.97])
    if show: plt.show()
    return fig, axes[-1]
