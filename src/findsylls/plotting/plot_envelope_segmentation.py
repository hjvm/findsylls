"""
Plot envelope-based segmentation results without evaluation.

Shows the waveform, envelope, syllable boundaries, and nuclei.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def plot_envelope_segmentation(
    audio: np.ndarray,
    sr: int,
    envelope: np.ndarray,
    envelope_times: np.ndarray,
    segments: List[Tuple[float, float, float]],
    title: str = "Envelope-Based Segmentation",
    figsize: Tuple[int, int] = (14, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot waveform with envelope and segmentation results.
    
    Displays:
    - Waveform in gray (background)
    - Amplitude envelope in blue
    - Vertical lines at syllable boundaries (green)
    - Dots at syllable nuclei/peaks (red)
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate in Hz
        envelope: Amplitude envelope
        envelope_times: Time points for envelope (in seconds)
        segments: List of (start, peak, end) tuples in seconds
        title: Plot title
        figsize: Figure size (width, height)
        ax: Optional matplotlib axes to plot on
        
    Returns:
        matplotlib Axes object
        
    Example:
        >>> from findsylls.audio.utils import load_audio
        >>> from findsylls.envelope import HilbertEnvelope
        >>> from findsylls.segmentation.peakdetect_segmenter import PeakdetectSegmenter
        >>> 
        >>> audio, sr = load_audio('audio.wav')
        >>> env_computer = HilbertEnvelope()
        >>> segmenter = PeakdetectSegmenter(env_computer)
        >>> 
        >>> envelope, times = env_computer.compute(audio, sr)
        >>> segments = segmenter.segment(audio, sr)
        >>> 
        >>> plot_envelope_segmentation(audio, sr, envelope, times, segments)
        >>> plt.show()
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create time axis for waveform
    t_audio = np.linspace(0, len(audio) / sr, len(audio))
    
    # Normalize waveform and envelope to [-1, 1] range
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-10)
    envelope_norm = envelope / (np.max(envelope) + 1e-10)
    
    # Plot waveform (background)
    ax.plot(t_audio, audio_norm, color='gray', alpha=0.3, linewidth=0.5, label='Waveform')
    
    # Plot envelope
    ax.plot(envelope_times, envelope_norm, color='steelblue', linewidth=1.5, label='Envelope')
    
    # Plot syllable boundaries (vertical lines)
    for i, (start, peak, end) in enumerate(segments):
        # Start boundary
        ax.axvline(
            start, 
            color='green', 
            linestyle='-', 
            linewidth=1.5, 
            alpha=0.6,
            label='Boundaries' if i == 0 else ''
        )
        # End boundary
        ax.axvline(
            end, 
            color='green', 
            linestyle='-', 
            linewidth=1.5, 
            alpha=0.6
        )
    
    # Plot syllable nuclei/peaks (dots)
    peaks = [peak for start, peak, end in segments]
    peak_amplitudes = np.interp(peaks, envelope_times, envelope_norm)
    ax.plot(
        peaks, 
        peak_amplitudes, 
        'ro', 
        markersize=8, 
        label='Nuclei',
        zorder=10  # Draw on top
    )
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Normalized Amplitude', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, t_audio[-1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)
    
    # Add segment count annotation
    ax.text(
        0.02, 0.98, 
        f'Segments: {len(segments)}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    return ax


def plot_multiple_envelope_segmentations(
    audio: np.ndarray,
    sr: int,
    results: dict,
    figsize: Tuple[int, int] = (14, 10),
    suptitle: str = "Envelope-Based Segmentation Comparison"
) -> plt.Figure:
    """
    Plot multiple envelope segmentation results in a grid.
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate in Hz
        results: Dictionary mapping method names to (envelope, times, segments) tuples
                 Example: {'Hilbert': (envelope, times, segments), ...}
        figsize: Figure size (width, height)
        suptitle: Overall figure title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> results = {
        ...     'Hilbert': (hilbert_env, times1, segments1),
        ...     'Theta': (theta_env, times2, segments2),
        ...     'SBS': (sbs_env, times3, segments3)
        ... }
        >>> fig = plot_multiple_envelope_segmentations(audio, sr, results)
        >>> plt.show()
    """
    n_methods = len(results)
    n_cols = 2
    n_rows = (n_methods + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    # Plot each method
    for idx, (method_name, (envelope, times, segments)) in enumerate(results.items()):
        ax = axes_flat[idx]
        plot_envelope_segmentation(
            audio, sr, envelope, times, segments,
            title=f"{method_name}",
            ax=ax
        )
    
    # Hide unused subplots
    for idx in range(len(results), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    return fig
