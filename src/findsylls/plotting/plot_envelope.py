"""Visualization utilities for envelope computation.

Provides functions to plot amplitude envelopes over audio waveforms
for validation and comparison of different envelope methods.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_envelope_over_waveform(
    audio: np.ndarray,
    sr: int,
    envelope: np.ndarray,
    envelope_times: np.ndarray,
    title: str = "Amplitude Envelope",
    figsize: Tuple[int, int] = (12, 4),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot amplitude envelope over audio waveform.
    
    Args:
        audio: Audio signal (1D array)
        sr: Sample rate in Hz
        envelope: Computed envelope (1D array)
        envelope_times: Time values for envelope (in seconds)
        title: Plot title
        figsize: Figure size if creating new figure
        ax: Existing axes to plot on, or None to create new figure
        
    Returns:
        Matplotlib axes object
        
    Example:
        from findsylls.audio.utils import load_audio
        from findsylls.envelope import HilbertEnvelope
        from findsylls.plotting import plot_envelope_over_waveform
        
        audio, sr = load_audio('audio.wav')
        envelope_computer = HilbertEnvelope()
        envelope, times = envelope_computer.compute(audio, sr)
        plot_envelope_over_waveform(audio, sr, envelope, times, title='Hilbert Envelope')
        plt.show()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Compute time axis for waveform
    waveform_times = np.linspace(0, len(audio) / sr, len(audio))
    
    # Plot waveform
    ax.plot(waveform_times, audio, 'k-', alpha=0.3, linewidth=0.5, label='Waveform')
    
    # Plot envelope
    ax.plot(envelope_times, envelope, 'r-', linewidth=2, label='Envelope')
    
    # Formatting
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set y-limits to match waveform range
    y_max = max(np.abs(audio).max(), envelope.max()) * 1.1
    ax.set_ylim(-y_max, y_max)
    
    return ax


def plot_multiple_envelopes(
    audio: np.ndarray,
    sr: int,
    envelopes: dict,
    figsize: Tuple[int, int] = (14, 10),
    suptitle: str = "Envelope Method Comparison"
) -> plt.Figure:
    """Plot multiple envelopes in a grid for comparison.
    
    Args:
        audio: Audio signal (1D array)
        sr: Sample rate in Hz
        envelopes: Dict mapping method names to (envelope, times) tuples
        figsize: Figure size
        suptitle: Overall figure title
        
    Returns:
        Matplotlib figure object
        
    Example:
        from findsylls.envelope import HilbertEnvelope, ThetaEnvelope, SBSEnvelope
        
        audio, sr = load_audio('audio.wav')
        
        envelopes = {}
        for name, computer in [
            ('Hilbert', HilbertEnvelope()),
            ('Theta', ThetaEnvelope()),
            ('SBS', SBSEnvelope())
        ]:
            env, times = computer.compute(audio, sr)
            envelopes[name] = (env, times)
        
        fig = plot_multiple_envelopes(audio, sr, envelopes)
        plt.show()
    """
    n_methods = len(envelopes)
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()  # Ensure 1D array of axes
    
    for idx, (method_name, (envelope, times)) in enumerate(envelopes.items()):
        ax = axes[idx]
        plot_envelope_over_waveform(
            audio, sr, envelope, times,
            title=f"{method_name} Envelope",
            ax=ax
        )
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave room for suptitle
    
    return fig
