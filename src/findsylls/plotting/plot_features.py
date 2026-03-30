"""
Plot feature extraction results without segmentation.

Shows feature matrices as heatmaps to visualize acoustic/neural representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict


def plot_feature_matrix(
    features: np.ndarray,
    feature_times: np.ndarray,
    title: str = "Feature Matrix",
    figsize: Tuple[int, int] = (12, 4),
    ax: Optional[plt.Axes] = None,
    aspect: str = 'auto',
) -> plt.Axes:
    """
    Plot a feature matrix as a heatmap.
    
    Args:
        features: Feature matrix (time × features) or (features × time)
        feature_times: Time points in seconds
        title: Plot title
        figsize: Figure size (width, height)
        ax: Optional matplotlib axes to plot on
        aspect: Aspect ratio for imshow ('auto' or 'equal')
        
    Returns:
        matplotlib Axes object
        
    Example:
        >>> from findsylls.audio.utils import load_audio
        >>> from findsylls.features import MFCCExtractor
        >>> 
        >>> audio, sr = load_audio('audio.wav')
        >>> extractor = MFCCExtractor(n_mfcc=13)
        >>> features = extractor.extract(audio, sr)
        >>> times = np.linspace(0, len(audio)/sr, features.shape[0])
        >>> 
        >>> plot_feature_matrix(features, times, title='MFCC Features')
        >>> plt.show()
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure features are (time × features) for consistent display
    if features.shape[0] == len(feature_times):
        # Already (time × features), transpose for display (features on Y axis)
        features_display = features.T
        extent = [feature_times[0], feature_times[-1], 0, features.shape[1]]
        ylabel = f'Feature Dimension (n={features.shape[1]})'
    else:
        # Assume (features × time), use directly
        features_display = features
        extent = [feature_times[0], feature_times[-1], 0, features.shape[0]]
        ylabel = f'Feature Dimension (n={features.shape[0]})'
    
    # Plot as heatmap
    im = ax.imshow(
        features_display,
        aspect=aspect,
        origin='lower',
        extent=extent,
        cmap='viridis',
        interpolation='nearest'
    )
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature Value', fontsize=10)
    
    # Add statistics annotation
    stats_text = (
        f'Shape: {features.shape[0]}×{features.shape[1]}\n'
        f'Min: {features.min():.3f}\n'
        f'Max: {features.max():.3f}\n'
        f'Mean: {features.mean():.3f}'
    )
    ax.text(
        0.02, 0.98, 
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=9,
        family='monospace'
    )
    
    return ax


def plot_multiple_feature_matrices(
    audio: np.ndarray,
    sr: int,
    feature_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (14, 10),
    suptitle: str = "Feature Extraction Comparison"
) -> plt.Figure:
    """
    Plot multiple feature extraction results in a grid.
    
    Args:
        audio: Audio signal (for duration reference)
        sr: Sample rate in Hz
        feature_results: Dictionary mapping method names to (features, times) tuples
                        Example: {'MFCC': (features, times), ...}
        figsize: Figure size (width, height)
        suptitle: Overall figure title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> results = {
        ...     'MFCC': (mfcc_features, times1),
        ...     'MelSpec': (melspec_features, times2),
        ...     'HuBERT': (hubert_features, times3)
        ... }
        >>> fig = plot_multiple_feature_matrices(audio, sr, results)
        >>> plt.show()
    """
    n_methods = len(feature_results)
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
    for idx, (method_name, (features, times)) in enumerate(feature_results.items()):
        ax = axes_flat[idx]
        plot_feature_matrix(
            features, times,
            title=f"{method_name}",
            ax=ax
        )
    
    # Hide unused subplots
    for idx in range(len(feature_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    return fig
