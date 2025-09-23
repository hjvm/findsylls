from matplotlib.patches import Rectangle
import numpy as np

def plot_nuclei_overlay(ax, t_env, A, matches=None, insertions=None, deletions=None, tg_intervals=None):
    matches = matches or []; insertions = insertions or []; deletions = deletions or []
    for p in matches:
        ax.plot(p, np.interp(p, t_env, A), 'go', label='TP Peak' if 'TP Peak' not in ax.get_legend_handles_labels()[1] else "")
    for p in insertions:
        ax.plot(p, np.interp(p, t_env, A), 'kx', label='FP Peak' if 'FP Peak' not in ax.get_legend_handles_labels()[1] else "")
    for s, e in deletions:
        center = (s + e) / 2
        ax.plot(center, np.interp(center, t_env, A), 'r^', label='FN Peak' if 'FN Peak' not in ax.get_legend_handles_labels()[1] else "")
    if tg_intervals:
        for start, end, label in tg_intervals:
            ax.text((start + end)/2, -0.2, label, ha='center', va='top', fontsize=9)

def plot_boundary_overlay(ax, matches=None, insertions=None, deletions=None, tg_intervals=None):
    matches = matches or []; insertions = insertions or []; deletions = deletions or []
    for idx, b in enumerate(matches):
        ax.axvline(b, color='green', linestyle='-', linewidth=1, label='TP boundary' if idx == 0 else "")
    for idx, i in enumerate(insertions):
        ax.axvline(i, color='gold', linestyle='-', linewidth=1, label='Insertion' if idx == 0 else "")
    for idx, d in enumerate(deletions):
        ax.axvline(d, color='red', linestyle='--', linewidth=1, label='Deletion' if idx == 0 else "")
    if tg_intervals:
        for start, end, label in tg_intervals:
            ax.text((start + end)/2, -0.2, label, ha='center', va='top', fontsize=9)

def plot_span_overlay(ax, matches=None, substitutions=None, insertions=None, deletions=None, tg_intervals=None):
    matches = matches or []; substitutions = substitutions or []; insertions = insertions or []; deletions = deletions or []
    y, h = 0, 0.2
    for s, e in matches:
        ax.add_patch(Rectangle((s, y), e - s, h, edgecolor='black', facecolor='green', label='Correct' if 'Correct' not in ax.get_legend_handles_labels()[1] else ""))
    for s, e in substitutions:
        ax.add_patch(Rectangle((s, y - h), e - s, h, edgecolor='black', facecolor='yellow', label='Substitution' if 'Substitution' not in ax.get_legend_handles_labels()[1] else ""))
    for s, e in insertions:
        ax.add_patch(Rectangle((s, y - h), e - s, h, edgecolor='black', facecolor='red', label='Insertion' if 'Insertion' not in ax.get_legend_handles_labels()[1] else ""))
    for s, e in deletions:
        ax.add_patch(Rectangle((s, y), e - s, h, edgecolor='black', facecolor='red', label='Deletion' if 'Deletion' not in ax.get_legend_handles_labels()[1] else ""))
    if tg_intervals:
        for start, end, label in tg_intervals:
            ax.text((start + end)/2, -0.2, label, ha='center', va='top', fontsize=9, rotation=45, color='blue')
