import numpy as np
import matplotlib.pyplot as plt

def plot_background(ax, t_audio, audio, t_env, A):
    ax.plot(t_audio, audio / np.max(np.abs(audio)), color='gray', alpha=0.3, label='Waveform')
    ax.plot(t_env, A, color='steelblue', linewidth=1.5, label='Envelope')
    ax.legend(loc='upper right')
