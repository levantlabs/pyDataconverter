"""
FFT Visualization Utilities
===========================

Plotting functions for FFT-based analysis. Separated from fft_analysis.py
so that computation and visualization concerns are independent.

Functions:
    plot_fft: Plot an FFT spectrum with automatic frequency unit scaling.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_fft(freqs: np.ndarray,
             mags: np.ndarray,
             title: str = "FFT",
             max_freq: float = None,
             min_db: float = None,
             max_db: float = None,
             fig=None,
             ax=None):
    """
    Plot an FFT spectrum with automatic frequency unit selection.

    Args:
        freqs: Frequency array in Hz
        mags: Magnitude array in dB
        title: Plot title
        max_freq: Maximum frequency to display in Hz. If None, shows full spectrum.
        min_db: Minimum dB value for y-axis
        max_db: Maximum dB value for y-axis. If None, auto-scales to data max + 2 dB.
        fig: Existing matplotlib figure to plot on (optional)
        ax: Existing matplotlib axis to plot on (optional)

    Returns:
        ax: The matplotlib axis used for the plot
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Apply frequency limit if specified
    if max_freq:
        mask = freqs <= max_freq
        freqs = freqs[mask]
        mags = mags[mask]
    else:
        max_freq = np.max(freqs)

    # Auto-select frequency unit
    if max_freq >= 1e9:
        freq_scale, freq_unit = 1e9, 'GHz'
    elif max_freq >= 1e6:
        freq_scale, freq_unit = 1e6, 'MHz'
    elif max_freq >= 1e3:
        freq_scale, freq_unit = 1e3, 'kHz'
    else:
        freq_scale, freq_unit = 1, 'Hz'

    scaled_max = max_freq / freq_scale
    plot_max = 1 if scaled_max <= 1 else np.ceil(scaled_max)

    ax.plot(freqs / freq_scale, mags)
    ax.grid(True)
    ax.set_xlabel(f'Frequency ({freq_unit})')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.set_xlim(0, plot_max)

    if max_db is None:
        max_db = np.ceil(np.max(mags)) + 2
    ax.set_ylim(min_db, max_db)

    return ax
