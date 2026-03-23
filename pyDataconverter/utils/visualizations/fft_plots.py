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
             title: str = "FFT Spectrum",
             max_freq: float = None,
             min_db: float = None,
             max_db: float = None,
             metrics: dict = None,
             show_metrics: bool = True,
             metrics_dbfs: bool = False,
             fig=None,
             ax=None):
    """
    Plot an FFT spectrum with automatic frequency unit selection.

    Infinite or NaN values (e.g. log10 of zero at DC) are clipped to the
    floor of the plot so they don't compress the y-axis scale.

    Args:
        freqs: Frequency array in Hz
        mags: Magnitude array in dB
        title: Plot title
        max_freq: Maximum frequency to display in Hz. If None, shows full spectrum.
        min_db: Floor of the y-axis in dB. If None, defaults to max_db - 120 (i.e. -120 dBFS).
        max_db: Ceiling of the y-axis in dB. If None, defaults to 0 dBFS.
        metrics: Dict of performance metrics from calculate_adc_dynamic_metrics.
                 If provided and show_metrics is True, key metrics are displayed
                 as an annotation box on the plot.
        show_metrics: Whether to render the metrics annotation (default True).
                      Set to False to suppress the box even if metrics are passed.
        metrics_dbfs: If True, labels SNR/SNDR/SFDR/THD as dBFS instead of dB.
                      Use when metrics were computed with dBFS normalization.
        fig: Existing matplotlib figure to plot on (optional)
        ax: Existing matplotlib axis to plot on (optional)

    Returns:
        ax: The matplotlib axis used for the plot
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Apply frequency limit
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

    # Determine y-axis bounds — top is always 0 dBFS unless overridden
    if max_db is None:
        max_db = 0
    if min_db is None:
        min_db = max_db - 120

    # Clip -inf / nan to floor so they render at the bottom, not off-axis
    mags_clipped = np.clip(np.where(np.isfinite(mags), mags, min_db), min_db, None)

    freqs_scaled = freqs / freq_scale

    # Fill under the spectrum, then draw the line on top
    ax.fill_between(freqs_scaled, min_db, mags_clipped,
                    alpha=0.15, color='steelblue')
    ax.plot(freqs_scaled, mags_clipped,
            color='steelblue', linewidth=0.8)

    # Styling
    ax.set_xlim(0, plot_max)
    ax.set_ylim(min_db, max_db)
    ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=11)
    ax.set_ylabel('Magnitude (dBFS)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Horizontal gridlines only, at every 20 dB
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.grid(False, axis='x')

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Metrics annotation box
    if metrics is not None and show_metrics:
        # Use dBFS variants if available and not explicitly overridden
        use_dbfs = metrics_dbfs if metrics_dbfs else 'SNR_dBFS' in metrics
        suffix = '_dBFS' if use_dbfs else ''
        db_label = 'dBFS' if use_dbfs else 'dB'
        keys = ['SNR', 'SNDR', 'SFDR', 'THD', 'ENOB']
        units = {'SNR': db_label, 'SNDR': db_label, 'SFDR': db_label, 'THD': db_label, 'ENOB': 'bits'}
        lines = []
        for k in keys:
            dict_key = (k + suffix) if k != 'ENOB' else k  # ENOB is in bits, no dBFS variant
            if dict_key in metrics:
                v = metrics[dict_key]
                u = units[k]
                lines.append(f'{k}: {v:.2f} {u}')
        if lines:
            box_text = '\n'.join(lines)
            ax.text(0.98, 0.97, box_text,
                    transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='lightgray', alpha=0.85))

    plt.tight_layout()
    return ax
