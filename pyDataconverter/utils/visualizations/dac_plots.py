"""
DAC Visualization Utilities
============================

Common plotting functions for DAC testbenches. These are architecture-agnostic
and can be used with any DAC that implements the DACBase interface.

Functions:
    plot_transfer_curve:  Ideal vs actual output voltage and error in LSBs.
    plot_inl_dnl:         INL and DNL bar charts from a code sweep.
    plot_output_spectrum: FFT of DAC output. Clips to first Nyquist zone by
                          default; pass full_spectrum=True to show all zones
                          with Nyquist shading and ZOH sinc envelope.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.visualizations.fft_plots import plot_fft
import pyDataconverter.utils.metrics as metrics


def _get_voltage(dac, code: int) -> float:
    """Extract scalar voltage from a DAC conversion (handles differential)."""
    result = dac.convert(code)
    if isinstance(result, tuple):
        return result[0] - result[1]
    return result


def plot_transfer_curve(dac,
                        title: Optional[str] = None) -> Tuple:
    """
    Sweep all codes through the DAC and plot the transfer curve with error.

    Compares the actual DAC output (with non-idealities) against an ideal
    reference DAC. For differential DACs, the differential voltage
    (v_pos - v_neg) is plotted.

    Args:
        dac: DAC instance with a convert(int) method and n_bits attribute.
        title: Plot title. Defaults to the DAC's __repr__.

    Returns:
        Tuple of (fig, (ax1, ax2))
    """
    n_codes = 2 ** dac.n_bits
    codes = np.arange(n_codes)

    # Ideal reference: same n_bits and v_ref, no non-idealities, single-ended
    ideal_dac = SimpleDAC(dac.n_bits, dac.v_ref, OutputType.SINGLE)
    ideal_voltages = np.array([ideal_dac.convert(c) for c in codes])

    # Actual DAC output
    actual_voltages = np.array([_get_voltage(dac, c) for c in codes])

    lsb = dac.v_ref / (n_codes - 1)
    error_lsb = (actual_voltages - ideal_voltages) / lsb

    plot_title = title if title is not None else repr(dac)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top: transfer curve ---
    ax1.plot(codes, ideal_voltages, color='gray', linestyle='--',
             linewidth=1.0, label='Ideal')
    ax1.plot(codes, actual_voltages, color='steelblue',
             linewidth=1.5, label='Actual')
    ax1.set_ylabel('Output Voltage (V)')
    ax1.set_title(f'DAC Transfer Curve: {plot_title}\nLSB = {lsb * 1000:.4f} mV')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.4)

    # --- Bottom: output error ---
    ax2.plot(codes, error_lsb, color='steelblue', linewidth=1.0)
    ax2.axhline( 0.5, color='gray', linewidth=0.7, linestyle=':')
    ax2.axhline( 0.0, color='black', linewidth=0.7)
    ax2.axhline(-0.5, color='gray', linewidth=0.7, linestyle=':')
    ax2.set_xlabel('Input Code')
    ax2.set_ylabel('Output Error (LSB)')
    ax2.set_ylim(-0.75, 0.75)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_inl_dnl(dac,
                 title: Optional[str] = None) -> Tuple:
    """
    Compute and plot INL and DNL for the DAC.

    DNL measures the deviation of each code step from the ideal 1-LSB step.
    INL measures the cumulative deviation from the ideal transfer function,
    with an endpoint-fit correction (removes offset and gain from INL per
    IEEE 1057).

    For differential DACs, the differential voltage (v_pos - v_neg) is used.

    Args:
        dac: DAC instance with a convert(int) method and n_bits attribute.
        title: Plot title prefix. Defaults to the DAC's __repr__.

    Returns:
        Tuple of (fig, (ax1, ax2))
    """
    n_codes = 2 ** dac.n_bits
    codes = np.arange(n_codes)
    ideal_lsb = dac.v_ref / (n_codes - 1)

    # Actual DAC output
    actual = np.array([_get_voltage(dac, c) for c in codes])

    # Ideal voltages
    ideal = codes * ideal_lsb

    # DNL: deviation of each step from ideal 1-LSB step
    steps = np.diff(actual)
    dnl = steps / ideal_lsb - 1.0

    # INL: endpoint-fit corrected deviation from ideal
    # Best-fit line through code 0 and code (2^N - 1)
    raw_inl = (actual - ideal) / ideal_lsb
    # Endpoint fit: subtract the line from code 0 to code N-1
    endpoint_line = np.linspace(raw_inl[0], raw_inl[-1], n_codes)
    inl = raw_inl - endpoint_line

    max_dnl = np.max(np.abs(dnl))
    max_inl = np.max(np.abs(inl))

    plot_title = title if title is not None else repr(dac)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Bar width: thin for high resolution, visible for low
    bar_width = 1.0 if dac.n_bits <= 8 else 1.0
    bar_lw = 0.5 if dac.n_bits <= 12 else 0

    # --- Top: DNL ---
    ax1.bar(codes[1:], dnl, width=bar_width, linewidth=bar_lw,
            color='steelblue', alpha=0.7)
    ax1.axhline( 0.0, color='black', linewidth=0.7)
    ax1.axhline( 0.5, color='gray', linewidth=0.7, linestyle=':')
    ax1.axhline(-0.5, color='gray', linewidth=0.7, linestyle=':')
    ax1.axhline( 1.0, color='red',  linewidth=0.7, linestyle=':')
    ax1.axhline(-1.0, color='red',  linewidth=0.7, linestyle=':')
    ax1.set_xlabel('Input Code')
    ax1.set_ylabel('DNL (LSB)')
    ax1.set_title(f'DNL \u2014 Max |DNL| = {max_dnl:.3f} LSB\n{plot_title}')
    ax1.grid(True, alpha=0.4)

    # --- Bottom: INL ---
    ax2.bar(codes, inl, width=bar_width, linewidth=bar_lw,
            color='steelblue', alpha=0.7)
    ax2.axhline( 0.0, color='black', linewidth=0.7)
    ax2.axhline( 0.5, color='gray', linewidth=0.7, linestyle=':')
    ax2.axhline(-0.5, color='gray', linewidth=0.7, linestyle=':')
    ax2.set_xlabel('Input Code')
    ax2.set_ylabel('INL (LSB)')
    ax2.set_title(f'INL \u2014 Max |INL| = {max_inl:.3f} LSB')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_output_spectrum(freqs: np.ndarray,
                         mags: np.ndarray,
                         fs: float,
                         title: Optional[str] = None,
                         nyquist_zone: int = 1,
                         precomputed_metrics: dict = None):
    """
    Plot a DAC output spectrum from a pre-computed FFT.

    The caller is responsible for generating the signal, running
    convert_sequence, and computing the FFT.  This function handles
    the DAC-specific display: metrics annotation for a single Nyquist
    zone, or full-bandwidth Nyquist zone shading with ZOH sinc envelope.

    Args:
        freqs: Frequency array from compute_fft (Hz).
        mags: Magnitude array from compute_fft (dBFS).
        fs: DAC update rate (Hz). Sets the Nyquist zone boundaries.
        title: Plot title.
        nyquist_zone: Which Nyquist zone to display. Defaults to 1 (0 to fs/2).
            Pass any integer N to show zone N ((N-1)*fs/2 to N*fs/2).
            Pass None to show the full bandwidth with zone shading and
            ZOH sinc envelope.
        precomputed_metrics: Optional metrics dict (ignored when
            nyquist_zone=None). Computed automatically when None.

    Returns:
        ax: The matplotlib axis used for the plot.
    """
    fs_fft     = len(freqs) * 2 * (freqs[1] - freqs[0])
    plot_title = title if title is not None else 'DAC Output Spectrum'

    if nyquist_zone is None:
        # Full bandwidth — all Nyquist zones with sinc envelope
        oversample   = max(1, round(fs_fft / fs))
        min_db       = -120
        mags_clipped = np.clip(np.where(np.isfinite(mags), mags, min_db), min_db, None)
        sinc_env     = np.clip(20 * np.log10(np.abs(np.sinc(freqs / fs)) + 1e-20), min_db, 0)
        n_zones      = int(np.ceil((fs_fft / 2) / (fs / 2)))
        zone_colors  = ['#ddeeff', '#ffeedd']

        fig, ax = plt.subplots(figsize=(14, 5))

        for k in range(n_zones):
            f_lo = k * fs / 2
            f_hi = min((k + 1) * fs / 2, fs_fft / 2)
            lbl  = f'Nyquist zone {k + 1}' if k < 2 else f'_z{k}'
            ax.axvspan(f_lo / 1e6, f_hi / 1e6, alpha=0.35,
                       color=zone_colors[k % 2], label=lbl)

        for k in range(1, n_zones):
            ax.axvline(k * fs / 2 / 1e6, color='gray', linewidth=0.7, linestyle=':')
            ax.text(k * fs / 2 / 1e6, -8, f'Z{k + 1}',
                    fontsize=7, ha='center', va='top', color='dimgray')

        ax.fill_between(freqs / 1e6, min_db, mags_clipped, alpha=0.25, color='steelblue')
        ax.plot(freqs / 1e6, mags_clipped, color='steelblue', linewidth=0.8,
                label='DAC spectrum')
        ax.plot(freqs / 1e6, sinc_env, color='tomato', linewidth=1.5,
                linestyle='--', label='ZOH sinc envelope')

        ax.set_xlim(0, fs_fft / 2 / 1e6)
        ax.set_ylim(min_db, 0)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Magnitude (dBFS)')
        n_fft = len(freqs) * 2
        ax.set_title(f'{plot_title} — All {n_zones} Nyquist Zones (NFFT={n_fft}, ×{oversample} ZOH)')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, axis='y', alpha=0.4, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
    else:
        # Single Nyquist zone with metrics
        f_lo      = (nyquist_zone - 1) * fs / 2
        f_hi      = nyquist_zone * fs / 2
        zone_mask = (freqs >= f_lo) & (freqs <= f_hi)
        freqs_z   = freqs[zone_mask]
        mags_z    = mags[zone_mask]
        pos_mask  = freqs_z > 0
        f0        = freqs_z[pos_mask][np.argmax(mags_z[pos_mask])]
        plot_metrics = precomputed_metrics if precomputed_metrics is not None else \
            metrics._calculate_dac_dynamic_metrics_from_fft(freqs=freqs_z, mags=mags_z,
                                                            fs=fs_fft, f0=f0, full_scale=None)
        zone_title = f'{plot_title} — Nyquist Zone {nyquist_zone}' if nyquist_zone > 1 else plot_title
        ax = plot_fft(freqs, mags, title=zone_title,
                      min_freq=f_lo if nyquist_zone > 1 else None,
                      max_freq=f_hi,
                      metrics=plot_metrics, metrics_dbfs=True)

    return ax
