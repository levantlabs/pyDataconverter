"""
DAC Visualization Utilities
============================

Common plotting functions for DAC testbenches. These are architecture-agnostic
and can be used with any DAC that implements the DACBase interface.

Functions:
    plot_transfer_curve: Ideal vs actual output voltage and error in LSBs.
    plot_inl_dnl: INL and DNL bar charts from a code sweep.
    plot_output_spectrum: FFT of DAC output driven by a sinusoidal code sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.fft_analysis import compute_fft, FFTNormalization
from pyDataconverter.utils.visualizations.fft_plots import plot_fft
from pyDataconverter.utils.signal_gen import generate_digital_sine


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


def plot_output_spectrum(dac,
                         fs: float,
                         f_sig: float,
                         n_fft: int = 4096,
                         window: str = 'hann',
                         title: Optional[str] = None,
                         metrics: dict = None):
    """
    Plot the output spectrum of a DAC driven by a sinusoidal code sequence.

    Generates a digital sine wave, converts each code through the DAC, then
    computes and plots the FFT of the output voltage. Coherent sampling is
    used (f_sig is snapped to the nearest FFT bin) to eliminate spectral
    leakage.

    Oversampling note: the DAC update rate ``fs`` sets the Nyquist bandwidth.
    When ``fs >> f_sig``, the noise floor spreads across a wider bandwidth,
    lowering the in-band noise density. This is visible in the spectrum as
    a lower noise floor relative to the signal.

    Args:
        dac: DAC instance.
        fs: DAC update rate (Hz).
        f_sig: Desired signal frequency (Hz).
        n_fft: FFT size (default 4096).
        window: Window function for FFT (default 'hann').
        title: Plot title. Defaults to the DAC's __repr__.
        metrics: Pre-computed metrics dict to display in annotation box.

    Returns:
        ax: The matplotlib axis used for the plot.
    """
    # Snap to nearest FFT bin for coherent sampling
    n_fin = max(1, round(f_sig * n_fft / fs))
    f_actual = n_fin / n_fft * fs
    duration = n_fft / fs

    # Generate digital sine code sequence
    codes = generate_digital_sine(dac.n_bits, f_actual, fs,
                                  amplitude=0.9, offset=0.5,
                                  duration=duration)

    # Convert each code through the DAC
    voltages = np.array([_get_voltage(dac, int(c)) for c in codes])

    # Compute FFT
    freqs, mags = compute_fft(voltages, fs,
                              window=window,
                              normalization=FFTNormalization.DBFS,
                              full_scale=dac.v_ref)

    # Plot
    plot_title = title if title is not None else f'DAC Output Spectrum: {repr(dac)}'
    ax = plot_fft(freqs, mags, title=plot_title,
                  max_freq=fs / 2,
                  metrics=metrics, metrics_dbfs=True)

    return ax
