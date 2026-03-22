"""
ADC Visualization Utilities
===========================

Common plotting functions for ADC testbenches. These are architecture-agnostic
and can be used with any ADC that implements the ADCBase interface.

Functions:
    plot_conversion: Time/voltage-domain plot of input signal and output codes.
    plot_transfer_function: Sweep an ADC with a ramp and plot code vs. vin
                            plus quantization error in LSBs.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def plot_conversion(x: np.ndarray,
                    input_signal: np.ndarray,
                    output_codes,
                    title: str,
                    xlabel: str = 'Time (s)',
                    ylabel: str = 'Voltage (V)') -> Tuple:
    """
    Plot ADC input signal and output codes on a shared x-axis.

    Args:
        x: X-axis values (time array, voltage array, etc.)
        input_signal: Analog input signal array
        output_codes: Digital output code array
        title: Plot title prefix for the input signal subplot
        xlabel: X-axis label (default 'Time (s)')
        ylabel: Y-axis label for the input signal (default 'Voltage (V)')

    Returns:
        Tuple of (fig, (ax1, ax2))
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(x, input_signal)
    ax1.set_title(f'Input Signal - {title}')
    ax1.set_ylabel(ylabel)
    ax1.grid(True)

    ax2.plot(x, output_codes, 'r.-', markersize=2)
    ax2.set_title('ADC Output Codes')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Code')
    ax2.grid(True)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_transfer_function(adc,
                           v_min: float,
                           v_max: float,
                           n_points: int = 10000,
                           title: Optional[str] = None) -> Tuple:
    """
    Sweep an ADC with a ramp and plot the transfer function and quantization
    error. Works with any single-ended ADC implementing the ADCBase interface.

    The quantization error is normalized to LSBs and centered around zero:
      - FLOOR mode: bin midpoint (code + 0.5) * LSB is used as the
        reconstructed voltage, giving error in [-0.5, +0.5] LSB.
      - SYMMETRIC mode (and any other): code * LSB is used directly,
        also giving error in [-0.5, +0.5] LSB.

    Args:
        adc: ADC instance with a convert(float) method and n_bits attribute.
             Must be configured for single-ended input.
        v_min: Minimum sweep voltage
        v_max: Maximum sweep voltage
        n_points: Number of ramp points (default 10000)
        title: Plot title. Defaults to the ADC's __repr__.

    Returns:
        Tuple of (fig, (ax1, ax2))
    """
    from pyDataconverter.dataconverter import QuantizationMode

    vin = np.linspace(v_min, v_max, n_points)
    codes = np.array([adc.convert(v) for v in vin])

    # Determine LSB and error formula based on quantization mode
    n_codes = 2 ** adc.n_bits
    v_range = v_max - v_min
    quant_mode = getattr(adc, 'quant_mode', None)

    if quant_mode == QuantizationMode.SYMMETRIC:
        # Code k represents voltage level k * lsb; error is already centered
        lsb = v_range / (n_codes - 1)
        error_lsb = codes - (vin - v_min) / lsb
    else:
        # FLOOR (default): use bin midpoint (code + 0.5) * lsb to center error
        lsb = v_range / n_codes
        error_lsb = (codes + 0.5) - (vin - v_min) / lsb

    plot_title = title if title is not None else repr(adc)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top: transfer function ---
    ax1.plot(vin, codes, color='steelblue', linewidth=1.5)
    ax1.set_ylabel('Output Code')
    ax1.set_title(f'Transfer Function: {plot_title}\nLSB = {lsb * 1000:.4f} mV')
    ax1.grid(True, alpha=0.4)

    # --- Bottom: quantization error ---
    ax2.plot(vin, error_lsb, color='steelblue', linewidth=1.0)
    ax2.axhline( 0.5, color='gray', linewidth=0.7, linestyle=':')
    ax2.axhline( 0.0, color='black', linewidth=0.7)
    ax2.axhline(-0.5, color='gray', linewidth=0.7, linestyle=':')
    ax2.set_xlabel('Input Voltage (V)')
    ax2.set_ylabel('Quantization Error (LSB)')
    ax2.set_ylim(-0.75, 0.75)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig, (ax1, ax2)
