"""
DNL / INL Monte Carlo Demo
===========================

Shows how DNL and INL vary with non-idealities for two converter families:

  1. Flash ADC — comparator offset mismatch (offset_std sweep)
  2. CurrentSteeringDAC — binary, segmented, and thermometer topologies
     with current-source mismatch

Each subplot overlays N_MC individual Monte Carlo traces (thin, transparent)
plus the mean ± 1-sigma band (thick line with shaded region).

Usage
-----
    python examples/dnl_inl_demo.py

The script produces two figures and keeps them open until the window is closed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.CurrentSteeringDAC import CurrentSteeringDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.metrics import (
    calculate_adc_static_metrics,
    calculate_dac_static_metrics,
)

# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

TRACE_ALPHA  = 0.18
TRACE_COLOR  = '#4a90d9'
MEAN_COLOR   = '#1a5fa8'
BAND_COLOR   = '#4a90d9'
BAND_ALPHA   = 0.30
ZERO_COLOR   = '#999999'


def _plot_mc(ax, x, curves, ylabel, title, zero_line=True):
    """
    Overlay individual traces plus mean ± std band on *ax*.

    Parameters
    ----------
    x      : 1-D array, common x-axis values
    curves : list of 1-D arrays (one per MC run)
    ylabel : y-axis label string
    title  : subplot title string
    zero_line : draw a horizontal dashed line at y=0
    """
    stack = np.vstack(curves)          # (N_MC, len_x)
    mean  = np.mean(stack, axis=0)
    std   = np.std(stack,  axis=0)

    for curve in curves:
        ax.plot(x, curve, color=TRACE_COLOR, alpha=TRACE_ALPHA, linewidth=0.6)

    ax.fill_between(x, mean - std, mean + std,
                    color=BAND_COLOR, alpha=BAND_ALPHA, label='mean ± 1σ')
    ax.plot(x, mean, color=MEAN_COLOR, linewidth=1.8, label='mean')

    if zero_line:
        ax.axhline(0, color=ZERO_COLOR, linewidth=0.7, linestyle='--')

    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Figure 1 — Flash ADC
# ---------------------------------------------------------------------------

def figure_flash_adc(n_bits=6, v_ref=1.0, n_ramp=5000, n_mc=30):
    """
    Flash ADC DNL/INL vs comparator offset_std.

    Two columns: low vs high offset standard deviation (in LSB units).
    Two rows:    DNL (length 2^N) and INL (length 2^N-1, endpoint method).
    """
    ideal_lsb = v_ref / (2 ** n_bits)
    offset_std_lsbs = [0.10, 0.30]          # offset σ expressed in LSB units

    input_voltages = np.linspace(0.0, v_ref, n_ramp)
    dnl_x = np.arange(2 ** n_bits)          # one bin per code
    inl_x = np.arange(1, 2 ** n_bits)       # transition index (1-based for readability)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        f'Flash ADC DNL / INL  —  {n_bits}-bit, {n_mc} Monte Carlo runs',
        fontsize=11, fontweight='bold'
    )

    for col, sigma_lsb in enumerate(offset_std_lsbs):
        sigma_v = sigma_lsb * ideal_lsb

        dnl_curves = []
        inl_curves = []

        for _ in range(n_mc):
            adc = FlashADC(n_bits=n_bits, v_ref=v_ref, offset_std=sigma_v)
            output_codes = np.array([adc.convert(v) for v in input_voltages])
            m = calculate_adc_static_metrics(
                input_voltages, output_codes, n_bits, v_ref,
                inl_method='endpoint'
            )
            dnl_curves.append(m['DNL'])
            inl_curves.append(m['INL'])

        _plot_mc(axes[0, col], dnl_x, dnl_curves,
                 ylabel='DNL (LSB)',
                 title=f'DNL  |  offset σ = {sigma_lsb:.2f} LSB')
        axes[0, col].set_xlabel('Code', fontsize=8)

        _plot_mc(axes[1, col], inl_x, inl_curves,
                 ylabel='INL (LSB)',
                 title=f'INL (endpoint)  |  offset σ = {sigma_lsb:.2f} LSB')
        axes[1, col].set_xlabel('Transition', fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — CurrentSteeringDAC (binary / segmented / thermometer)
# ---------------------------------------------------------------------------

def figure_current_steering_dac(n_bits=8, current_mismatch=0.02, n_mc=30):
    """
    CurrentSteeringDAC DNL/INL for three segmentation topologies.

    Three columns: binary (0T), segmented (4T+4B), thermometer (8T).
    Two rows:      DNL (length 2^N-1) and INL (length 2^N, endpoint method).

    With binary weighting the MSB source carries 2^(N-1) times the unit
    current; a 2 % mismatch there produces a DNL spike of ~2.56 LSB at the
    mid-scale transition (code 127→128).  Thermometer encoding distributes
    errors uniformly so DNL stays near ±mismatch per step.  Segmented is
    intermediate.

    Parameters
    ----------
    n_bits           : DAC resolution (8-bit by default for clear signatures).
    current_mismatch : σ of multiplicative per-source current mismatch.
    n_mc             : Number of Monte Carlo runs per topology.
    """
    # Choose i_unit and r_load so that v_ref = (2^N-1)*i_unit*r_load = 1 V
    r_load = 1000.0
    i_unit = 1.0 / ((2 ** n_bits - 1) * r_load)
    v_ref  = 1.0

    configs = [
        (0,      f'Binary (0T + {n_bits}B)'),
        (n_bits // 2, f'Segmented ({n_bits//2}T + {n_bits//2}B)'),
        (n_bits, f'Thermometer ({n_bits}T)'),
    ]

    n_codes = 2 ** n_bits
    dnl_x = np.arange(1, n_codes)           # steps between consecutive codes
    inl_x = np.arange(n_codes)              # one entry per code

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(
        f'CurrentSteeringDAC DNL / INL  —  {n_bits}-bit, '
        f'mismatch σ = {current_mismatch*100:.0f}%,  {n_mc} Monte Carlo runs',
        fontsize=11, fontweight='bold'
    )

    for col, (n_therm, label) in enumerate(configs):
        dnl_curves = []
        inl_curves = []

        for _ in range(n_mc):
            dac = CurrentSteeringDAC(
                n_bits=n_bits,
                v_ref=v_ref,
                n_therm_bits=n_therm,
                i_unit=i_unit,
                r_load=r_load,
                current_mismatch=current_mismatch,
                output_type=OutputType.SINGLE,
            )
            m = calculate_dac_static_metrics(dac, inl_method='endpoint')
            dnl_curves.append(m['DNL'])
            inl_curves.append(m['INL'])

        _plot_mc(axes[0, col], dnl_x, dnl_curves,
                 ylabel='DNL (LSB)',
                 title=f'DNL  |  {label}')
        axes[0, col].set_xlabel('Code step', fontsize=8)

        _plot_mc(axes[1, col], inl_x, inl_curves,
                 ylabel='INL (LSB)',
                 title=f'INL (endpoint)  |  {label}')
        axes[1, col].set_xlabel('Code', fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Running Flash ADC Monte Carlo (this may take a few seconds) ...')
    fig1 = figure_flash_adc(n_bits=6, v_ref=1.0, n_ramp=5000, n_mc=30)

    print('Running CurrentSteeringDAC Monte Carlo ...')
    fig2 = figure_current_steering_dac(n_bits=8, current_mismatch=0.02, n_mc=30)

    plt.show()
