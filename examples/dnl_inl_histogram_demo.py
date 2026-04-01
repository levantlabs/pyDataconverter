"""
Histogram DNL / INL Demo
=========================

Demonstrates the sine-wave histogram method for measuring ADC static
linearity, and compares it to the ramp (transition-voltage) method.

Two figures are produced:

  Figure 1 — Monte Carlo comparison: ramp vs histogram
    For N_MC Flash ADC instances with identical offset_std, both methods
    are applied.  Thin individual traces plus a mean ± 1σ band are shown
    for DNL and INL side-by-side so the agreement (and differences) between
    methods are immediately visible.

  Figure 2 — Histogram convergence with sample count
    A single Flash ADC instance is characterised repeatedly as the number
    of sine-wave samples grows from a few thousand to half a million.
    Shows how the INL estimate stabilises as the histogram fills up.

Usage
-----
    python examples/dnl_inl_histogram_demo.py
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.utils.metrics import (
    calculate_adc_static_metrics,
    calculate_adc_static_metrics_histogram,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_BITS      = 6
V_REF       = 1.0
OFFSET_STD  = 0.12          # comparator offset σ in LSB units
N_MC        = 25            # Monte Carlo runs per method
N_RAMP      = 5_000         # ramp points for transition-voltage method
N_SINE      = 300_000       # sine samples for histogram method
RNG_SEED    = 0

IDEAL_LSB       = V_REF / (2 ** N_BITS)
OFFSET_STD_V    = OFFSET_STD * IDEAL_LSB
AMPLITUDE       = V_REF / 2          # full-scale sine
SINE_OFFSET     = V_REF / 2          # centred in [0, V_REF]

# x-axes
DNL_X = np.arange(2 ** N_BITS)          # one entry per code
INL_X = np.arange(1, 2 ** N_BITS)       # one entry per transition (1-based)

# ---------------------------------------------------------------------------
# Shared plot helper
# ---------------------------------------------------------------------------

TRACE_ALPHA = 0.18
TRACE_COLOR = '#4a90d9'
MEAN_COLOR  = '#1a5fa8'
BAND_COLOR  = '#4a90d9'
BAND_ALPHA  = 0.28


def _plot_mc(ax, x, curves, ylabel, title):
    stack = np.vstack(curves)
    mean  = np.mean(stack, axis=0)
    std   = np.std(stack,  axis=0)

    for c in curves:
        ax.plot(x, c, color=TRACE_COLOR, alpha=TRACE_ALPHA, linewidth=0.6)

    ax.fill_between(x, mean - std, mean + std,
                    color=BAND_COLOR, alpha=BAND_ALPHA, label='mean ± 1σ')
    ax.plot(x, mean, color=MEAN_COLOR, linewidth=1.8, label='mean')
    ax.axhline(0, color='#aaaaaa', linewidth=0.7, linestyle='--')
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Helper: generate sine codes for a given ADC instance
# ---------------------------------------------------------------------------

def _sine_codes(adc, n_samples, rng):
    """Drive ADC with a full-scale sine and return the output code array."""
    phase = rng.uniform(0, 2 * np.pi)
    t     = np.linspace(0, 50 * np.pi, n_samples)
    vin   = SINE_OFFSET + AMPLITUDE * np.sin(t + phase)
    return np.array([adc.convert(float(v)) for v in vin])


# ---------------------------------------------------------------------------
# Figure 1 — Monte Carlo: ramp vs histogram
# ---------------------------------------------------------------------------

def figure_comparison():
    rng = np.random.default_rng(RNG_SEED)
    input_voltages = np.linspace(0.0, V_REF, N_RAMP)

    ramp_dnl, ramp_inl = [], []
    hist_dnl, hist_inl = [], []

    print(f"Running {N_MC} Monte Carlo ADC instances …")
    for i in range(N_MC):
        adc = FlashADC(n_bits=N_BITS, v_ref=V_REF, offset_std=OFFSET_STD_V)

        # Ramp method
        out_codes = np.array([adc.convert(float(v)) for v in input_voltages])
        m_ramp = calculate_adc_static_metrics(
            input_voltages, out_codes, N_BITS, V_REF, inl_method='endpoint')
        ramp_dnl.append(m_ramp['DNL'])
        ramp_inl.append(m_ramp['INL'])

        # Histogram method (same ADC instance)
        sine_out = _sine_codes(adc, N_SINE, rng)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            m_hist = calculate_adc_static_metrics_histogram(
                sine_out, N_BITS, V_REF, AMPLITUDE, inl_method='endpoint')
        hist_dnl.append(m_hist['DNL'])
        hist_inl.append(m_hist['INL'])

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{N_MC}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(
        f'Flash ADC DNL / INL  —  {N_BITS}-bit, offset σ = {OFFSET_STD:.2f} LSB, '
        f'{N_MC} Monte Carlo runs\n'
        f'Left: ramp method ({N_RAMP} pts)   '
        f'Right: histogram method ({N_SINE//1000}k sine samples)',
        fontsize=10, fontweight='bold'
    )

    _plot_mc(axes[0, 0], DNL_X, ramp_dnl, 'DNL (LSB)', 'DNL — ramp')
    axes[0, 0].set_xlabel('Code', fontsize=8)

    _plot_mc(axes[0, 1], DNL_X, hist_dnl, 'DNL (LSB)', 'DNL — histogram')
    axes[0, 1].set_xlabel('Code', fontsize=8)

    _plot_mc(axes[1, 0], INL_X, ramp_inl, 'INL (LSB)', 'INL — ramp (endpoint)')
    axes[1, 0].set_xlabel('Transition', fontsize=8)

    _plot_mc(axes[1, 1], INL_X, hist_inl, 'INL (LSB)', 'INL — histogram (endpoint)')
    axes[1, 1].set_xlabel('Transition', fontsize=8)

    # Match y-limits between ramp and histogram panels for easy comparison
    for row in range(2):
        ylims = [axes[row, col].get_ylim() for col in range(2)]
        combined = (min(y[0] for y in ylims), max(y[1] for y in ylims))
        for col in range(2):
            axes[row, col].set_ylim(combined)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Convergence of histogram INL with sample count
# ---------------------------------------------------------------------------

def figure_convergence():
    rng = np.random.default_rng(RNG_SEED + 1)
    sample_counts = [5_000, 20_000, 100_000, 500_000]

    # Use a single ADC with meaningful non-idealities
    adc = FlashADC(n_bits=N_BITS, v_ref=V_REF, offset_std=OFFSET_STD_V)

    # Ground-truth INL from a very dense ramp
    input_voltages = np.linspace(0.0, V_REF, 20_000)
    out_codes = np.array([adc.convert(float(v)) for v in input_voltages])
    m_ref = calculate_adc_static_metrics(
        input_voltages, out_codes, N_BITS, V_REF, inl_method='endpoint')
    ref_inl = m_ref['INL']

    fig, axes = plt.subplots(1, len(sample_counts), figsize=(14, 4), sharey=True)
    fig.suptitle(
        f'Histogram INL convergence  —  {N_BITS}-bit Flash ADC, '
        f'offset σ = {OFFSET_STD:.2f} LSB\n'
        'Grey dashed: ramp ground truth   Blue: histogram estimate',
        fontsize=10, fontweight='bold'
    )

    print('\nRunning convergence sweeps …')
    for ax, n in zip(axes, sample_counts):
        codes_subset = _sine_codes(adc, n, rng)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            m = calculate_adc_static_metrics_histogram(
                codes_subset, N_BITS, V_REF, AMPLITUDE, inl_method='endpoint')

        ax.plot(INL_X, ref_inl, color='#999999', linewidth=1.2,
                linestyle='--', label='ramp ref')
        ax.plot(INL_X, m['INL'], color=MEAN_COLOR, linewidth=1.4,
                label='histogram')
        ax.axhline(0, color='#cccccc', linewidth=0.6, linestyle=':')

        label = f'{n//1000}k' if n >= 1000 else str(n)
        ax.set_title(f'N = {label} samples', fontsize=9)
        ax.set_xlabel('Transition', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        print(f'  N={n:>7,}  MaxINL={m["MaxINL"]:.3f} LSB')

    axes[0].set_ylabel('INL (LSB)', fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    fig1 = figure_comparison()
    fig2 = figure_convergence()
    plt.show()
