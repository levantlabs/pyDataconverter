"""
SARADC Usage Examples
=====================

Demonstrates the use of SARADC with various configurations:
  1. Transfer function    — ideal SAR, single-ended and differential
  2. Conversion trace     — cycle-by-cycle binary search visualisation
  3. Cap mismatch         — effect on transfer function and static linearity (DNL/INL)
  4. Non-idealities       — sampling noise, offset, gain error, comparator noise
  5. Dynamic metrics      — FFT / SNR on a coherent sine wave
  6. SNR vs resolution    — comparator noise floor across different bit counts
  7. Architecture visualizer — static snapshot, interactive slider, and animations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.components.cdac import SingleEndedCDAC, DifferentialCDAC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine
from pyDataconverter.utils.visualizations.fft_plots import plot_fft
from pyDataconverter.utils.visualizations.adc_plots import plot_transfer_function
from pyDataconverter.utils.visualizations.visualize_SARADC import (
    visualize_sar_adc,
    animate_sar_conversion,
    animate_sar_adc,
)
import pyDataconverter.utils.metrics as metrics
import pyDataconverter.utils.fft_analysis as fftan


# ---------------------------------------------------------------------------
# 1. Transfer function — ideal SAR, single-ended and differential
# ---------------------------------------------------------------------------

def test_transfer_function():
    """Ideal 4-bit SAR transfer function: single-ended and differential."""
    print('--- Transfer function (4-bit, ideal) ---')

    n_bits = 4
    v_ref  = 1.0
    vin_se   = np.linspace(0.0, v_ref, 5000)
    vdiff_sw = np.linspace(-v_ref / 2, v_ref / 2, 5000)

    adc_se   = SARADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)
    adc_diff = SARADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.DIFFERENTIAL)

    codes_se   = [adc_se.convert(v) for v in vin_se]
    codes_diff = [adc_diff.convert((vd / 2, -vd / 2)) for vd in vdiff_sw]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{n_bits}-bit SAR ADC — Ideal Transfer Function', fontweight='bold')

    for ax, x, codes, xlabel, title, dac_vols in [
        (ax1, vin_se,   codes_se,
         'Input Voltage (V)', 'Single-Ended', adc_se.dac_voltages),
        (ax2, vdiff_sw, codes_diff,
         'Differential Input (V)', 'Differential', adc_diff.dac_voltages),
    ]:
        ax.step(x, codes, where='post', color='steelblue', linewidth=1.5)
        for threshold in dac_vols:
            ax.axvline(threshold, color='tomato', linewidth=0.6,
                       linestyle='--', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Output Code')
        ax.set_title(title)
        ax.set_yticks(range(0, 2 ** n_bits + 1, 2))
        ax.set_ylim(-0.5, 2 ** n_bits - 0.5)
        ax.grid(True, axis='y', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add dashed-line legend entry
    dac_line = mpatches.Patch(facecolor='tomato', alpha=0.5, label='DAC thresholds')
    ax2.legend(handles=[dac_line], fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Conversion trace — cycle-by-cycle binary search
# ---------------------------------------------------------------------------

def test_conversion_trace():
    """
    Visualise the SAR binary search for several representative input voltages.
    Each panel shows one conversion: the held input, trial DAC voltages per bit
    cycle, and how the register converges to the final code.
    """
    print('--- Conversion trace (5-bit, single-ended) ---')

    n_bits = 5
    v_ref  = 1.0
    adc    = SARADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)

    vin_examples = [0.13, 0.37, 0.62, 0.88]
    fig, axes = plt.subplots(1, len(vin_examples), figsize=(15, 5))
    fig.suptitle(f'{n_bits}-bit SAR ADC — Cycle-by-Cycle Conversion Trace',
                 fontweight='bold')

    for ax, vin in zip(axes, vin_examples):
        trace = adc.convert_with_trace(vin)

        cycles      = np.arange(1, n_bits + 1)
        dac_vs      = trace['dac_voltages']
        decisions   = trace['bit_decisions']
        reg_states  = trace['register_states']   # N+1 values

        # DAC trial voltage at each cycle
        kept_v  = [v for v, d in zip(dac_vs, decisions) if d == 1]
        kept_c  = [c for c, d in zip(cycles, decisions) if d == 1]
        clear_v = [v for v, d in zip(dac_vs, decisions) if d == 0]
        clear_c = [c for c, d in zip(cycles, decisions) if d == 0]

        ax.axhline(trace['sampled_voltage'], color='steelblue', linewidth=1.8,
                   linestyle='-', label=f'vin = {vin:.2f} V', zorder=1)

        ax.scatter(kept_c,  kept_v,  color='seagreen', zorder=3, s=70,
                   marker='o', label='bit kept (1)')
        ax.scatter(clear_c, clear_v, color='tomato',   zorder=3, s=70,
                   marker='x', linewidths=2, label='bit cleared (0)')

        # Register DAC output after each cycle (as a step trace)
        reg_v = [adc.cdac.get_voltage(r)[0] for r in reg_states]
        ax.step(np.arange(0, n_bits + 1), reg_v, where='post',
                color='darkorange', linewidth=1.2, linestyle=':', label='register DAC')

        ax.set_xlim(0.4, n_bits + 0.6)
        ax.set_ylim(-0.02, v_ref + 0.02)
        ax.set_xticks(cycles)
        ax.set_xticklabels([f'Bit {n_bits - k}' for k in range(n_bits)], fontsize=8)
        ax.set_xlabel('Bit Cycle (MSB → LSB)')
        ax.set_ylabel('Voltage (V)' if ax is axes[0] else '')
        ax.set_title(f'vin = {vin:.2f} V → code {trace["code"]}')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax is axes[0]:
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Cap mismatch — transfer function and static linearity
# ---------------------------------------------------------------------------

def test_cap_mismatch():
    """
    Show how capacitor mismatch degrades the SAR transfer function and
    produces DNL/INL errors for a single-ended 6-bit SAR ADC.
    """
    print('--- Cap mismatch — transfer function and static linearity (6-bit) ---')

    n_bits = 6
    v_ref  = 1.0
    lsb    = v_ref / 2 ** n_bits

    vin   = np.linspace(0.01, 0.99, 8000)
    n_pts = len(vin)

    mismatch_cases = [
        ('Ideal (no mismatch)', 0.00,  'black',     '-',  0),
        ('mismatch = 0.5 %',   0.005, 'steelblue', '--', 1),
        ('mismatch = 2 %',     0.020, 'tomato',    ':',  2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{n_bits}-bit SAR ADC — Capacitor Mismatch', fontweight='bold')
    ax_tf, ax_dnl, ax_inl = axes

    np.random.seed(0)
    for label, cm, color, ls, _ in mismatch_cases:
        adc   = SARADC(n_bits=n_bits, v_ref=v_ref, cap_mismatch=cm)
        codes = np.array([adc.convert(v) for v in vin])

        # Transfer function
        ax_tf.step(vin, codes, where='post', color=color, linestyle=ls,
                   linewidth=1.4, label=label, alpha=0.9)

        # Static metrics (DNL / INL)
        res = metrics.calculate_adc_static_metrics(vin, codes, n_bits, v_ref)
        dnl = res['DNL']
        inl = res['INL']

        ax_dnl.plot(range(len(dnl)), dnl, color=color, linestyle=ls,
                    linewidth=1.2, label=f'{label}\nMax={res["MaxDNL"]:.2f} LSB')
        ax_inl.plot(range(len(inl)), inl, color=color, linestyle=ls,
                    linewidth=1.2, label=f'{label}\nMax={res["MaxINL"]:.2f} LSB')

    # Style
    ax_tf.set_xlabel('Input Voltage (V)')
    ax_tf.set_ylabel('Output Code')
    ax_tf.set_title('Transfer Function')
    ax_tf.set_yticks(range(0, 2 ** n_bits + 1, 8))
    ax_tf.legend(fontsize=8)
    ax_tf.grid(True, axis='y', alpha=0.3)

    for ax, title, ylabel in [
        (ax_dnl, 'DNL', 'DNL (LSB)'),
        (ax_inl, 'INL', 'INL (LSB)'),
    ]:
        ax.axhline(0,    color='black', linewidth=0.7)
        ax.axhline( 0.5, color='gray', linewidth=0.6, linestyle=':')
        ax.axhline(-0.5, color='gray', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Code')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Non-idealities — sampling noise, offset, gain error, comparator noise
# ---------------------------------------------------------------------------

def test_nonidealities():
    """
    2×2 grid showing the effect of four input-referred non-idealities on the
    transfer function of an 8-bit single-ended SAR ADC.
    Each panel overlays ideal against the non-ideal case.
    """
    print('--- Non-idealities (8-bit, single-ended) ---')

    n_bits = 8
    v_ref  = 1.0
    lsb    = v_ref / 2 ** n_bits
    vin    = np.linspace(0.1, 0.9, 8000)

    cases = [
        # (title, param, values, param_label_unit)
        ('Sampling Noise (kT/C)',
         'noise_rms', [0.0, 2*lsb, 8*lsb],
         'noise_rms'),
        ('Input Offset',
         'offset', [0.0, 4*lsb, 16*lsb],
         'offset'),
        ('Gain Error',
         'gain_error', [0.0, 0.01, 0.04],
         'gain_error'),
        ('Comparator Noise',
         None, [0.0, 2*lsb, 8*lsb],
         'comp noise_rms'),
    ]

    colors    = ['black', 'steelblue', 'tomato']
    linstyles = ['-', '--', ':']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{n_bits}-bit SAR ADC — Non-Ideality Effects', fontweight='bold')

    np.random.seed(42)
    for ax, (title, param, values, label_key) in zip(axes.flat, cases):
        for val, color, ls in zip(values, colors, linstyles):
            if param is None:
                # Comparator noise via comparator_params
                adc = SARADC(n_bits=n_bits, v_ref=v_ref,
                             comparator_params={'noise_rms': val})
                lbl = f'{label_key} = {val*1e3:.1f} mV' if val else 'Ideal'
            else:
                kwargs = {param: val}
                adc = SARADC(n_bits=n_bits, v_ref=v_ref, **kwargs)
                lbl = (f'{label_key} = {val*1e3:.1f} mV' if val else 'Ideal')
                if param == 'gain_error' and val:
                    lbl = f'gain_error = {val*100:.1f} %'

            codes = np.array([adc.convert(v) for v in vin])
            ax.step(vin, codes, where='post', color=color, linestyle=ls,
                    linewidth=1.4, label=lbl, alpha=0.9)

        ax.set_xlabel('Input Voltage (V)')
        ax.set_ylabel('Output Code')
        ax.set_title(title)
        ax.set_yticks(range(0, 2 ** n_bits + 1, 32))
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Dynamic metrics — FFT and SNR on a coherent sine
# ---------------------------------------------------------------------------

def test_dynamic_metrics():
    """
    8-bit SAR ADC (single-ended) driven at full scale.
    Shows time-domain conversion and FFT with dynamic metrics.
    """
    print('--- Dynamic metrics (8-bit, single-ended, full-scale sine) ---')

    n_bits = 8
    v_ref  = 1.0
    fs     = 10e6
    NFFT   = 1024
    NFIN   = 11

    amplitude       = v_ref / 2 * 0.99      # near-full-scale, single-sided
    full_scale_codes = 2 ** n_bits

    adc = SARADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)

    sine, f_in = generate_coherent_sine(fs, NFFT, NFIN, amplitude=amplitude)
    sine_biased = sine + v_ref / 2   # centre within [0, v_ref]
    t    = np.arange(NFFT) / fs
    codes = np.array([adc.convert(v) for v in sine_biased])

    # Time domain
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t * 1e6, sine_biased, color='steelblue', linewidth=1)
    ax1.set_ylabel('Input Voltage (V)')
    ax1.set_title(f'{n_bits}-bit SAR ADC (Single-Ended) — {f_in / 1e6:.2f} MHz input')
    ax1.grid(True, alpha=0.4)
    ax2.step(t * 1e6, codes, where='post', color='tomato', linewidth=0.8)
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Output Code')
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # FFT and metrics
    freqs, mags = fftan.compute_fft(codes, fs,
                                    normalization=fftan.FFTNormalization.DBFS,
                                    full_scale=full_scale_codes)
    res = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags,
                                                fs=fs, f0=f_in,
                                                full_scale=full_scale_codes)

    plot_fft(freqs, mags,
             title=f'{n_bits}-bit SAR ADC — FFT (dBFS)', metrics=res)
    plt.show()

    print(f'  SNR  = {res["SNR"]:.1f} dB  ({res["SNR_dBFS"]:.1f} dBFS)')
    print(f'  SNDR = {res["SNDR"]:.1f} dB  ({res["SNDR_dBFS"]:.1f} dBFS)')
    print(f'  SFDR = {res["SFDR"]:.1f} dB  ({res["SFDR_dBFS"]:.1f} dBFS)')
    print(f'  ENOB = {res["ENOB"]:.2f} bits')
    print(f'  Ideal ENOB = {n_bits:.2f} bits  (6.02N + 1.76 = {6.02*n_bits+1.76:.1f} dB)')


# ---------------------------------------------------------------------------
# 6. SNR vs resolution — comparator noise floor
# ---------------------------------------------------------------------------

def test_noise_snr_vs_bits():
    """
    Shows how comparator noise and capacitor mismatch degrade SNR across
    different SAR ADC resolutions.
    """
    print('--- SNR vs. resolution (comparator noise & cap mismatch) ---')

    fs   = 10e6
    NFFT = 1024
    NFIN = 11

    bit_range = [4, 6, 8, 10, 12]

    noise_cases = [
        ('Ideal',              {},                      'black',     '-'),
        ('comp noise 0.5 mV',  {'noise_rms': 5e-4},    'steelblue', '--'),
        ('comp noise 2 mV',    {'noise_rms': 2e-3},    'tomato',    ':'),
    ]
    mismatch_cases = [
        ('Ideal',              0.000,  'black',     '-'),
        ('cap mismatch 0.1 %', 0.001,  'steelblue', '--'),
        ('cap mismatch 0.5 %', 0.005,  'tomato',    ':'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SAR ADC — SNR vs. Resolution', fontweight='bold')

    # Panel 1: comparator noise
    for label, comp_params, color, ls in noise_cases:
        snrs = []
        for n_bits in bit_range:
            v_ref = 1.0
            amp   = v_ref / 2 * 0.99
            fsc   = 2 ** n_bits
            np.random.seed(0)
            adc   = SARADC(n_bits=n_bits, v_ref=v_ref,
                           comparator_params=comp_params)
            sine, f_in = generate_coherent_sine(fs, NFFT, NFIN, amplitude=amp)
            codes = np.array([adc.convert(v + v_ref / 2) for v in sine])
            freqs, mags = fftan.compute_fft(codes, fs,
                                            normalization=fftan.FFTNormalization.DBFS,
                                            full_scale=fsc)
            res = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags,
                                                        fs=fs, f0=f_in)
            snrs.append(res['SNR'])
        ax1.plot(bit_range, snrs, color=color, linestyle=ls, linewidth=1.8,
                 marker='o', markersize=5, label=label)

    ideal_snr = [6.02 * n + 1.76 for n in bit_range]
    ax1.plot(bit_range, ideal_snr, 'gray', linewidth=1.0, linestyle=':',
             label='Ideal (6.02N+1.76)')
    ax1.set_xlabel('ADC Resolution (bits)')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Comparator Noise')
    ax1.set_xticks(bit_range)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel 2: capacitor mismatch
    for label, cm, color, ls in mismatch_cases:
        snrs = []
        for n_bits in bit_range:
            v_ref = 1.0
            amp   = v_ref / 2 * 0.99
            fsc   = 2 ** n_bits
            np.random.seed(0)
            adc   = SARADC(n_bits=n_bits, v_ref=v_ref, cap_mismatch=cm)
            sine, f_in = generate_coherent_sine(fs, NFFT, NFIN, amplitude=amp)
            codes = np.array([adc.convert(v + v_ref / 2) for v in sine])
            freqs, mags = fftan.compute_fft(codes, fs,
                                            normalization=fftan.FFTNormalization.DBFS,
                                            full_scale=fsc)
            res = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags,
                                                        fs=fs, f0=f_in)
            snrs.append(res['SNR'])
        ax2.plot(bit_range, snrs, color=color, linestyle=ls, linewidth=1.8,
                 marker='o', markersize=5, label=label)

    ax2.plot(bit_range, ideal_snr, 'gray', linewidth=1.0, linestyle=':',
             label='Ideal (6.02N+1.76)')
    ax2.set_xlabel('ADC Resolution (bits)')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('Capacitor Mismatch')
    ax2.set_xticks(bit_range)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7. Architecture visualizer — static snapshot, interactive slider, animations
# ---------------------------------------------------------------------------

def test_visualizer():
    """
    SAR ADC visualizer:
      - Static snapshot at a fixed input voltage (single-ended)
      - Static snapshot for differential mode
      - Interactive slider for real-time exploration
      - Bit-by-bit animation of a single conversion
      - Sweep animation across a sine-wave sequence of inputs
    """
    print('--- Architecture visualizer ---')

    adc_se = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)

    # Static snapshot — single-ended
    visualize_sar_adc(adc_se, input_voltage=0.37)
    plt.show()

    # Static snapshot — differential
    adc_diff = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
    visualize_sar_adc(adc_diff, input_voltage=0.22)
    plt.show()

    # Interactive slider — single-ended (slider controls v_in)
    visualize_sar_adc(adc_se, interactive=True)

    # Bit-by-bit animation of a single conversion
    animate_sar_conversion(adc_se, input_voltage=0.62)

    # Sweep animation across a sine-wave sequence
    t    = np.linspace(0, 2 * np.pi, 40)
    v_in = 0.5 + 0.45 * np.sin(t)
    animate_sar_adc(adc_se, input_voltages=v_in)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    test_transfer_function()
    test_conversion_trace()
    test_cap_mismatch()
    test_nonidealities()
    test_dynamic_metrics()
    test_noise_snr_vs_bits()
    test_visualizer()


if __name__ == '__main__':
    main()
