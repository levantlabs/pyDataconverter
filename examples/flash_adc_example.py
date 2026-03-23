"""
FlashADC Usage Examples
=======================

Demonstrates the use of FlashADC with various configurations:
  1. Transfer function — ideal 3-bit, single-ended and differential
  2. Non-idealities — comparator offset spread and resistor mismatch
  3. Encoder comparison — COUNT_ONES vs XOR on a clean signal and with bubbles
  4. Custom reference — ArbitraryReference with non-uniform thresholds
  5. Dynamic metrics — FFT/SNR on a sine wave at -6 dBFS
  6. Architecture visualizer — static snapshot of comparator bank
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.architectures.FlashADC import FlashADC, EncoderType
from pyDataconverter.components.reference import ReferenceLadder, ArbitraryReference
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import (
    generate_coherent_sine,
    generate_coherent_two_tone,
    convert_to_differential,
)
from pyDataconverter.utils.visualizations.fft_plots import plot_fft
from pyDataconverter.utils.visualizations.visualize_FlashADC import visualize_flash_adc
import pyDataconverter.utils.metrics as metrics
import pyDataconverter.utils.fft_analysis as fftan


# ---------------------------------------------------------------------------
# 1. Transfer function — ideal 3-bit
# ---------------------------------------------------------------------------

def test_transfer_function():
    """Ideal 3-bit transfer function: single-ended and differential side by side."""
    print('--- Transfer function (3-bit, ideal) ---')

    n_bits = 3
    v_ref  = 1.0
    vin_se   = np.linspace(0.0, v_ref, 5000)
    vdiff_sw = np.linspace(-v_ref / 2, v_ref / 2, 5000)

    adc_se   = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)
    adc_diff = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.DIFFERENTIAL)

    codes_se   = [adc_se.convert(v) for v in vin_se]
    codes_diff = [adc_diff.convert((0.5 + vd / 2, 0.5 - vd / 2)) for vd in vdiff_sw]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{n_bits}-bit Flash ADC — Ideal Transfer Function', fontweight='bold')

    for ax, x, codes, xlabel, title in [
        (ax1, vin_se,   codes_se,   'Input Voltage (V)',              'Single-Ended'),
        (ax2, vdiff_sw, codes_diff, 'Differential Input Voltage (V)', 'Differential'),
    ]:
        ax.step(x, codes, where='post', color='steelblue', linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Output Code')
        ax.set_title(title)
        ax.set_yticks(range(2 ** n_bits))
        ax.set_ylim(-0.5, 2 ** n_bits - 0.5)
        ax.grid(True, axis='y', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Mark reference voltages
        for vref_tap in adc_se.reference_voltages if ax is ax1 else adc_diff.reference_voltages:
            ax.axvline(vref_tap, color='tomato', linewidth=0.7, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Non-idealities — offset spread and resistor mismatch
# ---------------------------------------------------------------------------

def test_nonidealities():
    """
    2x2 grid showing the effect of comparator offset spread and resistor mismatch
    on both single-ended and differential 3-bit Flash ADCs.
    """
    print('--- Non-ideality comparison (offset spread & resistor mismatch) ---')

    n_bits = 3
    v_ref  = 1.0
    lsb    = v_ref / 2 ** n_bits

    vin_se   = np.linspace(0.0, v_ref, 5000)
    vdiff_sw = np.linspace(-v_ref / 2, v_ref / 2, 5000)
    vcm      = 0.5

    offset_cases = [
        ('Ideal',             0.00,  'black',     '-'),
        ('offset σ = 0.5 LSB', 0.5 * lsb, 'steelblue', '--'),
        ('offset σ = 2 LSB',  2.0 * lsb, 'tomato',    ':'),
    ]
    mismatch_cases = [
        ('Ideal',              0.00, 'black',     '-'),
        ('mismatch = 1 %',     0.01, 'steelblue', '--'),
        ('mismatch = 5 %',     0.05, 'tomato',    ':'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('3-bit Flash ADC Non-Idealities', fontweight='bold')

    def _style(ax, title):
        ax.set_yticks(range(2 ** n_bits))
        ax.set_ylim(-0.5, 2 ** n_bits - 0.5)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    np.random.seed(0)
    for label, offset_std, color, ls in offset_cases:
        adc = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                       offset_std=offset_std)
        codes = [adc.convert(v) for v in vin_se]
        axes[0, 0].step(vin_se, codes, where='post', color=color, linestyle=ls,
                        linewidth=1.6, label=label)
    axes[0, 0].set_xlabel('Input Voltage (V)')
    axes[0, 0].set_ylabel('Output Code')
    _style(axes[0, 0], 'Comparator Offset Spread — Single-Ended')

    np.random.seed(0)
    for label, mismatch, color, ls in mismatch_cases:
        adc = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                       resistor_mismatch=mismatch)
        codes = [adc.convert(v) for v in vin_se]
        axes[0, 1].step(vin_se, codes, where='post', color=color, linestyle=ls,
                        linewidth=1.6, label=label)
    axes[0, 1].set_xlabel('Input Voltage (V)')
    _style(axes[0, 1], 'Resistor Mismatch — Single-Ended')

    np.random.seed(0)
    for label, offset_std, color, ls in offset_cases:
        adc = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.DIFFERENTIAL,
                       offset_std=offset_std)
        codes = [adc.convert((vcm + vd / 2, vcm - vd / 2)) for vd in vdiff_sw]
        axes[1, 0].step(vdiff_sw, codes, where='post', color=color, linestyle=ls,
                        linewidth=1.6, label=label)
    axes[1, 0].set_xlabel('Differential Input Voltage (V)')
    axes[1, 0].set_ylabel('Output Code')
    _style(axes[1, 0], 'Comparator Offset Spread — Differential')

    np.random.seed(0)
    for label, mismatch, color, ls in mismatch_cases:
        adc = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.DIFFERENTIAL,
                       resistor_mismatch=mismatch)
        codes = [adc.convert((vcm + vd / 2, vcm - vd / 2)) for vd in vdiff_sw]
        axes[1, 1].step(vdiff_sw, codes, where='post', color=color, linestyle=ls,
                        linewidth=1.6, label=label)
    axes[1, 1].set_xlabel('Differential Input Voltage (V)')
    _style(axes[1, 1], 'Resistor Mismatch — Differential')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Encoder comparison — COUNT_ONES vs XOR
# ---------------------------------------------------------------------------

def test_encoder_comparison():
    """
    Compare COUNT_ONES and XOR encoders:
      - Left:  transfer functions on an ideal 3-bit ramp (should be identical)
      - Right: thermometer codes with an injected bubble, showing XOR sparkle error
    """
    print('--- Encoder comparison (COUNT_ONES vs XOR) ---')

    n_bits = 3
    v_ref  = 1.0
    vin    = np.linspace(0.0, v_ref, 5000)

    adc_co  = FlashADC(n_bits=n_bits, v_ref=v_ref, encoder_type=EncoderType.COUNT_ONES)
    adc_xor = FlashADC(n_bits=n_bits, v_ref=v_ref, encoder_type=EncoderType.XOR)

    codes_co  = [adc_co.convert(v)  for v in vin]
    codes_xor = [adc_xor.convert(v) for v in vin]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('3-bit Flash ADC — Encoder Comparison', fontweight='bold')

    # Left: overlaid transfer functions (should be identical for clean input)
    ax1.step(vin, codes_co,  where='post', color='steelblue', linewidth=2.0,
             label='COUNT_ONES')
    ax1.step(vin, codes_xor, where='post', color='tomato',    linewidth=1.5,
             linestyle='--', label='XOR')
    ax1.set_xlabel('Input Voltage (V)')
    ax1.set_ylabel('Output Code')
    ax1.set_title('Transfer Functions (clean input — should overlap)')
    ax1.set_yticks(range(2 ** n_bits))
    ax1.set_ylim(-0.5, 2 ** n_bits - 0.5)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: inject bubbles by hand and show the decoded codes for each encoder
    # A valid thermometer code for 7 comparators looks like [1,1,1,0,0,0,0].
    # A bubble inserts a 0 inside the run of 1s: [1,1,0,1,0,0,0].
    n_comp = 2 ** n_bits - 1
    test_codes = []
    labels     = []
    for k in range(n_comp + 1):
        # valid thermometer: k ones then zeros
        therm = np.array([1] * k + [0] * (n_comp - k))
        test_codes.append(therm)
        labels.append(f'k={k} (valid)')

    # bubble at position 2 in k=4 code: [1,1,0,1,0,0,0]
    bubble = np.array([1, 1, 0, 1, 0, 0, 0])
    test_codes.append(bubble)
    labels.append('bubble [1,1,0,1,0,0,0]')

    decoded_co  = [adc_co._encode(t)  for t in test_codes]
    decoded_xor = [adc_xor._encode(t) for t in test_codes]

    x = np.arange(len(test_codes))
    width = 0.35
    ax2.bar(x - width / 2, decoded_co,  width, label='COUNT_ONES', color='steelblue', alpha=0.8)
    ax2.bar(x + width / 2, decoded_xor, width, label='XOR',        color='tomato',    alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Decoded Output Code')
    ax2.set_title('Decoded Codes — Valid + Bubble Thermometer Inputs')
    ax2.set_yticks(range(2 ** n_bits + 1))
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Custom reference — ArbitraryReference with non-uniform thresholds
# ---------------------------------------------------------------------------

def test_custom_reference():
    """
    Inject an ArbitraryReference with non-uniform thresholds (compressed at the centre)
    and compare the resulting transfer function to the ideal uniform ladder.
    """
    print('--- Custom reference (non-uniform ArbitraryReference) ---')

    n_bits = 3
    v_ref  = 1.0
    n_comp = 2 ** n_bits - 1     # 7 taps for 3-bit

    # Uniform (ideal) thresholds
    uniform_thresholds = np.linspace(0.0, v_ref, n_comp + 2)[1:-1]

    # Non-uniform: compressed towards midscale (simulate gain nonlinearity)
    # Use a sine-shaped warp: map [0,1] -> [0,1] with compression at centre
    t = np.linspace(0, 1, n_comp + 2)[1:-1]
    nonuniform_thresholds = v_ref * (t + 0.15 * np.sin(np.pi * t))
    # Ensure strictly increasing (sine warp is monotone for small amplitude)

    ref_uniform    = ArbitraryReference(uniform_thresholds)
    ref_nonuniform = ArbitraryReference(nonuniform_thresholds)

    adc_uniform    = FlashADC(n_bits=n_bits, v_ref=v_ref, reference=ref_uniform)
    adc_nonuniform = FlashADC(n_bits=n_bits, v_ref=v_ref, reference=ref_nonuniform)

    vin = np.linspace(0.0, v_ref, 5000)
    codes_uniform    = [adc_uniform.convert(v)    for v in vin]
    codes_nonuniform = [adc_nonuniform.convert(v) for v in vin]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('3-bit Flash ADC — ArbitraryReference vs Uniform Ladder', fontweight='bold')

    # Left: transfer functions
    ax1.step(vin, codes_uniform,    where='post', color='steelblue', linewidth=2.0,
             label='Uniform (ideal)')
    ax1.step(vin, codes_nonuniform, where='post', color='tomato',    linewidth=1.5,
             linestyle='--', label='Non-uniform (compressed centre)')
    ax1.set_xlabel('Input Voltage (V)')
    ax1.set_ylabel('Output Code')
    ax1.set_title('Transfer Function')
    ax1.set_yticks(range(2 ** n_bits))
    ax1.set_ylim(-0.5, 2 ** n_bits - 0.5)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: reference threshold positions
    ax2.scatter(range(1, n_comp + 1), uniform_thresholds, color='steelblue',
                zorder=3, label='Uniform')
    ax2.scatter(range(1, n_comp + 1), nonuniform_thresholds, color='tomato',
                marker='x', s=80, zorder=3, label='Non-uniform')
    for k, (u, nu) in enumerate(zip(uniform_thresholds, nonuniform_thresholds)):
        ax2.plot([k + 1, k + 1], [u, nu], color='gray', linewidth=0.8, linestyle=':')
    ax2.set_xlabel('Comparator Index')
    ax2.set_ylabel('Threshold Voltage (V)')
    ax2.set_title('Reference Threshold Positions')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Dynamic metrics — sine wave FFT at -6 dBFS
# ---------------------------------------------------------------------------

def test_dynamic_metrics():
    """
    4-bit Flash ADC (single-ended) driven at -6 dBFS.
    Shows:
      - Time-domain conversion
      - FFT with dBFS metrics annotation
      - FFT with plain dB annotation
    """
    print('--- Dynamic metrics (4-bit, single-ended, -6 dBFS sine) ---')

    n_bits = 4
    v_ref  = 1.0
    fs     = 10e6    # 10 MHz sample rate
    NFFT   = 1024
    NFIN   = 11

    full_scale_peak  = v_ref            # single-ended FSR = v_ref
    amplitude_6dbfs  = full_scale_peak / 2  # -6 dBFS (amplitude = FSR/2)
    full_scale_codes = 2 ** n_bits      # single-ended: full scale

    adc = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)

    sine, f_in = generate_coherent_sine(fs, NFFT, NFIN, amplitude=amplitude_6dbfs)
    # Centre around v_ref/2 to stay within [0, v_ref]
    sine_biased = sine + v_ref / 2

    t     = np.arange(NFFT) / fs
    codes = [adc.convert(v) for v in sine_biased]

    # Time-domain plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t * 1e6, sine_biased, color='steelblue', linewidth=1)
    ax1.set_ylabel('Input Voltage (V)')
    ax1.set_title(f'{n_bits}-bit Flash ADC (Single-Ended) — {f_in/1e6:.2f} MHz input at -6 dBFS')
    ax1.grid(True, alpha=0.4)
    ax2.step(t * 1e6, codes, where='post', color='tomato', linewidth=1)
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Output Code')
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # FFT and metrics
    freqs, mags = fftan.compute_fft(codes, fs,
                                    normalization=fftan.FFTNormalization.DBFS,
                                    full_scale=full_scale_codes)
    res = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags, fs=fs, f0=f_in,
                                                full_scale=full_scale_codes)

    plot_fft(freqs, mags, title=f'{n_bits}-bit Flash ADC — FFT (dBFS metrics)', metrics=res)
    plt.show()

    res_db_only = {k: v for k, v in res.items() if not k.endswith('_dBFS')}
    plot_fft(freqs, mags, title=f'{n_bits}-bit Flash ADC — FFT (dB metrics)', metrics=res_db_only)
    plt.show()

    print(f'  SNR  = {res["SNR"]:.1f} dB  ({res["SNR_dBFS"]:.1f} dBFS)')
    print(f'  SNDR = {res["SNDR"]:.1f} dB  ({res["SNDR_dBFS"]:.1f} dBFS)')
    print(f'  SFDR = {res["SFDR"]:.1f} dB  ({res["SFDR_dBFS"]:.1f} dBFS)')
    print(f'  ENOB = {res["ENOB"]:.2f} bits')


# ---------------------------------------------------------------------------
# 6. Comparator noise effect on SNR vs. number of bits
# ---------------------------------------------------------------------------

def test_noise_snr_vs_bits():
    """
    Shows how comparator noise degrades SNR for different ADC resolutions.
    Each panel is a different noise level; curves show SNR vs. n_bits.
    """
    print('--- Comparator noise: SNR vs. resolution ---')

    fs    = 10e6
    NFFT  = 1024
    NFIN  = 11

    bit_range   = [3, 4, 5, 6, 8]
    noise_cases = [
        ('Ideal (no noise)',  0.0,    'black',    '-'),
        ('noise = 0.5 mV',    5e-4,   'steelblue', '--'),
        ('noise = 2 mV',      2e-3,   'tomato',    ':'),
        ('noise = 5 mV',      5e-3,   'seagreen',  '-.'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Flash ADC (Single-Ended): SNR vs. Resolution — Comparator Noise',
                 fontweight='bold')

    for label, noise_rms, color, ls in noise_cases:
        snrs = []
        for n_bits in bit_range:
            v_ref  = 1.0
            amp    = v_ref / 4       # -12 dBFS keeps signal well within range
            fs_c   = 2 ** n_bits     # full-scale codes

            adc    = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                              comparator_params={'noise_rms': noise_rms})
            sine, f_in = generate_coherent_sine(fs, NFFT, NFIN, amplitude=amp)
            sine_biased = sine + v_ref / 2

            codes = [adc.convert(v) for v in sine_biased]
            freqs, mags = fftan.compute_fft(codes, fs,
                                            normalization=fftan.FFTNormalization.DBFS,
                                            full_scale=fs_c)
            res = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags,
                                                        fs=fs, f0=f_in)
            snrs.append(res['SNR'])

        ax.plot(bit_range, snrs, color=color, linestyle=ls, linewidth=1.8,
                marker='o', markersize=5, label=label)

    # Ideal quantization SNR: 6.02*N + 1.76
    ideal_snr = [6.02 * n + 1.76 for n in bit_range]
    ax.plot(bit_range, ideal_snr, 'gray', linewidth=1.0, linestyle=':', label='Ideal (6.02N+1.76)')

    ax.set_xlabel('ADC Resolution (bits)')
    ax.set_ylabel('SNR (dB)')
    ax.set_xticks(bit_range)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7. Architecture visualizer
# ---------------------------------------------------------------------------

def test_visualizer():
    """Static snapshot of the comparator bank at a given input voltage."""
    print('--- Architecture visualizer ---')
    np.random.seed(1)
    adc = FlashADC(
        n_bits=3,
        v_ref=1.0,
        input_type=InputType.SINGLE,
        offset_std=0.01,
    )
    visualize_flash_adc(adc, input_voltage=0.4)
    plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    test_transfer_function()
    test_nonidealities()
    test_encoder_comparison()
    test_custom_reference()
    test_dynamic_metrics()
    test_noise_snr_vs_bits()
    test_visualizer()


if __name__ == '__main__':
    main()
