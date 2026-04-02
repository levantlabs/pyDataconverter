"""
SAR ADC Variants Example
========================

Demonstrates three advanced SAR ADC architectures:

  1. RedundantSARCDAC  — sub-binary radix with digital error correction (DEC)
  2. MultibitSARADC    — resolves multiple bits per cycle (fewer clock cycles)
  3. NoiseshapingSARADC — first-order noise shaping; SNR improves with OSR

Usage
-----
    python examples/sar_variants_example.py
"""

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pyDataconverter.architectures.SARADC import SARADC, MultibitSARADC, NoiseshapingSARADC
from pyDataconverter.components.cdac import RedundantSARCDAC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine
import pyDataconverter.utils.metrics as metrics
import pyDataconverter.utils.fft_analysis as fftan


# ---------------------------------------------------------------------------
# 1. RedundantSARCDAC — digital error correction
# ---------------------------------------------------------------------------

def demo_redundant_sar():
    """
    Build a 6-bit SAR ADC with a sub-binary (radix=1.85) C-DAC.

    The redundancy lets the converter correct a wrong comparator decision on
    any single bit cycle.  A DEC lookup table maps the raw register to the
    correct output code.

    We sweep the input, show the raw and corrected transfer functions, and
    compare to an ideal binary SAR.
    """
    print('--- 1. RedundantSARCDAC (radix=1.85, 6-bit) ---')

    n_bits = 6
    v_ref  = 1.0
    radix  = 1.85

    # Build redundant CDAC and inject it into a standard SARADC
    np.random.seed(42)
    redun_cdac = RedundantSARCDAC(n_bits=n_bits, v_ref=v_ref, radix=radix,
                                   cap_mismatch=0.0)
    adc_redun  = SARADC(n_bits=n_bits, v_ref=v_ref,
                        input_type=InputType.SINGLE, cdac=redun_cdac)
    adc_ideal  = SARADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)

    vin = np.linspace(0.01, 0.99, 4000)

    raw_codes      = []
    corrected_codes = []
    ideal_codes    = []

    for v in vin:
        trace = adc_redun.convert_with_trace(v)
        raw_codes.append(trace['code'])
        corrected_codes.append(redun_cdac.decode(trace['code']))
        ideal_codes.append(adc_ideal.convert(v))

    raw_codes       = np.array(raw_codes)
    corrected_codes = np.array(corrected_codes)
    ideal_codes     = np.array(ideal_codes)

    print(f'  Radix               : {radix}')
    print(f'  Nominal cap weights : {np.round(redun_cdac.cap_weights[:n_bits], 3)}')
    print(f'  Max raw code        : {raw_codes.max()}  (up to {2**n_bits - 1} for sub-binary)')
    print(f'  Max corrected code  : {corrected_codes.max()}  (should be {2**n_bits - 1})')
    print(f'  Max ideal code      : {ideal_codes.max()}')
    print(f'  Codes match after DEC: {np.all(corrected_codes == ideal_codes)}')

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{n_bits}-bit RedundantSARCDAC (radix={radix}) — Digital Error Correction',
                 fontweight='bold')

    axes[0].step(vin, raw_codes,       where='post', color='tomato',   linewidth=1.2)
    axes[0].set_title('Raw register (before DEC)')
    axes[0].set_xlabel('Input Voltage (V)')
    axes[0].set_ylabel('Raw Code')

    axes[1].step(vin, corrected_codes, where='post', color='steelblue', linewidth=1.2)
    axes[1].set_title('Corrected output (after DEC)')
    axes[1].set_xlabel('Input Voltage (V)')
    axes[1].set_ylabel('Output Code')

    axes[2].step(vin, ideal_codes,     where='post', color='black',    linewidth=1.2,
                 label='Ideal binary')
    axes[2].step(vin, corrected_codes, where='post', color='steelblue', linewidth=1.2,
                 linestyle='--', alpha=0.7, label='Redundant + DEC')
    axes[2].set_title('Overlay: ideal vs. DEC-corrected')
    axes[2].set_xlabel('Input Voltage (V)')
    axes[2].set_ylabel('Output Code')
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_yticks(range(0, 2 ** n_bits + 1, 8))
        ax.set_ylim(-1, 2 ** n_bits + 1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('redundant_sar_demo.png', dpi=100, bbox_inches='tight')
    print('  Figure saved: redundant_sar_demo.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. MultibitSARADC — fewer clock cycles per conversion
# ---------------------------------------------------------------------------

def demo_multibit_sar():
    """
    Compare conversion speed (number of clock cycles) between:
      - Standard 1-bit-per-cycle SAR (8 cycles for 8 bits)
      - MultibitSARADC with bits_per_cycle=2 (4 cycles for 8 bits)
      - MultibitSARADC with bits_per_cycle=4 (2 cycles for 8 bits)

    All three produce the same output codes on an ideal input.
    """
    print('\n--- 2. MultibitSARADC (8-bit, bits_per_cycle sweep) ---')

    n_bits = 8
    v_ref  = 1.0
    test_voltages = [0.13, 0.37, 0.62, 0.88]

    configs = [
        ('Standard SAR (1 bit/cycle)',  SARADC(n_bits=n_bits, v_ref=v_ref),        n_bits),
        ('Multibit SAR (2 bits/cycle)', MultibitSARADC(n_bits=n_bits, v_ref=v_ref,
                                                       bits_per_cycle=2),           math.ceil(n_bits / 2)),
        ('Multibit SAR (4 bits/cycle)', MultibitSARADC(n_bits=n_bits, v_ref=v_ref,
                                                       bits_per_cycle=4),           math.ceil(n_bits / 4)),
    ]

    print(f'  {"Architecture":<30} {"Cycles":>7}  Codes for vin = {test_voltages}')
    print(f'  {"-"*30} {"-"*7}  {"-"*40}')

    for label, adc, n_cycles in configs:
        codes = [adc.convert(v) for v in test_voltages]
        print(f'  {label:<30} {n_cycles:>7}  {codes}')

    # Sweep transfer function to confirm correctness
    vin = np.linspace(0.01, 0.99, 2000)
    codes_std  = np.array([SARADC(n_bits=n_bits, v_ref=v_ref).convert(v) for v in vin])
    adc_mb2    = MultibitSARADC(n_bits=n_bits, v_ref=v_ref, bits_per_cycle=2)
    codes_mb2  = np.array([adc_mb2.convert(v) for v in vin])

    match = np.all(codes_std == codes_mb2)
    print(f'  Transfer functions match (std vs 2-bit/cycle): {match}')

    print(f'  Speed-up (2 bits/cycle vs standard): {n_bits / math.ceil(n_bits/2):.1f}x fewer cycles')
    print(f'  Speed-up (4 bits/cycle vs standard): {n_bits / math.ceil(n_bits/4):.1f}x fewer cycles')


# ---------------------------------------------------------------------------
# 3. NoiseshapingSARADC — SNR improvement with oversampling
# ---------------------------------------------------------------------------

def demo_noiseshaping_sar():
    """
    Compare SNR of a standard 8-bit SAR vs. NoiseshapingSARADC at several
    oversampling ratios.

    The noise-shaping ADC accumulates the quantisation residue and feeds it
    back to the next sample, pushing quantisation noise toward high frequencies.
    A digital low-pass filter (decimation) recovers SNR proportional to OSR^3
    (first-order shaping: +9 dB per octave of OSR).
    """
    print('\n--- 3. NoiseshapingSARADC (8-bit, SNR vs OSR) ---')

    n_bits   = 8
    v_ref    = 1.0
    fs_base  = 1e6          # base sampling rate (Nyquist)
    n_fft    = 512
    n_fin    = 13           # coherent frequency bin

    adc_std = SARADC(n_bits=n_bits, v_ref=v_ref)
    adc_ns  = NoiseshapingSARADC(n_bits=n_bits, v_ref=v_ref)

    osr_list = [1, 2, 4, 8, 16]

    print(f'  {"ADC":<30} {"OSR":>5}  {"SNR (dB)":>10}  {"ENOB (bits)":>12}')
    print(f'  {"-"*30} {"-"*5}  {"-"*10}  {"-"*12}')

    # Standard SAR at OSR=1 (reference)
    amp = v_ref / 2 * 0.99
    sine, f_in = generate_coherent_sine(fs_base, n_fft, n_fin, amplitude=amp,
                                         offset=v_ref / 2)
    codes_std = np.array([adc_std.convert(float(v)) for v in sine])
    m_std = metrics.calculate_adc_dynamic_metrics(time_data=codes_std, fs=fs_base)
    print(f'  {"Standard SAR":<30} {"1":>5}  {m_std["SNR"]:>10.2f}  {m_std["ENOB"]:>12.2f}')

    snr_ns_list = []
    for osr in osr_list:
        fs_osr = fs_base * osr
        n_fft_osr = n_fft * osr
        adc_ns.reset()
        sine_osr, _ = generate_coherent_sine(fs_osr, n_fft_osr, n_fin * osr,
                                              amplitude=amp, offset=v_ref / 2)
        codes_osr = np.array([adc_ns.convert(float(v)) for v in sine_osr])

        # Decimate: keep only in-band bins (first n_fft/2 after decimation)
        m_osr = metrics.calculate_adc_dynamic_metrics(time_data=codes_osr, fs=fs_osr)
        snr_ns_list.append(m_osr['SNR'])
        print(f'  {"NoiseshapingSAR":<30} {osr:>5}  {m_osr["SNR"]:>10.2f}  {m_osr["ENOB"]:>12.2f}')

    # Simple plot: SNR vs OSR for noise-shaping SAR
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'{n_bits}-bit SAR Variants — SNR vs Oversampling Ratio',
                 fontweight='bold')

    ax.axhline(m_std['SNR'], color='black', linestyle='--', linewidth=1.5,
               label=f'Standard SAR (OSR=1): {m_std["SNR"]:.1f} dB')
    ax.plot(osr_list, snr_ns_list, color='steelblue', linewidth=2,
            marker='o', markersize=7, label='NoiseshapingSAR')

    ideal_improvement = [m_std['SNR'] + 9 * math.log2(osr) for osr in osr_list]
    ax.plot(osr_list, ideal_improvement, color='tomato', linewidth=1.2,
            linestyle=':', marker='s', markersize=5, label='+9 dB/octave (ideal 1st-order)')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Oversampling Ratio (OSR)')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Noise-Shaping Benefit vs. OSR')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('noiseshaping_sar_demo.png', dpi=100, bbox_inches='tight')
    print('  Figure saved: noiseshaping_sar_demo.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    demo_redundant_sar()
    demo_multibit_sar()
    demo_noiseshaping_sar()
    print('\nDone.')


if __name__ == '__main__':
    main()
