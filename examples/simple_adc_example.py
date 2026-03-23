"""
SimpleADC Usage Examples
=======================

Demonstrates the use of SimpleADC with various input signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType, QuantizationMode
from pyDataconverter.utils.signal_gen import (
    generate_coherent_sine,
    generate_coherent_two_tone,
    generate_ramp,
    generate_imd_tones,
    convert_to_differential
)
from pyDataconverter.utils.visualizations.adc_plots import plot_conversion, plot_transfer_function
from pyDataconverter.utils.visualizations.fft_plots import plot_fft
import pyDataconverter.utils.metrics as metrics
import pyDataconverter.utils.fft_analysis as fftan



def test_sine_wave():
    """Test ADC with sine wave input at -6 dBFS to demonstrate dB vs dBFS annotation."""
    print('--- Running sine wave input example (-6 dBFS) ---')
    # Setup ADC
    fsr = 1
    adc = SimpleADC(n_bits=12, v_ref=fsr, input_type=InputType.DIFFERENTIAL)

    # -6 dBFS: amplitude = full-scale-peak / 2  (20*log10(0.5) = -6.02 dBFS)
    full_scale_peak = fsr / 2   # differential full-scale peak is v_ref/2
    amplitude_6dbfs = full_scale_peak / 2  # -6 dBFS

    # Generate coherent sine wave (exact FFT bin frequency eliminates leakage)
    fs = 1e6   # 1 MHz sampling rate
    NFIN = 11
    NFFT = 1024
    sine, f_in = generate_coherent_sine(fs, NFFT, NFIN, amplitude=amplitude_6dbfs)

    # Convert to differential
    v_pos, v_neg = convert_to_differential(sine, vcm=0.5)

    # Convert using ADC
    t = np.arange(len(sine)) / fs
    codes = [adc.convert((vp, vn)) for vp, vn in zip(v_pos, v_neg)]

    # Plot time-domain result
    plot_conversion(t, v_pos - v_neg, codes, "Sine Wave (-6 dBFS)")
    plt.show()

    # Compute FFT (dBFS normalised)
    full_scale_codes = 2**adc.n_bits / 2
    freqs, mags = fftan.compute_fft(codes, fs, normalization=fftan.FFTNormalization.DBFS,
                                    full_scale=full_scale_codes)

    # Calculate metrics (with full_scale so dBFS variant keys are populated)
    results = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags, fs=fs, f0=f_in,
                                                    full_scale=full_scale_codes)

    # Plot 1: with dBFS metrics annotation (auto-detected from dict keys)
    plot_fft(freqs, mags, title="FFT Spectrum (-6 dBFS) — metrics in dBFS", metrics=results)
    plt.show()

    # Plot 2: suppress dBFS keys so annotation falls back to plain dB
    results_db_only = {k: v for k, v in results.items() if not k.endswith('_dBFS')}
    plot_fft(freqs, mags, title="FFT Spectrum (-6 dBFS) — metrics in dB", metrics=results_db_only)
    plt.show()

    print('Sine wave test results:')
    for k in results.keys():
        print('--- {}: {}'.format(k, results[k]))




def test_ramp():
    """Test ADC with ramp input, plotted as output code vs input voltage"""
    # Setup ADC
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    # Generate single-ended ramp from -0.6V to +0.6V (within differential range)
    samples = 1000
    ramp = generate_ramp(samples, -0.6, 0.6)

    # Convert to differential
    v_pos, v_neg = convert_to_differential(ramp, vcm=0.5)
    vdiff = v_pos - v_neg

    # Convert using ADC
    codes = [adc.convert((vp, vn)) for vp, vn in zip(v_pos, v_neg)]

    # Plot output code vs input voltage (transfer function view)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vdiff, codes, 'r.-', markersize=2)
    ax.set_xlabel('Input Voltage (V)')
    ax.set_ylabel('Output Code')
    ax.set_title('ADC Ramp Test — Output Code vs Input Voltage')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def test_two_tone():
    """Test ADC with two-tone input"""
    # Setup ADC
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    # Generate coherent two-tone signal
    fs = 1e6  # 1 MHz sampling rate
    NFFT = 1024
    signal, f1, f2 = generate_coherent_two_tone(fs, NFFT, n_fin1=11, n_fin2=13,
                                                 amplitude1=0.25, amplitude2=0.25)
    delta_f = f2 - f1

    # Compute expected IMD frequencies
    imd_freqs = {
        'f1': f1, 'f2': f2,
        'imd2': {'f1+f2': f1+f2, 'f2-f1': delta_f},
        'imd3': {'2f1-f2': 2*f1-f2, '2f2-f1': 2*f2-f1}
    }

    # Convert to differential
    v_pos, v_neg = convert_to_differential(signal, vcm=0.5)

    # Convert using ADC
    t = np.arange(len(signal)) / fs
    codes = [adc.convert((vp, vn)) for vp, vn in zip(v_pos, v_neg)]

    # Plot results
    plot_conversion(t, v_pos - v_neg, codes, "Two-Tone")
    plt.show()

    # Print expected IMD frequencies
    print("\nIMD Test Frequencies:")
    for key, value in imd_freqs.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, freq in value.items():
                print(f"  {subkey}: {freq / 1e3:.1f} kHz")
        else:
            print(f"{key}: {value / 1e3:.1f} kHz")

    # Compute FFT and metrics, then plot
    freqs, mags = fftan.compute_fft(codes, fs, normalization=fftan.FFTNormalization.DBFS,
                                    full_scale=2**adc.n_bits/2)
    results = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags, fs=fs, f0=f1,
                                                    full_scale=2**adc.n_bits/2)
    plot_fft(freqs, mags, title="Two-Tone FFT", metrics=results)
    plt.show()


def test_transfer_function_3bit():
    """
    Compare FLOOR and SYMMETRIC quantization modes on a 3-bit ADC using
    plot_transfer_function from adc_plots, with both modes overlaid for
    easy visual comparison of transition points and error shape.
    """
    print('--- 3-bit ADC transfer function comparison (FLOOR vs SYMMETRIC) ---')

    n_bits = 3
    v_ref = 1.0
    n_codes = 2 ** n_bits

    adc_floor = SimpleADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                          quant_mode=QuantizationMode.FLOOR)
    adc_sym   = SimpleADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                          quant_mode=QuantizationMode.SYMMETRIC)

    # Dense ramp across full input range
    vin = np.linspace(0, v_ref, 10000)
    codes_floor = np.array([adc_floor.convert(v) for v in vin])
    codes_sym   = np.array([adc_sym.convert(v) for v in vin])

    lsb_floor = v_ref / n_codes
    lsb_sym   = v_ref / (n_codes - 1)
    transitions_floor = [k * lsb_floor for k in range(1, n_codes)]
    transitions_sym   = [(k - 0.5) * lsb_sym for k in range(1, n_codes)]
    error_floor_lsb = (codes_floor + 0.5) - vin / lsb_floor
    error_sym_lsb   =  codes_sym          - vin / lsb_sym

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top: overlaid transfer functions ---
    ax1.plot(vin, codes_floor, color='steelblue', linewidth=2, label='FLOOR')
    ax1.plot(vin, codes_sym,   color='tomato',    linewidth=2, linestyle='--',
             label='SYMMETRIC')
    for t in transitions_floor:
        ax1.axvline(t, color='steelblue', linewidth=0.7, linestyle=':')
    for t in transitions_sym:
        ax1.axvline(t, color='tomato',    linewidth=0.7, linestyle=':')
    ax1.set_ylabel('Output Code')
    ax1.set_title(f'{n_bits}-bit ADC: FLOOR vs SYMMETRIC\n'
                  f'FLOOR LSB = {lsb_floor*1000:.1f} mV  |  '
                  f'SYMMETRIC LSB = {lsb_sym*1000:.2f} mV')
    ax1.set_yticks(range(n_codes))
    ax1.set_ylim(-0.5, n_codes - 0.5)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.4)

    # --- Bottom: overlaid quantization errors ---
    ax2.plot(vin, error_floor_lsb, color='steelblue', linewidth=1.5, label='FLOOR')
    ax2.plot(vin, error_sym_lsb,   color='tomato',    linewidth=1.5, linestyle='--',
             label='SYMMETRIC')
    ax2.axhline( 0.5, color='gray',  linewidth=0.7, linestyle=':')
    ax2.axhline( 0.0, color='black', linewidth=0.7)
    ax2.axhline(-0.5, color='gray',  linewidth=0.7, linestyle=':')
    ax2.set_xlabel('Input Voltage (V)')
    ax2.set_ylabel('Quantization Error (LSB)')
    ax2.set_xlim(0, v_ref)
    ax2.set_ylim(-0.75, 0.75)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # Also show each mode individually using the shared utility
    plot_transfer_function(adc_floor, v_min=0, v_max=v_ref,
                           title=f'{n_bits}-bit FLOOR')
    plt.show()
    plot_transfer_function(adc_sym, v_min=0, v_max=v_ref,
                           title=f'{n_bits}-bit SYMMETRIC')
    plt.show()


def main():
    # Run all tests
    test_transfer_function_3bit()
    test_sine_wave()
    test_ramp()
    test_two_tone()


if __name__ == "__main__":
    main()