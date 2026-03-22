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
    generate_sine,
    generate_ramp,
    generate_imd_tones,
    convert_to_differential
)
import pyDataconverter.utils.metrics as metrics
import pyDataconverter.utils.fft_analysis as fftan


def plot_conversion(time, input_signal, output_codes, title, ylabel="Voltage (V)"):
    """Helper function to plot input and output"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time, input_signal)
    ax1.set_title(f'Input Signal - {title}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(ylabel)
    ax1.grid(True)

    ax2.plot(time, output_codes, 'r.-')
    ax2.set_title('ADC Output Codes')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Code')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_frequencydomain(freqs, mags, title, xlabel='Frequency', ylabel='Power'):
    """Helper function to plot input and output"""
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    ax1.plot(freqs, mags)
    ax1.set_title(f'Input Signal - {title}')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(True)

    plt.tight_layout()

    return fig, ax1



def test_sine_wave():
    """Test ADC with sine wave input"""
    print('--- Running sine wave input example ---')
    # Setup ADC
    fsr = 1
    adc = SimpleADC(n_bits=12, v_ref=fsr, input_type=InputType.DIFFERENTIAL)

    # Generate sine wave
    fs = 1e6  # 1 MHz sampling rate
    NFIN = 11
    NFFT = 1024
    f_in = NFIN/NFFT*fs # Coherent input signal
    duration = NFFT / fs # Take full sine wave cycle

    # Generate single-ended signal
    sine = generate_sine(
        frequency=f_in,
        sampling_rate=fs,
        amplitude=fsr/2,
        duration=duration
    )

    # Convert to differential
    v_pos, v_neg = convert_to_differential(sine, vcm=0.5)

    # Convert using ADC
    t = np.arange(0, duration, 1 / fs)
    codes = [adc.convert((vp, vn)) for vp, vn in zip(v_pos, v_neg)]

    # Plot results
    plot_conversion(t, v_pos - v_neg, codes, "Sine Wave")

    #Now, analyze performance.  Calculate FFT and plot
    freqs, mags = fftan.compute_fft(codes, fs, normalization=fftan.FFTNormalization.DBFS, full_scale=2**adc.n_bits/2) # use default window. Normalize to dBFS
    fftan.plot_fft(freqs, mags)
    plt.show()

    #Calculate metrics and print
    results = metrics.calculate_adc_dynamic_metrics(freqs=freqs, mags=mags, fs=fs, f0=f_in)
    print('Sine wave test results:')
    for k in results.keys():
        print('--- {}: {}'.format(k, results[k]))




def test_ramp():
    """Test ADC with ramp input"""
    # Setup ADC
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    # Generate ramp
    samples = 1000
    duration = 1e-3

    # Generate single-ended ramp
    ramp = generate_ramp(samples, -0.6, 0.6)  # -0.6V to +0.6V for differential

    # Convert to differential
    v_pos, v_neg = convert_to_differential(ramp, vcm=0.5)

    # Convert using ADC
    t = np.linspace(0, duration, samples)
    codes = [adc.convert((vp, vn)) for vp, vn in zip(v_pos, v_neg)]

    # Plot results
    plot_conversion(t, v_pos - v_neg, codes, "Ramp")


def test_two_tone():
    """Test ADC with two-tone input"""
    # Setup ADC
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    # Generate two-tone signal
    fs = 1e6  # 1 MHz sampling rate
    NFIN1 = 11
    NFIN2 = 13
    NFFT = 1024
    f1 = NFIN1/NFFT * fs  # Coherent tone
    delta_f = (NFIN2-NFIN1)/NFFT * fs # Tone spacing
    duration = NFFT / fs

    # Generate signal and get IMD frequencies
    signal, imd_freqs = generate_imd_tones(f1, delta_f, fs, amplitude=0.5, duration=duration)

    # Convert to differential
    v_pos, v_neg = convert_to_differential(signal, vcm=0.5)

    # Convert using ADC
    t = np.arange(0, duration, 1 / fs)
    codes = [adc.convert((vp, vn)) for vp, vn in zip(v_pos, v_neg)]

    # Plot results
    plot_conversion(t, v_pos - v_neg, codes, "Two-Tone")

    # Print expected IMD frequencies
    print("\nIMD Test Frequencies:")
    for key, value in imd_freqs.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, freq in value.items():
                print(f"  {subkey}: {freq / 1e3:.1f} kHz")
        else:
            print(f"{key}: {value / 1e3:.1f} kHz")


def test_transfer_function_3bit():
    """
    Compare FLOOR and SYMMETRIC quantization modes on a 3-bit ADC.

    Sweeps a ramp from 0 to v_ref and overlays the transfer functions for
    both modes, making it easy to see where the transition points differ —
    particularly the half-LSB shift at the first and last code boundaries
    in SYMMETRIC mode.
    """
    print('--- 3-bit ADC transfer function comparison (FLOOR vs SYMMETRIC) ---')

    n_bits = 3
    v_ref = 1.0
    n_codes = 2 ** n_bits  # 8 codes: 0 to 7

    adc_floor = SimpleADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                          quant_mode=QuantizationMode.FLOOR)
    adc_sym   = SimpleADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                          quant_mode=QuantizationMode.SYMMETRIC)

    # Dense ramp across full input range
    vin = np.linspace(0, v_ref, 10000)
    codes_floor = [adc_floor.convert(v) for v in vin]
    codes_sym   = [adc_sym.convert(v) for v in vin]

    # Compute expected transition points for annotation
    lsb_floor = v_ref / n_codes                   # v_ref / 2^N
    lsb_sym   = v_ref / (n_codes - 1)             # v_ref / (2^N - 1)
    transitions_floor = [k * lsb_floor for k in range(1, n_codes)]
    transitions_sym   = [(k - 0.5) * lsb_sym for k in range(1, n_codes)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(vin, codes_floor, color='steelblue', linewidth=2, label='FLOOR')
    ax.plot(vin, codes_sym,   color='tomato',    linewidth=2, label='SYMMETRIC',
            linestyle='--')

    # Mark transition points on the x-axis
    for t in transitions_floor:
        ax.axvline(t, color='steelblue', linewidth=0.7, linestyle=':')
    for t in transitions_sym:
        ax.axvline(t, color='tomato', linewidth=0.7, linestyle=':')

    ax.set_xlabel('Input Voltage (V)')
    ax.set_ylabel('Output Code')
    ax.set_title(f'{n_bits}-bit ADC Transfer Function: FLOOR vs SYMMETRIC\n'
                 f'FLOOR LSB = {lsb_floor*1000:.1f} mV  |  '
                 f'SYMMETRIC LSB = {lsb_sym*1000:.2f} mV')
    ax.set_yticks(range(n_codes))
    ax.set_xlim(0, v_ref)
    ax.set_ylim(-0.5, n_codes - 0.5)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    plt.show()


def main():
    # Run all tests
    test_transfer_function_3bit()
    test_sine_wave()
    test_ramp()
    test_two_tone()


if __name__ == "__main__":
    main()