"""
SimpleADC Usage Examples
=======================

Demonstrates the use of SimpleADC with various input signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import (
    generate_sine,
    generate_ramp,
    generate_imd_tones,
    convert_to_differential
)


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


def test_sine_wave():
    """Test ADC with sine wave input"""
    # Setup ADC
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    # Generate sine wave
    fs = 1e6  # 1 MHz sampling rate
    f_in = 10e3  # 10 kHz input
    duration = 1e-3  # 1 ms

    # Generate single-ended signal
    sine = generate_sine(
        frequency=f_in,
        sampling_rate=fs,
        amplitude=0.4,
        duration=duration
    )

    # Convert to differential
    v_pos, v_neg = convert_to_differential(sine, vcm=0.5)

    # Convert using ADC
    t = np.arange(0, duration, 1 / fs)
    codes = [adc.convert((vp, vn)) for vp, vn in zip(v_pos, v_neg)]

    # Plot results
    plot_conversion(t, v_pos - v_neg, codes, "Sine Wave")


def test_ramp():
    """Test ADC with ramp input"""
    # Setup ADC
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    # Generate ramp
    samples = 1000
    duration = 1e-3

    # Generate single-ended ramp
    ramp = generate_ramp(samples, -0.5, 0.5)  # -0.5V to +0.5V for differential

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
    f1 = 10e3  # 10 kHz
    delta_f = 1e3  # 1 kHz spacing
    duration = 1e-3

    # Generate signal and get IMD frequencies
    signal, imd_freqs = generate_imd_tones(f1, delta_f, fs, amplitude=0.8, duration=duration)

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


def main():
    # Run all tests
    test_sine_wave()
    test_ramp()
    test_two_tone()


if __name__ == "__main__":
    main()