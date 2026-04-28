"""
signal_gen demo — analog and digital test-signal generators.

Run::

    python examples/signal_gen_demo.py

Exercises the sine, two-tone, IMD, ramp, step, and multitone generators
from ``pyDataconverter.utils.signal_gen``, in both analog (float) and
digital (integer-code) variants.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.utils.signal_gen import (
    convert_to_differential,
    generate_digital_imd_tones,
    generate_digital_multitone,
    generate_digital_ramp,
    generate_digital_sine,
    generate_digital_step,
    generate_digital_two_tone,
    generate_imd_tones,
    generate_sine,
    generate_two_tone,
)


def main() -> None:
    # Generate a sine wave and convert to differential
    sine = generate_sine(frequency=1e3, fs=1e6, amplitude=0.5)
    convert_to_differential(sine)            # vcm = 0
    convert_to_differential(sine, vcm=0.5)   # vcm = 0.5 V

    # Two-tone with phase control
    generate_two_tone(
        f1=1e3,
        f2=1.1e3,
        fs=1e6,
        amplitude1=0.4,
        amplitude2=0.4,
        phase1=0,
        phase2=np.pi / 2,
        duration=0.01,
    )

    # IMD test
    _, imd_frequencies = generate_imd_tones(
        f1=1e6, delta_f=1e3, fs=10e6, amplitude=0.8, order=3,
    )
    print("IMD Test Frequencies:")
    for key, value in imd_frequencies.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, freq in value.items():
                print(f"  {subkey}: {freq / 1e3:.1f} kHz")
        else:
            print(f"{key}: {value / 1e3:.1f} kHz")

    n_bits = 12
    fs = 1e6

    # 1. Digital sine wave
    print("\nTest 1: Digital Sine Wave")
    sine = generate_digital_sine(n_bits=n_bits, frequency=1e3, fs=fs,
                                 amplitude=0.9, duration=0.005)
    t = np.arange(len(sine)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, sine, '.-')
    plt.title('Digital Sine Wave'); plt.xlabel('Time (ms)'); plt.ylabel('Code')
    plt.grid(True); plt.show()

    # 2. Digital two-tone
    print("\nTest 2: Digital Two-Tone Signal")
    signal_2tone = generate_digital_two_tone(
        n_bits=n_bits, f1=1e3, f2=1.1e3, fs=fs,
        amplitude1=0.4, amplitude2=0.4, duration=0.01,
    )
    t = np.arange(len(signal_2tone)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, signal_2tone, '.-')
    plt.title('Digital Two-Tone Signal'); plt.xlabel('Time (ms)'); plt.ylabel('Code')
    plt.grid(True); plt.show()

    # 3. Digital IMD test
    print("\nTest 3: Digital IMD Test")
    _, imd_frequencies = generate_digital_imd_tones(
        n_bits=n_bits, f1=1e6, delta_f=1e3, fs=10e6,
        amplitude=0.8, duration=0.01,
    )
    print("IMD Test Frequencies:")
    for key, value in imd_frequencies.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, freq in value.items():
                print(f"  {subkey}: {freq / 1e3:.1f} kHz")
        else:
            print(f"{key}: {value / 1e3:.1f} kHz")

    # 4. Digital ramp
    print("\nTest 4: Digital Ramp")
    ramp = generate_digital_ramp(n_bits=n_bits, n_points=1000)
    plt.figure(figsize=(10, 4))
    plt.plot(ramp, '.-')
    plt.title('Digital Ramp'); plt.xlabel('Sample'); plt.ylabel('Code')
    plt.grid(True); plt.show()

    # 5. Digital step
    print("\nTest 5: Digital Step")
    step = generate_digital_step(
        n_bits=n_bits, samples=1000,
        step_points=[200, 400, 600, 800],
        levels=[0, 1000, 2000, 3000, 4000],
    )
    plt.figure(figsize=(10, 4))
    plt.plot(step, '.-')
    plt.title('Digital Step'); plt.xlabel('Sample'); plt.ylabel('Code')
    plt.grid(True); plt.show()

    # 6. Digital multitone
    print("\nTest 6: Digital Multitone")
    signal_multi = generate_digital_multitone(
        n_bits=n_bits,
        frequencies=[1e3, 2e3, 3e3, 4e3],
        fs=fs,
        amplitudes=[0.2, 0.2, 0.2, 0.2],
        duration=0.01,
    )
    t = np.arange(len(signal_multi)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, signal_multi, '.-')
    plt.title('Digital Multitone Signal'); plt.xlabel('Time (ms)'); plt.ylabel('Code')
    plt.grid(True); plt.show()


if __name__ == "__main__":
    main()
