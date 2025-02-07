"""
Signal Generation Utilities
==========================

Provides functions to generate various test signals for ADC testing.
"""

import numpy as np
from typing import Union, Tuple, List


def convert_to_differential(signal: np.ndarray, vcm: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert single-ended signal to differential with specified common mode.

    Args:
        signal: Input signal array
        vcm: Common mode voltage (default 0.0V)

    Returns:
        Tuple of (v_pos, v_neg) arrays
    """
    v_pos = vcm + signal / 2
    v_neg = vcm - signal / 2
    return v_pos, v_neg


def generate_sine(frequency: float,
                  sampling_rate: float,
                  amplitude: float = 1.0,
                  offset: float = 0.0,
                  duration: float = 1.0) -> np.ndarray:
    """
    Generate a sinusoidal test signal.

    Args:
        frequency: Signal frequency in Hz
        sampling_rate: Sampling rate in Hz
        amplitude: Peak amplitude in volts
        offset: DC offset in volts
        duration: Signal duration in seconds

    Returns:
        Signal array
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t) + offset


def generate_ramp(samples: int,
                  v_min: float = 0.0,
                  v_max: float = 1.0) -> np.ndarray:
    """
    Generate a ramp signal.

    Args:
        samples: Number of samples
        v_min: Minimum voltage
        v_max: Maximum voltage

    Returns:
        Signal array
    """
    return np.linspace(v_min, v_max, samples)


def generate_step(samples: int,
                  step_points: List[int],
                  levels: List[float]) -> np.ndarray:
    """
    Generate a multi-level step signal.

    Args:
        samples: Total number of samples
        step_points: List of points where steps occur
        levels: Voltage levels for each step

    Returns:
        Signal array
    """
    signal = np.zeros(samples)
    current_level = levels[0]

    for point, level in zip(step_points, levels[1:]):
        signal[point:] = level
        current_level = level

    return signal


def generate_two_tone(f1: float,
                      f2: float,
                      sampling_rate: float,
                      amplitude1: float = 0.5,
                      amplitude2: float = 0.5,
                      phase1: float = 0.0,
                      phase2: float = 0.0,
                      duration: float = 1.0) -> np.ndarray:
    """
    Generate a two-tone test signal.

    Args:
        f1: First tone frequency in Hz
        f2: Second tone frequency in Hz
        sampling_rate: Sampling rate in Hz
        amplitude1: First tone amplitude in volts
        amplitude2: Second tone amplitude in volts
        phase1: First tone phase in radians
        phase2: Second tone phase in radians
        duration: Signal duration in seconds

    Returns:
        Signal array
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    tone1 = amplitude1 * np.sin(2 * np.pi * f1 * t + phase1)
    tone2 = amplitude2 * np.sin(2 * np.pi * f2 * t + phase2)
    return tone1 + tone2


def generate_multitone(frequencies: List[float],
                       sampling_rate: float,
                       amplitudes: List[float] = None,
                       phases: List[float] = None,
                       duration: float = 1.0) -> np.ndarray:
    """
    Generate a multitone test signal.

    Args:
        frequencies: List of frequencies in Hz
        sampling_rate: Sampling rate in Hz
        amplitudes: List of amplitudes in volts. If None, all tones have amplitude 1/N
        phases: List of phases in radians. If None, all phases are 0
        duration: Signal duration in seconds

    Returns:
        Signal array

    Raises:
        ValueError: If lengths of frequencies, amplitudes, and phases don't match
    """
    n_tones = len(frequencies)

    if amplitudes is None:
        amplitudes = [1.0 / n_tones] * n_tones
    if phases is None:
        phases = [0.0] * n_tones

    if len(frequencies) != len(amplitudes) or len(frequencies) != len(phases):
        raise ValueError("Number of frequencies must match number of amplitudes and phases")

    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.zeros_like(t)

    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    return signal


def generate_imd_tones(f1: float,
                      delta_f: float,
                      sampling_rate: float,
                      amplitude: float = 0.5,
                      duration: float = 1.0,
                      order: int = 3) -> Tuple[np.ndarray, dict]:
    """
    Generate an Intermodulation Distortion (IMD) test signal.

    Args:
        f1: First tone frequency in Hz
        delta_f: Frequency spacing in Hz
        sampling_rate: Sampling rate in Hz
        amplitude: Total amplitude of both tones (split equally)
        duration: Signal duration in seconds
        order: IMD order to calculate expected frequencies (2 or 3)

    Returns:
        Tuple of (signal array, dict of expected IMD frequencies)

    Example:
        signal, imd_freqs = generate_imd_test(1e6, 1e3, 10e6)
        # imd_freqs contains frequencies of expected IMD products
    """
    f2 = f1 + delta_f
    amp_per_tone = amplitude / 2

    # Generate two-tone signal
    signal = generate_two_tone(f1, f2, sampling_rate, amp_per_tone, amp_per_tone, duration=duration)

    # Calculate expected IMD frequencies
    imd_freqs = {
        'f1': f1,
        'f2': f2,
        'imd2': {  # Second-order products
            'f1+f2': f1 + f2,
            'f2-f1': f2 - f1
        }
    }

    if order >= 3:
        imd_freqs['imd3'] = {  # Third-order products
            '2f1-f2': 2 * f1 - f2,
            '2f2-f1': 2 * f2 - f1,
            '2f1+f2': 2 * f1 + f2,
            '2f2+f1': 2 * f2 + f1
        }

    return signal, imd_freqs


def generate_digital_ramp(n_bits: int,
                          n_points: int = None) -> np.ndarray:
    """
    Generate digital ramp signal.

    Args:
        n_bits: DAC resolution
        n_points: Number of points. If None, generates all codes

    Returns:
        Array of digital codes
    """
    if n_points is None:
        n_points = 2 ** n_bits

    max_code = 2 ** n_bits - 1
    return np.linspace(0, max_code, n_points, dtype=int)


def generate_digital_step(n_bits: int,
                          step_points: List[int],
                          levels: List[int]) -> np.ndarray:
    """
    Generate digital step signal.

    Args:
        n_bits: DAC resolution
        step_points: List of points where steps occur
        levels: Digital codes for each step

    Returns:
        Array of digital codes
    """
    max_code = 2 ** n_bits - 1
    if any(level > max_code for level in levels):
        raise ValueError(f"All levels must be less than {max_code}")

    if any(level < 0 for level in levels):
        raise ValueError("All levels must be non-negative")

    signal = np.zeros(step_points[-1], dtype=int)
    current_level = levels[0]

    for point, level in zip(step_points[1:], levels[1:]):
        signal[point:] = level
        current_level = level

    return signal


def generate_digital_sine(n_bits: int,
                          frequency: float,
                          sampling_rate: float,
                          amplitude: float = 0.9,  # 90% of full scale
                          offset: float = 0.5,  # centered
                          duration: float = 1.0) -> np.ndarray:
    """
    Generate digital sine wave.

    Args:
        n_bits: DAC resolution
        frequency: Signal frequency in Hz
        sampling_rate: Sampling rate in Hz
        amplitude: Amplitude as fraction of full scale
        offset: DC offset as fraction of full scale
        duration: Signal duration in seconds

    Returns:
        Array of digital codes
    """
    max_code = 2 ** n_bits - 1
    t = np.arange(0, duration, 1 / sampling_rate)

    # Generate analog sine wave
    analog_sine = amplitude/2 * np.sin(2 * np.pi * frequency * t) + offset

    # Convert to digital codes
    digital_sine = np.round(analog_sine * max_code).astype(int)

    # Clip to valid range
    return np.clip(digital_sine, 0, max_code)


def generate_digital_two_tone(n_bits: int,
                              f1: float,
                              f2: float,
                              sampling_rate: float,
                              amplitude1: float = 0.45,  # 45% each
                              amplitude2: float = 0.45,
                              duration: float = 1.0) -> np.ndarray:
    """
    Generate digital two-tone signal.

    Args:
        n_bits: DAC resolution
        f1: First tone frequency in Hz
        f2: Second tone frequency in Hz
        sampling_rate: Sampling rate in Hz
        amplitude1: First tone amplitude as fraction of full scale
        amplitude2: Second tone amplitude as fraction of full scale
        duration: Signal duration in seconds

    Returns:
        Array of digital codes
    """
    max_code = 2 ** n_bits - 1
    t = np.arange(0, duration, 1 / sampling_rate)

    # Generate analog two-tone
    analog_signal = (amplitude1 * np.sin(2 * np.pi * f1 * t) +
                     amplitude2 * np.sin(2 * np.pi * f2 * t) + 0.5)

    # Convert to digital codes
    digital_signal = np.round(analog_signal * max_code).astype(int)

    # Clip to valid range
    return np.clip(digital_signal, 0, max_code)


def generate_digital_multitone(n_bits: int,
                               frequencies: List[float],
                               sampling_rate: float,
                               amplitudes: List[float] = None,
                               duration: float = 1.0) -> np.ndarray:
    """
    Generate digital multitone signal.

    Args:
        n_bits: DAC resolution
        frequencies: List of frequencies in Hz
        sampling_rate: Sampling rate in Hz
        amplitudes: List of amplitudes as fraction of full scale
        duration: Signal duration in seconds

    Returns:
        Array of digital codes
    """
    if amplitudes is None:
        # Default: equal amplitudes that sum to 0.9
        n_tones = len(frequencies)
        amplitudes = [0.9 / n_tones] * n_tones

    max_code = 2 ** n_bits - 1
    t = np.arange(0, duration, 1 / sampling_rate)

    # Generate analog multitone
    analog_signal = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        analog_signal += amp * np.sin(2 * np.pi * freq * t)

    # Add offset to center the signal
    analog_signal += 0.5

    # Convert to digital codes
    digital_signal = np.round(analog_signal * max_code).astype(int)

    # Clip to valid range
    return np.clip(digital_signal, 0, max_code)


def generate_digital_imd_tones(n_bits: int,
                               f1: float,
                               delta_f: float,
                               sampling_rate: float,
                               amplitude: float = 0.9,
                               duration: float = 1.0) -> Tuple[np.ndarray, dict]:
    """
    Generate digital IMD test signal.

    Args:
        n_bits: DAC resolution
        f1: First tone frequency in Hz
        delta_f: Frequency spacing in Hz
        sampling_rate: Sampling rate in Hz
        amplitude: Total amplitude as fraction of full scale
        duration: Signal duration in seconds

    Returns:
        Tuple of (digital codes, dict of IMD frequencies)
    """
    f2 = f1 + delta_f
    amp_per_tone = amplitude / 2

    # Generate digital two-tone
    signal = generate_digital_two_tone(n_bits, f1, f2, sampling_rate,
                                       amp_per_tone, amp_per_tone, duration)

    # Calculate expected IMD frequencies
    imd_freqs = {
        'f1': f1,
        'f2': f2,
        'imd2': {  # Second-order products
            'f1+f2': f1 + f2,
            'f2-f1': f2 - f1
        },
        'imd3': {  # Third-order products
            '2f1-f2': 2 * f1 - f2,
            '2f2-f1': 2 * f2 - f1,
            '2f1+f2': 2 * f1 + f2,
            '2f2+f1': 2 * f2 + f1
        }
    }

    return signal, imd_freqs

# Usage example
if __name__ == "__main__":
    # Generate a sine wave
    sine = generate_sine(frequency=1e3, sampling_rate=1e6, amplitude=0.5)

    # Convert to differential with different common mode voltages
    v_pos1, v_neg1 = convert_to_differential(sine)  # vcm = 0
    v_pos2, v_neg2 = convert_to_differential(sine, vcm=0.5)  # vcm = 0.5V


    # Two-tone with phase control
    signal_2tone = generate_two_tone(
        f1=1e3,
        f2=1.1e3,
        sampling_rate=1e6,
        amplitude1=0.4,
        amplitude2=0.4,
        phase1=0,
        phase2=np.pi / 2,  # 90 degree phase shift
        duration=0.01
    )

    # IMD test
    imd_signal, imd_frequencies = generate_imd_tones(
        f1=1e6,  # 1 MHz
        delta_f=1e3,  # 1 kHz spacing
        sampling_rate=10e6,
        amplitude=0.8,
        order=3
    )

    print("IMD Test Frequencies:")
    for key, value in imd_frequencies.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, freq in value.items():
                print(f"  {subkey}: {freq / 1e3:.1f} kHz")
        else:
            print(f"{key}: {value / 1e3:.1f} kHz")


    import matplotlib.pyplot as plt

    # Test parameters
    n_bits = 12
    fs = 1e6  # 1 MHz sampling rate

    # 1. Generate digital sine wave
    print("\nTest 1: Digital Sine Wave")
    print("--------------------------")
    sine = generate_digital_sine(
        n_bits=n_bits,
        frequency=1e3,  # 1 kHz
        sampling_rate=fs,
        amplitude=0.9,
        duration=0.005
    )

    t = np.arange(len(sine)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, sine, '.-')
    plt.title('Digital Sine Wave')
    plt.xlabel('Time (ms)')
    plt.ylabel('Code')
    plt.grid(True)
    plt.show()

    # 2. Generate digital two-tone signal
    print("\nTest 2: Digital Two-Tone Signal")
    print("-------------------------------")
    signal_2tone = generate_digital_two_tone(
        n_bits=n_bits,
        f1=1e3,  # 1 kHz
        f2=1.1e3,  # 1.1 kHz
        sampling_rate=fs,
        amplitude1=0.4,
        amplitude2=0.4,
        duration=0.01
    )

    t = np.arange(len(signal_2tone)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, signal_2tone, '.-')
    plt.title('Digital Two-Tone Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Code')
    plt.grid(True)
    plt.show()

    # 3. Generate digital IMD test signal
    print("\nTest 3: Digital IMD Test")
    print("------------------------")
    imd_signal, imd_frequencies = generate_digital_imd_tones(
        n_bits=n_bits,
        f1=1e6,  # 1 MHz
        delta_f=1e3,  # 1 kHz spacing
        sampling_rate=10e6,
        amplitude=0.8,
        duration=0.01
    )

    print("IMD Test Frequencies:")
    for key, value in imd_frequencies.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, freq in value.items():
                print(f"  {subkey}: {freq / 1e3:.1f} kHz")
        else:
            print(f"{key}: {value / 1e3:.1f} kHz")

    # 4. Generate digital ramp
    print("\nTest 4: Digital Ramp")
    print("-------------------")
    ramp = generate_digital_ramp(n_bits=n_bits, n_points=1000)

    plt.figure(figsize=(10, 4))
    plt.plot(ramp, '.-')
    plt.title('Digital Ramp')
    plt.xlabel('Sample')
    plt.ylabel('Code')
    plt.grid(True)
    plt.show()

    # 5. Generate digital step
    print("\nTest 5: Digital Step")
    print("-------------------")
    step_points = [0, 200, 400, 600, 800]
    levels = [0, 1000, 2000, 3000, 4000]
    step = generate_digital_step(n_bits=n_bits, step_points=step_points, levels=levels)

    plt.figure(figsize=(10, 4))
    plt.plot(step, '.-')
    plt.title('Digital Step')
    plt.xlabel('Sample')
    plt.ylabel('Code')
    plt.grid(True)
    plt.show()

    # 6. Generate digital multitone
    print("\nTest 6: Digital Multitone")
    print("------------------------")
    freqs = [1e3, 2e3, 3e3, 4e3]  # 1kHz, 2kHz, 3kHz, 4kHz
    amps = [0.2, 0.2, 0.2, 0.2]  # Equal amplitudes
    signal_multi = generate_digital_multitone(
        n_bits=n_bits,
        frequencies=freqs,
        sampling_rate=fs,
        amplitudes=amps,
        duration=0.01
    )

    t = np.arange(len(signal_multi)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, signal_multi, '.-')
    plt.title('Digital Multitone Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Code')
    plt.grid(True)
    plt.show()

