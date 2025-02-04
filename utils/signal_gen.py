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
