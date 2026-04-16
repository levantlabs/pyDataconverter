"""
Signal Generation Utilities
==========================

Provides functions to generate various test signals for ADC testing.
"""

import numpy as np
from typing import Optional, Union, Tuple, List


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
                  fs: float,
                  amplitude: float = 1.0,
                  offset: float = 0.0,
                  duration: float = 1.0,
                  phase: float = 0.0) -> np.ndarray:
    """
    Generate a sinusoidal test signal.

    Args:
        frequency: Signal frequency in Hz
        fs: Sampling rate in Hz
        amplitude: Peak amplitude in volts
        offset: DC offset in volts
        duration: Signal duration in seconds
        phase: Initial phase in radians (default 0.0)

    Returns:
        Signal array
    """
    t = np.arange(0, duration, 1 / fs)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset


def generate_chirp(fs: float,
                   n_samples: int,
                   f_start: float,
                   f_stop: float,
                   amplitude: float = 1.0,
                   offset: float = 0.0,
                   method: str = 'linear',
                   phi: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a swept-frequency (chirp) signal.

    Args:
        fs: Sampling rate in Hz.
        n_samples: Number of samples.
        f_start: Start frequency in Hz.
        f_stop: Stop frequency in Hz.
        amplitude: Peak amplitude in volts (default 1.0).
        offset: DC offset in volts (default 0.0).
        method: Frequency sweep method — 'linear' (default) or 'logarithmic'.
        phi: Phase offset in degrees (default 0.0), passed to scipy.signal.chirp.

    Returns:
        Tuple of (signal array, time array).
    """
    from scipy.signal import chirp as scipy_chirp
    t = np.arange(n_samples) / fs
    sig = scipy_chirp(t, f0=f_start, f1=f_stop, t1=t[-1], method=method, phi=phi)
    return amplitude * sig + offset, t


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

    Each entry of ``levels`` is the amplitude of one segment:

        signal[0                : step_points[0]]   = levels[0]
        signal[step_points[i-1] : step_points[i]]   = levels[i]   (1 ≤ i < N)
        signal[step_points[-1]  : samples]          = levels[N]

    where ``N = len(step_points)``, so ``len(levels) == N + 1``.

    Args:
        samples: Total number of samples.
        step_points: Sample indices where transitions occur. Must be
            non-decreasing and every entry must lie in ``[0, samples]``.
        levels: One value per segment; ``len(levels)`` must equal
            ``len(step_points) + 1``.

    Returns:
        Signal array of length ``samples``.

    Raises:
        ValueError: If ``len(levels) != len(step_points) + 1``, or if
            ``step_points`` is out-of-range or non-monotonic.
    """
    if len(levels) != len(step_points) + 1:
        raise ValueError(
            f"len(levels) must equal len(step_points) + 1 "
            f"(got len(levels)={len(levels)}, len(step_points)={len(step_points)})"
        )
    for p in step_points:
        if p < 0 or p > samples:
            raise ValueError(
                f"step_points entry {p} out of range [0, {samples}]"
            )
    for i in range(len(step_points) - 1):
        if step_points[i] > step_points[i + 1]:
            raise ValueError(
                f"step_points must be non-decreasing; got {list(step_points)}"
            )

    signal = np.full(samples, levels[0], dtype=float)
    for point, level in zip(step_points, levels[1:]):
        signal[point:] = level

    return signal


def generate_two_tone(f1: float,
                      f2: float,
                      fs: float,
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
        fs: Sampling rate in Hz
        amplitude1: First tone amplitude in volts
        amplitude2: Second tone amplitude in volts
        phase1: First tone phase in radians
        phase2: Second tone phase in radians
        duration: Signal duration in seconds

    Returns:
        Signal array
    """
    t = np.arange(0, duration, 1 / fs)
    tone1 = amplitude1 * np.sin(2 * np.pi * f1 * t + phase1)
    tone2 = amplitude2 * np.sin(2 * np.pi * f2 * t + phase2)
    return tone1 + tone2


def generate_multitone(frequencies: List[float],
                       fs: float,
                       amplitudes: List[float] = None,
                       phases: List[float] = None,
                       duration: float = 1.0) -> np.ndarray:
    """
    Generate a multitone test signal.

    Args:
        frequencies: List of frequencies in Hz
        fs: Sampling rate in Hz
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

    t = np.arange(0, duration, 1 / fs)
    frequencies = np.asarray(frequencies)
    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)

    # Vectorized: outer product gives (n_tones, n_samples) matrix
    signal = (amplitudes[:, np.newaxis] *
              np.sin(2 * np.pi * np.outer(frequencies, t) + phases[:, np.newaxis])
              ).sum(axis=0)

    return signal


def generate_imd_tones(f1: float,
                      delta_f: float,
                      fs: float,
                      amplitude: float = 0.5,
                      duration: float = 1.0,
                      order: int = 3) -> Tuple[np.ndarray, dict]:
    """
    Generate an Intermodulation Distortion (IMD) test signal.

    Args:
        f1: First tone frequency in Hz
        delta_f: Frequency spacing in Hz
        fs: Sampling rate in Hz
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
    signal = generate_two_tone(f1, f2, fs, amp_per_tone, amp_per_tone, duration=duration)

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
                          samples: int,
                          step_points: List[int],
                          levels: List[int]) -> np.ndarray:
    """
    Generate a digital step signal as an array of integer DAC codes.

    Same segment contract as :func:`generate_step`:

        signal[0                : step_points[0]]   = levels[0]
        signal[step_points[i-1] : step_points[i]]   = levels[i]   (1 ≤ i < N)
        signal[step_points[-1]  : samples]          = levels[N]

    where ``N = len(step_points)``, so ``len(levels) == N + 1``.

    Args:
        n_bits: DAC resolution.
        samples: Total number of samples in the output array.
        step_points: Sample indices where transitions occur. Must be
            non-decreasing and every entry must lie in ``[0, samples]``.
        levels: One integer code per segment, each in
            ``[0, 2^n_bits − 1]``. ``len(levels)`` must equal
            ``len(step_points) + 1``.

    Returns:
        Integer code array of length ``samples``.

    Raises:
        ValueError: If any code is out of range, if
            ``len(levels) != len(step_points) + 1``, or if
            ``step_points`` is out-of-range or non-monotonic.
    """
    max_code = 2 ** n_bits - 1
    if any(level > max_code for level in levels):
        raise ValueError(f"All levels must be less than {max_code}")

    if any(level < 0 for level in levels):
        raise ValueError("All levels must be non-negative")

    if len(levels) != len(step_points) + 1:
        raise ValueError(
            f"len(levels) must equal len(step_points) + 1 "
            f"(got len(levels)={len(levels)}, len(step_points)={len(step_points)})"
        )
    for p in step_points:
        if p < 0 or p > samples:
            raise ValueError(
                f"step_points entry {p} out of range [0, {samples}]"
            )
    for i in range(len(step_points) - 1):
        if step_points[i] > step_points[i + 1]:
            raise ValueError(
                f"step_points must be non-decreasing; got {list(step_points)}"
            )

    signal = np.full(samples, levels[0], dtype=int)
    for point, level in zip(step_points, levels[1:]):
        signal[point:] = level

    return signal


def generate_digital_sine(n_bits: int,
                          frequency: float,
                          fs: float,
                          amplitude: float = 0.9,  # 90% of full scale
                          offset: float = 0.5,  # centered
                          duration: float = 1.0) -> np.ndarray:
    """
    Generate digital sine wave.

    Args:
        n_bits: DAC resolution
        frequency: Signal frequency in Hz
        fs: Sampling rate in Hz
        amplitude: Amplitude as fraction of full scale
        offset: DC offset as fraction of full scale
        duration: Signal duration in seconds

    Returns:
        Array of digital codes
    """
    max_code = 2 ** n_bits - 1
    t = np.arange(0, duration, 1 / fs)

    # Generate analog sine wave
    analog_sine = amplitude/2 * np.sin(2 * np.pi * frequency * t) + offset

    # Convert to digital codes
    digital_sine = np.round(analog_sine * max_code).astype(int)

    # Clip to valid range
    return np.clip(digital_sine, 0, max_code)


def generate_digital_two_tone(n_bits: int,
                              f1: float,
                              f2: float,
                              fs: float,
                              amplitude1: float = 0.45,  # 45% each
                              amplitude2: float = 0.45,
                              duration: float = 1.0) -> np.ndarray:
    """
    Generate digital two-tone signal.

    Args:
        n_bits: DAC resolution
        f1: First tone frequency in Hz
        f2: Second tone frequency in Hz
        fs: Sampling rate in Hz
        amplitude1: First tone amplitude as fraction of full scale
        amplitude2: Second tone amplitude as fraction of full scale
        duration: Signal duration in seconds

    Returns:
        Array of digital codes
    """
    max_code = 2 ** n_bits - 1
    t = np.arange(0, duration, 1 / fs)

    # Generate analog two-tone
    analog_signal = (amplitude1 * np.sin(2 * np.pi * f1 * t) +
                     amplitude2 * np.sin(2 * np.pi * f2 * t) + 0.5)

    # Convert to digital codes
    digital_signal = np.round(analog_signal * max_code).astype(int)

    # Clip to valid range
    return np.clip(digital_signal, 0, max_code)


def generate_digital_multitone(n_bits: int,
                               frequencies: List[float],
                               fs: float,
                               amplitudes: List[float] = None,
                               duration: float = 1.0) -> np.ndarray:
    """
    Generate digital multitone signal.

    Args:
        n_bits: DAC resolution
        frequencies: List of frequencies in Hz
        fs: Sampling rate in Hz
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
    t = np.arange(0, duration, 1 / fs)

    # Generate analog multitone — vectorized
    frequencies_arr = np.asarray(frequencies)
    amplitudes_arr = np.asarray(amplitudes)
    analog_signal = (amplitudes_arr[:, np.newaxis] *
                     np.sin(2 * np.pi * np.outer(frequencies_arr, t))
                     ).sum(axis=0)

    # Add offset to center the signal
    analog_signal += 0.5

    # Convert to digital codes
    digital_signal = np.round(analog_signal * max_code).astype(int)

    # Clip to valid range
    return np.clip(digital_signal, 0, max_code)


def generate_digital_imd_tones(n_bits: int,
                               f1: float,
                               delta_f: float,
                               fs: float,
                               amplitude: float = 0.9,
                               duration: float = 1.0) -> Tuple[np.ndarray, dict]:
    """
    Generate digital IMD test signal.

    Args:
        n_bits: DAC resolution
        f1: First tone frequency in Hz
        delta_f: Frequency spacing in Hz
        fs: Sampling rate in Hz
        amplitude: Total amplitude as fraction of full scale
        duration: Signal duration in seconds

    Returns:
        Tuple of (digital codes, dict of IMD frequencies)
    """
    f2 = f1 + delta_f
    amp_per_tone = amplitude / 2

    # Generate digital two-tone
    signal = generate_digital_two_tone(n_bits, f1, f2, fs,
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

def generate_prbs(order: int,
                  n_samples: int,
                  amplitude: float = 1.0,
                  offset: float = 0.0,
                  seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Pseudo-Random Binary Sequence (PRBS).

    Uses a maximal-length linear feedback shift register (LFSR) to produce
    a balanced ±amplitude binary sequence with a flat power spectrum.

    Args:
        order: LFSR order (2–20). The full period length is 2^order − 1.
        n_samples: Number of output samples. The PRBS is tiled or truncated
            to this length.
        amplitude: Half-range of the output (default 1.0). Output values
            are +amplitude and −amplitude.
        offset: DC offset in volts (default 0.0).
        seed: Optional integer seed for the initial LFSR state. If None a
            fixed all-ones state is used, giving a deterministic sequence.

    Returns:
        Signal array of length n_samples with values {offset±amplitude}.
    """
    # Standard maximal-length LFSR taps (Fibonacci form, bit positions 1-indexed from LSB)
    _TAPS = {
        2:  [1, 2],
        3:  [1, 2],
        4:  [1, 2],
        5:  [1, 3],
        6:  [1, 2],
        7:  [1, 2],
        8:  [1, 2, 3, 8],
        9:  [1, 5],
        10: [1, 4],
        11: [1, 3],
        12: [1, 2, 3, 9],
        13: [1, 2, 3, 6],
        14: [1, 2, 3, 13],
        15: [1, 2],
        16: [1, 2, 4, 13],
        17: [1, 4],
        18: [1, 8],
        19: [1, 2, 3, 6],
        20: [1, 4],
    }
    if order not in _TAPS:
        raise ValueError(f"order must be between 2 and 20, got {order}")

    taps = _TAPS[order]
    period = 2 ** order - 1

    rng = np.random.default_rng(seed)
    state = int(rng.integers(1, 2**order)) if seed is not None else (2**order - 1)

    bits = np.zeros(period, dtype=np.int8)
    for i in range(period):
        bits[i] = state & 1
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (tap - 1)) & 1
        state = ((state >> 1) | (feedback << (order - 1))) & ((1 << order) - 1)

    # Tile to n_samples
    reps = (n_samples + period - 1) // period
    tiled = np.tile(bits, reps)[:n_samples]

    # Map 0/1 → -amplitude/+amplitude
    return tiled.astype(float) * 2 * amplitude - amplitude + offset


def apply_channel(signal: np.ndarray,
                  h: np.ndarray) -> np.ndarray:
    """
    Apply a channel impulse response to a signal (linear convolution).

    Output is truncated to the same length as the input so the return
    array can be fed directly to an ADC.

    Args:
        signal: Input signal array.
        h: Channel impulse response (FIR filter coefficients).

    Returns:
        Signal array of the same length as ``signal``.
    """
    from scipy.signal import fftconvolve
    out = fftconvolve(signal, h, mode='full')
    return out[:len(signal)]


def generate_gaussian_noise(n_samples: int,
                            std: float = 1.0,
                            offset: float = 0.0,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate Gaussian white noise.

    Args:
        n_samples: Number of samples.
        std: Standard deviation in volts (default 1.0).
        offset: DC offset in volts (default 0.0).
        rng: Optional numpy Generator for reproducibility. If None, uses
             the global numpy random state.

    Returns:
        Signal array of length n_samples.
    """
    if rng is None:
        return np.random.normal(loc=offset, scale=std, size=n_samples)
    return rng.normal(loc=offset, scale=std, size=n_samples)


def apply_window(signal: np.ndarray, window_type: str) -> np.ndarray:
    """
    Apply a window function to a signal.

    Supports the same window names used by ``compute_fft``:
    'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'flattop',
    'boxcar', 'tukey', 'cosine', 'exponential'.

    Args:
        signal: Input signal array.
        window_type: Name of the window (case-sensitive, scipy.signal.windows).

    Returns:
        Windowed signal array of the same length as ``signal``.

    Raises:
        ValueError: If window_type is not a recognised window name.
    """
    _ALLOWED_WINDOWS = {
        'hann', 'hamming', 'blackman', 'bartlett', 'kaiser',
        'flattop', 'boxcar', 'tukey', 'cosine', 'exponential',
    }
    if window_type not in _ALLOWED_WINDOWS:
        raise ValueError(
            f"Unknown window '{window_type}'. "
            f"Allowed: {sorted(_ALLOWED_WINDOWS)}")
    from scipy.signal import windows as sp_windows
    w = getattr(sp_windows, window_type)(len(signal))
    return signal * w


def generate_coherent_sine(fs: float,
                           n_fft: int,
                           n_fin: int,
                           amplitude: float = 1.0,
                           offset: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Generate a coherent sine wave for FFT-based ADC testing.

    Coherent sampling places the signal at an exact FFT bin frequency
    (f_in = n_fin / n_fft * fs), eliminating spectral leakage so that
    dynamic metrics such as SNR, SFDR, THD, and ENOB can be measured
    accurately from a single FFT window.

    Args:
        fs: Sampling rate in Hz
        n_fft: FFT size. Sets duration = n_fft / fs so that
               exactly one FFT window is captured.
        n_fin: Input frequency bin number (integer, 1 <= n_fin < n_fft/2).
               The actual frequency is f_in = n_fin / n_fft * fs.
        amplitude: Peak amplitude in volts (default 1.0)
        offset: DC offset in volts (default 0.0)

    Returns:
        Tuple of (signal array, input frequency in Hz)
    """
    f_in = n_fin / n_fft * fs
    duration = n_fft / fs
    signal = generate_sine(f_in, fs, amplitude, offset, duration)
    return signal, f_in


def generate_coherent_two_tone(fs: float,
                               n_fft: int,
                               n_fin1: int,
                               n_fin2: int,
                               amplitude1: float = 0.5,
                               amplitude2: float = 0.5,
                               phase1: float = 0.0,
                               phase2: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """
    Generate a coherent two-tone signal for FFT-based ADC testing.

    Both tones land on exact FFT bin frequencies, eliminating spectral
    leakage. Useful for IMD and two-tone intermodulation measurements
    where the in-band IMD products must be resolved accurately.

    Args:
        fs: Sampling rate in Hz
        n_fft: FFT size. Duration = n_fft / fs.
        n_fin1: First tone frequency bin number
        n_fin2: Second tone frequency bin number
        amplitude1: First tone peak amplitude in volts (default 0.5)
        amplitude2: Second tone peak amplitude in volts (default 0.5)
        phase1: First tone initial phase in radians (default 0.0)
        phase2: Second tone initial phase in radians (default 0.0)

    Returns:
        Tuple of (signal array, f1 in Hz, f2 in Hz)
    """
    f1 = n_fin1 / n_fft * fs
    f2 = n_fin2 / n_fft * fs
    duration = n_fft / fs
    signal = generate_two_tone(f1, f2, fs, amplitude1, amplitude2,
                               phase1, phase2, duration)
    return signal, f1, f2


# Usage example
if __name__ == "__main__":  # pragma: no cover
    # Generate a sine wave
    sine = generate_sine(frequency=1e3, fs=1e6, amplitude=0.5)

    # Convert to differential with different common mode voltages
    v_pos1, v_neg1 = convert_to_differential(sine)  # vcm = 0
    v_pos2, v_neg2 = convert_to_differential(sine, vcm=0.5)  # vcm = 0.5V


    # Two-tone with phase control
    signal_2tone = generate_two_tone(
        f1=1e3,
        f2=1.1e3,
        fs=1e6,
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
        fs=10e6,
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
        fs=fs,
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
        fs=fs,
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
        fs=10e6,
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
    step_points = [200, 400, 600, 800]
    levels = [0, 1000, 2000, 3000, 4000]
    step = generate_digital_step(n_bits=n_bits, samples=1000,
                                 step_points=step_points, levels=levels)

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
        fs=fs,
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

