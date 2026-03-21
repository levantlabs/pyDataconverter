"""
FFT Analysis Utilities
=====================

Provides functions for FFT computation and visualization.
"""

import numpy as np
from typing import Tuple, List
from scipy import signal
from enum import Enum

import matplotlib.pyplot as plt
from pyDataconverter.utils.signal_gen import generate_sine, generate_two_tone, convert_to_differential



class FFTNormalization(Enum):
    NONE = 'none'        # Raw FFT values in dB
    POWER = 'power'      # Normalize by FFT power
    DBFS = 'dbfs'        # Normalize to full scale


def compute_fft(time_data: np.ndarray,
                fs: float,
                window: str = None,
                remove_dc: bool = True,
                normalization: FFTNormalization = FFTNormalization.NONE,
                full_scale: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT with optional windowing and different normalization options.

    Args:
        time_data: Input signal array
        fs: Sampling frequency in Hz
        window: Window type ('hann', 'blackman', etc.), None for no window
        remove_dc: If True, removes DC component
        normalization: Type of normalization to apply
        full_scale: Full scale value for dBFS normalization

    Returns:
        Tuple of (frequencies, magnitudes_db)

    Raises:
        ValueError: If dBFS normalization is selected but full_scale is None
    """

    N = len(time_data)  # FFT length
    # Remove DC if requested
    if remove_dc:
        time_data = time_data - np.mean(time_data)

    # Apply window if specified
    if window is not None:
        window_func = getattr(signal.windows, window)
        time_data = time_data * window_func(N)

    # Compute FFT
    fft_data = np.fft.fft(time_data)
    freqs = np.fft.fftfreq(len(time_data), 1 / fs)

    # Convert to dB
    magnitudes_db = 20 * np.log10(np.abs(fft_data))

    # Apply normalization
    if normalization == FFTNormalization.POWER:
        # Normalize by FFT length

        magnitudes_db = magnitudes_db - 20 * np.log10(N)

    elif normalization == FFTNormalization.DBFS:
        if full_scale is None:
            raise ValueError("full_scale must be provided for dBFS normalization")
        # Normalize to full scale
        magnitudes_db = magnitudes_db - 20 * np.log10(full_scale) - 20 * np.log10(N/2)

    # Return positive frequencies only
    pos_mask = freqs >= 0
    return freqs[pos_mask], magnitudes_db[pos_mask]


def _get_harmonic(freqs: np.ndarray,
                  mags: np.ndarray,
                  f0: float,
                  fs: float,
                  n: int,
                  tol: float = 0.1) -> Tuple[float, float]:
    """
    Internal function to find the nth harmonic frequency and magnitude, accounting for aliasing.

    Args:
        freqs: Frequency array
        mags: Magnitude array (in dB)
        f0: Fundamental frequency
        fs: Sampling frequency
        n: Harmonic number (1 = fundamental)
        tol: Frequency tolerance as fraction of frequency spacing

    Returns:
        Tuple of (frequency, magnitude) for the harmonic

    Raises:
        ValueError: If harmonic not found within tolerance
    """
    freq_spacing = freqs[1] - freqs[0]
    nyquist = fs / 2

    # True harmonic frequency
    f_harm = n * f0

    # Calculate where this harmonic would appear due to aliasing
    if f_harm <= nyquist:
        target_freq = f_harm
    else:
        num_folds = int(np.ceil(f_harm / fs))
        if num_folds % 2 == 0:
            target_freq = f_harm % fs
        else:
            target_freq = fs - (f_harm % fs)

        if target_freq > nyquist:
            target_freq = fs - target_freq

    idx = np.argmin(np.abs(freqs - target_freq))

    if abs(freqs[idx] - target_freq) <= tol * freq_spacing:
        return (freqs[idx], mags[idx])
    else:
        raise ValueError(f"Harmonic {n} not found at expected frequency {target_freq}")


def find_fundamental(freqs: np.ndarray,
                     mags: np.ndarray,
                     f0: float,
                     fs: float,
                     tol: float = 0.1) -> Tuple[float, float]:
    """
    Find the fundamental frequency and its magnitude.

    Args:
        freqs: Frequency array
        mags: Magnitude array (in dB)
        f0: Expected fundamental frequency
        fs: Sampling frequency
        tol: Frequency tolerance as fraction of frequency spacing

    Returns:
        Tuple of (frequency, magnitude) for fundamental
    """
    return _get_harmonic(freqs, mags, f0, fs, n=1, tol=tol)


def find_harmonics(freqs: np.ndarray,
                   mags: np.ndarray,
                   f0: float,
                   fs: float,
                   num_harmonics: int = 5,
                   tol: float = 0.1) -> List[Tuple[float, float]]:
    """
    Find harmonic frequencies and their magnitudes, excluding fundamental.

    Args:
        freqs: Frequency array
        mags: Magnitude array (in dB)
        f0: Fundamental frequency
        fs: Sampling frequency
        num_harmonics: Number of harmonics to find (excluding fundamental)
        tol: Frequency tolerance as fraction of frequency spacing

    Returns:
        List of (frequency, magnitude) tuples for each harmonic found
    """
    harmonics = []
    for n in range(2, num_harmonics + 2):
        try:
            harm_freq, harm_mag = _get_harmonic(freqs, mags, f0, fs, n, tol)
            harmonics.append((harm_freq, harm_mag))
            print(f"Harmonic {n}: True freq = {n * f0 / 1e3:.1f} kHz, "
                  f"Aliased freq = {harm_freq / 1e3:.1f} kHz, "
                  f"Magnitude = {harm_mag:.1f} dB")
        except ValueError as e:
            print(f"Warning: {e}")

    return harmonics
def plot_fft(freqs: np.ndarray,
             mags: np.ndarray,
             title: str = "FFT",
             max_freq: float = None,
             min_db: float = None,
             max_db: float = None,
             fig=None,
             ax=None):
    """
    Plot FFT with automatic frequency unit selection.

    Args:
        freqs: Frequency array in Hz
        mags: Magnitude array in dB
        title: Plot title
        max_freq: Maximum frequency to display (in Hz), if None shows all frequencies
        min_db: Minimum dB value to display
        max_db: Maximum dB value to display, if None uses max of data + small margin
        fig: Matplotlib figure to plot on (optional)
        ax: Matplotlib axis to plot on (optional)
    """
    import matplotlib.pyplot as plt

    # Create figure if not provided
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Apply frequency limit if specified
    if max_freq:
        mask = freqs <= max_freq
        freqs = freqs[mask]
        mags = mags[mask]
    else:
        max_freq = np.max(freqs)

    # Determine frequency unit
    if max_freq >= 1e9:
        freq_scale = 1e9
        freq_unit = 'GHz'
    elif max_freq >= 1e6:
        freq_scale = 1e6
        freq_unit = 'MHz'
    elif max_freq >= 1e3:
        freq_scale = 1e3
        freq_unit = 'kHz'
    else:
        freq_scale = 1
        freq_unit = 'Hz'

    # Find nice maximum frequency for plot
    scaled_max = max_freq / freq_scale
    if scaled_max <= 1:
        plot_max = 1
    else:
        plot_max = np.ceil(scaled_max)

    # Plot
    ax.plot(freqs / freq_scale, mags)
    ax.grid(True)
    ax.set_xlabel(f'Frequency ({freq_unit})')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)

    # Set axis limits
    ax.set_xlim(0, plot_max)
    if max_db is None:
        max_db = np.ceil(np.max(mags)) + 2  # Add 2 dB margin
    ax.set_ylim(min_db, max_db)

    return ax
"""
FFT Analysis Demo
================

Demonstrates FFT analysis with various test signals.
"""


def demo_fft_analysis():
    """Demonstrate FFT analysis functions with various test signals"""
    import matplotlib.pyplot as plt

    # Test parameters
    fs = 1e6  # 1 MHz sampling rate
    NFFT = 1024
    NFIN = 11
    duration = NFFT / fs

    # 1. Single tone with harmonics
    print("\nDemo 1: Single tone with harmonics")
    f0 = NFIN / NFFT * fs  # Coherent signal

    # Generate fundamental and harmonics using signal generator
    signal1 = generate_sine(f0, fs, amplitude=1.0, duration=duration)
    signal2 = generate_sine(2 * f0, fs, amplitude=0.1, duration=duration)
    signal3 = generate_sine(3 * f0, fs, amplitude=0.05, duration=duration)
    signal = signal1 + signal2 + signal3

    # In demo_fft_analysis()
    freqs, mags = compute_fft(signal, fs)  # No window

    # Find fundamental
    fund_freq, fund_mag = find_fundamental(freqs, mags, f0, fs)
    print(f"Fundamental: freq = {fund_freq / 1e3:.1f} kHz, magnitude = {fund_mag:.1f} dB")

    # Find harmonics (excluding fundamental)
    harmonics = find_harmonics(freqs, mags, f0, fs, num_harmonics=3)


    fig, ax = plt.subplots(figsize=(12, 6))
    plot_fft(freqs, mags, "Single Tone with Harmonics (No Window)",
             max_freq=5 * f0, fig=fig, ax=ax)
    plt.show()

    # 2. Compare no window vs different windows
    print("\nDemo 2: Window function comparison")
    f1, f2 = f0, f0 + fs / NFFT  # Adjacent bins
    signal = generate_two_tone(f1, f2, fs, amplitude1=0.5, amplitude2=0.5, duration=duration)

    windows = [None, 'hann', 'blackman', 'hamming']
    #fig = plt.figure(figsize=(15, 10))
    fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(15, 10))
    axs = axs.flatten()


    for i, window in enumerate(windows):
        #ax = fig.subplots(2, 2, i + 1)
        freqs, mags = compute_fft(signal, fs, window=window)
        title = 'No Window' if window is None else f'{window.capitalize()} Window'
        print('Plotting window: {}'.format(title))
        print(axs[i])
        plot_fft(freqs, mags, title, max_freq=5 * f2, min_db=-100,
                 fig=fig, ax=axs[i])

    plt.tight_layout()
    plt.show()

    # 3. Spectral leakage demonstration
    print("\nDemo 3: Spectral Leakage")
    # Use non-coherent sampling
    f_leak = f0 * 1.5  # Non-coherent frequency
    signal_leak = generate_sine(f_leak, fs, amplitude=1.0, duration=duration)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for i, window in enumerate([None, 'hann']):
        freqs, mags = compute_fft(signal_leak, fs, window=window)
        title = 'Spectral Leakage - No Window' if window is None else 'Reduced Leakage - Hann Window'
        plot_fft(freqs, mags, title, max_freq=2 * f_leak, min_db=-100,
                 fig=fig, ax=[ax1, ax2][i])

    plt.tight_layout()
    plt.show()

    # 4. Frequency resolution demo
    print("\nDemo 4: Frequency resolution comparison")
    durations = [NFFT / fs, 4 * NFFT / fs]  # Compare different FFT lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for i, dur in enumerate(durations):
        signal = generate_sine(f0, fs, amplitude=0.5, duration=dur) + \
                 generate_sine(f0 + fs / NFFT / 2, fs, amplitude=0.5, duration=dur)

        freqs, mags = compute_fft(signal, fs)  # No window for resolution demo
        title = f'Duration: {dur * 1e6:.1f} µs\nΔf = {fs / len(signal):.1f} Hz'
        plot_fft(freqs, mags, title, max_freq=2 * f0, min_db=-100,
                 fig=fig, ax=[ax1, ax2][i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_fft_analysis()