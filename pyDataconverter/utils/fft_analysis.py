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
from pyDataconverter.utils.signal_gen import (generate_sine, generate_two_tone,
                                               convert_to_differential,
                                               generate_coherent_sine)



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
        full_scale: Full scale value for dBFS normalization.  This is the largest code value assuming offset binary

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
        _ALLOWED_WINDOWS = {
            'hann', 'hamming', 'blackman', 'bartlett', 'kaiser',
            'flattop', 'boxcar', 'tukey', 'cosine', 'exponential',
        }
        if window not in _ALLOWED_WINDOWS:
            raise ValueError(
                f"Unknown window '{window}'. Allowed: {sorted(_ALLOWED_WINDOWS)}")
        window_func = getattr(signal.windows, window)
        time_data = time_data * window_func(N)

    # Compute FFT
    fft_data = np.fft.fft(time_data)
    freqs = np.fft.fftfreq(len(time_data), 1 / fs)

    # Convert to dB, always normalizing by FFT length N so that magnitude
    # is independent of the number of samples captured.
    # The 1e-20 additive term prevents log(0) for bins that are exactly zero
    # (e.g. a perfectly zeroed DC bin on a balanced signal). It sets a
    # numerical floor at 20*log10(1e-20) = -400 dB, well below any physical
    # spectrum — apparent "floor" values near -400 dB should be interpreted
    # as empty bins, not real noise.
    magnitudes_db = 20 * np.log10(np.abs(fft_data) + 1e-20) - 20 * np.log10(N)

    # Apply additional normalization on top of the length correction
    if normalization == FFTNormalization.POWER:
        pass  # Length normalization already applied above

    elif normalization == FFTNormalization.DBFS:
        if full_scale is None:
            raise ValueError("full_scale must be provided for dBFS normalization")
        # Normalize to full scale (subtract the full-scale reference level)
        magnitudes_db = magnitudes_db - 20 * np.log10(full_scale / 2)

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

    # Calculate where this harmonic would appear due to aliasing.
    # Each time a frequency crosses fs/2 (Nyquist) it reflects back;
    # crossing fs wraps it to baseband. An even number of folds lands
    # on f % fs; an odd number mirrors it to fs - (f % fs).
    # The final check handles the edge case where the mirrored result
    # still exceeds Nyquist (e.g. f_harm is a multiple of fs).
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
                   tol: float = 0.1,
                   verbose: bool = False) -> List[Tuple[float, float]]:
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
            if verbose:
                print(f"Harmonic {n}: True freq = {n * f0 / 1e3:.1f} kHz, "
                      f"Aliased freq = {harm_freq / 1e3:.1f} kHz, "
                      f"Magnitude = {harm_mag:.1f} dB")
        except ValueError as e:
            if verbose:
                print(f"Warning: {e}")

    return harmonics
def demo_fft_analysis():
    """Demonstrate FFT analysis functions with various test signals"""
    import matplotlib.pyplot as plt
    from pyDataconverter.utils.visualizations.fft_plots import plot_fft

    # Test parameters
    fs = 1e6  # 1 MHz sampling rate
    NFFT = 1024
    NFIN = 11

    # 1. Single tone with harmonics
    print("\nDemo 1: Single tone with harmonics")
    signal1, f0 = generate_coherent_sine(fs, NFFT, NFIN, amplitude=1.0)
    signal2 = generate_sine(2 * f0, fs, amplitude=0.1, duration=NFFT / fs)
    signal3 = generate_sine(3 * f0, fs, amplitude=0.05, duration=NFFT / fs)
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
