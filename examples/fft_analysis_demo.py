"""
FFT analysis demo — windowing, leakage, and frequency-resolution comparisons.

Run::

    python examples/fft_analysis_demo.py

Walks through four scenarios using the helpers in
``pyDataconverter.utils.fft_analysis`` and the plotting utilities in
``pyDataconverter.utils.visualizations.fft_plots``:

    1. Single tone with harmonics (no window)
    2. Window function comparison (none / hann / blackman / hamming)
    3. Spectral leakage (coherent vs non-coherent sampling)
    4. Frequency-resolution comparison across FFT lengths
"""

import matplotlib.pyplot as plt

from pyDataconverter.utils.fft_analysis import (
    compute_fft, find_fundamental, find_harmonics,
)
from pyDataconverter.utils.signal_gen import (
    generate_coherent_sine, generate_sine, generate_two_tone,
)
from pyDataconverter.utils.visualizations.fft_plots import plot_fft


def main() -> None:
    # Test parameters
    fs = 1e6     # 1 MHz sampling rate
    NFFT = 1024
    NFIN = 11
    duration = NFFT / fs

    # 1. Single tone with harmonics
    print("\nDemo 1: Single tone with harmonics")
    signal1, f0 = generate_coherent_sine(fs, NFFT, NFIN, amplitude=1.0)
    signal2 = generate_sine(2 * f0, fs, amplitude=0.1, duration=duration)
    signal3 = generate_sine(3 * f0, fs, amplitude=0.05, duration=duration)
    signal = signal1 + signal2 + signal3

    freqs, mags = compute_fft(signal, fs)

    fund_freq, fund_mag = find_fundamental(freqs, mags, f0, fs)
    print(f"Fundamental: freq = {fund_freq / 1e3:.1f} kHz, "
          f"magnitude = {fund_mag:.1f} dB")
    find_harmonics(freqs, mags, f0, fs, num_harmonics=3)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_fft(freqs, mags, "Single Tone with Harmonics (No Window)",
             max_freq=5 * f0, fig=fig, ax=ax)
    plt.show()

    # 2. Compare no window vs different windows
    print("\nDemo 2: Window function comparison")
    f1, f2 = f0, f0 + fs / NFFT  # adjacent bins
    signal = generate_two_tone(f1, f2, fs, amplitude1=0.5, amplitude2=0.5,
                               duration=duration)

    windows = [None, 'hann', 'blackman', 'hamming']
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    axs = axs.flatten()

    for i, window in enumerate(windows):
        freqs, mags = compute_fft(signal, fs, window=window)
        title = 'No Window' if window is None else f'{window.capitalize()} Window'
        plot_fft(freqs, mags, title, max_freq=5 * f2, min_db=-100,
                 fig=fig, ax=axs[i])

    plt.tight_layout()
    plt.show()

    # 3. Spectral leakage demonstration
    print("\nDemo 3: Spectral Leakage")
    f_leak = f0 * 1.5  # non-coherent frequency
    signal_leak = generate_sine(f_leak, fs, amplitude=1.0, duration=duration)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    for i, window in enumerate([None, 'hann']):
        freqs, mags = compute_fft(signal_leak, fs, window=window)
        title = ('Spectral Leakage - No Window' if window is None
                 else 'Reduced Leakage - Hann Window')
        plot_fft(freqs, mags, title, max_freq=2 * f_leak, min_db=-100,
                 fig=fig, ax=[ax1, ax2][i])
    plt.tight_layout()
    plt.show()

    # 4. Frequency resolution demo
    print("\nDemo 4: Frequency resolution comparison")
    durations = [NFFT / fs, 4 * NFFT / fs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for i, dur in enumerate(durations):
        signal = (generate_sine(f0, fs, amplitude=0.5, duration=dur)
                  + generate_sine(f0 + fs / NFFT / 2, fs,
                                  amplitude=0.5, duration=dur))
        freqs, mags = compute_fft(signal, fs)
        title = f'Duration: {dur * 1e6:.1f} µs\nΔf = {fs / len(signal):.1f} Hz'
        plot_fft(freqs, mags, title, max_freq=2 * f0, min_db=-100,
                 fig=fig, ax=[ax1, ax2][i])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
