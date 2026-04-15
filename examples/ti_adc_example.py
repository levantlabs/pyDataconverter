"""
Time-Interleaved ADC Example
============================

Constructs a 4-channel 10-bit TI-ADC with explicit offset, gain, and
timing-skew mismatches, drives a coherent sine through it, and plots
the output time-series + spectrum. The spectrum should clearly show
mismatch spurs at fs/M, fs/M +/- f_in.

Run with:

    PYTHONPATH=. python examples/ti_adc_example.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly; comment out for interactive use
import matplotlib.pyplot as plt

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType


def _compute_spectrum(codes: np.ndarray, fs: float):
    """FFT spectrum of a code sequence, normalised to dBFS."""
    x = codes.astype(float) - np.mean(codes)
    spec = np.abs(np.fft.rfft(x))
    spec_norm = spec / (len(x) / 2)
    spec_db = 20 * np.log10(spec_norm + 1e-30)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, spec_db


def build_demo_ti_adc(fs: float = 1e9) -> TimeInterleavedADC:
    """4-channel 10-bit TI-ADC with explicit offset/gain/skew mismatches."""
    template = FlashADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE)
    offsets = np.array([2e-3, -2e-3, 1e-3, -1e-3])     # +/- 2 mV
    gains   = np.array([0.0015, -0.0015, 0.0008, -0.0008])  # +/- 0.15%
    skews   = np.array([3e-12, -3e-12, 1e-12, -1e-12])  # +/- 3 ps
    return TimeInterleavedADC(
        channels=4,
        sub_adc_template=template,
        fs=fs,
        offset=offsets,
        gain_error=gains,
        timing_skew=skews,
    )


def main():
    fs = 1e9
    n_fft = 2 ** 14
    adc = build_demo_ti_adc(fs=fs)

    # Coherent sine centred at v_ref/2 = 0.5 (the unipolar SimpleADC/FlashADC
    # midscale) with amplitude 0.4 so the swing is [0.1, 0.9] — safely within
    # the [0, v_ref] input range. The DC offset does not affect mismatch
    # spurs (they depend on amplitude only), so the spectrum annotation
    # shows the same spur locations as the bipolar analysis in the spec.
    n_fin_bins = 511
    f_in = n_fin_bins * fs / n_fft
    t = np.arange(n_fft) / fs
    v_in = 0.5 + 0.4 * np.sin(2 * np.pi * f_in * t)
    dvdt_in = 2 * np.pi * f_in * 0.4 * np.cos(2 * np.pi * f_in * t)

    # Pointwise path (offset/gain/skew only; no bandwidth here)
    codes = np.array([adc.convert(float(v_in[i]), dvdt=float(dvdt_in[i]))
                      for i in range(n_fft)], dtype=int)

    print("Time-Interleaved ADC demo (4-channel 10-bit)")
    print(f"  fs            = {fs/1e6:.1f} MHz")
    print(f"  f_in          = {f_in/1e6:.2f} MHz")
    print(f"  n_fft         = {n_fft}")
    print(f"  code range    = [{codes.min()}, {codes.max()}]")
    print(f"  output mean   = {codes.mean():.1f} (ideal ~512 for midscale)")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Time-series: first 64 samples
    axes[0].plot(codes[:64], 'o-', color="#4a9eff", markersize=4)
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("output code")
    axes[0].set_title("First 64 output samples (4-channel interleaving visible)")
    axes[0].grid(True, linestyle=":")

    # Spectrum with annotated spur locations
    freqs, mag_db = _compute_spectrum(codes, fs)
    axes[1].plot(freqs / 1e6, mag_db, color="#f4a261", linewidth=0.8)
    axes[1].set_xlabel("frequency (MHz)")
    axes[1].set_ylabel("magnitude (dBFS)")
    axes[1].set_title(
        f"Output spectrum — mismatch spurs visible at fs/M = {fs/4/1e6:.0f} MHz and fs/M ± f_in")
    axes[1].grid(True, linestyle=":")

    # Annotate the primary mismatch spurs
    for k in range(1, 4):
        axes[1].axvline(k * fs / 4 / 1e6, color="red", linestyle="--",
                         alpha=0.3, linewidth=0.7)
    axes[1].axvline((fs / 4 - f_in) / 1e6, color="purple", linestyle=":",
                     alpha=0.5, linewidth=0.7)
    axes[1].axvline((fs / 4 + f_in) / 1e6, color="purple", linestyle=":",
                     alpha=0.5, linewidth=0.7)

    plt.tight_layout()
    plt.savefig("ti_adc_spectrum.png", dpi=150)
    print("\nWrote ti_adc_spectrum.png")


if __name__ == "__main__":
    main()
