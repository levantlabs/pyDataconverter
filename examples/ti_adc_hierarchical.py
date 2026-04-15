"""
TI-ADC Hierarchical Mismatch
=============================

Demonstrates that in a 2-level [4, 2] hierarchy, the spur frequency is
controlled by which level has the mismatch:
  - Outer-level offset (M=4)  →  spur at fs/4  = 500 MHz
  - Inner-level offset (M=2)  →  spur at fs/8  = 250 MHz

Three spectra are overlaid: ideal (grey), outer-only, inner-only.

Run with:

    PYTHONPATH=. python examples/ti_adc_hierarchical.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly; comment out for interactive use
import matplotlib.pyplot as plt

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine

# ── Configuration ─────────────────────────────────────────────────────────────
N_BITS    = 10
V_REF     = 1.0
FS        = 2e9       # 2 GHz (outer sampling rate)
N_FFT     = 2**14
N_FIN     = 511       # prime, coprime with N_FFT
AMPLITUDE = 0.4
DC_OFFSET = 0.5
SIGMA_OFF = 1e-3      # 1 mV offset stddev — same for both levels for fair comparison
SEED      = 7

CHANNELS_PER_LEVEL = [4, 2]   # 8 total leaf channels
F_SPUR_OUTER = FS / CHANNELS_PER_LEVEL[0]          # 500 MHz
F_SPUR_INNER = FS / (CHANNELS_PER_LEVEL[0] * CHANNELS_PER_LEVEL[1])   # 250 MHz


def _build_template() -> FlashADC:
    return FlashADC(n_bits=N_BITS, v_ref=V_REF, input_type=InputType.SINGLE)


def _spectrum_db(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = codes.astype(float) - np.mean(codes)
    spec = np.abs(np.fft.rfft(x)) / (N_FFT / 2)
    spec_db = 20 * np.log10(spec + 1e-30)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FS)
    return freqs, spec_db


def _run(ti: TimeInterleavedADC, signal: np.ndarray) -> np.ndarray:
    return np.array([ti.convert(float(v)) for v in signal])


def _peak_near(freqs, spec_db, target_hz, window_hz=3 * FS / N_FFT) -> tuple[float, float]:
    """Return (peak_dBFS, actual_frequency_Hz) within ±window_hz."""
    mask = np.abs(freqs - target_hz) <= window_hz
    if not mask.any():
        return -200.0, target_hz
    idx = int(np.argmax(spec_db[mask]))
    return float(spec_db[mask][idx]), float(freqs[mask][idx])


def main():
    signal, f_in = generate_coherent_sine(FS, N_FFT, N_FIN, AMPLITUDE, DC_OFFSET)
    print(f"f_in = {f_in/1e6:.2f} MHz, fs = {FS/1e9:.0f} GHz, "
          f"hierarchy = {CHANNELS_PER_LEVEL}")
    print(f"Expected outer spur at {F_SPUR_OUTER/1e6:.0f} MHz, "
          f"inner spur at {F_SPUR_INNER/1e6:.0f} MHz")

    # Ideal
    ideal = TimeInterleavedADC.hierarchical(
        channels_per_level=CHANNELS_PER_LEVEL,
        sub_adc_template=_build_template(),
        fs=FS,
        offset_std_per_level=[0.0, 0.0],
        seed=SEED,
    )
    codes_ideal = _run(ideal, signal)
    freqs, spec_ideal = _spectrum_db(codes_ideal)

    # Outer-level offset only
    outer_only = TimeInterleavedADC.hierarchical(
        channels_per_level=CHANNELS_PER_LEVEL,
        sub_adc_template=_build_template(),
        fs=FS,
        offset_std_per_level=[SIGMA_OFF, 0.0],
        seed=SEED,
    )
    codes_outer = _run(outer_only, signal)
    _, spec_outer = _spectrum_db(codes_outer)

    # Inner-level offset only
    inner_only = TimeInterleavedADC.hierarchical(
        channels_per_level=CHANNELS_PER_LEVEL,
        sub_adc_template=_build_template(),
        fs=FS,
        offset_std_per_level=[0.0, SIGMA_OFF],
        seed=SEED,
    )
    codes_inner = _run(inner_only, signal)
    _, spec_inner = _spectrum_db(codes_inner)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs / 1e6, spec_ideal, color="silver", linewidth=0.8,
            label="Ideal (zero mismatch)", zorder=1)
    ax.plot(freqs / 1e6, spec_outer, color="C0",    linewidth=1.0,
            label=f"Outer offset only (σ={SIGMA_OFF*1e3:.0f} mV)", zorder=2)
    ax.plot(freqs / 1e6, spec_inner, color="C1",    linewidth=1.0,
            label=f"Inner offset only (σ={SIGMA_OFF*1e3:.0f} mV)", zorder=3)
    ax.axvline(F_SPUR_OUTER / 1e6, color="C0", linestyle="--", linewidth=0.9,
               label=f"fs/4 = {F_SPUR_OUTER/1e6:.0f} MHz")
    ax.axvline(F_SPUR_INNER / 1e6, color="C1", linestyle="--", linewidth=0.9,
               label=f"fs/8 = {F_SPUR_INNER/1e6:.0f} MHz")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(
        f"Hierarchical TI-ADC Mismatch\n"
        f"Outer (M={CHANNELS_PER_LEVEL[0]}) → Inner (M={CHANNELS_PER_LEVEL[1]}) "
        f"→ {N_BITS}-bit FlashADC"
    )
    ax.set_xlim(0, FS / 2 / 1e6)
    ax.set_ylim(-120, 10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("ti_adc_hierarchical.png", dpi=150)
    print("Saved ti_adc_hierarchical.png")

    # Printed summary
    outer_peak, outer_freq = _peak_near(freqs, spec_outer, F_SPUR_OUTER)
    inner_peak, inner_freq = _peak_near(freqs, spec_inner, F_SPUR_INNER)
    print(f"\nSpur summary:")
    print(f"  Outer-only: {outer_peak:.1f} dBFS @ {outer_freq/1e6:.1f} MHz "
          f"(expected {F_SPUR_OUTER/1e6:.0f} MHz)")
    print(f"  Inner-only: {inner_peak:.1f} dBFS @ {inner_freq/1e6:.1f} MHz "
          f"(expected {F_SPUR_INNER/1e6:.0f} MHz)")


if __name__ == "__main__":
    main()
