"""
TI-ADC Spur Anatomy
===================

Four-panel figure showing each canonical mismatch spur type for a 4-channel
12-bit TI-ADC. Each panel overlays the ideal (zero-mismatch) spectrum in grey
and marks the expected spur frequency with a dashed vertical line.

Run with:

    PYTHONPATH=. python examples/ti_adc_spurs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly; comment out for interactive use
import matplotlib.pyplot as plt

from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine

# ── Configuration ─────────────────────────────────────────────────────────────
N_BITS = 12
V_REF = 1.0
M = 4
FS = 1e9          # 1 GHz
N_FFT = 2**14
N_FIN = 511       # prime, coprime with N_FFT
AMPLITUDE = 0.4
DC_OFFSET = 0.5   # unipolar midscale

# Mismatch arrays (one value per channel)
OFFSET_MISMATCH = np.array([1e-3, -1e-3, 0.5e-3, -0.5e-3])   # V
GAIN_MISMATCH   = np.array([1e-3, -1e-3, 5e-4,   -5e-4])      # fractional
SKEW_MISMATCH   = np.array([5e-13, -5e-13, 2e-13, -2e-13])    # s
BW_MISMATCH     = np.array([2e8, 3e8, 1.5e8, 2.5e8])          # Hz (BW per channel)


def _build_template() -> SimpleADC:
    return SimpleADC(n_bits=N_BITS, v_ref=V_REF, input_type=InputType.SINGLE)


def _spectrum_db(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = codes.astype(float) - np.mean(codes)
    spec = np.abs(np.fft.rfft(x)) / (N_FFT / 2)
    spec_db = 20 * np.log10(spec + 1e-30)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FS)
    return freqs, spec_db


def _run_pointwise(ti: TimeInterleavedADC, signal: np.ndarray) -> np.ndarray:
    return np.array([ti.convert(float(v)) for v in signal])


def _run_waveform(ti: TimeInterleavedADC, signal: np.ndarray) -> np.ndarray:
    t = np.arange(N_FFT) / FS
    return ti.convert_waveform(signal, t)


def _peak_near(freqs, spec_db, target_hz, window_hz=20e6) -> float:
    """Return the peak dB value within ±window_hz of target_hz."""
    mask = np.abs(freqs - target_hz) <= window_hz
    return float(np.max(spec_db[mask])) if mask.any() else -200.0


def main():
    signal, f_in = generate_coherent_sine(FS, N_FFT, N_FIN, AMPLITUDE, DC_OFFSET)

    # Ideal spectrum (same ADC, zero mismatch)
    ideal_ti = TimeInterleavedADC(M, _build_template(), fs=FS)
    ideal_codes = _run_pointwise(ideal_ti, signal)
    freqs, ideal_db = _spectrum_db(ideal_codes)

    f_spur_offset = FS / M                    # 250 MHz  (offset tone)
    f_spur_image  = FS / M - f_in             # ≈219 MHz (gain/skew/BW image)

    panels = [
        ("Offset only",    "Offset spur @ fs/M",      f_spur_offset,
         TimeInterleavedADC(M, _build_template(), fs=FS,
                            offset=OFFSET_MISMATCH), _run_pointwise),
        ("Gain only",      "Gain image @ fs/M−f_in",  f_spur_image,
         TimeInterleavedADC(M, _build_template(), fs=FS,
                            gain_error=GAIN_MISMATCH), _run_pointwise),
        ("Timing skew only", "Skew image @ fs/M−f_in", f_spur_image,
         TimeInterleavedADC(M, _build_template(), fs=FS,
                            timing_skew=SKEW_MISMATCH), _run_pointwise),
        ("Bandwidth only", "BW image @ fs/M−f_in",    f_spur_image,
         TimeInterleavedADC(M, _build_template(), fs=FS,
                            bandwidth=BW_MISMATCH), _run_waveform),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(f"TI-ADC Spur Anatomy (M={M}, {N_BITS}-bit, fs={FS/1e9:.0f} GHz)", fontsize=13)

    for ax, (title, label, f_spur, ti, runner) in zip(axes.flat, panels):
        codes = runner(ti, signal)
        _, spec_db = _spectrum_db(codes)

        ax.plot(freqs / 1e6, ideal_db, color="silver", linewidth=0.8, label="Ideal")
        ax.plot(freqs / 1e6, spec_db,  color="C0",     linewidth=0.9, label="Mismatched")
        ax.axvline(f_spur / 1e6, color="C3", linestyle="--", linewidth=1.0)
        ax.text(f_spur / 1e6 + 2, np.max(spec_db) - 15, label, color="C3", fontsize=8)

        ax.set_title(title)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(0, FS / 2 / 1e6)
        ax.set_ylim(-120, 10)
        ax.legend(fontsize=7)

        measured = _peak_near(freqs, spec_db, f_spur)
        print(f"{title:20s}: spur @ {f_spur/1e6:.1f} MHz → {measured:.1f} dBFS")

    fig.tight_layout()
    fig.savefig("ti_adc_spurs.png", dpi=150)
    print("Saved ti_adc_spurs.png")


if __name__ == "__main__":
    main()
