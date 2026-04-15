"""
TI-ADC ENOB vs. Input Frequency (Bandwidth Mismatch)
=====================================================

Two-curve plot: ideal (flat) vs. bandwidth-mismatched TI-ADC. Bandwidth
mismatch degrades ENOB at high input frequencies while low-frequency ENOB
is unaffected.

Run with:

    PYTHONPATH=. python examples/ti_adc_bandwidth_sweep.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly; comment out for interactive use
import matplotlib.pyplot as plt

from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine
from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics

# ── Configuration ─────────────────────────────────────────────────────────────
N_BITS    = 12
V_REF     = 1.0
M         = 4
FS        = 1e9
N_FFT     = 2**13
AMPLITUDE = 0.4
DC_OFFSET = 0.5
BW_ARRAY  = np.array([2e8, 3e8, 1.5e8, 2.5e8])   # Hz, one per channel
BW_MEAN   = float(np.mean(BW_ARRAY))               # 250 MHz
N_FREQS   = 15
F_MIN     = 5e6
F_MAX     = 480e6


def _build_template() -> SimpleADC:
    return SimpleADC(n_bits=N_BITS, v_ref=V_REF, input_type=InputType.SINGLE)


def _find_nfin(f_target: float) -> int:
    """Nearest odd integer bin to f_target within N_FFT."""
    n = round(f_target * N_FFT / FS)
    n = max(1, min(n, N_FFT // 2 - 1))
    return n if n % 2 == 1 else n + 1


def _measure_enob(ti: TimeInterleavedADC, signal: np.ndarray,
                  use_waveform: bool) -> float:
    if use_waveform:
        t = np.arange(N_FFT) / FS
        codes = ti.convert_waveform(signal, t)
    else:
        codes = np.array([ti.convert(float(v)) for v in signal])
    voltages = codes.astype(float) / (2**N_BITS - 1) * V_REF
    metrics = calculate_adc_dynamic_metrics(
        time_data=voltages, fs=FS, full_scale=V_REF
    )
    return float(metrics["ENOB"])


def main():
    f_targets = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQS)

    ideal_ti = TimeInterleavedADC(M, _build_template(), fs=FS)
    mismatch_ti = TimeInterleavedADC(M, _build_template(), fs=FS,
                                     bandwidth=BW_ARRAY)

    enob_ideal    = np.zeros(N_FREQS)
    enob_mismatch = np.zeros(N_FREQS)

    for i, f_target in enumerate(f_targets):
        nfin = _find_nfin(f_target)
        signal, f_actual = generate_coherent_sine(
            FS, N_FFT, nfin, AMPLITUDE, DC_OFFSET
        )
        ideal_ti.reset()
        mismatch_ti.reset()
        enob_ideal[i]    = _measure_enob(ideal_ti,    signal, use_waveform=False)
        enob_mismatch[i] = _measure_enob(mismatch_ti, signal, use_waveform=True)
        print(f"  f_in={f_actual/1e6:6.1f} MHz  ->  "
              f"ideal ENOB={enob_ideal[i]:.2f}  "
              f"mismatch ENOB={enob_mismatch[i]:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(f_targets / 1e6, enob_ideal,    "o-", color="silver",
            linewidth=1.5, label="Ideal (zero BW mismatch)")
    ax.plot(f_targets / 1e6, enob_mismatch, "s-", color="C0",
            linewidth=1.5, label="Bandwidth mismatch")
    ax.axvline(BW_MEAN / 1e6, color="C3", linestyle="--", linewidth=1.0)

    # Use data-derived y position to ensure the annotation is visible
    y_pos = min(enob_mismatch.min(), enob_ideal.min()) - 0.3
    ax.text(BW_MEAN / 1e6 + 3, y_pos, "mean BW cutoff",
            color="C3", fontsize=9)

    ax.set_xlabel("Input Frequency (MHz)")
    ax.set_ylabel("ENOB (bits)")
    ax.set_title(f"TI-ADC ENOB vs. f_in -- Bandwidth Mismatch (M={M}, {N_BITS}-bit)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("ti_adc_bandwidth_sweep.png", dpi=150)
    print("Saved ti_adc_bandwidth_sweep.png")

    # Printed summary
    idx_lo = int(np.argmin(np.abs(f_targets - 10e6)))
    idx_hi = int(np.argmin(np.abs(f_targets - 400e6)))
    print(f"\nSummary:")
    print(f"  f_in ~= {f_targets[idx_lo]/1e6:.0f} MHz: "
          f"ideal={enob_ideal[idx_lo]:.2f}, mismatch={enob_mismatch[idx_lo]:.2f}")
    print(f"  f_in ~= {f_targets[idx_hi]/1e6:.0f} MHz: "
          f"ideal={enob_ideal[idx_hi]:.2f}, mismatch={enob_mismatch[idx_hi]:.2f}")


if __name__ == "__main__":
    main()
