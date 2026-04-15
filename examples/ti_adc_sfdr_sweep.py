"""
TI-ADC SFDR Sweep
=================

Three-panel figure: SFDR vs. mismatch RMS for offset, gain, and timing-skew
mismatches in a 4-channel 12-bit TI-ADC. A 60 dB reference line is drawn
on each panel; the printed summary reports the crossing point.

Run with:

    PYTHONPATH=. python examples/ti_adc_sfdr_sweep.py
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
N_BITS  = 12
V_REF   = 1.0
M       = 4
FS      = 1e9
N_FFT   = 2**13    # 8192 — faster than 2^14
N_FIN   = 257      # prime
AMPLITUDE = 0.4
DC_OFFSET = 0.5
SEED    = 42
N_SWEEP = 12       # points per curve
SFDR_REF_DB = 60.0

LSB = V_REF / (2**N_BITS)   # ≈ 244 µV


def _build_template() -> SimpleADC:
    return SimpleADC(n_bits=N_BITS, v_ref=V_REF, input_type=InputType.SINGLE)


def _measure_sfdr(ti: TimeInterleavedADC, signal: np.ndarray) -> float:
    codes = np.array([ti.convert(float(v)) for v in signal])
    voltages = codes.astype(float) / (2**N_BITS - 1) * V_REF
    metrics = calculate_adc_dynamic_metrics(
        time_data=voltages, fs=FS, full_scale=V_REF
    )
    return float(metrics["SFDR"])


def _sweep(param: str, magnitudes: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """
    For each scalar magnitude, build a TimeInterleavedADC with that stddev
    (seeded), run the signal, return SFDR array.
    """
    sfdr_vals = np.zeros(len(magnitudes))
    rng = np.random.default_rng(SEED)
    for i, mag in enumerate(magnitudes):
        offsets = rng.standard_normal(M) * mag
        if param == "offset":
            ti = TimeInterleavedADC(M, _build_template(), fs=FS, offset=offsets)
        elif param == "gain":
            ti = TimeInterleavedADC(M, _build_template(), fs=FS,
                                    gain_error=rng.standard_normal(M) * mag)
        else:  # skew
            ti = TimeInterleavedADC(M, _build_template(), fs=FS,
                                    timing_skew=rng.standard_normal(M) * mag)
        sfdr_vals[i] = _measure_sfdr(ti, signal)
    return sfdr_vals


def _crossing(x_vals: np.ndarray, sfdr_vals: np.ndarray) -> float | None:
    """Return first x where SFDR drops below SFDR_REF_DB, or None."""
    below = np.where(sfdr_vals < SFDR_REF_DB)[0]
    return float(x_vals[below[0]]) if len(below) else None


def main():
    signal, _ = generate_coherent_sine(FS, N_FFT, N_FIN, AMPLITUDE, DC_OFFSET)

    # Sweep ranges
    offset_range = np.linspace(0, 5 * LSB, N_SWEEP)        # 0 → ~1.2 mV
    gain_range   = np.linspace(0, 0.005,    N_SWEEP)        # 0 → 0.5%
    skew_range   = np.linspace(0, 500e-15,  N_SWEEP)        # 0 → 500 fs

    print("Running offset sweep ...")
    sfdr_off  = _sweep("offset", offset_range, signal)
    print("Running gain sweep ...")
    sfdr_gain = _sweep("gain",   gain_range,   signal)
    print("Running skew sweep ...")
    sfdr_skew = _sweep("skew",   skew_range,   signal)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle(f"TI-ADC SFDR vs. Mismatch (M={M}, {N_BITS}-bit)", fontsize=12)

    datasets = [
        (axes[0], offset_range / LSB,     sfdr_off,  "Offset RMS (LSB)",   "offset"),
        (axes[1], gain_range * 100,        sfdr_gain, "Gain RMS (%)",       "gain"),
        (axes[2], skew_range * 1e15,       sfdr_skew, "Skew RMS (fs)",      "skew"),
    ]

    for ax, x, sfdr, xlabel, name in datasets:
        ax.plot(x, sfdr, "o-", color="C0", linewidth=1.5)
        ax.axhline(SFDR_REF_DB, color="C3", linestyle="--", linewidth=1.0,
                   label=f"{SFDR_REF_DB:.0f} dB reference")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SFDR (dB)")
        ax.set_title(name.capitalize() + " mismatch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    y_min = min(sfdr_off.min(), sfdr_gain.min(), sfdr_skew.min()) - 5
    y_max = max(sfdr_off.max(), sfdr_gain.max(), sfdr_skew.max()) + 5
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig.savefig("ti_adc_sfdr_sweep.png", dpi=150)
    print("Saved ti_adc_sfdr_sweep.png")

    # Printed summary
    cross_off  = _crossing(offset_range / LSB, sfdr_off)
    cross_gain = _crossing(gain_range * 100,    sfdr_gain)
    cross_skew = _crossing(skew_range * 1e15,   sfdr_skew)
    print(f"SFDR crosses {SFDR_REF_DB:.0f} dB at:")
    print(f"  Offset: {f'{cross_off:.2f} LSB' if cross_off else 'not reached'}")
    print(f"  Gain:   {f'{cross_gain:.3f}%'   if cross_gain else 'not reached'}")
    print(f"  Skew:   {f'{cross_skew:.0f} fs' if cross_skew else 'not reached'}")


if __name__ == "__main__":
    main()
