# TI-ADC Examples & Quickstart Documentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four focused, runnable TI-ADC example scripts and update `docs/quickstart.md` with a complete TimeInterleavedADC section.

**Architecture:** Five independent file tasks: four new `examples/ti_adc_*.py` scripts plus one `docs/quickstart.md` edit. No changes to library source code. All examples are self-contained (~80–120 lines), headless (Agg backend), and runnable from the repo root with `PYTHONPATH=. python examples/<file>.py`.

**Tech Stack:** Python 3, numpy, matplotlib (Agg), scipy (bandwidth example), `pyDataconverter` (SimpleADC, FlashADC, TimeInterleavedADC, generate_coherent_sine, calculate_adc_dynamic_metrics).

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `examples/ti_adc_spurs.py` | 2×2 grid: one panel per mismatch type with ideal overlay and spur annotation |
| Create | `examples/ti_adc_sfdr_sweep.py` | 1×3 subplots: SFDR vs. mismatch severity for offset / gain / skew |
| Create | `examples/ti_adc_bandwidth_sweep.py` | ENOB vs. f_in — ideal flat vs. bandwidth-mismatched curve |
| Create | `examples/ti_adc_hierarchical.py` | Outer-level vs. inner-level offset spur on a 2-level tree |
| Modify | `docs/quickstart.md` | Insert `## TimeInterleavedADC` section; update `## Next Steps` bullets |

---

## Task 1: `ti_adc_spurs.py` — Spur anatomy (2×2 grid)

**Files:**
- Create: `examples/ti_adc_spurs.py`

**Concept:** Show the four canonical TI-ADC mismatch spur types in one figure. Each panel has an ideal overlay (grey) and a dashed vertical line at the expected spur frequency.

- [ ] **Step 1: Write the script**

```python
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
matplotlib.use("Agg")
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
    t = np.arange(N_FFT) / FS

    # Ideal spectrum (same ADC, zero mismatch)
    ideal_ti = TimeInterleavedADC(_build_template(), M=M, fs=FS)
    ideal_codes = _run_pointwise(ideal_ti, signal)
    freqs, ideal_db = _spectrum_db(ideal_codes)

    f_spur_offset = FS / M                    # 250 MHz  (offset tone)
    f_spur_image  = FS / M - f_in             # ≈219 MHz (gain/skew/BW image)

    panels = [
        ("Offset only",    "Offset spur @ fs/M",      f_spur_offset,
         TimeInterleavedADC(_build_template(), M=M, fs=FS,
                            offset=OFFSET_MISMATCH), _run_pointwise),
        ("Gain only",      "Gain image @ fs/M−f_in",  f_spur_image,
         TimeInterleavedADC(_build_template(), M=M, fs=FS,
                            gain_error=GAIN_MISMATCH), _run_pointwise),
        ("Timing skew only", "Skew image @ fs/M−f_in", f_spur_image,
         TimeInterleavedADC(_build_template(), M=M, fs=FS,
                            timing_skew=SKEW_MISMATCH), _run_pointwise),
        ("Bandwidth only", "BW image @ fs/M−f_in",    f_spur_image,
         TimeInterleavedADC(_build_template(), M=M, fs=FS,
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
```

- [ ] **Step 2: Run the script and verify output**

```bash
PYTHONPATH=. python examples/ti_adc_spurs.py
```

Expected: Four lines like `Offset only         : spur @ 250.0 MHz → -XX.X dBFS`, file `ti_adc_spurs.png` created. No errors.

- [ ] **Step 3: Commit**

```bash
git add examples/ti_adc_spurs.py ti_adc_spurs.png
git commit -m "feat(examples): add ti_adc_spurs.py — spur anatomy 2x2 grid"
```

---

## Task 2: `ti_adc_sfdr_sweep.py` — SFDR vs. mismatch severity

**Files:**
- Create: `examples/ti_adc_sfdr_sweep.py`

**Concept:** 1×3 subplot row. Each panel sweeps one mismatch parameter from zero to a large value and plots SFDR (dB). Horizontal reference line at 60 dB. Printed output identifies the mismatch level at which SFDR crosses below 60 dB.

- [ ] **Step 1: Write the script**

```python
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
matplotlib.use("Agg")
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
            ti = TimeInterleavedADC(_build_template(), M=M, fs=FS, offset=offsets)
        elif param == "gain":
            ti = TimeInterleavedADC(_build_template(), M=M, fs=FS,
                                    gain_error=rng.standard_normal(M) * mag)
        else:  # skew
            ti = TimeInterleavedADC(_build_template(), M=M, fs=FS,
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
```

- [ ] **Step 2: Run the script and verify output**

```bash
PYTHONPATH=. python examples/ti_adc_sfdr_sweep.py
```

Expected: "Running offset sweep ...", "Running gain sweep ...", "Running skew sweep ...", SFDR crossing summary, "Saved ti_adc_sfdr_sweep.png". No errors.

- [ ] **Step 3: Commit**

```bash
git add examples/ti_adc_sfdr_sweep.py ti_adc_sfdr_sweep.png
git commit -m "feat(examples): add ti_adc_sfdr_sweep.py — SFDR vs. mismatch severity"
```

---

## Task 3: `ti_adc_bandwidth_sweep.py` — ENOB vs. input frequency

**Files:**
- Create: `examples/ti_adc_bandwidth_sweep.py`

**Concept:** Two curves on one plot — ideal (flat ENOB) vs. bandwidth-mismatched. Shows ENOB degrades as f_in approaches and exceeds the per-channel BW cutoffs. Vertical annotation at mean BW cutoff (250 MHz).

**Important:** Bandwidth mismatch requires `convert_waveform`. For each input frequency, compute `n_fin = round(f_target * N_FFT / FS)` ensuring it's odd (prime-like), build a dense time vector, call `ti.convert_waveform(signal, t)`.

- [ ] **Step 1: Write the script**

```python
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
matplotlib.use("Agg")
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

    ideal_ti = TimeInterleavedADC(_build_template(), M=M, fs=FS)
    mismatch_ti = TimeInterleavedADC(_build_template(), M=M, fs=FS,
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
        print(f"  f_in={f_actual/1e6:6.1f} MHz  →  "
              f"ideal ENOB={enob_ideal[i]:.2f}  "
              f"mismatch ENOB={enob_mismatch[i]:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(f_targets / 1e6, enob_ideal,    "o-", color="silver",
            linewidth=1.5, label="Ideal (zero BW mismatch)")
    ax.plot(f_targets / 1e6, enob_mismatch, "s-", color="C0",
            linewidth=1.5, label="Bandwidth mismatch")
    ax.axvline(BW_MEAN / 1e6, color="C3", linestyle="--", linewidth=1.0)
    ax.text(BW_MEAN / 1e6 + 3, ax.get_ylim()[0] + 0.5, "mean BW cutoff",
            color="C3", fontsize=9)
    ax.set_xlabel("Input Frequency (MHz)")
    ax.set_ylabel("ENOB (bits)")
    ax.set_title(f"TI-ADC ENOB vs. f_in — Bandwidth Mismatch (M={M}, {N_BITS}-bit)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("ti_adc_bandwidth_sweep.png", dpi=150)
    print("Saved ti_adc_bandwidth_sweep.png")

    # Printed summary
    # Find indices closest to 10 MHz and 400 MHz
    idx_lo = int(np.argmin(np.abs(f_targets - 10e6)))
    idx_hi = int(np.argmin(np.abs(f_targets - 400e6)))
    print(f"\nSummary:")
    print(f"  f_in ≈ {f_targets[idx_lo]/1e6:.0f} MHz: "
          f"ideal={enob_ideal[idx_lo]:.2f}, mismatch={enob_mismatch[idx_lo]:.2f}")
    print(f"  f_in ≈ {f_targets[idx_hi]/1e6:.0f} MHz: "
          f"ideal={enob_ideal[idx_hi]:.2f}, mismatch={enob_mismatch[idx_hi]:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

```bash
PYTHONPATH=. python examples/ti_adc_bandwidth_sweep.py
```

Expected: 15 frequency lines printed, summary showing lower ENOB at 400 MHz for the mismatched curve vs. flat ideal, file `ti_adc_bandwidth_sweep.png` created. No errors.

- [ ] **Step 3: Commit**

```bash
git add examples/ti_adc_bandwidth_sweep.py ti_adc_bandwidth_sweep.png
git commit -m "feat(examples): add ti_adc_bandwidth_sweep.py — ENOB vs. f_in"
```

---

## Task 4: `ti_adc_hierarchical.py` — Outer vs. inner mismatch

**Files:**
- Create: `examples/ti_adc_hierarchical.py`

**Concept:** In a `[4, 2]` = 8-channel hierarchy, outer-level offset (σ=1 mV) places a spur at `fs/4 = 500 MHz`, inner-level offset (same σ) places a spur at `fs/8 = 250 MHz`. Three spectra on one axes: ideal (grey), outer-only, inner-only. Demonstrates tree-controlled spur placement.

**Key API:** `TimeInterleavedADC.hierarchical(channels_per_level=[4, 2], sub_adc_template=template, fs=2e9, offset_std_per_level=[σ_outer, σ_inner], seed=7)`. The `offset_std_per_level` parameter controls mismatch independently at each level.

- [ ] **Step 1: Write the script**

```python
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
matplotlib.use("Agg")
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


def _peak_near(freqs, spec_db, target_hz, window_hz=30e6) -> tuple[float, float]:
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
```

- [ ] **Step 2: Run the script and verify output**

```bash
PYTHONPATH=. python examples/ti_adc_hierarchical.py
```

Expected: Print showing f_in, expected spur locations, file `ti_adc_hierarchical.png` created, spur summary with outer spur near 500 MHz and inner spur near 250 MHz. No errors.

- [ ] **Step 3: Commit**

```bash
git add examples/ti_adc_hierarchical.py ti_adc_hierarchical.png
git commit -m "feat(examples): add ti_adc_hierarchical.py — outer vs. inner mismatch"
```

---

## Task 5: Update `docs/quickstart.md`

**Files:**
- Modify: `docs/quickstart.md` (insert new section before `## Putting It All Together`; update `## Next Steps`)

**Concept:** Add a `## TimeInterleavedADC` section that a new user can read to understand the TI-ADC API without consulting the full API reference. Then update the Next Steps bullet list to include all five TI-ADC example scripts.

**Existing structure (reference):**
- Line 595 begins: `## Putting It All Together`
- Line 637 begins: `## Next Steps`
- The new section must be inserted immediately before `## Putting It All Together`

- [ ] **Step 1: Insert the TI-ADC section into quickstart.md**

Find the exact line `## Putting It All Together` (preceded by a `---` separator) and insert the following block **before** it:

```markdown
## TimeInterleavedADC

A `TimeInterleavedADC` runs M sub-ADCs in round-robin rotation to achieve
a combined sampling rate M× higher than any single converter. It handles
offset, gain, timing-skew, and bandwidth mismatches — the four dominant
error sources in real TI-ADC designs.

### Basic construction

```python
import numpy as np
from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType

template = FlashADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE)
ti = TimeInterleavedADC(template, M=4, fs=1e9)

codes = np.array([ti.convert(v) for v in signal])
```

Each call to `convert()` rotates to the next sub-ADC automatically. All M
sub-ADCs are deep-copied from the template so they are independent.

### Mismatch

Pass per-channel arrays for offset, gain error, or timing skew, or a scalar
standard deviation (seeded for reproducibility):

```python
ti = TimeInterleavedADC(
    template, M=4, fs=1e9,
    offset=np.array([1e-3, -1e-3, 0.5e-3, -0.5e-3]),   # V per channel
    gain_error=1e-3,                                       # scalar std, seed=42
    timing_skew=np.array([5e-13, -5e-13, 2e-13, -2e-13]),# s per channel
    seed=42,
)
```

Scalar mismatches are expanded to random per-channel values drawn from
`N(0, σ)` using the given seed.

### Bandwidth mismatch requires `convert_waveform`

Bandwidth mismatch models a first-order low-pass filter inside each sub-ADC
channel. The filter needs dvdt (time-derivative of the input), so you must
supply a time vector via the waveform path:

```python
t = np.arange(len(signal)) / fs
codes = ti.convert_waveform(signal, t)   # returns np.ndarray of int
```

Calling `convert()` when bandwidth mismatch is active raises a `RuntimeError`.

### Hierarchical interleaving

```python
ti = TimeInterleavedADC.hierarchical(
    channels_per_level=[4, 2],   # outer M=4, inner M=2 → 8 total channels
    sub_adc_template=template,
    fs=2e9,
    offset_std_per_level=[1e-3, 0.0],  # mismatch at outer level only
    seed=7,
)
```

Outer-level offset mismatches produce spurs at `fs/4`; inner-level mismatches
produce spurs at `fs/8`.

---
```

- [ ] **Step 2: Update `## Next Steps` in quickstart.md**

Replace the existing Next Steps bullet list body with:

```markdown
- See `examples/` for complete runnable scripts:
  - `simple_adc_example.py` — ADC non-ideality sweeps
  - `simple_dac_example.py` — DAC non-ideality and spectrum examples
  - `flash_adc_example.py` — Flash ADC with non-idealities and animation
  - `sar_adc_example.py` — SAR ADC with C-DAC mismatch, static/dynamic metrics, and visualization
  - `ti_adc_example.py` — TI-ADC spectrum with offset, gain, and skew spurs
  - `ti_adc_spurs.py` — spur anatomy for all four mismatch types
  - `ti_adc_sfdr_sweep.py` — SFDR degradation vs. mismatch magnitude
  - `ti_adc_bandwidth_sweep.py` — ENOB vs. f_in under bandwidth mismatch
  - `ti_adc_hierarchical.py` — outer vs. inner mismatch in a 2-level tree
- See `docs/api_reference.md` for full parameter documentation on every class and function.
```

- [ ] **Step 3: Verify quickstart.md renders correctly**

```bash
grep -n "## TimeInterleavedADC\|## Putting It All Together\|## Next Steps\|ti_adc" docs/quickstart.md
```

Expected output should show:
- `## TimeInterleavedADC` before `## Putting It All Together`
- All five `ti_adc_*.py` names in the Next Steps section

- [ ] **Step 4: Commit**

```bash
git add docs/quickstart.md
git commit -m "docs(quickstart): add TimeInterleavedADC section and update Next Steps"
```

---

## Self-Review Against Spec

### Spec coverage

| Spec requirement | Covered by task |
|---|---|
| `ti_adc_spurs.py` — 2×2 grid, 4 mismatch types, ideal overlay, dashed line, text label | Task 1 ✓ |
| `ti_adc_sfdr_sweep.py` — 1×3 subplots, SFDR vs. mismatch, 60 dB reference | Task 2 ✓ |
| `ti_adc_bandwidth_sweep.py` — ENOB vs. f_in, ideal vs. mismatched, mean BW annotation | Task 3 ✓ |
| `ti_adc_hierarchical.py` — 3 spectra, outer/inner spur placement | Task 4 ✓ |
| Quickstart: TI-ADC section before "Putting It All Together" | Task 5 ✓ |
| Quickstart: Next Steps updated with all 5 TI-ADC examples | Task 5 ✓ |
| All examples: `matplotlib.use("Agg")` headless | All tasks ✓ |
| All examples: `PYTHONPATH=. python examples/<file>.py` | All tasks ✓ |
| All examples: module docstring with `Run with:` block | All tasks ✓ |
| All examples: `calculate_adc_dynamic_metrics(time_data=..., fs=..., full_scale=...)` | Tasks 2, 3 ✓ |
| Spurs example: printed spur summary per panel | Task 1 ✓ |
| SFDR sweep: 12-point sweep, printed crossing summary | Task 2 ✓ |
| BW sweep: 15 frequencies, printed ENOB at 10 MHz and 400 MHz | Task 3 ✓ |
| Hierarchical: printed spur magnitude and location | Task 4 ✓ |

### Placeholder scan: None found.

### Type consistency

- `TimeInterleavedADC(template, M=M, fs=FS, ...)` — consistent with Phase 1 implementation
- `TimeInterleavedADC.hierarchical(channels_per_level=[...], sub_adc_template=..., fs=..., offset_std_per_level=[...], seed=...)` — consistent with Phase 1 `hierarchical` classmethod signature
- `ti.convert_waveform(signal, t)` — consistent with Phase 1 implementation (signal first, t second)
- `calculate_adc_dynamic_metrics(time_data=..., fs=..., full_scale=...)` — consistent with spec and pipelined example
