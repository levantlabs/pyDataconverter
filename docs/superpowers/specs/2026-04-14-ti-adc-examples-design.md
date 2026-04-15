# TI-ADC Examples & Quickstart Documentation — Design Spec

**Date:** 2026-04-14
**Scope:** Four new example scripts and a `quickstart.md` TI-ADC section. No changes to core library code.

---

## Goal

Give engineers four focused, runnable examples that each teach one concrete TI-ADC concept, plus update `docs/quickstart.md` so a new user can find and understand the TI-ADC API without reading the full API reference.

---

## New files

| File | Concept |
|---|---|
| `examples/ti_adc_spurs.py` | Spur anatomy — one panel per mismatch type |
| `examples/ti_adc_sfdr_sweep.py` | SFDR vs. mismatch severity (budget curves) |
| `examples/ti_adc_bandwidth_sweep.py` | ENOB vs. input frequency under bandwidth mismatch |
| `examples/ti_adc_hierarchical.py` | Outer vs. inner mismatch in a 2-level tree |

Modified:

| File | Change |
|---|---|
| `docs/quickstart.md` | Add TI-ADC section + update Next Steps bullets |

---

## Example 1 — `ti_adc_spurs.py`

**Concept:** The four canonical TI-ADC mismatch spur types in one figure.

**Configuration:** 12-bit `SimpleADC` template, M=4, fs=1 GHz, n_fft=2¹⁴, n_fin=511, amplitude=0.4, DC offset=0.5 V (unipolar midscale).

**Layout:** 2×2 subplot grid.

| Panel | Active mismatch | Spur location | Annotation |
|---|---|---|---|
| Top-left | Offset only | `fs/M = 250 MHz` | "Offset spur @ fs/M" |
| Top-right | Gain only | `fs/M − f_in ≈ 219 MHz` | "Gain image @ fs/M−f_in" |
| Bottom-left | Timing skew only | `fs/M − f_in ≈ 219 MHz` | "Skew image @ fs/M−f_in" |
| Bottom-right | Bandwidth only | `fs/M − f_in ≈ 219 MHz` | "BW image @ fs/M−f_in" |

Each panel overlays:
- The **ideal spectrum** (grey, zero mismatch, same ADC) so the spur is visible by contrast.
- A **vertical dashed line** at the expected spur frequency.
- A **text label** giving the closed-form formula for that spur's magnitude.

The bandwidth panel uses `convert_waveform`; the others use the pointwise `convert` path.

Mismatch magnitudes chosen so spurs are clearly visible above the noise floor but don't dominate the fundamental:
- Offset: `[1e-3, −1e-3, 0.5e-3, −0.5e-3]` V
- Gain: `[1e-3, −1e-3, 5e-4, −5e-4]` (fractional)
- Skew: `[5e-13, −5e-13, 2e-13, −2e-13]` s
- Bandwidth: `[2e8, 3e8, 1.5e8, 2.5e8]` Hz

Output: `ti_adc_spurs.png`

**Printed summary:** For each panel, one line: `Offset spur: measured -47.2 dB, formula -45.8 dB (diff: 1.4 dB)`.

---

## Example 2 — `ti_adc_sfdr_sweep.py`

**Concept:** As mismatch RMS increases, SFDR degrades. Three curves on one plot.

**Configuration:** 12-bit `SimpleADC`, M=4, fs=1 GHz, n_fft=2¹³ (8192, faster than 2¹⁴), n_fin=257, amplitude=0.4.

**Sweep:** 12 log-spaced mismatch magnitudes per curve:
- **Offset:** σ_off = 0 to 5×LSB (LSB = v_ref/2¹² ≈ 244 µV), so range ≈ [0, 1.2 mV].
- **Gain:** σ_gain = 0 to 0.5% fractional.
- **Skew:** σ_skew = 0 to 500 fs.

At each point, construct a `TimeInterleavedADC` with that scalar stddev (seeded for reproducibility, seed=42), drive a coherent sine, compute `calculate_adc_dynamic_metrics(time_data=codes.astype(float)/max_code * v_ref, fs=fs, full_scale=v_ref)`, extract `SFDR`.

**Plot:** 1×3 subplot row, one panel per mismatch type. Each panel: x-axis in natural units (LSB for offset, % for gain, fs for skew), y-axis SFDR (dB), horizontal reference line at 60 dB. Shared y-axis range so the three panels are visually comparable.

Output: `ti_adc_sfdr_sweep.png`

**Printed summary:** For each curve, the mismatch level at which SFDR crosses 60 dB.

---

## Example 3 — `ti_adc_bandwidth_sweep.py`

**Concept:** Bandwidth mismatch degrades ENOB at high input frequencies while low-frequency ENOB is unaffected.

**Configuration:** 12-bit `SimpleADC`, M=4, fs=1 GHz. Bandwidth array: `[2e8, 3e8, 1.5e8, 2.5e8]` Hz (25% spread around 250 MHz mean). n_fft=2¹³. 15 input frequencies log-spaced from 5 MHz to 480 MHz.

**Two curves per plot:**
1. **Ideal** (zero bandwidth mismatch): ENOB flat across frequency, matches theoretical 12-bit ENOB.
2. **Mismatched** (bandwidth array above): ENOB degrades as f_in approaches and exceeds the LPF cutoffs.

Each point uses `convert_waveform`, extracts codes, calls `calculate_adc_dynamic_metrics` for ENOB. Use coherent sine (n_fin chosen as prime near each frequency target × n_fft/fs).

**Annotation:** A vertical line at the mean bandwidth cutoff (250 MHz) with label "mean BW cutoff".

Output: `ti_adc_bandwidth_sweep.png`

**Printed summary:** ENOB at 10 MHz and 400 MHz for both curves.

---

## Example 4 — `ti_adc_hierarchical.py`

**Concept:** In a 2-level `[4, 2]` = 8-channel hierarchy, outer-level offset mismatches create spurs at `fs/4`, inner-level offset mismatches create spurs at `fs/8` — showing that the tree structure controls spur frequency placement.

**Configuration:** 10-bit `FlashADC` leaf template, M_outer=4 / M_inner=2 → 8 total leaf channels, fs=2 GHz (outer), n_fft=2¹⁴, n_fin=511, amplitude=0.4.

**Three spectra on one axes:**
1. **Ideal** (zero mismatch): grey reference.
2. **Outer-level offset only:** `offset_std_per_level=[1e-3, 0]`, seed=7. Spur expected at `fs/4 = 500 MHz`.
3. **Inner-level offset only:** `offset_std_per_level=[0, 1e-3]`, seed=7. Spur expected at `fs_inner/4 = fs/8 = 250 MHz`.

Both mismatch cases use the same σ = 1 mV so the comparison is fair.

Show both the spectrum and a brief tree diagram in the title or subtitle: `Outer (M=4) → Inner (M=2) → 10-bit FlashADC`.

Output: `ti_adc_hierarchical.png`

**Printed summary:** Measured spur magnitude and location for outer-only and inner-only cases.

---

## Quickstart update (`docs/quickstart.md`)

Add a new section **"## TimeInterleavedADC"** before "## Putting It All Together". Content:

1. **One-paragraph intro:** What TI-ADC is and when to use it (high sample rate via M parallel sub-ADCs).
2. **Minimal construction snippet** (10 lines): build a 4-channel TI-ADC from a FlashADC template, pointwise convert.
3. **Mismatch snippet** (6 lines): show `offset=np.array([...])` and `gain_error=scalar_stddev` with `seed=`.
4. **`convert_waveform` note** (3 lines + snippet): bandwidth mismatch requires the waveform path; `RuntimeError` if you forget.
5. **Hierarchical one-liner** (3 lines + snippet): `TimeInterleavedADC.hierarchical(channels_per_level=[4, 2], ...)`.

Update the **"## Next Steps"** bullet list to include:
- `ti_adc_example.py` — spectrum with offset/gain/skew spurs
- `ti_adc_spurs.py` — spur anatomy for all four mismatch types
- `ti_adc_sfdr_sweep.py` — SFDR degradation vs. mismatch magnitude
- `ti_adc_bandwidth_sweep.py` — ENOB vs. f_in under bandwidth mismatch
- `ti_adc_hierarchical.py` — outer vs. inner mismatch in a 2-level tree

---

## Constraints

- All examples use `matplotlib.use("Agg")` (headless).
- All examples run from the repo root with `PYTHONPATH=. python examples/<file>.py`.
- No new dependencies — `numpy`, `matplotlib`, `scipy` already present.
- `calculate_adc_dynamic_metrics` called as: `calculate_adc_dynamic_metrics(time_data=codes_float, fs=fs, full_scale=v_ref)` where `codes_float = codes.astype(float) / (2**n_bits - 1) * v_ref`.
- Each example is self-contained (~80–120 lines). No shared helper module.
- Style: mirror `pipelined_adc_example.py` — top-level `main()`, helper functions, module docstring with `Run with:` block.

---

## Out of scope

- Calibration/correction algorithms (Phase 2 item).
- Animation or interactive plots.
- Changes to any library source files.
- New test coverage for the examples themselves (examples are not unit-tested).
