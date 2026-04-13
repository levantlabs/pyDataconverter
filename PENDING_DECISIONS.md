# Pending Decisions — Code Review Discussion

Items from the 2026-03-29 review session that require a decision before changes are made.
Items already applied in this session are noted as **[APPLIED]**.

---

## Already Applied This Session

### 14.1 — Gain error uses `dac.v_ref` directly `[APPLIED]`
- **File:** `pyDataconverter/utils/dac_metrics.py`, line 232
- **Change:** Replaced `dac.v_ref` with `ideal_span = (codes[-1] - codes[0]) * lsb` so the
  formula is correct for any subset of codes, not just a full 0→max sweep.

### 15.1 — Aliasing fold logic comment `[APPLIED]`
- **File:** `pyDataconverter/utils/fft_analysis.py`, line 121
- **Change:** Added inline comment explaining the even/odd fold logic in `_get_harmonic`.

### 15.2 — FFT length normalization `[APPLIED]`
- **File:** `pyDataconverter/utils/fft_analysis.py`, lines 71-84
- **Change:** `NONE` and `POWER` modes now both divide by N (`- 20*log10(N)`).
  `DBFS` only additionally subtracts the full-scale term.

---

## Pending Decisions

### 14.2 — Two functions named `calculate_dac_dynamic_metrics`
- **Files:**
  - `pyDataconverter/utils/metrics.py`, line 135 — takes pre-computed `freqs`/`mags`
  - `pyDataconverter/utils/dac_metrics.py`, line 247 — takes raw `voltages`, zone-aware
- **How it is used today:**
  `dac_plots.py` line 252 imports and calls the `metrics.py` version because
  `plot_output_spectrum` already has a zone-filtered FFT and no longer holds the
  original voltage array. The `dac_metrics.py` version is the public full-pipeline API.
- **Decision needed:** Rename the `metrics.py` version to
  `_calculate_dac_dynamic_metrics_from_fft` (private, underscore prefix) to avoid
  confusion, or leave as-is with a clarifying comment?

---

### 15.3 — `demo_fft_analysis` references undefined `duration`
- **File:** `pyDataconverter/utils/fft_analysis.py`, lines 233, 257
- **Bug:** `duration` is used in Demo 2 and Demo 3 of `demo_fft_analysis()` but is never
  defined. Demo 1 is fine (uses `NFFT / fs` inline). Demo 2+ will raise `NameError`
  at runtime if the demo is run.

  ```python
  # Line 233 (Demo 2) — breaks:
  signal = generate_two_tone(f1, f2, fs, amplitude1=0.5, amplitude2=0.5, duration=duration)

  # Line 257 (Demo 3) — breaks:
  signal_leak = generate_sine(f_leak, fs, amplitude=1.0, duration=duration)
  ```

- **Proposed fix:** Add `duration = NFFT / fs` at the top of the function (after the
  `NFFT = 1024` line), consistent with how Demo 1 uses `NFFT / fs` inline.
- **Decision needed:** Apply fix?

---

### 16.1 — `generate_step` ignores `levels[0]` `[APPLIED 2026-04-13]`
- **File:** `pyDataconverter/utils/signal_gen.py`
- **Resolution:** Contract flipped. New rule: `len(levels) == len(step_points) + 1`, `levels[0]` fills `[0, step_points[0])`, `levels[i]` fills `[step_points[i-1], step_points[i])`, and `levels[-1]` fills `[step_points[-1], samples)`. Implementation uses `np.full(samples, levels[0], dtype=float)` followed by the `zip(step_points, levels[1:])` fill loop. Full-length invariant added; non-monotonic and out-of-range step_points raise `ValueError`. Tests `test_initial_level_is_applied` and friends lock the new contract.

---

### 16.2 — `generate_digital_step` has the same initial-level bug `[APPLIED 2026-04-13]`
- **File:** `pyDataconverter/utils/signal_gen.py`
- **Resolution:** Fixed jointly with 16.1 and 16.3 by redesigning the whole function. New signature `(n_bits, samples, step_points, levels)`, same segment contract as `generate_step`. `levels[0]` is now the initial segment, `step_points[0]` is now an actual transition (no longer silently ignored). Internal `__main__` demo and all tests updated. New regression test `test_initial_level_is_applied` guards the fix.

---

### 16.3 — `generate_digital_step` signal length truncates the final level `[APPLIED 2026-04-13 — Option A]`
- **File:** `pyDataconverter/utils/signal_gen.py`
- **Resolution:** Adopted Option A. New signature `generate_digital_step(n_bits, samples, step_points, levels)` takes an explicit `samples` parameter so the caller controls total length. The final level now fills `[step_points[-1], samples)`. Regression test `test_final_level_appears` asserts `sig[800] == 200` and `sig[999] == 200` with step_points `[200, 400, 600, 800]` and 5 levels in a length-1000 array — the exact scenario that silently dropped the final level under the old contract.

---

### 16.4 — `generate_digital_ramp` uses `np.linspace(..., dtype=int)`
- **File:** `pyDataconverter/utils/signal_gen.py`, line 232
- **Bug:** `dtype=int` on `np.linspace` truncates (floors) the floating-point values,
  producing uneven code spacing and skipped codes.

  ```python
  # Current code:
  return np.linspace(0, max_code, n_points, dtype=int)
  ```

  Example — `n_bits=3` (max_code=7), `n_points=5`:
  - **Current** (`dtype=int`, truncation): `[0, 1, 3, 5, 7]`
  - **Proposed** (`round().astype(int)`):   `[0, 2, 4, 5, 7]`

  The proposed fix gives better distribution (rounds to nearest, not toward zero).
  Neither eliminates all unevenness for non-integer-spaced code counts — that is
  a mathematical constraint of fitting N codes into M points.

- **Proposed fix:**
  ```python
  return np.round(np.linspace(0, max_code, n_points)).astype(int)
  ```
- **Decision needed:** Apply fix, or document the current behaviour instead?
