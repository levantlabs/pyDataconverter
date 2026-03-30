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

### 16.1 — `generate_step` ignores `levels[0]`
- **File:** `pyDataconverter/utils/signal_gen.py`, lines 81-88
- **Bug:** `levels[0]` is stored in `current_level` but the signal is initialised to
  `np.zeros(samples)`. Samples before the first step point stay at 0.0 instead of
  `levels[0]`. This is original code (not introduced by our changes).

  ```python
  # Current code:
  signal = np.zeros(samples)
  current_level = levels[0]       # stored but never written to the signal

  for point, level in zip(step_points, levels[1:]):
      signal[point:] = level
  ```

  Example — `generate_step(100, [20, 50], [1.0, 2.0, 3.0])`:
  - **Got:**  `[0.0]*20 + [2.0]*30 + [3.0]*50`
  - **Want:** `[1.0]*20 + [2.0]*30 + [3.0]*50`

- **Proposed fix:**
  ```python
  signal = np.full(samples, levels[0])    # initialise to first level
  for point, level in zip(step_points, levels[1:]):
      signal[point:] = level
  ```
- **Decision needed:** Apply fix?

---

### 16.2 — `generate_digital_step` has the same initial-level bug
- **File:** `pyDataconverter/utils/signal_gen.py`, lines 256-263
- **Bug:** Identical pattern to 16.1 — `levels[0]` is never applied; signal starts
  at all zeros. The existing `__main__` demo hides the bug because it passes `levels[0] = 0`.
  This is original code.

  ```python
  # Current code (line 256):
  signal = np.zeros(step_points[-1], dtype=int)
  current_level = levels[0]       # stored but never applied

  for point, level in zip(step_points[1:], levels[1:]):
      signal[point:] = level
  ```

- **Where it is used:** `signal_gen.py` `__main__` block (line 616), and callable
  from any DAC testbench doing a step-response characterisation.
- **Proposed fix:** Same as 16.1 — `np.full(..., levels[0], dtype=int)` and iterate
  from `step_points[1:]` + `levels[1:]`.
- **Decision needed:** Apply fix?

---

### 16.3 — `generate_digital_step` signal length truncates the final level
- **File:** `pyDataconverter/utils/signal_gen.py`, line 256
- **Bug:** Signal length is set to `step_points[-1]`, so the array ends exactly at
  the last step boundary. The assignment `signal[step_points[-1]:]` sets an empty
  slice — the final level never appears in the output. This is original code.

  ```python
  # Current code:
  signal = np.zeros(step_points[-1], dtype=int)   # length = last step point index
  ```

  Example — `step_points = [0, 200, 400, 600, 800]`, `levels = [0, 1000, 2000, 3000, 4000]`:
  - Array length is 800 (indices 0–799).
  - `signal[800:] = 4000` → empty slice, level 4000 never appears.

- **Proposed fix option A:** Accept a `samples` parameter (mirrors `generate_step`
  signature) so the caller controls total length.
- **Proposed fix option B:** Default to `step_points[-1] + 1` extra sample so the
  last level appears for at least one sample.
- **Decision needed:** Which fix, or leave as-is?

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
