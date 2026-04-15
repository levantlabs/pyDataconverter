# pyDataconverter — Consolidated Code Review Report

*Synthesized from 5 independent reviews: Optimization, Security, Robustness, Correctness, and API Documentation.*
*Date: 2026-04-02 — last updated 2026-04-14. Section 5 adds Phase 2 full-codebase review.*

---

### Status summary (as of 2026-04-13)

| Item | Status |
|------|--------|
| P1-1 `OutputType.DIFFERENTIAL` doc typo | FIXED (`api_reference.md:196,204`) |
| P1-2 `CurrentSourceArray` therm_index range | FIXED doc-only (reviewer analysis was wrong; range is correctly `[0, n_therm_elements]` inclusive) |
| P1-3 dBFS ratio metrics | FIXED comment-only (calculation unchanged; no variable renamed) |
| P1-4 `MultibitSARADC` fallback `best_code=0` | NOT A BUG (monotonic flash sub-loop; lowest code is the correct fallback) |
| P2-1 `FlashADC` reference ladder check | ALREADY IMPLEMENTED |
| P2-2 `SimpleADC._quantize` range collapse | NOT A BUG (`v_min`/`v_max` come from `v_ref`, not from non-idealities) |
| P2-3 `SARADC` `v_sampled` out-of-range | NOT A BUG (`v_sampled` is not used to index the DAC) |
| P2-4 `SegmentedCDAC` `n_therm=0` | REJECTED BY DESIGN (`n_therm=0` must collapse to binary CDAC) |
| P2-5 R-2R non-finite tap voltages | LARGELY MITIGATED (solver catches LinAlgError; P2-6 closes the last pathway) |
| P2-6 `nodal_solver` index validation | FIXED (`nodal_solver.py:42-73`; rejects negative indices, out-of-range, non-positive R) |
| P2-7 `DifferentialComparator` `time_step<=0` | FIXED (`comparator.py:161-164`) |
| P2-8 `generate_step` unsorted `step_points` | FIXED (validation added; `levels[0]` contract flipped — `len(levels) == len(step_points) + 1`) |
| P2-9 `generate_digital_step` length mismatch | FIXED (validation added; contract redesigned, explicit `samples` parameter, `levels[0]` now applied, final level no longer truncated) |
| P2-10 `generate_prbs` seed annotation | FIXED (`signal_gen.py:459`) |
| P3-1 redundant FFT mask | FIXED (`_dynamic.py:47-88`; mask computed once, reused three times; ~40% less broadcast work per call) |
| P3-2 per-sample DAC/ADC batch API | DEFERRED (bigger refactor; separate design pass needed) |
| P3-3 `np.unpackbits` for bit extraction | FIXED (new `utils/_bits.py` helper; `decoder.py`, `cdac.py` (2x), `R2RDAC.py` all converted) |
| P3-4 reference.py property copies | FIXED (cached read-only views in `ReferenceLadder` and `ArbitraryReference`; `test_voltages_is_read_only_view` updated) |
| P3-5 NaN/Inf SNR in characterization | FIXED (`characterization.py:76-86`; `RuntimeWarning` on non-finite SNR) |
| P3-6 negative amplitude validation | DEFERRED (design question — negative amplitude is mathematically equivalent to a π-phase-shifted signal; rejecting would be breaking) |
| P3-7 `fft_analysis` 1e-20 bias comment | FIXED (`fft_analysis.py:71-78`) |
| P3-8 `NoiseshapingSARADC` docstring vs midpoint | NOT A BUG as reported (FLOOR+midpoint is the correct pairing; the residue of a FLOOR-quantised signal has zero mean at bin midpoints). **But** in reading the code I found a real bug: the reconstruction returned single-ended-equivalent voltage unconditionally, so in `input_type=DIFFERENTIAL` mode the residue was biased by +v_ref/2, the integrator saturated immediately, and noise shaping was silently disabled. Fixed at `SARADC.py:480-509` by subtracting `v_ref/2` from the reconstruction when `input_type == DIFFERENTIAL`. Regression test `test_differential_mode_integrator_does_not_saturate` added at `tests/test_sar_generalisation.py:253-273`. |
| P3-9 `FlashADC.reference_voltages` ×2 comment | FIXED (`FlashADC.py:158-169`) |
| P3-10 `QuantizationMode` isinstance guards | FIXED (`metrics/adc.py` — guards added at entry of `calculate_gain_offset_error` and `calculate_adc_static_metrics`; `SimpleADC.__init__` already had one) |

All fixes verified against 826 passing tests (9 new regression tests added).

### Newly discovered issues (not in original review)

- **Differential-mode NoiseshapingSARADC reconstruction bug** — discovered while verifying P3-8. Silently disabled noise shaping in differential mode. FIXED (see P3-8 row).

---

My comments are preceded by MEC:

## 1. Executive Summary

pyDataconverter is a well-structured simulation library with clear class hierarchies, good top-level input validation, and a readable API. The most serious concerns are not crashes-on-startup bugs but rather subtle silent failures: non-standard dBFS metric definitions that can mislead users comparing results against IEEE 1241, an off-by-one in `CurrentSourceArray.get_current` that permits a silent index overflow, and a broken code example in the published API documentation caused by a stale enum value (`'diff'` vs. the actual `'differential'`). Robustness gaps are concentrated at internal or derived computation boundaries — the public constructors guard their inputs well, but values produced by one component and consumed by another (e.g. DAC output fed into a noise calculation, or therm_code reaching a CDAC) are rarely re-validated. Performance is acceptable for typical use but degrades significantly on characterization sweeps due to redundant mask computations and per-sample Python function calls. None of the findings indicate fundamental architectural problems; all are targeted, locatable fixes.

---

## 2. Priority Action Items

### P1 — Fix Immediately (bugs, silent wrong results, broken public-facing material)

**P1-1. Fix stale `OutputType.DIFFERENTIAL` value in API docs**
- Reviews: API Documentation
- Files: `docs/api_reference.md` lines 196, 204
- The documentation states the enum value is `'diff'`; the actual value in `dataconverter.py` line 112 is `'differential'`. Example code that prints or compares `.value` will silently produce wrong results for any user who copies it.
- Fix: Update the table entry and example output in `api_reference.md` from `'diff'` to `'differential'`.
MEC: let's fix this
**STATUS: FIXED** — Updated `api_reference.md` table and example on lines 196 and 204.


**P1-2. Correct off-by-one in `CurrentSourceArray.get_current` thermometer bounds check**
- Reviews: Security
- Files: `pyDataconverter/components/current_source.py` line 295
- The guard uses `> n_therm_elements` but `n_therm_elements` is a valid exclusive upper bound; an index equal to `n_therm_elements` silently slices the full thermometer array without raising, bypassing the intended error.
- Fix: Change `therm_index > n_therm_elements` to `therm_index >= n_therm_elements` (when `n_therm_elements > 0`), or restate the valid range in the docstring and the guard consistently.
MEC:  let's fix this and update documentation
**STATUS: NO CODE CHANGE — REVIEWER ANALYSIS WAS INCORRECT.** After reading the actual code and call sites, `therm_index == n_therm_elements` is a **valid** state meaning all thermometer elements are steered to the positive rail (full-scale thermometer code). Changing `>` to `>=` would break fully-thermometer DACs at maximum code. The existing check is correct. The docstring was updated (`current_source.py` line 281) to explicitly state that the range is inclusive on both ends and explain what each endpoint means.

**P1-3. Document or standardize non-standard dBFS ratio metrics**
- Reviews: Correctness
- Files: `pyDataconverter/utils/metrics/_dynamic.py` lines 118–121
- `SNR_dBFS`, `SNDR_dBFS`, `THD_dBFS`, and `SFDR_dBFS` are computed by subtracting `fund_mag_dBFS` from already-dimensionless dB ratios. This convention is not IEEE 1241 / INSPEC standard and will confuse users comparing results with other tools or literature.
- Fix: Either remove the four `*_dBFS` ratio keys and retain only `FundamentalMagnitude_dBFS` and `HarmonicMags_dBFS` (which are genuine amplitude-in-dBFS quantities), or add a prominent docstring note explaining the custom definition and its rationale.
MEC:  this is actually a correct expression.  Pleaes add a comment along the lines that this is relevative to the full-scale maginutde.  Do not change the actual calculation
**STATUS: FIXED** — Added explanatory comment in `_dynamic.py` above lines 118–121 clarifying that these metrics express each ratio relative to the full-scale fundamental magnitude. No calculation changed.

**P1-4. Fix `MultibitSARADC` flash sub-ADC linear search producing incorrect best_code on first-iteration break**
- Reviews: Robustness, Correctness
- Files: `pyDataconverter/architectures/SARADC.py` lines 419–429
- The loop initialises `best_code = 0` and iterates from `n_levels-1` down to `0`. If the comparator returns `True` on the very first iteration (`sub_code == n_levels-1`), `best_code` is set and the loop breaks correctly. However if no `True` decision is ever reached (all trials fail), `best_code` stays at `0` rather than the lowest valid code, and there is no warning. For a well-functioning SAR this edge case should not occur, but it can under noise or mis-calibration conditions where it will produce a wrong code silently.
- Fix: After the loop, assert or log a warning if no `True` decision was found, and document the fallback behaviour explicitly.
MEC: I actually don't follow this. I would need to look at the code to understand what you are doing
**STATUS: NOT A BUG — REVIEWER ANALYSIS WAS INCORRECT.** The loop iterates `sub_code` from high to low and breaks on the first comparator `True`. Because the CDAC output is monotonic in `trial_code` and the comparator returns `1` iff `v_sampled > dac_output`, "first True from the top" is equivalent to "largest `sub_code` whose DAC output is ≤ `v_sampled`" — which is exactly the correct flash sub-conversion result. If the comparator never trips, the loop exits naturally with `sub_code == 0` and the fallback `best_code = 0` is the **correct** answer: the input is below every sub-threshold, so the lowest code wins. No silent wrong output. No code change needed.

---

### P2 — Fix Soon (robustness gaps and missing validation)

**P2-1. Validate `FlashADC` reference ladder length against comparator count**
- Reviews: Robustness
- Files: `pyDataconverter/architectures/FlashADC.py` lines 139–140
- No check that `n_references == n_comparators`; a mismatch causes a silent wrong conversion rather than an early error.
- Fix: Add an assertion in `__init__` and raise `ValueError` on mismatch.
MEC:  ok, this is fair.  Note the ref ladder may have a longer length, but the voltages the ADC gets for comparison should match
**STATUS: ALREADY IMPLEMENTED** — The check exists at `FlashADC.py` lines 128–131: raises `ValueError` when a custom reference has `n_references != n_comparators`. The auto-built ladder always has exactly `n_comparators = 2^n_bits − 1` taps by construction. No change needed.


**P2-2. Validate `SimpleADC._quantize` output range after nonideality application**

- Reviews: Robustness
- Files: `pyDataconverter/architectures/SimpleADC.py` lines 135–145
- Gain error, offset, and noise can collapse the effective range so that the quantisation step becomes zero or negative, producing NaN or inf output codes silently.
- Fix: Check that effective `v_max > v_min` after nonidealities are applied; raise a descriptive error.
MEC:  not sure I follow.  Gain error should definitely not change the transfer function to less than 0. If offset is big enough, there is still an ADC transfer function.  And noise wouldn't do this?
**STATUS: NOT A BUG — REVIEWER ANALYSIS WAS INCORRECT.** `_quantize` receives `v_min`/`v_max` from `_convert_input` as fixed constants derived from `v_ref` alone (`(0, v_ref)` for single-ended, `(−v_ref/2, v_ref/2)` for differential). Non-idealities (gain, offset, noise, jitter) are applied to the input signal `v` inside `_apply_nonidealities`, not to `v_min`/`v_max`. There is no code path where non-idealities can collapse the quantisation range. The existing `v_max > v_min` guard on line 135 is already sufficient. User's pushback is correct; no change needed.

**P2-3. Guard `SARADC` against gain-error-induced overflow of `v_sampled`**
- Reviews: Robustness
- Files: `pyDataconverter/architectures/SARADC.py` lines 290–297
- A large `gain_error` can push `v_sampled` outside the DAC reference range; subsequent DAC lookups return undefined results without warning.
- Fix: Clamp `v_sampled` to `[−v_ref/2, v_ref/2]` (or the appropriate range) and optionally emit a warning when clamping occurs.
MEC: why would subsequency DAC loopus return undefined results?  That shouldn't happen even if the input is out of range
**STATUS: NOT A BUG — REVIEWER ANALYSIS WAS INCORRECT.** `v_sampled` is never used as an index into the CDAC. In `_run_sar`, the CDAC is queried with an integer `trial_code` (`cdac.get_voltage(trial_code)`); `v_sampled` only appears as the comparator's signal input. An over-range `v_sampled` simply causes every comparator decision to go one way, so the SAR saturates to the max or min code — which is the correct saturation behaviour for an out-of-range input. No undefined lookups are possible. User's pushback is correct; no change needed.

**P2-4. Validate `SegmentedCDAC` thermometer code bounds and `n_therm` minimum**
- Reviews: Security, Robustness
- Files: `pyDataconverter/components/cdac.py` lines 500–533, 515–518
- When `n_therm=0`, a NaN can propagate silently through `get_voltage`. An out-of-range `therm_code` reaches the array without a bounds check.
- Fix: Enforce `1 <= n_therm < n_bits` in `__init__`, and add a bounds check on `therm_code` in `get_voltage`.
MEC:  no, a segmented CDAC should collapse to a binary dac if n_therm = 0.  
**STATUS: REJECTED BY DESIGN.** `n_therm = 0` is a valid degenerate case that must collapse to a pure binary CDAC, so the `n_therm >= 1` floor is wrong. The `therm_code` bounds-check sub-item is subsumed by the analogous fix already applied to `CurrentSourceArray.get_current` (see P1-2); if it turns out the CDAC path has a real silent overflow, it should be filed as a separate, narrower finding after reading `cdac.py` directly.

**P2-5. Validate `R2RDAC` and `ResistorStringDAC` tap voltages for finiteness**
- Reviews: Robustness
- Files: `pyDataconverter/architectures/R2RDAC.py` lines 192–200
- If the nodal solver produces non-finite tap voltages (due to a singular or near-singular resistor matrix), subsequent conversions silently return NaN/inf.
- Fix: After computing tap voltages, call `np.all(np.isfinite(_tap_voltages))` and raise `RuntimeError` with a diagnostic message if it fails.
MEC:  I'm not sure this is a real issue, can you analyze more?
**STATUS: LARGELY MITIGATED — NO CODE CHANGE.** `solve_resistor_network` in `nodal_solver.py` already catches `np.linalg.LinAlgError` and re-raises it as a `ValueError` with a clear message (lines 59–65). For a well-formed R-2R ladder built from a constant R value the conductance matrix is diagonally dominant and well-conditioned, so near-singular NaN/inf outputs are not reachable under any legitimate construction. Now that P2-6 additionally rejects non-positive resistances at the solver entry, the pathway that would produce non-finite tap voltages is closed. A defensive `np.isfinite` check would be belt-and-suspenders only.


**P2-6. Validate `nodal_solver` node index bounds**
- Reviews: Robustness
- Files: `pyDataconverter/utils/nodal_solver.py` lines 15–67
- Node indices are used to index into a matrix directly; out-of-range indices cause cryptic NumPy errors rather than informative messages.
- Fix: Validate that all node indices are in `[0, n_nodes)` before building the conductance matrix.
MEC:  please elaborate with actual code
**STATUS: FIXED (stronger than the original report).** The actual silent-failure mode was negative indices: NumPy wraps `G[-1, -1]` silently, so an upstream bug that produced `na = -1` would build the wrong network rather than erroring. The fix in `nodal_solver.py` now rejects: (a) non-positive `n_nodes`, (b) `na` or `nb` outside `[0, n_nodes)` (including negatives) in every resistor tuple, (c) non-positive resistance values, and (d) out-of-range node keys in `fixed_voltages`. All raise `ValueError` with a descriptive message. 816 existing tests still pass.


**P2-7. Guard `DifferentialComparator` bandwidth calculation against `time_step=0`**
- Reviews: Robustness
- Files: `pyDataconverter/components/comparator.py` lines 158–163
- Division by `time_step` produces `inf` bandwidth when `time_step=0`, which then silently clears any filtering effect.
- Fix: Raise `ValueError` if `time_step <= 0` when bandwidth is finite.
MEC:  please eslaborate with actual code
**STATUS: FIXED (reviewer's rationale was slightly off; fix is still correct).** The reviewer claimed `time_step=0` produced `inf` bandwidth. The actual failure mode is different: with `time_step=0`, `alpha = 0 / (0 + tau) = 0`, so `v_diff = (1 − 0) * _filtered_state + 0 * v_diff = _filtered_state` — the filter state is **frozen indefinitely**, not infinite bandwidth. Negative `time_step` would produce a nonsensical negative `alpha`. A `time_step > 0` guard fixes both cases; added in `comparator.py` with a `ValueError` that names the observed value, only active when `bandwidth is not None`.


**P2-8. Fix `signal_gen.generate_step()` handling of unsorted `step_points`**
- Reviews: Robustness
- Files: `pyDataconverter/utils/signal_gen.py` lines 99–120
- Step transitions are assigned assuming `step_points` is sorted; unsorted input produces wrong waveforms silently.
- Fix: Sort `step_points` at entry (with an optional warning) or add an explicit validation.
MEC:  I would like to see the actual code here
**STATUS: FIXED (contract flipped — breaking change, approved).**
- Added validation: every `step_points` entry must lie in `[0, samples]`; the list must be non-decreasing. Both raise `ValueError`.
- Contract redesigned so `levels[0]` is now the initial segment amplitude: `len(levels) == len(step_points) + 1`, `signal[0 : step_points[0]] = levels[0]`, `signal[step_points[i-1] : step_points[i]] = levels[i]`, `signal[step_points[-1] : samples] = levels[-1]`. The old "initialised to zero, `levels[0]` ignored" behaviour is gone.
- Docstring, `api_reference.md` example, and tests updated; new regression tests `test_initial_level_is_applied`, `test_levels_length_mismatch_raises`, `test_step_point_out_of_range_raises`, `test_non_monotonic_step_points_raises` added.


**P2-9. Fix `generate_digital_step()` potential length mismatch**
- Reviews: Robustness
- Files: `pyDataconverter/utils/signal_gen.py` lines 267–295
- When `n_samples` and `step_points` are inconsistent, the function can return an array of the wrong length without raising.
- Fix: Assert final array length equals `n_samples` before returning.
MEC:  let's see the code here
**STATUS: FIXED (contract redesigned — breaking change, approved).**
- New signature: `generate_digital_step(n_bits, samples, step_points, levels)` with explicit `samples` parameter. The output length is no longer smuggled through `step_points[-1]`.
- Three bugs in the original implementation are all now fixed:
  1. `levels[0]` silently discarded (same as P2-8).
  2. `step_points[0]` silently discarded.
  3. **Final level never appeared** — the old implementation set `len(signal) = step_points[-1]`, so `signal[step_points[-1]:] = levels[-1]` wrote to an empty slice. This is `PENDING_DECISIONS.md` item 16.3, previously unresolved.
- Contract matches `generate_step`: `len(levels) == len(step_points) + 1`. Internal `__main__` demo (`signal_gen.py:833-836`) updated to the new signature. Docstring, `api_reference.md` example, and tests all updated.
- New regression tests added: `test_initial_level_is_applied`, `test_final_level_appears`, `test_levels_length_mismatch_raises`, `test_step_point_out_of_range_raises`, `test_non_monotonic_step_points_raises`.


**P2-10. Correct `generate_prbs` seed type annotation**
- Reviews: API Documentation, Security (documentation mismatch)
- Files: `pyDataconverter/utils/signal_gen.py` line 459
- The annotation is `seed: int = None`, which is incorrect; `None` is not an `int`. The new-features doc correctly shows `int | None`.
- Fix: Change annotation to `seed: Optional[int] = None` (add `from typing import Optional` if not already imported).
MEc:  let's fix
**STATUS: FIXED** — Changed `seed: int = None` to `seed: Optional[int] = None` in `signal_gen.py` line 459; added `Optional` to the `typing` import.
---

MEC: didn't review the below yet
### P3 — Improve When Time Allows (performance, minor correctness, documentation polish)

**P3-1. Eliminate redundant frequency exclusion mask computation in `_dynamic.py`**
- Reviews: Optimization
- Files: `pyDataconverter/utils/metrics/_dynamic.py` lines 51–52 and 77–79
- The same `(num_harmonics+1, num_freqs)` broadcast mask is built twice per metrics call; move it above both uses and reuse the result.

**P3-2. Replace Python per-sample loops in DAC/ADC characterization with batch operations**
- Reviews: Optimization
- Files: `pyDataconverter/utils/metrics/dac.py` lines 102–109, `pyDataconverter/utils/characterization.py` lines 70–75
- Per-code and per-sample Python loops dominate runtime on high-resolution sweeps; consider a vectorized or batch-convert API.

**P3-3. Replace list-comprehension bit extraction with `np.unpackbits`**
- Reviews: Optimization
- Files: `pyDataconverter/components/decoder.py` line 153, `pyDataconverter/components/cdac.py` lines 254 and 689, `pyDataconverter/architectures/R2RDAC.py` line 169
- The pattern `[(code >> (n-1-k)) & 1 for k in range(n)]` is O(n) Python; `np.unpackbits` achieves the same in C with negligible overhead.

**P3-4. Avoid unconditional array copies in `reference.py` property accessors**
- Reviews: Optimization
- Files: `pyDataconverter/components/reference.py` lines 119, 125, 174, 180
- Every read of `.voltages` copies the full reference array; for a 12-bit Flash this is 4095 elements per conversion. Return a read-only view or cache the result.

**P3-5. Check for NaN/Inf SNR in characterization sweep and handle gracefully**
- Reviews: Robustness
- Files: `pyDataconverter/utils/characterization.py` lines 69–77
- When an FFT produces a degenerate spectrum, SNR can be `NaN` or `inf`; these values propagate into sweep results with no diagnostic.
- Fix: Detect and log a warning when NaN/Inf metrics are recorded.

**P3-6. Validate negative amplitude values in `signal_gen`**
- Reviews: Security
- Files: `pyDataconverter/utils/signal_gen.py` lines 28–49
- Negative amplitudes are accepted silently; they produce inverted waveforms that are likely unintentional.
- Fix: Raise `ValueError` for `amplitude < 0` or document that negative amplitudes are valid and what they mean.

**P3-7. Document the `1e-20` bias in `fft_analysis`**
- Reviews: Robustness
- Files: `pyDataconverter/utils/fft_analysis.py` line 73
- A small constant is added to avoid `log(0)` but is undocumented; users examining raw spectra will see an unexplained noise floor.
- Fix: Add a single inline comment explaining the regularisation and its magnitude.

**P3-8. Update `NoiseshapingSARADC` docstring to match midpoint reconstruction**
- Reviews: Correctness
- Files: `pyDataconverter/architectures/SARADC.py` line 488 and surrounding docstring
- The code correctly uses `(code + 0.5) / 2^N * v_ref` (midpoint), but the docstring says FLOOR quantization.
- Fix: Update the docstring; the implementation is correct.

**P3-9. Document the `FlashADC.reference_voltages` ×2 scaling factor**
- Reviews: Robustness
- Files: `pyDataconverter/architectures/FlashADC.py` lines 159–161
- An undocumented ×2 scale is applied to reference voltages; this will surprise users inspecting the ladder directly.
- Fix: Add a comment explaining why the scaling is applied.

**P3-10. Validate `QuantizationMode` argument with `isinstance` check**
- Reviews: Security
- Files: relevant quantization handling code
- The `QuantizationMode` parameter is not validated with an `isinstance` check, unlike most other enum parameters in the codebase.
- Fix: Add consistent `isinstance` guard matching the pattern used for `InputType` and `OutputType`.

---

## 3. Detailed Findings by Category

### 3.1 Correctness

| ID | File | Lines | Finding | Fix |
|----|------|--------|---------|-----|
| C-1 | `utils/metrics/_dynamic.py` | 118–121 | `SNR_dBFS`, `SNDR_dBFS`, `THD_dBFS`, `SFDR_dBFS` subtract `fund_mag_dBFS` from ratio metrics, producing a non-standard convention not compatible with IEEE 1241. | Remove `*_dBFS` ratio keys or add a docstring warning. See **P1-3**. |
| C-2 | `architectures/SARADC.py` | 419–429 | `MultibitSARADC` sub-code loop falls back to `best_code=0` with no diagnostic when no comparator trip is found. | Warn or assert on no-trip condition. See **P1-4**. |
| C-3 | `architectures/SARADC.py` | 488 | `NoiseshapingSARADC` uses midpoint reconstruction (`code+0.5`) while docstring documents FLOOR quantization. | Update docstring. See **P3-8**. |

### 3.2 Security

| ID | File | Lines | Finding | Fix |
|----|------|--------|---------|-----|
| S-1 | `components/current_source.py` | 295 | Off-by-one: `> n_therm_elements` should be `>= n_therm_elements`; valid maximum index silently accepted. | Change comparison operator. See **P1-2**. |
| S-2 | `components/cdac.py` | 515–518 | `SegmentedCDAC` with `n_therm=0` produces NaN without raising. | Enforce `n_therm >= 1` in `__init__`. See **P2-4**. |
| S-3 | `utils/signal_gen.py` | 28–49 | Negative amplitude accepted silently. | Raise `ValueError`. See **P3-6**. |
| S-4 | `architectures/SARADC.py` | 493–495 | Hardcoded clipping limits may not match actual reference range. | Derive limits from `v_ref`. |
| S-5 | `architectures/SimpleADC.py` | 55 | Inconsistent default `output_type` (`DIFFERENTIAL` vs `SINGLE` used elsewhere). | Standardise to `SINGLE` across the codebase and update docs. |
| S-6 | `components/comparator.py` | 129–131 | `_filtered_state` conditionally initialised; accessing before first `compare()` call could raise `AttributeError`. | Initialise unconditionally in `__init__`. |

*Note: The `DACBase` / `ADCBase` `n_bits=0` guard reported in the original Security review is already present in the current code (`dataconverter.py` lines 59–60 and 130–131) and does not require action.*

### 3.3 Robustness

| ID | File | Lines | Finding | Fix |
|----|------|--------|---------|-----|
| R-1 | `architectures/FlashADC.py` | 139–140 | Reference ladder length not checked against comparator count. | Add `ValueError` guard. See **P2-1**. |
| R-2 | `architectures/SimpleADC.py` | 135–145 | Range collapse after nonidealities not detected. | Validate effective range. See **P2-2**. |
| R-3 | `architectures/SARADC.py` | 290–297 | Gain error can push `v_sampled` out of DAC range. | Clamp with optional warning. See **P2-3**. |
| R-4 | `components/cdac.py` | 500–533 | `SegmentedCDAC` `therm_code` bounds not checked. | Add bounds check. See **P2-4**. |
| R-5 | `architectures/R2RDAC.py` | 192–200 | Tap voltages not checked for finiteness after nodal solve. | Check `np.isfinite`. See **P2-5**. |
| R-6 | `utils/nodal_solver.py` | 15–67 | Node indices not validated before matrix indexing. | Validate range. See **P2-6**. |
| R-7 | `components/comparator.py` | 158–163 | `time_step=0` produces `inf` bandwidth silently. | Raise `ValueError`. See **P2-7**. |
| R-8 | `utils/signal_gen.py` | 99–120 | Unsorted `step_points` produces wrong waveform silently. | Sort at entry. See **P2-8**. |
| R-9 | `utils/signal_gen.py` | 267–295 | `generate_digital_step` may return wrong-length array. | Assert final length. See **P2-9**. |
| R-10 | `utils/characterization.py` | 69–77 | NaN/Inf SNR not flagged during sweep. | Warn on non-finite metric. See **P3-5**. |
| R-11 | `components/reference.py` | 95–105 | Degenerate reference ladder (zero span) not caught. | Add minimum-range check in `__init__`. |
| R-12 | `utils/fft_analysis.py` | 73 | `1e-20` bias undocumented. | Add inline comment. See **P3-7**. |
| R-13 | `architectures/FlashADC.py` | 159–161 | ×2 reference voltage scaling undocumented. | Add comment. See **P3-9**. |
| R-14 | `utils/metrics/adc.py` | 426–434 | Histogram normalizer all-NaN fallback produces no diagnostic. | Log warning when fallback is triggered. |

### 3.4 Optimization

| ID | File | Lines | Finding | Fix |
|----|------|--------|---------|-----|
| O-1 | `utils/metrics/_dynamic.py` | 51–52, 77–79 | Frequency exclusion mask computed twice per call; 40–50% wasted computation on large FFTs. | Hoist to single computation, reuse. See **P3-1**. |
| O-2 | `utils/metrics/dac.py` | 102–109 | Per-code Python loop over all DAC codes; function call overhead dominates. | Vectorise or add batch API. See **P3-2**. |
| O-3 | `utils/characterization.py` | 70–75 | Per-sample ADC characterization loop. | Batch API. See **P3-2**. |
| O-4 | `components/decoder.py` | 153 | List comprehension for bit extraction. | Use `np.unpackbits`. See **P3-3**. |
| O-5 | `components/cdac.py` | 254, 689 | Same list-comprehension pattern. | Use `np.unpackbits`. See **P3-3**. |
| O-6 | `architectures/R2RDAC.py` | 169 | Same list-comprehension pattern. | Use `np.unpackbits`. See **P3-3**. |
| O-7 | `components/reference.py` | 119, 125, 174, 180 | Unconditional full array copy on every property access. | Return read-only view. See **P3-4**. |
| O-8 | `components/current_source.py` | 305 | `therm_sources[:therm_index]` creates a list slice before summing. | Use `sum(s.current for s in self.therm_sources[:therm_index])` or a numpy array sum. |
| O-9 | `architectures/FlashADC.py` | 218–224 | Sequential Python loop over 1023 comparators for 10-bit Flash. | Vectorise comparator evaluation. |

### 3.5 API Documentation

| ID | File | Lines | Finding | Fix |
|----|------|--------|---------|-----|
| D-1 | `docs/api_reference.md` | 196, 204 | `OutputType.DIFFERENTIAL` documented as `'diff'`; actual value is `'differential'`. Example code is broken. | Update table and example. See **P1-1**. |
| D-2 | `pyDataconverter/utils/signal_gen.py` | 459 | `seed: int = None` annotation is incorrect; `None` is not `int`. | Change to `Optional[int]`. See **P2-10**. |
| D-3 | `architectures/SARADC.py` | 488 | Docstring says FLOOR; code uses midpoint. | Fix docstring. See **P3-8**. |

---

## 4. Cross-Cutting Themes

### Input validation is strong at public API boundaries but absent at internal handoffs

The `ADCBase` and `DACBase` constructors validate `n_bits`, `v_ref`, and output type thoroughly (see `dataconverter.py` lines 56–68, 127–141). The same discipline is largely present in public `__init__` methods of the concrete architectures. However, values that flow between components — a `therm_code` produced by a decoder and consumed by a CDAC, a `v_sampled` produced by a track-and-hold and consumed by a SAR loop, tap voltages produced by a nodal solver and used in every conversion — pass without re-validation. This asymmetry explains why all five reviews found robustness issues at component interfaces rather than at the user-facing layer. A targeted pass adding lightweight precondition checks at the start of `get_voltage`, `get_current`, and `_quantize` methods would close most of these gaps with minimal code change.

### Silent numerical failures are the dominant risk category

Across all reviews, the most common failure mode is not an exception but a silent wrong number: NaN propagating through a CDAC, `inf` bandwidth in a comparator, a non-finite tap voltage in an R2R ladder, or an undetected range collapse in `SimpleADC`. Python and NumPy do not raise on NaN/inf arithmetic, which means these failures can travel through an entire characterisation sweep and appear only as anomalous metric values that a user might attribute to their DUT rather than the simulator. The recommended mitigation is a systematic "assert finite" pattern at the output of each component's core computation method, combined with a utility function such as `_check_finite(name, value)` that can be toggled off for production speed.

### Performance bottlenecks are predictable and targeted

The optimisation findings cluster around three patterns: redundant NumPy array construction (the double mask in `_dynamic.py`), Python-level loops over large iteration spaces (per-code DAC loops, per-sample ADC characterization, per-comparator Flash evaluation), and unnecessary data copies (reference voltage accessors). None of these require architectural changes. The highest-return fix is the mask deduplication in `_dynamic.py` because that function is called on every FFT result in a characterization sweep. The batch-API pattern for DAC/ADC loops would require a small interface addition but would yield large speedups on resolution sweeps. The bit-extraction and copy fixes are lower effort and should be applied opportunistically.

### Documentation and code are drifting apart in specific but high-visibility spots

The `OutputType.DIFFERENTIAL` value mismatch is the most serious documentation issue because it is in a quickstart-level example that a new user will run first and find broken. The `seed` annotation mismatch and the `NoiseshapingSARADC` docstring are lower severity but contribute to a pattern where internal changes are not propagated to documentation. Adding a lightweight documentation-check step to the CI pipeline — for example, a `doctest` run on the API reference examples — would catch regressions of this type automatically going forward.

---

## 5. Phase 2 Full-Codebase Review — 2026-04-14

*Scope: all library source files (ADC/DAC architectures, components, utils, visualizations, 4 new TI-ADC example scripts, updated quickstart). Three independent Explore-agent passes across the codebase; critical claims individually verified against source before inclusion.*

*943 tests passing at time of review. All test regressions introduced here would be new.*

---

### 5.1 Status summary

| ID | File | Severity | Title | Status |
|----|------|----------|-------|--------|
| R4-C1 | `utils/fft_analysis.py:242,266` | Minor (not Critical) | `duration` undefined in `demo_fft_analysis()` | FIXED |
| R4-I1 | `utils/visualizations/dac_plots.py:60` | ~~Important~~ | ~~DAC plot LSB uses wrong formula for default DACs~~ | FALSE POSITIVE |
| R4-I2 | `components/capacitor.py:97`, `current_source.py:103` | Important | Silent clipping of negative mismatch draws | FIXED |
| R4-I3 | `utils/characterization.py:92` | Important | Key rename creates inconsistency with source dict | FIXED |
| R4-I4 | `architectures/SARADC.py:109,175` | Important | `cap_mismatch` silently ignored when `cdac` provided | FIXED |
| R4-I5 | `components/cdac.py` | Important | Code bounds check only in `SegmentedCDAC`, not `SingleEndedCDAC`/`DifferentialCDAC` | Open |
| R4-I6 | `utils/visualizations/visualize_SARADC.py:103` | Important | Assumes CDAC `get_voltage` returns tuple; fails for single-ended | Open |
| R4-I7 | `architectures/R2RDAC.py:188` | Important | Double 2R-to-GND on LSB node when bit=0; verify linearity is unaffected | Needs verification |
| R4-I8 | `architectures/TimeInterleavedADC.py:290` | Important | `convert_waveform` assumes uniform time spacing; silently wrong for non-uniform `t` | Open |
| R4-M1 | `utils/signal_gen.py` | Minor | Inconsistent parameter names (`sampling_rate` vs `fs`) across functions | Open |
| R4-M2 | `components/comparator.py:99` | Minor | Docstring says "input-referred" but offset is applied post-bandwidth-filter | Open |
| R4-M3 | `utils/visualizations/adc_plots.py:88-99` | Minor | LSB calculated from swept `v_range` instead of `adc.v_ref`; wrong for differential ADCs | Open |
| R4-M4 | `architectures/TimeInterleavedADC.py:_resolve_mismatch` | Minor | Negative scalar stddev not rejected; silently treated as positive | Open |
| R4-M5 | `utils/metrics/adc.py:425` | Minor | PDF singularity threshold 0.999 is a magic number | Open |
| R4-M6 | `utils/signal_gen.py:559` | Minor | PRBS error message does not state the valid order range (2–20) | Open |

**False positives (verified against source, not bugs):**

| Claim | Why invalid |
|-------|-------------|
| PipelinedADC combiner formula `DOUT + DOUT*H + code` is wrong | Docstring at `PipelinedADC.py:152-154` explicitly states this is "bit-exactly matching the adc_book reference's accumulation formula." Intentional. |
| FlashADC differential reference scaling inconsistency | `_convert_input` passes `comp_refs[i]` and `comp_refs[n-1-i]` to the comparator. Effective threshold is `comp_refs[i] − comp_refs[n-1-i] = 2×comp_refs[i]`. The `reference_voltages` property is a public read accessor scaled ×2 for display. Code is correct. |

---

### 5.2 Detailed findings

---

**R4-C1 — `demo_fft_analysis()` crashes: `duration` undefined** — **FIXED**
- **File:** `pyDataconverter/utils/fft_analysis.py:242, 266`
- **Severity on reflection:** Minor — `demo_fft_analysis()` is a module-level `if __name__ == "__main__"` entry point, not imported anywhere in the library, tests, or examples. Known issue previously tracked in `CODE_REVIEW.md:119` (item 15.3), `PENDING_DECISIONS.md:42`, and `todo/fft_analysis_review.md`. Original "Critical" rating overstated its real-world impact — no user path reached it.
- **Description:** Demos 2 and 3 of `demo_fft_analysis()` used the variable name `duration` which was never assigned in the function scope. Calling `demo_fft_analysis()` (or running the module directly with `python -m pyDataconverter.utils.fft_analysis`) raised `NameError: name 'duration' is not defined` at Demo 2.
- **Fix:** Added `duration = NFFT / fs` to the test-parameters block (`fft_analysis.py:215`) and made Demo 1 reuse the same variable for consistency. Consolidated the three pre-existing coverage tests (`test_demo_fft_analysis_runs_partially`, `test_demo_fft_analysis_full_with_patched_duration`, `test_main_block_calls_demo`) — which had documented the bug as a pytest expectation via `pytest.raises(NameError)` — into two positive tests: `test_demo_fft_analysis_runs_to_completion` and an inverted `test_main_block_calls_demo`. 942 tests passing.

---

**R4-I1 — `dac_plots.py` LSB formula wrong for default DACs** — **FALSE POSITIVE (closed)**
- **File:** `pyDataconverter/utils/visualizations/dac_plots.py:60`
- **Verification:** `DACBase.__init__` at `dataconverter.py:207` sets `self.lsb = v_ref / (n_levels - 1)` where `n_levels` defaults to `2**n_bits`. So the default LSB **is** `v_ref / (2**n_bits - 1) = v_ref / (n_codes - 1)` — exactly what `dac_plots.py:60` computes. The `SimpleDAC` docstring at line 54 also states this. Original review finding misidentified the default formula. No code change.
- **Follow-up (deferred):** `dac_plots.py` could use `dac.lsb` directly instead of recomputing, as a single-source-of-truth cleanup. Not filed as an issue.

---

**R4-I2 — Silent clipping of negative capacitance / current from large mismatch draws** — **FIXED**
- **Files:** `pyDataconverter/components/capacitor.py:97-109`, `pyDataconverter/components/current_source.py:103-115`
- **Severity:** Important
- **Description:** Both components clip values to 0 silently when a large mismatch draw produces a negative result. A 0 F capacitor or 0 A current source is not a realistic mismatch model — it means the component is missing entirely. This masks over-large mismatch specifications (e.g., `mismatch=2.0`) with wrong simulation results rather than a user-visible error.
- **Fix:** Emit a `RuntimeWarning` at `stacklevel=2` whenever the clip fires, naming the drawn value and the mismatch stddev that produced it. Numerical behavior unchanged (still clips to zero); the component now tells the user that the clip happened so they can shrink the mismatch parameter. No input-validation rejection was added — per user request, warning only.

---

**R4-I3 — `characterization.measure_dynamic_range` exposes amplitude in only one unit** — **FIXED**
- **File:** `pyDataconverter/utils/characterization.py:88-106`
- **Severity:** Important
- **Description:** `measure_dynamic_range()` previously returned only `AmplitudeAtSNR0_dBFS` (silently renamed from the source function's `AmplitudeAtSNR0_dB` key). The sweep axis is in dBFS, so the renamed value was truthful — but callers working in absolute dBV units had no key available, and the `_dB` vs `_dBFS` pairing seen elsewhere in the metrics layer (e.g., `SNR_dB` / `SNR_dBFS` in `metrics/_dynamic.py`) was missing here.
- **Fix:** Expose both keys:
  - `AmplitudeAtSNR0_dBFS` — raw interpolation on the dBFS sweep axis (amplitude relative to full-scale).
  - `AmplitudeAtSNR0_dB` — same value converted to absolute dBV-style units via `+ 20·log₁₀(v_ref/2)`. For `v_ref=1 V` this is `dBFS − 6.02`. Reflects the physical signal level in dB re 1 V.

  Docstring updated to document both keys. `docs/api_new_features.md` return-key table updated. New regression test `test_measure_dynamic_range_amplitude_key_relationship` verifies the `20·log10(v_ref/2)` offset holds. 943 tests passing (+1 new).

---

**R4-I4 — `SARADC`: `cap_mismatch` silently ignored when custom `cdac` is provided** — **FIXED**
- **Files:** `pyDataconverter/architectures/SARADC.py`, `pyDataconverter/components/cdac.py`, `pyDataconverter/components/capacitor.py`
- **Severity:** Important
- **Description:** When `cdac=None`, `cap_mismatch > 0` correctly seeded a mismatched CDAC. When `cdac` was provided, `cap_mismatch` was documented as "ignored" — but that meant a caller building a topology with one CDAC and then constructing SARADC with a different `cap_mismatch` would silently lose the SARADC-level value. This blocked the natural Monte Carlo workflow of reusing one CDAC topology across many statistical draws.
- **Design change:** Promoted mismatch from a construction-time immutable to a re-drawable property on the CDAC.
  - **`UnitCapacitorBase.redraw_mismatch(stddev, rng)`** — new method (default raises `NotImplementedError`). `IdealCapacitor` implements it: preserves `_c_nominal`, draws a fresh ε ~ N(0, stddev), sets `_capacitance = c_nominal * (1 + ε)`. Same negative-clip warning as the construction-time path.
  - **`CDACBase.apply_mismatch(cap_mismatch, seed=None)`** — new method (default raises `NotImplementedError`). Implemented concretely in:
    - `SingleEndedCDAC` (inherited by `RedundantSARCDAC` and `SplitCapCDAC` — the redundant CDAC's DEC lookup uses ideal radix weights and is unaffected by re-draw)
    - `SegmentedCDAC` (delegates to the inner `SingleEndedCDAC`)
    - `DifferentialCDAC` (re-draws both positive and negative arrays independently from the same `Generator`, so `(pos, neg)` pair is reproducible from a single seed)
  - Each implementation refreshes the cached `_cap_weights` / `_cap_total` arrays after re-drawing.
- **`SARADC.__init__` change:** When `cdac is not None and cap_mismatch > 0`, SARADC now emits a `RuntimeWarning` (so the override is visible) and calls `cdac.apply_mismatch(cap_mismatch)`. The CDAC's nominal topology is preserved, the SARADC-level stddev wins. `cap_mismatch=0` with a supplied cdac is unchanged (no warning, cdac used as-is).
- **Tests added (22 new):** `test_capacitor.py::TestRedrawMismatch` (5 tests), `test_cdac.py::TestApplyMismatch*` (15 tests covering SingleEnded, Differential, Segmented, and base-class default), `test_sar_generalisation.py::TestSARADCCapMismatchPassthrough` (4 tests covering the warning, topology preservation, no-warn on zero, and differential passthrough). 965 tests passing (was 943; +22 new).
- **Monte Carlo workflow now supported:**
  ```python
  cdac = SingleEndedCDAC(n_bits=12, v_ref=1.0)  # ideal topology
  for seed in range(1000):
      cdac.apply_mismatch(0.002, seed=seed)  # fresh realization, same topology
      adc = SARADC(n_bits=12, cdac=cdac)     # no SARADC-level override needed
      # ... run one Monte Carlo iteration ...
  ```

---

**R4-I5 — CDAC code-range check exists only in `SegmentedCDAC`**
- **File:** `pyDataconverter/components/cdac.py`
- **Severity:** Important
- **Description:** `SegmentedCDAC.get_voltage()` validates that the code is in `[0, 2^n_bits - 1]` and raises `ValueError` on violation. `SingleEndedCDAC.get_voltage()` and `DifferentialCDAC.get_voltage()` have no such check; an out-of-range code silently passes to bit-extraction, where NumPy's shift arithmetic produces a wrong bit pattern rather than an error.
- **Fix:** Add the same bounds check to `SingleEndedCDAC.get_voltage()` and `DifferentialCDAC.get_voltage()` to match the `SegmentedCDAC` behavior.

---

**R4-I6 — `visualize_SARADC._bit_contributions` assumes differential CDAC**
- **File:** `pyDataconverter/utils/visualizations/visualize_SARADC.py:103`
- **Severity:** Important
- **Description:** Line 103 does `diff_base = v_base[0] - v_base[1]` where `v_base = adc.cdac.get_voltage(0)`. For a `SingleEndedCDAC`, `get_voltage` returns a scalar `float`, not a tuple; `v_base[0]` then indexes the first character of the float's string representation (via Python's iterator fallback) and raises `TypeError`. The visualization silently works only when the SAR uses a differential CDAC.
- **Fix:**
  ```python
  v_base = adc.cdac.get_voltage(0)
  diff_base = (v_base[0] - v_base[1]) if isinstance(v_base, tuple) else float(v_base)
  ```

---

**R4-I7 — `R2RDAC` LSB node has two parallel 2R arms to GND when bit=0**
- **File:** `pyDataconverter/architectures/R2RDAC.py:180-188`
- **Severity:** Important (needs verification)
- **Description:** The for-loop at lines 180–182 adds a 2R arm from each node k to `switch_node` (GND if bit=0, V_ref if bit=1). Line 188 adds a permanent additional 2R arm from node `n-1` to GND. When bit `n-1` (LSB) = 0, there are two parallel 2R arms from node `n-1` to GND → effective resistance R, not 2R. This changes the Thevenin impedance at the LSB node and would cause a non-linearity at all codes where LSB=0.

  The comment at line 184-187 says this is "separate from the bit switch arm" and required for binary weighting — but that rationale applies only when the bit switch arm connects to V_ref (bit=1). When bit=0 both arms go to the same node, creating a topology mismatch.
- **Fix / Verification needed:** Run `[dac.convert(c) for c in range(2**n_bits)]` on a known-linear configuration and check that adjacent-code steps are equal. If DNL is non-zero at LSB=0 codes, remove the permanent termination (line 188) and instead rely on the bit-switch arm to provide the termination.

---

**R4-I8 — `TimeInterleavedADC.convert_waveform` assumes uniform time spacing**
- **File:** `pyDataconverter/architectures/TimeInterleavedADC.py:290`
- **Severity:** Important
- **Description:** `dt = float(t_dense[1] - t_dense[0])` is computed once and used as the uniform sample period for all IIR bandwidth filters. If the caller passes a non-uniformly-spaced `t_dense` array (e.g., from a non-uniform resampler or a simulation with variable step size), every filter cutoff frequency is wrong, silently.
- **Fix:** Validate that `t_dense` is uniformly spaced before using it:
  ```python
  dt_all = np.diff(t_dense)
  if not np.allclose(dt_all, dt_all[0], rtol=1e-4):
      raise ValueError("convert_waveform requires uniformly spaced t_dense")
  dt = float(dt_all[0])
  ```

---

**R4-M1 — `signal_gen.py` inconsistent parameter names across functions**
- **File:** `pyDataconverter/utils/signal_gen.py`
- **Severity:** Minor
- **Description:** `generate_sine()` and `generate_multitone()` use `sampling_rate`; `generate_chirp()`, `generate_step()`, `generate_prbs()`, and `generate_coherent_sine()` use `fs`. Both mean the same thing, but the inconsistency forces callers to check each function's signature.
- **Fix:** Standardize to `fs` (shorter, matches scipy and NumPy conventions) in a single pass. Maintain backward compatibility with a deprecated `sampling_rate` alias if needed.

---

**R4-M2 — Comparator docstring says "input-referred" but offset is applied post-filter**
- **File:** `pyDataconverter/components/comparator.py:99`
- **Severity:** Minor
- **Description:** The `offset` parameter docstring says "DC input-referred offset voltage (V)". In the implementation, offset is added to `v_diff` after the bandwidth-limiting IIR filter. For a high-bandwidth signal, the pre-filter and post-filter values are identical, so this is only misleading for configurations with both `bandwidth` and `offset` active. The term "input-referred" implies the offset appears before all signal processing.
- **Fix:** Change the docstring to "DC offset voltage added to the filtered differential signal" to match the actual implementation.

---

**R4-M3 — `adc_plots.plot_transfer_function` uses swept range, not `v_ref`, for LSB**
- **File:** `pyDataconverter/utils/visualizations/adc_plots.py:88-99`
- **Severity:** Minor
- **Description:** The function computes `lsb = v_range / n_codes` where `v_range = v_max - v_min` is the sweep range. For a single-ended ADC tested at full range this is correct, but for a differential ADC (where v_range is the full differential swing but internal v_ref is per-rail) the LSB is wrong, and the plotted error axis will be miscalibrated.
- **Fix:** Prefer `lsb = adc.v_ref / n_codes` (or the symmetric variant when applicable) over the swept range.

---

**R4-M4 — `TimeInterleavedADC._resolve_mismatch` accepts negative scalar stddev**
- **File:** `pyDataconverter/architectures/TimeInterleavedADC.py` (`_resolve_mismatch` helper)
- **Severity:** Minor
- **Description:** A negative scalar (e.g., `-0.001`) is interpreted as a standard deviation and passed directly to `rng.normal(scale=negative)`, which raises `ValueError: scale < 0` deep inside NumPy — not at the TI-ADC constructor boundary.
- **Fix:** Add `if scalar < 0: raise ValueError(f"{name} scalar must be >= 0 (stddev), got {scalar}")` before the `rng.normal` call.

---

**R4-M5 — PDF singularity threshold 0.999 in `metrics/adc.py` is a magic number**
- **File:** `pyDataconverter/utils/metrics/adc.py:425`
- **Severity:** Minor
- **Description:** `in_range = np.abs(u) < 0.999` avoids the `1/sqrt(1-u²)` singularity at ±1. The value 0.999 is undocumented. The threshold affects INL accuracy near the rails.
- **Fix:** Extract as `_PDF_SINGULARITY_GUARD = 0.999  # avoid 1/sqrt(1-u²) divergence within 0.1% of ±full-scale` and add a brief comment.

---

**R4-M6 — PRBS error message does not indicate valid order range**
- **File:** `pyDataconverter/utils/signal_gen.py:559`
- **Severity:** Minor
- **Description:** `generate_prbs(order=21)` raises `ValueError` with a message that lists only the unsupported value, not the supported range.
- **Fix:** `raise ValueError(f"order must be between 2 and 20, got {order}")`.

---

### 5.3 Cross-cutting observation: LSB formula fragmentation

Three separate locations compute LSB with slightly different formulas:

| Location | Formula | Correct for |
|----------|---------|-------------|
| `dac_plots.py:60` | `v_ref / (n_codes - 1)` | Symmetric DACs only |
| `simple_dac_example.py:119` | `v_ref / (n_codes - 1)` | Same — wrong for default `SimpleDAC` |
| `adc_plots.py:94/98` | `v_range / (n_codes - 1)` or `v_range / n_codes` | Swept range — wrong for differential |

The root cause is that `DACBase` and `ADCBase` do not expose a `get_lsb()` method. Adding one would let all visualization and metrics code use a single authoritative source:

```python
# DACBase
def get_lsb(self) -> float:
    if self.n_levels is None:
        return self.v_ref / (2 ** self.n_bits)
    return self.v_ref / (self.n_levels - 1)
```

This is a low-effort, high-consistency fix.

---

### 5.4 Newly confirmed correct behaviors (do not re-file)

These were flagged by automated review agents but are intentional or already correct:

- **`PipelinedADC` combiner `DOUT = DOUT + DOUT*H + code`** — deliberately matches adc_book reference formula (docstring says so explicitly; formula is validated by full test suite).
- **`FlashADC` differential reference path** — `comp_refs[i]` and `comp_refs[n-1-i]` are single-ended values; the comparator's differential subtraction `(v_pos − v_refp[i]) − (v_neg − v_refn[i])` achieves the ×2 threshold automatically. The `reference_voltages` property returns ×2 for external display only.
- **`R2RDAC` nodal solver robustness** — `nodal_solver.py` already rejects non-positive R and out-of-range indices (P2-6 from prior review). Tap-voltage finiteness is guaranteed for well-formed ladders (P2-5 from prior review).
- **`SARADC` gain-error overflow** — `v_sampled` drives comparators, not DAC indices; over-range saturates to max/min code correctly (P2-3 from prior review).
