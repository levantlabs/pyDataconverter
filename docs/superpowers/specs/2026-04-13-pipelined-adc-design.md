# Pipelined ADC — Phase 1 Design

**Date:** 2026-04-13
**Status:** Design complete, pending user review. Implementation has not started.
**Scope:** Phase 1 of pyDataconverter's pipelined ADC support. Time-interleaved ADC is a separate design session scheduled after Phase 1 lands.

---

## 1. Context and motivation

pyDataconverter currently implements five ADC families: `SimpleADC`, `FlashADC`, `SARADC`, `MultibitSARADC`, and `NoiseshapingSARADC`. Pipelined ADCs — the dominant architecture for 10–14 bit converters at 50–500 MS/s — are missing. This design adds a pipelined ADC that:

1. Composes existing pyDataconverter sub-components (any `ADCBase` as the sub-ADC, any `DACBase` as the sub-DAC, a new `ResidueAmplifier` component), so users can swap in `FlashADC`, `SARADC`, `SimpleDAC`, `ResistorStringDAC`, etc. without the pipelined class defining its own mini-ADC or mini-DAC.
2. Matches the existing pyDataconverter conventions: inherits from `ADCBase`, uses the `_convert_input` pattern, validates constructor arguments in `__init__`, follows the SARADC style for S&H non-idealities.
3. Reproduces the exact output of a vetted reference implementation (Manar El-Chammas's `adc_book/python/pipelinedADC.py`) bit-exactly on a set of canonical configurations. The reference is used as a read-only oracle for validation; the new code is written fresh in pyDataconverter style, not ported as-is.
4. Preserves the reference's most distinctive modelling feature — the coupling between comparator metastability and residue-amplifier settling — as an emergent consequence of component composition rather than as monolithic stage logic.

The reference code was reviewed in full before this design was written. Its topology (heterogeneous stages with independent `N`, `FSR_ADC`, `FSR_DAC`, `G`, `minADCcode` per stage, plus a back-end flash) is adopted as the target topology. Its algebraic model for the metastability / settling coupling is adopted unchanged, derived symbolically in Appendix A, and translated into a cleaner component decomposition.

---

## 2. Architecture overview

### 2.1 Top-level cascade

```
           ┌──────────────┐   ┌──────────────┐       ┌──────────────┐   ┌────────────┐
 v_in ───▶ │  first S/H   │──▶│ PipelineStage│──▶...─▶│ PipelineStage│──▶│ backend ADC│──▶ code_be
           │  kT/C, gain, │   │      0       │       │     N-2      │   │(any ADCBase)│
           │  offset, jit │   └──────┬───────┘       └──────┬───────┘   └─────┬──────┘
           └──────────────┘          │                      │                 │
                                   code_0                code_N-2          code_be
                                     │                      │                 │
                                     ▼                      ▼                 ▼
                             ┌────────────────────────────────────────────────────┐
                             │     digital combiner — weighted accumulation       │
                             │     DOUT += DOUT * H + code  per stage + backend    │
                             └────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                                              final digital code
```

Input flows left to right: an optional first-stage sample-and-hold applies the same non-idealities the existing `SARADC._sample_input` applies (gain error, input-referred offset, kT/C thermal noise, aperture jitter), and the result enters the cascade.

Each `PipelineStage` instance converts its input into a digital stage code plus an amplified residue. The residue propagates forward to the next stage. After the last `PipelineStage`, a **required** backend ADC (any `ADCBase` subclass, but typically a `FlashADC` with an arbitrary number of comparators) digitises the final residue into one more code.

The digital combiner inside `PipelinedADC` reads the code from every stage and the backend, applies the reference's accumulation formula (`DOUT += DOUT * H + code` per stage, then once more with `backend_H` for the backend), optionally clips the result to `[0, 2^n_bits − 1]`, and returns an integer.

### 2.2 Inside one `PipelineStage`

```
                          ┌─────────────────┐
                ┌────────▶│ sub_adc:ADCBase │──── raw_code ──┐
                │         └─────────────────┘                │
                │                                            ▼
 v_sampled ─────┤                                 ┌─────────────────┐
                │                                 │ sub_dac:DACBase │─── v_dac ─┐
                │                                 └─────────────────┘           │
                │                                                               ▼
                └──────────────────────────────────────────────────────────▶ (v − v_dac + offset)
                                                                                │
                                                                                ▼
                             ┌──────────────────────┐               ┌──────────────────────┐
                             │ last_conversion_time │──── t_regen ──▶│  ResidueAmplifier    │
                             │ last_metastable_sign │──── sign   ──▶│  amplify(target,     │─── v_res ──▶
                             │     (on sub_adc)     │               │  initial_error,      │
                             └──────────────────────┘               │  t_budget)           │
                                                                    └──────────────────────┘
```

The stage is pure coordination. It does not implement any non-ideality itself. It asks the sub-ADC for a code and for its metastability state (regeneration time plus sign of the initial-condition error), asks the sub-DAC for the analog reconstruction of the raw code, computes the ideal residue, and hands the ideal target plus the metastability initial-condition error to the residue amp along with whatever time budget remains after the slowest comparator regenerated.

All physical modelling lives in the components: comparator regeneration is inside `DifferentialComparator`, amplifier settling is inside `ResidueAmplifier`. The coupling between them emerges from the stage's timing-budget calculation.

### 2.3 Architectural invariants

- **Pluggable sub-components.** `PipelineStage` accepts any `ADCBase` subclass as the sub-ADC and any `DACBase` subclass as the sub-DAC. Phase 1 default for the canonical example is `FlashADC` (with relaxed `n_comparators`) as the sub-ADC and `SimpleDAC` (with relaxed `n_levels` and a `code_errors` array) as the sub-DAC. Users can substitute any existing or future converter class.
- **Physical models live where they physically belong.** Comparator regeneration is on `DifferentialComparator`. Amplifier settling is on `ResidueAmplifier`. The metastability-to-settling coupling that the reference code hard-wires into `Stage.output` emerges here from composition: `stage` queries the sub-ADC's regen time and sign, scales the LSB-level initial-condition error by the stage's residue gain, and hands it to the amp, which returns `target + initial_error · exp(−t_budget/τ_amp)`.
- **The stage is just coordination.** It owns the sample rate, the timing-budget calculation, the subtraction, and the data-flow plumbing. It does not re-implement any non-ideality a component underneath it already owns.
- **Backend is required and is any `ADCBase`.** A `PipelinedADC` always ends in a backend ADC that digitises the final residue — never optional. The backend can be any `ADCBase` subclass, typically `FlashADC` with an arbitrary (possibly even) number of comparators for a high-resolution final stage.
- **Digital combiner is bit-exact to the reference.** `DOUT += DOUT * stage.H + code` is applied per stage and then once more with `backend_H` for the backend, reproducing the reference's `PipelinedADC.output` line 43 arithmetic without the `option=2` branch.

---

## 3. Component inventory

All of pipelined ADC's support falls into three buckets: new files, surgical extensions to existing files, and reused-unchanged files.

### 3.1 NEW — files created (six)

| Path | Role |
|---|---|
| `pyDataconverter/components/residue_amplifier.py` | `ResidueAmplifier` component. `amplify(target, initial_error, t_budget) -> v_out` applies finite gain, offset, optional slew limiting, and exponential settling for the supplied time budget. Ideal case (`settling_tau=0`, `slew_rate=∞`) degenerates to `gain * target + offset`. |
| `pyDataconverter/architectures/PipelinedADC.py` | Two classes: `PipelineStage` (helper, does not inherit from `ADCBase` — it takes a sub-ADC, a sub-DAC, a residue amp, a sample rate, an input-referred offset, a code offset, and an `H` weight) and `PipelinedADC(ADCBase)` (top-level — takes first-stage S&H non-idealities, a list of `PipelineStage`, a required backend `ADCBase`, backend `H` and code offset, and a clip-output flag; implements `_convert_input` and the digital combiner). |
| `tests/_reference/adc_book_pipelined.py` | Vendored reference — just the class definitions from `adc_book/python/pipelinedADC.py` with `__main__`, plotting, `ADCanalysis`, and `print` statements removed. Header attributes the source and marks read-only. |
| `tests/_reference/__init__.py` | Package marker. |
| `tests/test_pipelined_adc.py` | Unit tests for the new classes: construction validation, ideal transfer functions, parameter round-trips, `_convert_input` edge cases, digital-combiner weight accumulation, interaction with different sub-ADC / sub-DAC choices. |
| `tests/test_pipelined_adc_vs_reference.py` | Bit-exact comparison against the vendored reference across four canonical configurations. |
| `examples/pipelined_adc_example.py` | Clean Phase 1 example: builds the canonical 12-bit pipelined ADC, runs a coherent sine, plots the residue trace, computes SNDR via `utils/metrics`. Follows the style of `sar_adc_example.py`. |

### 3.2 EXTENDED — surgical changes to existing files

All changes are strictly additive: new kwargs default to "behaviour unchanged", no existing call sites break.

| Path | Change |
|---|---|
| `pyDataconverter/dataconverter.py` | `DACBase.__init__` gains an optional `n_levels: Optional[int] = None` kwarg. When provided, overrides `2^n_bits` as the number of valid codes. `self.lsb = v_ref / (n_levels - 1)` unconditionally. Default keeps `n_levels = 2^n_bits`, backward compatible. `DACBase.convert` validates codes against `[0, n_levels - 1]`. |
| `pyDataconverter/components/comparator.py` | `DifferentialComparator.__init__` gains `tau_regen: float = 0.0` and `vc_threshold: float = 0.5`. Adds a `last_regen_time` read-only property returning the regeneration time of the most recent `compare()` call as `tau_regen * ln(vc_threshold / max(|v_diff|, 1e-30))`. With `tau_regen=0` the property returns 0 and behaviour is unchanged from today. |
| `pyDataconverter/architectures/FlashADC.py` | Decouples `n_comparators` from `n_bits` via an optional `n_comparators: Optional[int] = None` kwarg. Default keeps `2^n_bits - 1` (backward compatible). Arbitrary positive counts allowed, including even numbers. Output code range expands from `[0, 2^n_bits - 1]` to `[0, n_comparators]` when the count is non-standard. Adds `last_conversion_time()` method returning `max(c.last_regen_time for c in self.comparators)`, and `last_metastable_sign()` method returning the sign (±1) indicating which side of its nearest threshold the last `v_sampled` landed on. With `tau_regen=0` on every comparator, both methods return 0. |
| `pyDataconverter/architectures/SimpleDAC.py` | Adds `n_levels: Optional[int] = None` (forwarded to `DACBase`) and `code_errors: Optional[np.ndarray] = None` kwargs. When `code_errors` is provided it must be a 1-D array of length `n_levels`; the DAC applies the per-code additive error inside `_convert_input` after the ideal transfer function and before `gain_error`/`offset`/`noise_rms`. Default `None` means no per-code error — existing users see no behaviour change. |
| `pyDataconverter/components/__init__.py` | Export `ResidueAmplifier`. |
| `pyDataconverter/architectures/__init__.py` | Export `PipelinedADC`, `PipelineStage`. |
| `docs/api_reference.md` | New sections for `PipelinedADC`, `PipelineStage`, `ResidueAmplifier`. New subsections under the existing `FlashADC`, `SimpleDAC`, `DifferentialComparator`, `DACBase` entries documenting the new kwargs. |
| `todo/adc_architectures.md` | Pipelined ADC entry updated from "STATUS: IN DESIGN" to "STATUS: PHASE 1 IMPLEMENTED" with a pointer to `PipelinedADC.py` and the example file. |

### 3.3 REUSED — unchanged

`dataconverter.ADCBase`/`InputType`/`OutputType`, `architectures/SARADC.py` (usable as a sub-ADC), `architectures/SimpleADC.py` (also usable as a sub-ADC), `components/reference.py` (`ReferenceLadder` and friends already support arbitrary lengths — the `FlashADC` change just relaxes the class-side constraint), `components/cdac.py` (usable as a sub-DAC), `architectures/R2RDAC.py` / `ResistorStringDAC.py` / `SegmentedResistorDAC.py` / `CurrentSteeringDAC.py` (all usable as sub-DACs without modification), `utils/metrics/_dynamic.py` (FFT analysis used by the example in place of the reference's `ADCanalysis` duplicate), `utils/_bits.py`, `utils/nodal_solver.py`, and every test suite already in place.

---

## 4. Data flow per conversion

Walking through what happens when `PipelinedADC.convert(v_in, dvdt=0.0)` is called. This is the sequence that must produce bit-exact agreement with the reference.

### 4.1 Sequence

1. **Input validation.** Inherited from `ADCBase.convert`: type check on `v_in` against `input_type` (SINGLE → float; DIFFERENTIAL → `(v_pos, v_neg)` tuple).

2. **First-stage sample-and-hold.** Same sequence as `SARADC._sample_input`:
   ```
   v = v_in (or v_pos − v_neg if differential)
   v = v * (1 + gain_error)            # gain error
   v = v + offset                      # input-referred offset
   v = v + N(0, noise_rms)             # kT/C / thermal noise
   v = v + dvdt * N(0, t_jitter)       # aperture jitter
   ```
   The result `v_sampled` is a scalar in the converter's voltage coordinate system.

3. **Cascade — iterate over every `PipelineStage`.**
   ```
   DOUT = 0
   for stage in self.stages:
       raw_code = stage.sub_adc.convert(v_sampled)
       v_dac    = stage.sub_dac.convert(raw_code)       # sub-DAC sees raw code
       delta_v  = v_sampled - v_dac + stage.offset
       target   = stage.residue_amp.gain * delta_v      # "ideal" residue target

       t_regen  = stage.sub_adc.last_conversion_time()
       sign     = stage.sub_adc.last_metastable_sign()
       initial_error = sign * stage.residue_amp.gain * stage.sub_dac.lsb

       t_budget = 1 / (2 * self.fs) - t_regen           # NOT clamped

       v_res = stage.residue_amp.amplify(
           target        = target,
           initial_error = initial_error,
           t_budget      = t_budget,
       )

       shifted_code = raw_code + stage.code_offset
       DOUT        += DOUT * stage.H + shifted_code
       v_sampled    = v_res
   ```

4. **Backend — digitise the final residue.**
   ```
   raw_backend  = self.backend.convert(v_sampled)
   shifted_be   = raw_backend + self.backend_code_offset
   DOUT        += DOUT * self.backend_H + shifted_be
   ```

5. **Optional clip and return.**
   ```
   if self.clip_output:
       DOUT = max(0, min(DOUT, 2**self.n_bits - 1))
   return int(DOUT)
   ```

### 4.2 Invariants

- **When `tau_regen = 0` on every comparator and `settling_tau = 0` on every residue amp, and per-stage `code_offset` matches the reference's `minADCcode`, `PipelinedADC.convert(v)` equals the reference's `PipelinedADC.output(v)` bit-exactly for every input in a dense sweep.**
- **When metastability is enabled on both sides with matched `tauC` / `tauA` / `fs` / per-stage gains, `PipelinedADC.convert(v)` still equals the reference's `PipelinedADC.output(v)` bit-exactly for every input in a dense sweep.**

The comparison tests in §6 enforce both invariants as hard assertions on parameterised configurations.

### 4.3 Why the data flow is deliberately unguarded in three places

1. **`t_budget = 1/(2·fs) − t_regen` is not clamped.** If the slowest comparator's regeneration time exceeds half the clock period, `t_budget` goes negative, `exp(−t_budget/τ_amp)` becomes `exp(+positive) > 1`, and the initial-condition correction overshoots. This is unphysical, but the reference code at line 122 is also unguarded, and matching its arithmetic is the point.

2. **No range check on sub-ADC output codes in the stage.** The stage trusts the sub-ADC's `convert` contract. If the sub-ADC returns an out-of-range integer, that is a sub-ADC bug, not a stage runtime condition to absorb.

3. **Comparator-regeneration `ln(Vc / |v_diff|)` has a `max(|v_diff|, 1e-30)` floor.** One tiny divergence from the reference: `|v_diff|=0` exactly (input sitting on a threshold) would produce `log(0) = −inf` in the reference and poison every downstream calculation. Adding a floor at `1e-30` is small enough to be below normal floating-point precision for any realistic converter voltage and so doesn't perturb any non-degenerate comparison result, but it prevents silent `inf`/`NaN` propagation on the measure-zero edge case. Documented explicitly in the `DifferentialComparator` docstring.

---

## 5. Error handling and validation

All validation fires in `__init__` where possible. Input validation at `convert()` time is inherited from `ADCBase`.

### 5.1 `PipelinedADC.__init__`

Inherited from `ADCBase`: `n_bits` is int in `[1, 32]`; `v_ref` is positive; `input_type` is an `InputType`. Added:

| Parameter | Validation | Error |
|---|---|---|
| `stages` | non-empty list of `PipelineStage` | `ValueError` / `TypeError` naming the offending index |
| `backend` | `ADCBase` instance (any subclass) | `TypeError` |
| `backend_H` | positive number | `ValueError` |
| `backend_code_offset` | int (default 0) | `TypeError` |
| `fs` | positive float, required (no default) | `ValueError` if `fs ≤ 0`; `TypeError` if not a number |
| `noise_rms`, `offset`, `gain_error`, `t_jitter` | same sign/range rules as `SARADC` | `ValueError` on negative where applicable |
| `clip_output` | bool (default `True`) | `TypeError` |

Soft consistency warning (not an error): if stage `i`'s sub-DAC `v_ref` does not match stage `i+1`'s sub-ADC `v_ref`, emit a `RuntimeWarning`. The reference deliberately uses mismatched `FSR_ADC`/`FSR_DAC` per stage to inject static gain errors, so this must not raise.

### 5.2 `PipelineStage.__init__`

| Parameter | Validation |
|---|---|
| `sub_adc` | `ADCBase` instance |
| `sub_dac` | `DACBase` instance |
| `residue_amp` | `ResidueAmplifier` instance |
| `offset` | real float (signed allowed) |
| `code_offset` | int (default 0), arbitrary sign |
| `H` | float, default `residue_amp.gain`, explicit override allowed for trimming/calibration cases |

### 5.3 `ResidueAmplifier.__init__`

| Parameter | Validation |
|---|---|
| `gain` | nonzero float, signed allowed |
| `offset` | real float |
| `slew_rate` | non-negative, `0` or `inf` disables slew limiting |
| `settling_tau` | non-negative, `0` means instantaneous settling |
| `output_swing` | optional `(v_min, v_max)`, `v_max > v_min` if supplied |

**`ResidueAmplifier.amplify(target, initial_error, t_budget)` edge cases** (documented in the class docstring, covered by unit tests):

- `settling_tau = 0` → amp is treated as instantaneous; return `target + offset` regardless of `initial_error` and `t_budget`. This is the "ideal amp" degenerate case and avoids `exp(−t_budget/0)` producing `NaN`.
- `t_budget = +∞` → full settling; return `target + offset`. Avoids `0·∞` indeterminate forms when `initial_error = 0`.
- `initial_error = 0` → short-circuit to `target + offset` without evaluating the exponential. Belt-and-suspenders against the above two cases co-occurring.
- `t_budget < 0` → computed as `target + offset + initial_error · exp(−t_budget/settling_tau)` unmodified; this makes the settling term `> 1` (overshoot), reproducing the reference's unguarded pathological behaviour.

### 5.4 Extensions to existing classes

- **`DifferentialComparator`**: `tau_regen` non-negative; `vc_threshold` positive. Default `(0.0, 0.5)` matches current behaviour and the reference's hardcoded `Vc = 0.5`.
- **`FlashADC`**: `n_comparators` positive int if supplied; `reference.n_references` must equal `n_comparators`.
- **`DACBase`**: `n_levels` positive int `≥ 2` if supplied.
- **`SimpleDAC`**: `code_errors` if supplied must be a 1-D array of length `n_levels`.

### 5.5 Error message conventions

All validation raises `TypeError` or `ValueError`, matches the `SARADC` / `FlashADC` message style, and names both the offending parameter and its observed value. Examples:

```python
raise ValueError("PipelinedADC requires at least one stage, got empty list")
raise TypeError(f"stages[{i}] must be a PipelineStage instance, got {type(stages[i]).__name__}")
raise ValueError(f"n_levels must be >= 2, got {n_levels}")
raise ValueError(f"fs must be positive when tau_regen > 0 (stage {i}), got fs={fs}")
raise TypeError(f"code_errors must have length {n_levels}, got {len(code_errors)}")
```

---

## 6. Testing strategy

### 6.1 Layer 1 — vendored reference and bit-exact comparison

**`tests/_reference/adc_book_pipelined.py`** contains just the five classes from the upstream file: `PipelinedADC`, `Stage`, `subADC`, `subDAC`, `sumGain`. Deleted: the ~630-line `__main__` block, the `ADCanalysis` function (duplicate of existing `utils/metrics/_dynamic.py`), the `print` statements at lines 150/151/169/170, and the `import matplotlib` lines. File header:

```python
"""
Vendored reference for validation of pyDataconverter's PipelinedADC.

SOURCE:   https://github.com/manar-c/adc_book (python/pipelinedADC.py)
IMPORTED: 2026-04-13, commit <short sha>
AUTHOR:   Manar El-Chammas (attribution preserved)

This file is READ-ONLY. It is used exclusively by
tests/test_pipelined_adc_vs_reference.py to cross-check the
pyDataconverter implementation against the vetted original. Do not
modify it — any bug found here should be investigated in the upstream
repository. If the upstream changes, re-vendor the relevant slice and
update the IMPORTED date above.

Classes included: PipelinedADC, Stage, subADC, subDAC, sumGain.
Nothing else from the upstream file is needed.
"""
```

**`tests/test_pipelined_adc_vs_reference.py`** is a single parameterised pytest module driving four canonical configurations. Each test builds the reference and the new implementation from a shared config dict, runs a 4001-point linear sweep across `[−FSR/2 · 0.99, +FSR/2 · 0.99]`, and asserts `int(reference.output(v)) == int(new.convert(v))` for every sample. The assertion message names the input voltage, both codes, and the config name so that any failure points directly at the offending sample.

The four configurations:

1. **`ideal_12bit`** — `Nstages=2`, `N=[8, 1026]`, `FSR=[1, 1]`, `G=[4, 512]`, `minADCcode=[−1, 0]`. Canonical reference `__main__` line 278. Tests the cascade, the digital combiner, and the relaxed `FlashADC` / `SimpleDAC` / `DACBase` extensions at their full exercise.
2. **`stage0_dac_error`** — identical to `ideal_12bit` but with the reference's per-code DAC error array applied to stage 0's sub-DAC: `[0, −0.2, 0.3, 0.05, −0.15, 0, 0.3, −0.3, 0] × 0.001`. Tests the `SimpleDAC.code_errors` path.
3. **`stage0_gain_error`** — identical to `ideal_12bit` but with `G[0] = 3.988` (0.3% gain error on stage 0). Tests `ResidueAmplifier(gain=3.988)` against the reference.
4. **`metastability_canned`** — `ideal_12bit` plus `timeResponse=[True, False]`, `SampleRate=500e6`, `tau_comparator=[30e-12, 0]`, `tau_amplifier=[50e-12, 0]`. Tests the metastability + settling coupling exactly, and is the binary correctness check for the entire ResidueAmplifier / `last_conversion_time` / `last_metastable_sign` plumbing.

Randomness policy: all four configurations are fully deterministic (no `noise_rms`, no `noisesigma`). Any future comparison config that needs RNG must seed both sides with the same `np.random.seed(...)` immediately before each `convert` call, in the same order, to match the reference's use of raw `np.random` calls.

### 6.2 Layer 2 — unit tests for new and extended classes

**`tests/test_pipelined_adc.py`** covers:

- **`TestResidueAmplifier`**: ideal amplification when `settling_tau=0` and `initial_error=0`; perfect settling when `t_budget → ∞` with nonzero `initial_error`; no settling when `t_budget=0`; exponential settling fraction (`amplify(0, 100, 2·τ) ≈ 100·e^−2`); divergence on negative `t_budget` as a documented pathological case; independence of gain/offset/slew from the settling term; constructor validation on `gain=0`, `settling_tau<0`, `output_swing` ordering.
- **`TestPipelineStage`**: ideal stage reproduces `G·(v_in − v_dac) + offset`; `code_offset` does not affect the sub-DAC input (sub-DAC sees raw code, combiner sees offset code); metastability sign is translated into the correct `initial_error` sign at the residue amp.
- **`TestPipelinedADC`**: construction validation (empty stages, wrong types, invalid `backend_H`, invalid `fs`); single-stage cascade with a trivial input; first-stage S&H non-idealities applied correctly; input-type switching (SINGLE vs DIFFERENTIAL); `clip_output` true vs false; digital-combiner formula against a closed-form computation.

Extensions added to existing test files:

- **`tests/test_FlashADC.py`**: `TestFlashADCRelaxed` class covering arbitrary `n_comparators` (including even), backward compatibility when the kwarg is omitted, `last_conversion_time()` returning the maximum regen across the bank, and `last_metastable_sign()` flipping at the midpoint.
- **`tests/test_comparator.py`**: `TestDifferentialComparatorRegen` covering zero-regen default, the `ln(Vc / |v_diff|)` formula, and the `1e-30` floor on `|v_diff|=0`.
- **`tests/test_dacbase.py`** (or the closest existing equivalent): `TestDACBaseRelaxed` covering the `n_levels` kwarg, `lsb = v_ref / (n_levels − 1)`, and out-of-range code validation.
- **SimpleDAC tests**: a new class covering `code_errors` application, length-mismatch validation, and interaction order with `offset`/`gain_error`/`noise_rms`.

### 6.3 Layer 3 — example as smoke test

**`examples/pipelined_adc_example.py`** constructs the `ideal_12bit` configuration, runs a coherent sine, prints SNDR from `utils/metrics`, and plots the residue trace and the FFT. Styled after `examples/sar_adc_example.py`. Functions as an end-to-end sanity check whose only pass criterion is "runs without error and prints an SNDR within 0.5 dB of the ideal 12-bit theoretical value."

### 6.4 Coverage target

New code targets ≥ 95% statement coverage under `pytest --cov`. Extensions to existing classes target 100% coverage of the new lines. The comparison layer is the binary correctness check: four passing configs means the entire data flow is right; any failure points at a specific sample and a specific config.

---

## 7. Phase boundaries

### 7.1 Phase 1 — this design, ships as one commit series

Deliverables (in rough build order):

1. `ResidueAmplifier` component.
2. `DACBase.n_levels` relaxation.
3. `SimpleDAC.code_errors` kwarg.
4. `FlashADC.n_comparators` relaxation plus `last_conversion_time()` and `last_metastable_sign()` methods.
5. `DifferentialComparator.tau_regen` / `vc_threshold` kwargs.
6. `PipelineStage` and `PipelinedADC` classes.
7. Vendored reference and its `__init__.py`.
8. Unit tests (`test_pipelined_adc.py`, extensions to `test_FlashADC.py` / `test_comparator.py` / SimpleDAC / DACBase tests).
9. Comparison tests (`test_pipelined_adc_vs_reference.py`).
10. Docs (`api_reference.md` sections, `todo/adc_architectures.md` status update) and example file.

Acceptance criteria for Phase 1 completion:

1. All four comparison configs pass bit-exact on a 4001-point sweep.
2. New code has ≥ 95% statement coverage.
3. The existing 826 tests still pass — no regressions from the `FlashADC` / `DACBase` / `SimpleDAC` / `DifferentialComparator` extensions.
4. `examples/pipelined_adc_example.py` runs end-to-end and prints SNDR within 0.5 dB of the 12-bit theoretical ideal.
5. `docs/api_reference.md` has complete entries for every new/extended public API.
6. `todo/adc_architectures.md` entry updated with a pointer to the shipped classes.

### 7.2 Phase 2 — deferred, noted in `todo/adc_architectures.md`

- First-stage S&H as an explicit `SampleAndHold` component (currently rolled into `PipelinedADC.__init__` to match `SARADC`'s pattern). Unlocks per-channel S&H reuse for the future TI-ADC.
- Residue-amp slew-rate limiting in the comparison harness. The `slew_rate` parameter exists in Phase 1, but the reference has no slew model, so Phase 1 can't cover it in comparison tests.
- `SARADC` metastability plumbing. Phase 1 adds `tau_regen` only to `DifferentialComparator`, with `FlashADC` aggregating across the bank. `SARADC` continues to report `0` for `last_conversion_time()` and `last_metastable_sign()`. Phase 2 adds the same mechanism through `SARADC`'s bit-cycling loop.
- Additional pipelined non-idealities: reference-voltage noise, sub-DAC capacitor mismatch, stage crosstalk, 1/f noise in the residue amp.
- Characterisation helpers in `utils/characterization.py` tailored to pipelined ADCs (stage-level residue transfer curves, per-stage INL sweeps).

### 7.3 Phase 3 — separate design sessions each

- **`MDAC` class** as a `DACBase` subclass bundling S&H + sub-DAC + residue amp. Simplifies `PipelineStage`'s constructor surface (one MDAC instead of separate sub-DAC and residue amp), unlocks pipelined-style DACs in other contexts. Not coming back into this design because it's a refactor of the stage boundaries.
- **Time-interleaved ADC wrapper** composing any ADC subclass as a per-channel sub-ADC. Separate brainstorming session; the interesting content is in the channel mismatch models, not the cascade structure.
- **`convert_batch` API** — parked in `todo/parking_lot.md` §1. Not coming back in this work.
- **Sigma-Delta ADC / DAC pair** — next HIGH-priority item after pipelined.

### 7.4 Deliberately NOT in scope — any phase of pipelined-ADC work

- New sub-ADC architectures beyond what's already in pyDataconverter. The pluggable design means any future `ADCBase` subclass becomes usable as a sub-ADC automatically.
- Subranging / two-step ADCs (conceptually pipelined with unity inter-stage gain, but a separate architecture entry in `todo/adc_architectures.md`).
- VCO- / TDC-based ADCs (completely different physical model).
- Any changes to `utils/metrics`, `utils/characterization`, or the existing plotting helpers beyond the new example file.
- Any edit to the vendored reference in `tests/_reference/`. That file is read-only.

---

## Appendix A — Algebraic derivation of the metastability model

This appendix preserves the derivation that connects the reference's imperative metastability logic to the `ResidueAmplifier.amplify(target, initial_error, t_budget)` contract. It is what makes the component-level decomposition in §2.2 and the data flow in §4.1 provably bit-exact to the reference.

### A.1 Reference code (annotated)

From `adc_book/python/pipelinedADC.py` lines 83–134, with `timeResponse` enabled:

```
stageoutput_0  = G * (v_in − DACOUT)                           # line 86
                                                                # DACOUT is computed from
                                                                # the fully-resolved sub-ADC code
sign           = +1 if subADC.ref[closest] > v_in else −1      # lines 98–101
delta_vin      = |subADC.ref - v_in|                           # line 95
deltaOutput    = G * (−subDAC.FSR / subDAC.N)                  # line 107
               = −G · sub_dac_lsb

# Pre-adjust (line 109):
stageoutput_1  = stageoutput_0 − sign · deltaOutput
               = stageoutput_0 + sign · G · sub_dac_lsb

# Regeneration and settling (lines 119–127):
t_regen        = tau_comparator · ln(Vc / delta_vin[closest])
TR             = 1/(2·FS) − t_regen
Gerror         = 1 − exp(−TR / tau_amplifier)

# Re-add (line 132):
stageoutput_2  = stageoutput_1 + sign · deltaOutput · Gerror
               = stageoutput_1 − sign · G · sub_dac_lsb · (1 − exp(−TR/tau_amp))
```

### A.2 Collapsing the two adjustments

Substituting `stageoutput_1` into `stageoutput_2`:

```
stageoutput_2
  = stageoutput_0 + sign · G · sub_dac_lsb − sign · G · sub_dac_lsb · (1 − exp(−TR/tau_amp))
  = stageoutput_0 + sign · G · sub_dac_lsb · [1 − (1 − exp(−TR/tau_amp))]
  = stageoutput_0 + sign · G · sub_dac_lsb · exp(−TR/tau_amp)
```

So the reference's final residue is:

> `stageoutput_final = ideal_residue + sign · G · sub_dac_lsb · exp(−TR / tau_amp)`

where `ideal_residue = G · (v_in − DACOUT)`, `sign = +1` when the nearest sub-ADC threshold is above `v_in`, and `TR = 1/(2·FS) − tau_comparator · ln(Vc / |v_in − nearest_threshold|)`.

### A.3 Matching this with `ResidueAmplifier.amplify`

`ResidueAmplifier.amplify(target, initial_error, t_budget)` models a first-order linear system starting at `target + initial_error` and settling toward `target` with time constant `settling_tau`:

```
v_out(t_budget) = target + initial_error · exp(−t_budget / settling_tau)
```

Feed it:

- `target = G · (v_sampled − v_dac) = stageoutput_0`
- `initial_error = G · sub_dac_lsb · sign`
- `t_budget = 1/(2·fs) − t_regen = TR`
- `settling_tau = tau_amp`

and it returns:

```
target + initial_error · exp(−t_budget / settling_tau)
  = stageoutput_0 + G · sub_dac_lsb · sign · exp(−TR / tau_amp)
```

This is exactly `stageoutput_final` from A.2. The new decomposition and the reference's imperative sequence compute the same value, up to floating-point precision, for every input.

### A.4 Responsibility split

Where each piece of this derivation lives in the new architecture:

| Quantity | Owner | How it is obtained |
|---|---|---|
| `v_in − nearest_threshold`, `sign` | `FlashADC` | Computed from the reference ladder and the last `v_sampled`. Exposed via `last_metastable_sign()`. |
| `t_regen` | `DifferentialComparator` aggregated by `FlashADC` | `tau_regen · ln(vc_threshold / max(|v_diff|, 1e-30))` per comparator; `FlashADC.last_conversion_time()` returns the max across the bank. |
| `sub_dac_lsb` | `DACBase` (via `SimpleDAC` or any subclass) | `v_ref / (n_levels − 1)`, computed at construction. Exposed via `sub_dac.lsb`. |
| `G` (stage gain) | `ResidueAmplifier` | Stored as `residue_amp.gain`. Read by `PipelineStage` when computing `target` and scaling `initial_error`. |
| `target`, `initial_error` | `PipelineStage` | Computed from the above. Passed to `residue_amp.amplify`. |
| `target + initial_error · exp(−t_budget / tau_amp)` | `ResidueAmplifier.amplify` | Settling exponential. Core formula. |
| `t_budget = 1/(2·fs) − t_regen` (unclamped) | `PipelineStage` | Pure coordination — no physics. |

Nothing about this derivation assumes a particular sub-ADC topology or a particular DAC topology. Any `ADCBase` that implements `last_conversion_time()` and `last_metastable_sign()` can drive the metastability coupling; any `DACBase` that exposes `lsb` can be the sub-DAC. Classes that do not model metastability (like the current `SARADC`) return `0` from both methods and degenerate the model to the ideal case.
