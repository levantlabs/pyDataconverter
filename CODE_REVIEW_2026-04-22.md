# pyDataconverter Code Review

**Review Date**: 2026-04-22
**Reviewer**: Claude (comprehensive codebase review)
**Scope**: Code quality, consistency, logic, and documentation

---

## Status Tracking

Legend: PENDING · DECIDED (plan agreed, code not changed) · FIXED · FALSE POSITIVE · WON'T FIX

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| 1-W1 | Empty top-level `__init__.py` | FIXED | Fully-namespaced approach. Added module docstring + `__version__` via `importlib.metadata`. Public API remains under `pyDataconverter.architectures`, `pyDataconverter.components`, `pyDataconverter.dataconverter`, `pyDataconverter.utils`. |
| 1-W2 | Empty `utils/__init__.py` | FIXED | Added explicit submodule re-exports (signal_gen, fft_analysis, characterization, nodal_solver, metrics, visualizations) with `__all__`. Created `utils/visualizations/__init__.py`. Broke an incidental util→architecture circular import in `dac_plots.py` by localising the `SimpleDAC` import. |
| 2.1 | ADC `input_type` default inconsistency | FIXED | Harmonised SARADC / FlashADC / PipelinedADC defaults to `InputType.DIFFERENTIAL` (matching `ADCBase`). Updated test fixtures (test_SARADC, test_FlashADC, test_sar_generalisation, test_metrics, test_characterization) to explicitly pin `input_type=InputType.SINGLE` where they were relying on the old default. FlashADC `__main__` demo also pinned. All 973 tests pass. |
| 2.2 | Quantization mode not unified | WON'T FIX (documented) | Architectural distinction is intentional: `QuantizationMode` parameterises behavioural ADCs (`SimpleADC`) and metric helpers; structural ADCs (SARADC/FlashADC/PipelinedADC/TI-ADC) are FLOOR-by-construction. Documented the applicability in `QuantizationMode`'s docstring and in each structural-ADC module docstring. |
| 2.3 | `n_levels` support inconsistent across DACs | WON'T FIX (documented) | Architectural distinction parallel to §2.2: `n_levels` is a behavioural-DAC feature (`SimpleDAC`) used primarily for pipelined / TI-ADC sub-DACs matching non-power-of-two flash sub-ADC outputs. Structural DACs are tied to `2**n_bits` by topology (R-2R is strict; others are power-of-two in their current implementations). Documented in `DACBase` docstring and each structural-DAC class docstring. |
| 2.4 | `__repr__` inconsistency | PARTIAL | `CurrentSteeringDAC` repr key renamed `output=` → `output_type=`. Derived finding (API consistency): all three structural single-ended DACs — `ResistorStringDAC`, `R2RDAC`, `SegmentedResistorDAC` — now expose `output_type` explicitly in their signatures and reject DIFFERENTIAL with a clear error pointing to the "instantiate two and combine externally" composition pattern. Class docstrings updated. 4 new tests added. Remaining §2.4 sub-items (decided to skip as cosmetic): SARADC embedded CDAC repr length, `TimeInterleavedADC` missing `v_ref`/`input_type`/`M=` naming, MultibitSARADC/NoiseshapingSARADC dropping `cdac`. |
| 3.1 | FlashADC differential ref range `±v_ref/4` | FALSE POSITIVE (doc clarified) | Code is functionally correct — the `[-v_ref/4, +v_ref/4]` ladder combined with symmetric `(v_refp=t[i], v_refn=t[n-1-i])` pairing produces the full `[-v_ref/2, +v_ref/2]` effective differential threshold range. Review acknowledged correctness but flagged the convention as "subtle". Added a "Reference ladder convention" section to the `FlashADC` class docstring so the scaling is visible without reading block comments. |
| 3.2 | DAC repr missing `seed` | FIXED (extended) | Beyond the review's narrow scope: added construction-time `seed` kwarg (or wired it through to repr+storage where it already existed) on every class that performs random mismatch draws at construction. Phase A — already accept seed, added storage + conditional repr: `ResistorStringDAC`, `R2RDAC`, `SegmentedResistorDAC`, `TimeInterleavedADC`. Phase B — added `seed` kwarg + reproducible draw: 5 CDAC classes (`SingleEndedCDAC`, `DifferentialCDAC`, `SegmentedCDAC`, `RedundantSARCDAC`, `SplitCapCDAC`), `ReferenceLadder`, `FlashADC`. CDAC refactor routes mismatch through the existing seeded `apply_mismatch` path. FlashADC streams sub-RNGs via `SeedSequence.spawn` so offset and resistor-mismatch draws are independent within a seeded instance. Per-conversion noise (sampling, jitter, comparator/reference noise) deliberately stays on `np.random` global state — see Level 2/3 deferred. 26 new tests covering same-seed match, different-seed distinguish, seed-in-repr-when-set, seed-omitted-when-None. |
| 3.3 | `SegmentedCDAC` private member access | FIXED | Added a protected `SingleEndedCDAC._voltage_from_bits(bits)` helper that computes the DAC voltage from an explicit bit pattern. `SingleEndedCDAC.get_voltage` and `SegmentedCDAC.get_voltage` both route through it — SegmentedCDAC no longer touches `_cdac._cap_weights` / `_cdac._cap_total` directly. Encapsulation restored without any per-call array copy (the public property would have allocated on every conversion). |
| 3.4 | `MultibitSARADC` ignores `dvdt` | FALSE POSITIVE | Reviewer expected `dvdt` to be passed to each `comparator.compare()` call inside `_run_sar`. Architecturally wrong: a SAR samples the input once then bit-cycles against the held value, so aperture jitter applies once at the S/H — which is exactly where `_dvdt` is used (`SARADC._sample_input:321–322`, inherited by `MultibitSARADC`). The comparator API takes `time_step` (for bandwidth filtering), not `dvdt`. Empirically verified: `dvdt + t_jitter` produces conversion spread on `MultibitSARADC` (5 distinct codes / 50 conversions); `dvdt=0` produces deterministic output. |
| 3.5 | `NoiseshapingSARADC` assumes FLOOR | FALSE POSITIVE | Premise wrong: `NoiseshapingSARADC.__init__` forwards `**kwargs` to `SARADC.__init__`, which has explicit named params and no `quant_mode`. Passing `quant_mode=SYMMETRIC` raises `TypeError`, so the DC-bias scenario can't occur. The reconstruction formula itself is correct for the actual FLOOR quantization (single-ended: bin midpoint at `(k+0.5)·LSB`; differential: same shifted by `−v_ref/2` to align with `DifferentialCDAC`'s symmetric threshold layout). Consistent with §2.2 — SARADC is FLOOR-by-construction. |
| 3.6 | `convert_waveform` return type annotation | FIXED (cosmetic) | Review's own conclusion was "no issues found" — implementations correctly return int arrays via `dtype=int`. Added `-> np.ndarray` return-type annotation to the two `def` signatures (`ADCBase.convert_waveform`, `TimeInterleavedADC.convert_waveform`) so the return type is visible to static checkers / IDE introspection. The dtype=int specifics stay in the docstring (numpy generic typing for dtype is verbose and not idiomatic). No runtime change. |
| 3.7 | FlashADC XOR encoder edge case | FALSE POSITIVE | No real overflow risk within the supported range. `ADCBase` validates `1 ≤ n_bits ≤ 32`; at the max, `values[i]` reaches `2^32 − 1` and `np.flatnonzero` returns int64 on 64-bit platforms — `bitwise_or.reduce` supports up to `2^63 − 1`, leaving ~31 bits of headroom. The review itself acknowledged this was not a concern for practical n_bits. |
| 3.8 | R2RDAC termination resistor double-use | FIXED | The LSB switch arm and the LSB-end termination are physically separate 2R resistors but were sharing `r2_values[n-1]`, so a Monte-Carlo sweep underestimated variance by treating them as perfectly correlated.  Now `r2_values` is length `n_bits+1`: indices 0..n-1 are switch arms, index n is the dedicated termination, each with an independent mismatch draw.  Class docstring + `_build_network` docstring rewritten to reflect the correct topology.  1 new test verifies the array length and that the LSB switch / termination draws are distinct.  Note: any user that pinned a seed against the prior model will see different `r_values` arrays — the new behaviour is more physically accurate. |
| 4.1 | Outdated module version-history blocks | FIXED | Replaced the multi-line `Version History:` block in 15 modules with a single-line `First written <date>; see git log for the change history.` pointer. Removes the maintenance burden of keeping in-file version logs in sync with code changes; git is now the single source of truth. Original first-written dates preserved. |
| 4.2 | Missing parameter documentation | FIXED | Two sub-items addressed. (a) `ResidueAmplifier`: added a "Gain contract" block at the top of the class docstring (before the Attributes table) showing the canonical caller-pre-multiplies-by-gain pattern with a worked example, and updated the `gain` attribute description to point at it. The contract was already correct in the `amplify()` method docstring; this surfaces it where a class-level reader will see it first. (b) `TimeInterleavedADC._convert_input`: expanded the docstring to spell out that `dvdt` is used in two places — the TI-level timing-skew mismatch (`dvdt * skew_k`) and the per-channel sub-ADC aperture jitter (forwarded as `convert(..., dvdt=self._dvdt)`) — and notes how the two effects compose linearly. |
| 4.3 | No Sphinx/RTD setup | DEFERRED | An exploratory Sphinx scaffold currently lives at `docs/source/` (untracked) but the user has flagged the output as poor quality and intends to redo the documentation build from scratch as a separate, larger effort. Closing this review item as DEFERRED rather than trying to polish the prototype. The prototype's notes-to-self in `docs/SPHINX_IMPROVEMENTS.md` (also untracked) should not be treated as a binding plan. |
| 4.4 | `ADCBase` docstring covers DAC too | FIXED | Reviewed the module docstring: line 5 already said "interfaces for both ADC and DAC implementations" so the narrow concern was largely a non-issue, but the "Classes:" listing genuinely omitted the three public enums (`InputType`, `OutputType`, `QuantizationMode`). Replaced with a structured "Public API:" block grouping base classes and enums separately, with one-line summaries for each enum (and a pointer to `QuantizationMode`'s applicability section). Both the review's narrow concern and the real gap closed. |
| 5.1 | Inconsistent type annotations | FIXED (narrow scope) | Normalised the specific inconsistency the review flagged: `np.random.Generator` was a string forward-ref (`"np.random.Generator"`) in `components/capacitor.py:53,149` but a bare class reference everywhere else (`TimeInterleavedADC.py`, `utils/signal_gen.py`). Numpy 1.17+ has it as a real importable class, so the string form isn't needed. Dropped the quotes — codebase now uses one form. The broader concern (≈119 of 275 `def` lines lack a return annotation) is out of scope for review closure; will be picked up by a future typing audit. |
| 5.2 | No `py.typed` marker | FIXED | Created the empty PEP 561 marker file `pyDataconverter/py.typed` and updated `setup.py` to ship it via `package_data` + `include_package_data=True`. Type-checker consumers (mypy, pyright) now pick up the inline annotations after `pip install pyDataconverter`. |
| 5.3 | Large `__main__` demo blocks | FIXED | Created `examples/` directory at repo root with 7 standalone demo scripts (one per source module's old `__main__` block) plus a `README.md` index. Stripped the corresponding `__main__` blocks from 6 modules (SARADC, FlashADC, SimpleADC, SimpleDAC, comparator, signal_gen) and removed the module-level `demo_fft_analysis()` function from `fft_analysis.py`. ~660 lines of demo code moved out of the library. Six tests that exercised the demo blocks for coverage were removed (demo coverage isn't library testing). 1000 tests still pass (was 1006; 6 demo-coverage tests removed). |
| 5.4 | `warnings` imported inside functions | PENDING | |
| 5.5 | Hardcoded magic numbers / format specifiers | PENDING | |
| 5.6 | No abstract property consistency check for `CDACBase.n_bits` | PENDING | |
| 5.7 | `SimpleDAC.convert_sequence` silent `code_errors` skip | PENDING | |
| 5.8 | `apply_mismatch` returns `None` without doc note | PENDING | |
| 6.1 | R2RDAC/ResistorStringDAC compute all codes at construction | PENDING | |
| 6.2 | `SegmentedCDAC.get_voltage` allocates per call | PENDING | |
| 6.3 | `SimpleDAC.convert_sequence` noise after repeat | PENDING | |
| 7.1 | No TI-ADC hierarchical tests | PENDING | |
| 7.2 | No `NoiseshapingSARADC` tests | PENDING | |
| 7.3 | No metastability coupling tests for `MultibitSARADC` | PENDING | |
| 7.4 | No `PipelinedADC` tests | PENDING | |
| 7.5 | Missing integration tests | PENDING | |
| 8.1 | Dead code in FlashADC | PENDING | |
| 8.2 | Visualization demo runs at import in FlashADC | PENDING | |
| 8.3 | `TimeInterleavedADC.__repr__` missing sub-ADC repr | PENDING | |
| 8.4 | `CurrentSteeringDAC.dac_currents` recomputes per access | PENDING | |
| 8.5 | `ResidueAmplifier.slew_rate` stored but unused | PENDING | |
| 8.6 | Demo code runs at module import in `comparator.py` | PENDING | |

---

## 1. Architecture & Design (Overall)

### Strengths
- **Clean class hierarchy**: `ADCBase` and `DACBase` abstract classes provide consistent interfaces for all converter implementations
- **Good component separation**: Comparator, C-DAC, reference, decoder, current source, etc. are properly abstracted and composable
- **Comprehensive architectures**: SAR, Flash, Pipeline, TI-ADC, SimpleADC/DAC, R-2R, Current-Steering, Resistor-String
- **Rich non-idealities modeling**: Noise, offset, gain error, jitter, hysteresis, mismatch, settling, metastability

### Weaknesses
- **Missing `__init__.py` exports** [FIXED]: ~~`pyDataconverter/__init__.py` is empty and `utils/__init__.py` is also empty, making public API discoverability poor~~. Resolved via fully-namespaced approach: the top-level package now carries a module docstring and `__version__`, `utils/__init__.py` explicitly re-exports its submodules, and `utils/visualizations/__init__.py` was created. Users access the API via `pyDataconverter.architectures.*`, `pyDataconverter.components.*`, `pyDataconverter.dataconverter.*`, and `pyDataconverter.utils.*`.
- **No top-level `__all__`** [WON'T FIX]: Deliberate design choice under the fully-namespaced model — `from pyDataconverter import *` is not an intended usage pattern, analogous to `scipy`/`sklearn`. Users import from the semantic subpackage.

---

## 2. Consistency Issues

### 2.1 Input/Output Type Enums
| Class | Input Type Param | Default | Enum Location |
|-------|-----------------|---------|---------------|
| `ADCBase` | `input_type` | `InputType.DIFFERENTIAL` | `dataconverter.py` |
| `SimpleADC` | `input_type` | `InputType.DIFFERENTIAL` | `dataconverter.py` |
| `SARADC` | `input_type` | `InputType.SINGLE` | `SARADC.py` |
| `FlashADC` | `input_type` | `InputType.SINGLE` | `FlashADC.py` |
| `SimpleDAC` | `output_type` | `OutputType.SINGLE` | `dataconverter.py` |
| `DACBase` | `output_type` | `OutputType.SINGLE` | `dataconverter.py` |

**Issue**: `SARADC` defaults to `SINGLE`, `FlashADC` defaults to `SINGLE`, but `ADCBase` defaults to `DIFFERENTIAL`. This inconsistency is confusing and can cause subtle bugs when users expect ADCs to be differential by default.

**Status: FIXED (2026-04-24)** — Harmonised all ADC architecture defaults to `InputType.DIFFERENTIAL` to match `ADCBase`. Affected files:
- `pyDataconverter/architectures/SARADC.py:105` (SARADC)
- `pyDataconverter/architectures/FlashADC.py:86` (FlashADC)
- `pyDataconverter/architectures/PipelinedADC.py:176` (PipelinedADC; same class of issue, not explicitly in the review but harmonised for consistency)

`MultibitSARADC` and `NoiseshapingSARADC` inherit via `**kwargs` and auto-follow. `TimeInterleavedADC` inherits from its template. `SimpleADC` already defaulted to DIFFERENTIAL (no change).

Tests that relied on the old SINGLE default (scalar `vin` inputs) were updated to explicitly pass `input_type=InputType.SINGLE`. FlashADC's `__main__` demo block was also pinned to SINGLE. All 973 tests pass.

### 2.2 Quantization Mode
- `SimpleADC` supports `QuantizationMode.FLOOR` and `SYMMETRIC` via `quant_mode` parameter
- `SARADC` hardcodes FLOOR quantization (documented in module docstring)
- Other ADCs do not expose this parameter at all

**Issue**: No unified quantization mode abstraction across architectures.

**Status: WON'T FIX — documented as an architectural distinction (2026-04-24)**

The split is intentional, not an oversight:

- `SimpleADC` is a *behavioural* ADC — it applies the quantization formula
  directly (see `SimpleADC._quantize`), so both FLOOR and SYMMETRIC are cheap
  parametric choices.
- `SARADC` / `FlashADC` / `PipelinedADC` / `TimeInterleavedADC` are
  *structural* ADC models. Their output comes from hardware dynamics
  (CDAC binary search, uniform ReferenceLadder comparator thresholds,
  residue-amp chains). LSB = v_ref / 2^N is baked into the physics —
  changing it would require shifting every comparator threshold (not what
  SYMMETRIC conventionally means) or post-processing codes arithmetically
  (a pedagogical fiction that misrepresents real silicon, which is itself
  FLOOR-like).
- DSP-style (zero-mean / SYMMETRIC) analysis of a structural ADC's code
  stream is already supported: `pyDataconverter.utils.metrics` functions
  (`calculate_gain_offset_error`, `calculate_adc_static_metrics`, etc.)
  accept `quant_mode` at metric-computation time. Use that path rather
  than expecting the ADC to remap codes.

Documentation updated (2026-04-24):
- `pyDataconverter/dataconverter.py` — `QuantizationMode` docstring now
  has an "Applicability" section spelling out which ADCs/helpers honour it.
- `pyDataconverter/architectures/SARADC.py` — existing "Quantisation"
  section expanded to state FLOOR is architectural and not configurable,
  with a pointer to the metric-time override.
- `pyDataconverter/architectures/FlashADC.py` — new "Quantisation" note in
  module docstring.
- `pyDataconverter/architectures/PipelinedADC.py` — new "Quantisation" note.
- `pyDataconverter/architectures/TimeInterleavedADC.py` — new "Quantisation"
  note explaining delegation to the channel template.

### 2.3 `n_levels` Parameter
- `DACBase` accepts optional `n_levels` (defaults to `2**n_bits`)
- `SimpleDAC` forwards it correctly
- `ResistorStringDAC`, `R2RDAC`, `SegmentedResistorDAC`, `CurrentSteeringDAC` do NOT accept `n_levels` — they're hardcoded to power-of-2

**Issue**: Inconsistent support for non-power-of-2 DACs across the codebase.

**Status: WON'T FIX — documented as an architectural distinction (2026-04-24)**

Parallel to §2.2 (behavioural vs structural split). The practical motivation
for `n_levels` is composing pipelined / TI-ADC stages where a flash sub-ADC
with N comparators produces N+1 codes and needs a matching sub-DAC with
N+1 distinct output levels. `SimpleDAC` is the designated tool for this
role — it is already used this way in `test_pipelined_adc.py` (lines 25,
81, 121) and `test_ti_adc.py:427` as
`SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0, ...)`.

The structural DACs are tied to `2**n_bits` by topology:
- **R2RDAC** is strict — the R-2R ladder is binary-weighted by construction.
- **ResistorStringDAC**, **SegmentedResistorDAC**, **CurrentSteeringDAC**
  are power-of-two in their current implementations (the resistor strings,
  R-2R fine sub-DAC, and decoder/element arrays are all sized for
  `2**n_bits`). In principle these could be extended to non-power-of-two,
  but there is no demonstrated demand beyond what `SimpleDAC` already
  satisfies.

Documentation updated (2026-04-24):
- `pyDataconverter/dataconverter.py` — `DACBase` class docstring gained a
  "Level count (n_levels)" section explaining behavioural vs structural
  applicability, with the pipelined / TI-ADC sub-DAC motivation spelled out.
- `pyDataconverter/architectures/R2RDAC.py` — class docstring notes R-2R is
  strictly power-of-two; points non-power-of-two users to `SimpleDAC`.
- `pyDataconverter/architectures/ResistorStringDAC.py` — same pattern.
- `pyDataconverter/architectures/SegmentedResistorDAC.py` — same pattern
  (R-2R fine sub-DAC pins topology to power-of-two).
- `pyDataconverter/architectures/CurrentSteeringDAC.py` — same pattern.

### 2.4 Repr Inconsistency
| Class | Repr Includes |
|-------|--------------|
| `ADCBase` | `n_bits, v_ref, input_type` |
| `DACBase` | `n_bits, v_ref, output_type` |
| `SARADC` | All above + `cdac=...` (full repr of CDAC object) |
| `TimeInterleavedADC` | `M, fs, template, n_bits` (missing `v_ref, input_type`) |
| `ResidueAmplifier` | Only non-default params |

**Issue**: `SARADC.__repr__` embeds the full CDAC repr which can be very long; `TimeInterleavedADC` omits `v_ref, input_type` from its base class.

**Status: PARTIAL (2026-04-24)** — working through sub-items one at a time.

Closed so far:
- `CurrentSteeringDAC.__repr__` used the key `output=` instead of `output_type=` (inconsistent with every other DAC/ADC repr). Renamed to `output_type=`.
- Derived finding during discussion: `ResistorStringDAC`, `R2RDAC`, and
  `SegmentedResistorDAC` silently rejected
  `output_type=OutputType.DIFFERENTIAL` with a generic `TypeError` because
  the base-class kwarg wasn't surfaced in their signatures. Physical
  reality is that each of these DACs is inherently single-ended — for
  differential output a user would instantiate two objects and combine
  them externally. Updated all three `__init__` methods to accept
  `output_type` explicitly with a `ValueError` on DIFFERENTIAL that teaches
  the composition pattern. Class docstrings gained "Output type" sections.
  4 new tests cover the accept-SINGLE and reject-DIFFERENTIAL paths.

Sub-items decided to skip as cosmetic:
- SARADC embedded CDAC repr length (100+ chars, redundant n_bits/v_ref).
  Useful for debugging; no functional harm.
- `MultibitSARADC` / `NoiseshapingSARADC` dropping `cdac` while parent
  `SARADC` includes it. Internal inconsistency only noticeable when
  comparing repr strings across classes.
- `TimeInterleavedADC` missing `v_ref` and `input_type` from repr; uses
  `M=` rather than `channels=`. Attributes still accessible via
  `ti.v_ref` / `ti.input_type`; repr omission is ergonomic not
  functional.

---

## 3. Logic Issues

### 3.1 FlashADC Differential Reference Range
`FlashADC.py:161-162`:
```python
v_min = -v_ref / 4 if input_type == InputType.DIFFERENTIAL else 0.0
v_max =  v_ref / 4 if input_type == InputType.DIFFERENTIAL else v_ref
```
Creates a `ReferenceLadder` spanning `[-v_ref/4, +v_ref/4]` for differential mode, yielding only `2^(n_bits-1)` effective range. This effectively halves the usable dynamic range for differential inputs.

**Impact**: The comparator's effective threshold range becomes `[-v_ref/2, +v_ref/2]` (after the `*2` scaling in `reference_voltages` property), which is correct for a differential ADC. However, this is subtle and could confuse users expecting `v_ref` to map to `[-v_ref/2, +v_ref/2]` directly.

**Status: FALSE POSITIVE — functionally correct; docstring clarified (2026-04-24)**

Code trace (single comparator `i`):

    v_refp[i] = t[i]              (from ascending-index ladder)
    v_refn[i] = t[n-1-i]          (from descending-index ladder)
    comparator sees (v_pos - v_neg) - (v_refp[i] - v_refn[i])
                  = (v_pos - v_neg) - (t[i] - t[n-1-i])

With the symmetric ladder around zero, `t[n-1-i] = -t[i]`, so the
effective threshold is `2 * t[i]`.  Since `t[i] ∈ [-v_ref/4, +v_ref/4]`,
the effective threshold ranges over `[-v_ref/2, +v_ref/2]` — the
standard differential swing for a reference `v_ref` (`v_ref`
peak-to-peak).  The `reference_voltages` property returns these
×2-scaled effective values so external inspection matches what the
comparators actually apply.

This is the conventional construction for a differential flash ADC; the
review itself conceded the math is correct ("... which is correct for a
differential ADC").  The remaining concern — that the convention is
"subtle and could confuse users" — was addressed by adding a
"Reference ladder convention" section to the `FlashADC` class docstring
(2026-04-24), so the scaling is visible to any reader of the class
without having to follow the block comments in `__init__` and
`_convert_input`.  No functional change.

### 3.2 Segmented Resistor DAC `__repr__` Does Not Include `seed`
`SegmentedResistorDAC.py:153-157` and `ResistorStringDAC.py:129-132`: Neither class includes the random seed in their repr. This means two DACs created with the same parameters but different seeds will have identical repr strings, making debugging harder.

**Status: FIXED — extended scope (2026-04-25)**

The narrow review item was about repr visibility on resistor DACs that
already accepted a `seed`.  During discussion we identified a deeper
gap: many classes that *model* construction-time random mismatch
*didn't accept a seed at all*, leaving the user no way to make
construction reproducible.  Resolved both gaps in one pass.

Convention adopted (matches scipy / sklearn):

- `seed=None` (default) — `np.random.default_rng(None)` uses OS entropy;
  non-deterministic, preserves prior behaviour.
- `seed=<int>` — reproducible mismatch draw.
- `self.seed = seed` is always stored on the instance so users can
  inspect `obj.seed`.
- Repr includes `seed=N` only when `seed is not None`; omitted (kept
  terse) for the default case.

Phase A — classes that already accepted `seed` at construction; added
storage + conditional-repr only:

- `ResistorStringDAC`, `R2RDAC`, `SegmentedResistorDAC`,
  `TimeInterleavedADC`.

Phase B — classes that did not previously accept `seed`; added kwarg
and routed construction-time draws through `np.random.default_rng(seed)`:

- All 5 CDAC classes (`SingleEndedCDAC`, `DifferentialCDAC`,
  `SegmentedCDAC`, `RedundantSARCDAC`, `SplitCapCDAC`).  Refactor
  builds capacitors with `mismatch=0` and routes the actual mismatch
  through the existing seeded `apply_mismatch(cap_mismatch, seed=...)`
  method, which already used `default_rng` correctly.  No change to
  `IdealCapacitor`'s public API.
- `ReferenceLadder` — replaced direct `np.random.normal` for resistor
  mismatch with `default_rng(seed).normal`.
- `FlashADC` — added `seed` kwarg.  Uses `np.random.SeedSequence(seed).spawn(2)`
  to derive *independent* sub-streams for the offset draw and the
  default-ladder mismatch draw, so the two are statistically
  independent within a single seeded instance instead of correlated
  through a shared Generator stream.

Out of scope (Level 2/3 deferred): per-conversion stochastic draws
(sampling noise / kT/C, aperture jitter, per-call comparator and
reference noise) continue to use `np.random` global state.  Making
those reproducible would require an `rng` attribute on every
ADC/DAC instance and `self._rng.normal(...)` everywhere — bigger
refactor with no concrete demand yet.

26 new tests cover: same-seed reproducibility, different-seed
distinguishability, seed visible in repr when set, seed omitted from
repr when `None`, and pos/neg independence within a single
`DifferentialCDAC` (a seeded instance must still draw the two arrays
independently from the same Generator stream).  Total suite: 1005
tests passing.

### 3.3 Segment`edCDAC` Accesses Private Members
`SegmentedCDAC.py:571-572`:
```python
cap_weights = self._cdac._cap_weights
cap_total   = self._cdac._cap_total
```
Direct access to `_cap_weights` and `_cap_total` bypasses the public property interface that returns copies. While `SingleEndedCDAC.cap_weights` returns `self._cap_weights.copy()`, this code uses the unprotected attribute directly, creating a potential inconsistency if the public interface ever adds mismatch computation.

**Status: FIXED (2026-04-25)**

The private access existed for a real reason: `cap_weights` /
`cap_total` properties return defensive copies, and a per-conversion
copy on the hot path is wasteful.  But the encapsulation breakage was
genuine — if `SingleEndedCDAC` ever changed how it computes the voltage
(e.g., parasitic capacitance, time-varying mismatch), `SegmentedCDAC`
would silently miss the update.

Resolution: extract the dot-product into a protected helper on
`SingleEndedCDAC`:

```python
def _voltage_from_bits(self, bits: np.ndarray) -> float:
    return float(np.dot(bits, self._cap_weights) / self._cap_total * self._v_ref)
```

Both `SingleEndedCDAC.get_voltage(code)` and `SegmentedCDAC.get_voltage(code)`
now route through it.  SegmentedCDAC builds its segmented bit pattern
and delegates the math to the inner CDAC — no longer touching
`_cap_weights` / `_cap_total` directly.  Performance is identical to
the prior private-access approach (no copy), and the storage layout
stays encapsulated inside `SingleEndedCDAC`.

All 1005 tests still pass — the refactor is functionally equivalent
and existing voltage-correctness tests cover any regression.

### 3.4 MultibitSARADC Ignores Per-Cycle DVdt
`SARADC.py:416-449`: `MultibitSARADC._run_sar()` does not pass `dvdt` to `self.comparator.compare()`. The parent's `_dvdt` attribute is set but never used in the multibit path.

**Status: FALSE POSITIVE — closed (2026-04-26)**

The premise is architecturally incorrect. In a SAR ADC the input is
sampled *once* by the sample-and-hold and then bit-cycled against the
*held* value. Aperture jitter is therefore a one-shot perturbation at
the sampling instant, not a per-comparator effect. The implementation
applies it in exactly the right place — `SARADC._sample_input` lines
321–322:

```python
if self.t_jitter and self._dvdt:
    v = v + self._dvdt * np.random.normal(0.0, self.t_jitter)
```

This method is inherited by `MultibitSARADC` (which only overrides
`_run_sar`), and the convert path is:

    MultibitSARADC.convert(vin, dvdt)
      -> SARADC.convert            (sets self._dvdt = dvdt)
      -> SARADC._convert_input     (inherited)
      -> SARADC._sample_input      (uses self._dvdt for jitter)  ✓
      -> MultibitSARADC._run_sar   (bit-cycling on the held value)

The parent `SARADC._run_sar` likewise doesn't pass `dvdt` to its
`comparator.compare()` call (line 357) — for the same reason. The
review apparently expected per-comparator `dvdt` application, but
the comparator API doesn't even take a `dvdt` parameter; it takes
`time_step` for bandwidth filtering, which is a different effect.

Empirical verification (with `t_jitter=1e-9` on an 8-bit, 2-bit/cycle
MultibitSARADC):

    convert(0.4, dvdt=3.14e6) × 50 → 5 distinct codes  (jitter active)
    convert(0.4, dvdt=0.0)    × 50 → 1 distinct code   (no jitter effect)

`dvdt` flows through correctly on the multibit path. No code change.

### 3.5 NoiseshapingSARADC Assumes FLOOR Quantization
`NoiseshapingSARADC.py:510`:
```python
v_reconstructed = (code + 0.5) / (2 ** self.n_bits) * self.v_ref
```
Uses the standard FLOOR quantization midpoint formula. If combined with `SYMMETRIC` mode (which `NoiseshapingSARADC.__init__` accepts via `**kwargs`), the noise-shaping integrator will have a DC bias because `v_reconstructed` won't match the actual quantization boundaries.

**Status: FALSE POSITIVE — closed (2026-04-27)**

Two-part finding, both parts rebut the concern.

(1) The premise is wrong. `NoiseshapingSARADC.__init__(n_bits, v_ref,
    **kwargs)` forwards kwargs to `SARADC.__init__`, which has explicit
    named parameters (`input_type`, `comparator_type`,
    `comparator_params`, `cdac`, `cap_mismatch`, `noise_rms`, `offset`,
    `gain_error`, `t_jitter`) and *no* `quant_mode`.  Passing
    `quant_mode=QuantizationMode.SYMMETRIC` actually raises
    `TypeError: SARADC.__init__() got an unexpected keyword argument
    'quant_mode'` — verified empirically.  The DC-bias scenario the
    review describes therefore cannot occur.

(2) The formula is correct for the implemented FLOOR quantizer.

    Single-ended: SARADC's CDAC produces thresholds at `k · LSB` with
    `LSB = v_ref / 2^N`; the FLOOR bin midpoint is `(k + 0.5) · LSB =
    (code + 0.5) / 2^N · v_ref`.  ✓

    Differential: `DifferentialCDAC` produces effective thresholds at
    `k/2^N · v_ref − v_ref/2` (symmetric around zero).  The bin
    midpoint is `(k + 0.5)/2^N · v_ref − v_ref/2`, matching the
    `if input_type == DIFFERENTIAL: v_reconstructed −= v_ref/2`
    branch.  ✓

    Midpoint reconstruction gives a zero-mean residue, which is
    exactly what the first-order noise-shaping integrator needs to
    avoid DC drift.

Consistent with §2.2's resolution: SARADC (and all structural ADCs)
are FLOOR-by-construction; `QuantizationMode.SYMMETRIC` is a
behavioural-ADC concern only.  No API or formula change required.

### 3.6 `convert_waveform` Return Type
`ADCBase.convert_waveform` returns `np.ndarray[int]` (annotated), but several architectures return plain `np.ndarray` without explicit dtype:
- `TimeInterleavedADC.convert_waveform`: Explicitly uses `dtype=int`
- `ADCBase.convert_waveform`: Uses `dtype=int` ✓
- No issues found, but type annotation should be `np.ndarray` with `dtype=int` constraint

**Status: FIXED (cosmetic, 2026-04-27)**

The review itself concluded "no issues found" — both implementations
already produce int arrays via `dtype=int`. Tightened the metadata
side: added `-> np.ndarray` return-type annotations to the two `def`
signatures so static checkers / IDE introspection see a real
annotation instead of `inspect.Signature.empty`.

The dtype=int constraint stays in the docstring rather than the
annotation. NumPy's generic typing form
`np.ndarray[Any, np.dtype[np.int_]]` is supported in modern numpy but
is verbose and not idiomatic for this codebase. The signature-level
`np.ndarray` plus the docstring `Returns: np.ndarray[int]` is the
common pyData convention.

No runtime change; 1005 tests still pass.

### 3.7 FlashADC `_encode` XOR Encoder Edge Case
`FlashADC.py:231-232`:
```python
values = active + 1
code = int(np.bitwise_or.reduce(values))
```
If `active` contains indices that result in values whose OR overflows standard integer range for large codes, the result could be incorrect. However, for practical `n_bits ≤ 12`, this is not a concern.

(The line numbers in the review point at the comparator construction
loop; the actual XOR encoder is around `FlashADC.py:279–282`.)

**Status: FALSE POSITIVE — closed (2026-04-27)**

No real overflow risk within the codebase's supported range:

- `ADCBase.__init__` validates `1 ≤ n_bits ≤ 32`, so the largest
  `values[i]` is `2^32 − 1`.
- `np.flatnonzero` returns `int64` on 64-bit platforms (verified).
- `np.bitwise_or.reduce` on `int64` supports values up to `2^63 − 1`,
  giving ~31 bits of headroom at the worst case (n_bits=32).

The review itself conceded "for practical n_bits ≤ 12, this is not a
concern."  The safe range actually extends to the codebase's hard
limit.  No code change.

### 3.8 R2RDAC Termination Resistor Usage
`R2RDAC.py:188` adds a termination resistor at node `n-1` to GND using `r2_values[n-1]`. Then `R2RDAC._build_network` at line 181 also adds a vertical arm for the LSB bit using `r2_values[n-1]`. So `r2_values[n-1]` is used twice for the LSB end. This matches the standard R-2R topology but is not clearly documented.

**Status: FIXED (2026-04-28)**

On closer reading the issue is more substantive than just
documentation.  A standard R-2R ladder has `n_bits + 1` distinct
physical 2R resistors:

  - `n_bits` switch arms (one per bit, switched between V_ref and GND)
  - `1` termination at the LSB end (always tied to GND)

The previous implementation allocated `r2_values` as length `n_bits`
and reused `r2_values[n-1]` for *both* the LSB switch arm *and* the
termination.  For ideal analysis (no mismatch) this is harmless —
both are nominally 2R.  But under `r2_mismatch > 0` the LSB switch
and termination became perfectly correlated rather than independent,
which under-estimates the LSB-region variance in a Monte-Carlo sweep
by roughly `1 / (2(n_bits + 1))` (about 4 % at `n_bits=10`).  The
constructor docstring on `r2_mismatch` reads "Std of multiplicative
mismatch for 2R (vertical) arms", which a user reasonably interprets
as "all 2R arms get independent draws".

Resolution:

  - `r2_values` is now length `n_bits + 1`.  Indices `0..n_bits-1`
    are the per-bit switch arms (MSB → LSB); index `n_bits` is the
    dedicated LSB-end termination.  All `n_bits + 1` arms receive
    independent mismatch draws.
  - `_build_network` now uses `r2_values[n_bits]` for the termination
    instead of `r2_values[n_bits - 1]`.
  - Class docstring updated; the prior ambiguous "doubled as
    termination" wording is replaced with explicit per-index meanings.
  - `_build_network` docstring rewritten — the "Wait — the standard
    R-2R analysis ..." mid-docstring correction is gone, replaced
    with a clean topology description.
  - 1 new test (`test_r2_values_includes_independent_termination`)
    asserts `r2_values.shape == (n_bits + 1,)` and that the LSB
    switch arm and termination at `seed=123` are distinct draws.

Behaviour change: for any user who pinned a seed against the old
model (only possible since the §3.2 fix added seed support), the
`r_values`, `r2_values`, and resulting `_tap_voltages` arrays are
slightly different now.  The new behaviour is more physically
accurate.  All 1006 tests pass.

---

## 4. Documentation Issues

### 4.1 Module Docstring Discrepancies
- `dataconverter.py` header says "Version History: 2026-03-22: Added QuantizationMode enum" but the code was clearly updated much later (comments mention 2026-04-13)
- Several module docstrings have "Version History" that are outdated relative to the actual code

**Status: FIXED (2026-04-28)**

The root cause is the pattern itself — duplicating change history in
module docstrings means it is *always* either wrong or about to be
wrong, and git already carries the canonical record.  Resolution:
strip the multi-line `Version History:` blocks across all 15 modules
and replace each with a single line preserving the first-written
date and pointing to git for the rest:

    First written <YYYY-MM-DD>; see ``git log`` for the change history.

Files touched (15):
  - `pyDataconverter/dataconverter.py` (2025-01-31)
  - `pyDataconverter/architectures/SimpleADC.py` (2025-02-01)
  - `pyDataconverter/architectures/SimpleDAC.py` (2025-02-06)
  - `pyDataconverter/architectures/FlashADC.py` (2024-02-07)
  - `pyDataconverter/components/comparator.py` (2024-02-07)
  - `pyDataconverter/architectures/SARADC.py` (2026-03-25)
  - `pyDataconverter/components/cdac.py` (2026-03-25)
  - `pyDataconverter/components/capacitor.py` (2026-03-25)
  - `pyDataconverter/components/decoder.py` (2026-03-25)
  - `pyDataconverter/components/current_source.py` (2026-03-25)
  - `pyDataconverter/architectures/CurrentSteeringDAC.py` (2026-03-25)
  - `pyDataconverter/components/reference.py` (2026-03-23)
  - `pyDataconverter/architectures/ResistorStringDAC.py` (2026-04-02)
  - `pyDataconverter/architectures/SegmentedResistorDAC.py` (2026-04-02)
  - `pyDataconverter/architectures/R2RDAC.py` (2026-04-02)

`grep -rn "Version History" pyDataconverter/` now returns nothing.
1006 tests still pass.

### 4.2 Missing Parameter Documentation
- `ResidueAmplifier.amplify()`: The docstring says the caller pre-multiplies by gain, but the `gain` attribute exists for callers to read. This contract is correct but not prominently documented in the class docstring.
- `TimeInterleavedADC`: The `convert()` method's `dvdt` parameter behavior is documented (timing skew via `dvdt * skew_k`) but the interaction with the sub-ADC's own jitter model is not explained.

**Status: FIXED (2026-04-28)**

(a) **`ResidueAmplifier`** — added a "Gain contract" block at the top
    of the class docstring (between the summary line and the Attributes
    table).  Shows the canonical caller-pre-multiplies pattern with a
    worked example:

        ra = ResidueAmplifier(gain=4.0, settling_tau=...)
        v_out = ra.amplify(
            target        = ra.gain * (v_in - v_dac),
            initial_error = sign * ra.gain * sub_dac.lsb,
            t_budget      = t_budget,
        )

    Updated the `gain` attribute description to cross-reference the
    contract block.  The `amplify()` method docstring already explained
    this clearly; the change just surfaces it where a class-level
    reader will see it first.

(b) **`TimeInterleavedADC._convert_input`** — expanded the docstring
    to call out the two distinct uses of `dvdt`:

      1. *Timing-skew mismatch* (TI level): `self._dvdt * skew_k` is
         added to the input-referred correction, modelling the channel
         sampling at a small offset from nominal — the canonical
         TI-ADC mismatch spur source.
      2. *Sub-ADC aperture jitter*: `dvdt` is forwarded unchanged to
         the channel's sub-ADC via `convert(..., dvdt=self._dvdt)`,
         so the sub-ADC's own `t_jitter` model applies on top.

    Notes how the two effects compose linearly (deterministic skew +
    stochastic jitter) and how to disable each independently.

### 4.3 No Sphinx/ReadTheDocs Setup
The project has no `docs/` folder with actual documentation build infrastructure, only a `docs/superpowers/` spec subfolder.

**Status: DEFERRED (2026-04-28)**

The review's premise was already partially obsolete: a working Sphinx
scaffold has been set up under `docs/source/` (untracked), with
`conf.py`, `index.rst`, `modules.rst`, an `api/` subtree, and a
`_build/` output.  It generates HTML successfully, albeit with ~101
warnings tracked in `docs/SPHINX_IMPROVEMENTS.md`.

However, the maintainer has flagged that scaffold as exploratory —
intended only to see what auto-generated Sphinx output looks like —
and judges the result to be of poor quality.  The plan is to redo
the documentation build from scratch as a separate, larger effort
(API style decisions, NumPy-format docstring conversion, hand-written
narrative pages, examples gallery, etc.) rather than polish the
current prototype.

This review item is therefore deferred.  The prototype's notes-to-self
file (`docs/SPHINX_IMPROVEMENTS.md`) should not be treated as a binding
plan — its quick-fixes list reflects observations from a single build
run, not the eventual documentation strategy.

Items in that list that overlap with completed review work:
- "Empty `__init__.py` files" was already addressed under §1 (commit
  `cad24ac`, fully-namespaced public API).

### 4.4 ADCBase Docstring
`dataconverter.py:8-9` says "Classes: ADCBase: abstract class for all ADC implementations" but then lists both ADC and DAC classes. The docstring header should clarify it covers both.

**Status: FIXED (2026-04-28)**

The narrow concern was largely already addressed — line 5 of the
module docstring read "This module provides interfaces for both ADC
and DAC implementations".  But the `Classes:` listing did have a real
gap: the three public enums living in this file (`InputType`,
`OutputType`, `QuantizationMode`) were not surfaced at all in the
module summary, even though they are part of the public API and
imported throughout the codebase.

Replaced the bare `Classes:` section with a structured `Public API:`
block split into "Base classes" and "Enums", with a one-line summary
for each enum (and a cross-reference to `QuantizationMode`'s
"Applicability" section so readers know it parameterises behavioural
ADCs only — see §2.2).  No code change; documentation only.

---

## 5. Code Quality Issues

### 5.1 Type Annotations Are Inconsistent
Many files mix typed and untyped parameters:
```python
# Good:
def __init__(self, n_bits: int, v_ref: float = 1.0, ...

# Missing type hints:
def redraw_mismatch(self, stddev: float, rng: "np.random.Generator") -> None:
```
Some type hints use string quotes for forward references, others don't.

**Status: FIXED — narrow scope (2026-04-28)**

The specific concrete inconsistency the review flagged — `np.random.Generator`
written as a string forward-reference in `components/capacitor.py`
(lines 53 and 149) but as a bare class reference everywhere else
(`TimeInterleavedADC.py:40`, `utils/signal_gen.py:607`) — is now
normalised.  numpy 1.17+ has `np.random.Generator` as a real
importable class; the string form was a pre-emptive forward-reference
that became unnecessary.  Dropped the quotes; the codebase now uses
one consistent form.

The broader concern that some `def` lines (about 119 of 275) lack
return-type annotations is out of scope for review closure.  Many
are legitimately bare (`__init__` returns None, `reset()` returns
None) and many would benefit from `-> None` for clarity, but a full
typing audit is its own thread of work.  Tracking as a future task
rather than under this review item.

### 5.2 No `py.typed` Marker
The package lacks a `py.typed` file, meaning it's not registered as a typed package for mypy consumers.

**Status: FIXED (2026-04-28)**

PEP 561 requires two pieces:

  1. An empty `py.typed` marker file at the package root.  Created at
     `pyDataconverter/py.typed`.
  2. The marker must be shipped in the wheel/sdist.  Added to
     `setup.py`:

         package_data={'pyDataconverter': ['py.typed']},
         include_package_data=True,

Downstream consumers running mypy, pyright, etc. against an installed
`pyDataconverter` will now consult the inline annotations.

### 5.3 `if __name__ == "__main__"` Blocks Are Extensive
- `SARADC.py:541-583`: ~42 lines of demo code in `__main__`
- `FlashADC.py:332-413`: ~82 lines
- `SimpleADC.py:184-205`: ~22 lines
- `SimpleDAC.py:209-226`: ~18 lines
- `signal_gen.py:725-884`: ~160 lines
- `fft_analysis.py:206-298`: ~93 lines

These should ideally be moved to separate example scripts under `examples/` or at minimum use `if __name__ == "__main__"` guards with a comment explaining they're demonstration code only.

**Status: FIXED (2026-04-28)**

Took the review's recommended approach: created `examples/` at repo
root with seven standalone demo scripts (one per source module's old
`__main__` block) plus a `README.md` index summarising what each
demo shows.

New files:
- `examples/saradc_demo.py`        — replaces `SARADC.__main__`
- `examples/flashadc_demo.py`      — replaces `FlashADC.__main__`
- `examples/simpleadc_demo.py`     — replaces `SimpleADC.__main__`
- `examples/simpledac_demo.py`     — replaces `SimpleDAC.__main__`
- `examples/comparator_demo.py`    — replaces `comparator.__main__`
- `examples/fft_analysis_demo.py`  — replaces both
  `fft_analysis.demo_fft_analysis()` *and* the `fft_analysis.__main__`
  call site
- `examples/signal_gen_demo.py`    — replaces `signal_gen.__main__`
- `examples/README.md`             — index with one-line per-demo
  description

Removed from library code:
- `__main__` blocks in six modules (SARADC, FlashADC, SimpleADC,
  SimpleDAC, comparator, signal_gen).
- The module-level `demo_fft_analysis()` function from
  `fft_analysis.py` (it existed only to be called by the demo
  block).
- The `from fft_analysis import demo_fft_analysis` import in
  `tests/test_fft_analysis_coverage.py`.

About 660 lines of demo code moved out of the library; the three
biggest files (signal_gen, fft_analysis, FlashADC) each lost
80–160 lines.

Tests: removed six demo-coverage tests that no longer have a
`__main__` block to exercise (`TestMainBlock` in test_FlashADC.py,
`TestSimpleDACMainBlock` in test_SimpleDAC.py,
`TestSimpleADCMainBlock` in test_SimpleADC.py,
`TestComparatorMainBlock` in test_comparator.py,
`TestDemoFFTAnalysis` in test_fft_analysis_coverage.py).
Demo-script coverage isn't library testing — examples are runnable
standalone, not part of the test suite.

1000 tests still pass (was 1006; the 6 removed tests were the only
delta).  All seven example scripts pass `ast.parse` and the
non-plotting ones (simpleadc_demo, simpledac_demo) produce the
expected output when run directly.

For users, the entry point is now:

    python examples/saradc_demo.py
    python examples/flashadc_demo.py
    ...

Discoverable in one place rather than scattered as
`python -m pyDataconverter.architectures.SARADC`.

### 5.4 Warnings Import Inside Functions
`cdac.py:123`, `capacitor.py:169`, `current_source.py:105`: All import `warnings` inside function bodies rather than at module level. This is a minor style inconsistency with the rest of the codebase.

### 5.5 Hardcoded Magic Numbers
- `ResidueAmplifier.__repr__` uses `parts.append(f"settling_tau={self.settling_tau:.3e}")` — the `.3e` format specifier is hardcoded; other attributes use `.2e`
- `_PDF_SINGULARITY_GUARD = 0.999` in `adc.py:14` — should be a parameter of the functions that use it, not a module-level constant

### 5.6 No Abstract Properties for `CDACBase.n_bits`
`CDACBase` declares `n_bits` as an `@property` + `@abstractmethod` but never checks implementations are consistent with `ADCBase.n_bits`. A CDAC could have a different `n_bits` than its parent SARADC, which would cause subtle bugs.

### 5.7 `SimpleDAC.convert_sequence` Note About `code_errors`
`SimpleDAC.py:156-163` has a detailed comment explaining that `code_errors` is NOT applied in `convert_sequence`, but this is a silent deviation from expected behavior. The docstring doesn't mention this.

### 5.8 `apply_mismatch` Returns `None` Implicitly
All `apply_mismatch` implementations in `CDACBase`, `SingleEndedCDAC`, `DifferentialCDAC`, `SegmentedCDAC` modify state in-place but return `None`. The docstring says "Re-draw..." without explicitly stating the mutation or return value. While technically correct (in-place mutation is clear from context), a clearer docstring would help.

---

## 6. Performance Concerns

### 6.1 R2RDAC Computes All Codes at Construction
`R2RDAC._compute_tap_voltages()` solves a linear system for all 2^N codes at construction time. For N=16, that's 65536 solves. This is O(2^N × N³) at startup. For large N, this can be slow.

**Same issue**: `ResistorStringDAC._compute_tap_voltages()` has the same problem.

### 6.2 `SegmentedCDAC.get_voltage` Creates Arrays Per Call
`SegmentedCDAC.py:558-566` creates two new numpy arrays per call:
```python
therm_bits = np.zeros(n_therm_caps, dtype=float)
binary_bits = np.array([...], dtype=float)
```
These could be pre-allocated buffers.

### 6.3 `SimpleDAC.convert_sequence` Applies Noise After Repeat
`SimpleDAC.py:176-177` applies noise to the already-oversampled waveform, which means more noise samples than unique codes. This is correct for thermal noise but wasteful if the user wanted one noise draw per code.

---

## 7. Testing Gaps

### 7.1 No Tests for Hierarchical TI-ADC
`TimeInterleavedADC.hierarchical()` has no tests covering the multi-level construction or the `outer_product` arithmetic.

### 7.2 No Tests for NoiseshapingSARADC
`NoiseshapingSARADC` has no dedicated tests.

### 7.3 No Tests for MultibitSARADC Beyond Basic Trace
The `MultibitSARADC._run_sar` override has basic test coverage but metastability coupling is not tested.

### 7.4 Pipelined ADC Has No Tests
`PipelinedADC` is listed in `architectures/__init__.py` but has no test file.

### 7.5 Missing Integration Tests
- TI-ADC + SARADC as backend: tested in `test_ti_adc.py`
- TI-ADC + hierarchical: no tests
- Pipelined ADC with real backends: no tests

---

## 8. Minor Issues

### 8.1 Dead Code in FlashADC
`FlashADC.py:392-401` creates a second `adc` instance that's never used.

### 8.2 Incomplete Visualization Demo
`FlashADC.py:407-413` has example code at module level (not in `__main__`) that imports and calls `visualize_flash_adc` and `animate_flash_adc`. This code runs when the module is imported, which may be unexpected.

### 8.3 `TimeInterleavedADC` Has No `__repr__` for Sub-ADCs
`TimeInterleavedADC.__repr__` shows the template class name but not the sub-ADC repr, making debugging interleaved ADCs harder.

### 8.4 `dac_currents` Property Computes on Every Access
`CurrentSteeringDAC.dac_currents` recomputes the entire array on every access. Should be cached or memoized.

### 8.5 `ResidueAmplifier.slew_rate` Is Stored But Never Used
The `slew_rate` attribute is validated in `__init__` but the `amplify()` method does not apply slew limiting.

### 8.6 `comparator.py` Has Code After `__main__` Block
Lines 273-301 contain demo code that runs when the module is imported. This pattern is repeated across multiple modules.

---

## 9. Recommendations (Prioritized)

### High Priority
1. **Fix `SARADC`/`FlashADC` default `input_type` inconsistency** — either make them consistent with `ADCBase` (DIFFERENTIAL) or clearly document the architecture-specific defaults
2. **Add `pyDataconverter/__init__.py` exports** — at minimum re-export all public classes
3. **Add missing test coverage** for `PipelinedADC`, `NoiseshapingSARADC`, `MultibitSARADC`, `TimeInterleavedADC.hierarchical()`
4. **Document the `ResidueAmplifier.gain` pre-multiplication contract** prominently in the class docstring

### Medium Priority
5. **Move `if __name__ == "__main__"` demo code to `examples/` directory** — these are 100s of lines of code that run on import
6. **Cache R2RDAC and ResistorStringDAC tap voltages with invalidation** — construction time is O(2^N) which is expensive for large N
7. **Add `py.typed` marker** for mypy consumers
8. **Standardize repr() formatting** — use consistent format specifiers and include all relevant parameters consistently

### Low Priority
9. **Consider a unified QuantizationMode for all ADCs** — `SARADC` hardcodes FLOOR but the infrastructure for SYMMETRIC exists
10. **Add `seed` to SegmentedResistorDAC and ResistorStringDAC repr**
11. **Consider pre-allocating buffers in SegmentedCDAC.get_voltage()** for hot-path performance