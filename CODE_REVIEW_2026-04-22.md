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
| 2.2 | Quantization mode not unified | PENDING | |
| 2.3 | `n_levels` support inconsistent across DACs | PENDING | |
| 2.4 | `__repr__` inconsistency | PENDING | |
| 3.1 | FlashADC differential ref range `±v_ref/4` | PENDING | |
| 3.2 | DAC repr missing `seed` | PENDING | |
| 3.3 | `SegmentedCDAC` private member access | PENDING | |
| 3.4 | `MultibitSARADC` ignores `dvdt` | PENDING | |
| 3.5 | `NoiseshapingSARADC` assumes FLOOR | PENDING | |
| 3.6 | `convert_waveform` return type annotation | PENDING | |
| 3.7 | FlashADC XOR encoder edge case | PENDING | |
| 3.8 | R2RDAC termination resistor double-use | PENDING | |
| 4.1 | Outdated module version-history blocks | PENDING | |
| 4.2 | Missing parameter documentation | PENDING | |
| 4.3 | No Sphinx/RTD setup | PENDING | See `docs/SPHINX_IMPROVEMENTS.md` |
| 4.4 | `ADCBase` docstring covers DAC too | PENDING | |
| 5.1 | Inconsistent type annotations | PENDING | |
| 5.2 | No `py.typed` marker | PENDING | |
| 5.3 | Large `__main__` demo blocks | PENDING | |
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

### 2.3 `n_levels` Parameter
- `DACBase` accepts optional `n_levels` (defaults to `2**n_bits`)
- `SimpleDAC` forwards it correctly
- `ResistorStringDAC`, `R2RDAC`, `SegmentedResistorDAC`, `CurrentSteeringDAC` do NOT accept `n_levels` — they're hardcoded to power-of-2

**Issue**: Inconsistent support for non-power-of-2 DACs across the codebase.

### 2.4 Repr Inconsistency
| Class | Repr Includes |
|-------|--------------|
| `ADCBase` | `n_bits, v_ref, input_type` |
| `DACBase` | `n_bits, v_ref, output_type` |
| `SARADC` | All above + `cdac=...` (full repr of CDAC object) |
| `TimeInterleavedADC` | `M, fs, template, n_bits` (missing `v_ref, input_type`) |
| `ResidueAmplifier` | Only non-default params |

**Issue**: `SARADC.__repr__` embeds the full CDAC repr which can be very long; `TimeInterleavedADC` omits `v_ref, input_type` from its base class.

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

### 3.2 Segmented Resistor DAC `__repr__` Does Not Include `seed`
`SegmentedResistorDAC.py:153-157` and `ResistorStringDAC.py:129-132`: Neither class includes the random seed in their repr. This means two DACs created with the same parameters but different seeds will have identical repr strings, making debugging harder.

### 3.3 Segment`edCDAC` Accesses Private Members
`SegmentedCDAC.py:571-572`:
```python
cap_weights = self._cdac._cap_weights
cap_total   = self._cdac._cap_total
```
Direct access to `_cap_weights` and `_cap_total` bypasses the public property interface that returns copies. While `SingleEndedCDAC.cap_weights` returns `self._cap_weights.copy()`, this code uses the unprotected attribute directly, creating a potential inconsistency if the public interface ever adds mismatch computation.

### 3.4 MultibitSARADC Ignores Per-Cycle DVdt
`SARADC.py:416-449`: `MultibitSARADC._run_sar()` does not pass `dvdt` to `self.comparator.compare()`. The parent's `_dvdt` attribute is set but never used in the multibit path.

### 3.5 NoiseshapingSARADC Assumes FLOOR Quantization
`NoiseshapingSARADC.py:510`:
```python
v_reconstructed = (code + 0.5) / (2 ** self.n_bits) * self.v_ref
```
Uses the standard FLOOR quantization midpoint formula. If combined with `SYMMETRIC` mode (which `NoiseshapingSARADC.__init__` accepts via `**kwargs`), the noise-shaping integrator will have a DC bias because `v_reconstructed` won't match the actual quantization boundaries.

### 3.6 `convert_waveform` Return Type
`ADCBase.convert_waveform` returns `np.ndarray[int]` (annotated), but several architectures return plain `np.ndarray` without explicit dtype:
- `TimeInterleavedADC.convert_waveform`: Explicitly uses `dtype=int`
- `ADCBase.convert_waveform`: Uses `dtype=int` ✓
- No issues found, but type annotation should be `np.ndarray` with `dtype=int` constraint

### 3.7 FlashADC `_encode` XOR Encoder Edge Case
`FlashADC.py:231-232`:
```python
values = active + 1
code = int(np.bitwise_or.reduce(values))
```
If `active` contains indices that result in values whose OR overflows standard integer range for large codes, the result could be incorrect. However, for practical `n_bits ≤ 12`, this is not a concern.

### 3.8 R2RDAC Termination Resistor Usage
`R2RDAC.py:188` adds a termination resistor at node `n-1` to GND using `r2_values[n-1]`. Then `R2RDAC._build_network` at line 181 also adds a vertical arm for the LSB bit using `r2_values[n-1]`. So `r2_values[n-1]` is used twice for the LSB end. This matches the standard R-2R topology but is not clearly documented.

---

## 4. Documentation Issues

### 4.1 Module Docstring Discrepancies
- `dataconverter.py` header says "Version History: 2026-03-22: Added QuantizationMode enum" but the code was clearly updated much later (comments mention 2026-04-13)
- Several module docstrings have "Version History" that are outdated relative to the actual code

### 4.2 Missing Parameter Documentation
- `ResidueAmplifier.amplify()`: The docstring says the caller pre-multiplies by gain, but the `gain` attribute exists for callers to read. This contract is correct but not prominently documented in the class docstring.
- `TimeInterleavedADC`: The `convert()` method's `dvdt` parameter behavior is documented (timing skew via `dvdt * skew_k`) but the interaction with the sub-ADC's own jitter model is not explained.

### 4.3 No Sphinx/ReadTheDocs Setup
The project has no `docs/` folder with actual documentation build infrastructure, only a `docs/superpowers/` spec subfolder.

### 4.4 ADCBase Docstring
`dataconverter.py:8-9` says "Classes: ADCBase: abstract class for all ADC implementations" but then lists both ADC and DAC classes. The docstring header should clarify it covers both.

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

### 5.2 No `py.typed` Marker
The package lacks a `py.typed` file, meaning it's not registered as a typed package for mypy consumers.

### 5.3 `if __name__ == "__main__"` Blocks Are Extensive
- `SARADC.py:541-583`: ~42 lines of demo code in `__main__`
- `FlashADC.py:332-413`: ~82 lines
- `SimpleADC.py:184-205`: ~22 lines
- `SimpleDAC.py:209-226`: ~18 lines
- `signal_gen.py:725-884`: ~160 lines
- `fft_analysis.py:206-298`: ~93 lines

These should ideally be moved to separate example scripts under `examples/` or at minimum use `if __name__ == "__main__"` guards with a comment explaining they're demonstration code only.

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