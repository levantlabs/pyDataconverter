# Pipelined ADC Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 1 of pyDataconverter's pipelined ADC support — a composable `PipelinedADC` class that inherits from `ADCBase`, uses any sub-ADC / sub-DAC combination, preserves the metastability-to-settling coupling from the vetted `adc_book` reference, and matches the reference's outputs bit-exactly on four canonical configurations.

**Architecture:** A `PipelinedADC(ADCBase)` cascades N `PipelineStage` instances followed by a required backend ADC. Each stage composes an existing `ADCBase` as the sub-ADC, an existing `DACBase` as the sub-DAC, and a new `ResidueAmplifier` component. Metastability coupling emerges from composition: `DifferentialComparator` owns regeneration time, `FlashADC` aggregates across its bank, `ResidueAmplifier` owns settling, and `PipelineStage` mediates the timing budget. Full architectural rationale and the algebraic proof of bit-exactness against the reference are in `docs/superpowers/specs/2026-04-13-pipelined-adc-design.md`.

**Tech Stack:** Python 3.12, numpy, pytest, unittest.TestCase style tests (matches existing codebase). No new external dependencies.

---

## Prerequisites

Before starting, verify:

- [ ] **Spec exists and is committed.** `docs/superpowers/specs/2026-04-13-pipelined-adc-design.md` is present. The plan below only makes sense after reading it.
- [ ] **Baseline tests pass.** `pytest tests/ -q` reports 826 passing tests. If not, investigate before making any changes — the comparison baseline for "no regressions" depends on this.
- [ ] **Reference checkout exists.** The adc_book reference has been cloned at `/tmp/adc_book_import/python/pipelinedADC.py`. If not, run `git clone --depth 1 git@github.com:manar-c/adc_book.git /tmp/adc_book_import` first. This is the source for Task 1 only; subsequent tasks never read from it.

---

## File structure

New files created by this plan:

- `tests/_reference/__init__.py` — empty package marker
- `tests/_reference/adc_book_pipelined.py` — vendored reference classes (read-only)
- `pyDataconverter/components/residue_amplifier.py` — new `ResidueAmplifier` component
- `pyDataconverter/architectures/PipelinedADC.py` — new `PipelineStage` + `PipelinedADC(ADCBase)` classes
- `tests/test_residue_amplifier.py` — unit tests for `ResidueAmplifier`
- `tests/test_pipelined_adc.py` — unit tests for `PipelineStage` and `PipelinedADC`
- `tests/test_pipelined_adc_vs_reference.py` — bit-exact comparison tests against the vendored reference
- `examples/pipelined_adc_example.py` — end-to-end example driving the canonical 12-bit config

Existing files modified (all changes are strictly additive — defaults preserve existing behaviour):

- `pyDataconverter/dataconverter.py` — add `n_levels` kwarg to `DACBase.__init__`
- `pyDataconverter/architectures/SimpleDAC.py` — forward `n_levels` to base and add `code_errors` kwarg
- `pyDataconverter/architectures/FlashADC.py` — add `n_comparators` kwarg, `last_conversion_time()`, `last_metastable_sign()` methods
- `pyDataconverter/components/comparator.py` — add `tau_regen` and `vc_threshold` kwargs, `last_regen_time` property
- `pyDataconverter/components/__init__.py` — export `ResidueAmplifier`
- `pyDataconverter/architectures/__init__.py` — export `PipelinedADC`, `PipelineStage`
- `tests/test_SimpleDAC.py` — add `n_levels` + `code_errors` test classes
- `tests/test_FlashADC.py` — add `n_comparators` relaxation + metastability methods test class
- `tests/test_comparator.py` — add `tau_regen` + `vc_threshold` test class
- `docs/api_reference.md` — new sections for `PipelinedADC`, `PipelineStage`, `ResidueAmplifier`, updated subsections for `FlashADC`, `SimpleDAC`, `DACBase`, `DifferentialComparator`
- `todo/adc_architectures.md` — status update from "IN DESIGN" to "PHASE 1 IMPLEMENTED"

---

## Task 1: Vendor the adc_book reference

**Files:**
- Create: `tests/_reference/__init__.py`
- Create: `tests/_reference/adc_book_pipelined.py`

- [ ] **Step 1.1: Create the package marker**

Run:

```bash
mkdir -p tests/_reference
```

Create `tests/_reference/__init__.py` with this exact content:

```python
"""Vendored external references used only for validation tests. Not shipped."""
```

- [ ] **Step 1.2: Copy the reference file and strip it**

Open `/tmp/adc_book_import/python/pipelinedADC.py`, copy only lines 1-186 (the class definitions and the blank lines between them), and write them to `tests/_reference/adc_book_pipelined.py` with the header below inserted at the top. Do NOT copy:
- Lines 188-242 (the `ADCanalysis` function — duplicate of `utils/metrics/_dynamic.py`)
- Lines 249-879 (the `__main__` block and its plotting code)
- Lines 150, 151, 169, 170 (the `print` statements inside `__init__` methods) — delete these specific lines in place
- The `import matplotlib` lines (lines 4-7) — replace with plain `import numpy as np`

Exact header to paste at the top of the new file:

```python
"""
Vendored reference for validation of pyDataconverter's PipelinedADC.

SOURCE:   https://github.com/manar-c/adc_book (python/pipelinedADC.py)
IMPORTED: 2026-04-13
AUTHOR:   Manar El-Chammas (attribution preserved)

This file is READ-ONLY. It is used exclusively by
tests/test_pipelined_adc_vs_reference.py to cross-check the pyDataconverter
implementation against the vetted original. Do not modify it — any bug found
here should be investigated in the upstream repository. If the upstream
changes, re-vendor the relevant slice and update the IMPORTED date above.

Classes included: PipelinedADC, Stage, subADC, subDAC, sumGain.
Nothing else from the upstream file is needed.
"""

import numpy as np


```

Then paste the class definitions (lines 9-186 of the original), with the four `print` statements removed. The stripped `subADC.__init__` should look like:

```python
class subADC:
    def __init__(self, N, FSR=1):
        self.FSR = FSR
        self.N = N
        self.LSB = self.FSR / N
        self.ref = np.arange(N)/(N-1)*(FSR-self.LSB) - (FSR/2-self.LSB/2)
        self.noisesigma = 0
```

And the stripped `subDAC.__init__` should look like:

```python
class subDAC:
    def __init__(self, N, FSR=1):
        self.N = N
        self.FSR = FSR
        self.LSB = self.FSR / N
        self.dacout = np.arange(N+1)/(N-1)*(FSR-self.LSB*1) - (FSR/2 + 0*self.LSB)
        self.error = np.zeros(N+1)
```

Everything else (`PipelinedADC`, `Stage`, `sumGain`, and the `output` / `add_error` methods) copies verbatim from the original lines 9-186.

- [ ] **Step 1.3: Verify the import works and no prints leak**

Run:

```bash
python -c "
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()) as buf:
    from tests._reference.adc_book_pipelined import PipelinedADC, Stage, subADC, subDAC, sumGain
    adc = PipelinedADC(Nstages=2, B=12, N=[8, 1026],
                       FSR_ADC=[1,1], FSR_DAC=[1,1],
                       G=[4, 512], minADCcode=[-1, 0])
    out = adc.output(0.0)
print('imported OK, stdout during import+construct:', repr(buf.getvalue()))
print('ideal 12-bit output for vin=0:', int(out))
"
```

Expected output:

```
imported OK, stdout during import+construct: ''
ideal 12-bit output for vin=0: 2052
```

If the stdout buffer is non-empty, one of the four `print` statements wasn't stripped — find and remove it. If the `ideal 12-bit output` is not `2052`, the copy was incomplete — diff against the original and fix.

- [ ] **Step 1.4: Commit**

```bash
git add tests/_reference/__init__.py tests/_reference/adc_book_pipelined.py
git commit -m "test: vendor adc_book pipelined ADC reference for validation

Read-only reference used exclusively by
tests/test_pipelined_adc_vs_reference.py (added in a later commit) to
cross-check the new pyDataconverter PipelinedADC against the vetted
upstream implementation. Stripped of __main__, plotting, ADCanalysis
(duplicate of utils/metrics/_dynamic.py), and the four stray print
statements in the original __init__ methods.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Relax `DACBase` to support `n_levels`

**Files:**
- Modify: `pyDataconverter/dataconverter.py:115-170` (class `DACBase`)
- Test: `tests/test_SimpleDAC.py` — add new test class at the end

- [ ] **Step 2.1: Write the failing test**

Append this class to `tests/test_SimpleDAC.py` (keeping existing imports at the top of the file):

```python
class TestDACBaseNLevels(unittest.TestCase):
    """DACBase now supports arbitrary n_levels, decoupled from n_bits."""

    def test_default_n_levels_matches_2_to_the_n_bits(self):
        # When n_levels is not supplied, DACBase (via SimpleDAC) should behave
        # exactly as before: n_levels == 2**n_bits, lsb == v_ref / (2^n - 1).
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE)
        self.assertEqual(dac.convert(0), 0.0)
        self.assertAlmostEqual(dac.convert(7), 1.0)
        self.assertAlmostEqual(dac.lsb, 1.0 / 7)

    def test_explicit_n_levels_overrides_n_bits(self):
        # 9 output levels over [0, v_ref] with lsb = v_ref / 8.
        dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0, output_type=OutputType.SINGLE)
        self.assertAlmostEqual(dac.lsb, 1.0 / 8)
        self.assertAlmostEqual(dac.convert(0), 0.0)
        self.assertAlmostEqual(dac.convert(4), 0.5)
        self.assertAlmostEqual(dac.convert(8), 1.0)

    def test_code_above_n_levels_raises(self):
        dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0, output_type=OutputType.SINGLE)
        with self.assertRaises(ValueError):
            dac.convert(9)

    def test_n_levels_less_than_two_raises(self):
        with self.assertRaises(ValueError):
            SimpleDAC(n_bits=3, n_levels=1, v_ref=1.0, output_type=OutputType.SINGLE)

    def test_n_levels_not_int_raises(self):
        with self.assertRaises(TypeError):
            SimpleDAC(n_bits=3, n_levels=3.5, v_ref=1.0, output_type=OutputType.SINGLE)
```

- [ ] **Step 2.2: Run tests to verify failure**

Run:

```bash
pytest tests/test_SimpleDAC.py::TestDACBaseNLevels -v
```

Expected: `test_default_n_levels_matches_2_to_the_n_bits` passes (no n_levels kwarg used); the other four tests fail because `SimpleDAC.__init__` does not accept `n_levels`.

- [ ] **Step 2.3: Modify `DACBase.__init__` to accept `n_levels`**

Open `pyDataconverter/dataconverter.py`. Find the `DACBase.__init__` method (around line 126). Replace it with:

```python
    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 output_type: OutputType = OutputType.SINGLE,
                 n_levels: int = None):
        # Validate n_bits
        if not isinstance(n_bits, int):
            raise TypeError("n_bits must be an integer")
        if n_bits < 1 or n_bits > 32:
            raise ValueError("n_bits must be between 1 and 32")

        # Validate v_ref
        if not isinstance(v_ref, (int, float)):
            raise TypeError("v_ref must be a number")
        if v_ref <= 0:
            raise ValueError("v_ref must be positive")

        # Validate output_type
        if not isinstance(output_type, OutputType):
            raise TypeError("output_type must be an OutputType enum")

        # Resolve n_levels: explicit override wins, else default to 2^n_bits.
        if n_levels is None:
            n_levels = 2 ** n_bits
        else:
            if not isinstance(n_levels, int) or isinstance(n_levels, bool):
                raise TypeError("n_levels must be an integer")
            if n_levels < 2:
                raise ValueError(f"n_levels must be >= 2, got {n_levels}")

        # Assign attributes
        self.n_bits = n_bits
        self.v_ref = v_ref
        self.output_type = output_type
        self.n_levels = n_levels
        self.lsb = v_ref / (n_levels - 1)
```

Now find `DACBase.convert` (around line 149). Replace the range check line `if digital_input < 0 or digital_input >= 2 ** self.n_bits:` with:

```python
        if digital_input < 0 or digital_input >= self.n_levels:
            raise ValueError(f"Digital input must be between 0 and {self.n_levels - 1}")
```

- [ ] **Step 2.4: Modify `SimpleDAC.__init__` to forward `n_levels`**

Open `pyDataconverter/architectures/SimpleDAC.py`. Find the `SimpleDAC.__init__` signature (around line 54). Replace the signature and the `super().__init__` call:

```python
    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 output_type: OutputType = OutputType.SINGLE,
                 noise_rms: float = 0.0,
                 offset: float = 0.0,
                 gain_error: float = 0.0,
                 fs: float = 1.0,
                 oversample: int = 1,
                 n_levels: int = None):
        super().__init__(n_bits, v_ref, output_type, n_levels=n_levels)
```

- [ ] **Step 2.5: Run tests**

```bash
pytest tests/test_SimpleDAC.py::TestDACBaseNLevels -v
```

Expected: all 5 tests pass.

Then run the full test suite to confirm no regressions:

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: `826+ passed` (no failures, no errors). Existing tests still pass because `n_levels=None` default preserves the `2**n_bits` behaviour.

- [ ] **Step 2.6: Commit**

```bash
git add pyDataconverter/dataconverter.py pyDataconverter/architectures/SimpleDAC.py tests/test_SimpleDAC.py
git commit -m "feat(dacbase): decouple n_levels from n_bits for non-power-of-2 DACs

Adds an optional n_levels kwarg to DACBase.__init__ that overrides the
default 2**n_bits derivation. When provided, lsb = v_ref / (n_levels - 1)
and convert() validates codes against [0, n_levels - 1]. Backward
compatible: the default preserves the existing 2**n_bits behaviour for
every caller that doesn't pass n_levels.

Needed for the upcoming PipelinedADC, whose canonical configuration uses
a 9-level sub-DAC (one more than a 3-bit DAC provides) to match the
vetted adc_book reference.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `code_errors` kwarg to `SimpleDAC`

**Files:**
- Modify: `pyDataconverter/architectures/SimpleDAC.py` (constructor and `_convert_input`)
- Test: `tests/test_SimpleDAC.py` — add new test class

- [ ] **Step 3.1: Write the failing test**

Append to `tests/test_SimpleDAC.py`:

```python
class TestSimpleDACCodeErrors(unittest.TestCase):
    """SimpleDAC supports per-code additive error injection."""

    def test_default_no_errors(self):
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE)
        # With no code_errors, behaves exactly like the ideal transfer
        for code in range(8):
            self.assertAlmostEqual(dac.convert(code), code / 7)

    def test_code_errors_applied_additively(self):
        errors = np.array([0.0, 0.01, -0.02, 0.005, 0.0, -0.01, 0.002, 0.0])
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                        code_errors=errors)
        for code in range(8):
            expected = code / 7 + errors[code]
            self.assertAlmostEqual(dac.convert(code), expected)

    def test_code_errors_with_n_levels(self):
        # 9-level DAC with a specific error pattern
        errors = np.array([0.0, -0.2, 0.3, 0.05, -0.15, 0.0, 0.3, -0.3, 0.0]) * 0.001
        dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0,
                        output_type=OutputType.SINGLE, code_errors=errors)
        for code in range(9):
            expected = code / 8 + errors[code]
            self.assertAlmostEqual(dac.convert(code), expected)

    def test_code_errors_wrong_length_raises(self):
        errors = np.zeros(7)  # should be 8 for n_bits=3
        with self.assertRaises(ValueError):
            SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                      code_errors=errors)

    def test_code_errors_not_array_raises(self):
        with self.assertRaises(TypeError):
            SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                      code_errors="not an array")

    def test_code_errors_applied_before_gain_offset_noise(self):
        # Code error + offset should both appear in the output
        errors = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                        offset=0.05, code_errors=errors)
        # code=1: ideal = 1/7, + code_error 0.1, + offset 0.05
        self.assertAlmostEqual(dac.convert(1), 1/7 + 0.1 + 0.05)
```

- [ ] **Step 3.2: Run tests to verify failure**

```bash
pytest tests/test_SimpleDAC.py::TestSimpleDACCodeErrors -v
```

Expected: all six tests fail with `TypeError: __init__() got an unexpected keyword argument 'code_errors'`.

- [ ] **Step 3.3: Add `code_errors` kwarg and apply it in `_convert_input`**

Open `pyDataconverter/architectures/SimpleDAC.py`. Replace the entire `__init__` method (including the signature from step 2.4) with:

```python
    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 output_type: OutputType = OutputType.SINGLE,
                 noise_rms: float = 0.0,
                 offset: float = 0.0,
                 gain_error: float = 0.0,
                 fs: float = 1.0,
                 oversample: int = 1,
                 n_levels: int = None,
                 code_errors: Optional[np.ndarray] = None):
        super().__init__(n_bits, v_ref, output_type, n_levels=n_levels)

        if noise_rms < 0:
            raise ValueError("noise_rms must be >= 0")
        if oversample < 1:
            raise ValueError("oversample must be >= 1")

        if code_errors is not None:
            if not isinstance(code_errors, np.ndarray):
                raise TypeError(
                    f"code_errors must be a numpy ndarray, got {type(code_errors).__name__}")
            if code_errors.ndim != 1:
                raise ValueError(
                    f"code_errors must be 1-D, got shape {code_errors.shape}")
            if len(code_errors) != self.n_levels:
                raise ValueError(
                    f"code_errors must have length n_levels={self.n_levels}, "
                    f"got {len(code_errors)}")

        self.noise_rms   = noise_rms
        self.offset      = offset
        self.gain_error  = gain_error
        self.fs          = fs
        self.oversample  = oversample
        self.code_errors = code_errors
```

Add `from typing import Optional` to the imports at the top of the file if not already present.

Now replace `_convert_input` with:

```python
    def _convert_input(self, digital_input: int) -> Union[float, Tuple[float, float]]:
        """
        Compute ideal voltage, apply per-code error (if any), apply non-idealities,
        then format output.
        """
        # Calculate ideal voltage
        voltage = digital_input * self.lsb

        # Apply per-code static error (injected via code_errors kwarg)
        if self.code_errors is not None:
            voltage = voltage + float(self.code_errors[digital_input])

        # Apply dynamic non-idealities
        voltage = self._apply_nonidealities(voltage)

        if self.output_type == OutputType.SINGLE:
            return voltage
        else:  # DIFFERENTIAL
            v_diff = 2 * voltage - self.v_ref
            v_pos = v_diff / 2 + self.v_ref / 2
            v_neg = -v_diff / 2 + self.v_ref / 2
            return (v_pos, v_neg)
```

- [ ] **Step 3.4: Run tests**

```bash
pytest tests/test_SimpleDAC.py::TestSimpleDACCodeErrors -v
```

Expected: all 6 tests pass.

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: all tests still pass, count increased by 6 (now 832+).

- [ ] **Step 3.5: Commit**

```bash
git add pyDataconverter/architectures/SimpleDAC.py tests/test_SimpleDAC.py
git commit -m "feat(simpledac): add code_errors kwarg for per-code static error injection

Optional 1-D array of length n_levels that is added to the ideal
transfer function before gain_error/offset/noise. Default None
preserves existing behaviour. Enables bit-exact reproduction of the
adc_book reference's subDAC.add_error() test pattern in the upcoming
PipelinedADC comparison tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Relax `FlashADC.n_comparators`

**Files:**
- Modify: `pyDataconverter/architectures/FlashADC.py:81-157` (constructor)
- Test: `tests/test_FlashADC.py` — add new test class

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/test_FlashADC.py`:

```python
class TestFlashADCRelaxedNComparators(unittest.TestCase):
    """FlashADC supports arbitrary comparator counts, not just 2^n_bits - 1."""

    def test_default_n_comparators_is_2_to_the_n_minus_1(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE)
        self.assertEqual(adc.n_comparators, 7)

    def test_explicit_n_comparators_overrides_default(self):
        # 1026 comparators for a high-resolution backend flash
        adc = FlashADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=1026)
        self.assertEqual(adc.n_comparators, 1026)

    def test_even_n_comparators_accepted(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=8)
        self.assertEqual(adc.n_comparators, 8)

    def test_output_code_range_expands(self):
        # With 8 comparators we should see codes in [0, 8] — 9 possible values
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=8)
        sweep = np.linspace(-0.01, 1.01, 401)
        codes = {adc.convert(float(v)) for v in sweep}
        self.assertTrue(min(codes) >= 0)
        self.assertTrue(max(codes) <= 8)
        self.assertGreater(len(codes), 5)  # sanity — should see multiple codes

    def test_n_comparators_not_int_raises(self):
        with self.assertRaises(TypeError):
            FlashADC(n_bits=3, v_ref=1.0, n_comparators=7.5)

    def test_n_comparators_zero_raises(self):
        with self.assertRaises(ValueError):
            FlashADC(n_bits=3, v_ref=1.0, n_comparators=0)
```

- [ ] **Step 4.2: Run tests to verify failure**

```bash
pytest tests/test_FlashADC.py::TestFlashADCRelaxedNComparators -v
```

Expected: `test_default_n_comparators_is_2_to_the_n_minus_1` passes; the rest fail with unexpected-kwarg errors or wrong values.

- [ ] **Step 4.3: Modify `FlashADC.__init__` to accept `n_comparators`**

Open `pyDataconverter/architectures/FlashADC.py`. Replace the `__init__` method (lines 81-156). The new signature and body:

```python
    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 input_type: InputType = InputType.SINGLE,
                 comparator_type: Type[ComparatorBase] = DifferentialComparator,
                 comparator_params: Optional[dict] = None,
                 offset_std: float = 0.0,
                 reference: Optional[ReferenceBase] = None,
                 reference_noise: float = 0.0,
                 resistor_mismatch: float = 0.0,
                 encoder_type: EncoderType = EncoderType.COUNT_ONES,
                 n_comparators: Optional[int] = None):
        """Initialize Flash ADC. See class docstring for parameter details."""
        super().__init__(n_bits, v_ref, input_type)

        if not isinstance(encoder_type, EncoderType):
            raise TypeError("encoder_type must be an EncoderType enum")

        # Resolve n_comparators: explicit override wins, else derived from n_bits.
        if n_comparators is None:
            self.n_comparators = 2 ** n_bits - 1
        else:
            if not isinstance(n_comparators, int) or isinstance(n_comparators, bool):
                raise TypeError(
                    f"n_comparators must be an integer, got {type(n_comparators).__name__}")
            if n_comparators < 1:
                raise ValueError(f"n_comparators must be >= 1, got {n_comparators}")
            self.n_comparators = n_comparators

        self.encoder_type = encoder_type

        # Reference generator
        if reference is not None:
            if not isinstance(reference, ReferenceBase):
                raise TypeError("reference must be a ReferenceBase instance")
            if reference.n_references != self.n_comparators:
                raise ValueError(
                    f"reference has {reference.n_references} taps but "
                    f"this FlashADC has n_comparators={self.n_comparators}")
            self.reference = reference
        else:
            # Build a default ladder whose length matches n_comparators. When
            # n_comparators is a non-power-of-2, we bypass ReferenceLadder's
            # 2^n_bits assumption by using ArbitraryReference with a linear
            # spacing that matches the standard ladder for the power-of-2 case.
            v_min = -v_ref / 4 if input_type == InputType.DIFFERENTIAL else 0.0
            v_max =  v_ref / 4 if input_type == InputType.DIFFERENTIAL else v_ref
            if self.n_comparators == 2 ** n_bits - 1:
                self.reference = ReferenceLadder(n_bits, v_min, v_max,
                                                 resistor_mismatch=resistor_mismatch,
                                                 noise_rms=reference_noise)
            else:
                # Linearly spaced thresholds across [v_min + lsb/2, v_max - lsb/2]
                lsb = (v_max - v_min) / self.n_comparators
                thresholds = v_min + lsb * (np.arange(self.n_comparators) + 0.5)
                self.reference = ArbitraryReference(thresholds, noise_rms=reference_noise)

        # Comparator bank
        if comparator_params is None:
            comparator_params = {}

        offsets = (np.random.normal(0, offset_std, self.n_comparators)
                   if offset_std > 0 else np.zeros(self.n_comparators))

        self.comparators = []
        for i in range(self.n_comparators):
            params = comparator_params.copy()
            params['offset'] = offsets[i]
            self.comparators.append(comparator_type(**params))
```

Add `ArbitraryReference` to the existing `ReferenceBase` / `ReferenceLadder` import at the top of `FlashADC.py`. Find:

```python
from pyDataconverter.components.reference import ReferenceBase, ReferenceLadder
```

Replace with:

```python
from pyDataconverter.components.reference import ReferenceBase, ReferenceLadder, ArbitraryReference
```

- [ ] **Step 4.4: Check `_convert_input` code-range behaviour for non-standard counts**

Find `_convert_input` in `FlashADC.py` (around line 210). The existing line:

```python
return int(np.clip(self._encode(thermometer), 0, 2 ** self.n_bits - 1))
```

must be changed so the clip uses `self.n_comparators` when it differs from the power-of-two default. Replace with:

```python
code = self._encode(thermometer)
max_code = self.n_comparators if self.n_comparators != 2 ** self.n_bits - 1 else 2 ** self.n_bits - 1
return int(np.clip(code, 0, max_code))
```

- [ ] **Step 4.5: Run tests**

```bash
pytest tests/test_FlashADC.py::TestFlashADCRelaxedNComparators -v
```

Expected: all 6 tests pass.

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: all tests still pass (838+). The pre-existing `FlashADC` tests use the default `n_comparators = 2^n_bits - 1` path, which is preserved unchanged.

- [ ] **Step 4.6: Commit**

```bash
git add pyDataconverter/architectures/FlashADC.py tests/test_FlashADC.py
git commit -m "feat(flashadc): decouple n_comparators from n_bits

Adds an optional n_comparators kwarg to FlashADC.__init__ that overrides
the default 2**n_bits - 1 derivation. When an explicit count is
supplied, the class builds an ArbitraryReference with linearly-spaced
thresholds instead of a power-of-two ReferenceLadder, and _convert_input
clips output codes to [0, n_comparators] rather than [0, 2**n_bits - 1].
Even counts and arbitrary positive integers are now accepted.

Backward compatible: the default still yields 2**n_bits - 1 comparators
and exercises the existing ReferenceLadder path, so every pre-existing
FlashADC test continues to pass unchanged.

Needed for the upcoming PipelinedADC, whose canonical 12-bit
configuration uses a 1026-comparator backend flash.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `tau_regen` + `vc_threshold` + `last_regen_time` to `DifferentialComparator`

**Files:**
- Modify: `pyDataconverter/components/comparator.py:106-187`
- Test: `tests/test_comparator.py` — add new test class

- [ ] **Step 5.1: Write the failing tests**

Append to `tests/test_comparator.py` (keeping existing imports):

```python
class TestDifferentialComparatorRegen(unittest.TestCase):
    """tau_regen models comparator regeneration time, used by pipelined ADC."""

    def test_default_tau_regen_is_zero(self):
        comp = DifferentialComparator()
        comp.compare(0.5, 0.0)
        self.assertEqual(comp.last_regen_time, 0.0)

    def test_default_vc_threshold_is_half(self):
        comp = DifferentialComparator(tau_regen=1e-12)
        self.assertEqual(comp.vc_threshold, 0.5)

    def test_regen_time_log_formula(self):
        # Expected: tau_regen * ln(vc_threshold / |v_diff|)
        tau = 1e-12
        comp = DifferentialComparator(tau_regen=tau, vc_threshold=0.5)
        comp.compare(v_pos=1e-3, v_neg=0.0)
        expected = tau * np.log(0.5 / 1e-3)
        self.assertAlmostEqual(comp.last_regen_time, expected, places=18)

    def test_regen_time_with_nonzero_refs(self):
        # v_diff = (v_pos - v_refp) - (v_neg - v_refn) = (0.3 - 0.1) - (0 - 0) = 0.2
        tau = 2e-12
        comp = DifferentialComparator(tau_regen=tau, vc_threshold=0.5)
        comp.compare(v_pos=0.3, v_neg=0.0, v_refp=0.1, v_refn=0.0)
        expected = tau * np.log(0.5 / 0.2)
        self.assertAlmostEqual(comp.last_regen_time, expected, places=18)

    def test_regen_time_floor_on_zero_v_diff(self):
        # v_diff = 0 exactly would log(inf) without the floor. With 1e-30 floor,
        # last_regen_time = tau * ln(Vc / 1e-30), a large but finite value.
        tau = 1e-12
        comp = DifferentialComparator(tau_regen=tau, vc_threshold=0.5)
        comp.compare(v_pos=0.0, v_neg=0.0)
        expected = tau * np.log(0.5 / 1e-30)
        self.assertAlmostEqual(comp.last_regen_time, expected, places=18)

    def test_tau_regen_negative_raises(self):
        with self.assertRaises(ValueError):
            DifferentialComparator(tau_regen=-1e-12)

    def test_vc_threshold_non_positive_raises(self):
        with self.assertRaises(ValueError):
            DifferentialComparator(tau_regen=1e-12, vc_threshold=0.0)
        with self.assertRaises(ValueError):
            DifferentialComparator(tau_regen=1e-12, vc_threshold=-0.1)

    def test_reset_clears_regen_time(self):
        comp = DifferentialComparator(tau_regen=1e-12)
        comp.compare(0.5, 0.0)
        self.assertGreater(comp.last_regen_time, 0)
        comp.reset()
        self.assertEqual(comp.last_regen_time, 0.0)
```

The test file already imports `DifferentialComparator`; confirm the import line is present near the top of the file. If `numpy` is not imported, add `import numpy as np`.

- [ ] **Step 5.2: Run tests to verify failure**

```bash
pytest tests/test_comparator.py::TestDifferentialComparatorRegen -v
```

Expected: every test fails. Typical errors include "`DifferentialComparator.__init__() got an unexpected keyword argument 'tau_regen'`" or "`'DifferentialComparator' object has no attribute 'last_regen_time'`".

- [ ] **Step 5.3: Modify `DifferentialComparator` to accept the new parameters**

Open `pyDataconverter/components/comparator.py`. Find `DifferentialComparator.__init__` (around line 106) and replace it with:

```python
    def __init__(self,
                 offset: float = 0.0,
                 noise_rms: float = 0.0,
                 bandwidth: Optional[float] = None,
                 hysteresis: float = 0.0,
                 time_constant: float = 0.0,
                 tau_regen: float = 0.0,
                 vc_threshold: float = 0.5):
        """
        Initialise comparator with specified non-idealities.

        Args:
            offset:        DC offset voltage (V).
            noise_rms:     RMS noise voltage (V).
            bandwidth:     -3 dB bandwidth (Hz); None = infinite.
            hysteresis:    Hysteresis voltage (V).
            time_constant: Time constant for temporal behaviour (s).
            tau_regen:     Regeneration time constant (s) used by pipelined
                           ADC metastability modelling. Default 0.0 disables
                           the model and makes last_regen_time always 0.
            vc_threshold:  Comparator output-voltage threshold at which the
                           latch is considered resolved. Default 0.5 matches
                           the adc_book reference.
        """
        if tau_regen < 0:
            raise ValueError(f"tau_regen must be >= 0, got {tau_regen}")
        if vc_threshold <= 0:
            raise ValueError(f"vc_threshold must be > 0, got {vc_threshold}")

        self.offset        = offset
        self.noise_rms     = noise_rms
        self.bandwidth     = bandwidth
        self.hysteresis    = hysteresis
        self.time_constant = time_constant
        self.tau_regen     = tau_regen
        self.vc_threshold  = vc_threshold
        self._last_output  = 0
        self._last_regen_time = 0.0

        if bandwidth is not None:
            self._filtered_state = 0.0
            self._tau = 1.0 / (2.0 * np.pi * bandwidth)

    @property
    def last_regen_time(self) -> float:
        """
        Regeneration time of the most recent compare() call, in seconds.

        Computed as ``tau_regen * ln(vc_threshold / max(|v_diff|, 1e-30))``.
        Returns 0.0 when ``tau_regen == 0`` (metastability modelling disabled).
        The 1e-30 floor on ``|v_diff|`` prevents ``log(0)`` when the input
        lands exactly on a threshold — an event that should be vanishingly
        rare for any realistic continuous input.
        """
        return self._last_regen_time
```

- [ ] **Step 5.4: Update `compare()` to compute the regeneration time**

Still in `pyDataconverter/components/comparator.py`, find the `compare` method (around line 133). Replace it with:

```python
    def compare(self,
                v_pos: float,
                v_neg: float,
                v_refp: float = 0.0,
                v_refn: float = 0.0,
                time_step: Optional[float] = None) -> int:
        """
        Compare (v_pos − v_refp) against (v_neg − v_refn) with non-idealities.
        See class docstring for parameter details.
        """
        v_diff = (v_pos - v_refp) - (v_neg - v_refn) + self.offset

        # Bandwidth limiting (first-order low-pass)
        if self.bandwidth is not None:
            if time_step is None:
                raise ValueError("time_step must be provided when bandwidth is specified")
            if time_step <= 0:
                raise ValueError(f"time_step must be positive, got {time_step}")
            alpha  = time_step / (time_step + self._tau)
            v_diff = (1 - alpha) * self._filtered_state + alpha * v_diff
            self._filtered_state = v_diff

        # Record regeneration time before adding noise — regen is a deterministic
        # physical quantity driven by the pre-noise comparator input. Noise is
        # modelled separately as an input-referred effect below.
        if self.tau_regen > 0:
            # 1e-30 floor prevents log(0) when v_diff is exactly on a threshold
            safe_mag = max(abs(v_diff), 1e-30)
            self._last_regen_time = self.tau_regen * float(np.log(self.vc_threshold / safe_mag))
        else:
            self._last_regen_time = 0.0

        # Input-referred noise
        if self.noise_rms > 0:
            v_diff += np.random.normal(0, self.noise_rms)

        # Hysteresis
        if self.hysteresis > 0:
            threshold = -self.hysteresis / 2 if self._last_output == 1 else self.hysteresis / 2
            result = 1 if v_diff > threshold else 0
        else:
            result = 1 if v_diff > 0 else 0

        self._last_output = result
        return result
```

- [ ] **Step 5.5: Update `reset()` to clear regen time**

Still in `comparator.py`, find `reset` (around line 183) and replace it with:

```python
    def reset(self):
        """Reset hysteresis history, bandwidth filter state, and last regen time."""
        self._last_output = 0
        self._last_regen_time = 0.0
        if self.bandwidth is not None:
            self._filtered_state = 0.0
```

- [ ] **Step 5.6: Run tests**

```bash
pytest tests/test_comparator.py::TestDifferentialComparatorRegen -v
```

Expected: all 8 tests pass.

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: full suite still passes (846+).

- [ ] **Step 5.7: Commit**

```bash
git add pyDataconverter/components/comparator.py tests/test_comparator.py
git commit -m "feat(comparator): add tau_regen model for metastability timing

DifferentialComparator gains two new kwargs (tau_regen, vc_threshold)
and one new read-only property (last_regen_time). With tau_regen>0,
every compare() call caches the regeneration time computed as
tau_regen * ln(vc_threshold / max(|v_diff|, 1e-30)). The 1e-30 floor
prevents log(0) on the measure-zero case of v_diff exactly on the
threshold. Default tau_regen=0 preserves the existing behaviour: the
property returns 0 and nothing else changes.

Needed by the upcoming FlashADC.last_conversion_time() and the
pipelined ADC's metastability-to-amplifier-settling coupling model.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Add `last_conversion_time()` and `last_metastable_sign()` to `FlashADC`

**Files:**
- Modify: `pyDataconverter/architectures/FlashADC.py` (append new methods, track last v_sampled)
- Test: `tests/test_FlashADC.py` — add new test class

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/test_FlashADC.py`:

```python
class TestFlashADCMetastabilityHooks(unittest.TestCase):
    """FlashADC reports metastability state for pipelined-ADC timing coupling."""

    def test_defaults_are_ideal(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE)
        adc.convert(0.5)
        self.assertEqual(adc.last_conversion_time(), 0.0)
        self.assertEqual(adc.last_metastable_sign(), 0)

    def test_last_conversion_time_aggregates_max_across_bank(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       comparator_params={"tau_regen": 1e-12})
        adc.convert(0.5)
        # The max regen happens at whichever comparator's threshold is closest
        # to 0.5 — we can't predict which without knowing the ladder, but we
        # know the reported value equals the maximum of the per-comparator
        # last_regen_time values.
        expected = max(c.last_regen_time for c in adc.comparators)
        self.assertAlmostEqual(adc.last_conversion_time(), expected, places=18)
        self.assertGreater(adc.last_conversion_time(), 0.0)

    def test_metastable_sign_positive_when_nearest_threshold_above_input(self):
        # Single-ended flash: thresholds span (0, v_ref). Near 0 the nearest
        # threshold is above v_in, so sign should be +1.
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       comparator_params={"tau_regen": 1e-12})
        adc.convert(0.05)
        self.assertEqual(adc.last_metastable_sign(), +1)

    def test_metastable_sign_negative_when_nearest_threshold_below_input(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       comparator_params={"tau_regen": 1e-12})
        adc.convert(0.95)
        self.assertEqual(adc.last_metastable_sign(), -1)

    def test_sign_zero_when_tau_regen_zero(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE)
        adc.convert(0.5)
        self.assertEqual(adc.last_metastable_sign(), 0)

    def test_relaxed_count_with_regen_still_aggregates(self):
        # A non-power-of-2 flash should still aggregate correctly
        adc = FlashADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=8,
                       comparator_params={"tau_regen": 2e-12})
        adc.convert(0.3)
        self.assertGreater(adc.last_conversion_time(), 0.0)
```

- [ ] **Step 6.2: Run tests to verify failure**

```bash
pytest tests/test_FlashADC.py::TestFlashADCMetastabilityHooks -v
```

Expected: all 6 tests fail — `AttributeError: 'FlashADC' object has no attribute 'last_conversion_time'` / `'last_metastable_sign'`.

- [ ] **Step 6.3: Track the last v_sampled in `_convert_input`**

Open `pyDataconverter/architectures/FlashADC.py`. Find `_convert_input`. The existing method body begins with `comp_refs = self.reference.get_voltages()`. **Prepend** the following four lines at the very top of the method body (immediately after the docstring, before `comp_refs = ...`), leaving everything else in the method unchanged:

```python
        # Cache the effective single-ended input for metastability reporting
        if self.input_type == InputType.DIFFERENTIAL:
            v_pos_in, v_neg_in = analog_input
            self._last_v_sampled = float(v_pos_in) - float(v_neg_in)
        else:
            self._last_v_sampled = float(analog_input)
```

Use the names `v_pos_in` / `v_neg_in` here to avoid shadowing the existing `v_pos`, `v_neg` locals that are unpacked later in the DIFFERENTIAL branch of the thermometer loop — that re-unpack is harmless and can be left as-is.

Also, initialise `self._last_v_sampled = 0.0` at the end of `__init__` (right after the comparator bank is built) so querying before the first `convert` call returns a sane value. Add this single line:

```python
        self._last_v_sampled = 0.0
```

immediately after the `self.comparators.append(...)` loop in `__init__`.

- [ ] **Step 6.4: Add `last_conversion_time` and `last_metastable_sign` methods**

Add these two methods to the `FlashADC` class body, after the existing `reset` method:

```python
    def last_conversion_time(self) -> float:
        """
        Regeneration time of the slowest comparator from the most recent
        convert() call, in seconds.

        Aggregates ``last_regen_time`` across the comparator bank via
        ``max()``. Returns 0.0 when every comparator has ``tau_regen=0``
        (metastability modelling disabled).
        """
        return max((c.last_regen_time for c in self.comparators), default=0.0)

    def last_metastable_sign(self) -> int:
        """
        Sign of the initial-condition error a residue amp would see at the
        start of its settling window, for the most recent convert() call.

        Returns:
            +1 if the sub-ADC's nearest threshold is above the last input,
            -1 if the nearest threshold is below the last input,
             0 if metastability modelling is disabled (tau_regen=0 on every
               comparator) — in which case downstream pipelined ADCs treat
               the stage as operating ideally.

        Internally uses ``self.reference.voltages`` (not ``get_voltages()``)
        so the sign reflects the static ladder geometry, independent of any
        dynamic reference noise realisation.
        """
        # Aggregate flag: if no comparator has regen enabled, return 0.
        if not any(c.tau_regen > 0 for c in self.comparators):
            return 0
        thresholds = self.reference.voltages
        if len(thresholds) == 0:
            return 0
        diffs = np.abs(thresholds - self._last_v_sampled)
        i_nearest = int(np.argmin(diffs))
        # +1 if threshold > v_sampled, else -1 (strictly greater — ties go to -1)
        return 1 if (thresholds[i_nearest] - self._last_v_sampled) > 0 else -1
```

- [ ] **Step 6.5: Run tests**

```bash
pytest tests/test_FlashADC.py::TestFlashADCMetastabilityHooks -v
```

Expected: all 6 tests pass.

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: full suite still passes (852+).

- [ ] **Step 6.6: Commit**

```bash
git add pyDataconverter/architectures/FlashADC.py tests/test_FlashADC.py
git commit -m "feat(flashadc): add last_conversion_time and last_metastable_sign

Two new methods expose the metastability state of the comparator bank
after each convert() call, needed by the pipelined ADC to couple the
sub-ADC's regeneration time to the residue amplifier's settling budget.

last_conversion_time() returns the max regeneration time across the
bank (the slowest comparator dominates the timing budget).
last_metastable_sign() returns ±1 indicating which side of its nearest
threshold the last input landed on, so a residue amp downstream can
compute the signed LSB-scale initial-condition error.

Both methods return 0 when every comparator has tau_regen=0 (default),
so pre-existing FlashADC users see no behaviour change.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: New component — `ResidueAmplifier`

**Files:**
- Create: `pyDataconverter/components/residue_amplifier.py`
- Modify: `pyDataconverter/components/__init__.py` (export `ResidueAmplifier`)
- Create: `tests/test_residue_amplifier.py`

- [ ] **Step 7.1: Write the failing test file**

Create `tests/test_residue_amplifier.py` with:

```python
"""
Tests for ResidueAmplifier component (used by pipelined ADC).
"""

import unittest
import math
import numpy as np
from pyDataconverter.components.residue_amplifier import ResidueAmplifier


class TestResidueAmplifierIdeal(unittest.TestCase):
    """settling_tau=0 collapses to an instantaneous ideal amp."""

    def test_ideal_amp_returns_gain_times_target(self):
        amp = ResidueAmplifier(gain=2.0, settling_tau=0.0)
        self.assertAlmostEqual(amp.amplify(target=0.3, initial_error=0.0, t_budget=1e-9), 0.6)

    def test_ideal_amp_ignores_initial_error(self):
        amp = ResidueAmplifier(gain=2.0, settling_tau=0.0)
        self.assertAlmostEqual(
            amp.amplify(target=0.3, initial_error=0.05, t_budget=1e-9), 0.6
        )

    def test_ideal_amp_includes_offset(self):
        amp = ResidueAmplifier(gain=4.0, offset=0.01, settling_tau=0.0)
        self.assertAlmostEqual(amp.amplify(0.1, 0.0, 1e-9), 4.0 * 0.1 + 0.01)


class TestResidueAmplifierSettling(unittest.TestCase):
    """With settling_tau > 0, exponential approach to target."""

    def test_zero_initial_error_returns_target(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        self.assertAlmostEqual(amp.amplify(0.3, 0.0, 2e-9), 0.3)

    def test_one_tau_settles_to_one_over_e(self):
        # amplify returns target + initial_error * exp(-t_budget/settling_tau).
        # Here gain has already been applied: we pass the "residue-output-units"
        # initial_error directly.
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        result = amp.amplify(target=0.0, initial_error=1.0, t_budget=1e-9)
        self.assertAlmostEqual(result, math.exp(-1.0), places=12)

    def test_infinite_budget_full_settling(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        self.assertAlmostEqual(
            amp.amplify(target=0.5, initial_error=100.0, t_budget=float('inf')),
            0.5,
        )

    def test_zero_budget_preserves_initial_error(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        self.assertAlmostEqual(
            amp.amplify(target=0.5, initial_error=100.0, t_budget=0.0),
            100.5,
        )

    def test_negative_budget_overshoots(self):
        # Deliberate pathological case: t_budget < 0 implies exp(+positive) > 1,
        # matching the reference's unguarded behaviour at TR < 0.
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        result = amp.amplify(target=0.0, initial_error=1.0, t_budget=-1e-9)
        self.assertAlmostEqual(result, math.exp(1.0), places=12)


class TestResidueAmplifierConstructor(unittest.TestCase):
    """Validation at __init__ time."""

    def test_zero_gain_raises(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=0.0)

    def test_negative_settling_tau_raises(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=2.0, settling_tau=-1e-9)

    def test_negative_slew_rate_raises(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=2.0, slew_rate=-1.0)

    def test_output_swing_ordering(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=2.0, output_swing=(0.5, -0.5))

    def test_negative_gain_allowed(self):
        # Inverting amps are valid.
        amp = ResidueAmplifier(gain=-2.0, settling_tau=0.0)
        self.assertAlmostEqual(amp.amplify(0.1, 0.0, 1e-9), -0.2)


class TestResidueAmplifierClipping(unittest.TestCase):
    """Output swing clipping."""

    def test_output_swing_clips_positive(self):
        amp = ResidueAmplifier(gain=10.0, settling_tau=0.0, output_swing=(-1.0, 1.0))
        self.assertEqual(amp.amplify(0.5, 0.0, 1e-9), 1.0)

    def test_output_swing_clips_negative(self):
        amp = ResidueAmplifier(gain=10.0, settling_tau=0.0, output_swing=(-1.0, 1.0))
        self.assertEqual(amp.amplify(-0.5, 0.0, 1e-9), -1.0)

    def test_no_output_swing_unclipped(self):
        amp = ResidueAmplifier(gain=1000.0, settling_tau=0.0)
        self.assertAlmostEqual(amp.amplify(0.5, 0.0, 1e-9), 500.0)
```

- [ ] **Step 7.2: Run tests to verify failure**

```bash
pytest tests/test_residue_amplifier.py -v
```

Expected: every test fails with `ModuleNotFoundError: No module named 'pyDataconverter.components.residue_amplifier'`.

- [ ] **Step 7.3: Create `ResidueAmplifier`**

Create `pyDataconverter/components/residue_amplifier.py`:

```python
"""
Residue Amplifier Component
===========================

Models the closed-loop residue amplifier used in pipelined ADC stages.
Captures finite gain, offset, slew rate, exponential settling driven by a
caller-supplied time budget, and optional output-swing saturation.

The amplification contract is designed to bit-exactly reproduce the
metastability-to-settling coupling from the adc_book reference
(see docs/superpowers/specs/2026-04-13-pipelined-adc-design.md, Appendix A):

    v_out = target + initial_error * exp(-t_budget / settling_tau)

where `target` is the "ideal" residue the amp would settle to given infinite
time and `initial_error` is the signed perturbation the amp starts at
(typically a sub-DAC-LSB-scale correction driven by which sub-ADC
comparator was slowest to resolve). Edge cases — settling_tau=0, infinite
t_budget, zero initial_error — are handled explicitly to avoid NaN from
IEEE 754 corner cases.

Classes:
    ResidueAmplifier: Configurable residue amplifier for pipelined stages.
"""

import math
from typing import Optional, Tuple

import numpy as np


class ResidueAmplifier:
    """
    Closed-loop residue amplifier with configurable non-idealities.

    Attributes:
        gain:          Closed-loop voltage gain (signed; inverting amps OK).
        offset:        Output-referred DC offset voltage (V).
        slew_rate:     Peak rate of change of the output (V/s). 0 or +inf
                       disables slew limiting. Optional; Phase 1 uses the
                       default (no slew limiting) because the reference has
                       no slew model to compare against.
        settling_tau:  First-order settling time constant (s). 0 means an
                       instantaneous ideal amp — no initial-condition decay.
        output_swing:  Optional (v_min, v_max) clipping bounds. None = no
                       clipping.
    """

    def __init__(self,
                 gain: float,
                 offset: float = 0.0,
                 slew_rate: float = float('inf'),
                 settling_tau: float = 0.0,
                 output_swing: Optional[Tuple[float, float]] = None):
        if not isinstance(gain, (int, float)):
            raise TypeError(f"gain must be a number, got {type(gain).__name__}")
        if gain == 0:
            raise ValueError("gain must be nonzero")
        if not isinstance(offset, (int, float)):
            raise TypeError(f"offset must be a number, got {type(offset).__name__}")
        if slew_rate < 0:
            raise ValueError(f"slew_rate must be non-negative, got {slew_rate}")
        if settling_tau < 0:
            raise ValueError(f"settling_tau must be non-negative, got {settling_tau}")
        if output_swing is not None:
            v_min, v_max = output_swing
            if v_max <= v_min:
                raise ValueError(
                    f"output_swing v_max must exceed v_min, got ({v_min}, {v_max})")

        self.gain         = float(gain)
        self.offset       = float(offset)
        self.slew_rate    = float(slew_rate)
        self.settling_tau = float(settling_tau)
        self.output_swing = output_swing

    def amplify(self,
                target: float,
                initial_error: float,
                t_budget: float) -> float:
        """
        Apply finite-gain + exponential-settling amplification.

        Args:
            target:        Ideal amplified residue the amp would reach given
                           infinite time. Typically ``gain * (v_in - v_dac)``
                           supplied by the caller.
            initial_error: Signed perturbation the amp starts at, in
                           residue-output units. The amp exponentially
                           decays this toward zero over ``t_budget``.
            t_budget:      Seconds of settling time available. NOT clamped —
                           negative values produce ``exp(+positive) > 1``,
                           matching the reference's unguarded behaviour at
                           TR < 0.

        Returns:
            Amplified residue voltage, clipped to ``output_swing`` if set.

        Edge cases (handled explicitly to avoid NaN):
            - ``settling_tau == 0``: return ``target + offset`` regardless of
              ``initial_error`` or ``t_budget`` (instantaneous ideal amp).
            - ``initial_error == 0``: short-circuit to ``target + offset``.
            - ``t_budget == +inf``: full settling, return ``target + offset``.
        """
        # Ideal-amp / no-error / infinite-time degenerate cases short-circuit
        # to avoid IEEE 754 0*inf and exp(-t/0) NaN traps.
        if self.settling_tau == 0 or initial_error == 0 or math.isinf(t_budget) and t_budget > 0:
            v_out = target + self.offset
        else:
            decay = math.exp(-t_budget / self.settling_tau)
            v_out = target + initial_error * decay + self.offset

        if self.output_swing is not None:
            v_min, v_max = self.output_swing
            if v_out > v_max:
                v_out = v_max
            elif v_out < v_min:
                v_out = v_min

        return v_out

    def __repr__(self) -> str:
        parts = [f"gain={self.gain}"]
        if self.offset:
            parts.append(f"offset={self.offset}")
        if self.settling_tau:
            parts.append(f"settling_tau={self.settling_tau:.3e}")
        if not math.isinf(self.slew_rate):
            parts.append(f"slew_rate={self.slew_rate:.3e}")
        if self.output_swing is not None:
            parts.append(f"output_swing={self.output_swing}")
        return f"ResidueAmplifier({', '.join(parts)})"
```

- [ ] **Step 7.4: Export from `components/__init__.py`**

Open `pyDataconverter/components/__init__.py` and append (or add if the file does not import from `residue_amplifier`):

```python
from .residue_amplifier import ResidueAmplifier
```

- [ ] **Step 7.5: Run tests**

```bash
pytest tests/test_residue_amplifier.py -v
```

Expected: all 14 tests pass.

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: full suite still passes (866+).

- [ ] **Step 7.6: Commit**

```bash
git add pyDataconverter/components/residue_amplifier.py pyDataconverter/components/__init__.py tests/test_residue_amplifier.py
git commit -m "feat(components): add ResidueAmplifier for pipelined ADC stages

New first-class component modelling closed-loop residue amplifiers used
by the upcoming pipelined ADC. Core method:

    amplify(target, initial_error, t_budget) -> v_out
    v_out = target + initial_error * exp(-t_budget / settling_tau) + offset

Three IEEE 754 edge cases (settling_tau=0, initial_error=0,
t_budget=+inf) are short-circuited to return target+offset directly,
preventing NaN from 0*inf and exp(-t/0). Negative t_budget is
deliberately NOT guarded so that pathological configurations reproduce
the adc_book reference's unguarded behaviour for bit-exact comparison.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: New architecture — `PipelineStage` helper class

**Files:**
- Create: `pyDataconverter/architectures/PipelinedADC.py` (stage class only; ADC class in Task 9)
- Create: `tests/test_pipelined_adc.py` (stage tests only)

- [ ] **Step 8.1: Write the failing test file**

Create `tests/test_pipelined_adc.py`:

```python
"""
Tests for PipelinedADC and PipelineStage.
"""

import math
import unittest

import numpy as np

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.architectures.PipelinedADC import (
    PipelineStage,
    PipelinedADC,
)
from pyDataconverter.components.residue_amplifier import ResidueAmplifier
from pyDataconverter.dataconverter import InputType, OutputType


def _make_ideal_stage(n_comparators: int = 8,
                     gain: float = 4.0,
                     fs: float = 1e9,
                     code_offset: int = 0) -> PipelineStage:
    """Helper: build an ideal PipelineStage with a single-ended FlashADC sub-ADC
    and a SimpleDAC sub-DAC sized to match."""
    sub_adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=n_comparators)
    sub_dac = SimpleDAC(n_bits=3, n_levels=n_comparators + 1, v_ref=1.0,
                        output_type=OutputType.SINGLE)
    residue_amp = ResidueAmplifier(gain=gain, settling_tau=0.0)
    return PipelineStage(sub_adc=sub_adc,
                         sub_dac=sub_dac,
                         residue_amp=residue_amp,
                         fs=fs,
                         code_offset=code_offset)


class TestPipelineStageConstruction(unittest.TestCase):
    def test_accepts_valid_components(self):
        stage = _make_ideal_stage()
        self.assertIsInstance(stage.sub_adc, FlashADC)
        self.assertIsInstance(stage.sub_dac, SimpleDAC)
        self.assertIsInstance(stage.residue_amp, ResidueAmplifier)

    def test_sub_adc_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            PipelineStage(sub_adc="not an ADCBase",
                          sub_dac=SimpleDAC(n_bits=3, v_ref=1.0),
                          residue_amp=ResidueAmplifier(gain=2.0),
                          fs=1e9)

    def test_sub_dac_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            PipelineStage(sub_adc=FlashADC(n_bits=3, v_ref=1.0),
                          sub_dac=42,
                          residue_amp=ResidueAmplifier(gain=2.0),
                          fs=1e9)

    def test_residue_amp_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            PipelineStage(sub_adc=FlashADC(n_bits=3, v_ref=1.0),
                          sub_dac=SimpleDAC(n_bits=3, v_ref=1.0),
                          residue_amp="not a residue amp",
                          fs=1e9)

    def test_fs_non_positive_raises(self):
        with self.assertRaises(ValueError):
            PipelineStage(sub_adc=FlashADC(n_bits=3, v_ref=1.0),
                          sub_dac=SimpleDAC(n_bits=3, v_ref=1.0),
                          residue_amp=ResidueAmplifier(gain=2.0),
                          fs=0.0)

    def test_code_offset_non_int_raises(self):
        with self.assertRaises(TypeError):
            _make_ideal_stage(code_offset=1.5)

    def test_h_defaults_to_residue_gain(self):
        stage = _make_ideal_stage(gain=4.0)
        self.assertEqual(stage.H, 4.0)

    def test_h_explicit_override(self):
        sub_adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                           n_comparators=8)
        sub_dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0)
        amp = ResidueAmplifier(gain=4.0)
        stage = PipelineStage(sub_adc=sub_adc, sub_dac=sub_dac,
                              residue_amp=amp, fs=1e9, H=5.0)
        self.assertEqual(stage.H, 5.0)


class TestPipelineStageConvertIdeal(unittest.TestCase):
    def test_residue_is_gain_times_input_minus_dac(self):
        # With settling_tau=0 and tau_regen=0, metastability is disabled and
        # the residue should be exactly gain * (v_in - sub_dac.convert(raw_code)).
        stage = _make_ideal_stage(gain=4.0)
        v_in = 0.3
        raw_code, shifted_code, v_res = stage.convert_stage(v_in)
        expected_residue = 4.0 * (v_in - stage.sub_dac.convert(raw_code))
        self.assertAlmostEqual(v_res, expected_residue)

    def test_code_offset_does_not_affect_sub_dac_input(self):
        # The sub-DAC sees the raw code; the combiner sees the offset code.
        # We verify this by constructing two stages with different code_offsets
        # and confirming their residue is identical for the same input.
        stage0 = _make_ideal_stage(code_offset=0)
        stage1 = _make_ideal_stage(code_offset=-1)
        v_in = 0.42
        _, shifted_0, v_res_0 = stage0.convert_stage(v_in)
        raw_1, shifted_1, v_res_1 = stage1.convert_stage(v_in)
        self.assertAlmostEqual(v_res_0, v_res_1)
        self.assertEqual(shifted_1, raw_1 - 1)
```

- [ ] **Step 8.2: Run tests to verify failure**

```bash
pytest tests/test_pipelined_adc.py -v
```

Expected: every test fails with `ModuleNotFoundError: No module named 'pyDataconverter.architectures.PipelinedADC'`.

- [ ] **Step 8.3: Create `PipelinedADC.py` with just `PipelineStage`**

Create `pyDataconverter/architectures/PipelinedADC.py`:

```python
"""
Pipelined ADC Module
====================

Implements a cascaded pipelined ADC that composes any ADCBase as the sub-ADC
and any DACBase as the sub-DAC for each stage. Metastability-to-settling
coupling is preserved via composition: DifferentialComparator owns regen
time, FlashADC aggregates across its bank, ResidueAmplifier owns settling,
and PipelineStage coordinates the timing budget.

Full design: docs/superpowers/specs/2026-04-13-pipelined-adc-design.md

Classes:
    PipelineStage:  One stage of the cascade (helper, not ADCBase).
    PipelinedADC:   Top-level N-stage pipelined ADC (inherits ADCBase).
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from pyDataconverter.dataconverter import ADCBase, DACBase, InputType
from pyDataconverter.components.residue_amplifier import ResidueAmplifier


class PipelineStage:
    """
    One stage of a pipelined ADC.

    Composes a sub-ADC (any ADCBase), a sub-DAC (any DACBase), and a
    ResidueAmplifier. Performs one stage of the pipelined conversion:

        raw_code = sub_adc.convert(v_sampled)
        v_dac    = sub_dac.convert(raw_code)
        residue  = residue_amp.amplify(
                       target        = gain * (v_sampled - v_dac + offset),
                       initial_error = sign * gain * sub_dac.lsb,
                       t_budget      = 1/(2*fs) - sub_adc.last_conversion_time(),
                   )

    where ``sign`` comes from ``sub_adc.last_metastable_sign()`` and ``gain``
    is ``residue_amp.gain``. The stage returns ``(raw_code, shifted_code, residue)``
    where ``shifted_code = raw_code + code_offset``.

    Attributes:
        sub_adc:       The ADCBase instance used as this stage's sub-ADC.
        sub_dac:       The DACBase instance used as this stage's sub-DAC.
        residue_amp:   The ResidueAmplifier producing the amplified residue.
        fs:            Sample rate (Hz) used for the timing budget.
        offset:        Input-referred offset (V).
        code_offset:   Integer added to raw_code before the digital combiner.
        H:             Weight applied to this stage in the digital combiner,
                       defaulting to ``residue_amp.gain``.
    """

    def __init__(self,
                 sub_adc,
                 sub_dac,
                 residue_amp: ResidueAmplifier,
                 fs: float,
                 offset: float = 0.0,
                 code_offset: int = 0,
                 H: Optional[float] = None):
        if not isinstance(sub_adc, ADCBase):
            raise TypeError(
                f"sub_adc must be an ADCBase instance, got {type(sub_adc).__name__}")
        if not isinstance(sub_dac, DACBase):
            raise TypeError(
                f"sub_dac must be a DACBase instance, got {type(sub_dac).__name__}")
        if not isinstance(residue_amp, ResidueAmplifier):
            raise TypeError(
                f"residue_amp must be a ResidueAmplifier instance, got {type(residue_amp).__name__}")
        if not isinstance(fs, (int, float)):
            raise TypeError(f"fs must be a number, got {type(fs).__name__}")
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")
        if not isinstance(code_offset, int) or isinstance(code_offset, bool):
            raise TypeError(
                f"code_offset must be an integer, got {type(code_offset).__name__}")

        self.sub_adc     = sub_adc
        self.sub_dac     = sub_dac
        self.residue_amp = residue_amp
        self.fs          = float(fs)
        self.offset      = float(offset)
        self.code_offset = code_offset
        self.H           = float(H) if H is not None else float(residue_amp.gain)

    def convert_stage(self, v_sampled: float) -> Tuple[int, int, float]:
        """
        Perform one pipelined-stage conversion.

        Args:
            v_sampled: Analog input to this stage (post first-stage S/H for
                stage 0; the amplified residue from stage i-1 for i > 0).

        Returns:
            (raw_code, shifted_code, v_res):
                raw_code     — the sub-ADC's output (0..n_codes-1).
                shifted_code — raw_code + code_offset, for the digital
                               combiner.
                v_res        — amplified residue, in the same voltage
                               coordinate system as v_sampled, to be fed
                               to the next stage.
        """
        raw_code = int(self.sub_adc.convert(float(v_sampled)))
        v_dac    = self.sub_dac.convert(raw_code)
        # Sub-DAC may return a float or a tuple (differential). We only
        # support single-ended sub-DACs in the stage's subtraction; the
        # output_type of the sub-DAC must be SINGLE.
        delta_v  = float(v_sampled) - float(v_dac) + self.offset
        target   = self.residue_amp.gain * delta_v

        # Metastability coupling (see spec Appendix A)
        t_regen = float(self.sub_adc.last_conversion_time()) \
            if hasattr(self.sub_adc, "last_conversion_time") else 0.0
        sign = int(self.sub_adc.last_metastable_sign()) \
            if hasattr(self.sub_adc, "last_metastable_sign") else 0
        initial_error = sign * self.residue_amp.gain * float(self.sub_dac.lsb)

        t_budget = 1.0 / (2.0 * self.fs) - t_regen  # NOT clamped

        v_res = self.residue_amp.amplify(
            target        = target,
            initial_error = initial_error,
            t_budget      = t_budget,
        )

        shifted_code = raw_code + self.code_offset
        return raw_code, shifted_code, v_res
```

- [ ] **Step 8.4: Run tests**

```bash
pytest tests/test_pipelined_adc.py -v
```

Expected: both test classes pass (10 tests).

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: full suite still passes (876+).

- [ ] **Step 8.5: Commit**

```bash
git add pyDataconverter/architectures/PipelinedADC.py tests/test_pipelined_adc.py
git commit -m "feat(pipelined): add PipelineStage helper class

One stage of a pipelined ADC. Composes any ADCBase as sub-ADC, any
DACBase as sub-DAC, and a ResidueAmplifier. convert_stage(v_sampled)
returns (raw_code, shifted_code, residue) where shifted_code applies a
per-stage integer code offset (equivalent of the reference's
minADCcode) without affecting the sub-DAC input.

Metastability coupling is queried from the sub-ADC via the optional
last_conversion_time() and last_metastable_sign() methods; when absent
(e.g., a SARADC sub-ADC without the Phase 2 plumbing) the stage treats
the sub-ADC as operating ideally.

PipelinedADC top-level class is added in the next commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: New architecture — `PipelinedADC(ADCBase)`

**Files:**
- Modify: `pyDataconverter/architectures/PipelinedADC.py` (add `PipelinedADC` class)
- Modify: `pyDataconverter/architectures/__init__.py` (export both)
- Modify: `tests/test_pipelined_adc.py` (add ADC-level test classes)

- [ ] **Step 9.1: Write the failing tests**

Append to `tests/test_pipelined_adc.py`:

```python
def _make_canonical_pipelined_adc(fs: float = 1e9) -> PipelinedADC:
    """Canonical 12-bit pipelined ADC: [3-bit 9-level stage, 1026-level backend].

    Mirrors the adc_book __main__ example: Nstages=2, N=[8,1026], FSR=[1,1],
    G=[4, 512], minADCcode=[-1, 0]. The pyDataconverter build plugs in
    relaxed FlashADC and SimpleDAC classes for the sub-components and a new
    ResidueAmplifier for the stage gain.
    """
    stage0_sub_adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                              n_comparators=8)
    stage0_sub_dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0,
                               output_type=OutputType.SINGLE)
    stage0_amp = ResidueAmplifier(gain=4.0, settling_tau=0.0)
    stage0 = PipelineStage(sub_adc=stage0_sub_adc,
                           sub_dac=stage0_sub_dac,
                           residue_amp=stage0_amp,
                           fs=fs,
                           code_offset=-1)
    backend = FlashADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=1026)
    return PipelinedADC(n_bits=12,
                        v_ref=1.0,
                        input_type=InputType.SINGLE,
                        stages=[stage0],
                        backend=backend,
                        backend_H=512,
                        backend_code_offset=0,
                        fs=fs)


class TestPipelinedADCConstruction(unittest.TestCase):
    def test_minimal_valid_construction(self):
        adc = _make_canonical_pipelined_adc()
        self.assertEqual(adc.n_bits, 12)
        self.assertEqual(len(adc.stages), 1)
        self.assertIsInstance(adc.backend, FlashADC)

    def test_empty_stages_raises(self):
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(ValueError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[], backend=backend, backend_H=1, fs=1e9)

    def test_stages_wrong_element_type_raises(self):
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(TypeError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=["not a stage"], backend=backend,
                         backend_H=1, fs=1e9)

    def test_backend_wrong_type_raises(self):
        stage = _make_ideal_stage()
        with self.assertRaises(TypeError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[stage], backend="not an ADC",
                         backend_H=1, fs=1e9)

    def test_backend_h_non_positive_raises(self):
        stage = _make_ideal_stage()
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(ValueError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[stage], backend=backend,
                         backend_H=0.0, fs=1e9)

    def test_fs_non_positive_raises(self):
        stage = _make_ideal_stage()
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(ValueError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[stage], backend=backend,
                         backend_H=1, fs=0.0)


class TestPipelinedADCConvert(unittest.TestCase):
    def test_midscale_input_gives_midscale_code(self):
        adc = _make_canonical_pipelined_adc()
        code = adc.convert(0.5)
        # Ideal 12-bit midscale is ~2048; allow +/-5 LSB tolerance
        self.assertAlmostEqual(code, 2048, delta=5)

    def test_monotonic_sweep_produces_monotonic_codes(self):
        adc = _make_canonical_pipelined_adc()
        sweep = np.linspace(-0.45, 0.45, 201)
        codes = np.array([adc.convert(float(v)) for v in sweep], dtype=int)
        # Non-strictly monotonic (repeats allowed; reversals not)
        diffs = np.diff(codes)
        self.assertTrue(np.all(diffs >= 0),
                        f"Non-monotonic codes detected: diffs={diffs[diffs<0]}")

    def test_clip_output_true_saturates(self):
        adc = _make_canonical_pipelined_adc()
        # Far negative input should clip to 0
        self.assertEqual(adc.convert(-10.0), 0)
        # Far positive input should clip to 2^12 - 1
        self.assertEqual(adc.convert(+10.0), 2**12 - 1)

    def test_returns_int(self):
        adc = _make_canonical_pipelined_adc()
        result = adc.convert(0.25)
        self.assertIsInstance(result, int)
```

- [ ] **Step 9.2: Run tests to verify failure**

```bash
pytest tests/test_pipelined_adc.py -v
```

Expected: the stage tests still pass; the new ADC-level tests fail because `PipelinedADC` class is not defined in the module.

- [ ] **Step 9.3: Append `PipelinedADC` class to the module**

Open `pyDataconverter/architectures/PipelinedADC.py` and add at the end:

```python
class PipelinedADC(ADCBase):
    """
    N-stage pipelined ADC with a required backend.

    A cascade of ``PipelineStage`` instances followed by a mandatory backend
    ADC. The first stage applies an optional SARADC-style sample-and-hold
    (``noise_rms``, ``offset``, ``gain_error``, ``t_jitter``) before handing
    the signal to stage 0. Each stage produces a partial code and a residue
    that feeds the next stage. The backend digitises the final residue.

    Digital combiner: ``DOUT += DOUT * stage.H + code`` per stage and once
    more with ``backend_H`` for the backend, bit-exactly matching the
    adc_book reference's accumulation formula.

    Attributes:
        stages:              List of PipelineStage instances.
        backend:             Backend ADC (any ADCBase subclass).
        backend_H:           Digital combiner weight applied to the backend.
        backend_code_offset: Integer added to backend's raw code before
                             combining (analogous to stage.code_offset).
        fs:                  Sample rate (Hz).
        clip_output:         Whether to clip the final DOUT to
                             [0, 2**n_bits - 1].
    """

    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 input_type: InputType = InputType.SINGLE,
                 stages: Optional[List[PipelineStage]] = None,
                 backend: Optional[ADCBase] = None,
                 backend_H: float = 1.0,
                 backend_code_offset: int = 0,
                 fs: float = 1.0,
                 noise_rms: float = 0.0,
                 offset: float = 0.0,
                 gain_error: float = 0.0,
                 t_jitter: float = 0.0,
                 clip_output: bool = True):
        super().__init__(n_bits, v_ref, input_type)

        if stages is None or not isinstance(stages, list) or len(stages) == 0:
            raise ValueError(
                "PipelinedADC requires at least one stage, got " +
                (repr(stages) if not isinstance(stages, list) else "empty list"))
        for i, s in enumerate(stages):
            if not isinstance(s, PipelineStage):
                raise TypeError(
                    f"stages[{i}] must be a PipelineStage instance, got {type(s).__name__}")

        if backend is None or not isinstance(backend, ADCBase):
            raise TypeError(
                f"backend must be an ADCBase instance, got {type(backend).__name__}")

        if not isinstance(backend_H, (int, float)):
            raise TypeError(f"backend_H must be a number, got {type(backend_H).__name__}")
        if backend_H <= 0:
            raise ValueError(f"backend_H must be positive, got {backend_H}")
        if not isinstance(backend_code_offset, int) or isinstance(backend_code_offset, bool):
            raise TypeError(
                f"backend_code_offset must be an integer, got {type(backend_code_offset).__name__}")

        if not isinstance(fs, (int, float)):
            raise TypeError(f"fs must be a number, got {type(fs).__name__}")
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")

        if noise_rms < 0:
            raise ValueError(f"noise_rms must be >= 0, got {noise_rms}")
        if t_jitter < 0:
            raise ValueError(f"t_jitter must be >= 0, got {t_jitter}")

        self.stages              = stages
        self.backend             = backend
        self.backend_H           = float(backend_H)
        self.backend_code_offset = backend_code_offset
        self.fs                  = float(fs)
        self.noise_rms           = float(noise_rms)
        self.offset              = float(offset)
        self.gain_error          = float(gain_error)
        self.t_jitter            = float(t_jitter)
        self.clip_output         = bool(clip_output)

    def _sample_input(self, analog_input) -> float:
        """Apply first-stage S&H non-idealities. Mirrors SARADC._sample_input."""
        if self.input_type == InputType.SINGLE:
            v = float(analog_input)
        else:
            v_pos, v_neg = analog_input
            v = float(v_pos) - float(v_neg)

        if self.gain_error:
            v = v * (1.0 + self.gain_error)
        if self.offset:
            v = v + self.offset
        if self.noise_rms:
            v = v + float(np.random.normal(0.0, self.noise_rms))
        if self.t_jitter and self._dvdt:
            v = v + self._dvdt * float(np.random.normal(0.0, self.t_jitter))
        return v

    def _convert_input(self, analog_input) -> int:
        """Cascade conversion + digital combiner."""
        v_sampled = self._sample_input(analog_input)

        DOUT = 0
        for stage in self.stages:
            raw_code, shifted_code, v_res = stage.convert_stage(v_sampled)
            DOUT = DOUT + DOUT * stage.H + shifted_code
            v_sampled = v_res

        # Backend
        raw_backend = int(self.backend.convert(float(v_sampled)))
        shifted_be  = raw_backend + self.backend_code_offset
        DOUT = DOUT + DOUT * self.backend_H + shifted_be

        if self.clip_output:
            lo, hi = 0, 2 ** self.n_bits - 1
            if DOUT < lo:
                DOUT = lo
            elif DOUT > hi:
                DOUT = hi

        return int(DOUT)

    def __repr__(self) -> str:
        return (f"PipelinedADC(n_bits={self.n_bits}, v_ref={self.v_ref}, "
                f"input_type={self.input_type.name}, stages={len(self.stages)}, "
                f"backend={type(self.backend).__name__}, fs={self.fs:.3e})")
```

Note the digital combiner line: `DOUT = DOUT + DOUT * stage.H + shifted_code`. This is the exact equivalent of the reference's `DOUT += DOUT*H + code`, written as an assignment for clarity. It evaluates to `DADC[0] * (1 + H[1]) + DADC[1]` for 2-element cascades with initial `DOUT=0` — numerically verified in Appendix A of the spec.

- [ ] **Step 9.4: Export both classes**

Open `pyDataconverter/architectures/__init__.py`. Append (or add in the appropriate alphabetical position):

```python
from .PipelinedADC import PipelineStage, PipelinedADC
```

- [ ] **Step 9.5: Run tests**

```bash
pytest tests/test_pipelined_adc.py -v
```

Expected: all tests pass (construction + cascade).

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: full suite passes (886+).

- [ ] **Step 9.6: Commit**

```bash
git add pyDataconverter/architectures/PipelinedADC.py pyDataconverter/architectures/__init__.py tests/test_pipelined_adc.py
git commit -m "feat(pipelined): add PipelinedADC(ADCBase) top-level class

Cascades N PipelineStage instances plus a required backend ADC. First-
stage S&H supports the usual SARADC-style non-idealities (noise_rms,
offset, gain_error, t_jitter). Digital combiner accumulates
DOUT = DOUT + DOUT*H + shifted_code per stage and once more with
backend_H, matching the adc_book reference's formula.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Bit-exact comparison tests against the vendored reference

**Files:**
- Create: `tests/test_pipelined_adc_vs_reference.py`

- [ ] **Step 10.1: Create the comparison harness**

Create `tests/test_pipelined_adc_vs_reference.py`:

```python
"""
Bit-exact comparison tests for PipelinedADC against the vendored adc_book reference.

Each test configures the same pipelined ADC in both implementations, runs a
dense linear sweep across the input range, and asserts that every output
code matches exactly. If any test fails, the assertion message names the
offending input, both output codes, and the config name — debugging starts
by running the stage-by-stage inspection on that single input.

See docs/superpowers/specs/2026-04-13-pipelined-adc-design.md §6 for the
harness design rationale.
"""

import io
import contextlib
import unittest
from typing import Callable, Dict, Any

import numpy as np

# Suppress the vendored reference's per-construct print spam. Although we
# already removed the four known print() calls from the vendored copy in
# Task 1, some constructors can still emit numpy warnings; redirect during
# import to stay noise-free.
with contextlib.redirect_stdout(io.StringIO()):
    from tests._reference.adc_book_pipelined import (
        PipelinedADC as RefPipelinedADC,
    )

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.architectures.PipelinedADC import (
    PipelineStage,
    PipelinedADC as NewPipelinedADC,
)
from pyDataconverter.components.residue_amplifier import ResidueAmplifier
from pyDataconverter.dataconverter import InputType, OutputType


# ------------------------------------------------------------
# Config dictionaries — one per scenario
# ------------------------------------------------------------

def _config_ideal_12bit() -> Dict[str, Any]:
    return dict(name="ideal_12bit",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=4.0, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=False,
                code_errors=None, extra_gain_error=0.0,
                fs=500e6)


def _config_stage0_dac_error() -> Dict[str, Any]:
    return dict(name="stage0_dac_error",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=4.0, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=False,
                code_errors=np.array(
                    [0.0, -0.2, 0.3, 0.05, -0.15, 0.0, 0.3, -0.3, 0.0]
                ) * 0.001,
                extra_gain_error=0.0,
                fs=500e6)


def _config_stage0_gain_error() -> Dict[str, Any]:
    return dict(name="stage0_gain_error",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=3.988, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=False,
                code_errors=None, extra_gain_error=0.0,
                fs=500e6)


def _config_metastability_canned() -> Dict[str, Any]:
    return dict(name="metastability_canned",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=4.0, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=True,
                tauC=30e-12, tauA=50e-12,
                code_errors=None, extra_gain_error=0.0,
                fs=500e6)


ALL_CONFIGS = [
    _config_ideal_12bit(),
    _config_stage0_dac_error(),
    _config_stage0_gain_error(),
    _config_metastability_canned(),
]


# ------------------------------------------------------------
# Builders
# ------------------------------------------------------------

def _build_reference(cfg: Dict[str, Any]) -> RefPipelinedADC:
    """Build the vendored reference ADC with the given configuration."""
    with contextlib.redirect_stdout(io.StringIO()):
        if cfg["time_response"]:
            ref = RefPipelinedADC(
                Nstages=2, B=cfg["n_bits"],
                N=[cfg["stage_n_comparators"], cfg["backend_n_comparators"]],
                FSR_ADC=[cfg["v_ref"], cfg["v_ref"]],
                FSR_DAC=[cfg["v_ref"], cfg["v_ref"]],
                G=[cfg["stage_gain"], cfg["backend_H"]],
                minADCcode=[cfg["stage_code_offset"], 0],
                timeResponse=[True, False],
                SampleRate=cfg["fs"],
                tau_comparator=[cfg["tauC"], 0],
                tau_amplifier=[cfg["tauA"], 0],
            )
        else:
            ref = RefPipelinedADC(
                Nstages=2, B=cfg["n_bits"],
                N=[cfg["stage_n_comparators"], cfg["backend_n_comparators"]],
                FSR_ADC=[cfg["v_ref"], cfg["v_ref"]],
                FSR_DAC=[cfg["v_ref"], cfg["v_ref"]],
                G=[cfg["stage_gain"], cfg["backend_H"]],
                minADCcode=[cfg["stage_code_offset"], 0],
            )
        if cfg.get("code_errors") is not None:
            # Reference subDAC has N+1 entries; pad the 9-entry pattern exactly
            # as the reference test does (it provides 9 entries for N=8).
            ref.stage[0].subDAC.add_error(np.asarray(cfg["code_errors"]))
    return ref


def _build_new(cfg: Dict[str, Any]) -> NewPipelinedADC:
    """Build the pyDataconverter PipelinedADC with the matching configuration."""
    n_comp = cfg["stage_n_comparators"]
    n_lev = cfg["stage_n_levels"]

    sub_adc_kwargs = {"n_bits": 3, "v_ref": cfg["v_ref"],
                      "input_type": InputType.SINGLE,
                      "n_comparators": n_comp}
    if cfg["time_response"]:
        sub_adc_kwargs["comparator_params"] = {
            "tau_regen": cfg["tauC"],
            "vc_threshold": 0.5,
        }
    sub_adc = FlashADC(**sub_adc_kwargs)

    sub_dac = SimpleDAC(
        n_bits=3, n_levels=n_lev, v_ref=cfg["v_ref"],
        output_type=OutputType.SINGLE,
        code_errors=cfg.get("code_errors"),
    )

    amp_kwargs = {"gain": cfg["stage_gain"]}
    if cfg["time_response"]:
        amp_kwargs["settling_tau"] = cfg["tauA"]
    else:
        amp_kwargs["settling_tau"] = 0.0
    residue_amp = ResidueAmplifier(**amp_kwargs)

    stage = PipelineStage(
        sub_adc=sub_adc, sub_dac=sub_dac, residue_amp=residue_amp,
        fs=cfg["fs"], code_offset=cfg["stage_code_offset"],
    )

    backend = FlashADC(n_bits=10, v_ref=cfg["v_ref"],
                       input_type=InputType.SINGLE,
                       n_comparators=cfg["backend_n_comparators"])

    return NewPipelinedADC(
        n_bits=cfg["n_bits"], v_ref=cfg["v_ref"],
        input_type=InputType.SINGLE,
        stages=[stage], backend=backend,
        backend_H=cfg["backend_H"], backend_code_offset=0,
        fs=cfg["fs"],
    )


# ------------------------------------------------------------
# The parameterised comparison test
# ------------------------------------------------------------

class TestPipelinedADCAgainstReference(unittest.TestCase):
    """Bit-exact sweep comparison for each canonical configuration."""

    def _run_sweep(self, cfg: Dict[str, Any]):
        ref = _build_reference(cfg)
        new = _build_new(cfg)
        v_sweep = np.linspace(-0.495, 0.495, 4001)
        for v in v_sweep:
            ref_out = int(ref.output(float(v)))
            new_out = int(new.convert(float(v)))
            if ref_out != new_out:
                self.fail(
                    f"[{cfg['name']}] mismatch at v={v:+.6f}: "
                    f"ref={ref_out}, new={new_out}"
                )

    def test_ideal_12bit(self):
        self._run_sweep(_config_ideal_12bit())

    def test_stage0_dac_error(self):
        self._run_sweep(_config_stage0_dac_error())

    def test_stage0_gain_error(self):
        self._run_sweep(_config_stage0_gain_error())

    def test_metastability_canned(self):
        self._run_sweep(_config_metastability_canned())
```

- [ ] **Step 10.2: Run the comparison tests**

```bash
pytest tests/test_pipelined_adc_vs_reference.py -v 2>&1 | tail -30
```

Expected: all four tests pass. If any mismatch appears, the assertion message names the exact input and both output codes.

**Debugging protocol for mismatches** (do NOT modify the vendored reference under any circumstances):

1. Reproduce the single failing input in isolation:
   ```python
   v = <the v value from the error message>
   print("ref:", ref.output(v), "stage0 DADC:", ref.stage[0].DADC, "backend DADC:", ref.stage[1].DADC)
   print("new:", new.convert(v))
   ```
2. Compare stage-0 intermediate values. If the sub-ADC raw code differs → FlashADC relaxation or threshold alignment bug. If the sub-DAC output differs → SimpleDAC n_levels or code_errors application bug. If the residue amp output differs → PipelineStage timing budget or initial_error sign bug.
3. For the metastability config, compare `ref.stage[0].tauC`, `ref.stage[0].tauA`, and `ref.stage[0].FS` to the corresponding fields on the new instance's components. Any mismatch there is a parameter-passing bug in `_build_new`.

If any mismatch resists diagnosis after 15 minutes, re-read `docs/superpowers/specs/2026-04-13-pipelined-adc-design.md` Appendix A and verify the algebraic derivation holds for the specific failing input.

- [ ] **Step 10.3: Run full suite to confirm no regressions**

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: 890+ tests pass.

- [ ] **Step 10.4: Commit**

```bash
git add tests/test_pipelined_adc_vs_reference.py
git commit -m "test(pipelined): bit-exact comparison against vendored adc_book reference

Four parameterised scenarios drive both the vendored reference and the
new pyDataconverter PipelinedADC with identical configurations and
assert every output code matches exactly across a 4001-point linear
sweep:

  1. ideal_12bit          - N=[8,1026] canonical config, no non-idealities
  2. stage0_dac_error     - explicit 9-element per-code sub-DAC error array
  3. stage0_gain_error    - G[0]=3.988 (0.3%) on the first-stage residue amp
  4. metastability_canned - tauC=30ps, tauA=50ps, fs=500MHz on stage 0

Test 4 is the binary correctness check for the entire
ResidueAmplifier + FlashADC.last_conversion_time +
FlashADC.last_metastable_sign plumbing. If any future refactor breaks
the metastability coupling, it fails here first.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Example file

**Files:**
- Create: `examples/pipelined_adc_example.py`

- [ ] **Step 11.1: Create the example**

Create `examples/pipelined_adc_example.py`:

```python
"""
Pipelined ADC Example
=====================

Constructs the canonical 12-bit pipelined ADC (3-bit thermometer first
stage + 1026-level backend flash), runs a coherent sine through it,
computes SNDR via pyDataconverter's metrics utilities, and plots the
output spectrum and the stage-0 residue transfer curve.

Run with:

    python examples/pipelined_adc_example.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.architectures.PipelinedADC import PipelineStage, PipelinedADC
from pyDataconverter.components.residue_amplifier import ResidueAmplifier
from pyDataconverter.dataconverter import InputType, OutputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine
from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics


def _compute_spectrum(codes: np.ndarray, fs: float):
    """FFT spectrum of a code sequence, in dB, independent of metric keys."""
    x = codes - np.mean(codes)
    spec = np.abs(np.fft.rfft(x))
    spec_db = 20 * np.log10(spec + 1e-30)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, spec_db


def build_canonical_pipelined_adc(fs: float = 500e6) -> PipelinedADC:
    """12-bit pipelined ADC: 3-bit stage + 1026-level backend."""
    sub_adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=8)
    sub_dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0,
                        output_type=OutputType.SINGLE)
    residue_amp = ResidueAmplifier(gain=4.0, settling_tau=0.0)
    stage = PipelineStage(sub_adc=sub_adc, sub_dac=sub_dac,
                          residue_amp=residue_amp, fs=fs, code_offset=-1)
    backend = FlashADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=1026)
    return PipelinedADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[stage], backend=backend,
                         backend_H=512, backend_code_offset=0, fs=fs)


def main():
    fs = 500e6
    n_fft = 2 ** 14
    adc = build_canonical_pipelined_adc(fs=fs)

    # Coherent sine at ~5 bins away from 0, below full-scale by 0.5 dB
    vin, _ = generate_coherent_sine(fs, n_fft, n_fin=127,
                                    amplitude=0.45, offset=0.0)
    codes = np.array([adc.convert(float(v)) for v in vin], dtype=float)

    metrics = calculate_adc_dynamic_metrics(time_data=codes, fs=fs)
    print("Pipelined ADC — canonical 12-bit configuration")
    print(f"  fs       = {fs/1e6:.1f} MHz")
    print(f"  n_fft    = {n_fft}")
    print(f"  SNR      = {metrics['SNR']:.2f} dB")
    print(f"  SNDR     = {metrics['SNDR']:.2f} dB")
    print(f"  SFDR     = {metrics['SFDR']:.2f} dB")
    print(f"  ENOB     = {metrics['ENOB']:.2f} bits")

    # Plot 1: output time series (first 256 samples)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(codes[:256], color="#4a9eff")
    axes[0].set_xlabel("sample")
    axes[0].set_ylabel("output code")
    axes[0].set_title("First 256 ADC output samples")
    axes[0].grid(True, linestyle=":")

    # Plot 2: output spectrum (computed locally; independent of metric dict keys)
    fft_freqs, fft_mags_db = _compute_spectrum(codes, fs)
    axes[1].plot(fft_freqs / 1e6, fft_mags_db, color="#f4a261")
    axes[1].set_xlabel("frequency (MHz)")
    axes[1].set_ylabel("magnitude (dB)")
    axes[1].set_title(
        f"Output spectrum, SNDR={metrics['SNDR']:.1f} dB, "
        f"ENOB={metrics['ENOB']:.2f} b")
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig("pipelined_adc_spectrum.png", dpi=150)
    print("\nWrote pipelined_adc_spectrum.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Run the example**

```bash
python examples/pipelined_adc_example.py
```

Expected output (approximate; SNDR depends on coherent frequency choice):

```
Pipelined ADC — canonical 12-bit configuration
  fs       = 500.0 MHz
  n_fft    = 16384
  SNR      = ~72.00 dB
  SNDR     = ~72.00 dB
  SFDR     = ~85.00 dB
  ENOB     = ~11.50 bits

Wrote pipelined_adc_spectrum.png
```

If SNDR is more than 2 dB below the 12-bit ideal (~74 dB), investigate: either the stage configuration is wrong or there is a subtle off-by-one in the digital combiner. The comparison tests from Task 10 should have caught structural errors; a small degradation here is usually a coherent-frequency mismatch.

If `calculate_adc_dynamic_metrics` does not expose `SNR`/`SNDR`/`SFDR`/`ENOB` keys under those exact names, check `pyDataconverter/utils/metrics/_dynamic.py` for the actual metric names and adjust the print statements accordingly. The spectrum plot is computed locally via `_compute_spectrum` and does not depend on any metric dict keys.

- [ ] **Step 11.3: Commit**

```bash
git add examples/pipelined_adc_example.py
git commit -m "example: add pipelined ADC demo script

Builds the canonical 12-bit configuration (3-bit thermometer first
stage + 1026-level backend flash), runs a coherent sine, reports SNR
/SNDR/SFDR/ENOB, and plots the output spectrum and time series.
Mirrors the style of examples/sar_adc_example.py.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Documentation and roadmap status update

**Files:**
- Modify: `docs/api_reference.md` (new sections)
- Modify: `todo/adc_architectures.md` (status update)

- [ ] **Step 12.1: Add API reference entries**

Open `docs/api_reference.md`. Find the section where `SARADC` is documented (search for `## SARADC` or similar). Immediately after that section, add:

````markdown
---

## PipelinedADC

*`pyDataconverter.architectures.PipelinedADC`*

N-stage pipelined ADC with a required backend. Cascades `PipelineStage` instances and combines their partial codes via the digital combiner. See `docs/superpowers/specs/2026-04-13-pipelined-adc-design.md` for the full design.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_bits` | int | — | Resolution in bits. |
| `v_ref` | float | 1.0 | Reference voltage (V). |
| `input_type` | InputType | SINGLE | Single-ended or differential. |
| `stages` | List[PipelineStage] | — | Non-empty list of early stages. |
| `backend` | ADCBase | — | Required backend ADC (any subclass). |
| `backend_H` | float | 1.0 | Digital combiner weight applied to the backend code. Typically equal to the last stage's expected residue gain. |
| `backend_code_offset` | int | 0 | Integer added to the backend's raw code before accumulation. |
| `fs` | float | 1.0 | Sample rate (Hz). Required positive. |
| `noise_rms` | float | 0.0 | First-stage S&H thermal / kT/C noise (V RMS). |
| `offset` | float | 0.0 | First-stage input-referred offset (V). |
| `gain_error` | float | 0.0 | First-stage fractional gain error. |
| `t_jitter` | float | 0.0 | First-stage aperture jitter (s RMS). Requires `dvdt != 0` to `convert()`. |
| `clip_output` | bool | True | Clip final DOUT to `[0, 2**n_bits - 1]`. |

**Example**

```python
from pyDataconverter.architectures.PipelinedADC import PipelineStage, PipelinedADC
from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.components.residue_amplifier import ResidueAmplifier
from pyDataconverter.dataconverter import InputType, OutputType

stage0 = PipelineStage(
    sub_adc=FlashADC(n_bits=3, v_ref=1.0, n_comparators=8),
    sub_dac=SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0, output_type=OutputType.SINGLE),
    residue_amp=ResidueAmplifier(gain=4.0),
    fs=500e6, code_offset=-1,
)
adc = PipelinedADC(
    n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
    stages=[stage0],
    backend=FlashADC(n_bits=10, v_ref=1.0, n_comparators=1026),
    backend_H=512, backend_code_offset=0, fs=500e6,
)
code = adc.convert(0.25)
```

---

## PipelineStage

*`pyDataconverter.architectures.PipelinedADC`*

One stage of a pipelined ADC. Composes a sub-ADC (`ADCBase`), a sub-DAC (`DACBase`), and a `ResidueAmplifier`. Not an ADC itself; used only as a building block for `PipelinedADC`.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sub_adc` | ADCBase | — | Any ADCBase instance. Code range must match `sub_dac.n_levels`. |
| `sub_dac` | DACBase | — | Any DACBase instance. Must produce single-ended output. |
| `residue_amp` | ResidueAmplifier | — | Residue amplifier. |
| `fs` | float | — | Sample rate (Hz), required positive. |
| `offset` | float | 0.0 | Input-referred offset added to `(v_in - v_dac)`. |
| `code_offset` | int | 0 | Integer added to `raw_code` before accumulation in the combiner. Equivalent to the `adc_book` reference's `minADCcode`. |
| `H` | float | `residue_amp.gain` | Digital combiner weight. Override to decouple from physical residue gain (trimming / calibration). |

**Method**

```python
raw_code, shifted_code, v_res = stage.convert_stage(v_sampled)
```

Returns the sub-ADC's raw output code, the offset-shifted code used by the combiner, and the amplified residue fed to the next stage.

---

## ResidueAmplifier

*`pyDataconverter.components.residue_amplifier`*

Closed-loop residue amplifier used in pipelined ADC stages. Core method:

```python
v_out = amp.amplify(target, initial_error, t_budget)
# returns target + initial_error * exp(-t_budget / settling_tau) + offset
```

Handles three IEEE 754 corner cases explicitly (`settling_tau=0`, `initial_error=0`, `t_budget=+inf`) to prevent NaN. Negative `t_budget` is deliberately not guarded to reproduce the `adc_book` reference's unguarded behaviour.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gain` | float | — | Closed-loop voltage gain. Nonzero; signed allowed. |
| `offset` | float | 0.0 | Output-referred DC offset. |
| `slew_rate` | float | +inf | Peak output slew rate (V/s). 0 or +inf disables slew limiting. |
| `settling_tau` | float | 0.0 | First-order settling time constant (s). 0 means instantaneous ideal amp. |
| `output_swing` | (float, float) | None | Optional `(v_min, v_max)` clipping bounds. |
````

Then find the `FlashADC` section and add a subsection after its existing content:

```markdown
**New in pipelined-ADC support:** `FlashADC` now accepts an optional `n_comparators: int` kwarg that decouples the comparator count from `2**n_bits - 1`. Even counts and arbitrary positive integers are allowed. When set to a non-standard value, output codes range over `[0, n_comparators]` and the reference ladder is built via `ArbitraryReference` with linear spacing instead of the default `ReferenceLadder`. Two new methods — `last_conversion_time()` and `last_metastable_sign()` — expose the comparator bank's metastability state for use by pipelined ADCs; both return 0 when every comparator has `tau_regen=0` (default).
```

Find the `SimpleDAC` section and add:

```markdown
**New in pipelined-ADC support:** `SimpleDAC` now accepts two optional kwargs. `n_levels: int` decouples the number of output codes from `2**n_bits` (inherited from the relaxed `DACBase`). `code_errors: np.ndarray` injects a per-code additive error pattern (length must equal `n_levels`), applied after the ideal transfer function and before gain/offset/noise.
```

Find the `DACBase` section (if one exists; otherwise skip) and add:

```markdown
**New in pipelined-ADC support:** `DACBase.__init__` now accepts an optional `n_levels: int` kwarg. When provided, it overrides the default `2**n_bits` code count and `lsb = v_ref / (n_levels - 1)`. `convert()` validates codes against `[0, n_levels - 1]`. Default behaviour (`n_levels = 2**n_bits`) is unchanged.
```

Find the `DifferentialComparator` section and add:

```markdown
**New in pipelined-ADC support:** `DifferentialComparator.__init__` now accepts `tau_regen: float` (default 0.0) and `vc_threshold: float` (default 0.5) kwargs, and exposes a `last_regen_time` read-only property. With `tau_regen > 0`, every `compare()` call caches `tau_regen * ln(vc_threshold / max(|v_diff|, 1e-30))` as the physical comparator regeneration time, used by `FlashADC.last_conversion_time()` to aggregate across a comparator bank.
```

- [ ] **Step 12.2: Update the ADC architecture roadmap status**

Open `todo/adc_architectures.md`. Find the line `## 1. Pipelined ADC  [HIGH] — STATUS: IN DESIGN (2026-04-13)` and replace the STATUS with:

```markdown
## 1. Pipelined ADC  [HIGH] — STATUS: PHASE 1 IMPLEMENTED (2026-04-13)

Shipped classes: `pyDataconverter/architectures/PipelinedADC.py` (`PipelineStage`, `PipelinedADC`), `pyDataconverter/components/residue_amplifier.py` (`ResidueAmplifier`). Extensions: relaxed `FlashADC.n_comparators`, `DACBase.n_levels`, `SimpleDAC.code_errors`, `DifferentialComparator.tau_regen`. Bit-exact against the vetted `adc_book` reference on four canonical configurations (see `tests/test_pipelined_adc_vs_reference.py`). Example: `examples/pipelined_adc_example.py`. Full design: `docs/superpowers/specs/2026-04-13-pipelined-adc-design.md`.

Phase 2 items (still open): explicit `SampleAndHold` component, residue-amp slew-rate limiting in comparison harness, `SARADC` metastability plumbing, additional non-idealities (reference noise, sub-DAC cap mismatch, stage crosstalk, 1/f noise), pipelined-specific characterisation helpers in `utils/characterization.py`.
```

- [ ] **Step 12.3: Run the full suite one more time**

```bash
pytest tests/ -q 2>&1 | tail -5
```

Expected: 890+ passed.

- [ ] **Step 12.4: Commit**

```bash
git add docs/api_reference.md todo/adc_architectures.md
git commit -m "docs: document PipelinedADC Phase 1 and update roadmap

New sections in docs/api_reference.md for PipelinedADC, PipelineStage,
and ResidueAmplifier, plus subsections on the new kwargs added to
FlashADC, SimpleDAC, DACBase, and DifferentialComparator.

Roadmap entry in todo/adc_architectures.md updated from IN DESIGN to
PHASE 1 IMPLEMENTED with pointers to the shipped classes, tests,
example, and spec.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

After Task 12, run the full acceptance check:

```bash
pytest tests/ -q 2>&1 | tail -5
pytest tests/test_pipelined_adc_vs_reference.py -v
python examples/pipelined_adc_example.py
```

Expected:
- Full suite: 890+ passed
- All four comparison configs: `PASSED`
- Example: prints SNDR within 0.5 dB of ~74 dB (12-bit ideal)

At that point every acceptance criterion from the spec §7.1 is satisfied:
1. ✅ Four comparison configs pass bit-exact on 4001-point sweep.
2. ✅ New code has ≥ 95% statement coverage (measurable via `pytest --cov=pyDataconverter tests/test_pipelined_adc.py tests/test_residue_amplifier.py`).
3. ✅ Existing 826 tests still pass.
4. ✅ `examples/pipelined_adc_example.py` runs end-to-end.
5. ✅ `docs/api_reference.md` updated.
6. ✅ `todo/adc_architectures.md` updated.

Phase 1 is done. Phase 2 items (explicit `SampleAndHold`, slew-rate comparison, SARADC metastability, additional non-idealities, characterisation helpers) are noted in the roadmap and become separate future work.
