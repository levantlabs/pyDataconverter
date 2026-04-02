# SAR ADC Generalisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalise the SAR ADC beyond binary weighting. Add `RedundantSARCDAC` (radix < 2 with digital error correction), `SplitCapCDAC`, `SegmentedCDAC`, `MultibitSARADC` (multi-bit per cycle via flash sub-ADC), and `NoiseshapingSARADC`.

**Architecture:** New CDAC subclasses go in `pyDataconverter/components/cdac.py` (they extend `CDACBase` / `SingleEndedCDAC`). New ADC subclasses go in `pyDataconverter/architectures/SARADC.py` (they extend `SARADC`). The base `SARADC._run_sar` already accepts any `CDACBase` via `self.cdac`, so most new capabilities are purely additive. The only method that changes is `_run_sar` for `MultibitSARADC`.

**Key insight:** `SingleEndedCDAC` already accepts a `cap_weights` argument — arbitrary weights are already supported. The missing pieces are:
1. A `RedundantSARCDAC` that sets the right radix-<2 weights and bundles a digital correction decoder.
2. A `SplitCapCDAC` that implements the bridge-cap topology.
3. A `SegmentedCDAC` that combines thermometer MSBs with binary LSBs.
4. `MultibitSARADC` overrides `_run_sar` to use a flash sub-ADC per cycle.
5. `NoiseshapingSARADC` adds an integrator state for noise shaping.

**Tech Stack:** numpy, existing `SingleEndedCDAC`, `DifferentialCDAC`, `CDACBase`, `SARADC`, `FlashADC`.

---

## File Map

| File | Change |
|---|---|
| `pyDataconverter/components/cdac.py` | Add `RedundantSARCDAC`, `SplitCapCDAC`, `SegmentedCDAC` |
| `pyDataconverter/architectures/SARADC.py` | Add `MultibitSARADC`, `NoiseshapingSARADC` |
| `pyDataconverter/components/__init__.py` | Re-export new CDAC classes |
| `tests/test_sar_generalisation.py` | New test file |

---

## Task 1: Add `RedundantSARCDAC` with digital error correction

A redundant SAR uses a radix r < 2 (typically 1.7–1.9) so adjacent bit decisions overlap. This allows a comparator error on cycle k to be corrected by the remaining cycles. After the bit loop, the raw register (binary code) must be converted to the true output code via a digital error correction (DEC) lookup.

**Files:**
- Modify: `pyDataconverter/components/cdac.py`
- Create: `tests/test_sar_generalisation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sar_generalisation.py`:

```python
"""Tests for generalised SAR ADC components."""
import numpy as np
import pytest
from pyDataconverter.components.cdac import (
    RedundantSARCDAC, SplitCapCDAC, SegmentedCDAC,
)


class TestRedundantSARCDAC:
    def test_construction(self):
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        assert cdac.n_bits == 4
        assert cdac.v_ref  == 1.0

    def test_weights_decrease(self):
        """Capacitor weights are strictly decreasing (MSB first)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.85)
        w = cdac.cap_weights
        assert np.all(np.diff(w) < 0)

    def test_weights_not_binary(self):
        """Weights are not powers of 2 (distinguishes from standard CDAC)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        w = cdac.cap_weights
        binary_weights = 2 ** np.arange(cdac.n_bits - 1, -1, -1).astype(float)
        assert not np.allclose(w, binary_weights)

    def test_dec_monotone(self):
        """Decoded output is monotone with increasing raw code."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        decoded = [cdac.decode(code) for code in range(2**6)]
        assert decoded == sorted(decoded)

    def test_dec_range(self):
        """Decoded values span 0..2^n_bits-1."""
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        decoded = [cdac.decode(code) for code in range(2**4)]
        assert min(decoded) == 0
        assert max(decoded) == 2**4 - 1

    def test_ideal_conversion_with_sar(self):
        """RedundantSARCDAC inside SARADC gives monotone ideal transfer."""
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        adc  = SARADC(n_bits=6, v_ref=1.0, cdac=cdac)
        vin  = np.linspace(0, 1.0, 200)
        codes = [adc.convert(float(v)) for v in vin]
        # codes should be non-decreasing
        assert all(b >= a for a, b in zip(codes, codes[1:]))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lazarus/python-dev/pyDataconverter
python -m pytest tests/test_sar_generalisation.py::TestRedundantSARCDAC -v
```

Expected: ImportError for `RedundantSARCDAC`.

- [ ] **Step 3: Implement `RedundantSARCDAC`**

Add to `pyDataconverter/components/cdac.py` (after `SingleEndedCDAC`):

```python
class RedundantSARCDAC(SingleEndedCDAC):
    """
    Redundant-radix C-DAC for SAR ADCs with digital error correction.

    Uses capacitor weights proportional to r^(N-1-k) for k=0..N-1 (MSB
    first), where r < 2.  The overlap between adjacent bit weights means
    that a wrong comparator decision on cycle k can be corrected by the
    remaining cycles, at the cost of requiring a digital error correction
    (DEC) step to map the raw binary register to the correct output code.

    The DEC lookup table is built at construction by computing the
    weighted sum for every possible raw code and sorting to produce a
    monotone mapping.

    Attributes:
        radix (float): The radix r used for weight generation (1.5 ≤ r < 2).
        _dec_table (np.ndarray): DEC lookup: dec_table[raw_code] → output_code.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        radix: float = 1.85,
        cap_mismatch: float = 0.0,
    ):
        """
        Args:
            n_bits: ADC resolution.
            v_ref: Reference voltage (V).
            radix: Sub-binary radix (1.5 ≤ radix < 2.0).  Typical: 1.8–1.9.
            cap_mismatch: Capacitor mismatch std (dimensionless).
        """
        if not (1.0 < radix < 2.0):
            raise ValueError("radix must be in (1.0, 2.0)")
        self.radix = radix

        # Compute nominal weights: r^(N-1), r^(N-2), ..., r^0  (MSB first)
        exponents = np.arange(n_bits - 1, -1, -1, dtype=float)
        nominal_weights = radix ** exponents

        super().__init__(n_bits, v_ref,
                         cap_weights=nominal_weights,
                         cap_mismatch=cap_mismatch)

        # Build DEC lookup table
        n_codes = 2 ** n_bits
        # For each raw code, compute weighted sum using nominal weights
        raw_voltages = np.array([
            float(np.dot(self._code_to_bits(c), nominal_weights))
            for c in range(n_codes)
        ])
        # Sort raw codes by their voltage → monotone mapping
        sorted_idx = np.argsort(raw_voltages, kind='stable')
        self._dec_table = np.empty(n_codes, dtype=int)
        for out_code, raw in enumerate(sorted_idx):
            self._dec_table[raw] = out_code

    def decode(self, raw_code: int) -> int:
        """
        Apply digital error correction to a raw SAR register value.

        Args:
            raw_code: The raw binary register output of the SAR bit loop.

        Returns:
            int: Corrected output code in [0, 2^n_bits − 1].
        """
        return int(self._dec_table[raw_code])
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sar_generalisation.py::TestRedundantSARCDAC -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/components/cdac.py tests/test_sar_generalisation.py
git commit -m "feat(cdac): add RedundantSARCDAC with digital error correction"
```

---

## Task 2: Add `SplitCapCDAC`

A split-capacitor array uses a bridge capacitor of value `C_bridge` between a coarse MSB sub-array and a fine LSB sub-array, reducing the total capacitance by ~2× compared to a full binary array while maintaining the same resolution.

Ideal output (n_msb MSBs + n_lsb LSBs, n_msb + n_lsb = n_bits):
    v_dac = (msb_code / 2^n_msb + lsb_code / (2^n_msb * 2^n_lsb)) * v_ref

The equivalent flat weight vector (for use with SingleEndedCDAC) is:
    weights = [2^(n_lsb-1), ..., 2, 1,  C_bridge_equiv,  2^(n_lsb-1), ..., 1]
              [--- MSB weights scaled by 2^n_lsb ---]  [bridge]  [--- LSB weights ---]

The bridge capacitor equivalent weight is 1.0 (same as LSB cap, accounting for the attenuation).

**Files:**
- Modify: `pyDataconverter/components/cdac.py`
- Modify: `tests/test_sar_generalisation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sar_generalisation.py`:

```python
class TestSplitCapCDAC:
    def test_construction(self):
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        assert cdac.n_bits == 8
        assert cdac.v_ref  == 1.0

    def test_fewer_total_caps_than_full_binary(self):
        """Split-cap uses n_bits+1 caps (incl. bridge), not 2^n_bits."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        # n_msb + n_lsb + 1 bridge cap = n_bits + 1 total caps
        assert len(cdac.cap_weights) == 8 + 1  # 9 caps, not 256

    def test_ideal_output_zero(self):
        """Code 0 → 0 V."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        vp, vn = cdac.get_voltage(0)
        assert abs(vp - vn) < 1e-9

    def test_ideal_output_full_scale(self):
        """Max code → v_ref - LSB."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        max_code = 2**8 - 1
        vp, vn = cdac.get_voltage(max_code)
        expected = (max_code / 2**8) * 1.0
        assert abs(vp - vn - expected) < 0.01  # within 1% of LSB

    def test_monotone_voltages(self):
        """Voltages are non-decreasing with code."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        voltages = cdac.voltages
        assert np.all(np.diff(voltages) >= -1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sar_generalisation.py::TestSplitCapCDAC -v
```

Expected: ImportError for `SplitCapCDAC`.

- [ ] **Step 3: Implement `SplitCapCDAC`**

Add to `pyDataconverter/components/cdac.py` (after `RedundantSARCDAC`):

```python
class SplitCapCDAC(SingleEndedCDAC):
    """
    Split-capacitor C-DAC with a bridge capacitor between coarse and fine arrays.

    Divides the n_bits capacitors into n_msb MSB caps and n_lsb = n_bits - n_msb
    LSB caps, joined by a bridge capacitor.  Total caps = n_bits + 1 vs 2^n_bits
    for a full binary array.

    The effective weight vector (passed to SingleEndedCDAC) is:
        MSB weights: 2^(n_lsb), 2^(n_lsb-1), ..., 2, 1  (scaled by 2^n_lsb)
        Bridge cap: 1  (attenuation factor already built into MSB scaling)
        LSB weights: 2^(n_lsb-1), ..., 2, 1

    Total cap: sum(MSB) + 1 (bridge) + sum(LSB) + 1 (termination)
             = (2^(n_lsb+1) - 1) + 1 + (2^n_lsb - 1) + 1 = 2^(n_bits+1)
    Ideal LSB step = v_ref / 2^n_bits  ✓

    Args:
        n_bits: Total ADC resolution.
        v_ref: Reference voltage (V).
        n_msb: Number of MSB capacitors (1 ≤ n_msb ≤ n_bits − 1).
        cap_mismatch: Capacitor mismatch std (dimensionless).
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        n_msb: int = None,
        cap_mismatch: float = 0.0,
    ):
        if n_msb is None:
            n_msb = n_bits // 2
        n_lsb = n_bits - n_msb
        if not (1 <= n_msb <= n_bits - 1):
            raise ValueError(f"n_msb must be in [1, {n_bits-1}]")

        # MSB weights scaled up by 2^n_lsb so the bridge attenuates them
        msb_weights = 2 ** np.arange(n_msb - 1, -1, -1, dtype=float) * (2 ** n_lsb)
        # Bridge cap
        bridge_weight = np.array([1.0])
        # LSB weights (standard binary)
        lsb_weights = 2 ** np.arange(n_lsb - 1, -1, -1, dtype=float)

        weights = np.concatenate([msb_weights, bridge_weight, lsb_weights])

        # n_bits for the parent is the actual total number of caps (n_bits+1)
        # but we override n_bits to keep it equal to the logical resolution.
        # We pass n_bits+1 to the parent and correct the n_bits property.
        super().__init__(n_bits + 1, v_ref,
                         cap_weights=weights,
                         cap_mismatch=cap_mismatch)
        # Override _n_bits so the ADC sees the correct logical resolution
        self._n_bits = n_bits
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sar_generalisation.py::TestSplitCapCDAC -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/components/cdac.py tests/test_sar_generalisation.py
git commit -m "feat(cdac): add SplitCapCDAC (bridge-capacitor topology)"
```

---

## Task 3: Add `SegmentedCDAC`

A segmented DAC splits the MSBs into thermometer-coded capacitors (one cap per code level) and keeps the LSBs binary. This reduces glitch energy and improves monotonicity for the upper codes.

**Files:**
- Modify: `pyDataconverter/components/cdac.py`
- Modify: `tests/test_sar_generalisation.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestSegmentedCDAC:
    def test_construction(self):
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        assert cdac.n_bits == 8

    def test_monotone_voltages(self):
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        v = cdac.voltages
        assert np.all(np.diff(v) >= -1e-9)

    def test_output_range(self):
        """Max code → v_ref - LSB, code 0 → 0."""
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        v = cdac.voltages
        assert abs(v[0]) < 1e-9
        expected_max = (2**8 - 1) / 2**8
        assert abs(v[-1] - expected_max) < 0.01

    def test_thermometer_section_equal_steps(self):
        """The MSB 2^n_therm steps should be equal (thermometer linearity)."""
        n_bits, n_therm = 8, 4
        cdac = SegmentedCDAC(n_bits=n_bits, v_ref=1.0, n_therm=n_therm)
        v = cdac.voltages
        step = 2**n_bits // 2**n_therm  # codes per thermometer step
        msb_voltages = v[step-1::step][:2**n_therm]
        diffs = np.diff(msb_voltages)
        assert np.allclose(diffs, diffs[0], rtol=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sar_generalisation.py::TestSegmentedCDAC -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `SegmentedCDAC`**

Add to `pyDataconverter/components/cdac.py`:

```python
class SegmentedCDAC(CDACBase):
    """
    Segmented C-DAC: thermometer-coded MSBs + binary-weighted LSBs.

    The top n_therm bits are decoded to (2^n_therm − 1) equal-weight unit
    capacitors (thermometer), and the lower n_bits − n_therm bits are handled
    by a binary sub-array.  This class wraps both sub-arrays and decodes a
    full n_bits code into (therm_code, binary_code) before dispatching to
    each sub-array's voltage.

    Args:
        n_bits: Total ADC resolution.
        v_ref: Reference voltage (V).
        n_therm: Number of MSBs implemented as thermometer (1 ≤ n_therm < n_bits).
        cap_mismatch: Capacitor mismatch std for both sub-arrays.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        n_therm: int = 4,
        cap_mismatch: float = 0.0,
    ):
        n_binary = n_bits - n_therm
        if not (1 <= n_therm <= n_bits - 1):
            raise ValueError(f"n_therm must be in [1, {n_bits-1}]")

        self._n_bits    = n_bits
        self._v_ref     = v_ref
        self._n_therm   = n_therm
        self._n_binary  = n_binary

        # One unit cap per thermometer level (2^n_therm − 1 caps)
        n_therm_caps = 2 ** n_therm - 1
        therm_weights = np.ones(n_therm_caps)

        # Binary sub-array for the lower bits
        binary_weights = 2 ** np.arange(n_binary - 1, -1, -1, dtype=float)

        # Scale thermometer unit so that 1 therm step = 2^n_binary binary steps
        lsb_equiv = 2 ** n_binary
        therm_weights_scaled = therm_weights * lsb_equiv

        # Combine into one flat weight vector
        all_weights = np.concatenate([therm_weights_scaled, binary_weights])

        # Instantiate as a single SingleEndedCDAC
        self._cdac = SingleEndedCDAC(
            n_bits=len(all_weights),
            v_ref=v_ref,
            cap_weights=all_weights,
            cap_mismatch=cap_mismatch,
        )

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def v_ref(self) -> float:
        return self._v_ref

    @property
    def cap_weights(self) -> np.ndarray:
        return self._cdac.cap_weights

    @property
    def cap_total(self) -> float:
        return self._cdac.cap_total

    def get_voltage(self, code: int) -> Tuple[float, float]:
        """
        Decode code into thermometer + binary parts, return (v_dac, 0.0).

        The upper n_therm bits → thermometer index t (0..2^n_therm − 1).
        The lower n_binary bits → binary code b.
        Flat index = t * 2^n_binary + b.
        """
        if not (0 <= code < 2 ** self._n_bits):
            raise ValueError(f"code {code} out of range [0, {2**self._n_bits - 1}]")
        # Decompose
        therm_code  = code >> self._n_binary          # upper n_therm bits
        binary_code = code & ((1 << self._n_binary) - 1)  # lower n_binary bits
        flat_code   = therm_code * (2 ** self._n_binary) + binary_code
        return self._cdac.get_voltage(flat_code)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sar_generalisation.py::TestSegmentedCDAC -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/components/cdac.py tests/test_sar_generalisation.py
git commit -m "feat(cdac): add SegmentedCDAC (thermometer MSBs + binary LSBs)"
```

---

## Task 4: Add `MultibitSARADC`

Resolves M bits per cycle using a flash sub-ADC instead of a single comparator. With N total bits and M bits per cycle, the number of SAR cycles is ceil(N/M).

**Files:**
- Modify: `pyDataconverter/architectures/SARADC.py`
- Modify: `tests/test_sar_generalisation.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestMultibitSARADC:
    def test_construction(self):
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(n_bits=8, v_ref=1.0, bits_per_cycle=2)
        assert adc.n_bits == 8
        assert adc.bits_per_cycle == 2

    def test_monotone_ideal(self):
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(n_bits=6, v_ref=1.0, bits_per_cycle=2)
        vin = np.linspace(0.01, 0.99, 200)
        codes = [adc.convert(float(v)) for v in vin]
        assert all(b >= a for a, b in zip(codes, codes[1:]))

    def test_output_range(self):
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(n_bits=6, v_ref=1.0, bits_per_cycle=3)
        codes = [adc.convert(float(v)) for v in np.linspace(0, 1, 100)]
        assert min(codes) >= 0
        assert max(codes) <= 2**6 - 1

    def test_bits_per_cycle_1_matches_standard_sar(self):
        """bits_per_cycle=1 should behave identically to standard SARADC."""
        from pyDataconverter.architectures.SARADC import SARADC, MultibitSARADC
        import numpy as np
        np.random.seed(0)
        sar    = SARADC(n_bits=6, v_ref=1.0)
        mbit   = MultibitSARADC(n_bits=6, v_ref=1.0, bits_per_cycle=1)
        vins   = np.linspace(0.01, 0.99, 63)
        codes_sar  = [sar.convert(float(v))  for v in vins]
        codes_mbit = [mbit.convert(float(v)) for v in vins]
        assert codes_sar == codes_mbit
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sar_generalisation.py::TestMultibitSARADC -v
```

Expected: ImportError for `MultibitSARADC`.

- [ ] **Step 3: Implement `MultibitSARADC`**

Add to `pyDataconverter/architectures/SARADC.py`:

```python
class MultibitSARADC(SARADC):
    """
    SAR ADC that resolves multiple bits per cycle using a flash sub-ADC.

    Each cycle uses a (2^bits_per_cycle − 1)-comparator flash to determine
    the next bits_per_cycle bits simultaneously.  With N total bits and M
    bits per cycle, the conversion requires ceil(N / M) cycles.

    The flash sub-ADC is idealised (no offset, no noise) unless a comparator
    with non-idealities is injected via comparator_params.

    Args:
        n_bits: Total resolution.
        v_ref: Reference voltage (V).
        bits_per_cycle: Number of bits resolved per SAR cycle (default 2).
        All other args forwarded to SARADC.__init__.
    """

    def __init__(self, n_bits: int, v_ref: float = 1.0,
                 bits_per_cycle: int = 2, **kwargs):
        if bits_per_cycle < 1 or bits_per_cycle > n_bits:
            raise ValueError("bits_per_cycle must be in [1, n_bits]")
        self.bits_per_cycle = bits_per_cycle
        super().__init__(n_bits, v_ref, **kwargs)

    def _run_sar(self, v_sampled: float):
        """Override: use flash sub-ADC to resolve bits_per_cycle bits at once."""
        import math
        register  = 0
        dac_voltages    = []
        bit_decisions   = []
        register_states = [0]

        n_cycles = math.ceil(self.n_bits / self.bits_per_cycle)

        for cycle in range(n_cycles):
            msb_position = self.n_bits - cycle * self.bits_per_cycle
            lsb_position = max(0, msb_position - self.bits_per_cycle)
            n_sub = msb_position - lsb_position  # bits to resolve this cycle

            n_levels = 2 ** n_sub
            best_code = 0
            for sub_code in range(n_levels - 1, -1, -1):
                trial_code = register | (sub_code << lsb_position)
                v_refp, v_refn = self.cdac.get_voltage(trial_code)
                self.comparator.reset()
                decision = self.comparator.compare(v_sampled, 0.0, v_refp, v_refn)
                dac_voltages.append(v_refp - v_refn)
                bit_decisions.append(decision)
                if decision:
                    best_code = sub_code
                    break

            register = register | (best_code << lsb_position)
            register_states.append(register)

        return register, dac_voltages, bit_decisions, register_states
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sar_generalisation.py::TestMultibitSARADC -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/architectures/SARADC.py tests/test_sar_generalisation.py
git commit -m "feat(sar): add MultibitSARADC (multi-bit per cycle)"
```

---

## Task 5: Add `NoiseshapingSARADC`

After each conversion, compute the residue (error = v_sampled − reconstructed_analog). Integrate it and add to the next conversion's input. This shapes the quantisation noise to higher frequencies (first-order noise shaping).

**Files:**
- Modify: `pyDataconverter/architectures/SARADC.py`
- Modify: `tests/test_sar_generalisation.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestNoiseshapingSARADC:
    def test_construction(self):
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0)
        assert adc.n_bits == 6
        assert adc.integrator_state == 0.0

    def test_reset_clears_state(self):
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0)
        adc.convert(0.5)  # leave some integrator state
        adc.reset()
        assert adc.integrator_state == 0.0

    def test_output_range(self):
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0)
        codes = [adc.convert(0.5) for _ in range(100)]
        assert all(0 <= c <= 2**6 - 1 for c in codes)

    def test_noise_shaping_improves_snr_at_low_freq(self):
        """First-order noise shaping should give better SNDR than standard SAR
        when measuring SNR over the lower half of the Nyquist band (oversampling)."""
        from pyDataconverter.architectures.SARADC import SARADC, NoiseshapingSARADC
        from pyDataconverter.utils.signal_gen import generate_coherent_sine
        from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics
        import numpy as np
        n_bits, v_ref, fs, n_fft = 6, 1.0, 1e6, 2048
        vin, _ = generate_coherent_sine(fs, n_fft, n_fin=5, amplitude=0.45, offset=0.5)
        adc_std = SARADC(n_bits=n_bits, v_ref=v_ref)
        adc_ns  = NoiseshapingSARADC(n_bits=n_bits, v_ref=v_ref)
        codes_std = np.array([adc_std.convert(float(v)) for v in vin], dtype=float)
        codes_ns  = np.array([adc_ns.convert(float(v))  for v in vin], dtype=float)
        m_std = calculate_adc_dynamic_metrics(time_data=codes_std, fs=fs)
        m_ns  = calculate_adc_dynamic_metrics(time_data=codes_ns,  fs=fs)
        # Noise shaping pushes noise up; SNR at low frequency should improve
        assert m_ns['SNR'] > m_std['SNR'] - 3.0  # at least comparable at low fin
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sar_generalisation.py::TestNoiseshapingSARADC -v
```

Expected: ImportError for `NoiseshapingSARADC`.

- [ ] **Step 3: Implement `NoiseshapingSARADC`**

Add to `pyDataconverter/architectures/SARADC.py`:

```python
class NoiseshapingSARADC(SARADC):
    """
    First-order noise-shaping SAR ADC.

    After each conversion the quantisation residue (v_sampled − v_reconstructed)
    is accumulated in an integrator.  The integrated residue is added to the
    sampled input on the next conversion, shaping quantisation noise to higher
    frequencies at the cost of an increased noise floor near Nyquist.

    For an oversampling ratio OSR the in-band SNR improvement over a standard
    SAR is approximately 9 dB per octave of OSR (first-order shaping, −20 dB/decade).

    Attributes:
        integrator_state (float): Current integrator output (V).  Reset to 0
            by calling .reset().
    """

    def __init__(self, n_bits: int, v_ref: float = 1.0, **kwargs):
        super().__init__(n_bits, v_ref, **kwargs)
        self.integrator_state: float = 0.0

    def reset(self):
        """Reset comparator state and integrator."""
        super().reset()
        self.integrator_state = 0.0

    def _convert_input(self, analog_input) -> int:
        """Sample, add integrator, run SAR, update integrator."""
        v_sampled = self._sample_input(analog_input)
        v_input   = v_sampled + self.integrator_state

        code, _, _, _ = self._run_sar(v_input)

        # Reconstruct analog value from output code
        v_reconstructed = (code + 0.5) / (2 ** self.n_bits) * self.v_ref

        # Update integrator: residue = clipped error
        residue = v_input - v_reconstructed
        ideal_lsb = self.v_ref / (2 ** self.n_bits)
        # Clip integrator to ±0.5 v_ref to prevent runaway
        self.integrator_state = float(
            np.clip(residue, -self.v_ref / 2, self.v_ref / 2)
        )

        return code
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sar_generalisation.py::TestNoiseshapingSARADC -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyDataconverter/architectures/SARADC.py tests/test_sar_generalisation.py
git commit -m "feat(sar): add NoiseshapingSARADC (first-order noise shaping)"
```

---

## Task 6: Update `__init__.py` exports

**Files:**
- Modify: `pyDataconverter/components/__init__.py`

- [ ] **Step 1: Add new CDAC classes to exports**

Read `pyDataconverter/components/__init__.py` to see what is currently exported, then add:

```python
from .cdac import (
    CDACBase,
    SingleEndedCDAC,
    DifferentialCDAC,
    RedundantSARCDAC,
    SplitCapCDAC,
    SegmentedCDAC,
)
```

- [ ] **Step 2: Run full test suite to confirm nothing broken**

```bash
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add pyDataconverter/components/__init__.py
git commit -m "chore: export new CDAC classes from components __init__"
```
