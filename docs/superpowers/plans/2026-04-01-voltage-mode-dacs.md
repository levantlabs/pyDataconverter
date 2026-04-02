# Voltage-Mode DACs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resistor network solver, `ResistorStringDAC`, `R2RDAC`, and `SegmentedResistorDAC` to pyDataconverter. All three produce voltage outputs directly (voltage-mode), contrasting with the current-steering and CDAC models.

**Architecture:**
- `pyDataconverter/utils/nodal_solver.py` — Modified Nodal Analysis (MNA) solver for general resistor networks. Used internally by all three DAC classes.
- `pyDataconverter/architectures/ResistorStringDAC.py` — 2^N equal resistors in series; switches tap the output node.
- `pyDataconverter/architectures/R2RDAC.py` — R-2R ladder; each bit switches between V_ref and GND.
- `pyDataconverter/architectures/SegmentedResistorDAC.py` — Thermometer-coded coarse resistor string + binary fine R-2R.

All three inherit from `DACBase` (found in `pyDataconverter/dataconverter.py`). Check what `DACBase` / `SimpleDAC` look like before implementing.

**Tech Stack:** numpy (linear algebra for MNA), existing `DACBase`.

---

## File Map

| File | Change |
|---|---|
| `pyDataconverter/utils/nodal_solver.py` | New: MNA resistor network solver |
| `pyDataconverter/architectures/ResistorStringDAC.py` | New |
| `pyDataconverter/architectures/R2RDAC.py` | New |
| `pyDataconverter/architectures/SegmentedResistorDAC.py` | New |
| `pyDataconverter/architectures/__init__.py` | Re-export new classes |
| `tests/test_voltage_mode_dacs.py` | New test file |

---

## Preparation: Read `DACBase` and `SimpleDAC` before implementing

Read `pyDataconverter/dataconverter.py` to understand `DACBase` (its `__init__` signature, abstract methods, and `convert` interface) and read `pyDataconverter/architectures/SimpleDAC.py` to see a concrete implementation before writing any DAC code.

---

## Task 1: Add `nodal_solver.py`

The MNA solver builds the conductance matrix G and current vector I for a resistor network with fixed voltage sources, then solves G·V = I for all node voltages.

**Files:**
- Create: `pyDataconverter/utils/nodal_solver.py`
- Create: `tests/test_voltage_mode_dacs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voltage_mode_dacs.py`:

```python
"""Tests for voltage-mode DAC models and the nodal solver."""
import numpy as np
import pytest


class TestNodalSolver:
    def test_voltage_divider(self):
        """Two equal resistors between V_ref and GND gives V_ref/2 at midpoint."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        # Nodes: 0=GND (fixed 0V), 1=midpoint (unknown), 2=V_ref (fixed 1V)
        # R between node 0 and 1, R between node 1 and 2
        nodes = 3
        resistors = [(0, 1, 1e3), (1, 2, 1e3)]  # (node_a, node_b, R_ohms)
        fixed = {0: 0.0, 2: 1.0}                 # node: voltage
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert abs(voltages[1] - 0.5) < 1e-9

    def test_unequal_divider(self):
        """1kΩ / 3kΩ divider gives 0.75 V at midpoint."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        nodes = 3
        resistors = [(0, 1, 3e3), (1, 2, 1e3)]
        fixed = {0: 0.0, 2: 1.0}
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert abs(voltages[1] - 0.75) < 1e-9

    def test_three_node_network(self):
        """Star network: three 1kΩ resistors meeting at a centre node."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        # Nodes: 0=fixed 0V, 1=fixed 1V, 2=fixed 0.5V, 3=centre (unknown)
        nodes = 4
        resistors = [(0, 3, 1e3), (1, 3, 1e3), (2, 3, 1e3)]
        fixed = {0: 0.0, 1: 1.0, 2: 0.5}
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert abs(voltages[3] - 0.5) < 1e-9

    def test_returns_all_node_voltages(self):
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        nodes = 4
        resistors = [(0, 1, 1e3), (1, 2, 1e3), (2, 3, 1e3)]
        fixed = {0: 0.0, 3: 1.0}
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert len(voltages) == nodes
        assert abs(voltages[1] - 1/3) < 1e-9
        assert abs(voltages[2] - 2/3) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lazarus/python-dev/pyDataconverter
python -m pytest tests/test_voltage_mode_dacs.py::TestNodalSolver -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `nodal_solver.py`**

Create `pyDataconverter/utils/nodal_solver.py`:

```python
"""
Nodal Solver
============

Modified Nodal Analysis (MNA) solver for resistor networks.

Solves for all unknown node voltages given a set of resistors and
fixed (source) node voltages, using numpy's linear algebra solver.
"""

import numpy as np
from typing import Dict, List, Tuple


def solve_resistor_network(
        n_nodes: int,
        resistors: List[Tuple[int, int, float]],
        fixed_voltages: Dict[int, float],
) -> np.ndarray:
    """
    Solve for all node voltages in a resistor network.

    Uses Modified Nodal Analysis: builds the conductance matrix G and
    solves G · V = I, with fixed-voltage nodes handled via row substitution.

    Args:
        n_nodes: Total number of nodes (0-indexed).
        resistors: List of (node_a, node_b, resistance_ohms) tuples.
            Each resistor connects node_a to node_b with the given resistance.
        fixed_voltages: Dict mapping node index → fixed voltage (V).
            Must include at least one node (e.g. ground node = 0 V).

    Returns:
        np.ndarray of shape (n_nodes,): solved voltage at every node.
        Fixed-voltage nodes have their prescribed values; unknown nodes
        are solved by the linear system.

    Raises:
        ValueError: If the system is singular (disconnected network or
            insufficient fixed nodes).
    """
    G = np.zeros((n_nodes, n_nodes))
    I = np.zeros(n_nodes)

    # Build conductance matrix
    for na, nb, r in resistors:
        g = 1.0 / r
        G[na, na] += g
        G[nb, nb] += g
        G[na, nb] -= g
        G[nb, na] -= g

    # Apply fixed voltages via row substitution
    for node, voltage in fixed_voltages.items():
        G[node, :] = 0.0
        G[node, node] = 1.0
        I[node] = voltage

    try:
        voltages = np.linalg.solve(G, I)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Resistor network is singular — check that all nodes are "
            "reachable and at least one voltage is fixed."
        ) from exc

    return voltages
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_voltage_mode_dacs.py::TestNodalSolver -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/utils/nodal_solver.py tests/test_voltage_mode_dacs.py
git commit -m "feat: add nodal_solver (MNA resistor network solver)"
```

---

## Task 2: Add `ResistorStringDAC`

2^N resistors in series between V_ref and GND. For code k, the output is tapped at node k. Inherently monotonic.

**Files:**
- Create: `pyDataconverter/architectures/ResistorStringDAC.py`
- Modify: `tests/test_voltage_mode_dacs.py`

- [ ] **Step 1: Read `DACBase` and `SimpleDAC`**

Read `pyDataconverter/dataconverter.py` (focus on `DACBase`) and `pyDataconverter/architectures/SimpleDAC.py` to understand the required interface before writing any code.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_voltage_mode_dacs.py`:

```python
class TestResistorStringDAC:
    def test_construction(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0, r_unit=1e3)
        assert dac.n_bits == 4
        assert dac.v_ref  == 1.0

    def test_ideal_output_zero(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0)
        assert abs(dac.convert(0)) < 1e-9

    def test_ideal_output_full_scale(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0)
        max_code = 2**4 - 1
        expected = max_code / 2**4 * 1.0
        assert abs(dac.convert(max_code) - expected) < 1e-9

    def test_ideal_monotone(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=6, v_ref=1.0)
        voltages = [dac.convert(c) for c in range(2**6)]
        assert all(b >= a for a, b in zip(voltages, voltages[1:]))

    def test_ideal_dnl_inl_small(self):
        """Ideal (no mismatch) DAC has DNL and INL < 0.001 LSB."""
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        import numpy as np
        dac = ResistorStringDAC(n_bits=6, v_ref=1.0)
        codes = np.arange(2**6)
        voltages = np.array([dac.convert(int(c)) for c in codes])
        m = calculate_dac_static_metrics(codes, voltages, 6, 1.0)
        assert m['MaxDNL'] < 0.001
        assert m['MaxINL'] < 0.001

    def test_mismatch_increases_inl(self):
        """Non-zero mismatch produces larger INL than ideal."""
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        import numpy as np
        codes = np.arange(2**6)
        dac_ideal    = ResistorStringDAC(n_bits=6, v_ref=1.0, r_mismatch=0.0)
        dac_mismatch = ResistorStringDAC(n_bits=6, v_ref=1.0, r_mismatch=0.05,
                                          seed=42)
        v_ideal    = np.array([dac_ideal.convert(int(c))    for c in codes])
        v_mismatch = np.array([dac_mismatch.convert(int(c)) for c in codes])
        m_ideal    = calculate_dac_static_metrics(codes, v_ideal,    6, 1.0)
        m_mismatch = calculate_dac_static_metrics(codes, v_mismatch, 6, 1.0)
        assert m_mismatch['MaxINL'] > m_ideal['MaxINL']
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_voltage_mode_dacs.py::TestResistorStringDAC -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 4: Implement `ResistorStringDAC`**

Create `pyDataconverter/architectures/ResistorStringDAC.py`:

```python
"""
Resistor String DAC
===================

A voltage-mode DAC using 2^N equal resistors in series between V_ref and GND.
Output is the voltage at the tap corresponding to the digital code.

Inherently monotonic: DNL > -1 LSB for any resistor mismatch.
"""

import numpy as np
from typing import Optional

from pyDataconverter.dataconverter import DACBase
from pyDataconverter.utils.nodal_solver import solve_resistor_network


class ResistorStringDAC(DACBase):
    """
    Resistor string (Kelvin divider) DAC.

    2^N resistors are connected in series between V_ref and GND.
    For code k, the output is tapped at node k (counting from GND).

    Resistor mismatch is modelled as multiplicative Gaussian error
    drawn once at construction: R_k = r_unit * (1 + ε_k),
    ε_k ~ N(0, r_mismatch²).

    Attributes:
        r_unit (float): Nominal unit resistor value (Ω).
        r_mismatch (float): Std of multiplicative resistor mismatch.
        r_values (np.ndarray): Actual resistor values (with mismatch), GND→Vref order.
        _tap_voltages (np.ndarray): Pre-computed output voltage per code.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        r_unit: float = 1e3,
        r_mismatch: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_bits: DAC resolution.
            v_ref: Reference voltage (V).
            r_unit: Nominal unit resistor (Ω), default 1 kΩ.
            r_mismatch: Std of multiplicative mismatch (e.g. 0.01 = 1 %).
            seed: Random seed for mismatch draw (None = non-deterministic).
        """
        super().__init__(n_bits, v_ref)
        self.r_unit     = r_unit
        self.r_mismatch = r_mismatch

        n_codes = 2 ** n_bits
        rng = np.random.default_rng(seed)
        epsilons = rng.normal(0.0, r_mismatch, size=n_codes) if r_mismatch > 0 else np.zeros(n_codes)
        self.r_values = r_unit * (1.0 + epsilons)

        self._tap_voltages = self._compute_tap_voltages()

    def _compute_tap_voltages(self) -> np.ndarray:
        """Solve the resistor ladder for all 2^N tap voltages."""
        n_codes  = 2 ** self.n_bits
        n_nodes  = n_codes + 1  # node 0 = GND, node n_codes = V_ref

        resistors = [(k, k + 1, float(self.r_values[k])) for k in range(n_codes)]
        fixed     = {0: 0.0, n_nodes - 1: self.v_ref}

        voltages = solve_resistor_network(n_nodes, resistors, fixed)
        # Tap k → node k (code 0 → node 0 = 0 V, code n_codes-1 → node n_codes-1)
        return voltages[:n_codes]

    def convert(self, code: int) -> float:
        """
        Convert digital code to output voltage.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            float: Output voltage (V).
        """
        if not (0 <= code < 2 ** self.n_bits):
            raise ValueError(f"code {code} out of range")
        return float(self._tap_voltages[code])

    def __repr__(self) -> str:
        return (f"ResistorStringDAC(n_bits={self.n_bits}, v_ref={self.v_ref}, "
                f"r_unit={self.r_unit}, r_mismatch={self.r_mismatch})")
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_voltage_mode_dacs.py::TestResistorStringDAC -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add pyDataconverter/architectures/ResistorStringDAC.py tests/test_voltage_mode_dacs.py
git commit -m "feat: add ResistorStringDAC (resistor ladder, voltage-mode)"
```

---

## Task 3: Add `R2RDAC`

R-2R ladder: each of the N bit nodes either connects to V_ref (bit = 1) or GND (bit = 0) via a 2R resistor. Horizontal arms are R. Only 2N resistors total.

**Files:**
- Create: `pyDataconverter/architectures/R2RDAC.py`
- Modify: `tests/test_voltage_mode_dacs.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestR2RDAC:
    def test_construction(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=4, v_ref=1.0)
        assert dac.n_bits == 4

    def test_ideal_output_zero(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        assert abs(dac.convert(0)) < 1e-9

    def test_ideal_output_msb_only(self):
        """MSB only (code = 2^(N-1)) → V_ref/2."""
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        assert abs(dac.convert(2**5) - 0.5) < 1e-6

    def test_ideal_monotone(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        voltages = [dac.convert(c) for c in range(2**6)]
        assert all(b >= a - 1e-9 for a, b in zip(voltages, voltages[1:]))

    def test_ideal_dnl_inl_small(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        import numpy as np
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        codes = np.arange(2**6)
        voltages = np.array([dac.convert(int(c)) for c in codes])
        m = calculate_dac_static_metrics(codes, voltages, 6, 1.0)
        assert m['MaxDNL'] < 0.01
        assert m['MaxINL'] < 0.01

    def test_r_mismatch_independent_r_and_2r(self):
        """Separate R and 2R mismatch parameters can be set."""
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0, r_mismatch=0.02, r2_mismatch=0.03,
                     seed=0)
        assert dac.r_mismatch  == 0.02
        assert dac.r2_mismatch == 0.03

    def test_mismatch_increases_inl(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        import numpy as np
        codes = np.arange(2**6)
        dac_ideal    = R2RDAC(n_bits=6, v_ref=1.0)
        dac_mismatch = R2RDAC(n_bits=6, v_ref=1.0, r_mismatch=0.05,
                               r2_mismatch=0.05, seed=99)
        v_ideal    = np.array([dac_ideal.convert(int(c))    for c in codes])
        v_mismatch = np.array([dac_mismatch.convert(int(c)) for c in codes])
        m_ideal    = calculate_dac_static_metrics(codes, v_ideal,    6, 1.0)
        m_mismatch = calculate_dac_static_metrics(codes, v_mismatch, 6, 1.0)
        assert m_mismatch['MaxINL'] > m_ideal['MaxINL']
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_voltage_mode_dacs.py::TestR2RDAC -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `R2RDAC`**

Create `pyDataconverter/architectures/R2RDAC.py`:

```python
"""
R-2R Ladder DAC
===============

Voltage-mode R-2R ladder DAC using nodal analysis.

The R-2R network has N+1 nodes (0=GND terminal, 1..N=bit nodes, with node N
connected to the output buffer).  Each bit node k is driven to V_ref (bit=1)
or GND (bit=0) via a 2R resistor.  The horizontal rungs between adjacent bit
nodes are R resistors.

Ideal output:
    V_out = sum(b_k * V_ref / 2^(N-k+1))  for k=1..N  (MSB first)

This implementation directly solves the network for each code using the
MNA nodal solver.  The output is taken at node N (the MSB node), which
connects to an ideal output buffer.

Separate mismatch parameters for the R (horizontal) and 2R (vertical) arms
allow more realistic modelling of actual process variation.
"""

import numpy as np
from typing import Optional

from pyDataconverter.dataconverter import DACBase
from pyDataconverter.utils.nodal_solver import solve_resistor_network


class R2RDAC(DACBase):
    """
    R-2R ladder voltage-mode DAC.

    Attributes:
        r_unit (float): Nominal R value (Ω).
        r_mismatch (float): Std of multiplicative mismatch for R arms.
        r2_mismatch (float): Std of multiplicative mismatch for 2R arms.
        r_values (np.ndarray): Actual R-arm values, index k = rung between
            node k and node k+1, length n_bits.
        r2_values (np.ndarray): Actual 2R-arm values, index k = vertical arm
            at node k, length n_bits.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        r_unit: float = 1e3,
        r_mismatch: float = 0.0,
        r2_mismatch: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_bits: DAC resolution.
            v_ref: Reference voltage (V).
            r_unit: Nominal R value (Ω).  The 2R arms use 2 * r_unit nominally.
            r_mismatch: Mismatch std for horizontal R arms.
            r2_mismatch: Mismatch std for vertical 2R arms.
            seed: Random seed for mismatch (None = non-deterministic).
        """
        super().__init__(n_bits, v_ref)
        self.r_unit     = r_unit
        self.r_mismatch  = r_mismatch
        self.r2_mismatch = r2_mismatch

        rng = np.random.default_rng(seed)

        eps_r  = rng.normal(0, r_mismatch,  size=n_bits) if r_mismatch  > 0 else np.zeros(n_bits)
        eps_r2 = rng.normal(0, r2_mismatch, size=n_bits) if r2_mismatch > 0 else np.zeros(n_bits)

        self.r_values  = r_unit       * (1.0 + eps_r)
        self.r2_values = 2.0 * r_unit * (1.0 + eps_r2)

    def _build_network(self, code: int):
        """
        Build node list and resistor list for the given code.

        Node layout:
            0       = output (Thévenin) / MSB end
            1..N-1  = internal ladder nodes
            N       = GND (fixed 0 V)
            N+1     = V_ref source (fixed v_ref) — used as switch target for bit=1

        For bit k (MSB=0, LSB=N-1):
            - Horizontal rung: node k ↔ node k+1 with r_values[k]
            - Vertical arm: node k ↔ GND (bit=0) or V_ref (bit=1) with r2_values[k]
        The termination 2R on the far end (node N-1) connects to GND always.
        """
        n     = self.n_bits
        # Nodes: 0..n-1 = ladder nodes (0 is MSB end = output)
        #        n = GND, n+1 = V_ref
        n_nodes  = n + 2
        gnd_node = n
        vref_node = n + 1

        resistors = []
        # Horizontal rungs (R arms)
        for k in range(n - 1):
            resistors.append((k, k + 1, float(self.r_values[k])))
        # Termination at LSB end (2R to GND)
        resistors.append((n - 1, gnd_node, float(self.r2_values[-1])))

        # Vertical (2R) arms — bit k is MSB at node 0, LSB at node n-1
        bits = [(code >> (n - 1 - k)) & 1 for k in range(n)]
        for k in range(n - 1):  # skip last (already handled as termination)
            switch_node = vref_node if bits[k] else gnd_node
            resistors.append((k, switch_node, float(self.r2_values[k])))

        fixed = {gnd_node: 0.0, vref_node: self.v_ref}
        return n_nodes, resistors, fixed

    def convert(self, code: int) -> float:
        """
        Convert digital code to output voltage at the MSB node (node 0).

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            float: Output voltage (V).
        """
        if not (0 <= code < 2 ** self.n_bits):
            raise ValueError(f"code {code} out of range")
        n_nodes, resistors, fixed = self._build_network(code)
        voltages = solve_resistor_network(n_nodes, resistors, fixed)
        return float(voltages[0])

    def __repr__(self) -> str:
        return (f"R2RDAC(n_bits={self.n_bits}, v_ref={self.v_ref}, "
                f"r_unit={self.r_unit}, r_mismatch={self.r_mismatch}, "
                f"r2_mismatch={self.r2_mismatch})")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_voltage_mode_dacs.py::TestR2RDAC -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/architectures/R2RDAC.py tests/test_voltage_mode_dacs.py
git commit -m "feat: add R2RDAC (R-2R ladder, voltage-mode)"
```

---

## Task 4: Add `SegmentedResistorDAC`

Combines a thermometer-coded coarse resistor string (for the top n_therm bits) with a binary R-2R fine sub-DAC (for the lower n_bits − n_therm bits). The coarse string sets the base voltage; the fine R-2R adds a fraction of one coarse LSB.

**Files:**
- Create: `pyDataconverter/architectures/SegmentedResistorDAC.py`
- Modify: `tests/test_voltage_mode_dacs.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestSegmentedResistorDAC:
    def test_construction(self):
        from pyDataconverter.architectures.SegmentedResistorDAC import SegmentedResistorDAC
        dac = SegmentedResistorDAC(n_bits=8, v_ref=1.0, n_therm=4)
        assert dac.n_bits == 8

    def test_ideal_output_zero(self):
        from pyDataconverter.architectures.SegmentedResistorDAC import SegmentedResistorDAC
        dac = SegmentedResistorDAC(n_bits=8, v_ref=1.0, n_therm=4)
        assert abs(dac.convert(0)) < 1e-6

    def test_ideal_monotone(self):
        from pyDataconverter.architectures.SegmentedResistorDAC import SegmentedResistorDAC
        dac = SegmentedResistorDAC(n_bits=8, v_ref=1.0, n_therm=4)
        voltages = [dac.convert(c) for c in range(2**8)]
        assert all(b >= a - 1e-9 for a, b in zip(voltages, voltages[1:]))

    def test_ideal_dnl_inl_small(self):
        from pyDataconverter.architectures.SegmentedResistorDAC import SegmentedResistorDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        import numpy as np
        dac = SegmentedResistorDAC(n_bits=8, v_ref=1.0, n_therm=4)
        codes = np.arange(2**8)
        voltages = np.array([dac.convert(int(c)) for c in codes])
        m = calculate_dac_static_metrics(codes, voltages, 8, 1.0)
        assert m['MaxDNL'] < 0.01
        assert m['MaxINL'] < 0.01

    def test_output_range(self):
        from pyDataconverter.architectures.SegmentedResistorDAC import SegmentedResistorDAC
        dac = SegmentedResistorDAC(n_bits=8, v_ref=1.0, n_therm=4)
        max_code = 2**8 - 1
        assert abs(dac.convert(max_code) - max_code / 2**8) < 0.005
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_voltage_mode_dacs.py::TestSegmentedResistorDAC -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `SegmentedResistorDAC`**

Create `pyDataconverter/architectures/SegmentedResistorDAC.py`:

```python
"""
Segmented Resistor DAC
======================

Combines a thermometer-coded coarse resistor string (top n_therm bits)
with a binary R-2R fine sub-DAC (lower n_bits - n_therm bits).

The coarse string divides [0, V_ref] into 2^n_therm equal steps.
The fine R-2R subdivides one coarse step into 2^n_fine equal sub-steps.

V_out = V_coarse_base + V_fine * (coarse_lsb / V_ref_fine)
"""

import numpy as np
from typing import Optional

from pyDataconverter.dataconverter import DACBase
from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
from pyDataconverter.architectures.R2RDAC import R2RDAC


class SegmentedResistorDAC(DACBase):
    """
    Segmented resistor DAC: coarse thermometer string + fine R-2R.

    Args:
        n_bits: Total DAC resolution.
        v_ref: Full-scale reference voltage (V).
        n_therm: Number of MSBs handled by the thermometer string.
        r_unit: Nominal unit resistor (Ω).
        r_mismatch: Mismatch std for all resistors.
        seed: Random seed for mismatch.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        n_therm: int = 4,
        r_unit: float = 1e3,
        r_mismatch: float = 0.0,
        seed: Optional[int] = None,
    ):
        n_fine = n_bits - n_therm
        if not (1 <= n_therm <= n_bits - 1):
            raise ValueError(f"n_therm must be in [1, {n_bits-1}]")

        super().__init__(n_bits, v_ref)
        self._n_therm = n_therm
        self._n_fine  = n_fine

        coarse_lsb = v_ref / 2 ** n_therm  # voltage per coarse step

        # Coarse string: 2^n_therm codes spanning [0, v_ref]
        rng = np.random.default_rng(seed)
        coarse_seed = int(rng.integers(0, 2**31))
        fine_seed   = int(rng.integers(0, 2**31))

        self._coarse = ResistorStringDAC(
            n_bits=n_therm, v_ref=v_ref, r_unit=r_unit,
            r_mismatch=r_mismatch, seed=coarse_seed,
        )
        # Fine R-2R spans one coarse LSB
        self._fine = R2RDAC(
            n_bits=n_fine, v_ref=coarse_lsb, r_unit=r_unit,
            r_mismatch=r_mismatch, seed=fine_seed,
        )

    def convert(self, code: int) -> float:
        """
        Convert code to output voltage.

        Splits code into coarse (upper n_therm bits) and fine (lower n_fine bits),
        evaluates each sub-DAC independently, and sums.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            float: Output voltage (V).
        """
        if not (0 <= code < 2 ** self.n_bits):
            raise ValueError(f"code {code} out of range")
        coarse_code = code >> self._n_fine
        fine_code   = code & ((1 << self._n_fine) - 1)
        v_coarse = self._coarse.convert(coarse_code)
        v_fine   = self._fine.convert(fine_code)
        return float(v_coarse + v_fine)

    def __repr__(self) -> str:
        return (f"SegmentedResistorDAC(n_bits={self.n_bits}, v_ref={self.v_ref}, "
                f"n_therm={self._n_therm})")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_voltage_mode_dacs.py::TestSegmentedResistorDAC -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyDataconverter/architectures/SegmentedResistorDAC.py tests/test_voltage_mode_dacs.py
git commit -m "feat: add SegmentedResistorDAC (thermometer coarse + R-2R fine)"
```

---

## Task 5: Update exports

**Files:**
- Modify: `pyDataconverter/architectures/__init__.py`

- [ ] **Step 1: Add new DACs to exports**

Read `pyDataconverter/architectures/__init__.py`, then add:

```python
from .ResistorStringDAC import ResistorStringDAC
from .R2RDAC import R2RDAC
from .SegmentedResistorDAC import SegmentedResistorDAC
```

- [ ] **Step 2: Run full suite one final time**

```bash
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add pyDataconverter/architectures/__init__.py
git commit -m "chore: export ResistorStringDAC, R2RDAC, SegmentedResistorDAC"
```
