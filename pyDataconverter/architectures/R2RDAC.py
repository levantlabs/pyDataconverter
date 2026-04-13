"""
R-2R Ladder DAC
===============

Voltage-mode R-2R ladder DAC using nodal analysis.

The R-2R network has N+1 nodes.  Node 0 is the output (MSB end); nodes 1
through N-1 are internal ladder nodes; node N is GND; node N+1 is V_ref.

For each bit position k (k=0 is MSB, k=N-1 is LSB):
  - A horizontal R arm connects node k to node k+1.
  - A vertical 2R arm connects node k to GND (bit=0) or V_ref (bit=1).

The LSB end (node N-1) is terminated with a 2R arm to GND (this provides
the correct Thevenin termination for the ladder).

The output voltage is taken at node 0 (MSB end).

Ideal output:
    V_out = sum(b_k * V_ref / 2^(k+1))  for k=0..N-1  (MSB first)

Separate mismatch parameters for the R (horizontal) and 2R (vertical) arms
allow realistic modelling of actual process variation.

Classes:
    R2RDAC: R-2R ladder voltage-mode DAC with optional resistor mismatch.

Version History:
---------------
1.0.0 (2026-04-02):
    - Initial release
"""

import numpy as np
from typing import Optional

from pyDataconverter.dataconverter import DACBase, OutputType
from pyDataconverter.utils.nodal_solver import solve_resistor_network
from pyDataconverter.utils._bits import code_to_bits_msb_first


class R2RDAC(DACBase):
    """
    R-2R ladder voltage-mode DAC.

    MSB is connected at node 0 (output end); LSB at node N-1.
    Each bit switches its 2R arm between V_ref (bit=1) and GND (bit=0).
    Horizontal R arms run along the ladder; the far end is terminated with
    a 2R arm to GND.

    Mismatch is modelled as multiplicative Gaussian error drawn once at
    construction, with separate standard deviations for R and 2R arms.

    Tap voltages are pre-computed at construction for all 2^N codes so
    that repeated calls to ``convert`` are O(1).

    Attributes:
        r_unit (float): Nominal R value (Ω).
        r_mismatch (float): Std of multiplicative mismatch for R (horizontal) arms.
        r2_mismatch (float): Std of multiplicative mismatch for 2R (vertical) arms.
        r_values (np.ndarray): Actual R-arm values, length n_bits-1 (rungs).
        r2_values (np.ndarray): Actual 2R-arm values, length n_bits (vertical arms,
            including termination at LSB end).
        _tap_voltages (np.ndarray): Pre-computed output voltage per code.
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
            n_bits:      DAC resolution (1–32).
            v_ref:       Reference voltage (V, > 0).
            r_unit:      Nominal R value (Ω), default 1 kΩ.
                         The nominal 2R arms use 2 * r_unit.
            r_mismatch:  Std of multiplicative mismatch for R (horizontal) arms
                         (e.g. 0.01 = 1 %).  Must be >= 0.
            r2_mismatch: Std of multiplicative mismatch for 2R (vertical) arms
                         (e.g. 0.01 = 1 %).  Must be >= 0.
            seed:        Random seed for mismatch draw (None = non-deterministic).

        Raises:
            TypeError:  If n_bits is not an integer or v_ref is not a number.
            ValueError: If n_bits is out of range, v_ref <= 0, r_unit <= 0,
                        r_mismatch < 0, or r2_mismatch < 0.
        """
        super().__init__(n_bits=n_bits, v_ref=v_ref, output_type=OutputType.SINGLE)

        if not isinstance(r_unit, (int, float)) or r_unit <= 0:
            raise ValueError("r_unit must be a positive number")
        if not isinstance(r_mismatch, (int, float)) or r_mismatch < 0:
            raise ValueError("r_mismatch must be >= 0")
        if not isinstance(r2_mismatch, (int, float)) or r2_mismatch < 0:
            raise ValueError("r2_mismatch must be >= 0")

        self.r_unit = float(r_unit)
        self.r_mismatch = float(r_mismatch)
        self.r2_mismatch = float(r2_mismatch)

        rng = np.random.default_rng(seed)

        # n_bits-1 horizontal R arms (rungs between adjacent nodes)
        n_rungs = n_bits - 1
        if r_mismatch > 0:
            eps_r = rng.normal(0.0, r_mismatch, size=n_rungs)
        else:
            eps_r = np.zeros(n_rungs)
        self.r_values = self.r_unit * (1.0 + eps_r)

        # n_bits vertical 2R arms (one per bit, including LSB termination)
        if r2_mismatch > 0:
            eps_r2 = rng.normal(0.0, r2_mismatch, size=n_bits)
        else:
            eps_r2 = np.zeros(n_bits)
        self.r2_values = 2.0 * self.r_unit * (1.0 + eps_r2)

        self._tap_voltages = self._compute_tap_voltages()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_network(self, code: int):
        """
        Build the resistor network for the given digital code.

        Node layout:
            0 .. n-1  = ladder nodes (node 0 = MSB end = output)
            n         = GND  (fixed 0 V)
            n+1       = V_ref (fixed v_ref)

        Resistors:
            Horizontal rungs: node k ↔ node k+1, value r_values[k],
                              for k = 0 .. n-2.
            Vertical (2R) arms:
                For k = 0 .. n-2: node k ↔ GND or V_ref based on bit k.
                For k = n-1 (LSB end): always terminated to GND with r2_values[n-1],
                              regardless of the LSB bit — the bit is handled via its
                              own separate 2R arm to GND or V_ref.

        Wait — the standard R-2R analysis has EACH bit node connected to GND
        (bit=0) or V_ref (bit=1) via 2R, AND the far end terminated to GND via
        a separate 2R.  So there are N vertical arms (one per bit) PLUS the
        termination.  We use r2_values[k] for bit k and r2_values[n-1] doubled
        as termination.

        Revised node layout (matching standard R-2R ladder):
            Nodes 0..n-1 are ladder nodes.
            Node n   = GND.
            Node n+1 = V_ref.

            Horizontal rungs: node k ↔ node k+1 with r_values[k], k=0..n-2.
            Vertical arms: node k ↔ (GND or V_ref) with r2_values[k], k=0..n-1.
            Termination: node n-1 also gets a 2R to GND (the standard ladder
                         requires this for the Thevenin equivalent to hold;
                         implemented as a separate resistor using r2_values[n-1]).
        """
        n = self.n_bits
        n_nodes = n + 2
        gnd_node = n
        vref_node = n + 1

        # Decode bits: bit k = MSB at k=0, LSB at k=n-1
        bits = code_to_bits_msb_first(code, n, dtype=np.int8)

        resistors = []

        # Horizontal rungs (R arms): node k ↔ node k+1
        for k in range(n - 1):
            resistors.append((k, k + 1, float(self.r_values[k])))

        # Vertical (2R) switch arms: one per bit node
        # bit=1 → switch connects to V_ref; bit=0 → switch connects to GND
        for k in range(n):
            switch_node = vref_node if bits[k] else gnd_node
            resistors.append((k, switch_node, float(self.r2_values[k])))

        # Termination 2R at the LSB end (node n-1) to GND.
        # Standard R-2R ladders require this resistor so that the Thevenin
        # resistance looking into each node is exactly R, enabling ideal
        # binary weighting.  It is separate from the bit switch arm above.
        resistors.append((n - 1, gnd_node, float(self.r2_values[n - 1])))

        fixed = {gnd_node: 0.0, vref_node: self.v_ref}
        return n_nodes, resistors, fixed

    def _compute_tap_voltages(self) -> np.ndarray:
        """Solve the R-2R network for all 2^N codes via MNA."""
        n_codes = 2 ** self.n_bits
        tap_voltages = np.empty(n_codes)
        for code in range(n_codes):
            n_nodes, resistors, fixed = self._build_network(code)
            voltages = solve_resistor_network(n_nodes, resistors, fixed)
            tap_voltages[code] = voltages[0]  # output at node 0 (MSB end)
        return tap_voltages

    # ------------------------------------------------------------------
    # DACBase interface
    # ------------------------------------------------------------------

    def _convert_input(self, digital_input: int) -> float:
        """
        Return the pre-computed output voltage for the given code.

        Args:
            digital_input: Pre-validated code in [0, 2^n_bits − 1].

        Returns:
            float: Output voltage (V).
        """
        return float(self._tap_voltages[digital_input])

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"R2RDAC(n_bits={self.n_bits}, v_ref={self.v_ref}, "
            f"r_unit={self.r_unit}, r_mismatch={self.r_mismatch}, "
            f"r2_mismatch={self.r2_mismatch})"
        )
