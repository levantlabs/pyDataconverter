"""
Resistor String DAC
===================

A voltage-mode DAC using 2^N equal resistors in series between V_ref and GND.
Output is the voltage at the tap corresponding to the digital code.

Inherently monotonic: DNL > -1 LSB for any resistor mismatch.

Classes:
    ResistorStringDAC: Kelvin divider DAC with optional resistor mismatch.

First written 2026-04-02; see ``git log`` for the change history.
"""

import numpy as np
from typing import Optional

from pyDataconverter.dataconverter import DACBase, OutputType
from pyDataconverter.utils.nodal_solver import solve_resistor_network


class ResistorStringDAC(DACBase):
    """
    Resistor string (Kelvin divider) DAC.

    2^N resistors are connected in series between V_ref (top) and GND (bottom).
    For code k, the output is tapped at node k (counting from GND).

    Ideal output: V_out = code / 2^N * v_ref
    (code 0 → 0 V; code 2^N-1 → v_ref * (2^N-1)/2^N)

    Resistor mismatch is modelled as multiplicative Gaussian error drawn once
    at construction: R_k = r_unit * (1 + ε_k), ε_k ~ N(0, r_mismatch²).

    Tap voltages are pre-computed at construction via nodal analysis so that
    repeated calls to ``convert`` are O(1).

    Level count: the current implementation uses 2^N resistors for a
    power-of-two code space.  Non-power-of-two level counts (e.g., for
    pipelined-ADC sub-DACs) are not supported here — use ``SimpleDAC``.
    ``n_levels`` is therefore not exposed in the constructor.

    Output type: single-ended only.  A resistor string is inherently
    single-ended; for differential output instantiate two
    ``ResistorStringDAC`` objects and combine their outputs externally.
    The ``output_type`` kwarg is exposed in the constructor so the
    constraint is visible in the signature; passing
    ``OutputType.DIFFERENTIAL`` raises ``ValueError`` with a pointer to
    the composition pattern.

    Attributes:
        r_unit (float): Nominal unit resistor value (Ω).
        r_mismatch (float): Std of multiplicative resistor mismatch.
        r_values (np.ndarray): Actual resistor values (with mismatch),
            ordered from GND to V_ref (length 2^N).
        _tap_voltages (np.ndarray): Pre-computed output voltage per code.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        r_unit: float = 1e3,
        r_mismatch: float = 0.0,
        output_type: OutputType = OutputType.SINGLE,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_bits:      DAC resolution (1–32).
            v_ref:       Reference voltage (V, > 0).
            r_unit:      Nominal unit resistor (Ω), default 1 kΩ.
            r_mismatch:  Std of multiplicative mismatch (e.g. 0.01 = 1 %).
                         Must be >= 0.
            output_type: Must be ``OutputType.SINGLE``.  A resistor string is
                         inherently single-ended; for differential output,
                         instantiate two ``ResistorStringDAC`` objects and
                         combine their outputs externally.  Accepted as an
                         explicit kwarg so the constraint is visible in the
                         signature and rejected with a clear error instead of
                         silently raising ``TypeError`` on the base class.
            seed:        Random seed for mismatch draw (None = non-deterministic).

        Raises:
            TypeError:  If n_bits is not an integer or v_ref is not a number.
            ValueError: If n_bits is out of range, v_ref <= 0, r_unit <= 0,
                        r_mismatch < 0, or output_type is not SINGLE.
        """
        if output_type != OutputType.SINGLE:
            raise ValueError(
                "ResistorStringDAC models a single-ended resistor ladder and "
                "only supports output_type=OutputType.SINGLE. For a differential "
                "output, instantiate two ResistorStringDAC objects and combine "
                "their outputs externally."
            )
        super().__init__(n_bits=n_bits, v_ref=v_ref, output_type=OutputType.SINGLE)

        if not isinstance(r_unit, (int, float)) or r_unit <= 0:
            raise ValueError("r_unit must be a positive number")
        if not isinstance(r_mismatch, (int, float)) or r_mismatch < 0:
            raise ValueError("r_mismatch must be >= 0")

        self.r_unit = float(r_unit)
        self.r_mismatch = float(r_mismatch)
        self.seed = seed

        n_codes = 2 ** n_bits
        rng = np.random.default_rng(seed)
        if r_mismatch > 0:
            epsilons = rng.normal(0.0, r_mismatch, size=n_codes)
        else:
            epsilons = np.zeros(n_codes)
        self.r_values = self.r_unit * (1.0 + epsilons)

        self._tap_voltages = self._compute_tap_voltages()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_tap_voltages(self) -> np.ndarray:
        """Solve the resistor ladder for all 2^N tap voltages via MNA."""
        n_codes = 2 ** self.n_bits
        n_nodes = n_codes + 1  # node 0 = GND, node n_codes = V_ref

        # Resistor k connects node k to node k+1 (GND side to V_ref side)
        resistors = [(k, k + 1, float(self.r_values[k])) for k in range(n_codes)]
        fixed = {0: 0.0, n_nodes - 1: self.v_ref}

        voltages = solve_resistor_network(n_nodes, resistors, fixed)
        # Tap for code k is node k (code 0 → node 0 = 0 V)
        return voltages[:n_codes]

    # ------------------------------------------------------------------
    # DACBase interface
    # ------------------------------------------------------------------

    def _convert_input(self, digital_input: int) -> float:
        """
        Return the pre-computed tap voltage for the given code.

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
        parts = [
            f"n_bits={self.n_bits}",
            f"v_ref={self.v_ref}",
            f"r_unit={self.r_unit}",
            f"r_mismatch={self.r_mismatch}",
        ]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return f"ResistorStringDAC({', '.join(parts)})"
