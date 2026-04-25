"""
Segmented Resistor DAC
======================

Combines a thermometer-coded coarse resistor string (top n_therm bits)
with a binary R-2R fine sub-DAC (lower n_bits - n_therm bits).

The coarse resistor string divides [0, V_ref] into 2^n_therm equal steps.
The fine R-2R sub-DAC spans exactly one coarse step (coarse_lsb = V_ref /
2^n_therm) and subdivides it into 2^n_fine equal sub-steps.

Output formula (ideal, no mismatch):
    V_out = V_coarse(coarse_code) + V_fine(fine_code)

where:
    coarse_code = code >> n_fine         (upper n_therm bits)
    fine_code   = code & (2^n_fine - 1)  (lower n_fine bits)
    V_coarse(k) = k / 2^n_therm * v_ref
    V_fine(k)   = k / 2^n_fine  * coarse_lsb

Classes:
    SegmentedResistorDAC: Thermometer coarse + R-2R fine segmented DAC.

Version History:
---------------
1.0.0 (2026-04-02):
    - Initial release
"""

import numpy as np
from typing import Optional

from pyDataconverter.dataconverter import DACBase, OutputType
from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
from pyDataconverter.architectures.R2RDAC import R2RDAC


class SegmentedResistorDAC(DACBase):
    """
    Segmented resistor DAC: thermometer-coded coarse string + R-2R fine sub-DAC.

    The top n_therm bits select a segment on the coarse resistor string.
    The lower n_fine = n_bits - n_therm bits drive the R-2R fine sub-DAC,
    which spans one coarse LSB voltage.

    Level count: power-of-two by construction (2^n_therm coarse segments ×
    2^n_fine fine steps = 2^n_bits total codes).  The R-2R fine sub-DAC
    alone pins the topology to a power of two; ``n_levels`` is not
    exposed.  For non-power-of-two DACs, use ``SimpleDAC``.

    Output type: single-ended only, inherited from the underlying
    single-ended coarse resistor string and R-2R fine sub-DAC.  For
    differential output instantiate two ``SegmentedResistorDAC`` objects
    and combine their outputs externally.  The ``output_type`` kwarg is
    exposed in the constructor so the constraint is visible in the
    signature; passing ``OutputType.DIFFERENTIAL`` raises ``ValueError``
    with a pointer to the composition pattern.

    Attributes:
        n_therm (int): Number of MSBs handled by the thermometer coarse string.
        n_fine (int): Number of LSBs handled by the R-2R fine sub-DAC
            (= n_bits - n_therm).
        r_unit (float): Nominal unit resistor (Ω).
        r_mismatch (float): Std of multiplicative resistor mismatch applied to
            both the coarse and fine stages.
        _coarse (ResistorStringDAC): Internal coarse stage.
        _fine (R2RDAC): Internal fine stage.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        n_therm: int = 4,
        r_unit: float = 1e3,
        r_mismatch: float = 0.0,
        output_type: OutputType = OutputType.SINGLE,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_bits:      Total DAC resolution (2–32).
            v_ref:       Full-scale reference voltage (V, > 0).
            n_therm:     Number of MSBs for the thermometer coarse string.
                         Must satisfy 1 <= n_therm <= n_bits - 1.
            r_unit:      Nominal unit resistor (Ω), default 1 kΩ.
            r_mismatch:  Std of multiplicative resistor mismatch (e.g. 0.01 = 1 %).
                         Applied independently to coarse and fine stages.
                         Must be >= 0.
            output_type: Must be ``OutputType.SINGLE``.  Both sub-stages
                         (coarse resistor string + fine R-2R) are
                         inherently single-ended; for differential output,
                         instantiate two ``SegmentedResistorDAC`` objects
                         and combine their outputs externally.
            seed:        Random seed for reproducible mismatch draws.

        Raises:
            TypeError:  If n_bits is not an integer or v_ref is not a number.
            ValueError: If n_bits is out of range, v_ref <= 0, r_unit <= 0,
                        r_mismatch < 0, n_therm is out of [1, n_bits-1],
                        or output_type is not SINGLE.
        """
        if output_type != OutputType.SINGLE:
            raise ValueError(
                "SegmentedResistorDAC models a single-ended thermometer + R-2R "
                "stack and only supports output_type=OutputType.SINGLE. For a "
                "differential output, instantiate two SegmentedResistorDAC "
                "objects and combine their outputs externally."
            )
        super().__init__(n_bits=n_bits, v_ref=v_ref, output_type=OutputType.SINGLE)

        if not isinstance(n_therm, int) or not (1 <= n_therm <= n_bits - 1):
            raise ValueError(
                f"n_therm must be an integer in [1, {n_bits - 1}], got {n_therm}"
            )
        if not isinstance(r_unit, (int, float)) or r_unit <= 0:
            raise ValueError("r_unit must be a positive number")
        if not isinstance(r_mismatch, (int, float)) or r_mismatch < 0:
            raise ValueError("r_mismatch must be >= 0")

        self.n_therm = n_therm
        self.n_fine = n_bits - n_therm
        self.r_unit = float(r_unit)
        self.r_mismatch = float(r_mismatch)
        self.seed = seed

        # Derive separate seeds for coarse and fine stages so the mismatch
        # draws are independent but the combined result is fully reproducible.
        rng = np.random.default_rng(seed)
        coarse_seed = int(rng.integers(0, 2**31))
        fine_seed = int(rng.integers(0, 2**31))

        # Coarse stage: n_therm-bit resistor string spanning [0, v_ref]
        self._coarse = ResistorStringDAC(
            n_bits=n_therm,
            v_ref=v_ref,
            r_unit=r_unit,
            r_mismatch=r_mismatch,
            seed=coarse_seed,
        )

        # Fine stage: n_fine-bit R-2R spanning one coarse LSB
        coarse_lsb = v_ref / (2 ** n_therm)
        self._fine = R2RDAC(
            n_bits=self.n_fine,
            v_ref=coarse_lsb,
            r_unit=r_unit,
            r_mismatch=r_mismatch,
            seed=fine_seed,
        )

    # ------------------------------------------------------------------
    # DACBase interface
    # ------------------------------------------------------------------

    def _convert_input(self, digital_input: int) -> float:
        """
        Convert a pre-validated code to output voltage.

        Splits the code into coarse (upper n_therm bits) and fine (lower
        n_fine bits), evaluates each sub-DAC, and sums the results.

        Args:
            digital_input: Pre-validated code in [0, 2^n_bits − 1].

        Returns:
            float: Output voltage (V).
        """
        fine_mask = (1 << self.n_fine) - 1
        coarse_code = digital_input >> self.n_fine
        fine_code = digital_input & fine_mask

        v_coarse = self._coarse.convert(coarse_code)
        v_fine = self._fine.convert(fine_code)
        return float(v_coarse + v_fine)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [
            f"n_bits={self.n_bits}",
            f"v_ref={self.v_ref}",
            f"n_therm={self.n_therm}",
            f"r_unit={self.r_unit}",
            f"r_mismatch={self.r_mismatch}",
        ]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return f"SegmentedResistorDAC({', '.join(parts)})"
