"""
Current-Steering DAC Module
============================

Models a current-steering DAC in binary, thermometer, or segmented
configuration.  The architecture is controlled by two composable components:

    Decoder        — translates the N-bit code into thermometer and binary
                     control signals (see components/decoder.py).
    CurrentSourceArray — holds the physical element arrays and computes the
                     total selected current (see components/current_source.py).

All sources are always conducting; they are *steered* between positive and
negative output rails, keeping i_total constant regardless of code.  This is
the defining property of current-steering DACs.

Segmentation is controlled by n_therm_bits:

    n_therm_bits = 0        → fully binary-weighted
    n_therm_bits = n_bits   → fully thermometer (unary)
    0 < n_therm_bits < n_bits → segmented (MSBs thermometer, LSBs binary)

Output voltage (single-ended or differential) is derived from the steered
current through a resistive load r_load.

Classes:
    CurrentSteeringDAC: Current-steering DAC inheriting from DACBase.

First written 2026-03-25; see ``git log`` for the change history.

Notes:
------
Output conventions
------------------
Single-ended (OutputType.SINGLE):
    v_out = i_selected * r_load

Differential (OutputType.DIFFERENTIAL):
    i_pos = i_selected
    i_neg = i_total - i_selected
    Returns (v_pos, v_neg) = (i_pos * r_load, i_neg * r_load)

    The differential output voltage is:
        v_diff = (i_pos - i_neg) * r_load
               = (2 * i_selected - i_total) * r_load

    For ideal binary code with no mismatch and full-scale input:
        i_selected ≈ i_total → v_diff ≈ i_total * r_load

Component dependencies
----------------------
Decoder         : pyDataconverter.components.decoder.SegmentedDecoder
CurrentSourceArray : pyDataconverter.components.current_source.CurrentSourceArray
"""

import functools
from typing import Optional, Tuple, Type, Union
import numpy as np

from pyDataconverter.dataconverter import DACBase, OutputType
from pyDataconverter.components.decoder import (
    DecoderBase, BinaryDecoder, ThermometerDecoder, SegmentedDecoder,
)
from pyDataconverter.components.current_source import (
    CurrentSourceArray, UnitCurrentSourceBase, IdealCurrentSource,
)


class CurrentSteeringDAC(DACBase):
    """
    Current-steering DAC with configurable segmentation.

    Supports binary, thermometer, and segmented topologies through the
    n_therm_bits parameter.  A custom decoder or current source array can
    be injected at construction time for advanced use cases.

    Level count: the current implementation uses 2^n_bits codes regardless
    of segmentation (the decoder/element arrays are sized accordingly).
    ``n_levels`` is not exposed in the constructor.  For non-power-of-two
    DACs — e.g., a pipelined-ADC sub-DAC matching an N-comparator flash
    sub-ADC's N+1 codes — use ``SimpleDAC``.

    Attributes:
        n_therm_bits (int):     Number of MSBs decoded as thermometer.
        n_binary_bits (int):    Number of LSBs remaining as binary.
        i_unit (float):         Nominal unit current (A).
        r_load (float):         Load resistance (Ω) for I→V conversion.
        current_mismatch (float): Std of multiplicative current source mismatch.
        decoder (DecoderBase):  Decoder instance used for code splitting.
        current_array (CurrentSourceArray): Element array instance.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        n_therm_bits: int = 0,
        i_unit: float = 100e-6,
        r_load: float = 1000.0,
        current_mismatch: float = 0.0,
        output_type: OutputType = OutputType.DIFFERENTIAL,
        decoder: Optional[DecoderBase] = None,
        current_array: Optional[CurrentSourceArray] = None,
        source_class: Type[UnitCurrentSourceBase] = IdealCurrentSource,
    ):
        """
        Args:
            n_bits:           DAC resolution in bits (1–32).
            v_ref:            Reference voltage (V, > 0).  Stored for
                              compatibility with DACBase; output voltage is
                              determined by i_unit and r_load.
            n_therm_bits:     MSBs decoded as thermometer (0 = fully binary,
                              n_bits = fully thermometer).
            i_unit:           Nominal unit current source value (A, > 0).
            r_load:           Load resistance (Ω, > 0) for I→V conversion.
            current_mismatch: Std of multiplicative mismatch per source (≥ 0).
                              Each source draws independently at construction.
            output_type:      OutputType.SINGLE or OutputType.DIFFERENTIAL.
            decoder:          Optional pre-built DecoderBase instance.  If
                              None a SegmentedDecoder is created from n_bits
                              and n_therm_bits.
            current_array:    Optional pre-built CurrentSourceArray.  If None
                              one is created from the other parameters.
            source_class:     UnitCurrentSourceBase subclass to use when
                              building the current array automatically.

        Raises:
            ValueError: If parameters are out of range.
            TypeError:  If decoder or current_array have wrong types.
        """
        super().__init__(n_bits=n_bits, v_ref=v_ref, output_type=output_type)

        if not isinstance(n_therm_bits, int) or n_therm_bits < 0 or n_therm_bits > n_bits:
            raise ValueError(
                f"n_therm_bits must be an integer in [0, n_bits={n_bits}]"
            )
        if not isinstance(i_unit, (int, float)) or i_unit <= 0:
            raise ValueError("i_unit must be a positive number")
        if not isinstance(r_load, (int, float)) or r_load <= 0:
            raise ValueError("r_load must be a positive number")
        if not isinstance(current_mismatch, (int, float)) or current_mismatch < 0:
            raise ValueError("current_mismatch must be >= 0")

        self._n_therm_bits = n_therm_bits
        self._i_unit       = float(i_unit)
        self._r_load       = float(r_load)
        self._current_mismatch = float(current_mismatch)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        if decoder is not None:
            if not isinstance(decoder, DecoderBase):
                raise TypeError("decoder must be a DecoderBase instance")
            if decoder.n_bits != n_bits:
                raise ValueError(
                    f"decoder.n_bits={decoder.n_bits} does not match n_bits={n_bits}"
                )
            self.decoder = decoder
        else:
            self.decoder = SegmentedDecoder(n_bits=n_bits, n_therm_bits=n_therm_bits)

        # ------------------------------------------------------------------
        # Current source array
        # ------------------------------------------------------------------
        if current_array is not None:
            if not isinstance(current_array, CurrentSourceArray):
                raise TypeError("current_array must be a CurrentSourceArray instance")
            if current_array.n_bits != n_bits:
                raise ValueError(
                    f"current_array.n_bits={current_array.n_bits} does not match "
                    f"n_bits={n_bits}"
                )
            self.current_array = current_array
        else:
            self.current_array = CurrentSourceArray(
                n_therm_bits  = n_therm_bits,
                n_binary_bits = n_bits - n_therm_bits,
                i_unit        = self._i_unit,
                current_mismatch = current_mismatch,
                source_class  = source_class,
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_therm_bits(self) -> int:
        """Number of MSBs decoded as thermometer."""
        return self._n_therm_bits

    @property
    def n_binary_bits(self) -> int:
        """Number of LSBs that remain as binary."""
        return self.n_bits - self._n_therm_bits

    @property
    def i_unit(self) -> float:
        """Nominal unit current (A)."""
        return self._i_unit

    @property
    def r_load(self) -> float:
        """Load resistance (Ω)."""
        return self._r_load

    @property
    def current_mismatch(self) -> float:
        """Standard deviation of multiplicative mismatch used at construction."""
        return self._current_mismatch

    @property
    def i_total(self) -> float:
        """
        Total current from all sources (A).

        Constant regardless of code — this is the fundamental property of a
        current-steering DAC.
        """
        return self.current_array.i_total

    @functools.cached_property
    def dac_currents(self) -> np.ndarray:
        """
        Ideal output current (positive rail) for every code.

        Cached after first access — the decode + lookup is done once per
        instance.  Mismatch is fixed at construction (CurrentSteeringDAC
        has no in-place re-draw method), so the cache never needs to be
        invalidated.

        Returns:
            np.ndarray: Shape (2^n_bits,).  Entry k is i_selected for code k.
        """
        n_codes = 2 ** self.n_bits
        currents = np.zeros(n_codes)
        for code in range(n_codes):
            therm_idx, bin_bits = self.decoder.decode(code)
            i_sel, _ = self.current_array.get_current(therm_idx, bin_bits)
            currents[code] = i_sel
        return currents

    # ------------------------------------------------------------------
    # DACBase interface
    # ------------------------------------------------------------------

    def _convert_input(
        self, digital_input: int
    ) -> Union[float, Tuple[float, float]]:
        """
        Convert a validated digital code to an analog output voltage.

        Args:
            digital_input: Pre-validated code in [0, 2^n_bits − 1].

        Returns:
            float if output_type == SINGLE: v_out = i_selected * r_load.
            Tuple[float, float] if output_type == DIFFERENTIAL:
                (v_pos, v_neg) = (i_pos * r_load, i_neg * r_load).
        """
        therm_idx, bin_bits = self.decoder.decode(digital_input)
        i_selected, i_total = self.current_array.get_current(therm_idx, bin_bits)

        if self.output_type == OutputType.SINGLE:
            return i_selected * self._r_load
        else:
            i_pos = i_selected
            i_neg = i_total - i_selected
            return (i_pos * self._r_load, i_neg * self._r_load)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = (
            'binary'       if self._n_therm_bits == 0 else
            'thermometer'  if self._n_therm_bits == self.n_bits else
            f'segmented({self._n_therm_bits}T+{self.n_binary_bits}B)'
        )
        parts = [
            f"n_bits={self.n_bits}",
            f"mode={mode}",
            f"i_unit={self._i_unit:.3g}A",
            f"r_load={self._r_load:.3g}Ω",
            f"output_type={self.output_type.name}",
        ]
        if self._current_mismatch > 0:
            parts.append(f"current_mismatch={self._current_mismatch:.3g}")
        return f"CurrentSteeringDAC({', '.join(parts)})"
