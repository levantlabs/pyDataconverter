"""
Unit Current Source Components
================================

Unit current source models and the current source array used by current-
steering DACs.  Each UnitCurrentSource instance represents one physical
current source cell; the CurrentSourceArray holds a thermometer segment
(uniform sources) and a binary segment (binary-weighted sources) and
computes the total selected current given decoder control signals.

Swapping in a different UnitCurrentSource subclass (e.g. CascodeCurrentSource
with higher output impedance) requires no changes to CurrentSourceArray or
the DAC.

Classes:
    UnitCurrentSourceBase:  Abstract base defining the current source interface.
    IdealCurrentSource:     Fixed current with static Gaussian mismatch.
    CurrentSourceArray:     Thermometer + binary arrays; evaluates total DAC output current.

First written 2026-03-25; see ``git log`` for the change history.
"""

from abc import ABC, abstractmethod
import warnings
from typing import List, Tuple, Type
import numpy as np

from pyDataconverter.components.decoder import DecoderBase


class UnitCurrentSourceBase(ABC):
    """
    Abstract base class for unit current source models.

    Defines the interface used by CurrentSourceArray.  Subclasses model
    different physical implementations (ideal, cascode, regulated cascode)
    while keeping the array and DAC code unchanged.

    All subclasses must expose:
        i_nominal — the designed (target) current.
        current   — the actual current used in simulation.
    """

    @property
    @abstractmethod
    def i_nominal(self) -> float:
        """Designed (nominal) current in amperes."""

    @property
    @abstractmethod
    def current(self) -> float:
        """
        Actual current used in simulation.

        For ideal sources this equals i_nominal * (1 + mismatch_draw).
        Subclasses may model output-impedance effects, supply dependence, etc.
        """

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"i_nominal={self.i_nominal:.4g}A, "
                f"current={self.current:.4g}A)")


class IdealCurrentSource(UnitCurrentSourceBase):
    """
    Ideal current source with static Gaussian mismatch drawn once at construction.

    The actual current is ``current = i_nominal * (1 + ε)``
    where ``ε ~ N(0, mismatch)`` is drawn once and held fixed, modelling
    process-induced static spread (threshold voltage variation, etc.).

    Attributes:
        i_nominal (float):  Designed current (A).
        mismatch (float):   Standard deviation of multiplicative mismatch.
        current (float):    Actual current after mismatch draw.
    """

    def __init__(self, i_nominal: float = 100e-6, mismatch: float = 0.0):
        """
        Args:
            i_nominal: Designed current in amperes (> 0).
            mismatch:  Standard deviation of multiplicative mismatch (≥ 0).
                       A draw ``ε ~ N(0, mismatch)`` scales i_nominal once at
                       construction to give the actual current.

        Raises:
            ValueError: If i_nominal ≤ 0 or mismatch < 0.
        """
        if not isinstance(i_nominal, (int, float)) or i_nominal <= 0:
            raise ValueError("i_nominal must be a positive number")
        if not isinstance(mismatch, (int, float)) or mismatch < 0:
            raise ValueError("mismatch must be >= 0")

        self._i_nominal = float(i_nominal)
        self._mismatch  = float(mismatch)

        if mismatch > 0:
            drawn = self._i_nominal * (1.0 + np.random.normal(0.0, mismatch))
            if drawn < 0.0:
                warnings.warn(
                    f"IdealCurrentSource: mismatch draw produced negative current "
                    f"({drawn:.3e} A); clipping to 0 A. This usually means the "
                    f"mismatch stddev ({mismatch}) is too large relative to unity — "
                    f"a realistic fractional mismatch should be much less than 1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                drawn = 0.0
            self._current = drawn
        else:
            self._current = self._i_nominal

    @property
    def i_nominal(self) -> float:
        return self._i_nominal

    @property
    def mismatch(self) -> float:
        """Standard deviation of mismatch used at construction."""
        return self._mismatch

    @property
    def current(self) -> float:
        return self._current

    def __repr__(self) -> str:
        return (f"IdealCurrentSource(i_nominal={self._i_nominal:.4g}A, "
                f"mismatch={self._mismatch:.4g}, "
                f"current={self._current:.4g}A)")


class CurrentSourceArray:
    """
    Array of current sources for a current-steering DAC.

    Holds two segments that mirror the decoder output:

    Thermometer segment
        n_therm_elements = 2^n_therm_bits − 1 unit sources, each with
        nominal current i_unit.  For a given therm_index k, the first k
        sources are switched to the positive output rail.

    Binary segment
        n_binary_bits sources with binary-weighted nominals.
        ``source[j].i_nominal = 2^j * i_unit`` (j=0 is LSB).
        MSB is at index n_binary_bits − 1.

    All sources are always conducting — they are *steered* between the
    positive and negative output rails, keeping the total current constant
    regardless of code.  This is the fundamental property of a current-
    steering DAC.

    get_current() returns (i_selected, i_total) so that the DAC can compute:
        i_pos = i_selected
        i_neg = i_total − i_selected
        v_diff = (i_pos − i_neg) * r_load   (differential output)

    Attributes:
        therm_sources (List[UnitCurrentSourceBase]):
            Thermometer unit sources, in switching order (index 0 switches on
            first for code = 1).
        binary_sources (List[UnitCurrentSourceBase]):
            Binary-weighted sources, index 0 = LSB weight (1×i_unit),
            index n_binary_bits−1 = MSB weight (2^(n_binary_bits−1)×i_unit).
        i_unit (float):    Nominal unit current (A).
        current_mismatch (float): Mismatch std used at construction.
    """

    def __init__(
        self,
        n_therm_bits: int,
        n_binary_bits: int,
        i_unit: float = 100e-6,
        current_mismatch: float = 0.0,
        source_class: Type[UnitCurrentSourceBase] = IdealCurrentSource,
    ):
        """
        Args:
            n_therm_bits:     Number of thermometer MSBs.  Creates
                              2^n_therm_bits − 1 unit sources.  May be 0.
            n_binary_bits:    Number of binary LSBs.  Creates n_binary_bits
                              binary-weighted sources.  May be 0.
            i_unit:           Nominal unit current (A, > 0).
            current_mismatch: Std of multiplicative mismatch for every source
                              (≥ 0).  Each source draws independently.
            source_class:     UnitCurrentSourceBase subclass to instantiate.
                              Defaults to IdealCurrentSource.

        Raises:
            ValueError: If arguments are invalid or both segments are empty.
        """
        if not isinstance(n_therm_bits, int) or n_therm_bits < 0:
            raise ValueError("n_therm_bits must be a non-negative integer")
        if not isinstance(n_binary_bits, int) or n_binary_bits < 0:
            raise ValueError("n_binary_bits must be a non-negative integer")
        if n_therm_bits == 0 and n_binary_bits == 0:
            raise ValueError("At least one of n_therm_bits or n_binary_bits must be > 0")
        if not isinstance(i_unit, (int, float)) or i_unit <= 0:
            raise ValueError("i_unit must be a positive number")
        if not isinstance(current_mismatch, (int, float)) or current_mismatch < 0:
            raise ValueError("current_mismatch must be >= 0")
        if not (isinstance(source_class, type) and
                issubclass(source_class, UnitCurrentSourceBase)):
            raise TypeError("source_class must be a UnitCurrentSourceBase subclass")

        self._n_therm_bits  = n_therm_bits
        self._n_binary_bits = n_binary_bits
        self._i_unit        = float(i_unit)
        self._current_mismatch = float(current_mismatch)

        n_therm_elements = max(0, 2 ** n_therm_bits - 1)

        # Thermometer unit current must equal 2^n_binary_bits * i_unit so that
        # one thermometer step exactly spans the full binary sub-range, ensuring
        # monotonicity at every segment rollover boundary.
        # When n_binary_bits == 0 (pure thermometer) this reduces to i_unit.
        i_therm_unit = (2 ** n_binary_bits) * self._i_unit

        # Thermometer segment: all unit sources share the same nominal current
        self.therm_sources: List[UnitCurrentSourceBase] = [
            source_class(i_nominal=i_therm_unit, mismatch=current_mismatch)
            for _ in range(n_therm_elements)
        ]

        # Binary segment: source[j] has nominal 2^j * i_unit (j=0 is LSB)
        self.binary_sources: List[UnitCurrentSourceBase] = [
            source_class(i_nominal=(2 ** j) * self._i_unit, mismatch=current_mismatch)
            for j in range(n_binary_bits)
        ]

        # Cache total current (constant — all sources always conduct)
        self._i_total = (sum(s.current for s in self.therm_sources) +
                         sum(s.current for s in self.binary_sources))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_therm_bits(self) -> int:
        """Number of thermometer MSB bits."""
        return self._n_therm_bits

    @property
    def n_binary_bits(self) -> int:
        """Number of binary LSB bits."""
        return self._n_binary_bits

    @property
    def n_bits(self) -> int:
        """Total DAC resolution (n_therm_bits + n_binary_bits)."""
        return self._n_therm_bits + self._n_binary_bits

    @property
    def i_unit(self) -> float:
        """Nominal unit current (A)."""
        return self._i_unit

    @property
    def current_mismatch(self) -> float:
        """Standard deviation of multiplicative mismatch used at construction."""
        return self._current_mismatch

    @property
    def i_total(self) -> float:
        """
        Total current conducted by all sources (A).

        This is constant regardless of code — sources are steered, not
        switched off.
        """
        return self._i_total

    def get_current(
        self,
        therm_index: int,
        binary_bits: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute DAC output current for given decoder control signals.

        Sums the currents of the selected thermometer elements and the
        asserted binary-weighted elements.

        Args:
            therm_index:  Number of thermometer unit elements to steer to the
                          positive output (integer in [0, n_therm_elements],
                          inclusive on both ends).
                          0 = all thermometer elements steered to negative rail;
                          n_therm_elements = all elements steered to positive
                          rail (maximum thermometer code).
            binary_bits:  Bit array for the binary segment, MSB first (length
                          n_binary_bits).  Bit k=1 steers source to positive
                          output; k=0 steers to negative output.

        Returns:
            Tuple[float, float]:
                i_selected — current steered to the positive output rail (A).
                i_total    — total current from all sources (A); constant.

        Raises:
            ValueError: If therm_index or binary_bits are out of range.
        """
        n_therm_elements = len(self.therm_sources)
        # The valid range is [0, n_therm_elements] inclusive.  therm_index == 0
        # means no thermometer elements are active; therm_index == n_therm_elements
        # means all are active (full-scale thermometer code).  Values strictly
        # greater than n_therm_elements are invalid.
        if therm_index < 0 or therm_index > n_therm_elements:
            raise ValueError(
                f"therm_index {therm_index} out of range [0, {n_therm_elements}] inclusive"
            )
        if len(binary_bits) != self._n_binary_bits:
            raise ValueError(
                f"binary_bits length {len(binary_bits)} != n_binary_bits {self._n_binary_bits}"
            )

        # Thermometer: first therm_index sources go to positive rail
        i_therm = sum(s.current for s in self.therm_sources[:therm_index])

        # Binary: MSB is binary_bits[0], LSB is binary_bits[-1]
        # binary_sources[0] = LSB (weight 1), binary_sources[-1] = MSB
        # binary_bits is MSB-first, so reverse for indexing
        i_binary = 0.0
        for bit, src in zip(reversed(binary_bits), self.binary_sources):
            if bit:
                i_binary += src.current

        i_selected = i_therm + i_binary
        return i_selected, self._i_total

    def __repr__(self) -> str:
        n_elem = len(self.therm_sources)
        return (f"CurrentSourceArray("
                f"n_therm_bits={self._n_therm_bits} ({n_elem} elements), "
                f"n_binary_bits={self._n_binary_bits}, "
                f"i_unit={self._i_unit:.4g}A, "
                f"current_mismatch={self._current_mismatch:.4g})")
