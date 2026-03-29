"""
Capacitive DAC (C-DAC) Components
==================================

Abstract base and concrete implementations of capacitive DAC models for use
in SAR ADC simulations.

Classes:
    CDACBase:          Abstract base class defining the C-DAC interface.
    SingleEndedCDAC:   Binary-weighted C-DAC for single-ended SAR ADCs.
    DifferentialCDAC:  Binary-weighted C-DAC with complementary arrays for
                       differential SAR ADCs.

Version History:
---------------
1.0.0 (2026-03-25):
    - Initial release
1.1.0 (2026-03-25):
    - Capacitor arrays now hold UnitCapacitorBase instances (IdealCapacitor by
      default) rather than bare numpy weight arrays.  Public API is unchanged;
      cap_instances / cap_instances_neg properties added for per-element access.

Notes:
------
The C-DAC is the core of every SAR ADC.  It generates the trial voltages that
the comparator evaluates during each bit cycle.

get_voltage(code) always returns a (v_refp, v_refn) pair that maps directly
onto the DifferentialComparator 4-input signature:

    compare(v_signal, 0.0, v_refp, v_refn)
    effective_diff = (v_signal − v_refp) − (0 − v_refn)
                   = v_signal − (v_refp − v_refn)

SingleEndedCDAC returns (v_dac, 0.0) so v_refn = 0 (ground reference).
DifferentialCDAC returns (v_dacp, v_dacn) with both rails actively driven.

Capacitor mismatch is modelled as a multiplicative Gaussian error drawn once
per capacitor at construction and held fixed, producing static DNL/INL.  Each
UnitCapacitorBase instance owns its own mismatch draw, making it possible to
inspect, replace, or extend individual elements without touching the CDAC or
SAR ADC code.

DifferentialCDAC draws independent mismatch for the positive and negative
arrays.  Correlated (identical) mismatch would cancel in the differential
output, producing no nonlinearity; independent mismatch is required to model
realistic differential errors.

Termination cap:
    Both classes include a fixed termination capacitor of 1.0 unit (equal to
    the ideal LSB cap).  For binary-weighted caps this gives:
        cap_total = (2^N − 1) + 1 = 2^N
    and the ideal DAC output is code / 2^N * v_ref (FLOOR quantisation).
    When custom cap_weights are supplied the user should scale them so that
    the LSB capacitor is ≈ 1.0 unit to preserve this convention.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type
import numpy as np

from pyDataconverter.components.capacitor import UnitCapacitorBase, IdealCapacitor


class CDACBase(ABC):
    """
    Abstract base class for capacitive DAC models.

    A C-DAC owns a set of capacitor weights and converts a digital code to a
    pair of reference voltages (v_refp, v_refn) that the comparator evaluates
    against the held input signal.
    """

    @property
    @abstractmethod
    def n_bits(self) -> int:
        """ADC resolution; number of bit capacitors."""

    @property
    @abstractmethod
    def v_ref(self) -> float:
        """Reference voltage (V)."""

    @property
    @abstractmethod
    def cap_weights(self) -> np.ndarray:
        """
        Actual capacitor weights (with mismatch applied), MSB first.
        Length == n_bits.
        """

    @property
    @abstractmethod
    def cap_total(self) -> float:
        """
        Total effective capacitance including the termination cap.
        For the positive array of a DifferentialCDAC.
        """

    @abstractmethod
    def get_voltage(self, code: int) -> Tuple[float, float]:
        """
        Return (v_refp, v_refn) for the given code.

        The pair is passed directly to the comparator's reference inputs:
            compare(v_signal, 0.0, v_refp, v_refn)
        The effective decision threshold is v_refp − v_refn.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            Tuple[float, float]: (v_refp, v_refn).
        """

    @property
    def voltages(self) -> np.ndarray:
        """
        Effective output voltage (v_refp − v_refn) for every code.

        Returns:
            np.ndarray: Shape (2^n_bits,).  Entry k is the DAC threshold for
            code k, i.e. the value the SAR compares the signal against.
        """
        n_codes = 2 ** self.n_bits
        return np.array([vp - vn
                         for vp, vn in (self.get_voltage(c) for c in range(n_codes))])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_bits={self.n_bits}, v_ref={self.v_ref})"


class SingleEndedCDAC(CDACBase):
    """
    Binary-weighted capacitive DAC for single-ended SAR ADCs.

    The positive reference output (v_refp) is the weighted sum of selected
    capacitors normalised to v_ref.  The negative reference output (v_refn)
    is always 0 V (ground).

    Ideal output (no mismatch):
        v_dac = code / 2^n_bits * v_ref
    Output range: [0, v_ref − LSB] where LSB = v_ref / 2^n_bits.

    Attributes:
        cap_weights (np.ndarray): Actual capacitor values (with mismatch),
            MSB first, in units of C_unit.
        cap_total (float): sum(cap_weights) + termination cap (1.0).
        cap_mismatch (float): Std of multiplicative mismatch used at init.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        cap_weights: Optional[np.ndarray] = None,
        cap_mismatch: float = 0.0,
        cap_class: Type[UnitCapacitorBase] = IdealCapacitor,
    ):
        """
        Args:
            n_bits: Resolution in bits.
            v_ref: Reference voltage (V).
            cap_weights: Nominal capacitor weights, MSB first.  Length must
                equal n_bits and all values must be positive.  If None, binary
                weights [2^(N-1), …, 2, 1] are used.
            cap_mismatch: Standard deviation of multiplicative capacitor
                mismatch (dimensionless, e.g. 0.001 = 0.1 %).  Each capacitor
                instance draws its own mismatch independently at construction.
            cap_class: UnitCapacitorBase subclass to instantiate for each
                capacitor in the array.  Defaults to IdealCapacitor.
        """
        if not isinstance(n_bits, int) or n_bits < 1 or n_bits > 32:
            raise ValueError("n_bits must be an integer in [1, 32]")
        if not isinstance(v_ref, (int, float)) or v_ref <= 0:
            raise ValueError("v_ref must be a positive number")
        if cap_mismatch < 0:
            raise ValueError("cap_mismatch must be >= 0")

        self._n_bits = n_bits
        self._v_ref  = v_ref
        self.cap_mismatch = cap_mismatch

        intended = self._resolve_weights(n_bits, cap_weights)

        self._cap_instances: List[UnitCapacitorBase] = [
            cap_class(c_nominal=float(w), mismatch=cap_mismatch)
            for w in intended
        ]
        self._cap_weights = np.array([c.capacitance for c in self._cap_instances])
        self._cap_total   = float(np.sum(self._cap_weights) + 1.0)

    # ------------------------------------------------------------------
    # CDACBase interface
    # ------------------------------------------------------------------

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def v_ref(self) -> float:
        return self._v_ref

    @property
    def cap_weights(self) -> np.ndarray:
        return self._cap_weights.copy()

    @property
    def cap_total(self) -> float:
        return self._cap_total

    @property
    def cap_instances(self) -> List[UnitCapacitorBase]:
        """Individual capacitor objects, MSB first.  Length == n_bits."""
        return list(self._cap_instances)

    def get_voltage(self, code: int) -> Tuple[float, float]:
        """
        Return (v_dac, 0.0) for the given code.

        v_dac = dot(bits, cap_weights) / cap_total * v_ref

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            Tuple[float, float]: (v_dac, 0.0).
        """
        bits  = self._code_to_bits(code)
        v_dac = float(np.dot(bits, self._cap_weights) / self._cap_total * self._v_ref)
        return (v_dac, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weights(n_bits: int, cap_weights) -> np.ndarray:
        if cap_weights is None:
            return np.array([2 ** (n_bits - 1 - k) for k in range(n_bits)],
                            dtype=float)
        w = np.asarray(cap_weights, dtype=float)
        if w.ndim != 1 or len(w) != n_bits:
            raise ValueError(
                f"cap_weights must be a 1-D array of length n_bits={n_bits}, "
                f"got shape {w.shape}")
        if not np.all(w > 0):
            raise ValueError("All cap_weights must be positive")
        return w

    def _code_to_bits(self, code: int) -> np.ndarray:
        return np.array(
            [(code >> (self._n_bits - 1 - k)) & 1 for k in range(self._n_bits)],
            dtype=float,
        )

    def __repr__(self) -> str:
        return (f"SingleEndedCDAC(n_bits={self._n_bits}, v_ref={self._v_ref}, "
                f"cap_mismatch={self.cap_mismatch})")


class DifferentialCDAC(CDACBase):
    """
    Binary-weighted capacitive DAC with complementary arrays for differential
    SAR ADCs.

    Both a positive and a negative capacitor array are modelled.  They switch
    in a complementary fashion:
        bit = 1 → positive cap connects to v_ref/2, negative cap to 0 V
        bit = 0 → positive cap connects to 0 V,     negative cap to v_ref/2

    Ideal outputs:
        v_dacp = code / 2^n_bits * v_ref / 2
        v_dacn = (1 − code / 2^n_bits) * v_ref / 2
        v_dac_diff = v_dacp − v_dacn = (2·code / 2^n_bits − 1) · v_ref / 2

    Output range of v_dac_diff: [−v_ref/2, ≈ +v_ref/2].

    The positive and negative arrays receive **independent** mismatch draws.
    Correlated (identical) mismatch cancels in the differential output and
    produces no nonlinearity; independent mismatch is needed to model
    realistic differential errors.

    Attributes:
        cap_weights (np.ndarray):     Actual positive-side cap values (with
                                      mismatch), MSB first.
        cap_weights_neg (np.ndarray): Actual negative-side cap values (with
                                      mismatch), MSB first.
        cap_total (float):            Positive-side total capacitance.
        cap_total_neg (float):        Negative-side total capacitance.
        cap_mismatch (float):         Mismatch std used at construction.
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        cap_weights: Optional[np.ndarray] = None,
        cap_mismatch: float = 0.0,
        cap_class: Type[UnitCapacitorBase] = IdealCapacitor,
    ):
        """
        Args:
            n_bits: Resolution in bits.
            v_ref: Reference voltage (V).
            cap_weights: Nominal capacitor weights (applied to both arrays
                before mismatch), MSB first, length n_bits, all positive.
                If None, binary weights [2^(N-1), …, 2, 1] are used.
            cap_mismatch: Standard deviation of multiplicative mismatch
                (dimensionless).  Positive and negative arrays receive
                independent mismatch draws per capacitor.
            cap_class: UnitCapacitorBase subclass to instantiate for each
                capacitor in both arrays.  Defaults to IdealCapacitor.
        """
        if not isinstance(n_bits, int) or n_bits < 1 or n_bits > 32:
            raise ValueError("n_bits must be an integer in [1, 32]")
        if not isinstance(v_ref, (int, float)) or v_ref <= 0:
            raise ValueError("v_ref must be a positive number")
        if cap_mismatch < 0:
            raise ValueError("cap_mismatch must be >= 0")

        self._n_bits = n_bits
        self._v_ref  = v_ref
        self.cap_mismatch = cap_mismatch

        intended = SingleEndedCDAC._resolve_weights(n_bits, cap_weights)

        self._cap_instances_pos: List[UnitCapacitorBase] = [
            cap_class(c_nominal=float(w), mismatch=cap_mismatch) for w in intended
        ]
        self._cap_instances_neg: List[UnitCapacitorBase] = [
            cap_class(c_nominal=float(w), mismatch=cap_mismatch) for w in intended
        ]

        self._cap_weights_pos = np.array([c.capacitance for c in self._cap_instances_pos])
        self._cap_weights_neg = np.array([c.capacitance for c in self._cap_instances_neg])

        self._cap_total_pos = float(np.sum(self._cap_weights_pos) + 1.0)
        self._cap_total_neg = float(np.sum(self._cap_weights_neg) + 1.0)

    # ------------------------------------------------------------------
    # CDACBase interface
    # ------------------------------------------------------------------

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def v_ref(self) -> float:
        return self._v_ref

    @property
    def cap_weights(self) -> np.ndarray:
        """Positive-side capacitor weights (with mismatch), MSB first."""
        return self._cap_weights_pos.copy()

    @property
    def cap_weights_neg(self) -> np.ndarray:
        """Negative-side capacitor weights (with mismatch), MSB first."""
        return self._cap_weights_neg.copy()

    @property
    def cap_total(self) -> float:
        """Positive-side total capacitance (including termination cap)."""
        return self._cap_total_pos

    @property
    def cap_total_neg(self) -> float:
        """Negative-side total capacitance (including termination cap)."""
        return self._cap_total_neg

    @property
    def cap_instances(self) -> List[UnitCapacitorBase]:
        """Positive-side capacitor objects, MSB first.  Length == n_bits."""
        return list(self._cap_instances_pos)

    @property
    def cap_instances_neg(self) -> List[UnitCapacitorBase]:
        """Negative-side capacitor objects, MSB first.  Length == n_bits."""
        return list(self._cap_instances_neg)

    def get_voltage(self, code: int) -> Tuple[float, float]:
        """
        Return (v_dacp, v_dacn) for the given code.

        v_dacp = dot(bits, cap_weights_pos) / cap_total_pos * v_ref / 2
        v_dacn = (cap_total_neg − dot(bits, cap_weights_neg)) / cap_total_neg * v_ref / 2

        The effective threshold passed to the comparator is v_dacp − v_dacn.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            Tuple[float, float]: (v_dacp, v_dacn).
        """
        bits   = self._code_to_bits(code)
        half   = self._v_ref / 2.0
        v_dacp = float(np.dot(bits, self._cap_weights_pos) / self._cap_total_pos * half)
        v_dacn = float((self._cap_total_neg - np.dot(bits, self._cap_weights_neg))
                       / self._cap_total_neg * half)
        return (v_dacp, v_dacn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _code_to_bits(self, code: int) -> np.ndarray:
        return np.array(
            [(code >> (self._n_bits - 1 - k)) & 1 for k in range(self._n_bits)],
            dtype=float,
        )

    def __repr__(self) -> str:
        return (f"DifferentialCDAC(n_bits={self._n_bits}, v_ref={self._v_ref}, "
                f"cap_mismatch={self.cap_mismatch})")
