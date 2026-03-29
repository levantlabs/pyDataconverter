"""
Unit Capacitor Components
=========================

Unit capacitor models for use in capacitive DAC (C-DAC) arrays.  Each
instance represents a single physical capacitor; arrays of these objects
are held by the C-DAC to allow per-element mismatch, and future subclasses
can add behaviour (leakage, voltage-dependence) without touching the array
or the DAC.

Classes:
    UnitCapacitorBase:  Abstract base defining the capacitor interface.
    IdealCapacitor:     Fixed capacitance with static Gaussian mismatch.

Version History:
---------------
1.0.0 (2026-03-25):
    - Initial release
"""

from abc import ABC, abstractmethod
import numpy as np


class UnitCapacitorBase(ABC):
    """
    Abstract base class for unit capacitor models.

    Defines the interface used by C-DAC arrays.  Subclasses model
    different physical effects (leakage, voltage dependence, etc.)
    while keeping the array and DAC code unchanged.

    All subclasses must expose:
        c_nominal   — the designed (target) capacitance.
        capacitance — the actual capacitance used in simulation
                      (includes any static or dynamic non-idealities).
    """

    @property
    @abstractmethod
    def c_nominal(self) -> float:
        """Designed (nominal) capacitance in farads or normalised units."""

    @property
    @abstractmethod
    def capacitance(self) -> float:
        """
        Actual capacitance used in simulation.

        For ideal capacitors this equals c_nominal * (1 + mismatch_draw).
        Subclasses may compute this dynamically (e.g. voltage-dependent Cgg).
        """

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"c_nominal={self.c_nominal:.4g}, "
                f"capacitance={self.capacitance:.4g})")


class IdealCapacitor(UnitCapacitorBase):
    """
    Ideal capacitor with static Gaussian mismatch drawn once at construction.

    The actual capacitance is:
        capacitance = c_nominal * (1 + ε)
    where ε ~ N(0, mismatch) is drawn once and held fixed for the lifetime
    of the object, modelling process-induced static spread.

    Attributes:
        c_nominal (float):   Designed capacitance.
        mismatch (float):    Standard deviation of multiplicative mismatch
                             (dimensionless, e.g. 0.01 = 1 %).
        capacitance (float): Actual capacitance after mismatch draw.
    """

    def __init__(self, c_nominal: float = 1.0, mismatch: float = 0.0):
        """
        Args:
            c_nominal: Designed capacitance (> 0).  May be in any consistent
                       unit (F, fF, or normalised to a unit cap).
            mismatch:  Standard deviation of multiplicative mismatch (≥ 0).
                       A draw ε ~ N(0, mismatch) is applied once at init:
                           actual = c_nominal * (1 + ε)

        Raises:
            ValueError: If c_nominal ≤ 0 or mismatch < 0.
        """
        if not isinstance(c_nominal, (int, float)) or c_nominal <= 0:
            raise ValueError("c_nominal must be a positive number")
        if not isinstance(mismatch, (int, float)) or mismatch < 0:
            raise ValueError("mismatch must be >= 0")

        self._c_nominal = float(c_nominal)
        self._mismatch  = float(mismatch)

        if mismatch > 0:
            self._capacitance = max(0.0, self._c_nominal * (1.0 + np.random.normal(0.0, mismatch)))
        else:
            self._capacitance = self._c_nominal

    @property
    def c_nominal(self) -> float:
        return self._c_nominal

    @property
    def mismatch(self) -> float:
        """Standard deviation of mismatch used at construction."""
        return self._mismatch

    @property
    def capacitance(self) -> float:
        return self._capacitance

    def __repr__(self) -> str:
        return (f"IdealCapacitor(c_nominal={self._c_nominal:.4g}, "
                f"mismatch={self._mismatch:.4g}, "
                f"capacitance={self._capacitance:.4g})")
