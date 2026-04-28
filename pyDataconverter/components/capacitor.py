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

First written 2026-03-25; see ``git log`` for the change history.
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

    def redraw_mismatch(self,
                        stddev: float,
                        rng: np.random.Generator) -> None:
        """
        Re-draw mismatch from the stored nominal value.

        The nominal value (``c_nominal``) is preserved; a fresh multiplicative
        Gaussian ε ~ N(0, stddev) is generated and the effective
        ``capacitance`` is updated to ``c_nominal * (1 + ε)``.  Used by CDAC
        ``apply_mismatch()`` to refresh a statistical draw without rebuilding
        the topology.

        Subclasses that do not support in-place re-draw (e.g. voltage-
        dependent or time-varying models) should override with a
        ``NotImplementedError`` and document their limitation.

        Args:
            stddev: New mismatch standard deviation (>= 0).
            rng: NumPy random Generator used to draw ε.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.redraw_mismatch is not implemented; "
            f"subclasses must override to support in-place mismatch re-draw.")

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
            drawn = self._c_nominal * (1.0 + np.random.normal(0.0, mismatch))
            if drawn < 0.0:
                import warnings
                warnings.warn(
                    f"IdealCapacitor: mismatch draw produced negative capacitance "
                    f"({drawn:.3e} F); clipping to 0 F. This usually means the "
                    f"mismatch stddev ({mismatch}) is too large relative to unity — "
                    f"a realistic fractional mismatch should be much less than 1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                drawn = 0.0
            self._capacitance = drawn
        else:
            self._capacitance = self._c_nominal

    @property
    def c_nominal(self) -> float:
        return self._c_nominal

    @property
    def mismatch(self) -> float:
        """Standard deviation of mismatch used at the latest draw."""
        return self._mismatch

    @property
    def capacitance(self) -> float:
        return self._capacitance

    def redraw_mismatch(self,
                        stddev: float,
                        rng: np.random.Generator) -> None:
        """
        Re-draw ε ~ N(0, stddev) on the preserved nominal value.

        The new effective capacitance is ``c_nominal * (1 + ε)``.  The
        prior realization is discarded; ``c_nominal`` is unchanged so the
        topology is preserved across re-draws.  ``stddev=0`` restores the
        ideal value.
        """
        if not isinstance(stddev, (int, float)) or stddev < 0:
            raise ValueError(f"stddev must be >= 0, got {stddev}")
        self._mismatch = float(stddev)
        if stddev == 0:
            self._capacitance = self._c_nominal
            return
        drawn = self._c_nominal * (1.0 + rng.normal(0.0, stddev))
        if drawn < 0.0:
            import warnings
            warnings.warn(
                f"IdealCapacitor.redraw_mismatch: draw produced negative "
                f"capacitance ({drawn:.3e} F); clipping to 0 F. Mismatch "
                f"stddev ({stddev}) is too large — a realistic fractional "
                f"mismatch should be much less than 1.",
                RuntimeWarning,
                stacklevel=2,
            )
            drawn = 0.0
        self._capacitance = drawn

    def __repr__(self) -> str:
        return (f"IdealCapacitor(c_nominal={self._c_nominal:.4g}, "
                f"mismatch={self._mismatch:.4g}, "
                f"capacitance={self._capacitance:.4g})")
