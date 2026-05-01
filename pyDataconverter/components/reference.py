"""
Voltage Reference Components
============================

Abstract base and concrete implementations of voltage reference generators
for use in Flash ADC simulations.

Classes:
    ReferenceBase:      Abstract base class defining the reference interface.
    ReferenceLadder:    Uniform resistor-ladder divider with optional mismatch and noise.
    ArbitraryReference: User-supplied threshold array with optional noise.

First written 2026-03-23; see ``git log`` for the change history.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ReferenceBase(ABC):
    """
    Abstract base class for voltage reference generators.

    A reference generator owns a set of threshold voltages used by the
    comparator bank in a Flash ADC.  Static non-idealities (e.g. resistor
    mismatch) are fixed at construction; dynamic noise is redrawn on every
    call to get_voltages().
    """

    @property
    @abstractmethod
    def n_references(self) -> int:
        """Number of reference voltages (= 2^N - 1 for an N-bit Flash ADC)."""
        pass

    @property
    @abstractmethod
    def voltages(self) -> np.ndarray:
        """
        Static (noiseless) reference voltages.
        Useful for plotting the ideal transfer function or inspecting mismatch.
        """
        pass

    @abstractmethod
    def get_voltages(self) -> np.ndarray:
        """
        Return reference voltages for one conversion.

        Dynamic noise (if configured) is redrawn on every call.
        Static mismatch is already baked into the returned array.

        Returns:
            np.ndarray: Reference voltages of length n_references.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_references={self.n_references})"


class ReferenceLadder(ReferenceBase):
    """
    Uniform resistor-ladder voltage reference.

    Generates 2^n_bits - 1 equally-spaced thresholds between v_min and v_max,
    with optional static resistor mismatch and per-sample dynamic noise.

    Attributes:
        n_references (int): Number of reference taps (2^n_bits - 1).
        voltages (np.ndarray): Static thresholds including mismatch (no noise).
        noise_rms (float): RMS dynamic noise applied per conversion (V).
    """

    def __init__(self,
                 n_bits: int,
                 v_min: float,
                 v_max: float,
                 resistor_mismatch: float = 0.0,
                 noise_rms: float = 0.0,
                 seed: Optional[int] = None):
        """
        Args:
            n_bits: ADC resolution; determines the number of taps (2^n_bits - 1).
            v_min: Bottom of the reference range (V).
            v_max: Top of the reference range (V).
            resistor_mismatch: Standard deviation of multiplicative resistor
                mismatch (e.g. 0.01 = 1 %).  Drawn once at construction.
            noise_rms: RMS dynamic noise added to every reference voltage on
                each call to get_voltages() (V).
            seed: Optional integer seed for the construction-time mismatch
                draw.  ``None`` (default) uses OS entropy
                (non-deterministic); an integer makes the mismatch
                realisation reproducible.  Per-call noise (``noise_rms``)
                continues to use ``np.random`` global state.
        """
        if v_max <= v_min:
            raise ValueError("v_max must be greater than v_min")
        if resistor_mismatch < 0:
            raise ValueError("resistor_mismatch must be >= 0")
        if noise_rms < 0:
            raise ValueError("noise_rms must be >= 0")

        self._n_references = 2 ** n_bits - 1
        self.noise_rms = noise_rms
        self.seed = seed

        ideal = np.linspace(v_min, v_max, self._n_references + 2)[1:-1]

        if resistor_mismatch > 0:
            rng = np.random.default_rng(seed)
            mismatch = rng.normal(0, resistor_mismatch, self._n_references)
            self._voltages = ideal * (1.0 + mismatch)
        else:
            self._voltages = ideal.copy()

        # Cached read-only view used by .voltages and the no-noise branch of
        # get_voltages(). Avoids allocating a full copy on every access —
        # meaningful on hot paths for 10+ bit Flash ADCs (hundreds to
        # thousands of elements per conversion).
        self._voltages_ro = self._voltages.view()
        self._voltages_ro.flags.writeable = False

    @property
    def n_references(self) -> int:
        return self._n_references

    @property
    def voltages(self) -> np.ndarray:
        return self._voltages_ro

    def get_voltages(self) -> np.ndarray:
        if self.noise_rms > 0:
            return self._voltages + np.random.normal(0, self.noise_rms,
                                                     self._n_references)
        return self._voltages_ro

    def __repr__(self) -> str:
        parts = [
            f"n_references={self._n_references}",
            f"noise_rms={self.noise_rms}",
        ]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return f"ReferenceLadder({', '.join(parts)})"


class ArbitraryReference(ReferenceBase):
    """
    User-defined voltage reference.

    Accepts an explicit array of threshold voltages, allowing non-uniform
    spacing or any custom reference profile.  Optional dynamic noise can
    be added per conversion.

    Attributes:
        n_references (int): Number of thresholds.
        voltages (np.ndarray): Static threshold array (no noise).
        noise_rms (float): RMS dynamic noise applied per conversion (V).
    """

    def __init__(self,
                 thresholds,
                 noise_rms: float = 0.0):
        """
        Args:
            thresholds: Array-like of reference voltages, must be strictly
                increasing.
            noise_rms: RMS dynamic noise added on each get_voltages() call (V).
        """
        thresholds = np.asarray(thresholds, dtype=float)
        if thresholds.ndim != 1 or len(thresholds) == 0:
            raise ValueError("thresholds must be a non-empty 1-D array")
        if not np.all(np.isfinite(thresholds)):
            raise ValueError("thresholds must all be finite (no NaN or Inf)")
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("thresholds must be strictly increasing")
        if noise_rms < 0:
            raise ValueError("noise_rms must be >= 0")

        self._voltages = thresholds
        self.noise_rms = noise_rms

        # Cached read-only view (see ReferenceLadder for rationale).
        self._voltages_ro = self._voltages.view()
        self._voltages_ro.flags.writeable = False

    @property
    def n_references(self) -> int:
        return len(self._voltages)

    @property
    def voltages(self) -> np.ndarray:
        return self._voltages_ro

    def get_voltages(self) -> np.ndarray:
        if self.noise_rms > 0:
            return self._voltages + np.random.normal(0, self.noise_rms,
                                                     len(self._voltages))
        return self._voltages_ro

    def __repr__(self) -> str:
        return (f"ArbitraryReference(n_references={len(self._voltages)}, "
                f"noise_rms={self.noise_rms})")
