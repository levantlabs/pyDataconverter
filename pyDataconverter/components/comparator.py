"""
Comparator Component Module
==========================

This module provides comparator models for use in data converter simulations.

Classes:
    ComparatorBase:          Abstract base class defining the comparator interface.
    DifferentialComparator:  Standard latch comparator with optional reference injection.

Version History:
---------------
1.0.0 (2024-02-07):
    - Initial release (Comparator class)
1.0.1 (2024-02-07):
    - Added time constant parameter for future temporal modeling
1.1.0 (2026-03-23):
    - Introduced ComparatorBase ABC
    - Renamed Comparator → DifferentialComparator
    - Extended compare() to 4-input signature:
      compare(v_pos, v_neg, v_refp=0.0, v_refn=0.0)
      effective_diff = (v_pos − v_refp) − (v_neg − v_refn)
    - Backward-compatible: calling compare(v_pos, v_neg) gives same result as before

Notes:
------
The 4-input compare() signature allows the Flash ADC to pass signal and reference
rails separately. When v_refp=v_refn=0 (defaults), the comparator behaves as a
pure differential comparator — identical to the original 2-input interface.

Future variants (e.g. PreamplifiedComparator) subclass DifferentialComparator and
override compare() while inheriting all non-ideality parameters.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ComparatorBase(ABC):
    """
    Abstract base class for all comparator models.

    Defines the interface that the Flash ADC (and any other architecture)
    uses to interact with a comparator. All subclasses must implement
    compare() with the 4-input signature.
    """

    @abstractmethod
    def compare(self,
                v_pos: float,
                v_neg: float,
                v_refp: float = 0.0,
                v_refn: float = 0.0,
                time_step: Optional[float] = None) -> int:
        """
        Compare two differential inputs against an optional differential reference.

        The decision is based on:
            effective_diff = (v_pos − v_refp) − (v_neg − v_refn)

        When v_refp = v_refn = 0 (defaults) this reduces to v_pos − v_neg,
        giving the same result as the original 2-input interface.

        Args:
            v_pos:      Positive signal input.
            v_neg:      Negative signal input.
            v_refp:     Positive reference voltage (default 0).
            v_refn:     Negative reference voltage (default 0).
            time_step:  Time step for bandwidth calculations (required only when
                        bandwidth is set).

        Returns:
            1 if effective_diff > threshold (accounting for non-idealities), else 0.
        """

    @abstractmethod
    def reset(self):
        """Reset internal state (hysteresis history, bandwidth filter)."""


class DifferentialComparator(ComparatorBase):
    """
    Differential latch comparator with optional parallel reference injection.

    The effective input difference is:
        effective_diff = (v_pos − v_refp) − (v_neg − v_refn) + offset

    With default reference voltages (v_refp = v_refn = 0) this is identical
    to the classic v_pos − v_neg interface.

    The Flash ADC uses this by passing the signal and reference rails separately:
        compare(v_pos, v_neg, v_refp=ref_tap_pos, v_refn=ref_tap_neg)

    A different comparator model (e.g. PreamplifiedComparator) can be substituted
    at FlashADC construction time via the comparator_type argument.

    Attributes:
        offset:        DC input-referred offset voltage (V).
        noise_rms:     RMS input-referred noise voltage (V).
        bandwidth:     -3 dB bandwidth (Hz); None for infinite bandwidth.
        hysteresis:    Hysteresis window (V), symmetric around the threshold.
        time_constant: Reserved for future temporal modelling (s).
    """

    def __init__(self,
                 offset: float = 0.0,
                 noise_rms: float = 0.0,
                 bandwidth: Optional[float] = None,
                 hysteresis: float = 0.0,
                 time_constant: float = 0.0):
        """
        Initialise comparator with specified non-idealities.

        Args:
            offset:        DC offset voltage (V).
            noise_rms:     RMS noise voltage (V).
            bandwidth:     -3 dB bandwidth (Hz); None = infinite.
            hysteresis:    Hysteresis voltage (V).
            time_constant: Time constant for temporal behaviour (s).
        """
        self.offset        = offset
        self.noise_rms     = noise_rms
        self.bandwidth     = bandwidth
        self.hysteresis    = hysteresis
        self.time_constant = time_constant
        self._last_output  = 0

        if bandwidth is not None:
            self._last_input = 0.0
            self._tau = 1.0 / (2.0 * np.pi * bandwidth)

    def compare(self,
                v_pos: float,
                v_neg: float,
                v_refp: float = 0.0,
                v_refn: float = 0.0,
                time_step: Optional[float] = None) -> int:
        """
        Compare (v_pos − v_refp) against (v_neg − v_refn) with non-idealities.

        Args:
            v_pos:      Positive signal input.
            v_neg:      Negative signal input.
            v_refp:     Positive reference voltage (default 0).
            v_refn:     Negative reference voltage (default 0).
            time_step:  Time step for bandwidth calculations.

        Returns:
            1 if effective input > threshold, else 0.

        Raises:
            ValueError: If bandwidth is set but time_step is None.
        """
        v_diff = (v_pos - v_refp) - (v_neg - v_refn) + self.offset

        # Bandwidth limiting (first-order low-pass)
        if self.bandwidth is not None:
            if time_step is None:
                raise ValueError("time_step must be provided when bandwidth is specified")
            alpha  = time_step / (time_step + self._tau)
            v_diff = (1 - alpha) * self._last_input + alpha * v_diff
            self._last_input = v_diff

        # Input-referred noise
        if self.noise_rms > 0:
            v_diff += np.random.normal(0, self.noise_rms)

        # Hysteresis
        if self.hysteresis > 0:
            threshold = -self.hysteresis / 2 if self._last_output == 1 else self.hysteresis / 2
            result = 1 if v_diff > threshold else 0
        else:
            result = 1 if v_diff > 0 else 0

        self._last_output = result
        return result

    def reset(self):
        """Reset hysteresis history and bandwidth filter state."""
        self._last_output = 0
        if self.bandwidth is not None:
            self._last_input = 0.0

    def __repr__(self) -> str:
        params = [f"offset={self.offset:.2e}V",
                  f"noise_rms={self.noise_rms:.2e}V",
                  f"hysteresis={self.hysteresis:.2e}V"]
        if self.bandwidth is not None:
            params.append(f"bandwidth={self.bandwidth:.2e}Hz")
        if self.time_constant > 0:
            params.append(f"time_constant={self.time_constant:.2e}s")
        return f"DifferentialComparator({', '.join(params)})"


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
Comparator = DifferentialComparator


# ---------------------------------------------------------------------------
# Example / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    comp = DifferentialComparator(
        offset=0.001,
        noise_rms=0.0005,
        hysteresis=0.002,
    )
    print(comp)

    v_diff = np.linspace(-0.01, 0.01, 1000)
    n_trials = 500
    results = []
    for _ in range(n_trials):
        comp.reset()
        results.append([comp.compare(v, 0.0) for v in v_diff])

    prob_high = np.mean(results, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(v_diff * 1e3, prob_high)
    plt.axvline(comp.offset * 1e3, color='r', linestyle='--',
                label=f'Offset: {comp.offset*1e3:.1f} mV')
    plt.xlabel('Differential Input (mV)')
    plt.ylabel('P(output = 1)')
    plt.title('DifferentialComparator Transfer Characteristic')
    plt.legend()
    plt.grid(True)
    plt.show()
