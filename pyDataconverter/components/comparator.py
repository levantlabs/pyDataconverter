"""
Comparator Component Module
==========================

This module provides comparator models for use in data converter simulations.

Classes:
    ComparatorBase:          Abstract base class defining the comparator interface.
    DifferentialComparator:  Standard latch comparator with optional reference injection.

First written 2024-02-07; see ``git log`` for the change history.

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
        offset:        DC offset voltage added to the filtered differential signal (V).
        noise_rms:     RMS noise voltage (V).
        bandwidth:     -3 dB bandwidth (Hz); None for infinite bandwidth.
        hysteresis:    Hysteresis window (V), symmetric around the threshold.
        time_constant: Reserved for future temporal modelling (s).
        tau_regen:     Regeneration time constant (s) used by pipelined ADC
                       metastability modelling. Default 0.0 disables the model.
        vc_threshold:  Comparator output-voltage threshold at which the latch
                       is considered resolved. Default 0.5.
    """

    def __init__(self,
                 offset: float = 0.0,
                 noise_rms: float = 0.0,
                 bandwidth: Optional[float] = None,
                 hysteresis: float = 0.0,
                 time_constant: float = 0.0,
                 tau_regen: float = 0.0,
                 vc_threshold: float = 0.5):
        """
        Initialise comparator with specified non-idealities.

        Args:
            offset:        DC offset voltage (V).
            noise_rms:     RMS noise voltage (V).
            bandwidth:     -3 dB bandwidth (Hz); None = infinite.
            hysteresis:    Hysteresis voltage (V).
            time_constant: Time constant for temporal behaviour (s).
            tau_regen:     Regeneration time constant (s) used by pipelined
                           ADC metastability modelling. Default 0.0 disables
                           the model and makes last_regen_time always 0.
            vc_threshold:  Comparator output-voltage threshold at which the
                           latch is considered resolved. Default 0.5 matches
                           the adc_book reference.
        """
        if tau_regen < 0:
            raise ValueError(f"tau_regen must be >= 0, got {tau_regen}")
        if vc_threshold <= 0:
            raise ValueError(f"vc_threshold must be > 0, got {vc_threshold}")

        self.offset        = offset
        self.noise_rms     = noise_rms
        self.bandwidth     = bandwidth
        self.hysteresis    = hysteresis
        self.time_constant = time_constant
        self.tau_regen     = tau_regen
        self.vc_threshold  = vc_threshold
        self._last_output  = 0
        self._last_regen_time = 0.0

        if bandwidth is not None:
            self._filtered_state = 0.0
            self._tau = 1.0 / (2.0 * np.pi * bandwidth)

    @property
    def last_regen_time(self) -> float:
        """
        Regeneration time of the most recent compare() call, in seconds.

        Computed as ``tau_regen * ln(vc_threshold / max(|v_diff|, 1e-30))``.
        Returns 0.0 when ``tau_regen == 0`` (metastability modelling disabled).
        The 1e-30 floor on ``|v_diff|`` prevents ``log(0)`` when the input
        lands exactly on a threshold — an event that should be vanishingly
        rare for any realistic continuous input.

        Note: the returned value can be NEGATIVE when ``|v_diff| > vc_threshold``.
        This is physically meaningful — a large differential input resolves
        before the nominal comparator output threshold, so the "time to
        resolve" is negative relative to the latch clock edge. Callers that
        add ``last_regen_time`` to a timing budget must handle this case
        explicitly; the pipelined ADC does so via its amplifier-settling
        budget calculation, which allows negative budgets on purpose to
        reproduce the reference model's unguarded arithmetic.
        """
        return self._last_regen_time

    def compare(self,
                v_pos: float,
                v_neg: float,
                v_refp: float = 0.0,
                v_refn: float = 0.0,
                time_step: Optional[float] = None) -> int:
        """
        Compare (v_pos − v_refp) against (v_neg − v_refn) with non-idealities.

        The effective input difference is::

            v_diff = (v_pos − v_refp) − (v_neg − v_refn) + self.offset

        followed by bandwidth limiting (if enabled), regeneration-time
        caching (if ``tau_regen > 0``), input-referred noise injection,
        and hysteresis comparison.

        Args:
            v_pos:      Positive signal input.
            v_neg:      Negative signal input.
            v_refp:     Positive reference voltage (default 0).
            v_refn:     Negative reference voltage (default 0).
            time_step:  Time step (s) used by the bandwidth first-order LPF.
                        Required only when ``self.bandwidth is not None``.

        Returns:
            1 if the effective input exceeds the (possibly hysteretic)
            zero threshold, otherwise 0.

        Raises:
            ValueError: If ``self.bandwidth is not None`` and ``time_step``
                is ``None`` or non-positive.
        """
        v_diff = (v_pos - v_refp) - (v_neg - v_refn) + self.offset

        # Bandwidth limiting (first-order low-pass)
        if self.bandwidth is not None:
            if time_step is None:
                raise ValueError("time_step must be provided when bandwidth is specified")
            if time_step <= 0:
                # time_step=0 would make alpha=0 (filter holds indefinitely);
                # negative values yield a nonsensical negative alpha.
                raise ValueError(f"time_step must be positive, got {time_step}")
            alpha  = time_step / (time_step + self._tau)
            v_diff = (1 - alpha) * self._filtered_state + alpha * v_diff
            self._filtered_state = v_diff

        # Record regeneration time before adding noise — regen is a deterministic
        # physical quantity driven by the pre-noise comparator input. Noise is
        # modelled separately as an input-referred effect below.
        if self.tau_regen > 0:
            # 1e-30 floor prevents log(0) when v_diff is exactly on a threshold
            safe_mag = max(abs(v_diff), 1e-30)
            self._last_regen_time = self.tau_regen * float(np.log(self.vc_threshold / safe_mag))
        else:
            self._last_regen_time = 0.0

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
        """Reset hysteresis history, bandwidth filter state, and last regen time."""
        self._last_output = 0
        self._last_regen_time = 0.0
        if self.bandwidth is not None:
            self._filtered_state = 0.0

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
