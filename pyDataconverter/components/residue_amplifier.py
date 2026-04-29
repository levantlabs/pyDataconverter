"""
Residue Amplifier Component
===========================

Models the closed-loop residue amplifier used in pipelined ADC stages.
Captures finite gain, offset, slew rate, exponential settling driven by a
caller-supplied time budget, and optional output-swing saturation.

The amplification contract is designed to bit-exactly reproduce the
metastability-to-settling coupling from the adc_book reference
(see docs/superpowers/specs/2026-04-13-pipelined-adc-design.md, Appendix A):

    v_out = target + initial_error * exp(-t_budget / settling_tau) + offset

where `target` is the "ideal" residue the amp would settle to given infinite
time and `initial_error` is the signed perturbation the amp starts at
(typically a sub-DAC-LSB-scale correction driven by which sub-ADC
comparator was slowest to resolve). Edge cases — settling_tau=0, infinite
t_budget, zero initial_error — are handled explicitly to avoid NaN from
IEEE 754 corner cases.

Classes:
    ResidueAmplifier: Configurable residue amplifier for pipelined stages.
"""

import math
from typing import Optional, Tuple

import numpy as np


class ResidueAmplifier:
    """
    Closed-loop residue amplifier with configurable non-idealities.

    Gain contract (read this before calling ``amplify``):
        ``self.gain`` is *not* applied inside ``amplify()``.  The caller
        is expected to pre-multiply by ``self.gain`` when constructing
        the ``target`` and ``initial_error`` arguments — for example:

            ra = ResidueAmplifier(gain=4.0, settling_tau=...)
            v_out = ra.amplify(
                target        = ra.gain * (v_in - v_dac),
                initial_error = sign * ra.gain * sub_dac.lsb,
                t_budget      = t_budget,
            )

        ``amplify()`` then performs only the exponential-settling
        evolution and offset/clipping; multiplying inside it would
        double-apply the gain.  The attribute is exposed so callers
        can read it when composing those expressions
        (e.g. ``stage.residue_amp.gain``).

    Attributes:
        gain:          Closed-loop voltage gain (signed; inverting amps OK).
                       NOT applied inside ``amplify()`` — see the gain
                       contract above; the caller pre-multiplies.
        offset:        Output-referred DC offset voltage (V).
        slew_rate:     **NOT YET IMPLEMENTED.**  The parameter is accepted,
                       validated, stored, and shown in ``repr()`` for API
                       compatibility, but ``amplify()`` does not currently
                       apply slew limiting — the output settles purely
                       exponentially regardless of the requested rate.
                       Phase 1 deliberately omitted slew limiting because
                       the reference implementation we calibrate against
                       (see ``docs/superpowers/specs/2026-04-13-pipelined-adc-design.md``
                       Appendix A) has no slew model to compare against.
                       Pass any non-negative value (default ``+inf``) — it
                       has no effect on output.  Slew-limited amplification
                       is tracked for a future Phase.
        settling_tau:  First-order settling time constant (s). 0 means an
                       instantaneous ideal amp — no initial-condition decay.
        output_swing:  Optional (v_min, v_max) clipping bounds. None = no
                       clipping.
    """

    def __init__(self,
                 gain: float,
                 offset: float = 0.0,
                 slew_rate: float = float('inf'),
                 settling_tau: float = 0.0,
                 output_swing: Optional[Tuple[float, float]] = None):
        if not isinstance(gain, (int, float)):
            raise TypeError(f"gain must be a number, got {type(gain).__name__}")
        if gain == 0:
            raise ValueError("gain must be nonzero")
        if not isinstance(offset, (int, float)):
            raise TypeError(f"offset must be a number, got {type(offset).__name__}")
        if slew_rate < 0:
            raise ValueError(f"slew_rate must be non-negative, got {slew_rate}")
        if settling_tau < 0:
            raise ValueError(f"settling_tau must be non-negative, got {settling_tau}")
        if output_swing is not None:
            v_min, v_max = output_swing
            if v_max <= v_min:
                raise ValueError(
                    f"output_swing v_max must exceed v_min, got ({v_min}, {v_max})")

        self.gain         = float(gain)
        self.offset       = float(offset)
        self.slew_rate    = float(slew_rate)
        self.settling_tau = float(settling_tau)
        self.output_swing = output_swing

    def amplify(self,
                target: float,
                initial_error: float,
                t_budget: float) -> float:
        """
        Apply exponential-settling amplification on a pre-scaled target.

        The caller (typically a PipelineStage) is responsible for
        multiplying by the amp's gain when building ``target`` and
        ``initial_error``. This method does NOT re-multiply by
        ``self.gain``; the attribute exists so callers can read it
        (e.g. ``stage.residue_amp.gain``) when scaling their own values.

        Args:
            target:        Pre-scaled ideal residue the amp would reach
                           given infinite settling time. Typically built
                           by the caller as ``self.gain * (v_in - v_dac)``.
            initial_error: Pre-scaled signed perturbation the amp starts
                           at, in residue-output units. Typically built
                           by the caller as ``sign * self.gain * sub_dac.lsb``.
                           The amp exponentially decays this toward zero
                           over ``t_budget``.
            t_budget:      Seconds of settling time available. NOT clamped —
                           negative values produce ``exp(+positive) > 1``,
                           matching the reference's unguarded behaviour at
                           TR < 0.

        Returns:
            Amplified residue voltage, clipped to ``output_swing`` if set.

        .. note::
           ``slew_rate`` (set at construction) is **not yet honoured by
           this method** — see the class docstring for details.  Settling
           is purely exponential at ``settling_tau``; the amp is treated
           as if it has infinite slew capability regardless of the
           configured ``slew_rate``.

        Edge cases (handled explicitly to avoid NaN):
            - ``settling_tau == 0``: return ``target + offset`` regardless
              of ``initial_error`` or ``t_budget`` (instantaneous ideal amp).
            - ``initial_error == 0``: short-circuit to ``target + offset``.
            - ``t_budget == +inf``: full settling, return ``target + offset``.
        """
        # Ideal-amp / no-error / infinite-time degenerate cases short-circuit
        # to avoid IEEE 754 0*inf and exp(-t/0) NaN traps.
        if self.settling_tau == 0 or initial_error == 0 or (math.isinf(t_budget) and t_budget > 0):
            v_out = target + self.offset
        else:
            decay = math.exp(-t_budget / self.settling_tau)
            v_out = target + initial_error * decay + self.offset

        if self.output_swing is not None:
            v_min, v_max = self.output_swing
            if v_out > v_max:
                v_out = v_max
            elif v_out < v_min:
                v_out = v_min

        return v_out

    def __repr__(self) -> str:
        parts = [f"gain={self.gain}"]
        if self.offset:
            parts.append(f"offset={self.offset}")
        if self.settling_tau:
            parts.append(f"settling_tau={self.settling_tau:.3e}")
        if not math.isinf(self.slew_rate):
            parts.append(f"slew_rate={self.slew_rate:.3e}")
        if self.output_swing is not None:
            parts.append(f"output_swing={self.output_swing}")
        return f"ResidueAmplifier({', '.join(parts)})"
