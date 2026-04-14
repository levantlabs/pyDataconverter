"""
Time-Interleaved ADC Module
===========================

Composable time-interleaved ADC. Wraps a sub-ADC template as M identical
channels, injects per-channel offset/gain/timing-skew/bandwidth mismatches,
and produces a single interleaved output stream whose spectrum reproduces
the standard TI-ADC mismatch spurs.

Inherits from ADCBase so it can be used as a sub-component inside other
architectures (e.g. as a PipelinedADC backend or inside another
TimeInterleavedADC for hierarchical interleaving).

Full design rationale: docs/superpowers/specs/2026-04-14-ti-adc-design.md

Classes:
    TimeInterleavedADC: N-channel TI-ADC wrapping any ADCBase as its template.
"""

import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.signal

from pyDataconverter.dataconverter import ADCBase, InputType


_FORBIDDEN_KWARGS = ("n_bits", "v_ref", "input_type")


def _resolve_mismatch(value, M: int, rng: np.random.Generator, name: str) -> np.ndarray:
    """
    Resolve a mismatch argument into a length-M numpy array.

    - None or 0 or 0.0 → length-M zero array (mismatch disabled).
    - scalar float → interpreted as stddev; draw M values from N(0, value).
    - 1-D array of length M → used as explicit per-channel values.
    - 1-D array of wrong length → raise ValueError naming the observed length.
    """
    if value is None:
        return np.zeros(M, dtype=float)

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        scalar = float(arr)
        if scalar == 0.0:
            return np.zeros(M, dtype=float)
        # Scalar = stddev. Draw from N(0, scalar).
        return rng.normal(loc=0.0, scale=scalar, size=M)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or 1-D array, got shape {arr.shape}")
    if len(arr) != M:
        raise ValueError(
            f"{name} array length {len(arr)} does not match channels={M}")
    return arr


class TimeInterleavedADC(ADCBase):
    """
    M-channel time-interleaved ADC with per-channel mismatch modelling.

    Every channel is a deep copy of the ``sub_adc_template`` at construction.
    The class maintains a stateful channel counter that advances by one on
    each ``convert()`` call, routing samples to ``channels[counter % M]``.
    Per-channel mismatches are applied as input-referred first-order
    corrections before the channel's sub-ADC is invoked.

    Bandwidth mismatch cannot be modelled pointwise (it is a convolution).
    To inject bandwidth mismatch, use ``convert_waveform(v_dense, t_dense)``,
    which applies per-channel ``scipy.signal.butter`` LPFs before sampling.
    Calling ``convert()`` on a TI-ADC with any nonzero bandwidth raises
    ``RuntimeError`` with a clear redirect.

    Attributes:
        M (int): Number of channels.
        channels (list[ADCBase]): The M deep-copied sub-ADC instances.
        offset (np.ndarray): Length-M per-channel input-referred offsets (V).
        gain_error (np.ndarray): Length-M per-channel fractional gain errors.
        timing_skew (np.ndarray): Length-M per-channel sampling clock phase
            errors (seconds).
        bandwidth (np.ndarray): Length-M per-channel front-end LPF cutoffs
            (Hz). Zero means the channel is wideband (no LPF applied).
        fs (float): Aggregate sample rate (Hz). Per-channel rate is fs / M.
    """

    def __init__(self,
                 channels: int,
                 sub_adc_template: ADCBase,
                 fs: float,
                 offset=None,
                 gain_error=None,
                 timing_skew=None,
                 bandwidth=None,
                 seed: Optional[int] = None,
                 **forbidden):
        # Forbid n_bits / v_ref / input_type — they're inherited from the template.
        for bad in _FORBIDDEN_KWARGS:
            if bad in forbidden:
                raise TypeError(
                    f"{bad} must not be passed to TimeInterleavedADC; "
                    f"it is inherited from sub_adc_template.{bad}")
        if forbidden:
            # Any OTHER unexpected kwargs also raise — no silent swallowing.
            unknown = ", ".join(sorted(forbidden.keys()))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")

        if not isinstance(channels, int) or isinstance(channels, bool):
            raise TypeError(
                f"channels must be an integer, got {type(channels).__name__}")
        if channels < 2:
            raise ValueError(f"channels must be >= 2, got {channels}")

        if not isinstance(sub_adc_template, ADCBase):
            raise TypeError(
                f"sub_adc_template must be an ADCBase instance, got "
                f"{type(sub_adc_template).__name__}")

        if not isinstance(fs, (int, float)) or isinstance(fs, bool):
            raise TypeError(f"fs must be a number, got {type(fs).__name__}")
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")

        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")

        # Inherit n_bits / v_ref / input_type from the template
        super().__init__(
            sub_adc_template.n_bits,
            sub_adc_template.v_ref,
            sub_adc_template.input_type,
        )
        self.M = channels
        self.fs = float(fs)

        rng = np.random.default_rng(seed)
        self.offset      = _resolve_mismatch(offset,      self.M, rng, "offset")
        self.gain_error  = _resolve_mismatch(gain_error,  self.M, rng, "gain_error")
        self.timing_skew = _resolve_mismatch(timing_skew, self.M, rng, "timing_skew")
        self.bandwidth   = _resolve_mismatch(bandwidth,   self.M, rng, "bandwidth")

        # Deep-copy the template M times so each channel is an independent object.
        self.channels: List[ADCBase] = [
            copy.deepcopy(sub_adc_template) for _ in range(self.M)
        ]

        # Stateful channel counter — advances on each convert() call.
        self._counter = 0
        self._last_channel: Optional[int] = None

    @property
    def last_channel(self) -> Optional[int]:
        """Index of the most recently used channel, or None before any convert() call."""
        return self._last_channel

    def _convert_input(self, analog_input):
        """
        Pointwise per-channel conversion.

        Applies input-referred offset, gain, and timing-skew corrections for
        the current channel, routes to that channel's sub-ADC, and advances
        the channel counter.

        Bandwidth mismatch is not representable pointwise (it is a
        convolution) — if any channel has nonzero bandwidth, this method
        raises RuntimeError and directs the caller to use convert_waveform.
        """
        if np.any(self.bandwidth != 0):
            raise RuntimeError(
                "bandwidth mismatch requires convert_waveform(); use that "
                "method or disable bandwidth to use the pointwise convert().")

        k = self._counter % self.M

        # Resolve the effective scalar input. For differential input_type the
        # TI-ADC applies all corrections to the scalar v_diff = v_pos - v_neg,
        # then repackages the result for the sub-ADC.
        if self.input_type == InputType.DIFFERENTIAL:
            v_pos, v_neg = analog_input
            v = float(v_pos) - float(v_neg)
        else:
            v = float(analog_input)

        offset_k = float(self.offset[k])
        gain_k   = float(self.gain_error[k])
        skew_k   = float(self.timing_skew[k])

        # First-order per-channel input-referred correction.
        v_eff = v * (1.0 + gain_k) + offset_k + self._dvdt * skew_k

        # Repackage for the sub-ADC's input_type.
        if self.input_type == InputType.DIFFERENTIAL:
            sub_input = (v_eff / 2 + self.v_ref / 2,
                         -v_eff / 2 + self.v_ref / 2)
        else:
            sub_input = v_eff

        raw_code = int(self.channels[k].convert(sub_input, dvdt=self._dvdt))

        self._last_channel = k
        self._counter += 1
        return raw_code

    def __repr__(self) -> str:
        return (f"TimeInterleavedADC(M={self.M}, fs={self.fs:.3e}, "
                f"template={type(self.channels[0]).__name__}, "
                f"n_bits={self.n_bits})")
