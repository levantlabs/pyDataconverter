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

    def reset(self):
        """
        Reset the channel rotation state.

        After calling ``reset()``, the next ``convert()`` sample is routed to
        channel 0. ``last_channel`` becomes ``None`` again. Does not reset
        the per-channel sub-ADCs (they manage their own state, e.g. hysteresis
        or bandwidth-filter state, independently).
        """
        self._counter = 0
        self._last_channel = None

    def split_by_channel(self, codes) -> np.ndarray:
        """
        Reshape a 1-D code array into a 2-D per-channel view.

        Args:
            codes: 1-D array of length N·M (must be an integer multiple of
                ``self.M``). Typically the output of ``convert_waveform`` or
                a list of consecutive ``convert()`` return values.

        Returns:
            np.ndarray of shape ``(M, N)``: row k is the subsequence
            ``codes[k::M]``, i.e. the codes produced by channel k in time
            order.

        Raises:
            ValueError: If ``codes`` is not 1-D or ``len(codes) % M != 0``.
        """
        codes = np.asarray(codes)
        if codes.ndim != 1:
            raise ValueError(f"codes must be 1-D, got shape {codes.shape}")
        if len(codes) % self.M != 0:
            raise ValueError(
                f"len(codes)={len(codes)} is not a multiple of M={self.M}; "
                f"pass an array whose length is an integer multiple of the "
                f"channel count.")
        N_per_channel = len(codes) // self.M
        # codes[0::M], codes[1::M], ... via reshape + transpose.
        return codes.reshape(N_per_channel, self.M).T

    def convert_waveform(self, v_dense, t_dense):
        """
        Convert a dense time-domain waveform with per-channel mismatches.

        Overrides ``ADCBase.convert_waveform`` to apply per-channel
        ``scipy.signal.butter`` first-order LPFs when any channel has nonzero
        ``bandwidth``. When bandwidth is zero for every channel, the result
        matches the default (pointwise) path exactly.

        Offset, gain, and timing-skew mismatches are applied as input-referred
        first-order corrections (same arithmetic as the pointwise path).
        Bandwidth mismatch filters the dense waveform per channel BEFORE
        sampling the value at the channel's nominal sample index.

        The channel counter advances by ``len(v_dense)`` over the call, so
        back-to-back invocations pick up where the previous one left off.

        Args:
            v_dense: 1-D numpy array of input voltages. For differential
                input_type, pass the scalar ``v_pos - v_neg``; tuple form
                is not supported for waveforms.
            t_dense: 1-D numpy array of sample times (seconds), same length
                as ``v_dense``. Assumed regularly spaced when any
                ``bandwidth`` is nonzero (used to derive the effective
                sampling rate for the LPF cutoff normalisation).

        Returns:
            np.ndarray[int]: Output codes, same length as ``v_dense``.
        """
        v_dense = np.asarray(v_dense, dtype=float)
        t_dense = np.asarray(t_dense, dtype=float)
        if v_dense.shape != t_dense.shape:
            raise ValueError(
                f"v_dense and t_dense must have the same shape, got "
                f"{v_dense.shape} and {t_dense.shape}")
        if v_dense.ndim != 1:
            raise ValueError(
                f"v_dense and t_dense must be 1-D, got shape {v_dense.shape}")

        N = len(v_dense)
        dvdt_dense = np.gradient(v_dense, t_dense)

        # Build per-channel filtered waveforms if any channel has bandwidth set.
        if np.any(self.bandwidth != 0):
            dt = float(t_dense[1] - t_dense[0])
            fs_dense = 1.0 / dt
            nyquist = fs_dense / 2.0
            v_per_channel = np.empty((self.M, N))
            for k in range(self.M):
                bw_k = float(self.bandwidth[k])
                if bw_k > 0:
                    # First-order Butterworth LPF at bw_k.
                    wn = bw_k / nyquist
                    if wn >= 1.0:
                        # Cutoff at or above Nyquist → effectively pass-through.
                        v_per_channel[k] = v_dense
                    else:
                        b, a = scipy.signal.butter(1, wn, btype='low')
                        v_per_channel[k] = scipy.signal.lfilter(b, a, v_dense)
                else:
                    v_per_channel[k] = v_dense
        else:
            v_per_channel = None  # sentinel: use v_dense directly per sample

        codes = np.empty(N, dtype=int)
        for i in range(N):
            k = (self._counter + i) % self.M
            v_in = v_per_channel[k, i] if v_per_channel is not None else v_dense[i]

            offset_k = float(self.offset[k])
            gain_k   = float(self.gain_error[k])
            skew_k   = float(self.timing_skew[k])
            dvdt_i   = float(dvdt_dense[i])

            v_eff = v_in * (1.0 + gain_k) + offset_k + dvdt_i * skew_k

            if self.input_type == InputType.DIFFERENTIAL:
                sub_input = (v_eff / 2 + self.v_ref / 2,
                             -v_eff / 2 + self.v_ref / 2)
            else:
                sub_input = v_eff

            codes[i] = int(self.channels[k].convert(sub_input, dvdt=dvdt_i))

        self._counter += N
        self._last_channel = (self._counter - 1) % self.M
        return codes

    @classmethod
    def hierarchical(cls,
                     channels_per_level: List[int],
                     sub_adc_template: ADCBase,
                     fs: float,
                     offset_std_per_level: Optional[List[float]] = None,
                     gain_std_per_level: Optional[List[float]] = None,
                     timing_skew_std_per_level: Optional[List[float]] = None,
                     bandwidth_std_per_level: Optional[List[float]] = None,
                     seed: Optional[int] = None) -> "TimeInterleavedADC":
        """
        Build a multi-level interleaving tree from a list of per-level factors.

        ``channels_per_level[0]`` is the OUTERMOST (fastest) interleaving
        factor; ``channels_per_level[-1]`` is the innermost (slowest). The
        resulting tree has one ``TimeInterleavedADC`` per level, nested:
        the innermost level's sub-ADC is ``sub_adc_template``, each outer
        level's sub-ADC is the inner ``TimeInterleavedADC``. Total leaf
        channel count is the product of the per-level factors.

        Per-level mismatch arguments are parallel lists aligned with
        ``channels_per_level``: entry ``[0]`` configures the outermost
        level, entry ``[-1]`` the innermost. Each entry is a scalar stddev
        passed as the corresponding mismatch kwarg to that level's
        ``TimeInterleavedADC``. Missing lists default to zero at every
        level.

        Aggregate sample rate ``fs`` is the top-level rate; each inner
        level sees ``fs / (product of outer factors)``.

        Args:
            channels_per_level: Non-empty list of ints >= 2.
            sub_adc_template: Innermost leaf template (any ADCBase).
            fs: Top-level aggregate sample rate (Hz).
            offset_std_per_level: Optional parallel list of per-level
                offset stddevs (V). Default: all zeros.
            gain_std_per_level: Optional parallel list of per-level
                gain-error stddevs (fractional). Default: all zeros.
            timing_skew_std_per_level: Optional parallel list of per-level
                timing-skew stddevs (seconds). Default: all zeros.
            bandwidth_std_per_level: Optional parallel list of per-level
                bandwidth stddevs (Hz). Default: all zeros.
            seed: Optional RNG seed forwarded to every level's
                TimeInterleavedADC constructor. Each level uses the same
                seed but draws different mismatch values because each
                constructor creates its own Generator instance.

        Returns:
            TimeInterleavedADC: The outermost (top-level) TI-ADC instance.

        Raises:
            ValueError: If channels_per_level is empty, contains entries
                < 2, or any per-level list has the wrong length.
        """
        if not channels_per_level:
            raise ValueError("channels_per_level must have at least one entry")
        for i, m in enumerate(channels_per_level):
            if not isinstance(m, int) or isinstance(m, bool) or m < 2:
                raise ValueError(
                    f"channels_per_level[{i}]={m}, must be integer >= 2")

        L = len(channels_per_level)
        offset_levels = cls._resolve_per_level(
            offset_std_per_level, L, "offset_std_per_level")
        gain_levels = cls._resolve_per_level(
            gain_std_per_level, L, "gain_std_per_level")
        skew_levels = cls._resolve_per_level(
            timing_skew_std_per_level, L, "timing_skew_std_per_level")
        bw_levels = cls._resolve_per_level(
            bandwidth_std_per_level, L, "bandwidth_std_per_level")

        # Walk from innermost outward. fs_at_level is the aggregate rate at
        # the CURRENT level we're constructing; inner levels see slower rates.
        # Compute the inner level's fs by dividing fs by the product of all
        # outer factors.
        outer_product = 1
        for m in channels_per_level[:-1]:
            outer_product *= m

        current_template: ADCBase = sub_adc_template
        # Innermost level first
        for level_index in reversed(range(L)):
            M_level = channels_per_level[level_index]
            fs_at_level = fs / outer_product
            current_template = cls(
                channels=M_level,
                sub_adc_template=current_template,
                fs=fs_at_level,
                offset=offset_levels[level_index],
                gain_error=gain_levels[level_index],
                timing_skew=skew_levels[level_index],
                bandwidth=bw_levels[level_index],
                seed=seed,
            )
            # Moving outward: next iteration builds the level above this one,
            # which sees a FASTER rate. Multiply outer_product back up by the
            # CURRENT level's M (since the level above this one has fewer
            # "outer" factors relative to itself).
            if level_index > 0:
                outer_product //= channels_per_level[level_index - 1]

        return current_template  # outermost TI-ADC

    @staticmethod
    def _resolve_per_level(values, L: int, name: str) -> List[float]:
        """Resolve a per-level stddev argument into a length-L list."""
        if values is None:
            return [0.0] * L
        if len(values) != L:
            raise ValueError(
                f"{name} has length {len(values)}, expected {L}")
        return [float(v) for v in values]

    def __repr__(self) -> str:
        return (f"TimeInterleavedADC(M={self.M}, fs={self.fs:.3e}, "
                f"template={type(self.channels[0]).__name__}, "
                f"n_bits={self.n_bits})")
