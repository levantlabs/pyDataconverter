"""
Pipelined ADC Module
====================

Implements a cascaded pipelined ADC that composes any ADCBase as the sub-ADC
and any DACBase as the sub-DAC for each stage. Metastability-to-settling
coupling is preserved via composition: DifferentialComparator owns regen
time, FlashADC aggregates across its bank, ResidueAmplifier owns settling,
and PipelineStage coordinates the timing budget.

Full design: docs/superpowers/specs/2026-04-13-pipelined-adc-design.md

Classes:
    PipelineStage:  One stage of the cascade (helper, not ADCBase).
    PipelinedADC:   Top-level N-stage pipelined ADC (inherits ADCBase) — added in Task 9.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from pyDataconverter.dataconverter import ADCBase, DACBase, InputType
from pyDataconverter.components.residue_amplifier import ResidueAmplifier


class PipelineStage:
    """
    One stage of a pipelined ADC.

    Composes a sub-ADC (any ADCBase), a sub-DAC (any DACBase), and a
    ResidueAmplifier. Performs one stage of the pipelined conversion:

        raw_code = sub_adc.convert(v_sampled)
        v_dac    = sub_dac.convert(raw_code)
        residue  = residue_amp.amplify(
                       target        = gain * (v_sampled - v_dac + offset),
                       initial_error = sign * gain * sub_dac.lsb,
                       t_budget      = 1/(2*fs) - sub_adc.last_conversion_time(),
                   )

    where ``sign`` comes from ``sub_adc.last_metastable_sign()`` and ``gain``
    is ``residue_amp.gain``. The caller (this stage) pre-multiplies target
    and initial_error by gain so the amplify() method just does the
    exponential settling — see docs/superpowers/specs/2026-04-13-pipelined-adc-design.md
    §4.1 and Appendix A for the derivation.

    The stage returns ``(raw_code, shifted_code, residue)`` where
    ``shifted_code = raw_code + code_offset`` (equivalent of the reference's
    per-stage ``minADCcode``).

    Attributes:
        sub_adc:       The ADCBase instance used as this stage's sub-ADC.
        sub_dac:       The DACBase instance used as this stage's sub-DAC.
                       Must be single-ended (OutputType.SINGLE).
        residue_amp:   The ResidueAmplifier producing the amplified residue.
        fs:            Sample rate (Hz) used for the timing budget.
        offset:        Input-referred offset (V).
        code_offset:   Integer added to raw_code before the digital combiner.
                       Does NOT affect the sub-DAC input — the sub-DAC always
                       sees the raw code.
        H:             Weight applied to this stage in the digital combiner,
                       defaulting to ``residue_amp.gain``. Explicit override
                       allowed for trimming/calibration scenarios.
    """

    def __init__(self,
                 sub_adc,
                 sub_dac,
                 residue_amp: ResidueAmplifier,
                 fs: float,
                 offset: float = 0.0,
                 code_offset: int = 0,
                 H: Optional[float] = None):
        if not isinstance(sub_adc, ADCBase):
            raise TypeError(
                f"sub_adc must be an ADCBase instance, got {type(sub_adc).__name__}")
        if not isinstance(sub_dac, DACBase):
            raise TypeError(
                f"sub_dac must be a DACBase instance, got {type(sub_dac).__name__}")
        if not isinstance(residue_amp, ResidueAmplifier):
            raise TypeError(
                f"residue_amp must be a ResidueAmplifier instance, got {type(residue_amp).__name__}")
        if not isinstance(fs, (int, float)):
            raise TypeError(f"fs must be a number, got {type(fs).__name__}")
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")
        if not isinstance(code_offset, int) or isinstance(code_offset, bool):
            raise TypeError(
                f"code_offset must be an integer, got {type(code_offset).__name__}")

        self.sub_adc     = sub_adc
        self.sub_dac     = sub_dac
        self.residue_amp = residue_amp
        self.fs          = float(fs)
        self.offset      = float(offset)
        self.code_offset = code_offset
        self.H           = float(H) if H is not None else float(residue_amp.gain)

    def convert_stage(self, v_sampled: float) -> Tuple[int, int, float]:
        """
        Perform one pipelined-stage conversion.

        Args:
            v_sampled: Analog input to this stage (post first-stage S/H for
                stage 0; the amplified residue from stage i-1 for i > 0).

        Returns:
            (raw_code, shifted_code, v_res):
                raw_code     — the sub-ADC's output (0..n_codes-1).
                shifted_code — raw_code + code_offset, for the digital
                               combiner.
                v_res        — amplified residue, in the same voltage
                               coordinate system as v_sampled, to be fed
                               to the next stage.
        """
        raw_code = int(self.sub_adc.convert(float(v_sampled)))
        v_dac    = self.sub_dac.convert(raw_code)
        # Sub-DAC may return a float or a tuple (differential). We only
        # support single-ended sub-DACs in the stage's subtraction; the
        # output_type of the sub-DAC must be SINGLE.
        delta_v  = float(v_sampled) - float(v_dac) + self.offset
        target   = self.residue_amp.gain * delta_v

        # Metastability coupling (see spec Appendix A)
        t_regen = float(self.sub_adc.last_conversion_time()) \
            if hasattr(self.sub_adc, "last_conversion_time") else 0.0
        sign = int(self.sub_adc.last_metastable_sign()) \
            if hasattr(self.sub_adc, "last_metastable_sign") else 0
        initial_error = sign * self.residue_amp.gain * float(self.sub_dac.lsb)

        t_budget = 1.0 / (2.0 * self.fs) - t_regen  # NOT clamped

        v_res = self.residue_amp.amplify(
            target        = target,
            initial_error = initial_error,
            t_budget      = t_budget,
        )

        shifted_code = raw_code + self.code_offset
        return raw_code, shifted_code, v_res


class PipelinedADC(ADCBase):
    """
    N-stage pipelined ADC with a required backend.

    A cascade of ``PipelineStage`` instances followed by a mandatory backend
    ADC. The first stage applies an optional SARADC-style sample-and-hold
    (``noise_rms``, ``offset``, ``gain_error``, ``t_jitter``) before handing
    the signal to stage 0. Each stage produces a partial code and a residue
    that feeds the next stage. The backend digitises the final residue.

    Digital combiner: ``DOUT = DOUT + DOUT * stage.H + shifted_code`` per
    stage and once more with ``backend_H`` for the backend, bit-exactly
    matching the adc_book reference's accumulation formula.

    Attributes:
        stages:              List of PipelineStage instances.
        backend:             Backend ADC (any ADCBase subclass).
        backend_H:           Digital combiner weight applied to the backend.
        backend_code_offset: Integer added to backend's raw code before
                             combining (analogous to stage.code_offset).
        fs:                  Sample rate (Hz). The default of 1.0 is
                             a placeholder — any configuration with
                             ``tau_regen > 0`` on any comparator or
                             ``settling_tau > 0`` on any residue amp
                             MUST override this to the real sample rate.
                             Leaving the default produces nonsensical
                             timing budgets (500 ms per stage half-period).
        clip_output:         Whether to clip the final DOUT to
                             [0, 2**n_bits - 1].
    """

    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 input_type: InputType = InputType.SINGLE,
                 stages: Optional[List[PipelineStage]] = None,
                 backend: Optional[ADCBase] = None,
                 backend_H: float = 1.0,
                 backend_code_offset: int = 0,
                 fs: float = 1.0,
                 noise_rms: float = 0.0,
                 offset: float = 0.0,
                 gain_error: float = 0.0,
                 t_jitter: float = 0.0,
                 clip_output: bool = True):
        super().__init__(n_bits, v_ref, input_type)

        if stages is None or not isinstance(stages, list) or len(stages) == 0:
            raise ValueError(
                "PipelinedADC requires at least one stage, got " +
                (repr(stages) if not isinstance(stages, list) else "empty list"))
        for i, s in enumerate(stages):
            if not isinstance(s, PipelineStage):
                raise TypeError(
                    f"stages[{i}] must be a PipelineStage instance, got {type(s).__name__}")

        if backend is None or not isinstance(backend, ADCBase):
            raise TypeError(
                f"backend must be an ADCBase instance, got {type(backend).__name__}")

        if not isinstance(backend_H, (int, float)):
            raise TypeError(f"backend_H must be a number, got {type(backend_H).__name__}")
        if backend_H <= 0:
            raise ValueError(f"backend_H must be positive, got {backend_H}")
        if not isinstance(backend_code_offset, int) or isinstance(backend_code_offset, bool):
            raise TypeError(
                f"backend_code_offset must be an integer, got {type(backend_code_offset).__name__}")

        if not isinstance(fs, (int, float)):
            raise TypeError(f"fs must be a number, got {type(fs).__name__}")
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")

        if noise_rms < 0:
            raise ValueError(f"noise_rms must be >= 0, got {noise_rms}")
        if t_jitter < 0:
            raise ValueError(f"t_jitter must be >= 0, got {t_jitter}")

        self.stages              = stages
        self.backend             = backend
        self.backend_H           = float(backend_H)
        self.backend_code_offset = backend_code_offset
        self.fs                  = float(fs)
        self.noise_rms           = float(noise_rms)
        self.offset              = float(offset)
        self.gain_error          = float(gain_error)
        self.t_jitter            = float(t_jitter)
        self.clip_output         = bool(clip_output)

    def _sample_input(self, analog_input) -> float:
        """Apply first-stage S&H non-idealities. Mirrors SARADC._sample_input."""
        if self.input_type == InputType.SINGLE:
            v = float(analog_input)
        else:
            v_pos, v_neg = analog_input
            v = float(v_pos) - float(v_neg)

        if self.gain_error:
            v = v * (1.0 + self.gain_error)
        if self.offset:
            v = v + self.offset
        if self.noise_rms:
            v = v + float(np.random.normal(0.0, self.noise_rms))
        if self.t_jitter and self._dvdt:
            v = v + self._dvdt * float(np.random.normal(0.0, self.t_jitter))
        return v

    def _convert_input(self, analog_input) -> int:
        """Cascade conversion + digital combiner."""
        v_sampled = self._sample_input(analog_input)

        DOUT = 0
        for stage in self.stages:
            raw_code, shifted_code, v_res = stage.convert_stage(v_sampled)
            DOUT = DOUT + DOUT * stage.H + shifted_code
            v_sampled = v_res

        # Backend
        raw_backend = int(self.backend.convert(float(v_sampled)))
        shifted_be  = raw_backend + self.backend_code_offset
        DOUT = DOUT + DOUT * self.backend_H + shifted_be

        if self.clip_output:
            lo, hi = 0, 2 ** self.n_bits - 1
            if DOUT < lo:
                DOUT = lo
            elif DOUT > hi:
                DOUT = hi

        return int(DOUT)

    def __repr__(self) -> str:
        return (f"PipelinedADC(n_bits={self.n_bits}, v_ref={self.v_ref}, "
                f"input_type={self.input_type.name}, stages={len(self.stages)}, "
                f"backend={type(self.backend).__name__}, fs={self.fs:.3e})")
