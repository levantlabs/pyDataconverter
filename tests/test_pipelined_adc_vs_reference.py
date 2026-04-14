"""
Bit-exact comparison tests for PipelinedADC against the vendored adc_book reference.

Each test configures the same pipelined ADC in both implementations, runs a
dense linear sweep across the input range, and asserts that every output
code matches exactly. If any test fails, the assertion message names the
offending input, both output codes, and the config name — debugging starts
by running the stage-by-stage inspection on that single input.

Bipolar/unipolar reconciliation: the reference uses bipolar sub-ADC
thresholds in [-FSR/2, +FSR/2] and a subDAC with bipolar output levels.
pyDataconverter's FlashADC with InputType.SINGLE defaults to unipolar
[0, v_ref] thresholds and SimpleDAC produces unipolar [0, v_ref] output.
To achieve bit-exactness we pass the reference's exact threshold arrays
to each FlashADC via ArbitraryReference, and use SimpleDAC(offset=-v_ref/2)
to constant-shift the sub-DAC output into the bipolar range that matches
the reference's dacout[code] = code*lsb - v_ref/2 formula. The arithmetic
equivalence was verified manually before this harness was written.

See docs/superpowers/specs/2026-04-13-pipelined-adc-design.md §6 for the
harness design rationale.
"""

import io
import contextlib
import unittest
from typing import Dict, Any, Tuple

import numpy as np

# Import the vendored reference (stdout redirect kept defensively even
# though Task 1 already stripped all print() calls from the vendored copy).
with contextlib.redirect_stdout(io.StringIO()):
    from tests._reference.adc_book_pipelined import (
        PipelinedADC as RefPipelinedADC,
    )

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.architectures.PipelinedADC import (
    PipelineStage,
    PipelinedADC as NewPipelinedADC,
)
from pyDataconverter.components.residue_amplifier import ResidueAmplifier
from pyDataconverter.components.reference import ArbitraryReference
from pyDataconverter.dataconverter import InputType, OutputType


# ------------------------------------------------------------
# Config dictionaries — one per scenario
# ------------------------------------------------------------

def _config_ideal_12bit() -> Dict[str, Any]:
    return dict(name="ideal_12bit",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=4.0, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=False,
                code_errors=None,
                fs=500e6)


def _config_stage0_dac_error() -> Dict[str, Any]:
    return dict(name="stage0_dac_error",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=4.0, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=False,
                code_errors=np.array(
                    [0.0, -0.2, 0.3, 0.05, -0.15, 0.0, 0.3, -0.3, 0.0]
                ) * 0.001,
                fs=500e6)


def _config_stage0_gain_error() -> Dict[str, Any]:
    return dict(name="stage0_gain_error",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=3.988, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=False,
                code_errors=None,
                fs=500e6)


def _config_metastability_canned() -> Dict[str, Any]:
    return dict(name="metastability_canned",
                n_bits=12, v_ref=1.0,
                stage_n_comparators=8, stage_n_levels=9,
                stage_gain=4.0, stage_code_offset=-1,
                backend_n_comparators=1026, backend_H=512,
                time_response=True,
                tauC=30e-12, tauA=50e-12,
                code_errors=None,
                fs=500e6)


# ------------------------------------------------------------
# Reference builder
# ------------------------------------------------------------

def _build_reference(cfg: Dict[str, Any]) -> RefPipelinedADC:
    """Build the vendored reference ADC with the given configuration."""
    with contextlib.redirect_stdout(io.StringIO()):
        if cfg["time_response"]:
            ref = RefPipelinedADC(
                Nstages=2, B=cfg["n_bits"],
                N=[cfg["stage_n_comparators"], cfg["backend_n_comparators"]],
                FSR_ADC=[cfg["v_ref"], cfg["v_ref"]],
                FSR_DAC=[cfg["v_ref"], cfg["v_ref"]],
                G=[cfg["stage_gain"], cfg["backend_H"]],
                minADCcode=[cfg["stage_code_offset"], 0],
                timeResponse=[True, False],
                SampleRate=cfg["fs"],
                tau_comparator=[cfg["tauC"], 0],
                tau_amplifier=[cfg["tauA"], 0],
            )
        else:
            ref = RefPipelinedADC(
                Nstages=2, B=cfg["n_bits"],
                N=[cfg["stage_n_comparators"], cfg["backend_n_comparators"]],
                FSR_ADC=[cfg["v_ref"], cfg["v_ref"]],
                FSR_DAC=[cfg["v_ref"], cfg["v_ref"]],
                G=[cfg["stage_gain"], cfg["backend_H"]],
                minADCcode=[cfg["stage_code_offset"], 0],
            )
        if cfg.get("code_errors") is not None:
            ref.stage[0].subDAC.add_error(np.asarray(cfg["code_errors"]))
    return ref


# ------------------------------------------------------------
# New (pyDataconverter) builder — uses the reference's own threshold
# arrays to guarantee bit-identical bipolar geometry.
# ------------------------------------------------------------

def _build_new(cfg: Dict[str, Any],
               ref: RefPipelinedADC) -> NewPipelinedADC:
    """
    Build the pyDataconverter PipelinedADC with a configuration that
    mirrors the supplied reference exactly.

    The reference's bipolar sub-ADC threshold arrays and sub-DAC LSB
    values are extracted directly from the reference instance and used
    to construct matching pyDataconverter components. This guarantees
    threshold-level bit-identity and avoids re-deriving the reference's
    unusual index-based threshold formula.
    """
    # --- Stage 0 sub-ADC: use the reference's exact threshold array ---
    stage0_thresholds = np.asarray(ref.stage[0].subADC.ref, dtype=float)
    stage0_ref = ArbitraryReference(stage0_thresholds, noise_rms=0.0)

    sub_adc_kwargs = {
        "n_bits": 3, "v_ref": cfg["v_ref"],
        "input_type": InputType.SINGLE,
        "n_comparators": cfg["stage_n_comparators"],
        "reference": stage0_ref,
    }
    if cfg["time_response"]:
        sub_adc_kwargs["comparator_params"] = {
            "tau_regen": cfg["tauC"],
            "vc_threshold": 0.5,
        }
    stage0_sub_adc = FlashADC(**sub_adc_kwargs)

    # --- Stage 0 sub-DAC: bipolar-shift via offset=-v_ref/2 ---
    # Reference's subDAC output: dacout[code] = code * (FSR/N) - FSR/2
    # SimpleDAC(n_levels=N+1, offset=-FSR/2) produces:
    #   code * lsb + (code_errors[code] if provided) + offset
    #   = code * (FSR/N) + code_errors[code] - FSR/2
    # which equals ref_dacout[code] + code_errors[code].
    stage0_sub_dac = SimpleDAC(
        n_bits=3,
        n_levels=cfg["stage_n_levels"],
        v_ref=cfg["v_ref"],
        output_type=OutputType.SINGLE,
        offset=-cfg["v_ref"] / 2.0,
        code_errors=cfg.get("code_errors"),
    )

    # --- Residue amplifier ---
    amp_kwargs = {"gain": cfg["stage_gain"]}
    if cfg["time_response"]:
        amp_kwargs["settling_tau"] = cfg["tauA"]
    else:
        amp_kwargs["settling_tau"] = 0.0
    residue_amp = ResidueAmplifier(**amp_kwargs)

    # --- Pipeline stage 0 ---
    stage = PipelineStage(
        sub_adc=stage0_sub_adc,
        sub_dac=stage0_sub_dac,
        residue_amp=residue_amp,
        fs=cfg["fs"],
        code_offset=cfg["stage_code_offset"],
    )

    # --- Backend sub-ADC: same strategy — copy reference thresholds ---
    backend_thresholds = np.asarray(ref.stage[1].subADC.ref, dtype=float)
    backend_ref = ArbitraryReference(backend_thresholds, noise_rms=0.0)
    backend = FlashADC(
        n_bits=10, v_ref=cfg["v_ref"],
        input_type=InputType.SINGLE,
        n_comparators=cfg["backend_n_comparators"],
        reference=backend_ref,
    )

    return NewPipelinedADC(
        n_bits=cfg["n_bits"],
        v_ref=cfg["v_ref"],
        input_type=InputType.SINGLE,
        stages=[stage],
        backend=backend,
        backend_H=cfg["backend_H"],
        backend_code_offset=0,
        fs=cfg["fs"],
    )


# ------------------------------------------------------------
# The parameterised comparison test
# ------------------------------------------------------------

class TestPipelinedADCAgainstReference(unittest.TestCase):
    """Bit-exact sweep comparison for each canonical configuration."""

    def _run_sweep(self, cfg: Dict[str, Any]):
        ref = _build_reference(cfg)
        new = _build_new(cfg, ref)
        v_sweep = np.linspace(-0.495, 0.495, 4001)
        for v in v_sweep:
            ref_out = int(ref.output(float(v)))
            new_out = int(new.convert(float(v)))
            if ref_out != new_out:
                self.fail(
                    f"[{cfg['name']}] mismatch at v={v:+.6f}: "
                    f"ref={ref_out}, new={new_out}"
                )

    def test_ideal_12bit(self):
        self._run_sweep(_config_ideal_12bit())

    def test_stage0_dac_error(self):
        self._run_sweep(_config_stage0_dac_error())

    def test_stage0_gain_error(self):
        self._run_sweep(_config_stage0_gain_error())

    def test_metastability_canned(self):
        self._run_sweep(_config_metastability_canned())
