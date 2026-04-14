"""
Tests for PipelinedADC and PipelineStage.
"""

import math
import unittest

import numpy as np

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.architectures.PipelinedADC import PipelineStage, PipelinedADC
from pyDataconverter.components.residue_amplifier import ResidueAmplifier
from pyDataconverter.dataconverter import InputType, OutputType


def _make_ideal_stage(n_comparators: int = 8,
                     gain: float = 4.0,
                     fs: float = 1e9,
                     code_offset: int = 0) -> PipelineStage:
    """Helper: build an ideal PipelineStage with a single-ended FlashADC sub-ADC
    and a SimpleDAC sub-DAC sized to match."""
    sub_adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=n_comparators)
    sub_dac = SimpleDAC(n_bits=3, n_levels=n_comparators + 1, v_ref=1.0,
                        output_type=OutputType.SINGLE)
    residue_amp = ResidueAmplifier(gain=gain, settling_tau=0.0)
    return PipelineStage(sub_adc=sub_adc,
                         sub_dac=sub_dac,
                         residue_amp=residue_amp,
                         fs=fs,
                         code_offset=code_offset)


class TestPipelineStageConstruction(unittest.TestCase):
    def test_accepts_valid_components(self):
        stage = _make_ideal_stage()
        self.assertIsInstance(stage.sub_adc, FlashADC)
        self.assertIsInstance(stage.sub_dac, SimpleDAC)
        self.assertIsInstance(stage.residue_amp, ResidueAmplifier)

    def test_sub_adc_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            PipelineStage(sub_adc="not an ADCBase",
                          sub_dac=SimpleDAC(n_bits=3, v_ref=1.0),
                          residue_amp=ResidueAmplifier(gain=2.0),
                          fs=1e9)

    def test_sub_dac_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            PipelineStage(sub_adc=FlashADC(n_bits=3, v_ref=1.0),
                          sub_dac=42,
                          residue_amp=ResidueAmplifier(gain=2.0),
                          fs=1e9)

    def test_residue_amp_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            PipelineStage(sub_adc=FlashADC(n_bits=3, v_ref=1.0),
                          sub_dac=SimpleDAC(n_bits=3, v_ref=1.0),
                          residue_amp="not a residue amp",
                          fs=1e9)

    def test_fs_non_positive_raises(self):
        with self.assertRaises(ValueError):
            PipelineStage(sub_adc=FlashADC(n_bits=3, v_ref=1.0),
                          sub_dac=SimpleDAC(n_bits=3, v_ref=1.0),
                          residue_amp=ResidueAmplifier(gain=2.0),
                          fs=0.0)

    def test_code_offset_non_int_raises(self):
        with self.assertRaises(TypeError):
            _make_ideal_stage(code_offset=1.5)

    def test_h_defaults_to_residue_gain(self):
        stage = _make_ideal_stage(gain=4.0)
        self.assertEqual(stage.H, 4.0)

    def test_h_explicit_override(self):
        sub_adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                           n_comparators=8)
        sub_dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0)
        amp = ResidueAmplifier(gain=4.0)
        stage = PipelineStage(sub_adc=sub_adc, sub_dac=sub_dac,
                              residue_amp=amp, fs=1e9, H=5.0)
        self.assertEqual(stage.H, 5.0)


class TestPipelineStageConvertIdeal(unittest.TestCase):
    def test_residue_is_gain_times_input_minus_dac(self):
        # With settling_tau=0 and tau_regen=0, metastability is disabled and
        # the residue should be exactly gain * (v_in - sub_dac.convert(raw_code)).
        stage = _make_ideal_stage(gain=4.0)
        v_in = 0.3
        raw_code, shifted_code, v_res = stage.convert_stage(v_in)
        expected_residue = 4.0 * (v_in - stage.sub_dac.convert(raw_code))
        self.assertAlmostEqual(v_res, expected_residue)

    def test_code_offset_does_not_affect_sub_dac_input(self):
        # The sub-DAC sees the raw code; the combiner sees the offset code.
        # We verify this by constructing two stages with different code_offsets
        # and confirming their residue is identical for the same input.
        stage0 = _make_ideal_stage(code_offset=0)
        stage1 = _make_ideal_stage(code_offset=-1)
        v_in = 0.42
        _, shifted_0, v_res_0 = stage0.convert_stage(v_in)
        raw_1, shifted_1, v_res_1 = stage1.convert_stage(v_in)
        self.assertAlmostEqual(v_res_0, v_res_1)
        self.assertEqual(shifted_1, raw_1 - 1)


def _make_canonical_pipelined_adc(fs: float = 1e9) -> PipelinedADC:
    """Canonical 12-bit pipelined ADC: [3-bit 9-level stage, 1026-level backend].

    Mirrors the adc_book __main__ example: Nstages=2, N=[8,1026], FSR=[1,1],
    G=[4, 512], minADCcode=[-1, 0]. The pyDataconverter build plugs in
    relaxed FlashADC and SimpleDAC classes for the sub-components and a new
    ResidueAmplifier for the stage gain.
    """
    stage0_sub_adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                              n_comparators=8)
    stage0_sub_dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0,
                               output_type=OutputType.SINGLE)
    stage0_amp = ResidueAmplifier(gain=4.0, settling_tau=0.0)
    stage0 = PipelineStage(sub_adc=stage0_sub_adc,
                           sub_dac=stage0_sub_dac,
                           residue_amp=stage0_amp,
                           fs=fs,
                           code_offset=-1)
    backend = FlashADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                       n_comparators=1026)
    return PipelinedADC(n_bits=12,
                        v_ref=1.0,
                        input_type=InputType.SINGLE,
                        stages=[stage0],
                        backend=backend,
                        backend_H=512,
                        backend_code_offset=0,
                        fs=fs)


class TestPipelinedADCConstruction(unittest.TestCase):
    def test_minimal_valid_construction(self):
        adc = _make_canonical_pipelined_adc()
        self.assertEqual(adc.n_bits, 12)
        self.assertEqual(len(adc.stages), 1)
        self.assertIsInstance(adc.backend, FlashADC)

    def test_empty_stages_raises(self):
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(ValueError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[], backend=backend, backend_H=1, fs=1e9)

    def test_stages_wrong_element_type_raises(self):
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(TypeError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=["not a stage"], backend=backend,
                         backend_H=1, fs=1e9)

    def test_backend_wrong_type_raises(self):
        stage = _make_ideal_stage()
        with self.assertRaises(TypeError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[stage], backend="not an ADC",
                         backend_H=1, fs=1e9)

    def test_backend_h_non_positive_raises(self):
        stage = _make_ideal_stage()
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(ValueError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[stage], backend=backend,
                         backend_H=0.0, fs=1e9)

    def test_fs_non_positive_raises(self):
        stage = _make_ideal_stage()
        backend = FlashADC(n_bits=10, v_ref=1.0, n_comparators=1024)
        with self.assertRaises(ValueError):
            PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                         stages=[stage], backend=backend,
                         backend_H=1, fs=0.0)


class TestPipelinedADCConvert(unittest.TestCase):
    def test_midscale_input_gives_midscale_code(self):
        adc = _make_canonical_pipelined_adc()
        # With unipolar FlashADC/SimpleDAC (SINGLE, v_ref=1.0) and code_offset=-1
        # on stage0, the effective midscale output (~2048) is reached at v≈0.6,
        # not v=0.5. This offset arises because code_offset=-1 shifts stage0's
        # digital code by -1, moving the midscale input up by one LSB bin.
        code = adc.convert(0.6)
        # Ideal 12-bit midscale is ~2048; allow +/-5 LSB tolerance
        self.assertAlmostEqual(code, 2048, delta=5)

    def test_monotonic_sweep_produces_monotonic_codes(self):
        adc = _make_canonical_pipelined_adc()
        sweep = np.linspace(-0.45, 0.45, 201)
        codes = np.array([adc.convert(float(v)) for v in sweep], dtype=int)
        # Non-strictly monotonic (repeats allowed; reversals not)
        diffs = np.diff(codes)
        self.assertTrue(np.all(diffs >= 0),
                        f"Non-monotonic codes detected: diffs={diffs[diffs<0]}")

    def test_clip_output_true_saturates(self):
        adc = _make_canonical_pipelined_adc()
        # Far negative input should clip to 0
        self.assertEqual(adc.convert(-10.0), 0)
        # Far positive input should clip to 2**12 - 1
        self.assertEqual(adc.convert(+10.0), 2**12 - 1)

    def test_returns_int(self):
        adc = _make_canonical_pipelined_adc()
        result = adc.convert(0.25)
        self.assertIsInstance(result, int)
