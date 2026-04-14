"""
Tests for PipelinedADC and PipelineStage.
"""

import math
import unittest

import numpy as np

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.architectures.PipelinedADC import PipelineStage
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
