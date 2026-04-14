"""
Tests for ADCBase.convert_waveform default implementation.

Exercises the default path on several non-TI ADCBase subclasses.
The default computes dvdt via np.gradient and loops calling
self.convert(v[i], dvdt=dvdt[i]).
"""

import unittest
import numpy as np

from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.dataconverter import InputType


class TestConvertWaveformDefault(unittest.TestCase):
    """ADCBase.convert_waveform default implementation across non-TI subclasses."""

    def test_simple_adc_matches_pointwise_loop(self):
        adc = SimpleADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        t = np.linspace(0, 1e-6, 65)
        v = 0.5 + 0.4 * np.sin(2 * np.pi * 1e6 * t)
        dvdt = np.gradient(v, t)
        expected = np.array([adc.convert(float(v[i]), dvdt=float(dvdt[i]))
                             for i in range(len(v))], dtype=int)
        # Need a fresh ADC because SimpleADC has no hysteresis but this
        # keeps the comparison symmetric for subclasses that do.
        adc2 = SimpleADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        actual = adc2.convert_waveform(v, t)
        np.testing.assert_array_equal(actual, expected)

    def test_flash_adc_matches_pointwise_loop(self):
        adc = FlashADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        t = np.linspace(0, 1e-6, 65)
        v = 0.5 + 0.4 * np.sin(2 * np.pi * 1e6 * t)
        dvdt = np.gradient(v, t)
        expected = np.array([adc.convert(float(v[i]), dvdt=float(dvdt[i]))
                             for i in range(len(v))], dtype=int)
        adc2 = FlashADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        actual = adc2.convert_waveform(v, t)
        np.testing.assert_array_equal(actual, expected)

    def test_sar_adc_matches_pointwise_loop(self):
        adc = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        t = np.linspace(0, 1e-6, 65)
        v = 0.5 + 0.4 * np.sin(2 * np.pi * 1e6 * t)
        dvdt = np.gradient(v, t)
        expected = np.array([adc.convert(float(v[i]), dvdt=float(dvdt[i]))
                             for i in range(len(v))], dtype=int)
        adc2 = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        actual = adc2.convert_waveform(v, t)
        np.testing.assert_array_equal(actual, expected)

    def test_returns_int_array(self):
        adc = SimpleADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        t = np.linspace(0, 1e-6, 16)
        v = 0.5 * np.ones_like(t)
        codes = adc.convert_waveform(v, t)
        self.assertEqual(codes.dtype.kind, "i")
        self.assertEqual(len(codes), 16)

    def test_mismatched_lengths_raise(self):
        adc = SimpleADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        v = np.zeros(16)
        t = np.zeros(17)
        with self.assertRaises(ValueError):
            adc.convert_waveform(v, t)

    def test_non_1d_raises(self):
        adc = SimpleADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        v = np.zeros((4, 4))
        t = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            adc.convert_waveform(v, t)
