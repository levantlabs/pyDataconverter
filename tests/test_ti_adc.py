"""
Tests for TimeInterleavedADC.

Construction validation in this file; pointwise conversion tests,
split_by_channel, and composition tests land in later commits.
"""

import unittest
import numpy as np

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType


def _make_template(n_bits=6, input_type=InputType.SINGLE):
    return FlashADC(n_bits=n_bits, v_ref=1.0, input_type=input_type)


class TestTIADCConstruction(unittest.TestCase):
    """Basic constructor validation and attribute inheritance from template."""

    def test_minimal_valid_construction(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        self.assertEqual(ti.M, 4)
        self.assertEqual(ti.n_bits, 6)
        self.assertEqual(ti.v_ref, 1.0)
        self.assertEqual(ti.input_type, InputType.SINGLE)
        self.assertEqual(ti.fs, 1e9)
        self.assertEqual(len(ti.channels), 4)
        # Mismatch arrays default to length-M zero arrays
        np.testing.assert_array_equal(ti.offset, np.zeros(4))
        np.testing.assert_array_equal(ti.gain_error, np.zeros(4))
        np.testing.assert_array_equal(ti.timing_skew, np.zeros(4))
        np.testing.assert_array_equal(ti.bandwidth, np.zeros(4))

    def test_channels_less_than_two_raises(self):
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=1, sub_adc_template=_make_template(), fs=1e9)
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=0, sub_adc_template=_make_template(), fs=1e9)

    def test_channels_not_int_raises(self):
        with self.assertRaises(TypeError):
            TimeInterleavedADC(channels=4.0, sub_adc_template=_make_template(), fs=1e9)

    def test_template_not_adcbase_raises(self):
        with self.assertRaises(TypeError):
            TimeInterleavedADC(channels=4, sub_adc_template="not an ADCBase", fs=1e9)

    def test_fs_non_positive_raises(self):
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=0.0)
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=-1e9)

    def test_scalar_offset_interpreted_as_stddev(self):
        # Scalar mismatch => random draw of length M with std approximately equal
        # to the scalar value (for reasonable seeds; just check the shape and
        # that nonzero values are present).
        ti = TimeInterleavedADC(channels=64, sub_adc_template=_make_template(),
                                fs=1e9, offset=1e-3, seed=42)
        self.assertEqual(ti.offset.shape, (64,))
        # Not all zero
        self.assertGreater(np.std(ti.offset), 0.0)

    def test_explicit_array_offset(self):
        explicit = np.array([1e-3, -1e-3, 0.5e-3, -0.5e-3])
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(),
                                fs=1e9, offset=explicit)
        np.testing.assert_array_equal(ti.offset, explicit)

    def test_explicit_array_wrong_length_raises(self):
        explicit = np.array([1e-3, -1e-3, 0.5e-3])  # length 3, M=4
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=4, sub_adc_template=_make_template(),
                               fs=1e9, offset=explicit)

    def test_same_seed_gives_same_mismatch(self):
        ti1 = TimeInterleavedADC(channels=8, sub_adc_template=_make_template(),
                                 fs=1e9, offset=1e-3, gain_error=0.001, seed=123)
        ti2 = TimeInterleavedADC(channels=8, sub_adc_template=_make_template(),
                                 fs=1e9, offset=1e-3, gain_error=0.001, seed=123)
        np.testing.assert_array_equal(ti1.offset, ti2.offset)
        np.testing.assert_array_equal(ti1.gain_error, ti2.gain_error)

    def test_forbidden_kwargs_raise(self):
        # n_bits, v_ref, input_type must be inherited from the template, not
        # overridden at the TI-ADC level. Catching these at construction
        # prevents silent mismatches between wrapper and template.
        for kw in ("n_bits", "v_ref", "input_type"):
            with self.assertRaises(TypeError, msg=f"{kw} must raise"):
                TimeInterleavedADC(
                    channels=4,
                    sub_adc_template=_make_template(),
                    fs=1e9,
                    **{kw: 8 if kw == "n_bits" else (1.0 if kw == "v_ref" else InputType.SINGLE)},
                )

    def test_channels_are_deep_copies(self):
        # Each channel should be an independent copy so mutating one
        # (e.g. changing an attribute) does not affect the others.
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        self.assertEqual(len(ti.channels), 4)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertIsNot(ti.channels[i], ti.channels[j])
        # Not the same object as the template either
        tmpl = _make_template()
        ti2 = TimeInterleavedADC(channels=4, sub_adc_template=tmpl, fs=1e9)
        for ch in ti2.channels:
            self.assertIsNot(ch, tmpl)
