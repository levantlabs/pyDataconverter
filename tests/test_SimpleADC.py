"""
Test cases for SimpleADC class
"""

import unittest
import numpy as np
from pyDataconverter.dataconverter import InputType, QuantizationMode
from pyDataconverter.architectures.SimpleADC import SimpleADC


class TestSimpleADC(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.n_bits = 12
        self.v_ref = 1.0

    # ------------------------------------------------------------------ #
    # FLOOR mode — single-ended                                            #
    # ------------------------------------------------------------------ #

    def test_floor_single_ended(self):
        """FLOOR mode: check zero, mid, full-scale, and clipping"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)

        self.assertEqual(adc.convert(0.0), 0)
        # Mid-scale: floor(0.5 * 2^N) = 2^(N-1) = 2048
        self.assertEqual(adc.convert(self.v_ref / 2), 2 ** (self.n_bits - 1))
        # Full-scale clips to max code
        self.assertEqual(adc.convert(self.v_ref), 2 ** self.n_bits - 1)
        # Out of range clips
        self.assertEqual(adc.convert(-0.1), 0)
        self.assertEqual(adc.convert(self.v_ref + 0.1), 2 ** self.n_bits - 1)

    # ------------------------------------------------------------------ #
    # FLOOR mode — differential                                            #
    # ------------------------------------------------------------------ #

    def test_floor_differential(self):
        """FLOOR mode: check zero diff, full-scale pos/neg, quarter-scale, clipping"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL)

        # Zero differential maps to mid-scale code
        self.assertEqual(adc.convert((0.5, 0.5)), 2 ** (self.n_bits - 1))

        # Full-scale positive clips to max code
        self.assertEqual(adc.convert((self.v_ref / 2, 0)), 2 ** self.n_bits - 1)

        # Full-scale negative gives code 0
        self.assertEqual(adc.convert((0, self.v_ref / 2)), 0)

        # Quarter-scale positive: vdiff=0.125, shifted v=0.625
        # floor(0.625 * 2^N) = floor(2560) = 2560
        expected_quarter = int(2 ** (self.n_bits - 1) * (1 + 0.25))
        self.assertEqual(adc.convert((0.2, 0.075)), expected_quarter)

        # Out of range clips
        self.assertEqual(adc.convert((1.0, -1.0)), 2 ** self.n_bits - 1)
        self.assertEqual(adc.convert((-1.0, 1.0)), 0)

    # ------------------------------------------------------------------ #
    # SYMMETRIC mode — single-ended                                        #
    # ------------------------------------------------------------------ #

    def test_symmetric_single_ended(self):
        """SYMMETRIC mode: check zero, mid, full-scale"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                        quant_mode=QuantizationMode.SYMMETRIC)

        self.assertEqual(adc.convert(0.0), 0)
        # Mid-scale: floor(0.5*(2^N-1) + 0.5) = floor(2048) = 2048
        self.assertEqual(adc.convert(self.v_ref / 2), 2 ** (self.n_bits - 1))
        self.assertEqual(adc.convert(self.v_ref), 2 ** self.n_bits - 1)
        # Out of range clips
        self.assertEqual(adc.convert(-0.1), 0)
        self.assertEqual(adc.convert(self.v_ref + 0.1), 2 ** self.n_bits - 1)

    def test_symmetric_first_transition(self):
        """SYMMETRIC mode: first transition is at 0.5*LSB, not 1*LSB"""
        adc_sym   = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.SYMMETRIC)
        adc_floor = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.FLOOR)

        # SYMMETRIC LSB = v_ref / (2^N - 1); first transition at 0.5 * LSB
        lsb_sym = self.v_ref / (2 ** self.n_bits - 1)
        first_transition = 0.5 * lsb_sym

        # Just below: both modes stay at code 0
        vin_below = first_transition - 1e-9
        self.assertEqual(adc_sym.convert(vin_below), 0)
        self.assertEqual(adc_floor.convert(vin_below), 0)

        # At and just above first transition: SYMMETRIC moves to code 1,
        # FLOOR is still 0 (its first transition is at a full LSB = v_ref/2^N)
        vin_above = first_transition + 1e-9
        self.assertEqual(adc_sym.convert(vin_above), 1)
        self.assertEqual(adc_floor.convert(vin_above), 0)

    def test_symmetric_last_transition(self):
        """SYMMETRIC mode: last transition is at v_ref - 0.5*LSB"""
        adc_sym   = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.SYMMETRIC)
        adc_floor = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.FLOOR)

        lsb_sym = self.v_ref / (2 ** self.n_bits - 1)
        last_transition = self.v_ref - 0.5 * lsb_sym

        # Just below last transition: SYMMETRIC is still at 2^N-2,
        # FLOOR has already reached 2^N-1 (its last transition is earlier)
        vin_below = last_transition - 1e-9
        self.assertEqual(adc_sym.convert(vin_below), 2 ** self.n_bits - 2)

        # At and above last transition: SYMMETRIC reaches max code
        vin_above = last_transition + 1e-9
        self.assertEqual(adc_sym.convert(vin_above), 2 ** self.n_bits - 1)

    # ------------------------------------------------------------------ #
    # SYMMETRIC mode — differential                                        #
    # ------------------------------------------------------------------ #

    def test_symmetric_differential(self):
        """SYMMETRIC mode: check zero diff, full-scale pos/neg"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL,
                        quant_mode=QuantizationMode.SYMMETRIC)

        # Zero differential maps to mid-scale
        self.assertEqual(adc.convert((0.5, 0.5)), 2 ** (self.n_bits - 1))
        # Full-scale positive
        self.assertEqual(adc.convert((self.v_ref / 2, 0)), 2 ** self.n_bits - 1)
        # Full-scale negative
        self.assertEqual(adc.convert((0, self.v_ref / 2)), 0)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def test_input_validation(self):
        """Input type mismatches raise TypeError"""
        adc_se = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        with self.assertRaises(TypeError):
            adc_se.convert((0.5, 0.5))  # Tuple not allowed for single-ended

        adc_diff = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL)
        with self.assertRaises(TypeError):
            adc_diff.convert(0.5)       # Single value not allowed for differential
        with self.assertRaises(TypeError):
            adc_diff.convert((0.5,))    # Must be 2-tuple

    def test_quant_mode_validation(self):
        """Invalid quant_mode raises TypeError"""
        with self.assertRaises(TypeError):
            SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                      quant_mode='floor')

    # ------------------------------------------------------------------ #
    # Resolution and repr                                                  #
    # ------------------------------------------------------------------ #

    def test_resolution(self):
        """Max code is correct across resolutions for both modes"""
        for bits in [8, 12, 16]:
            for mode in QuantizationMode:
                adc = SimpleADC(bits, self.v_ref, InputType.SINGLE, quant_mode=mode)
                self.assertEqual(adc.convert(self.v_ref), 2 ** bits - 1)
                self.assertEqual(adc.convert(0.0), 0)

    def test_repr(self):
        """repr includes quant_mode"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                        quant_mode=QuantizationMode.SYMMETRIC)
        self.assertIn('SYMMETRIC', repr(adc))


if __name__ == '__main__':
    unittest.main()
