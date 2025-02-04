"""
Test cases for SimpleADC class
"""

import unittest
import numpy as np
from pyDataconverter.dataconverter import InputType
from pyDataconverter.architectures.SimpleADC import SimpleADC


class TestSimpleADC(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.n_bits = 12
        self.v_ref = 1.0

    def test_single_ended(self):
        """Test single-ended conversion"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)

        # Test zero
        self.assertEqual(adc.convert(0.0), 0)

        # Test full scale
        self.assertEqual(adc.convert(self.v_ref), 2 ** self.n_bits - 1)

        # Test mid scale
        self.assertEqual(adc.convert(self.v_ref / 2), 2 ** (self.n_bits - 1)-1)

        # Test out of range (should clip)
        self.assertEqual(adc.convert(-0.1), 0)
        self.assertEqual(adc.convert(self.v_ref + 0.1), 2 ** self.n_bits - 1)

    def test_differential(self):
        """Test differential conversion"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL)

        # Test zero differential
        self.assertEqual(adc.convert((0.5, 0.5)), 2 ** (self.n_bits - 1))

        # Test full-scale positive
        self.assertEqual(adc.convert((self.v_ref / 2, 0)), 2 ** self.n_bits - 1)

        # Test full-scale negative
        self.assertEqual(adc.convert((0, self.v_ref / 2)), 0)

        # Test quarter-scale
        expected_quarter = int(2 ** (self.n_bits - 1) * (1 + 0.25))
        self.assertEqual(adc.convert((0.2, 0.075)), expected_quarter)

        # Test out of range (should clip)
        self.assertEqual(adc.convert((1.0, -1.0)), 2 ** self.n_bits - 1)
        self.assertEqual(adc.convert((-1.0, 1.0)), 0)

    def test_input_validation(self):
        """Test input validation"""
        # Single-ended ADC
        adc_se = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        with self.assertRaises(TypeError):
            adc_se.convert((0.5, 0.5))  # Tuple not allowed for single-ended

        # Differential ADC
        adc_diff = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL)
        with self.assertRaises(TypeError):
            adc_diff.convert(0.5)  # Single value not allowed for differential
        with self.assertRaises(TypeError):
            adc_diff.convert((0.5,))  # Must be 2-tuple

    def test_resolution(self):
        """Test different resolutions"""
        test_bits = [8, 12, 16]
        for bits in test_bits:
            adc = SimpleADC(bits, self.v_ref, InputType.SINGLE)
            # Test that max code is correct
            self.assertEqual(adc.convert(self.v_ref), 2 ** bits - 1)


if __name__ == '__main__':
    unittest.main()