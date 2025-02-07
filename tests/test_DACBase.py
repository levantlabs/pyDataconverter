"""Test cases for DACBase class"""

import unittest
from typing import Union, Tuple
from pyDataconverter.dataconverter import DACBase, OutputType


class TestDACBase(unittest.TestCase):
    """Test cases for DACBase class"""

    def setUp(self):
        """Create a concrete test class since DACBase is abstract"""

        class TestDAC(DACBase):
            def _convert_input(self, digital_input: int) -> Union[float, Tuple[float, float]]:
                # Simple linear conversion
                voltage = digital_input * self.lsb
                if self.output_type == OutputType.SINGLE:
                    return voltage
                else:  # DIFFERENTIAL
                    v_diff = voltage - self.v_ref / 2
                    return (v_diff / 2 + self.v_ref / 2, -v_diff / 2 + self.v_ref / 2)

        self.DACClass = TestDAC

    def test_initialization(self):
        """Test valid and invalid initialization parameters"""
        # Valid initialization
        dac = self.DACClass(n_bits=12, v_ref=1.0)
        self.assertEqual(dac.n_bits, 12)
        self.assertEqual(dac.v_ref, 1.0)
        self.assertEqual(dac.output_type, OutputType.SINGLE)

        # Invalid n_bits
        with self.assertRaises(TypeError):
            self.DACClass(n_bits=12.5)
        with self.assertRaises(ValueError):
            self.DACClass(n_bits=0)
        with self.assertRaises(ValueError):
            self.DACClass(n_bits=33)

        # Invalid v_ref
        with self.assertRaises(ValueError):
            self.DACClass(n_bits=12, v_ref=-1.0)
        with self.assertRaises(TypeError):
            self.DACClass(n_bits=12, v_ref="1.0")

    def test_output_types(self):
        """Test single-ended and differential outputs"""
        # Single-ended
        dac_se = self.DACClass(12, output_type=OutputType.SINGLE)
        output = dac_se.convert(2048)  # Mid-scale for 12 bits
        self.assertIsInstance(output, float)
        self.assertAlmostEqual(output, 0.5)  # Should be v_ref/2

        # Differential
        dac_diff = self.DACClass(12, output_type=OutputType.DIFFERENTIAL)
        output = dac_diff.convert(2048)  # Mid-scale
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        self.assertAlmostEqual(output[0] - output[1], 0.0)  # Should be 0V differential

    def test_input_validation(self):
        """Test input code validation"""
        dac = self.DACClass(12)

        # Valid inputs
        dac.convert(0)  # Min
        dac.convert(4095)  # Max for 12 bits

        # Invalid inputs
        with self.assertRaises(ValueError):
            dac.convert(-1)
        with self.assertRaises(ValueError):
            dac.convert(4096)
        with self.assertRaises(TypeError):
            dac.convert(1.5)

    def test_lsb_calculation(self):
        """Test LSB calculation"""
        dac = self.DACClass(12, v_ref=1.0)
        expected_lsb = 1.0 / (2 ** 12)
        self.assertAlmostEqual(dac.lsb, expected_lsb)

    def test_string_representation(self):
        """Test string representation"""
        dac = self.DACClass(12, v_ref=1.0, output_type=OutputType.DIFFERENTIAL)
        expected_str = "TestDAC(n_bits=12, v_ref=1.0, output_type=DIFFERENTIAL)"
        self.assertEqual(str(dac), expected_str)

    def test_full_scale_output(self):
        """Test full scale output values"""
        dac = self.DACClass(12, v_ref=1.0)

        # Test minimum (code 0)
        min_output = dac.convert(0)
        self.assertAlmostEqual(min_output, 0.0)

        # Test maximum (code 2^N - 1)
        max_output = dac.convert(4095)  # 2^12 - 1
        self.assertAlmostEqual(max_output, 1.0)

    def test_differential_range(self):
        """Test differential output range"""
        dac = self.DACClass(12, v_ref=1.0, output_type=OutputType.DIFFERENTIAL)

        # Test mid-scale (should give 0V differential)
        mid_output = dac.convert(2048)
        v_pos, v_neg = mid_output
        self.assertAlmostEqual(v_pos - v_neg, 0.0)

        # Test full-scale (should give maximum differential voltage)
        max_output = dac.convert(4095)
        v_pos, v_neg = max_output
        self.assertAlmostEqual(v_pos - v_neg, 0.5)

        # Test minimum (should give minimum differential voltage)
        min_output = dac.convert(0)
        v_pos, v_neg = min_output
        self.assertAlmostEqual(v_pos - v_neg, -1.0)


if __name__ == '__main__':
    unittest.main()