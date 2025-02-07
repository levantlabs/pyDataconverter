"""
Test cases for SimpleDAC class
"""

import unittest
import numpy as np
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType


class TestSimpleDAC(unittest.TestCase):
    """Test cases for SimpleDAC class"""

    def setUp(self):
        """Set up test cases"""
        self.n_bits = 12
        self.v_ref = 1.0

    def test_single_ended(self):
        """Test single-ended conversion"""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)

        # Test zero scale
        self.assertEqual(dac.convert(0), 0)

        # Test mid scale
        mid_code = 2 ** (self.n_bits - 1)
        self.assertAlmostEqual(dac.convert(mid_code), self.v_ref / 2)

        # Test full scale
        max_code = 2 ** self.n_bits - 1
        self.assertAlmostEqual(dac.convert(max_code), self.v_ref)

        # Test quarter scale
        quarter_code = 2 ** (self.n_bits - 2)
        self.assertAlmostEqual(dac.convert(quarter_code), self.v_ref / 4)

    def test_differential(self):
        """Test differential conversion"""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)

        # Test mid scale (should give 0V differential)
        mid_code = 2 ** (self.n_bits - 1)
        v_pos, v_neg = dac.convert(mid_code)
        self.assertAlmostEqual(v_pos - v_neg, 0.0)
        self.assertAlmostEqual(v_pos, self.v_ref / 2)
        self.assertAlmostEqual(v_neg, self.v_ref / 2)

        # Test full scale
        max_code = 2 ** self.n_bits - 1
        v_pos, v_neg = dac.convert(max_code)
        self.assertAlmostEqual(v_pos - v_neg, self.v_ref)

        # Test zero scale
        v_pos, v_neg = dac.convert(0)
        self.assertAlmostEqual(v_pos - v_neg, -self.v_ref)

        # Test quarter scale
        quarter_code = 2 ** (self.n_bits - 2)
        v_pos, v_neg = dac.convert(quarter_code)
        self.assertAlmostEqual(v_pos - v_neg, -self.v_ref / 2)

    def test_lsb_steps(self):
        """Test LSB-sized steps"""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)

        # Test consecutive codes
        code = 1000  # Arbitrary starting point
        v1 = dac.convert(code)
        v2 = dac.convert(code + 1)

        # Check step size is one LSB
        self.assertAlmostEqual(v2 - v1, dac.lsb)

    def test_monotonicity(self):
        """Test DAC is monotonic"""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)

        # Test over a range of codes
        codes = range(0, 2 ** self.n_bits, 100)  # Test subset for speed
        voltages = [dac.convert(code) for code in codes]

        # Check each step is positive
        differences = np.diff(voltages)
        self.assertTrue(np.all(differences >= 0))

    def test_different_references(self):
        """Test different reference voltages"""
        test_refs = [1.0, 2.5, 3.3, 5.0]

        for v_ref in test_refs:
            dac = SimpleDAC(self.n_bits, v_ref, OutputType.SINGLE)

            # Test full scale
            max_code = 2 ** self.n_bits - 1
            self.assertAlmostEqual(dac.convert(max_code), v_ref)

            # Test mid scale
            mid_code = 2 ** (self.n_bits - 1)
            self.assertAlmostEqual(dac.convert(mid_code), v_ref / 2)

    def test_different_resolutions(self):
        """Test different bit resolutions"""
        test_bits = [8, 10, 12, 16]

        for n_bits in test_bits:
            dac = SimpleDAC(n_bits, self.v_ref, OutputType.SINGLE)

            # Test full scale
            max_code = 2 ** n_bits - 1
            self.assertAlmostEqual(dac.convert(max_code), self.v_ref)

            # Verify LSB size
            expected_lsb = self.v_ref / (2 ** n_bits)
            self.assertAlmostEqual(dac.lsb, expected_lsb)

    def test_code_range_validation(self):
        """Test input code validation"""
        dac = SimpleDAC(self.n_bits, self.v_ref)
        max_code = 2 ** self.n_bits - 1

        # Test valid range
        dac.convert(0)  # Should not raise
        dac.convert(max_code)  # Should not raise

        # Test invalid codes
        with self.assertRaises(ValueError):
            dac.convert(-1)
        with self.assertRaises(ValueError):
            dac.convert(max_code + 1)


if __name__ == '__main__':
    unittest.main()