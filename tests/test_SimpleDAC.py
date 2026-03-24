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

        # Test mid scale (no exact v_ref/2 with (2^N-1) LSB; check within 1 LSB)
        mid_code = 2 ** (self.n_bits - 1)
        expected_mid = mid_code * self.v_ref / (2**self.n_bits - 1)
        self.assertAlmostEqual(dac.convert(mid_code), expected_mid)

        # Test full scale
        max_code = 2 ** self.n_bits - 1
        self.assertAlmostEqual(dac.convert(max_code), self.v_ref)

        # Test quarter scale
        quarter_code = 2 ** (self.n_bits - 2)
        expected_quarter = quarter_code * self.v_ref / (2**self.n_bits - 1)
        self.assertAlmostEqual(dac.convert(quarter_code), expected_quarter)

    def test_differential(self):
        """Test differential conversion"""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)

        # Test near mid-scale (no integer code maps exactly to 0V diff with (2^N-1) LSB)
        mid_code = 2 ** (self.n_bits - 1)
        v_pos, v_neg = dac.convert(mid_code)
        self.assertAlmostEqual(v_pos - v_neg, 0.0, delta=dac.lsb)
        self.assertAlmostEqual(v_pos, self.v_ref / 2, delta=dac.lsb)
        self.assertAlmostEqual(v_neg, self.v_ref / 2, delta=dac.lsb)

        # Test full scale (max code maps exactly to v_ref with (2^N-1) LSB)
        max_code = 2 ** self.n_bits - 1
        v_pos, v_neg = dac.convert(max_code)
        self.assertAlmostEqual(v_pos - v_neg, self.v_ref)

        # Test zero scale
        v_pos, v_neg = dac.convert(0)
        self.assertAlmostEqual(v_pos - v_neg, -self.v_ref)

        # Test quarter scale (within 1 LSB of -v_ref/2)
        quarter_code = 2 ** (self.n_bits - 2)
        v_pos, v_neg = dac.convert(quarter_code)
        self.assertAlmostEqual(v_pos - v_neg, -self.v_ref / 2, delta=dac.lsb)

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

            # Test near mid scale
            mid_code = 2 ** (self.n_bits - 1)
            expected_mid = mid_code * v_ref / (2**self.n_bits - 1)
            self.assertAlmostEqual(dac.convert(mid_code), expected_mid)

    def test_different_resolutions(self):
        """Test different bit resolutions"""
        test_bits = [8, 10, 12, 16]

        for n_bits in test_bits:
            dac = SimpleDAC(n_bits, self.v_ref, OutputType.SINGLE)

            # Test full scale
            max_code = 2 ** n_bits - 1
            self.assertAlmostEqual(dac.convert(max_code), self.v_ref)

            # Verify LSB size
            expected_lsb = self.v_ref / (2 ** n_bits - 1)
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

    # ------------------------------------------------------------------ #
    # Non-idealities                                                       #
    # ------------------------------------------------------------------ #

    def test_offset_shifts_output(self):
        """A positive offset shifts all output voltages up by offset amount."""
        offset = 0.01  # 10 mV
        dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        dac_offset = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                               offset=offset)

        for code in [0, 1000, 2048, 4095]:
            ideal_v = dac_ideal.convert(code)
            offset_v = dac_offset.convert(code)
            self.assertAlmostEqual(offset_v - ideal_v, offset,
                                   msg=f"code={code}: offset shift wrong")

    def test_negative_offset(self):
        """Negative offset shifts outputs down."""
        offset = -0.005
        dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        dac_offset = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                               offset=offset)

        for code in [0, 1000, 2048, 4095]:
            ideal_v = dac_ideal.convert(code)
            offset_v = dac_offset.convert(code)
            self.assertAlmostEqual(offset_v - ideal_v, offset,
                                   msg=f"code={code}: negative offset shift wrong")

    def test_gain_error_scales_output(self):
        """gain_error=0.01 (+1%) scales the ideal voltage accordingly."""
        gain_error = 0.01
        dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        dac_gain = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                             gain_error=gain_error)

        for code in [1000, 2048, 3000]:
            ideal_v = dac_ideal.convert(code)
            gain_v = dac_gain.convert(code)
            expected_v = ideal_v * (1.0 + gain_error)
            self.assertAlmostEqual(gain_v, expected_v,
                                   msg=f"code={code}: gain error scaling wrong")

    def test_noise_spreads_output(self):
        """Large noise_rms causes repeated conversions of the same code to vary."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        noise_rms=0.1)
        voltages = [dac.convert(2048) for _ in range(200)]
        self.assertGreater(len(set(voltages)), 1,
                           "Voltages should vary with large noise_rms")

    def test_noise_rms_magnitude(self):
        """Statistically verify output std ~ noise_rms over many samples."""
        noise_rms = 0.005
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        noise_rms=noise_rms)
        voltages = np.array([dac.convert(2048) for _ in range(5000)])
        measured_std = np.std(voltages)
        # Allow 30% tolerance for statistical variation
        self.assertAlmostEqual(measured_std, noise_rms,
                               delta=noise_rms * 0.3,
                               msg=f"Measured std {measured_std} != expected {noise_rms}")

    def test_zero_nonidealities_ideal(self):
        """offset=0, gain_error=0, noise_rms=0 gives exact ideal output."""
        dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        dac_zero = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                             noise_rms=0.0, offset=0.0, gain_error=0.0)

        for code in [0, 1, 1000, 2048, 4095]:
            self.assertEqual(dac_zero.convert(code), dac_ideal.convert(code))

    def test_combined_nonidealities(self):
        """offset + gain_error + noise all applied simultaneously.

        Order: gain_error -> offset -> noise.
        With noise_rms=0, result should be deterministic.
        """
        gain_error = 0.02
        offset = 0.005
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        gain_error=gain_error, offset=offset, noise_rms=0.0)
        dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)

        for code in [500, 2048, 3500]:
            ideal_v = dac_ideal.convert(code)
            # Expected: gain first, then offset
            expected_v = ideal_v * (1.0 + gain_error) + offset
            actual_v = dac.convert(code)
            self.assertAlmostEqual(actual_v, expected_v,
                                   msg=f"code={code}: combined non-ideality mismatch")

    def test_nonideality_with_differential(self):
        """offset/gain/noise work correctly in differential mode."""
        offset = 0.01
        gain_error = 0.01
        dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)
        dac_ni = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL,
                           offset=offset, gain_error=gain_error, noise_rms=0.0)

        # Non-idealities are applied to ideal voltage before differential split
        code = 2048
        ideal_voltage = code * dac_ideal.lsb
        expected_voltage = ideal_voltage * (1.0 + gain_error) + offset

        v_pos, v_neg = dac_ni.convert(code)
        # Differential output is derived from the modified voltage
        v_diff_expected = 2 * expected_voltage - self.v_ref
        self.assertAlmostEqual(v_pos - v_neg, v_diff_expected,
                               msg="Differential non-ideality mismatch")

    def test_invalid_parameters(self):
        """Invalid noise_rms raises ValueError."""
        with self.assertRaises(ValueError):
            SimpleDAC(self.n_bits, self.v_ref, noise_rms=-1.0)

    def test_repr_includes_nonidealities(self):
        """__repr__ reflects non-zero non-ideality parameters."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        noise_rms=1e-4, offset=1e-3, gain_error=0.001)
        r = repr(dac)
        self.assertIn('noise_rms', r)
        self.assertIn('offset', r)
        self.assertIn('gain_error', r)

        # Ideal DAC repr should NOT include non-ideality params
        dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        r_ideal = repr(dac_ideal)
        self.assertNotIn('noise_rms', r_ideal)
        self.assertNotIn('offset', r_ideal)
        self.assertNotIn('gain_error', r_ideal)


if __name__ == '__main__':
    unittest.main()