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


    # ------------------------------------------------------------------ #
    # Constructor: fs and oversample defaults / validation                  #
    # ------------------------------------------------------------------ #

    def test_defaults_fs_and_oversample(self):
        """fs defaults to 1.0 and oversample defaults to 1."""
        dac = SimpleDAC(self.n_bits, self.v_ref)
        self.assertEqual(dac.fs, 1.0)
        self.assertEqual(dac.oversample, 1)

    def test_oversample_zero_raises(self):
        """oversample=0 raises ValueError."""
        with self.assertRaises(ValueError):
            SimpleDAC(self.n_bits, self.v_ref, oversample=0)

    def test_oversample_negative_raises(self):
        """oversample=-1 raises ValueError."""
        with self.assertRaises(ValueError):
            SimpleDAC(self.n_bits, self.v_ref, oversample=-1)

    # ------------------------------------------------------------------ #
    # convert_sequence — single-ended                                       #
    # ------------------------------------------------------------------ #

    def test_convert_sequence_returns_two_arrays(self):
        """Single-ended convert_sequence returns (t, voltages)."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        result = dac.convert_sequence(np.array([0, 1, 2]))
        self.assertEqual(len(result), 2)

    def test_convert_sequence_length(self):
        """Output length == len(codes) * oversample."""
        oversample = 4
        codes = np.array([0, 100, 200, 300])
        dac = SimpleDAC(self.n_bits, self.v_ref, oversample=oversample)
        t, v = dac.convert_sequence(codes)
        expected_len = len(codes) * oversample
        self.assertEqual(len(t), expected_len)
        self.assertEqual(len(v), expected_len)

    def test_convert_sequence_time_axis(self):
        """Time axis starts at 0, ends correctly, has uniform spacing."""
        fs = 100.0
        oversample = 4
        codes = np.array([0, 100, 200])
        dac = SimpleDAC(self.n_bits, self.v_ref, fs=fs, oversample=oversample)
        t, _ = dac.convert_sequence(codes)

        self.assertAlmostEqual(t[0], 0.0)
        expected_last = (len(codes) * oversample - 1) / (fs * oversample)
        self.assertAlmostEqual(t[-1], expected_last)
        spacing = np.diff(t)
        np.testing.assert_allclose(spacing, 1.0 / (fs * oversample))

    def test_convert_sequence_zoh_hold(self):
        """With oversample=4, each group of 4 consecutive voltages are equal."""
        oversample = 4
        codes = np.array([0, 2048, 4095])
        dac = SimpleDAC(self.n_bits, self.v_ref, oversample=oversample)
        _, v = dac.convert_sequence(codes)
        for i in range(len(codes)):
            group = v[i * oversample:(i + 1) * oversample]
            np.testing.assert_allclose(group, group[0])

    def test_convert_sequence_code_zero(self):
        """Ideal DAC, code=0 → voltages all 0.0."""
        dac = SimpleDAC(self.n_bits, self.v_ref, oversample=2)
        _, v = dac.convert_sequence(np.array([0, 0, 0]))
        np.testing.assert_allclose(v, 0.0)

    def test_convert_sequence_code_full_scale(self):
        """Ideal DAC, code=4095 (12-bit) → voltages ≈ v_ref."""
        max_code = 2 ** self.n_bits - 1
        dac = SimpleDAC(self.n_bits, self.v_ref, oversample=2)
        _, v = dac.convert_sequence(np.array([max_code, max_code]))
        np.testing.assert_allclose(v, self.v_ref, atol=1e-9)

    def test_convert_sequence_gain_and_offset(self):
        """Gain error + offset applied per code, held across oversample steps."""
        gain_error = 0.02
        offset = 0.005
        oversample = 4
        codes = np.array([1000, 2048, 3000])
        dac = SimpleDAC(self.n_bits, self.v_ref,
                        gain_error=gain_error, offset=offset,
                        oversample=oversample)
        _, v = dac.convert_sequence(codes)

        for i, code in enumerate(codes):
            ideal_v = code * dac.lsb
            expected = ideal_v * (1.0 + gain_error) + offset
            group = v[i * oversample:(i + 1) * oversample]
            np.testing.assert_allclose(group, expected,
                                       err_msg=f"code={code}: gain+offset mismatch")

    def test_convert_sequence_noise_per_sample(self):
        """Noise is independent per output sample; std ≈ noise_rms (within 20%)."""
        noise_rms = 0.01
        dac = SimpleDAC(self.n_bits, self.v_ref, noise_rms=noise_rms,
                        oversample=1)
        codes = np.full(10000, 2048)
        _, v = dac.convert_sequence(codes)
        ideal_v = 2048 * dac.lsb
        measured_std = np.std(v - ideal_v)
        self.assertAlmostEqual(measured_std, noise_rms,
                               delta=noise_rms * 0.2,
                               msg=f"Measured std {measured_std} != expected {noise_rms}")

    def test_convert_sequence_time_spacing_high_fs(self):
        """fs=10e6, oversample=8: spacing = 1/(10e6*8)."""
        fs = 10e6
        oversample = 8
        codes = np.array([0, 100])
        dac = SimpleDAC(self.n_bits, self.v_ref, fs=fs, oversample=oversample)
        t, _ = dac.convert_sequence(codes)
        spacing = np.diff(t)
        np.testing.assert_allclose(spacing, 1.0 / (fs * oversample))

    def test_convert_sequence_clips_codes(self):
        """Codes outside [0, 2^n_bits-1] are clipped, not rejected."""
        dac = SimpleDAC(self.n_bits, self.v_ref)
        max_code = 2 ** self.n_bits - 1
        codes = np.array([-5, 0, max_code, max_code + 100])
        _, v = dac.convert_sequence(codes)
        # -5 clipped to 0, max_code+100 clipped to max_code
        self.assertAlmostEqual(v[0], 0.0)
        self.assertAlmostEqual(v[-1], self.v_ref, places=9)

    # ------------------------------------------------------------------ #
    # convert_sequence — differential                                       #
    # ------------------------------------------------------------------ #

    def test_convert_sequence_differential_returns_three(self):
        """Differential convert_sequence returns (t, v_pos, v_neg)."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)
        result = dac.convert_sequence(np.array([0, 2048, 4095]))
        self.assertEqual(len(result), 3)

    def test_convert_sequence_differential_full_scale(self):
        """Full-scale: v_pos - v_neg ≈ +v_ref."""
        max_code = 2 ** self.n_bits - 1
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)
        t, vp, vn = dac.convert_sequence(np.array([max_code]))
        np.testing.assert_allclose(vp - vn, self.v_ref, atol=1e-9)

    def test_convert_sequence_differential_zero_scale(self):
        """Zero-scale: v_pos - v_neg ≈ -v_ref."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)
        t, vp, vn = dac.convert_sequence(np.array([0]))
        np.testing.assert_allclose(vp - vn, -self.v_ref, atol=1e-9)

    def test_convert_sequence_differential_common_mode(self):
        """v_pos + v_neg ≈ v_ref (common mode) for all codes."""
        codes = np.array([0, 1000, 2048, 3000, 4095])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)
        t, vp, vn = dac.convert_sequence(codes)
        np.testing.assert_allclose(vp + vn, self.v_ref, atol=1e-9)

    # ------------------------------------------------------------------ #
    # __repr__ with fs / oversample                                         #
    # ------------------------------------------------------------------ #

    def test_repr_default_omits_fs_oversample(self):
        """Default fs/oversample should NOT appear in repr."""
        dac = SimpleDAC(self.n_bits, self.v_ref)
        r = repr(dac)
        self.assertNotIn('fs=', r)
        self.assertNotIn('oversample=', r)

    def test_repr_nondefault_shows_fs_oversample(self):
        """Non-default fs/oversample should appear in repr."""
        dac = SimpleDAC(self.n_bits, self.v_ref, fs=48000.0, oversample=8)
        r = repr(dac)
        self.assertIn('fs=48000.0', r)
        self.assertIn('oversample=8', r)


# ===========================================================================
# __main__ block coverage
# ===========================================================================

class TestDACBaseNLevels(unittest.TestCase):
    """DACBase now supports arbitrary n_levels, decoupled from n_bits."""

    def test_default_n_levels_matches_2_to_the_n_bits(self):
        # When n_levels is not supplied, DACBase (via SimpleDAC) should behave
        # exactly as before: n_levels == 2**n_bits, lsb == v_ref / (2^n - 1).
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE)
        self.assertEqual(dac.convert(0), 0.0)
        self.assertAlmostEqual(dac.convert(7), 1.0)
        self.assertAlmostEqual(dac.lsb, 1.0 / 7)

    def test_explicit_n_levels_overrides_n_bits(self):
        # 9 output levels over [0, v_ref] with lsb = v_ref / 8.
        dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0, output_type=OutputType.SINGLE)
        self.assertAlmostEqual(dac.lsb, 1.0 / 8)
        self.assertAlmostEqual(dac.convert(0), 0.0)
        self.assertAlmostEqual(dac.convert(4), 0.5)
        self.assertAlmostEqual(dac.convert(8), 1.0)

    def test_code_above_n_levels_raises(self):
        dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0, output_type=OutputType.SINGLE)
        with self.assertRaises(ValueError):
            dac.convert(9)

    def test_n_levels_less_than_two_raises(self):
        with self.assertRaises(ValueError):
            SimpleDAC(n_bits=3, n_levels=1, v_ref=1.0, output_type=OutputType.SINGLE)

    def test_n_levels_not_int_raises(self):
        with self.assertRaises(TypeError):
            SimpleDAC(n_bits=3, n_levels=3.5, v_ref=1.0, output_type=OutputType.SINGLE)


class TestSimpleDACCodeErrors(unittest.TestCase):
    """SimpleDAC supports per-code additive error injection."""

    def test_default_no_errors(self):
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE)
        # With no code_errors, behaves exactly like the ideal transfer
        for code in range(8):
            self.assertAlmostEqual(dac.convert(code), code / 7)

    def test_code_errors_applied_additively(self):
        errors = np.array([0.0, 0.01, -0.02, 0.005, 0.0, -0.01, 0.002, 0.0])
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                        code_errors=errors)
        for code in range(8):
            expected = code / 7 + errors[code]
            self.assertAlmostEqual(dac.convert(code), expected)

    def test_code_errors_with_n_levels(self):
        # 9-level DAC with a specific error pattern
        errors = np.array([0.0, -0.2, 0.3, 0.05, -0.15, 0.0, 0.3, -0.3, 0.0]) * 0.001
        dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0,
                        output_type=OutputType.SINGLE, code_errors=errors)
        for code in range(9):
            expected = code / 8 + errors[code]
            self.assertAlmostEqual(dac.convert(code), expected)

    def test_code_errors_wrong_length_raises(self):
        errors = np.zeros(7)  # should be 8 for n_bits=3
        with self.assertRaises(ValueError):
            SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                      code_errors=errors)

    def test_code_errors_not_array_raises(self):
        with self.assertRaises(TypeError):
            SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                      code_errors="not an array")

    def test_code_errors_applied_before_gain_offset_noise(self):
        # Code error + offset should both appear in the output
        errors = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dac = SimpleDAC(n_bits=3, v_ref=1.0, output_type=OutputType.SINGLE,
                        offset=0.05, code_errors=errors)
        # code=1: ideal = 1/7, + code_error 0.1, + offset 0.05
        self.assertAlmostEqual(dac.convert(1), 1/7 + 0.1 + 0.05)


class TestSimpleDACConvertSequenceNLevels(unittest.TestCase):
    """convert_sequence must honour n_levels, not fall back to 2**n_bits."""

    def test_convert_sequence_respects_n_levels(self):
        # 9-level DAC: codes 0..8 must all produce distinct outputs
        dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0,
                        output_type=OutputType.SINGLE)
        codes = np.arange(9)
        t, voltages = dac.convert_sequence(codes)
        # Code 8 must produce v_ref (= 1.0), NOT v_ref * 7/8 (= 0.875)
        # under the bug, code 8 would be clipped to 7 and return 0.875.
        self.assertAlmostEqual(float(voltages[-1]), 1.0, places=10)
        # All codes should be monotonically increasing (9 distinct values)
        self.assertEqual(len(np.unique(voltages)), 9)


if __name__ == '__main__':
    unittest.main()