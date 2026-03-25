"""
Tests for SimpleDAC ZOH / convert_sequence and constructor additions (fs, oversample).
"""

import unittest
import numpy as np
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType


class TestSimpleDACConstructor(unittest.TestCase):
    """Tests for fs / oversample constructor parameters."""

    def test_defaults(self):
        """oversample defaults to 1, fs defaults to 1.0."""
        dac = SimpleDAC(n_bits=12, v_ref=1.0)
        self.assertEqual(dac.oversample, 1)
        self.assertEqual(dac.fs, 1.0)

    def test_oversample_zero_raises(self):
        """oversample=0 raises ValueError."""
        with self.assertRaises(ValueError):
            SimpleDAC(12, 1.0, oversample=0)

    def test_oversample_negative_raises(self):
        """oversample=-1 raises ValueError."""
        with self.assertRaises(ValueError):
            SimpleDAC(12, 1.0, oversample=-1)


class TestConvertSequenceSingleEnded(unittest.TestCase):
    """Tests for convert_sequence in single-ended mode."""

    def setUp(self):
        self.n_bits = 12
        self.v_ref = 1.0
        self.max_code = 2**self.n_bits - 1

    def test_output_tuple_length(self):
        """Single-ended output is a 2-element tuple (t, voltages)."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        result = dac.convert_sequence(np.array([0, 100, 200]))
        self.assertEqual(len(result), 2)

    def test_output_array_lengths(self):
        """len(t) == len(voltages) == len(codes) * oversample."""
        oversample = 4
        codes = np.array([0, 1000, 2048, 4095])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        oversample=oversample)
        t, voltages = dac.convert_sequence(codes)
        expected_len = len(codes) * oversample
        self.assertEqual(len(t), expected_len)
        self.assertEqual(len(voltages), expected_len)

    def test_time_axis_start(self):
        """t[0] == 0."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE, fs=1e6, oversample=4)
        t, _ = dac.convert_sequence(np.array([0, 100, 200]))
        self.assertEqual(t[0], 0.0)

    def test_time_axis_end(self):
        """t[-1] == (len(codes)*oversample - 1) / (fs * oversample)."""
        fs = 10e6
        oversample = 8
        codes = np.array([0, 500, 1000, 2000, 4095])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        fs=fs, oversample=oversample)
        t, _ = dac.convert_sequence(codes)
        expected_end = (len(codes) * oversample - 1) / (fs * oversample)
        self.assertAlmostEqual(t[-1], expected_end, places=15)

    def test_time_axis_spacing(self):
        """Time spacing is uniform at 1 / (fs * oversample)."""
        fs = 10e6
        oversample = 8
        codes = np.array([0, 500, 1000])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        fs=fs, oversample=oversample)
        t, _ = dac.convert_sequence(codes)
        dt = np.diff(t)
        expected_dt = 1.0 / (fs * oversample)
        np.testing.assert_allclose(dt, expected_dt, rtol=1e-12)

    def test_zoh_hold_with_oversample(self):
        """With oversample=4, each group of 4 consecutive voltages is identical (noise_rms=0)."""
        oversample = 4
        codes = np.array([0, 1000, 2048, 4095])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        oversample=oversample, noise_rms=0.0)
        _, voltages = dac.convert_sequence(codes)
        for i in range(len(codes)):
            group = voltages[i * oversample:(i + 1) * oversample]
            np.testing.assert_array_equal(group, group[0],
                                          err_msg=f"ZOH hold broken at code index {i}")

    def test_ideal_code_zero(self):
        """Ideal DAC, code=0: all voltages are 0.0."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE, oversample=4)
        _, voltages = dac.convert_sequence(np.array([0]))
        np.testing.assert_array_equal(voltages, 0.0)

    def test_ideal_code_full_scale(self):
        """Ideal DAC, code=4095 (12-bit): all voltages ≈ v_ref."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE, oversample=4)
        _, voltages = dac.convert_sequence(np.array([self.max_code]))
        np.testing.assert_allclose(voltages, self.v_ref, atol=1e-9)

    def test_gain_error_and_offset_per_code(self):
        """Gain error + offset applied per code, held across oversample steps."""
        gain_error = 0.02
        offset = 0.005
        oversample = 4
        codes = np.array([500, 2048, 3500])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        gain_error=gain_error, offset=offset,
                        noise_rms=0.0, oversample=oversample)
        _, voltages = dac.convert_sequence(codes)

        lsb = self.v_ref / self.max_code
        for i, code in enumerate(codes):
            expected = code * lsb * (1.0 + gain_error) + offset
            group = voltages[i * oversample:(i + 1) * oversample]
            np.testing.assert_allclose(group, expected, rtol=1e-12,
                                       err_msg=f"Gain+offset wrong at code {code}")

    def test_noise_rms_magnitude(self):
        """With noise_rms>0, std of a long output ≈ noise_rms (within 20%)."""
        noise_rms = 0.005
        # Use a single code repeated many times to isolate noise
        codes = np.full(5000, 2048)
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        noise_rms=noise_rms)
        _, voltages = dac.convert_sequence(codes)
        measured_std = np.std(voltages)
        self.assertAlmostEqual(measured_std, noise_rms,
                               delta=noise_rms * 0.2,
                               msg=f"Measured std {measured_std} not within 20% of {noise_rms}")

    def test_fs_oversample_time_spacing(self):
        """fs=10e6, oversample=8: time spacing = 1/(10e6*8)."""
        fs = 10e6
        oversample = 8
        codes = np.array([0, 1000, 2000])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                        fs=fs, oversample=oversample)
        t, _ = dac.convert_sequence(codes)
        dt = np.diff(t)
        expected_dt = 1.0 / (fs * oversample)
        np.testing.assert_allclose(dt, expected_dt, rtol=1e-12)


class TestConvertSequenceDifferential(unittest.TestCase):
    """Tests for convert_sequence in differential mode."""

    def setUp(self):
        self.n_bits = 12
        self.v_ref = 1.0
        self.max_code = 2**self.n_bits - 1

    def test_output_tuple_length(self):
        """Differential output is a 3-element tuple (t, v_pos, v_neg)."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)
        result = dac.convert_sequence(np.array([0, 100, 200]))
        self.assertEqual(len(result), 3)

    def test_full_scale_differential(self):
        """Full-scale code: v_pos - v_neg ≈ +v_ref."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL,
                        oversample=4, noise_rms=0.0)
        t, v_pos, v_neg = dac.convert_sequence(np.array([self.max_code]))
        np.testing.assert_allclose(v_pos - v_neg, self.v_ref, atol=1e-9)

    def test_zero_scale_differential(self):
        """Zero-scale code: v_pos - v_neg ≈ -v_ref."""
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL,
                        oversample=4, noise_rms=0.0)
        t, v_pos, v_neg = dac.convert_sequence(np.array([0]))
        np.testing.assert_allclose(v_pos - v_neg, -self.v_ref, atol=1e-9)

    def test_common_mode(self):
        """v_pos + v_neg ≈ v_ref (common mode) for all codes."""
        codes = np.array([0, 1024, 2048, 3072, self.max_code])
        dac = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL,
                        oversample=2, noise_rms=0.0)
        t, v_pos, v_neg = dac.convert_sequence(codes)
        np.testing.assert_allclose(v_pos + v_neg, self.v_ref, atol=1e-9)


class TestRegressionExistingTests(unittest.TestCase):
    """Ensure pre-existing SimpleDAC functionality still works after changes."""

    def test_single_convert_still_works(self):
        """Basic single convert() call still returns correct value."""
        dac = SimpleDAC(12, 1.0, OutputType.SINGLE)
        self.assertAlmostEqual(dac.convert(4095), 1.0)
        self.assertEqual(dac.convert(0), 0.0)

    def test_differential_convert_still_works(self):
        """Basic differential convert() call still returns correct tuple."""
        dac = SimpleDAC(12, 1.0, OutputType.DIFFERENTIAL)
        v_pos, v_neg = dac.convert(4095)
        self.assertAlmostEqual(v_pos - v_neg, 1.0)


if __name__ == '__main__':
    unittest.main()
