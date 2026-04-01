"""
Tests for DAC static and dynamic performance metrics (dac_metrics.py).
"""

import unittest
import numpy as np

from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.metrics import (
    calculate_dac_static_metrics,
    calculate_dac_dynamic_metrics,
)


class TestCalculateDACStaticMetrics(unittest.TestCase):
    """Tests for calculate_dac_static_metrics."""

    def test_ideal_8bit_dac(self):
        """Ideal DAC should have near-zero DNL, INL, offset, and gain error."""
        dac = SimpleDAC(n_bits=8, v_ref=1.0, output_type=OutputType.SINGLE)
        result = calculate_dac_static_metrics(dac)
        self.assertLess(result["MaxDNL"], 1e-9)
        self.assertLess(result["MaxINL"], 1e-9)
        self.assertAlmostEqual(result["Offset"], 0.0, places=9)
        self.assertAlmostEqual(result["GainError"], 0.0, places=9)

    def test_dac_with_offset(self):
        """Offset parameter should appear in the Offset metric."""
        dac = SimpleDAC(n_bits=8, v_ref=1.0, output_type=OutputType.SINGLE,
                        offset=5e-3)
        result = calculate_dac_static_metrics(dac)
        self.assertAlmostEqual(result["Offset"], 5e-3, places=6)

    def test_dac_with_gain_error(self):
        """Gain error parameter should appear in the GainError metric."""
        dac = SimpleDAC(n_bits=8, v_ref=1.0, output_type=OutputType.SINGLE,
                        gain_error=0.01)
        result = calculate_dac_static_metrics(dac)
        self.assertAlmostEqual(result["GainError"], 0.01, places=4)

    def test_differential_dac(self):
        """Differential DAC metrics computed from v_diff = v_pos - v_neg."""
        dac = SimpleDAC(n_bits=8, v_ref=1.0, output_type=OutputType.DIFFERENTIAL)
        result = calculate_dac_static_metrics(dac)
        # Differential output spans [-v_ref, v_ref], so all expected keys present
        self.assertIn("MaxDNL", result)
        self.assertIn("MaxINL", result)
        # Endpoint-fit INL should still be near zero (linear transfer)
        self.assertLess(result["MaxINL"], 1e-9)
        # Voltages span [-v_ref, v_ref] for differential
        self.assertAlmostEqual(result["Voltages"][0], -1.0, places=9)
        self.assertAlmostEqual(result["Voltages"][-1], 1.0, places=9)

    def test_return_keys(self):
        """Result dict must contain all expected keys."""
        dac = SimpleDAC(n_bits=4, v_ref=1.0)
        result = calculate_dac_static_metrics(dac)
        expected_keys = {"DNL", "INL", "MaxDNL", "MaxINL", "Offset",
                         "GainError", "Codes", "Voltages"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_dnl_length(self):
        """DNL length should be 2^n_bits - 1."""
        n_bits = 6
        dac = SimpleDAC(n_bits=n_bits, v_ref=1.0)
        result = calculate_dac_static_metrics(dac)
        self.assertEqual(len(result["DNL"]), 2 ** n_bits - 1)

    def test_inl_length(self):
        """INL length should be 2^n_bits."""
        n_bits = 6
        dac = SimpleDAC(n_bits=n_bits, v_ref=1.0)
        result = calculate_dac_static_metrics(dac)
        self.assertEqual(len(result["INL"]), 2 ** n_bits)

    def test_n_points(self):
        """n_points limits the number of swept codes."""
        dac = SimpleDAC(n_bits=8, v_ref=1.0)
        result = calculate_dac_static_metrics(dac, n_points=100)
        self.assertEqual(len(result["Codes"]), 100)
        self.assertEqual(len(result["Voltages"]), 100)


class TestCalculateDACDynamicMetrics(unittest.TestCase):
    """Tests for calculate_dac_dynamic_metrics."""

    @staticmethod
    def _ideal_dac_capture(n_bits=12, v_ref=1.0, fs=1e6, n_samples=8192):
        """Generate an ideal DAC output waveform (quantised sine).

        Uses coherent sampling (f_sig is an exact FFT bin) to minimise
        spectral leakage and maximise measured SNR.
        """
        # Pick a coherent tone: exact bin number (prime to avoid harmonic overlap)
        n_cycles = 43
        f_sig = n_cycles * fs / n_samples
        dac = SimpleDAC(n_bits=n_bits, v_ref=v_ref, output_type=OutputType.SINGLE)
        t = np.arange(n_samples) / fs
        analog = (v_ref / 2) * (1 + np.sin(2 * np.pi * f_sig * t))
        max_code = 2 ** n_bits - 1
        codes = np.clip(np.round(analog / dac.lsb).astype(int), 0, max_code)
        voltages = np.array([dac.convert(int(c)) for c in codes])
        return voltages, fs

    def test_ideal_12bit_zone1(self):
        """Ideal 12-bit DAC in zone 1 should have SNR >= 70 dB and ENOB >= 11."""
        voltages, fs = self._ideal_dac_capture()
        result = calculate_dac_dynamic_metrics(voltages, fs)
        self.assertGreaterEqual(result["SNR"], 70)
        self.assertGreaterEqual(result["ENOB"], 11)

    def test_fs_update_none_defaults_to_fs(self):
        """fs_update=None should produce the same result as fs_update=fs."""
        voltages, fs = self._ideal_dac_capture()
        r1 = calculate_dac_dynamic_metrics(voltages, fs, fs_update=None)
        r2 = calculate_dac_dynamic_metrics(voltages, fs, fs_update=fs)
        self.assertAlmostEqual(r1["SNR"], r2["SNR"], places=6)
        self.assertAlmostEqual(r1["ENOB"], r2["ENOB"], places=6)

    def test_valueerror_fs_below_nyquist(self):
        """Should raise ValueError when fs < nyquist_zone * fs_update."""
        voltages = np.sin(np.linspace(0, 2 * np.pi, 256))
        with self.assertRaises(ValueError):
            calculate_dac_dynamic_metrics(voltages, fs=1e6,
                                          fs_update=1e6, nyquist_zone=2)

    def test_full_scale_none_no_dbfs_keys(self):
        """full_scale=None should not include _dBFS keys."""
        voltages, fs = self._ideal_dac_capture()
        result = calculate_dac_dynamic_metrics(voltages, fs, full_scale=None)
        dbfs_keys = [k for k in result if k.endswith("_dBFS")]
        self.assertEqual(len(dbfs_keys), 0)

    def test_full_scale_adds_dbfs_keys(self):
        """full_scale=v_ref should add _dBFS keys."""
        voltages, fs = self._ideal_dac_capture(v_ref=1.0)
        result = calculate_dac_dynamic_metrics(voltages, fs, full_scale=1.0)
        expected_dbfs = {"SNR_dBFS", "SNDR_dBFS", "SFDR_dBFS", "THD_dBFS",
                         "FundamentalMagnitude_dBFS"}
        self.assertTrue(expected_dbfs.issubset(set(result.keys())))

    def test_zone_band_hz_zone1(self):
        """Zone 1 band should be (0, fs_update/2)."""
        voltages, fs = self._ideal_dac_capture()
        result = calculate_dac_dynamic_metrics(voltages, fs)
        self.assertEqual(result["ZoneBandHz"], (0.0, fs / 2))

    def test_nyquist_zone_key(self):
        """NyquistZone key should equal the passed value."""
        voltages, fs = self._ideal_dac_capture()
        result = calculate_dac_dynamic_metrics(voltages, fs, nyquist_zone=1)
        self.assertEqual(result["NyquistZone"], 1)

    def test_zone2_oversampled(self):
        """Zone 2 on an oversampled capture should return valid metrics."""
        fs_update = 1e6
        fs = 4 * fs_update  # oversample 4x so zone 2 is visible
        n_samples = 4096
        f_sig = fs_update / 2 + 50e3  # signal in zone 2
        t = np.arange(n_samples) / fs
        voltages = 0.5 * np.sin(2 * np.pi * f_sig * t)
        result = calculate_dac_dynamic_metrics(voltages, fs,
                                               fs_update=fs_update,
                                               nyquist_zone=2)
        self.assertEqual(result["NyquistZone"], 2)
        expected_band = (fs_update / 2, fs_update)
        self.assertEqual(result["ZoneBandHz"], expected_band)
        self.assertIn("ENOB", result)


if __name__ == "__main__":
    unittest.main()
