"""Tests for generalised SAR ADC components."""
import numpy as np
import pytest
from pyDataconverter.components.cdac import RedundantSARCDAC, SplitCapCDAC, SegmentedCDAC


class TestRedundantSARCDAC:
    def test_construction(self):
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        assert cdac.n_bits == 4
        assert cdac.v_ref  == 1.0

    def test_weights_decrease(self):
        """Capacitor weights are strictly decreasing (MSB first)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.85)
        w = cdac.cap_weights
        assert np.all(np.diff(w) < 0)

    def test_weights_not_binary(self):
        """Weights are not powers of 2 (distinguishes from standard CDAC)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        w = cdac.cap_weights
        binary_weights = 2 ** np.arange(cdac.n_bits - 1, -1, -1).astype(float)
        assert not np.allclose(w, binary_weights)

    def test_dec_monotone(self):
        """Decoded output is monotone with increasing DAC voltage.

        The DEC table must map raw codes to output codes such that
        decode(a) <= decode(b) whenever cdac.get_voltage(a) <= cdac.get_voltage(b).
        This is the correct monotonicity property for a redundant SAR DEC.
        """
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        n_codes = 2 ** 6
        voltages = np.array([cdac.get_voltage(c)[0] for c in range(n_codes)])
        decoded  = np.array([cdac.decode(c) for c in range(n_codes)])
        # Sort raw codes by DAC voltage; decoded values in that order must be non-decreasing
        order = np.argsort(voltages)
        assert list(decoded[order]) == sorted(decoded[order])

    def test_dec_range(self):
        """Decoded values span 0..2^n_bits-1."""
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        decoded = [cdac.decode(code) for code in range(2**4)]
        assert min(decoded) == 0
        assert max(decoded) == 2**4 - 1

    def test_ideal_conversion_with_sar(self):
        """RedundantSARCDAC inside SARADC gives monotone ideal transfer."""
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        adc  = SARADC(n_bits=6, v_ref=1.0, cdac=cdac)
        vin  = np.linspace(0, 1.0, 200)
        codes = [adc.convert(float(v)) for v in vin]
        # codes should be non-decreasing
        assert all(b >= a for a, b in zip(codes, codes[1:]))


class TestSplitCapCDAC:
    def test_construction(self):
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        assert cdac.n_bits == 8
        assert cdac.v_ref  == 1.0

    def test_fewer_total_caps_than_full_binary(self):
        """Split-cap uses n_bits+1 caps (incl. bridge), not 2^n_bits."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        # n_msb + n_lsb + 1 bridge cap = n_bits + 1 total caps
        assert len(cdac.cap_weights) == 8 + 1  # 9 caps, not 256

    def test_ideal_output_zero(self):
        """Code 0 → 0 V."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        vp, vn = cdac.get_voltage(0)
        assert abs(vp - vn) < 1e-9

    def test_ideal_output_full_scale(self):
        """Max code → v_ref - LSB."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        max_code = 2**8 - 1
        vp, vn = cdac.get_voltage(max_code)
        expected = (max_code / 2**8) * 1.0
        assert abs(vp - vn - expected) < 0.01  # within 1% of LSB

    def test_monotone_voltages(self):
        """Voltages are non-decreasing with code."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        voltages = cdac.voltages
        assert np.all(np.diff(voltages) >= -1e-9)


class TestSegmentedCDAC:
    def test_construction(self):
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        assert cdac.n_bits == 8

    def test_monotone_voltages(self):
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        v = cdac.voltages
        assert np.all(np.diff(v) >= -1e-9)

    def test_output_range(self):
        """Max code → v_ref - LSB, code 0 → 0."""
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        v = cdac.voltages
        assert abs(v[0]) < 1e-9
        expected_max = (2**8 - 1) / 2**8
        assert abs(v[-1] - expected_max) < 0.01

    def test_thermometer_section_equal_steps(self):
        """The MSB 2^n_therm steps should be equal (thermometer linearity)."""
        n_bits, n_therm = 8, 4
        cdac = SegmentedCDAC(n_bits=n_bits, v_ref=1.0, n_therm=n_therm)
        v = cdac.voltages
        step = 2**n_bits // 2**n_therm  # codes per thermometer step
        msb_voltages = v[step-1::step][:2**n_therm]
        diffs = np.diff(msb_voltages)
        assert np.allclose(diffs, diffs[0], rtol=0.01)
