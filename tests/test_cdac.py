"""
Tests for CDAC components (SingleEndedCDAC, DifferentialCDAC).

Supplements the CDAC tests in test_SARADC.py with additional edge cases.
"""

import numpy as np
import pytest
from pyDataconverter.components.cdac import (
    CDACBase, SingleEndedCDAC, DifferentialCDAC,
    RedundantSARCDAC, SplitCapCDAC, SegmentedCDAC,
)
from pyDataconverter.components.capacitor import IdealCapacitor


# ===========================================================================
# SingleEndedCDAC — validation edge cases
# ===========================================================================

class TestSingleEndedCDACValidation:

    def test_zero_bits_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=0, v_ref=1.0)

    def test_negative_bits_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=-1, v_ref=1.0)

    def test_33_bits_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=33, v_ref=1.0)

    def test_float_bits_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=3.0, v_ref=1.0)

    def test_zero_vref_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=3, v_ref=0.0)

    def test_negative_vref_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=3, v_ref=-1.0)

    def test_string_vref_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=3, v_ref="1.0")

    def test_2d_cap_weights_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=3, v_ref=1.0,
                            cap_weights=np.array([[4., 2.], [1., 0.5]]))

    def test_1_bit_cdac(self):
        cdac = SingleEndedCDAC(n_bits=1, v_ref=1.0)
        v0, _ = cdac.get_voltage(0)
        v1, _ = cdac.get_voltage(1)
        assert v0 == pytest.approx(0.0)
        assert v1 == pytest.approx(0.5)

    def test_cap_instances_length(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        assert len(cdac.cap_instances) == 4
        assert all(isinstance(c, IdealCapacitor) for c in cdac.cap_instances)

    def test_repr(self):
        cdac = SingleEndedCDAC(n_bits=3, v_ref=1.5, cap_mismatch=0.01)
        r = repr(cdac)
        assert 'SingleEndedCDAC' in r
        assert 'n_bits=3' in r
        assert 'v_ref=1.5' in r


# ===========================================================================
# DifferentialCDAC — validation edge cases
# ===========================================================================

class TestDifferentialCDACValidation:

    def test_zero_bits_raises(self):
        with pytest.raises(ValueError):
            DifferentialCDAC(n_bits=0, v_ref=1.0)

    def test_zero_vref_raises(self):
        with pytest.raises(ValueError):
            DifferentialCDAC(n_bits=3, v_ref=0.0)

    def test_string_vref_raises(self):
        with pytest.raises(ValueError):
            DifferentialCDAC(n_bits=3, v_ref="1.0")

    def test_1_bit_cdac(self):
        cdac = DifferentialCDAC(n_bits=1, v_ref=1.0, cap_mismatch=0.0)
        v_p0, v_n0 = cdac.get_voltage(0)
        v_p1, v_n1 = cdac.get_voltage(1)
        # code 0: differential = -v_ref/2
        assert pytest.approx(v_p0 - v_n0) == -0.5
        # code 1: differential = 0 (not +v_ref/2 due to FLOOR)
        assert v_p1 - v_n1 == pytest.approx(0.0)

    def test_cap_instances_neg_length(self):
        cdac = DifferentialCDAC(n_bits=4, v_ref=1.0)
        assert len(cdac.cap_instances) == 4
        assert len(cdac.cap_instances_neg) == 4

    def test_repr(self):
        cdac = DifferentialCDAC(n_bits=3, v_ref=2.0, cap_mismatch=0.005)
        r = repr(cdac)
        assert 'DifferentialCDAC' in r
        assert 'n_bits=3' in r


# ===========================================================================
# CDACBase.voltages property
# ===========================================================================

class TestCDACVoltagesProperty:

    def test_single_ended_voltages_length(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        assert len(cdac.voltages) == 16

    def test_differential_voltages_length(self):
        cdac = DifferentialCDAC(n_bits=4, v_ref=1.0)
        assert len(cdac.voltages) == 16

    def test_cdac_base_repr(self):
        cdac = SingleEndedCDAC(n_bits=3, v_ref=1.0)
        # CDACBase.__repr__ should work
        base_repr = CDACBase.__repr__(cdac)
        assert 'n_bits=3' in base_repr


# ===========================================================================
# RedundantSARCDAC — line 297 (radix validation) and 334 (__repr__)
# ===========================================================================

class TestRedundantSARCDAC:

    def test_radix_too_low_raises(self):
        """Line 297: radix <= 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="radix must be in"):
            RedundantSARCDAC(n_bits=4, radix=1.0)

    def test_radix_exactly_two_raises(self):
        """Line 297: radix >= 2.0 must raise ValueError."""
        with pytest.raises(ValueError, match="radix must be in"):
            RedundantSARCDAC(n_bits=4, radix=2.0)

    def test_radix_below_one_raises(self):
        with pytest.raises(ValueError, match="radix must be in"):
            RedundantSARCDAC(n_bits=4, radix=0.5)

    def test_valid_radix_constructs(self):
        cdac = RedundantSARCDAC(n_bits=4, radix=1.85)
        assert cdac.radix == pytest.approx(1.85)

    def test_repr(self):
        """Line 334: __repr__ coverage."""
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        r = repr(cdac)
        assert 'RedundantSARCDAC' in r
        assert 'n_bits=4' in r
        assert 'radix=1.8' in r


# ===========================================================================
# SplitCapCDAC — lines 372, 375 (n_msb validation) and 419 (__repr__)
# ===========================================================================

class TestSplitCapCDAC:

    def test_default_n_msb(self):
        """Line 372: n_msb defaults to n_bits // 2."""
        cdac = SplitCapCDAC(n_bits=8)
        assert cdac._n_msb == 4

    def test_n_msb_zero_raises(self):
        """Line 375: n_msb=0 is out of [1, n_bits-1]."""
        with pytest.raises(ValueError, match="n_msb must be in"):
            SplitCapCDAC(n_bits=4, n_msb=0)

    def test_n_msb_equals_n_bits_raises(self):
        """Line 375: n_msb=n_bits is out of [1, n_bits-1]."""
        with pytest.raises(ValueError, match="n_msb must be in"):
            SplitCapCDAC(n_bits=4, n_msb=4)

    def test_valid_construction(self):
        cdac = SplitCapCDAC(n_bits=6, n_msb=3)
        assert cdac._n_msb == 3

    def test_repr(self):
        """Line 419: __repr__ coverage."""
        cdac = SplitCapCDAC(n_bits=6, v_ref=1.0, n_msb=3)
        r = repr(cdac)
        assert 'SplitCapCDAC' in r
        assert 'n_bits=6' in r
        assert 'n_msb=3' in r


# ===========================================================================
# SegmentedCDAC — lines 455, 490, 494, 498, 510, 536 (init, properties, get_voltage, __repr__)
# ===========================================================================

class TestSegmentedCDAC:

    def test_n_therm_zero_raises(self):
        """Line 455: n_therm=0 is out of [1, n_bits-1]."""
        with pytest.raises(ValueError, match="n_therm must be in"):
            SegmentedCDAC(n_bits=8, n_therm=0)

    def test_n_therm_equals_n_bits_raises(self):
        """Line 455: n_therm=n_bits is out of [1, n_bits-1]."""
        with pytest.raises(ValueError, match="n_therm must be in"):
            SegmentedCDAC(n_bits=8, n_therm=8)

    def test_valid_construction(self):
        cdac = SegmentedCDAC(n_bits=8, n_therm=4)
        assert cdac._n_therm == 4
        assert cdac._n_binary == 4

    def test_n_bits_property(self):
        """Line 490: n_bits property."""
        cdac = SegmentedCDAC(n_bits=8, n_therm=4)
        assert cdac.n_bits == 8

    def test_v_ref_property(self):
        """Line 494: v_ref property."""
        cdac = SegmentedCDAC(n_bits=8, v_ref=2.0, n_therm=4)
        assert cdac.v_ref == pytest.approx(2.0)

    def test_cap_weights_property(self):
        """Line 494: cap_weights property delegates to internal CDAC."""
        cdac = SegmentedCDAC(n_bits=8, n_therm=4)
        w = cdac.cap_weights
        assert isinstance(w, np.ndarray)
        assert len(w) > 0

    def test_cap_total_property(self):
        """Line 498: cap_total property."""
        cdac = SegmentedCDAC(n_bits=8, n_therm=4)
        assert cdac.cap_total > 0

    def test_get_voltage_out_of_range_raises(self):
        """Line 510: code out of [0, 2^n_bits - 1] must raise."""
        cdac = SegmentedCDAC(n_bits=8, n_therm=4)
        with pytest.raises(ValueError, match="out of range"):
            cdac.get_voltage(-1)
        with pytest.raises(ValueError, match="out of range"):
            cdac.get_voltage(256)

    def test_get_voltage_all_zeros(self):
        """Line 510 path: code=0 should return (0.0, 0.0)."""
        cdac = SegmentedCDAC(n_bits=8, n_therm=4, cap_mismatch=0.0)
        v, zero = cdac.get_voltage(0)
        assert zero == 0.0
        assert v == pytest.approx(0.0)

    def test_get_voltage_all_ones(self):
        """Line 536: get_voltage for max code."""
        cdac = SegmentedCDAC(n_bits=8, n_therm=4, cap_mismatch=0.0)
        v, zero = cdac.get_voltage(255)
        assert zero == 0.0
        assert 0.0 <= v <= 1.0

    def test_repr(self):
        """Line 536: __repr__ coverage."""
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        r = repr(cdac)
        assert 'SegmentedCDAC' in r
        assert 'n_bits=8' in r
        assert 'n_therm=4' in r


# ===========================================================================
# DifferentialCDAC — line 629 (v_ref property)
# ===========================================================================

class TestDifferentialCDACProperties:

    def test_v_ref_property(self):
        """Line 629: v_ref property on DifferentialCDAC."""
        cdac = DifferentialCDAC(n_bits=4, v_ref=2.5)
        assert cdac.v_ref == pytest.approx(2.5)

    def test_cap_total_pos_and_neg(self):
        cdac = DifferentialCDAC(n_bits=4, v_ref=1.0, cap_mismatch=0.0)
        assert cdac.cap_total > 0
        assert cdac.cap_total_neg > 0

    def test_cap_weights_neg(self):
        cdac = DifferentialCDAC(n_bits=4, v_ref=1.0, cap_mismatch=0.0)
        w = cdac.cap_weights_neg
        assert isinstance(w, np.ndarray)
        assert len(w) == 4
