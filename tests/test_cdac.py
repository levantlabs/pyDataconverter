"""
Tests for CDAC components (SingleEndedCDAC, DifferentialCDAC).

Supplements the CDAC tests in test_SARADC.py with additional edge cases.
"""

import numpy as np
import pytest
from pyDataconverter.components.cdac import (
    CDACBase, SingleEndedCDAC, DifferentialCDAC,
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
