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

    def test_get_voltage_negative_code_raises(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        with pytest.raises(ValueError, match="out of range"):
            cdac.get_voltage(-1)

    def test_get_voltage_overflow_code_raises(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        with pytest.raises(ValueError, match="out of range"):
            cdac.get_voltage(16)

    def test_get_voltage_max_code_ok(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        v, _ = cdac.get_voltage(15)
        assert v > 0


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

    def test_get_voltage_negative_code_raises(self):
        cdac = DifferentialCDAC(n_bits=4, v_ref=1.0)
        with pytest.raises(ValueError, match="out of range"):
            cdac.get_voltage(-1)

    def test_get_voltage_overflow_code_raises(self):
        cdac = DifferentialCDAC(n_bits=4, v_ref=1.0)
        with pytest.raises(ValueError, match="out of range"):
            cdac.get_voltage(16)

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


# ===========================================================================
# apply_mismatch — Monte Carlo workflow
# ===========================================================================

class TestApplyMismatchSingleEnded:

    def test_preserves_nominal_topology(self):
        """Nominal c_nominal of each cap is unchanged across re-draws."""
        cdac = SingleEndedCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        nominals_before = [c.c_nominal for c in cdac.cap_instances]
        cdac.apply_mismatch(0.05, seed=1)
        cdac.apply_mismatch(0.10, seed=2)
        nominals_after = [c.c_nominal for c in cdac.cap_instances]
        assert nominals_before == nominals_after

    def test_effective_caps_change(self):
        cdac = SingleEndedCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        weights_ideal = cdac.cap_weights.copy()
        cdac.apply_mismatch(0.05, seed=1)
        weights_drawn = cdac.cap_weights
        assert not np.allclose(weights_ideal, weights_drawn)

    def test_cap_total_refreshed(self):
        cdac = SingleEndedCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        total_ideal = cdac.cap_total
        cdac.apply_mismatch(0.05, seed=1)
        total_drawn = cdac.cap_total
        expected = float(np.sum(cdac.cap_weights) + 1.0)
        assert total_drawn == pytest.approx(expected)
        assert total_drawn != total_ideal

    def test_seeded_reproducibility(self):
        c1 = SingleEndedCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        c2 = SingleEndedCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        c1.apply_mismatch(0.03, seed=42)
        c2.apply_mismatch(0.03, seed=42)
        np.testing.assert_array_equal(c1.cap_weights, c2.cap_weights)

    def test_zero_stddev_restores_nominal(self):
        cdac = SingleEndedCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.1)
        cdac.apply_mismatch(0.0)
        nominals = np.array([c.c_nominal for c in cdac.cap_instances])
        np.testing.assert_array_equal(cdac.cap_weights, nominals)

    def test_monte_carlo_loop(self):
        """Reusing one CDAC for many statistical draws produces distinct realizations."""
        cdac = SingleEndedCDAC(n_bits=8, v_ref=1.0, cap_mismatch=0.0)
        realizations = []
        for seed in range(5):
            cdac.apply_mismatch(0.02, seed=seed)
            realizations.append(cdac.cap_weights.copy())
        for i in range(len(realizations)):
            for j in range(i + 1, len(realizations)):
                assert not np.allclose(realizations[i], realizations[j])

    def test_negative_stddev_raises(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        with pytest.raises(ValueError):
            cdac.apply_mismatch(-0.01)


class TestApplyMismatchDifferential:

    def test_both_arrays_redrawn(self):
        cdac = DifferentialCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        pos_before = cdac.cap_weights.copy()
        neg_before = cdac.cap_weights_neg.copy()
        cdac.apply_mismatch(0.05, seed=1)
        assert not np.allclose(cdac.cap_weights, pos_before)
        assert not np.allclose(cdac.cap_weights_neg, neg_before)

    def test_independent_draws(self):
        """Pos and neg arrays must receive different realizations (not correlated)."""
        cdac = DifferentialCDAC(n_bits=8, v_ref=1.0, cap_mismatch=0.0)
        cdac.apply_mismatch(0.05, seed=7)
        assert not np.allclose(cdac.cap_weights, cdac.cap_weights_neg)

    def test_nominal_preserved(self):
        cdac = DifferentialCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        nominals_pos = [c.c_nominal for c in cdac.cap_instances]
        nominals_neg = [c.c_nominal for c in cdac.cap_instances_neg]
        cdac.apply_mismatch(0.1, seed=3)
        assert [c.c_nominal for c in cdac.cap_instances] == nominals_pos
        assert [c.c_nominal for c in cdac.cap_instances_neg] == nominals_neg


class TestApplyMismatchSegmented:

    def test_delegates_to_inner(self):
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=3, cap_mismatch=0.0)
        weights_before = cdac.cap_weights.copy()
        cdac.apply_mismatch(0.05, seed=1)
        assert not np.allclose(cdac.cap_weights, weights_before)

    def test_seeded_reproducibility(self):
        c1 = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=3, cap_mismatch=0.0)
        c2 = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=3, cap_mismatch=0.0)
        c1.apply_mismatch(0.02, seed=42)
        c2.apply_mismatch(0.02, seed=42)
        np.testing.assert_array_equal(c1.cap_weights, c2.cap_weights)


class TestApplyMismatchBaseClassDefault:

    def test_raises_not_implemented(self):
        """CDACBase.apply_mismatch raises by default for non-overriding subclasses."""
        class DummyCDAC(CDACBase):
            @property
            def n_bits(self): return 4
            @property
            def v_ref(self): return 1.0
            @property
            def cap_weights(self): return np.ones(4)
            @property
            def cap_total(self): return 5.0
            def get_voltage(self, code): return (0.0, 0.0)
        with pytest.raises(NotImplementedError):
            DummyCDAC().apply_mismatch(0.01)
