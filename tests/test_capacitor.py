"""
Tests for capacitor components (UnitCapacitorBase, IdealCapacitor).

Supplements the capacitor tests in test_CurrentSteeringDAC.py with
additional edge cases and boundary conditions.
"""

import numpy as np
import pytest
from pyDataconverter.components.capacitor import UnitCapacitorBase, IdealCapacitor


class TestIdealCapacitorEdgeCases:

    def test_very_small_nominal(self):
        cap = IdealCapacitor(c_nominal=1e-18)
        assert cap.c_nominal == pytest.approx(1e-18)
        assert cap.capacitance == pytest.approx(1e-18)

    def test_very_large_nominal(self):
        cap = IdealCapacitor(c_nominal=1e6)
        assert cap.c_nominal == pytest.approx(1e6)

    def test_integer_nominal_accepted(self):
        cap = IdealCapacitor(c_nominal=2)
        assert cap.c_nominal == pytest.approx(2.0)

    def test_negative_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCapacitor(c_nominal=-1.0)

    def test_zero_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCapacitor(c_nominal=0.0)

    def test_string_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCapacitor(c_nominal="1.0")

    def test_string_mismatch_raises(self):
        with pytest.raises(ValueError):
            IdealCapacitor(c_nominal=1.0, mismatch="0.01")

    def test_zero_mismatch_exact_nominal(self):
        cap = IdealCapacitor(c_nominal=4.0, mismatch=0.0)
        assert cap.capacitance == 4.0

    def test_mismatch_property(self):
        cap = IdealCapacitor(c_nominal=1.0, mismatch=0.05)
        assert cap.mismatch == 0.05

    def test_mismatch_statistical_distribution(self):
        """With many draws, mean capacitance should be near c_nominal."""
        np.random.seed(42)
        caps = [IdealCapacitor(c_nominal=1.0, mismatch=0.01) for _ in range(1000)]
        mean_cap = np.mean([c.capacitance for c in caps])
        assert abs(mean_cap - 1.0) < 0.005

    def test_repr_contains_values(self):
        cap = IdealCapacitor(c_nominal=2.0, mismatch=0.01)
        r = repr(cap)
        assert 'IdealCapacitor' in r
        assert 'c_nominal=' in r
        assert 'mismatch=' in r
        assert 'capacitance=' in r

    def test_is_unit_capacitor_base(self):
        assert isinstance(IdealCapacitor(), UnitCapacitorBase)

    def test_base_class_repr(self):
        """UnitCapacitorBase.__repr__ is accessible via super()."""
        cap = IdealCapacitor(c_nominal=1.0)
        # IdealCapacitor overrides __repr__, but base class one should also work
        base_repr = UnitCapacitorBase.__repr__(cap)
        assert 'IdealCapacitor' in base_repr
