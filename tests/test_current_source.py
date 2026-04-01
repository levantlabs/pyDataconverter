"""
Tests for current source components (IdealCurrentSource, CurrentSourceArray).

Supplements the tests in test_CurrentSteeringDAC.py with edge cases.
"""

import numpy as np
import pytest
from pyDataconverter.components.current_source import (
    UnitCurrentSourceBase, IdealCurrentSource, CurrentSourceArray,
)


class TestIdealCurrentSourceEdgeCases:

    def test_very_small_nominal(self):
        src = IdealCurrentSource(i_nominal=1e-12)
        assert src.i_nominal == pytest.approx(1e-12)

    def test_integer_nominal_accepted(self):
        src = IdealCurrentSource(i_nominal=1)
        assert src.i_nominal == pytest.approx(1.0)

    def test_zero_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCurrentSource(i_nominal=0.0)

    def test_negative_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCurrentSource(i_nominal=-1e-6)

    def test_string_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCurrentSource(i_nominal="100e-6")

    def test_string_mismatch_raises(self):
        with pytest.raises(ValueError):
            IdealCurrentSource(i_nominal=100e-6, mismatch="0.01")

    def test_mismatch_property(self):
        src = IdealCurrentSource(i_nominal=100e-6, mismatch=0.02)
        assert src.mismatch == 0.02

    def test_repr_format(self):
        src = IdealCurrentSource(i_nominal=100e-6, mismatch=0.01)
        r = repr(src)
        assert 'IdealCurrentSource' in r
        assert 'i_nominal=' in r

    def test_base_class_repr(self):
        src = IdealCurrentSource()
        base_repr = UnitCurrentSourceBase.__repr__(src)
        assert 'IdealCurrentSource' in base_repr


class TestCurrentSourceArrayEdgeCases:

    def test_pure_therm_1_bit(self):
        arr = CurrentSourceArray(n_therm_bits=1, n_binary_bits=0, i_unit=100e-6)
        assert len(arr.therm_sources) == 1
        assert len(arr.binary_sources) == 0

    def test_pure_binary_1_bit(self):
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=1, i_unit=100e-6)
        assert len(arr.therm_sources) == 0
        assert len(arr.binary_sources) == 1

    def test_n_bits_property(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=5, i_unit=100e-6)
        assert arr.n_bits == 8

    def test_negative_therm_bits_raises(self):
        with pytest.raises(ValueError):
            CurrentSourceArray(n_therm_bits=-1, n_binary_bits=4, i_unit=100e-6)

    def test_negative_binary_bits_raises(self):
        with pytest.raises(ValueError):
            CurrentSourceArray(n_therm_bits=4, n_binary_bits=-1, i_unit=100e-6)

    def test_zero_i_unit_raises(self):
        with pytest.raises(ValueError):
            CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit=0.0)

    def test_negative_i_unit_raises(self):
        with pytest.raises(ValueError):
            CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit=-100e-6)

    def test_string_i_unit_raises(self):
        with pytest.raises(ValueError):
            CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit="100e-6")

    def test_negative_mismatch_raises(self):
        with pytest.raises(ValueError):
            CurrentSourceArray(n_therm_bits=3, n_binary_bits=0,
                               i_unit=100e-6, current_mismatch=-0.01)

    def test_invalid_source_class_raises(self):
        with pytest.raises(TypeError):
            CurrentSourceArray(n_therm_bits=3, n_binary_bits=0,
                               i_unit=100e-6, source_class=int)

    def test_negative_therm_index_raises(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit=100e-6)
        with pytest.raises(ValueError):
            arr.get_current(-1, np.array([]))

    def test_therm_unit_current_scales_with_binary_bits(self):
        """Thermometer unit current = 2^n_binary_bits * i_unit."""
        i_unit = 100e-6
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=4, i_unit=i_unit)
        for src in arr.therm_sources:
            assert src.i_nominal == pytest.approx(16 * i_unit)

    def test_repr_format(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=4, i_unit=100e-6)
        r = repr(arr)
        assert 'CurrentSourceArray' in r
        assert 'n_therm_bits=3' in r
        assert 'n_binary_bits=4' in r
