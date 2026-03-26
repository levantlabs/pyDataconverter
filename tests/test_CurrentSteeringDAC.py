"""
Tests for decoder, capacitor, current source, and CurrentSteeringDAC components.
"""

import numpy as np
import pytest

from pyDataconverter.components.decoder import (
    DecoderBase, BinaryDecoder, ThermometerDecoder, SegmentedDecoder,
)
from pyDataconverter.components.capacitor import UnitCapacitorBase, IdealCapacitor
from pyDataconverter.components.current_source import (
    UnitCurrentSourceBase, IdealCurrentSource, CurrentSourceArray,
)
from pyDataconverter.architectures.CurrentSteeringDAC import CurrentSteeringDAC
from pyDataconverter.dataconverter import OutputType


# ===========================================================================
# BinaryDecoder
# ===========================================================================

class TestBinaryDecoder:

    def test_n_therm_bits_is_zero(self):
        d = BinaryDecoder(n_bits=4)
        assert d.n_therm_bits == 0

    def test_n_binary_bits_equals_n_bits(self):
        d = BinaryDecoder(n_bits=4)
        assert d.n_binary_bits == 4

    def test_decode_zero(self):
        d = BinaryDecoder(n_bits=4)
        therm, bits = d.decode(0)
        assert therm == 0
        np.testing.assert_array_equal(bits, [0., 0., 0., 0.])

    def test_decode_max(self):
        d = BinaryDecoder(n_bits=4)
        therm, bits = d.decode(15)
        assert therm == 0
        np.testing.assert_array_equal(bits, [1., 1., 1., 1.])

    def test_decode_msb_first(self):
        d = BinaryDecoder(n_bits=4)
        _, bits = d.decode(0b1000)   # 8
        np.testing.assert_array_equal(bits, [1., 0., 0., 0.])

    def test_decode_known_code(self):
        d = BinaryDecoder(n_bits=4)
        _, bits = d.decode(0b1011)   # 11
        np.testing.assert_array_equal(bits, [1., 0., 1., 1.])

    def test_invalid_code_raises(self):
        d = BinaryDecoder(n_bits=4)
        with pytest.raises(ValueError):
            d.decode(16)

    def test_negative_code_raises(self):
        d = BinaryDecoder(n_bits=4)
        with pytest.raises(ValueError):
            d.decode(-1)

    def test_float_code_raises(self):
        d = BinaryDecoder(n_bits=4)
        with pytest.raises(TypeError):
            d.decode(3.0)

    def test_invalid_n_bits_raises(self):
        with pytest.raises(ValueError):
            BinaryDecoder(n_bits=0)


# ===========================================================================
# ThermometerDecoder
# ===========================================================================

class TestThermometerDecoder:

    def test_n_therm_bits_equals_n_bits(self):
        d = ThermometerDecoder(n_bits=3)
        assert d.n_therm_bits == 3

    def test_n_binary_bits_is_zero(self):
        d = ThermometerDecoder(n_bits=3)
        assert d.n_binary_bits == 0

    def test_n_elements(self):
        d = ThermometerDecoder(n_bits=3)
        assert d.n_elements == 7

    def test_decode_zero(self):
        d = ThermometerDecoder(n_bits=3)
        therm, bits = d.decode(0)
        assert therm == 0
        assert len(bits) == 0

    def test_decode_max(self):
        d = ThermometerDecoder(n_bits=3)
        therm, bits = d.decode(7)
        assert therm == 7
        assert len(bits) == 0

    def test_decode_midscale(self):
        d = ThermometerDecoder(n_bits=3)
        therm, _ = d.decode(4)
        assert therm == 4

    def test_binary_bits_always_empty(self):
        d = ThermometerDecoder(n_bits=4)
        for code in range(16):
            _, bits = d.decode(code)
            assert len(bits) == 0

    def test_therm_index_equals_code(self):
        d = ThermometerDecoder(n_bits=4)
        for code in range(16):
            therm, _ = d.decode(code)
            assert therm == code

    def test_n_bits_too_large_raises(self):
        with pytest.raises(ValueError):
            ThermometerDecoder(n_bits=17)


# ===========================================================================
# SegmentedDecoder
# ===========================================================================

class TestSegmentedDecoder:

    def test_n_bits_stored(self):
        d = SegmentedDecoder(n_bits=8, n_therm_bits=4)
        assert d.n_bits == 8

    def test_n_therm_bits_stored(self):
        d = SegmentedDecoder(n_bits=8, n_therm_bits=4)
        assert d.n_therm_bits == 4

    def test_n_binary_bits_derived(self):
        d = SegmentedDecoder(n_bits=8, n_therm_bits=4)
        assert d.n_binary_bits == 4

    def test_n_therm_elements(self):
        d = SegmentedDecoder(n_bits=8, n_therm_bits=4)
        assert d.n_therm_elements == 15   # 2^4 - 1

    def test_zero_therm_bits_is_binary(self):
        d = SegmentedDecoder(n_bits=4, n_therm_bits=0)
        therm, bits = d.decode(0b1011)
        assert therm == 0
        assert len(bits) == 4

    def test_all_therm_bits_is_thermometer(self):
        d = SegmentedDecoder(n_bits=4, n_therm_bits=4)
        therm, bits = d.decode(7)
        assert therm == 7
        assert len(bits) == 0

    def test_split_upper_lower(self):
        d = SegmentedDecoder(n_bits=6, n_therm_bits=4)
        therm, bits = d.decode(0b101101)   # 45 = upper 0b1011=11, lower 0b01
        assert therm == 0b1011
        np.testing.assert_array_equal(bits, [0., 1.])

    def test_binary_bits_length(self):
        d = SegmentedDecoder(n_bits=8, n_therm_bits=3)
        for code in [0, 100, 255]:
            _, bits = d.decode(code)
            assert len(bits) == 5

    def test_therm_index_range(self):
        d = SegmentedDecoder(n_bits=6, n_therm_bits=3)
        for code in range(64):
            therm, _ = d.decode(code)
            assert 0 <= therm <= 7   # 2^3 - 1

    def test_invalid_n_therm_bits_raises(self):
        with pytest.raises(ValueError):
            SegmentedDecoder(n_bits=4, n_therm_bits=5)

    def test_n_therm_bits_too_large_raises(self):
        with pytest.raises(ValueError):
            SegmentedDecoder(n_bits=17, n_therm_bits=17)

    def test_is_decoder_base(self):
        assert isinstance(SegmentedDecoder(n_bits=4, n_therm_bits=2), DecoderBase)


# ===========================================================================
# IdealCapacitor
# ===========================================================================

class TestIdealCapacitor:

    def test_default_nominal(self):
        cap = IdealCapacitor()
        assert cap.c_nominal == 1.0

    def test_nominal_stored(self):
        cap = IdealCapacitor(c_nominal=10e-15)
        assert cap.c_nominal == pytest.approx(10e-15)

    def test_no_mismatch_equals_nominal(self):
        cap = IdealCapacitor(c_nominal=1.0, mismatch=0.0)
        assert cap.capacitance == pytest.approx(1.0)

    def test_mismatch_changes_capacitance(self):
        np.random.seed(42)
        cap = IdealCapacitor(c_nominal=1.0, mismatch=0.01)
        assert cap.capacitance != pytest.approx(1.0)

    def test_capacitance_positive(self):
        np.random.seed(0)
        for _ in range(50):
            cap = IdealCapacitor(c_nominal=1.0, mismatch=0.05)
            assert cap.capacitance > 0

    def test_non_positive_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCapacitor(c_nominal=0.0)

    def test_negative_mismatch_raises(self):
        with pytest.raises(ValueError):
            IdealCapacitor(mismatch=-0.01)

    def test_is_base_class(self):
        assert isinstance(IdealCapacitor(), UnitCapacitorBase)


# ===========================================================================
# IdealCurrentSource
# ===========================================================================

class TestIdealCurrentSource:

    def test_default_nominal(self):
        src = IdealCurrentSource()
        assert src.i_nominal == pytest.approx(100e-6)

    def test_no_mismatch_equals_nominal(self):
        src = IdealCurrentSource(i_nominal=100e-6, mismatch=0.0)
        assert src.current == pytest.approx(100e-6)

    def test_mismatch_changes_current(self):
        np.random.seed(7)
        src = IdealCurrentSource(i_nominal=100e-6, mismatch=0.01)
        assert src.current != pytest.approx(100e-6)

    def test_current_positive(self):
        np.random.seed(0)
        for _ in range(50):
            src = IdealCurrentSource(i_nominal=100e-6, mismatch=0.05)
            assert src.current > 0

    def test_non_positive_nominal_raises(self):
        with pytest.raises(ValueError):
            IdealCurrentSource(i_nominal=0.0)

    def test_negative_mismatch_raises(self):
        with pytest.raises(ValueError):
            IdealCurrentSource(mismatch=-0.01)

    def test_is_base_class(self):
        assert isinstance(IdealCurrentSource(), UnitCurrentSourceBase)


# ===========================================================================
# CurrentSourceArray
# ===========================================================================

class TestCurrentSourceArray:

    def test_therm_only_element_count(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit=100e-6)
        assert len(arr.therm_sources) == 7   # 2^3 - 1
        assert len(arr.binary_sources) == 0

    def test_binary_only_element_count(self):
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=4, i_unit=100e-6)
        assert len(arr.therm_sources) == 0
        assert len(arr.binary_sources) == 4

    def test_segmented_element_counts(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=3, i_unit=100e-6)
        assert len(arr.therm_sources) == 7
        assert len(arr.binary_sources) == 3

    def test_therm_sources_nominal_equal(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit=50e-6)
        for src in arr.therm_sources:
            assert src.i_nominal == pytest.approx(50e-6)

    def test_binary_sources_binary_weighted(self):
        i_unit = 100e-6
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=4, i_unit=i_unit)
        for j, src in enumerate(arr.binary_sources):
            assert src.i_nominal == pytest.approx((2 ** j) * i_unit)

    def test_i_total_constant_ideal(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=3, i_unit=100e-6)
        total = arr.i_total
        for therm_idx in range(8):
            for code in range(8):
                bits = np.array([(code >> (2 - k)) & 1 for k in range(3)], dtype=float)
                _, i_tot = arr.get_current(therm_idx, bits)
                assert i_tot == pytest.approx(total)

    def test_get_current_zero_code(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=3, i_unit=100e-6)
        i_sel, _ = arr.get_current(0, np.zeros(3))
        assert i_sel == pytest.approx(0.0)

    def test_get_current_full_therm(self):
        i_unit = 100e-6
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit=i_unit)
        i_sel, i_tot = arr.get_current(7, np.array([]))
        assert i_sel == pytest.approx(i_tot)

    def test_get_current_binary_lsb(self):
        i_unit = 100e-6
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=4, i_unit=i_unit)
        # code=1 → only LSB source switched on
        i_sel, _ = arr.get_current(0, np.array([0., 0., 0., 1.]))
        assert i_sel == pytest.approx(i_unit)

    def test_get_current_binary_msb(self):
        i_unit = 100e-6
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=4, i_unit=i_unit)
        # code=8 → only MSB source switched on (weight 2^3 * i_unit)
        i_sel, _ = arr.get_current(0, np.array([1., 0., 0., 0.]))
        assert i_sel == pytest.approx(8 * i_unit)

    def test_get_current_monotone_therm(self):
        arr = CurrentSourceArray(n_therm_bits=4, n_binary_bits=0, i_unit=100e-6)
        currents = [arr.get_current(k, np.array([]))[0] for k in range(16)]
        assert all(currents[i] <= currents[i + 1] for i in range(len(currents) - 1))

    def test_invalid_therm_index_raises(self):
        arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=0, i_unit=100e-6)
        with pytest.raises(ValueError):
            arr.get_current(8, np.array([]))

    def test_invalid_binary_bits_length_raises(self):
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=4, i_unit=100e-6)
        with pytest.raises(ValueError):
            arr.get_current(0, np.array([1., 0.]))

    def test_both_zero_raises(self):
        with pytest.raises(ValueError):
            CurrentSourceArray(n_therm_bits=0, n_binary_bits=0, i_unit=100e-6)

    def test_mismatch_changes_currents(self):
        np.random.seed(42)
        arr_ideal    = CurrentSourceArray(n_therm_bits=4, n_binary_bits=0, i_unit=100e-6)
        np.random.seed(42)
        arr_mismatch = CurrentSourceArray(n_therm_bits=4, n_binary_bits=0,
                                          i_unit=100e-6, current_mismatch=0.01)
        currents_ideal    = [s.current for s in arr_ideal.therm_sources]
        currents_mismatch = [s.current for s in arr_mismatch.therm_sources]
        assert not np.allclose(currents_ideal, currents_mismatch)


# ===========================================================================
# CurrentSteeringDAC — construction
# ===========================================================================

class TestCurrentSteeringDACConstruction:

    def test_default_binary_mode(self):
        dac = CurrentSteeringDAC(n_bits=8)
        assert dac.n_therm_bits == 0
        assert dac.n_binary_bits == 8

    def test_thermometer_mode(self):
        dac = CurrentSteeringDAC(n_bits=4, n_therm_bits=4)
        assert dac.n_therm_bits == 4
        assert dac.n_binary_bits == 0

    def test_segmented_mode(self):
        dac = CurrentSteeringDAC(n_bits=8, n_therm_bits=4)
        assert dac.n_therm_bits == 4
        assert dac.n_binary_bits == 4

    def test_i_unit_stored(self):
        dac = CurrentSteeringDAC(n_bits=8, i_unit=50e-6)
        assert dac.i_unit == pytest.approx(50e-6)

    def test_r_load_stored(self):
        dac = CurrentSteeringDAC(n_bits=8, r_load=2000.0)
        assert dac.r_load == pytest.approx(2000.0)

    def test_invalid_n_therm_bits_raises(self):
        with pytest.raises(ValueError):
            CurrentSteeringDAC(n_bits=4, n_therm_bits=5)

    def test_invalid_i_unit_raises(self):
        with pytest.raises(ValueError):
            CurrentSteeringDAC(n_bits=4, i_unit=0.0)

    def test_invalid_r_load_raises(self):
        with pytest.raises(ValueError):
            CurrentSteeringDAC(n_bits=4, r_load=-100.0)

    def test_negative_mismatch_raises(self):
        with pytest.raises(ValueError):
            CurrentSteeringDAC(n_bits=4, current_mismatch=-0.01)

    def test_custom_decoder_accepted(self):
        decoder = BinaryDecoder(n_bits=4)
        dac = CurrentSteeringDAC(n_bits=4, decoder=decoder)
        assert dac.decoder is decoder

    def test_custom_decoder_wrong_nbits_raises(self):
        decoder = BinaryDecoder(n_bits=3)
        with pytest.raises(ValueError):
            CurrentSteeringDAC(n_bits=4, decoder=decoder)

    def test_custom_array_accepted(self):
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=4, i_unit=100e-6)
        dac = CurrentSteeringDAC(n_bits=4, current_array=arr)
        assert dac.current_array is arr

    def test_custom_array_wrong_nbits_raises(self):
        arr = CurrentSourceArray(n_therm_bits=0, n_binary_bits=3, i_unit=100e-6)
        with pytest.raises(ValueError):
            CurrentSteeringDAC(n_bits=4, current_array=arr)


# ===========================================================================
# CurrentSteeringDAC — ideal conversion (binary, single-ended)
# ===========================================================================

class TestCurrentSteeringDACBinarySingleEnded:

    def setup_method(self):
        self.i_unit = 100e-6
        self.r_load = 1000.0
        self.dac = CurrentSteeringDAC(
            n_bits=4, n_therm_bits=0,
            i_unit=self.i_unit, r_load=self.r_load,
            output_type=OutputType.SINGLE,
        )

    def test_zero_gives_zero_output(self):
        v = self.dac.convert(0)
        assert v == pytest.approx(0.0)

    def test_max_code_gives_max_output(self):
        v = self.dac.convert(15)
        # All 4 binary sources on: (1+2+4+8)*i_unit*r_load
        expected = (1 + 2 + 4 + 8) * self.i_unit * self.r_load
        assert v == pytest.approx(expected)

    def test_lsb_code_gives_lsb_output(self):
        v = self.dac.convert(1)
        assert v == pytest.approx(self.i_unit * self.r_load)

    def test_output_monotone(self):
        codes = list(range(16))
        voltages = [self.dac.convert(c) for c in codes]
        assert all(voltages[i] <= voltages[i + 1] for i in range(len(voltages) - 1))

    def test_output_is_float(self):
        assert isinstance(self.dac.convert(7), float)

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError):
            self.dac.convert(16)


# ===========================================================================
# CurrentSteeringDAC — differential output
# ===========================================================================

class TestCurrentSteeringDACDifferential:

    def setup_method(self):
        self.i_unit = 100e-6
        self.r_load = 1000.0
        self.dac = CurrentSteeringDAC(
            n_bits=4, n_therm_bits=0,
            i_unit=self.i_unit, r_load=self.r_load,
            output_type=OutputType.DIFFERENTIAL,
        )

    def test_returns_tuple(self):
        result = self.dac.convert(7)
        assert isinstance(result, tuple) and len(result) == 2

    def test_zero_code_all_current_to_neg(self):
        v_pos, v_neg = self.dac.convert(0)
        assert v_pos == pytest.approx(0.0)
        assert v_neg == pytest.approx(self.dac.i_total * self.r_load)

    def test_max_code_all_current_to_pos(self):
        v_pos, v_neg = self.dac.convert(15)
        assert v_pos == pytest.approx(self.dac.i_total * self.r_load)
        assert v_neg == pytest.approx(0.0)

    def test_v_pos_plus_v_neg_is_constant(self):
        """i_total is constant so v_pos + v_neg = i_total * r_load for all codes."""
        i_total_v = self.dac.i_total * self.r_load
        for code in range(16):
            v_pos, v_neg = self.dac.convert(code)
            assert v_pos + v_neg == pytest.approx(i_total_v)

    def test_differential_output_monotone(self):
        diffs = [self.dac.convert(c)[0] - self.dac.convert(c)[1] for c in range(16)]
        assert all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1))


# ===========================================================================
# CurrentSteeringDAC — thermometer mode
# ===========================================================================

class TestCurrentSteeringDACThermometer:

    def setup_method(self):
        self.i_unit = 100e-6
        self.r_load = 1000.0
        self.dac = CurrentSteeringDAC(
            n_bits=4, n_therm_bits=4,
            i_unit=self.i_unit, r_load=self.r_load,
            output_type=OutputType.SINGLE,
        )

    def test_zero_gives_zero(self):
        assert self.dac.convert(0) == pytest.approx(0.0)

    def test_unit_steps_equal(self):
        """Each code step increases output by exactly i_unit * r_load."""
        voltages = [self.dac.convert(c) for c in range(16)]
        diffs = np.diff(voltages)
        lsb_v = self.i_unit * self.r_load
        np.testing.assert_allclose(diffs, lsb_v, rtol=1e-9)

    def test_monotone(self):
        voltages = [self.dac.convert(c) for c in range(16)]
        assert all(voltages[i] <= voltages[i + 1] for i in range(len(voltages) - 1))


# ===========================================================================
# CurrentSteeringDAC — segmented mode
# ===========================================================================

class TestCurrentSteeringDACSegmented:

    def setup_method(self):
        self.dac = CurrentSteeringDAC(
            n_bits=8, n_therm_bits=4,
            i_unit=100e-6, r_load=1000.0,
            output_type=OutputType.SINGLE,
        )

    def test_output_monotone(self):
        voltages = [self.dac.convert(c) for c in range(256)]
        assert all(voltages[i] <= voltages[i + 1] for i in range(len(voltages) - 1))

    def test_zero_gives_zero(self):
        assert self.dac.convert(0) == pytest.approx(0.0)

    def test_max_code_gives_max(self):
        v_max = self.dac.convert(255)
        v_mid = self.dac.convert(127)
        assert v_max > v_mid > 0


# ===========================================================================
# CurrentSteeringDAC — mismatch
# ===========================================================================

class TestCurrentSteeringDACMismatch:

    def test_mismatch_breaks_uniform_steps(self):
        np.random.seed(42)
        dac = CurrentSteeringDAC(n_bits=4, n_therm_bits=4,
                                 i_unit=100e-6, r_load=1000.0,
                                 current_mismatch=0.01,
                                 output_type=OutputType.SINGLE)
        voltages = [dac.convert(c) for c in range(16)]
        diffs = np.diff(voltages)
        # Steps are no longer perfectly equal
        assert not np.allclose(diffs, diffs[0])

    def test_mismatch_still_monotone_for_small_sigma(self):
        """With small mismatch, monotonicity is expected to hold."""
        np.random.seed(0)
        dac = CurrentSteeringDAC(n_bits=4, n_therm_bits=4,
                                 i_unit=100e-6, r_load=1000.0,
                                 current_mismatch=0.001,
                                 output_type=OutputType.SINGLE)
        voltages = [dac.convert(c) for c in range(16)]
        assert all(voltages[i] <= voltages[i + 1] for i in range(len(voltages) - 1))

    def test_ideal_and_mismatch_differ(self):
        np.random.seed(7)
        dac_ideal    = CurrentSteeringDAC(n_bits=8, n_therm_bits=4, i_unit=100e-6)
        np.random.seed(7)
        dac_mismatch = CurrentSteeringDAC(n_bits=8, n_therm_bits=4, i_unit=100e-6,
                                          current_mismatch=0.005)
        v_ideal    = [dac_ideal.convert(c) for c in range(0, 256, 16)]
        v_mismatch = [dac_mismatch.convert(c) for c in range(0, 256, 16)]
        assert v_ideal != v_mismatch


# ===========================================================================
# CurrentSteeringDAC — repr
# ===========================================================================

class TestCurrentSteeringDACRepr:

    def test_repr_binary_mode(self):
        dac = CurrentSteeringDAC(n_bits=8)
        assert 'binary' in repr(dac)

    def test_repr_thermometer_mode(self):
        dac = CurrentSteeringDAC(n_bits=4, n_therm_bits=4)
        assert 'thermometer' in repr(dac)

    def test_repr_segmented_mode(self):
        dac = CurrentSteeringDAC(n_bits=8, n_therm_bits=4)
        assert 'segmented' in repr(dac)

    def test_repr_contains_n_bits(self):
        dac = CurrentSteeringDAC(n_bits=12)
        assert 'n_bits=12' in repr(dac)

    def test_repr_mismatch_shown_when_nonzero(self):
        dac = CurrentSteeringDAC(n_bits=4, current_mismatch=0.01)
        assert 'current_mismatch' in repr(dac)

    def test_repr_mismatch_hidden_when_zero(self):
        dac = CurrentSteeringDAC(n_bits=4, current_mismatch=0.0)
        assert 'current_mismatch' not in repr(dac)
