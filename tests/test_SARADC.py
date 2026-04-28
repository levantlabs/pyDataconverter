"""
Tests for SARADC, SingleEndedCDAC, and DifferentialCDAC.
"""

import numpy as np
import pytest
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.components.cdac import (
    CDACBase, SingleEndedCDAC, DifferentialCDAC,
)
from pyDataconverter.components.comparator import DifferentialComparator
from pyDataconverter.dataconverter import InputType


# ===========================================================================
# SingleEndedCDAC
# ===========================================================================

class TestSingleEndedCDACConstruction:

    def test_default_binary_weights(self):
        cdac = SingleEndedCDAC(n_bits=3, v_ref=1.0)
        expected = np.array([4., 2., 1.])
        np.testing.assert_array_equal(cdac.cap_weights, expected)

    def test_cap_total_is_power_of_two(self):
        """For ideal binary weights, cap_total == 2^n_bits."""
        for n in [3, 4, 8]:
            cdac = SingleEndedCDAC(n_bits=n, v_ref=1.0)
            assert cdac.cap_total == 2 ** n

    def test_custom_cap_weights(self):
        weights = np.array([8., 4., 2., 1.])
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0, cap_weights=weights)
        np.testing.assert_array_equal(cdac.cap_weights, weights)

    def test_custom_weights_wrong_length_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=4, v_ref=1.0, cap_weights=[4., 2., 1.])

    def test_custom_weights_non_positive_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=3, v_ref=1.0, cap_weights=[4., 0., 1.])

    def test_negative_cap_mismatch_raises(self):
        with pytest.raises(ValueError):
            SingleEndedCDAC(n_bits=3, v_ref=1.0, cap_mismatch=-0.01)

    def test_cap_mismatch_changes_weights(self):
        np.random.seed(1)
        cdac_ideal    = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        np.random.seed(1)
        cdac_mismatch = SingleEndedCDAC(n_bits=4, v_ref=1.0, cap_mismatch=0.01)
        assert not np.allclose(cdac_ideal.cap_weights, cdac_mismatch.cap_weights)


class TestSingleEndedCDACVoltages:

    def setup_method(self):
        self.cdac = SingleEndedCDAC(n_bits=3, v_ref=1.0)

    def test_code_zero_gives_zero(self):
        v, v_n = self.cdac.get_voltage(0)
        assert v == 0.0
        assert v_n == 0.0

    def test_max_code_below_vref(self):
        v, v_n = self.cdac.get_voltage(7)
        assert v < 1.0
        assert v_n == 0.0

    def test_midscale_is_half_vref(self):
        v, _ = self.cdac.get_voltage(4)
        assert pytest.approx(v) == 0.5

    def test_negative_reference_always_zero(self):
        for code in range(8):
            _, v_n = self.cdac.get_voltage(code)
            assert v_n == 0.0

    def test_voltages_property_shape(self):
        vols = self.cdac.voltages
        assert vols.shape == (8,)

    def test_voltages_property_ascending(self):
        vols = self.cdac.voltages
        assert np.all(np.diff(vols) > 0)

    def test_voltages_ideal_formula(self):
        """Ideal: v[k] = k / 2^N * v_ref."""
        vols = self.cdac.voltages
        expected = np.arange(8) / 8.0
        np.testing.assert_allclose(vols, expected)

    def test_vref_scales_output(self):
        cdac2 = SingleEndedCDAC(n_bits=3, v_ref=2.0)
        v1, _ = self.cdac.get_voltage(4)
        v2, _ = cdac2.get_voltage(4)
        assert pytest.approx(v2) == 2 * v1


# ===========================================================================
# DifferentialCDAC
# ===========================================================================

class TestDifferentialCDACConstruction:

    def test_default_binary_weights(self):
        cdac = DifferentialCDAC(n_bits=3, v_ref=1.0)
        expected = np.array([4., 2., 1.])
        np.testing.assert_array_equal(cdac.cap_weights, expected)

    def test_cap_weights_neg_same_nominal(self):
        """With no mismatch, positive and negative weights are identical."""
        cdac = DifferentialCDAC(n_bits=3, v_ref=1.0, cap_mismatch=0.0)
        np.testing.assert_array_equal(cdac.cap_weights, cdac.cap_weights_neg)

    def test_independent_mismatch_draws(self):
        """With mismatch, positive and negative sides differ (independent draws)."""
        np.random.seed(42)
        cdac = DifferentialCDAC(n_bits=4, v_ref=1.0, cap_mismatch=0.01)
        assert not np.allclose(cdac.cap_weights, cdac.cap_weights_neg)

    def test_cap_total_is_power_of_two_ideal(self):
        for n in [3, 4, 8]:
            cdac = DifferentialCDAC(n_bits=n, v_ref=1.0, cap_mismatch=0.0)
            assert cdac.cap_total == 2 ** n
            assert cdac.cap_total_neg == 2 ** n

    def test_negative_cap_mismatch_raises(self):
        with pytest.raises(ValueError):
            DifferentialCDAC(n_bits=3, v_ref=1.0, cap_mismatch=-0.01)


class TestDifferentialCDACVoltages:

    def setup_method(self):
        self.cdac = DifferentialCDAC(n_bits=3, v_ref=1.0, cap_mismatch=0.0)

    def test_midscale_code_gives_zero_diff(self):
        """Code 4 (midscale) → v_dacp = v_dacn → differential = 0."""
        v_p, v_n = self.cdac.get_voltage(4)
        assert pytest.approx(v_p - v_n) == 0.0

    def test_code_zero_gives_minimum_diff(self):
        """Code 0 → most negative differential output = −v_ref/2."""
        v_p, v_n = self.cdac.get_voltage(0)
        assert pytest.approx(v_p - v_n) == -0.5

    def test_max_code_gives_near_positive_max(self):
        """Code 7 → differential just below +v_ref/2."""
        v_p, v_n = self.cdac.get_voltage(7)
        diff = v_p - v_n
        assert 0.0 < diff < 0.5

    def test_voltages_property_ascending(self):
        vols = self.cdac.voltages
        assert np.all(np.diff(vols) > 0)

    def test_voltages_property_shape(self):
        assert self.cdac.voltages.shape == (8,)

    def test_voltages_range(self):
        vols = self.cdac.voltages
        assert vols[0] == pytest.approx(-0.5)
        assert vols[-1] < 0.5

    def test_voltages_not_symmetric(self):
        """FLOOR mode: code 0 → −v_ref/2, code 2^N−1 → just below +v_ref/2."""
        vols = self.cdac.voltages
        assert vols[0] == pytest.approx(-0.5)
        assert vols[-1] < 0.5   # not +v_ref/2 because of FLOOR convention

    def test_both_rails_in_range(self):
        """v_dacp and v_dacn must both be in [0, v_ref/2]."""
        for code in range(8):
            v_p, v_n = self.cdac.get_voltage(code)
            assert 0.0 <= v_p <= 0.5
            assert 0.0 <= v_n <= 0.5

    def test_mismatch_breaks_linearity(self):
        np.random.seed(7)
        cdac_m = DifferentialCDAC(n_bits=8, v_ref=1.0, cap_mismatch=0.005)
        cdac_i = DifferentialCDAC(n_bits=8, v_ref=1.0, cap_mismatch=0.0)
        assert not np.allclose(cdac_m.voltages, cdac_i.voltages)


# ===========================================================================
# CDACBase interface via isinstance
# ===========================================================================

class TestCDACBaseInterface:

    def test_single_ended_is_cdac_base(self):
        assert isinstance(SingleEndedCDAC(n_bits=3), CDACBase)

    def test_differential_is_cdac_base(self):
        assert isinstance(DifferentialCDAC(n_bits=3), CDACBase)


# ===========================================================================
# CDAC seed reproducibility (added 2026-04-25)
# ===========================================================================

class TestCDACSeed:
    """Construction-time seed parameter on CDAC classes (§3.2 review item)."""

    def test_single_ended_same_seed_matches(self):
        a = SingleEndedCDAC(n_bits=8, cap_mismatch=0.01, seed=42)
        b = SingleEndedCDAC(n_bits=8, cap_mismatch=0.01, seed=42)
        np.testing.assert_array_equal(a.cap_weights, b.cap_weights)

    def test_single_ended_diff_seed_distinguishes(self):
        a = SingleEndedCDAC(n_bits=8, cap_mismatch=0.01, seed=1)
        b = SingleEndedCDAC(n_bits=8, cap_mismatch=0.01, seed=2)
        assert not np.array_equal(a.cap_weights, b.cap_weights)

    def test_single_ended_seed_in_repr_when_set(self):
        cdac = SingleEndedCDAC(n_bits=4, cap_mismatch=0.01, seed=99)
        assert "seed=99" in repr(cdac)

    def test_single_ended_seed_omitted_from_repr_when_none(self):
        cdac = SingleEndedCDAC(n_bits=4, cap_mismatch=0.01)
        assert "seed=" not in repr(cdac)

    def test_differential_same_seed_matches(self):
        a = DifferentialCDAC(n_bits=8, cap_mismatch=0.01, seed=42)
        b = DifferentialCDAC(n_bits=8, cap_mismatch=0.01, seed=42)
        np.testing.assert_array_equal(a.cap_weights, b.cap_weights)
        np.testing.assert_array_equal(a.cap_weights_neg, b.cap_weights_neg)

    def test_differential_pos_neg_independent_within_instance(self):
        """With the same seed, pos and neg arrays must still differ
        (independent draws from the same Generator stream)."""
        cdac = DifferentialCDAC(n_bits=8, cap_mismatch=0.01, seed=42)
        assert not np.array_equal(cdac.cap_weights, cdac.cap_weights_neg)

    def test_differential_seed_in_repr_when_set(self):
        cdac = DifferentialCDAC(n_bits=4, cap_mismatch=0.01, seed=7)
        assert "seed=7" in repr(cdac)

    def test_redundant_sar_seed_propagates(self):
        from pyDataconverter.components.cdac import RedundantSARCDAC
        a = RedundantSARCDAC(n_bits=6, radix=1.85, cap_mismatch=0.01, seed=11)
        b = RedundantSARCDAC(n_bits=6, radix=1.85, cap_mismatch=0.01, seed=11)
        np.testing.assert_array_equal(a.cap_weights, b.cap_weights)
        assert "seed=11" in repr(a)

    def test_split_cap_seed_propagates(self):
        from pyDataconverter.components.cdac import SplitCapCDAC
        a = SplitCapCDAC(n_bits=8, n_msb=4, cap_mismatch=0.01, seed=13)
        b = SplitCapCDAC(n_bits=8, n_msb=4, cap_mismatch=0.01, seed=13)
        np.testing.assert_array_equal(a.cap_weights, b.cap_weights)
        assert "seed=13" in repr(a)

    def test_segmented_seed_propagates(self):
        from pyDataconverter.components.cdac import SegmentedCDAC
        a = SegmentedCDAC(n_bits=8, n_therm=4, cap_mismatch=0.01, seed=17)
        b = SegmentedCDAC(n_bits=8, n_therm=4, cap_mismatch=0.01, seed=17)
        np.testing.assert_array_equal(a.cap_weights, b.cap_weights)
        assert "seed=17" in repr(a)

    def test_seed_none_is_nondeterministic(self):
        """Default seed=None still draws from OS entropy on each construction."""
        a = SingleEndedCDAC(n_bits=8, cap_mismatch=0.01)
        b = SingleEndedCDAC(n_bits=8, cap_mismatch=0.01)
        assert not np.array_equal(a.cap_weights, b.cap_weights)


# ===========================================================================
# SARADC — construction
# ===========================================================================

class TestSARAdcConstruction:

    def test_default_single_ended_cdac(self):
        adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)
        assert isinstance(adc.cdac, SingleEndedCDAC)

    def test_default_differential_cdac(self):
        adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        assert isinstance(adc.cdac, DifferentialCDAC)

    def test_custom_cdac_accepted(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        adc  = SARADC(n_bits=4, v_ref=1.0, cdac=cdac)
        assert adc.cdac is cdac

    def test_custom_cdac_wrong_nbits_raises(self):
        cdac = SingleEndedCDAC(n_bits=3, v_ref=1.0)
        with pytest.raises(ValueError):
            SARADC(n_bits=4, v_ref=1.0, cdac=cdac)

    def test_custom_cdac_wrong_vref_raises(self):
        cdac = SingleEndedCDAC(n_bits=4, v_ref=2.0)
        with pytest.raises(ValueError):
            SARADC(n_bits=4, v_ref=1.0, cdac=cdac)

    def test_custom_cdac_wrong_type_raises(self):
        with pytest.raises(TypeError):
            SARADC(n_bits=4, v_ref=1.0, cdac="not_a_cdac")

    def test_comparator_type_forwarded(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE)
        assert isinstance(adc.comparator, DifferentialComparator)

    def test_negative_noise_rms_raises(self):
        with pytest.raises(ValueError):
            SARADC(n_bits=4, input_type=InputType.SINGLE, noise_rms=-0.001)

    def test_negative_t_jitter_raises(self):
        with pytest.raises(ValueError):
            SARADC(n_bits=4, input_type=InputType.SINGLE, t_jitter=-1e-12)

    def test_negative_cap_mismatch_raises(self):
        with pytest.raises(ValueError):
            SARADC(n_bits=4, input_type=InputType.SINGLE, cap_mismatch=-0.001)


# ===========================================================================
# SARADC — single-ended ideal conversion
# ===========================================================================

class TestSARAdcSingleEndedIdeal:

    def setup_method(self):
        self.adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)

    def test_zero_input_gives_code_zero(self):
        assert self.adc.convert(0.0) == 0

    def test_full_scale_gives_max_code(self):
        assert self.adc.convert(1.0) == 15

    def test_over_range_saturates_high(self):
        assert self.adc.convert(2.0) == 15

    def test_under_range_saturates_low(self):
        assert self.adc.convert(-1.0) == 0

    def test_ramp_monotone(self):
        v_in = np.linspace(0.0, 1.0, 500)
        codes = [self.adc.convert(v) for v in v_in]
        assert all(codes[i] <= codes[i + 1] for i in range(len(codes) - 1))

    def test_output_in_valid_range(self):
        for v in np.linspace(0, 1, 50):
            assert 0 <= self.adc.convert(v) <= 15

    def test_known_mid_bin_code(self):
        """v = 0.35 should land in bin 5 (5/16 = 0.3125, 6/16 = 0.375)."""
        assert self.adc.convert(0.35) == 5

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError):
            self.adc.convert((0.5, 0.0))

    def test_dac_voltages_property(self):
        vols = self.adc.dac_voltages
        assert vols.shape == (16,)
        assert np.all(np.diff(vols) > 0)


# ===========================================================================
# SARADC — differential ideal conversion
# ===========================================================================

class TestSARAdcDifferentialIdeal:

    def setup_method(self):
        self.adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    def test_zero_diff_gives_midscale(self):
        """v_pos == v_neg → code near 2^(N-1)."""
        code = self.adc.convert((0.5, 0.5))
        assert code in (7, 8)

    def test_full_positive_diff_gives_max_code(self):
        assert self.adc.convert((1.0, 0.0)) == 15

    def test_full_negative_diff_gives_min_code(self):
        assert self.adc.convert((0.0, 1.0)) == 0

    def test_ramp_monotone(self):
        v_diff = np.linspace(-0.499, 0.499, 500)
        codes  = [self.adc.convert((v / 2, -v / 2)) for v in v_diff]
        assert all(codes[i] <= codes[i + 1] for i in range(len(codes) - 1))

    def test_output_in_valid_range(self):
        for v in np.linspace(-0.4, 0.4, 50):
            assert 0 <= self.adc.convert((v, 0.0)) <= 15

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError):
            self.adc.convert(0.5)


# ===========================================================================
# SARADC — convert_with_trace
# ===========================================================================

class TestSARAdcTrace:

    def setup_method(self):
        self.adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)

    def test_trace_keys(self):
        trace = self.adc.convert_with_trace(0.37)
        assert set(trace.keys()) == {
            'code', 'sampled_voltage', 'dac_voltages',
            'bit_decisions', 'register_states',
        }

    def test_trace_code_matches_convert(self):
        np.random.seed(0)
        code_direct = self.adc.convert(0.37)
        np.random.seed(0)
        trace = self.adc.convert_with_trace(0.37)
        assert trace['code'] == code_direct

    def test_trace_lengths(self):
        trace = self.adc.convert_with_trace(0.6)
        assert len(trace['dac_voltages'])   == 4
        assert len(trace['bit_decisions'])  == 4
        assert len(trace['register_states']) == 5  # initial + one per bit

    def test_register_states_first_is_zero(self):
        trace = self.adc.convert_with_trace(0.6)
        assert trace['register_states'][0] == 0

    def test_register_states_last_matches_code(self):
        trace = self.adc.convert_with_trace(0.6)
        assert trace['register_states'][-1] == trace['code']

    def test_bit_decisions_consistent_with_register(self):
        """Each register state is consistent with the bit decisions so far."""
        trace = self.adc.convert_with_trace(0.7)
        reg = 0
        for k, (decision, reg_after) in enumerate(
            zip(trace['bit_decisions'], trace['register_states'][1:])
        ):
            trial = reg | (1 << (4 - 1 - k))
            expected_reg = trial if decision else reg
            assert reg_after == expected_reg
            reg = reg_after

    def test_dac_voltages_ascending_then_falling(self):
        """Trial DAC voltages follow the binary search pattern."""
        trace = self.adc.convert_with_trace(0.7)
        # First trial is always v_ref/2 (MSB)
        assert pytest.approx(trace['dac_voltages'][0]) == 0.5

    def test_sampled_voltage_stored(self):
        trace = self.adc.convert_with_trace(0.4)
        assert isinstance(trace['sampled_voltage'], float)


# ===========================================================================
# SARADC — non-idealities
# ===========================================================================

class TestSARAdcNonidealities:

    def test_noise_rms_causes_code_spread(self):
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, noise_rms=0.02)
        codes = {adc.convert(0.5) for _ in range(50)}
        assert len(codes) > 1

    def test_offset_shifts_codes(self):
        adc_ideal  = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        adc_offset = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, offset=0.05)
        v_in = np.linspace(0.1, 0.9, 50)
        codes_ideal  = [adc_ideal.convert(v)  for v in v_in]
        codes_offset = [adc_offset.convert(v) for v in v_in]
        assert codes_ideal != codes_offset

    def test_gain_error_changes_codes(self):
        adc_ideal = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        adc_gain  = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, gain_error=0.02)
        v_in = np.linspace(0.1, 0.9, 50)
        codes_ideal = [adc_ideal.convert(v) for v in v_in]
        codes_gain  = [adc_gain.convert(v)  for v in v_in]
        assert codes_ideal != codes_gain

    def test_comparator_noise_causes_spread(self):
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE,
                     comparator_params={'noise_rms': 0.01})
        codes = {adc.convert(0.5) for _ in range(50)}
        assert len(codes) > 1

    def test_cap_mismatch_breaks_linearity(self):
        # Pass an explicitly-seeded CDAC so the mismatch realisation is
        # deterministic (np.random.seed has no effect on the CDAC anymore;
        # the cap-mismatch draw goes through default_rng(seed=...) per
        # §3.2's reproducibility plumbing).  Seed 42 happens to produce
        # mismatch large enough to perturb at least one of 100 sweep codes.
        adc_ideal = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        cdac_mismatch = SingleEndedCDAC(n_bits=8, v_ref=1.0,
                                        cap_mismatch=0.005, seed=42)
        adc_mismatch = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE,
                              cdac=cdac_mismatch)
        v_in = np.linspace(0.05, 0.95, 100)
        codes_ideal    = [adc_ideal.convert(v)    for v in v_in]
        codes_mismatch = [adc_mismatch.convert(v) for v in v_in]
        assert codes_ideal != codes_mismatch

    def test_no_nonidealities_deterministic(self):
        """Without noise, identical conversions are identical."""
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        codes = [adc.convert(0.5) for _ in range(10)]
        assert len(set(codes)) == 1

    def test_t_jitter_with_dvdt_causes_spread(self):
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, t_jitter=1e-9)
        # 1 MHz sine, amplitude 0.5 V → peak dvdt ≈ 3.14e6 V/s
        # jitter voltage ≈ 3.14e6 * 1e-9 ≈ 3 mV ≈ 0.75 LSB → visible spread
        dvdt = 0.5 * 2 * np.pi * 1e6
        codes = {adc.convert(0.4, dvdt=dvdt) for _ in range(50)}
        assert len(codes) > 1

    def test_t_jitter_without_dvdt_no_effect(self):
        """Jitter has no effect when dvdt=0 (default)."""
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, t_jitter=1e-9)
        codes = [adc.convert(0.5) for _ in range(20)]
        assert len(set(codes)) == 1


# ===========================================================================
# SARADC — reset
# ===========================================================================

class TestSARAdcReset:

    def test_reset_does_not_raise(self):
        adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE,
                     comparator_params={'hysteresis': 0.01})
        for v in np.linspace(0, 1, 20):
            adc.convert(v)
        adc.reset()

    def test_reset_restores_determinism(self):
        """After reset, same input gives same code regardless of history."""
        adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE,
                     comparator_params={'hysteresis': 0.05})
        for _ in range(10):
            adc.convert(1.0)
        adc.reset()
        code_a = adc.convert(0.5)
        adc.reset()
        code_b = adc.convert(0.5)
        assert code_a == code_b


# ===========================================================================
# SARADC — repr
# ===========================================================================

class TestSARAdcRepr:

    def test_repr_contains_class_name(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE)
        assert 'SARADC' in repr(adc)

    def test_repr_contains_n_bits(self):
        adc = SARADC(n_bits=6, input_type=InputType.SINGLE)
        assert 'n_bits=6' in repr(adc)

    def test_repr_contains_input_type(self):
        adc = SARADC(n_bits=4, input_type=InputType.DIFFERENTIAL)
        assert 'DIFFERENTIAL' in repr(adc)

    def test_repr_contains_noise_when_set(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, noise_rms=0.001)
        assert 'noise_rms' in repr(adc)

    def test_repr_omits_noise_when_zero(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, noise_rms=0.0)
        assert 'noise_rms' not in repr(adc)

    def test_repr_contains_offset_when_set(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, offset=0.01)
        assert 'offset=0.01' in repr(adc)

    def test_repr_omits_offset_when_zero(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, offset=0.0)
        assert 'offset=' not in repr(adc)

    def test_repr_contains_gain_error_when_set(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, gain_error=0.02)
        assert 'gain_error=0.02' in repr(adc)

    def test_repr_omits_gain_error_when_zero(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, gain_error=0.0)
        assert 'gain_error=' not in repr(adc)

    def test_repr_contains_t_jitter_when_set(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, t_jitter=1e-12)
        assert 't_jitter=' in repr(adc)

    def test_repr_omits_t_jitter_when_zero(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, t_jitter=0.0)
        assert 't_jitter=' not in repr(adc)

    def test_repr_all_nonidealities(self):
        adc = SARADC(n_bits=4, input_type=InputType.SINGLE, noise_rms=0.001, offset=0.01,
                     gain_error=0.02, t_jitter=1e-12)
        r = repr(adc)
        assert 'noise_rms' in r
        assert 'offset' in r
        assert 'gain_error' in r
        assert 't_jitter' in r


# ===========================================================================
# SARADC — convert_with_trace differential
# ===========================================================================

class TestSARAdcTraceDifferential:

    def setup_method(self):
        self.adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

    def test_trace_differential_keys(self):
        trace = self.adc.convert_with_trace((0.3, 0.1))
        assert set(trace.keys()) == {
            'code', 'sampled_voltage', 'dac_voltages',
            'bit_decisions', 'register_states',
        }

    def test_trace_differential_code_matches_convert(self):
        code_direct = self.adc.convert((0.3, 0.1))
        trace = self.adc.convert_with_trace((0.3, 0.1))
        assert trace['code'] == code_direct

    def test_trace_differential_sampled_voltage_is_diff(self):
        """Sampled voltage should be v_pos - v_neg for differential."""
        trace = self.adc.convert_with_trace((0.7, 0.3))
        assert pytest.approx(trace['sampled_voltage'], abs=1e-10) == 0.4

    def test_trace_differential_lengths(self):
        trace = self.adc.convert_with_trace((0.4, 0.1))
        assert len(trace['dac_voltages']) == 4
        assert len(trace['bit_decisions']) == 4
        assert len(trace['register_states']) == 5

    def test_trace_differential_register_first_zero(self):
        trace = self.adc.convert_with_trace((0.4, 0.1))
        assert trace['register_states'][0] == 0

    def test_trace_differential_register_last_matches_code(self):
        trace = self.adc.convert_with_trace((0.4, 0.1))
        assert trace['register_states'][-1] == trace['code']


# ===========================================================================
# SARADC — convert_with_trace with dvdt
# ===========================================================================

class TestSARAdcTraceWithDvdt:

    def test_trace_with_dvdt_returns_valid(self):
        adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE, t_jitter=1e-9)
        trace = adc.convert_with_trace(0.5, dvdt=1e6)
        assert 'code' in trace
        assert 0 <= trace['code'] <= 15

    def test_trace_dvdt_causes_spread(self):
        """With jitter and dvdt, repeated traces yield different codes."""
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, t_jitter=1e-9)
        dvdt = 0.5 * 2 * np.pi * 1e6
        codes = {adc.convert_with_trace(0.5, dvdt=dvdt)['code']
                 for _ in range(50)}
        assert len(codes) > 1

    def test_trace_dvdt_zero_no_jitter_effect(self):
        """dvdt=0 means jitter has no effect even when t_jitter > 0."""
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, t_jitter=1e-9)
        codes = {adc.convert_with_trace(0.5, dvdt=0.0)['code']
                 for _ in range(20)}
        assert len(codes) == 1


# ===========================================================================
# SARADC — sample_input coverage for differential non-idealities
# ===========================================================================

class TestSARAdcDifferentialNonidealities:

    def test_offset_shifts_differential_codes(self):
        adc_ideal = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        adc_offset = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.DIFFERENTIAL,
                            offset=0.05)
        code_ideal = adc_ideal.convert((0.4, 0.1))
        code_offset = adc_offset.convert((0.4, 0.1))
        assert code_ideal != code_offset

    def test_gain_error_scales_differential(self):
        adc_ideal = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        adc_gain = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.DIFFERENTIAL,
                          gain_error=0.05)
        code_ideal = adc_ideal.convert((0.4, 0.1))
        code_gain = adc_gain.convert((0.4, 0.1))
        assert code_ideal != code_gain

    def test_noise_causes_differential_spread(self):
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.DIFFERENTIAL,
                     noise_rms=0.02)
        codes = {adc.convert((0.3, 0.05)) for _ in range(50)}
        assert len(codes) > 1

    def test_jitter_with_dvdt_differential(self):
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.DIFFERENTIAL,
                     t_jitter=1e-9)
        dvdt = 0.5 * 2 * np.pi * 1e6
        codes = {adc.convert((0.3, 0.05), dvdt=dvdt) for _ in range(50)}
        assert len(codes) > 1

    def test_all_nonidealities_differential(self):
        """All non-idealities applied together on differential input."""
        np.random.seed(42)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.DIFFERENTIAL,
                     noise_rms=0.001, offset=0.01, gain_error=0.02,
                     t_jitter=1e-10)
        trace = adc.convert_with_trace((0.4, 0.1), dvdt=1e6)
        assert 0 <= trace['code'] <= 255
        assert isinstance(trace['sampled_voltage'], float)
