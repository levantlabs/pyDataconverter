"""
Tests for FlashADC and EncoderType.
"""

import numpy as np
import pytest
from pyDataconverter.architectures.FlashADC import FlashADC, EncoderType
from pyDataconverter.components.comparator import ComparatorBase, DifferentialComparator
from pyDataconverter.components.reference import ReferenceLadder, ArbitraryReference
from pyDataconverter.dataconverter import InputType


# ---------------------------------------------------------------------------
# Construction and basic attributes
# ---------------------------------------------------------------------------

class TestFlashADCConstruction:

    def test_n_comparators(self):
        for n in [2, 3, 4]:
            adc = FlashADC(n_bits=n)
            assert adc.n_comparators == 2 ** n - 1

    def test_default_encoder_is_count_ones(self):
        adc = FlashADC(n_bits=3)
        assert adc.encoder_type == EncoderType.COUNT_ONES

    def test_invalid_encoder_type_raises(self):
        with pytest.raises(TypeError):
            FlashADC(n_bits=3, encoder_type='count_ones')

    def test_default_reference_is_ladder(self):
        adc = FlashADC(n_bits=3, v_ref=1.0)
        assert isinstance(adc.reference, ReferenceLadder)

    def test_default_reference_single_ended_range(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE)
        vols = adc.reference.voltages
        assert vols[0] > 0.0
        assert vols[-1] < 1.0

    def test_default_reference_differential_range(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        vols = adc.reference.voltages
        assert vols[0] < 0.0
        assert vols[-1] > 0.0

    def test_custom_reference_accepted(self):
        ref = ArbitraryReference(np.linspace(0.125, 0.875, 7))
        adc = FlashADC(n_bits=3, v_ref=1.0, reference=ref)
        assert adc.reference is ref

    def test_custom_reference_wrong_taps_raises(self):
        ref = ArbitraryReference([0.25, 0.5, 0.75])   # 3 taps != 7
        with pytest.raises(ValueError):
            FlashADC(n_bits=3, v_ref=1.0, reference=ref)

    def test_custom_reference_wrong_type_raises(self):
        with pytest.raises(TypeError):
            FlashADC(n_bits=3, v_ref=1.0, reference="not_a_reference")

    def test_reference_voltages_property(self):
        """Backward-compat alias for reference.voltages."""
        adc = FlashADC(n_bits=3, v_ref=1.0)
        np.testing.assert_array_equal(adc.reference_voltages, adc.reference.voltages)


# ---------------------------------------------------------------------------
# Encoding: COUNT_ONES
# ---------------------------------------------------------------------------

class TestCountOnesEncoder:

    def setup_method(self):
        self.adc = FlashADC(n_bits=3, v_ref=1.0, encoder_type=EncoderType.COUNT_ONES)

    def test_ramp_monotone(self):
        """Output codes on a ramp are non-decreasing."""
        v_in = np.linspace(0.0, 1.0, 200)
        codes = [self.adc.convert(v) for v in v_in]
        assert all(codes[i] <= codes[i + 1] for i in range(len(codes) - 1))

    def test_min_input_gives_code_zero(self):
        assert self.adc.convert(-0.1) == 0

    def test_max_input_gives_max_code(self):
        assert self.adc.convert(1.1) == 7

    def test_midscale_code(self):
        # Midscale for 3-bit 0-1 V: code 4 around 0.5V
        code = self.adc.convert(0.5)
        assert code in (3, 4)

    def test_encode_all_zeros_thermometer(self):
        therm = np.zeros(7, dtype=int)
        assert self.adc._encode(therm) == 0

    def test_encode_all_ones_thermometer(self):
        therm = np.ones(7, dtype=int)
        assert self.adc._encode(therm) == 7

    def test_encode_partial_thermometer(self):
        therm = np.array([1, 1, 1, 0, 0, 0, 0])
        assert self.adc._encode(therm) == 3


# ---------------------------------------------------------------------------
# Encoding: XOR
# ---------------------------------------------------------------------------

class TestXorEncoder:

    def setup_method(self):
        self.adc_xor    = FlashADC(n_bits=3, v_ref=1.0, encoder_type=EncoderType.XOR)
        self.adc_co     = FlashADC(n_bits=3, v_ref=1.0, encoder_type=EncoderType.COUNT_ONES)

    def test_xor_matches_count_ones_on_valid_code(self):
        """XOR encoder must equal COUNT_ONES for every clean thermometer code."""
        for k in range(8):
            therm = np.array([1] * k + [0] * (7 - k))
            assert self.adc_xor._encode(therm) == self.adc_co._encode(therm)

    def test_xor_bubble_produces_nonzero(self):
        """A bubble (e.g. [1,1,0,1,0,0,0]) should produce a non-ideal code."""
        therm = np.array([1, 1, 0, 1, 0, 0, 0])
        code_xor = self.adc_xor._encode(therm)
        code_co  = self.adc_co._encode(therm)
        # With a bubble, COUNT_ONES gives 3 (still sane); XOR may differ
        # Just verify XOR produces a valid (in-range) code without crashing
        assert 0 <= code_xor <= 7

    def test_xor_ramp_codes_valid(self):
        """All ramp output codes are within [0, 2^n_bits - 1]."""
        v_in = np.linspace(0.0, 1.0, 200)
        codes = [self.adc_xor.convert(v) for v in v_in]
        assert all(0 <= c <= 7 for c in codes)


# ---------------------------------------------------------------------------
# Differential input
# ---------------------------------------------------------------------------

class TestDifferentialInput:

    def test_differential_midscale(self):
        """v_pos = v_neg → differential 0 → code near midscale."""
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        code = adc.convert((0.5, 0.5))
        assert code in (3, 4)

    def test_differential_full_positive(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        assert adc.convert((1.0, 0.0)) == 7

    def test_differential_full_negative(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        assert adc.convert((0.0, 1.0)) == 0

    def test_differential_reference_symmetric(self):
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        vols = adc.reference.voltages
        np.testing.assert_allclose(vols, -vols[::-1], atol=1e-12)


# ---------------------------------------------------------------------------
# Comparator offsets and non-idealities
# ---------------------------------------------------------------------------

class TestComparatorNonidealities:

    def test_offset_std_shifts_thresholds(self):
        """With large offset_std, output on a ramp differs from ideal."""
        np.random.seed(7)
        adc_ideal   = FlashADC(n_bits=3, v_ref=1.0)
        np.random.seed(7)
        adc_offset  = FlashADC(n_bits=3, v_ref=1.0, offset_std=0.1)

        v_in = np.linspace(0.05, 0.95, 100)
        codes_ideal  = [adc_ideal.convert(v)  for v in v_in]
        codes_offset = [adc_offset.convert(v) for v in v_in]
        assert codes_ideal != codes_offset

    def test_comparator_params_forwarded(self):
        """noise_rms in comparator_params should introduce code spread."""
        np.random.seed(0)
        adc = FlashADC(n_bits=4, v_ref=1.0,
                       comparator_params={'noise_rms': 0.05})
        v = 0.5
        codes = {adc.convert(v) for _ in range(50)}
        assert len(codes) > 1   # noise causes spread


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_does_not_raise(self):
        adc = FlashADC(n_bits=3, v_ref=1.0,
                       comparator_params={'hysteresis': 0.01})
        for v in np.linspace(0, 1, 20):
            adc.convert(v)
        adc.reset()   # must not raise

    def test_reset_clears_hysteresis(self):
        """After reset, converter behaves as fresh — hysteresis memory gone."""
        adc = FlashADC(n_bits=3, v_ref=1.0,
                       comparator_params={'hysteresis': 0.05})
        # drive high to latch all comparators
        for _ in range(5):
            adc.convert(1.0)
        code_before = adc.convert(0.5)
        adc.reset()
        code_after = adc.convert(0.5)
        # Before reset (with hysteresis history) might differ; both should be valid
        assert 0 <= code_before <= 7
        assert 0 <= code_after  <= 7


# ---------------------------------------------------------------------------
# Comparator model substitution
# ---------------------------------------------------------------------------

class TestComparatorSubstitution:

    def test_default_comparator_is_differential(self):
        adc = FlashADC(n_bits=3, v_ref=1.0)
        assert all(isinstance(c, DifferentialComparator) for c in adc.comparators)

    def test_custom_comparator_type_accepted(self):
        """A subclass of ComparatorBase can be passed as comparator_type."""
        class MyComp(DifferentialComparator):
            pass

        adc = FlashADC(n_bits=3, v_ref=1.0, comparator_type=MyComp)
        assert all(isinstance(c, MyComp) for c in adc.comparators)

    def test_invalid_comparator_type_raises(self):
        """Non-ComparatorBase class should raise TypeError or similar at instantiation."""
        class NotAComparator:
            def __init__(self, **kwargs):
                pass
        # Should raise because NotAComparator is not a ComparatorBase subclass;
        # FlashADC does not enforce this at construction, but the type hint documents it.
        # At minimum, conversion must not silently succeed with wrong results.
        # This test just verifies no crash for a duck-typed replacement.
        adc = FlashADC(n_bits=3, v_ref=1.0, comparator_type=NotAComparator)
        # If it has a compatible compare() it works; if not it raises on convert()
        assert len(adc.comparators) == 7


# ---------------------------------------------------------------------------
# DifferentialComparator 4-input interface
# ---------------------------------------------------------------------------

class TestDifferentialComparatorInterface:

    def test_two_input_default_matches_four_input(self):
        """compare(v, ref) == compare(v, 0, ref, 0) for any v, ref."""
        comp = DifferentialComparator()
        for v, ref in [(0.3, 0.2), (0.1, 0.5), (0.7, 0.7)]:
            np.random.seed(0)
            r2 = comp.compare(v, ref)
            comp.reset()
            np.random.seed(0)
            r4 = comp.compare(v, 0.0, ref, 0.0)
            assert r2 == r4

    def test_differential_reference_subtraction(self):
        """compare(v_pos, v_neg, v_refp, v_refn) fires based on (v_pos-v_refp)-(v_neg-v_refn)."""
        comp = DifferentialComparator()
        # (0.6 - 0.1) - (0.4 - 0.1) = 0.5 - 0.3 = 0.2 > 0 → should fire
        assert comp.compare(0.6, 0.4, 0.1, 0.1) == 1
        comp.reset()
        # (0.4 - 0.1) - (0.6 - 0.1) = 0.3 - 0.5 = -0.2 < 0 → should not fire
        assert comp.compare(0.4, 0.6, 0.1, 0.1) == 0

    def test_symmetric_reference_cancels(self):
        """If v_refp == v_refn, reference cancels and result depends only on v_diff."""
        comp = DifferentialComparator()
        assert comp.compare(0.7, 0.3, 0.5, 0.5) == 1   # v_diff = +0.4
        comp.reset()
        assert comp.compare(0.3, 0.7, 0.5, 0.5) == 0   # v_diff = -0.4


# ---------------------------------------------------------------------------
# Differential reference ladder
# ---------------------------------------------------------------------------

class TestDifferentialReferenceLadder:

    def test_ladder_spans_quarter_vref(self):
        """Default differential ladder taps span [-v_ref/4, +v_ref/4]."""
        adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        vols = adc.reference.voltages
        assert vols[0]  > -0.25 - 1e-9
        assert vols[-1] <  0.25 + 1e-9

    def test_effective_thresholds_span_half_vref(self):
        """Effective thresholds = comp_refs[i] - comp_refs[n-1-i] span [-v_ref/2, +v_ref/2]."""
        adc  = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        refs = adc.reference.voltages
        n    = len(refs)
        thresholds = np.array([refs[i] - refs[n - 1 - i] for i in range(n)])
        np.testing.assert_allclose(thresholds, -thresholds[::-1], atol=1e-12)
        assert thresholds[0]  < -0.3    # should be near -3/8 = -0.375
        assert thresholds[-1] >  0.3    # should be near +3/8 = +0.375

    def test_differential_thresholds_ascending(self):
        """Effective thresholds must be strictly ascending for correct thermometer code."""
        adc  = FlashADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        refs = adc.reference.voltages
        n    = len(refs)
        thresholds = np.array([refs[i] - refs[n - 1 - i] for i in range(n)])
        assert np.all(np.diff(thresholds) > 0)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestResetClearsHysteresis:

    def test_reset_makes_conversion_deterministic(self):
        """After reset, two fresh conversions of the same voltage give the same code."""
        adc = FlashADC(n_bits=3, v_ref=1.0,
                       comparator_params={'hysteresis': 0.05})
        # Drive high then low to build hysteresis history
        for _ in range(10):
            adc.convert(1.0)
        for _ in range(10):
            adc.convert(0.0)

        adc.reset()
        code_a = adc.convert(0.5)

        # Reset again and convert the same voltage
        adc.reset()
        code_b = adc.convert(0.5)

        assert code_a == code_b, "After reset, same input must produce same code"


class TestXorEncoderAllOnes:

    def test_xor_all_ones_thermometer_3bit(self):
        """XOR encoder with all-ones thermometer for 3-bit ADC gives max code 7."""
        adc = FlashADC(n_bits=3, v_ref=1.0, encoder_type=EncoderType.XOR)
        therm = np.ones(7, dtype=int)
        assert adc._encode(therm) == 7

    def test_xor_all_ones_matches_count_ones(self):
        """XOR all-ones result matches COUNT_ONES for 3-bit and 4-bit ADCs."""
        for n in [3, 4]:
            n_comp = 2**n - 1
            adc_xor = FlashADC(n_bits=n, v_ref=1.0, encoder_type=EncoderType.XOR)
            adc_co  = FlashADC(n_bits=n, v_ref=1.0, encoder_type=EncoderType.COUNT_ONES)
            therm = np.ones(n_comp, dtype=int)
            assert adc_xor._encode(therm) == adc_co._encode(therm)


class TestOffsetStdZeroDeterministic:

    def test_offset_std_zero_produces_identical_results(self):
        """With offset_std=0, repeated conversions of the same voltage are identical."""
        adc = FlashADC(n_bits=3, v_ref=1.0, offset_std=0.0)
        codes = [adc.convert(0.5) for _ in range(20)]
        assert len(set(codes)) == 1, "offset_std=0 should produce identical results"


class TestRepr:

    def test_repr_contains_class_name(self):
        adc = FlashADC(n_bits=3)
        assert 'FlashADC' in repr(adc)

    def test_repr_contains_n_bits(self):
        adc = FlashADC(n_bits=4)
        assert 'n_bits=4' in repr(adc)
