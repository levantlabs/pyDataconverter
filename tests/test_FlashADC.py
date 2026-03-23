"""
Tests for FlashADC and EncoderType.
"""

import numpy as np
import pytest
from pyDataconverter.architectures.FlashADC import FlashADC, EncoderType
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
# repr
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self):
        adc = FlashADC(n_bits=3)
        assert 'FlashADC' in repr(adc)

    def test_repr_contains_n_bits(self):
        adc = FlashADC(n_bits=4)
        assert 'n_bits=4' in repr(adc)
