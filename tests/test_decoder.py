"""
Tests for decoder components (BinaryDecoder, ThermometerDecoder, SegmentedDecoder).

Supplements the decoder tests already in test_CurrentSteeringDAC.py with
additional edge cases and boundary conditions.
"""

import numpy as np
import pytest
from pyDataconverter.components.decoder import (
    DecoderBase, BinaryDecoder, ThermometerDecoder, SegmentedDecoder,
)


# ===========================================================================
# BinaryDecoder — edge cases
# ===========================================================================

class TestBinaryDecoderEdgeCases:

    def test_1_bit_decode_zero(self):
        d = BinaryDecoder(n_bits=1)
        therm, bits = d.decode(0)
        assert therm == 0
        np.testing.assert_array_equal(bits, [0.])

    def test_1_bit_decode_one(self):
        d = BinaryDecoder(n_bits=1)
        therm, bits = d.decode(1)
        assert therm == 0
        np.testing.assert_array_equal(bits, [1.])

    def test_1_bit_out_of_range(self):
        d = BinaryDecoder(n_bits=1)
        with pytest.raises(ValueError):
            d.decode(2)

    def test_32_bit_construction(self):
        d = BinaryDecoder(n_bits=32)
        assert d.n_bits == 32
        assert d.n_binary_bits == 32

    def test_33_bit_raises(self):
        with pytest.raises(ValueError):
            BinaryDecoder(n_bits=33)

    def test_zero_bits_raises(self):
        with pytest.raises(ValueError):
            BinaryDecoder(n_bits=0)

    def test_negative_bits_raises(self):
        with pytest.raises(ValueError):
            BinaryDecoder(n_bits=-1)

    def test_float_bits_raises(self):
        with pytest.raises(ValueError):
            BinaryDecoder(n_bits=4.0)

    def test_repr(self):
        d = BinaryDecoder(n_bits=4)
        r = repr(d)
        assert 'BinaryDecoder' in r
        assert 'n_bits=4' in r
        assert 'n_therm_bits=0' in r


# ===========================================================================
# ThermometerDecoder — edge cases
# ===========================================================================

class TestThermometerDecoderEdgeCases:

    def test_1_bit_has_1_element(self):
        d = ThermometerDecoder(n_bits=1)
        assert d.n_elements == 1

    def test_1_bit_decode(self):
        d = ThermometerDecoder(n_bits=1)
        therm, bits = d.decode(0)
        assert therm == 0
        assert len(bits) == 0
        therm, bits = d.decode(1)
        assert therm == 1

    def test_16_bit_max_allowed(self):
        d = ThermometerDecoder(n_bits=16)
        assert d.n_elements == 65535

    def test_17_bit_raises(self):
        with pytest.raises(ValueError):
            ThermometerDecoder(n_bits=17)

    def test_zero_bits_raises(self):
        with pytest.raises(ValueError):
            ThermometerDecoder(n_bits=0)

    def test_repr(self):
        d = ThermometerDecoder(n_bits=3)
        r = repr(d)
        assert 'ThermometerDecoder' in r
        assert 'n_therm_bits=3' in r


# ===========================================================================
# SegmentedDecoder — edge cases
# ===========================================================================

class TestSegmentedDecoderEdgeCases:

    def test_1_bit_all_therm(self):
        d = SegmentedDecoder(n_bits=1, n_therm_bits=1)
        therm, bits = d.decode(0)
        assert therm == 0
        assert len(bits) == 0
        therm, bits = d.decode(1)
        assert therm == 1

    def test_1_bit_all_binary(self):
        d = SegmentedDecoder(n_bits=1, n_therm_bits=0)
        therm, bits = d.decode(0)
        assert therm == 0
        np.testing.assert_array_equal(bits, [0.])
        therm, bits = d.decode(1)
        assert therm == 0
        np.testing.assert_array_equal(bits, [1.])

    def test_n_therm_elements_zero_when_zero_therm_bits(self):
        d = SegmentedDecoder(n_bits=4, n_therm_bits=0)
        assert d.n_therm_elements == 0

    def test_negative_n_therm_bits_raises(self):
        with pytest.raises(ValueError):
            SegmentedDecoder(n_bits=4, n_therm_bits=-1)

    def test_n_therm_bits_greater_than_n_bits_raises(self):
        with pytest.raises(ValueError):
            SegmentedDecoder(n_bits=4, n_therm_bits=5)

    def test_full_code_range_roundtrip(self):
        """Every code produces valid therm_index and binary_bits."""
        d = SegmentedDecoder(n_bits=6, n_therm_bits=3)
        for code in range(64):
            therm, bits = d.decode(code)
            assert 0 <= therm <= 7
            assert len(bits) == 3
            assert all(b in (0.0, 1.0) for b in bits)

    def test_decode_reconstructs_code(self):
        """therm_index * 2^n_binary_bits + binary_value == original code."""
        d = SegmentedDecoder(n_bits=8, n_therm_bits=4)
        for code in [0, 1, 127, 128, 255]:
            therm, bits = d.decode(code)
            binary_val = sum(int(b) * (2 ** (len(bits) - 1 - i))
                             for i, b in enumerate(bits))
            reconstructed = therm * (2 ** d.n_binary_bits) + binary_val
            assert reconstructed == code

    def test_validate_code_non_integer_raises(self):
        d = SegmentedDecoder(n_bits=4, n_therm_bits=2)
        with pytest.raises(TypeError):
            d.decode(3.5)

    def test_validate_code_string_raises(self):
        d = SegmentedDecoder(n_bits=4, n_therm_bits=2)
        with pytest.raises(TypeError):
            d.decode("3")
