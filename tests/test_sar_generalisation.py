"""Tests for generalised SAR ADC components."""
import numpy as np
import pytest
from pyDataconverter.components.cdac import RedundantSARCDAC


class TestRedundantSARCDAC:
    def test_construction(self):
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        assert cdac.n_bits == 4
        assert cdac.v_ref  == 1.0

    def test_weights_decrease(self):
        """Capacitor weights are strictly decreasing (MSB first)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.85)
        w = cdac.cap_weights
        assert np.all(np.diff(w) < 0)

    def test_weights_not_binary(self):
        """Weights are not powers of 2 (distinguishes from standard CDAC)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        w = cdac.cap_weights
        binary_weights = 2 ** np.arange(cdac.n_bits - 1, -1, -1).astype(float)
        assert not np.allclose(w, binary_weights)

    def test_dec_monotone(self):
        """Decoded output is monotone with increasing DAC voltage.

        The DEC table must map raw codes to output codes such that
        decode(a) <= decode(b) whenever cdac.get_voltage(a) <= cdac.get_voltage(b).
        This is the correct monotonicity property for a redundant SAR DEC.
        """
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        n_codes = 2 ** 6
        voltages = np.array([cdac.get_voltage(c)[0] for c in range(n_codes)])
        decoded  = np.array([cdac.decode(c) for c in range(n_codes)])
        # Sort raw codes by DAC voltage; decoded values in that order must be non-decreasing
        order = np.argsort(voltages)
        assert list(decoded[order]) == sorted(decoded[order])

    def test_dec_range(self):
        """Decoded values span 0..2^n_bits-1."""
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        decoded = [cdac.decode(code) for code in range(2**4)]
        assert min(decoded) == 0
        assert max(decoded) == 2**4 - 1

    def test_ideal_conversion_with_sar(self):
        """RedundantSARCDAC inside SARADC gives monotone ideal transfer."""
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        adc  = SARADC(n_bits=6, v_ref=1.0, cdac=cdac)
        vin  = np.linspace(0, 1.0, 200)
        codes = [adc.convert(float(v)) for v in vin]
        # codes should be non-decreasing
        assert all(b >= a for a, b in zip(codes, codes[1:]))
