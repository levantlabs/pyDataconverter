"""Tests for pyDataconverter.utils.characterization."""
import numpy as np
import pytest
from pyDataconverter.architectures.FlashADC import FlashADC


def _make_adc(n_bits=8, offset_std=0.0):
    return FlashADC(n_bits=n_bits, v_ref=1.0, offset_std=offset_std)


def test_measure_dynamic_range_returns_dict():
    from pyDataconverter.utils.characterization import measure_dynamic_range
    adc = _make_adc(n_bits=8)
    result = measure_dynamic_range(adc, n_bits=8, v_ref=1.0, fs=1e6,
                                   n_fft=512, n_fin=13)
    assert isinstance(result, dict)
    assert 'DR_dB' in result
    assert 'Amplitudes_dBFS' in result
    assert 'SNR_values' in result


def test_measure_dynamic_range_ideal_adc():
    """8-bit ideal ADC should have DR close to 8*6.02 + 1.76 ≈ 49.9 dB."""
    from pyDataconverter.utils.characterization import measure_dynamic_range
    adc = _make_adc(n_bits=8)
    result = measure_dynamic_range(adc, n_bits=8, v_ref=1.0, fs=1e6,
                                   n_fft=1024, n_fin=13, n_amplitudes=15)
    # DR for an ideal N-bit ADC is approximately 6.02*N + 1.76 dB
    expected_dr = 6.02 * 8 + 1.76
    assert abs(result['DR_dB'] - expected_dr) < 5.0  # within 5 dB


def test_measure_dynamic_range_array_lengths():
    """Amplitudes_dBFS and SNR_values have n_amplitudes entries."""
    from pyDataconverter.utils.characterization import measure_dynamic_range
    adc = _make_adc(n_bits=6)
    n_amp = 10
    result = measure_dynamic_range(adc, n_bits=6, v_ref=1.0, fs=1e6,
                                   n_fft=512, n_fin=7, n_amplitudes=n_amp)
    assert len(result['Amplitudes_dBFS']) == n_amp
    assert len(result['SNR_values']) == n_amp


def test_measure_dynamic_range_duck_typing():
    """Any object with .convert(float)->int works as the ADC."""
    from pyDataconverter.utils.characterization import measure_dynamic_range

    class SimpleQuantizer:
        def __init__(self, n_bits, v_ref):
            self.n_bits = n_bits
            self.v_ref  = v_ref
        def convert(self, v):
            code = int(v / self.v_ref * 2**self.n_bits)
            return max(0, min(2**self.n_bits - 1, code))

    adc = SimpleQuantizer(n_bits=6, v_ref=1.0)
    result = measure_dynamic_range(adc, n_bits=6, v_ref=1.0, fs=1e6,
                                   n_fft=512, n_fin=7, n_amplitudes=8)
    assert 'DR_dB' in result
    assert result['DR_dB'] > 0
