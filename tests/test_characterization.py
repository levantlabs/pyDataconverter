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
    assert 'AmplitudeAtSNR0_dBFS' in result
    assert 'AmplitudeAtSNR0_dB' in result


def test_measure_dynamic_range_amplitude_key_relationship():
    """AmplitudeAtSNR0_dB = AmplitudeAtSNR0_dBFS + 20·log10(v_ref/2)."""
    from pyDataconverter.utils.characterization import measure_dynamic_range
    v_ref = 1.0
    adc = _make_adc(n_bits=8)
    result = measure_dynamic_range(adc, n_bits=8, v_ref=v_ref, fs=1e6,
                                   n_fft=1024, n_fin=13, n_amplitudes=15)
    expected_offset = 20.0 * np.log10(v_ref / 2.0)
    actual_offset = result['AmplitudeAtSNR0_dB'] - result['AmplitudeAtSNR0_dBFS']
    assert np.isclose(actual_offset, expected_offset, atol=1e-9)


def test_measure_dynamic_range_ideal_adc():
    """8-bit ideal ADC should have DR within practical bounds."""
    from pyDataconverter.utils.characterization import measure_dynamic_range
    adc = _make_adc(n_bits=8)
    result = measure_dynamic_range(adc, n_bits=8, v_ref=1.0, fs=1e6,
                                   n_fft=1024, n_fin=13, n_amplitudes=15)
    # Check that DR is a positive finite number within practical bounds
    dr = result['DR_dB']
    assert np.isfinite(dr), "DR_dB must be a finite number"
    assert dr > 20, "DR_dB must be at least 20 dB (above noise floor)"
    assert dr <= 100, "DR_dB must be no more than 100 dB (reasonable upper bound)"


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


def test_measure_erbw_returns_dict():
    from pyDataconverter.utils.characterization import measure_erbw
    adc = _make_adc(n_bits=8)
    result = measure_erbw(adc, n_bits=8, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=8)
    assert isinstance(result, dict)
    assert 'ERBW_Hz' in result
    assert 'Frequencies_Hz' in result
    assert 'ENOB_values' in result


def test_measure_erbw_array_lengths():
    from pyDataconverter.utils.characterization import measure_erbw
    adc = _make_adc(n_bits=6)
    n_freq = 7
    result = measure_erbw(adc, n_bits=6, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=n_freq)
    assert len(result['Frequencies_Hz']) == n_freq
    assert len(result['ENOB_values']) == n_freq


def test_measure_erbw_positive():
    """ERBW is a positive finite frequency."""
    from pyDataconverter.utils.characterization import measure_erbw
    adc = _make_adc(n_bits=8)
    result = measure_erbw(adc, n_bits=8, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=8)
    assert np.isfinite(result['ERBW_Hz'])
    assert result['ERBW_Hz'] > 0


def test_measure_erbw_duck_typing():
    """measure_erbw works with any object having .convert(float)->int."""
    from pyDataconverter.utils.characterization import measure_erbw

    class SimpleQuantizer:
        def __init__(self, n_bits, v_ref):
            self.n_bits = n_bits
            self.v_ref  = v_ref
        def convert(self, v):
            code = int(v / self.v_ref * 2**self.n_bits)
            return max(0, min(2**self.n_bits - 1, code))

    adc = SimpleQuantizer(n_bits=6, v_ref=1.0)
    result = measure_erbw(adc, n_bits=6, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=6)
    assert 'ERBW_Hz' in result
