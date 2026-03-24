"""
Integration tests for pyDataconverter.

Tests that verify end-to-end behavior across multiple components.
"""

import numpy as np
import pytest
from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType, QuantizationMode
from pyDataconverter.utils.signal_gen import generate_coherent_sine
from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics


# ---------------------------------------------------------------------------
# Test 20: FlashADC round-trip accuracy (within 1 LSB)
# ---------------------------------------------------------------------------

class TestFlashADCRoundTrip:

    def test_code_to_voltage_within_1_lsb(self):
        """FlashADC encode then check code-to-voltage is within 1 LSB."""
        n_bits = 3
        v_ref = 1.0
        adc = FlashADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)
        lsb = v_ref / 2**n_bits

        # Test at each ideal code center
        for expected_code in range(2**n_bits):
            # Voltage at the center of each code bin
            v_center = (expected_code + 0.5) * lsb
            if v_center >= v_ref:
                v_center = v_ref - lsb / 4  # stay within range for max code

            actual_code = adc.convert(v_center)
            # Reconstruct voltage from code (midpoint of code bin)
            reconstructed_v = (actual_code + 0.5) * lsb
            error = abs(reconstructed_v - v_center)
            assert error <= lsb, (
                f"Round-trip error {error:.6f} V > 1 LSB ({lsb:.6f} V) "
                f"at v_center={v_center:.4f}, got code={actual_code}"
            )


# ---------------------------------------------------------------------------
# Test 21: SimpleADC SNR within 3 dB of theoretical
# ---------------------------------------------------------------------------

class TestSimpleADCSNR:

    def test_snr_within_3db_of_theoretical(self):
        """SimpleADC computed SNR for pure sinusoid is within 3 dB of theoretical."""
        n_bits = 8
        v_ref = 1.0
        fs = 1e6
        n_fft = 4096
        n_fin = 11  # coherent bin

        adc = SimpleADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE)

        # Generate coherent sine at ~90% full scale
        amplitude = 0.45 * v_ref
        offset = 0.5 * v_ref
        signal_analog, f0 = generate_coherent_sine(fs, n_fft, n_fin,
                                                    amplitude=amplitude,
                                                    offset=offset)

        # Quantize through the ADC
        codes = np.array([adc.convert(v) for v in signal_analog], dtype=float)

        # Calculate metrics
        metrics = calculate_adc_dynamic_metrics(codes, fs, f0)

        # Theoretical SNR = 6.02 * n_bits + 1.76
        theoretical_snr = 6.02 * n_bits + 1.76
        measured_snr = metrics['SNR']

        assert abs(measured_snr - theoretical_snr) < 3.0, (
            f"Measured SNR ({measured_snr:.2f} dB) not within 3 dB of "
            f"theoretical ({theoretical_snr:.2f} dB)"
        )


# ---------------------------------------------------------------------------
# Test 22: SimpleADC noise_rms matches output noise (±20%)
# ---------------------------------------------------------------------------

class TestSimpleADCNoiseRMS:

    def test_output_noise_matches_noise_rms(self):
        """Noise RMS in SimpleADC output matches noise_rms constructor parameter (±20%)."""
        n_bits = 16  # high resolution to minimize quantization noise
        v_ref = 1.0
        noise_rms = 0.01  # 10 mV RMS noise
        n_samples = 10000

        adc = SimpleADC(n_bits=n_bits, v_ref=v_ref, input_type=InputType.SINGLE,
                        noise_rms=noise_rms)
        lsb = v_ref / 2**n_bits

        # Convert a fixed DC voltage many times
        vin = 0.5
        codes = np.array([adc.convert(vin) for _ in range(n_samples)])

        # Convert codes back to voltages and measure std
        voltages = codes * lsb
        measured_noise_rms = np.std(voltages)

        # Should be within 20% of the set noise_rms
        assert abs(measured_noise_rms - noise_rms) / noise_rms < 0.2, (
            f"Measured noise RMS ({measured_noise_rms:.6f}) not within 20% of "
            f"set noise_rms ({noise_rms:.6f})"
        )
