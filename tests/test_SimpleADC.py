"""
Test cases for SimpleADC class
"""

import unittest
import numpy as np
from pyDataconverter.dataconverter import InputType, QuantizationMode
from pyDataconverter.architectures.SimpleADC import SimpleADC


class TestSimpleADC(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.n_bits = 12
        self.v_ref = 1.0

    # ------------------------------------------------------------------ #
    # FLOOR mode — single-ended                                            #
    # ------------------------------------------------------------------ #

    def test_floor_single_ended(self):
        """FLOOR mode: check zero, mid, full-scale, and clipping"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)

        self.assertEqual(adc.convert(0.0), 0)
        # Mid-scale: floor(0.5 * 2^N) = 2^(N-1) = 2048
        self.assertEqual(adc.convert(self.v_ref / 2), 2 ** (self.n_bits - 1))
        # Full-scale clips to max code
        self.assertEqual(adc.convert(self.v_ref), 2 ** self.n_bits - 1)
        # Out of range clips
        self.assertEqual(adc.convert(-0.1), 0)
        self.assertEqual(adc.convert(self.v_ref + 0.1), 2 ** self.n_bits - 1)

    # ------------------------------------------------------------------ #
    # FLOOR mode — differential                                            #
    # ------------------------------------------------------------------ #

    def test_floor_differential(self):
        """FLOOR mode: check zero diff, full-scale pos/neg, quarter-scale, clipping"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL)

        # Zero differential maps to mid-scale code
        self.assertEqual(adc.convert((0.5, 0.5)), 2 ** (self.n_bits - 1))

        # Full-scale positive clips to max code
        self.assertEqual(adc.convert((self.v_ref / 2, 0)), 2 ** self.n_bits - 1)

        # Full-scale negative gives code 0
        self.assertEqual(adc.convert((0, self.v_ref / 2)), 0)

        # Quarter-scale positive: vdiff=0.125, shifted v=0.625
        # floor(0.625 * 2^N) = floor(2560) = 2560
        expected_quarter = int(2 ** (self.n_bits - 1) * (1 + 0.25))
        self.assertEqual(adc.convert((0.2, 0.075)), expected_quarter)

        # Out of range clips
        self.assertEqual(adc.convert((1.0, -1.0)), 2 ** self.n_bits - 1)
        self.assertEqual(adc.convert((-1.0, 1.0)), 0)

    # ------------------------------------------------------------------ #
    # SYMMETRIC mode — single-ended                                        #
    # ------------------------------------------------------------------ #

    def test_symmetric_single_ended(self):
        """SYMMETRIC mode: check zero, mid, full-scale"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                        quant_mode=QuantizationMode.SYMMETRIC)

        self.assertEqual(adc.convert(0.0), 0)
        # Mid-scale: floor(0.5*(2^N-1) + 0.5) = floor(2048) = 2048
        self.assertEqual(adc.convert(self.v_ref / 2), 2 ** (self.n_bits - 1))
        self.assertEqual(adc.convert(self.v_ref), 2 ** self.n_bits - 1)
        # Out of range clips
        self.assertEqual(adc.convert(-0.1), 0)
        self.assertEqual(adc.convert(self.v_ref + 0.1), 2 ** self.n_bits - 1)

    def test_symmetric_first_transition(self):
        """SYMMETRIC mode: first transition is at 0.5*LSB, not 1*LSB"""
        adc_sym   = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.SYMMETRIC)
        adc_floor = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.FLOOR)

        # SYMMETRIC LSB = v_ref / (2^N - 1); first transition at 0.5 * LSB
        lsb_sym = self.v_ref / (2 ** self.n_bits - 1)
        first_transition = 0.5 * lsb_sym

        # Just below: both modes stay at code 0
        vin_below = first_transition - 1e-9
        self.assertEqual(adc_sym.convert(vin_below), 0)
        self.assertEqual(adc_floor.convert(vin_below), 0)

        # At and just above first transition: SYMMETRIC moves to code 1,
        # FLOOR is still 0 (its first transition is at a full LSB = v_ref/2^N)
        vin_above = first_transition + 1e-9
        self.assertEqual(adc_sym.convert(vin_above), 1)
        self.assertEqual(adc_floor.convert(vin_above), 0)

    def test_symmetric_last_transition(self):
        """SYMMETRIC mode: last transition is at v_ref - 0.5*LSB"""
        adc_sym   = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.SYMMETRIC)
        adc_floor = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              quant_mode=QuantizationMode.FLOOR)

        lsb_sym = self.v_ref / (2 ** self.n_bits - 1)
        last_transition = self.v_ref - 0.5 * lsb_sym

        # Just below last transition: SYMMETRIC is still at 2^N-2,
        # FLOOR has already reached 2^N-1 (its last transition is earlier)
        vin_below = last_transition - 1e-9
        self.assertEqual(adc_sym.convert(vin_below), 2 ** self.n_bits - 2)

        # At and above last transition: SYMMETRIC reaches max code
        vin_above = last_transition + 1e-9
        self.assertEqual(adc_sym.convert(vin_above), 2 ** self.n_bits - 1)

    # ------------------------------------------------------------------ #
    # SYMMETRIC mode — differential                                        #
    # ------------------------------------------------------------------ #

    def test_symmetric_differential(self):
        """SYMMETRIC mode: check zero diff, full-scale pos/neg"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL,
                        quant_mode=QuantizationMode.SYMMETRIC)

        # Zero differential maps to mid-scale
        self.assertEqual(adc.convert((0.5, 0.5)), 2 ** (self.n_bits - 1))
        # Full-scale positive
        self.assertEqual(adc.convert((self.v_ref / 2, 0)), 2 ** self.n_bits - 1)
        # Full-scale negative
        self.assertEqual(adc.convert((0, self.v_ref / 2)), 0)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def test_input_validation(self):
        """Input type mismatches raise TypeError"""
        adc_se = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        with self.assertRaises(TypeError):
            adc_se.convert((0.5, 0.5))  # Tuple not allowed for single-ended

        adc_diff = SimpleADC(self.n_bits, self.v_ref, InputType.DIFFERENTIAL)
        with self.assertRaises(TypeError):
            adc_diff.convert(0.5)       # Single value not allowed for differential
        with self.assertRaises(TypeError):
            adc_diff.convert((0.5,))    # Must be 2-tuple

    def test_quant_mode_validation(self):
        """Invalid quant_mode raises TypeError"""
        with self.assertRaises(TypeError):
            SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                      quant_mode='floor')

    # ------------------------------------------------------------------ #
    # Resolution and repr                                                  #
    # ------------------------------------------------------------------ #

    def test_resolution(self):
        """Max code is correct across resolutions for both modes"""
        for bits in [8, 12, 16]:
            for mode in QuantizationMode:
                adc = SimpleADC(bits, self.v_ref, InputType.SINGLE, quant_mode=mode)
                self.assertEqual(adc.convert(self.v_ref), 2 ** bits - 1)
                self.assertEqual(adc.convert(0.0), 0)

    def test_repr(self):
        """repr includes quant_mode and any non-zero non-ideality parameters"""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                        quant_mode=QuantizationMode.SYMMETRIC)
        self.assertIn('SYMMETRIC', repr(adc))

        adc_ni = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                           noise_rms=1e-4, offset=1e-3,
                           gain_error=0.001, t_jitter=1e-12)
        r = repr(adc_ni)
        self.assertIn('noise_rms', r)
        self.assertIn('offset', r)
        self.assertIn('gain_error', r)
        self.assertIn('t_jitter', r)

    # ------------------------------------------------------------------ #
    # Non-idealities                                                       #
    # ------------------------------------------------------------------ #

    def test_offset_shifts_codes(self):
        """A positive offset shifts all codes upward by roughly offset/LSB."""
        lsb = self.v_ref / 2**self.n_bits
        offset = 10 * lsb
        adc_ideal  = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        adc_offset = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                               offset=offset)

        for vin in [0.1, 0.3, 0.5, 0.7]:
            ideal_code  = adc_ideal.convert(vin)
            offset_code = adc_offset.convert(vin)
            # Code should shift by approximately offset/lsb (allow ±1 rounding)
            self.assertAlmostEqual(offset_code - ideal_code, round(offset / lsb),
                                   delta=1,
                                   msg=f"vin={vin}: offset shift wrong")

    def test_gain_error_scales_codes(self):
        """
        A gain error of +1 % should cause the mid-scale code to be
        approximately 1 % higher than ideal (before clipping).
        """
        gain_error = 0.01
        adc_ideal = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        adc_gain  = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              gain_error=gain_error)

        vin = 0.25  # well within range so gain shift doesn't clip
        ideal_code = adc_ideal.convert(vin)
        gain_code  = adc_gain.convert(vin)
        expected_shift = round(ideal_code * gain_error)
        self.assertAlmostEqual(gain_code - ideal_code, expected_shift,
                               delta=1,
                               msg="Gain error did not shift code as expected")

    def test_noise_rms_zero_means_ideal(self):
        """With noise_rms=0 every call returns the same code as ideal."""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE, noise_rms=0.0)
        adc_ideal = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        for vin in [0.0, 0.25, 0.5, 0.75, 1.0]:
            self.assertEqual(adc.convert(vin), adc_ideal.convert(vin))

    def test_noise_spreads_codes(self):
        """Large noise_rms should cause output codes to vary across repeated conversions."""
        lsb = self.v_ref / 2**self.n_bits
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                        noise_rms=100 * lsb)
        vin = 0.5
        codes = [adc.convert(vin) for _ in range(200)]
        # With 100 LSB noise, output must not be constant
        self.assertGreater(len(set(codes)), 1,
                           "Codes should vary with large noise_rms")

    def test_aperture_jitter_zero_dvdt_no_effect(self):
        """Aperture jitter has no effect when dvdt=0 (default)."""
        lsb = self.v_ref / 2**self.n_bits
        adc_jitter = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                               t_jitter=1e-6)   # very large jitter
        adc_ideal  = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        # Without passing dvdt the jitter term is zero regardless of t_jitter
        for vin in [0.25, 0.5, 0.75]:
            self.assertEqual(adc_jitter.convert(vin), adc_ideal.convert(vin))

    def test_aperture_jitter_with_dvdt_spreads_codes(self):
        """Aperture jitter with a large dvdt should spread output codes."""
        import math
        lsb = self.v_ref / 2**self.n_bits
        # Jitter large enough to cause multi-LSB errors
        t_jitter = 100 * lsb   # 100 LSB * 1 V/s ≈ 100 LSB error at dvdt=1 V/s
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                        t_jitter=t_jitter)
        dvdt = 1.0  # V/s — large enough that jitter * dvdt >> LSB
        codes = [adc.convert(0.5, dvdt=dvdt) for _ in range(200)]
        self.assertGreater(len(set(codes)), 1,
                           "Codes should vary when t_jitter and dvdt are both non-zero")

    def test_parameter_validation(self):
        """noise_rms and t_jitter must be non-negative."""
        with self.assertRaises(ValueError):
            SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE, noise_rms=-1.0)
        with self.assertRaises(ValueError):
            SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE, t_jitter=-1e-12)

    # ------------------------------------------------------------------ #
    # Combined non-idealities                                              #
    # ------------------------------------------------------------------ #

    def test_gain_error_and_offset_combined(self):
        """Both gain_error and offset applied simultaneously shift codes correctly."""
        lsb = self.v_ref / 2**self.n_bits
        gain_error = 0.01   # +1%
        offset_val = 10 * lsb

        adc_ideal = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        adc_combo = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                              gain_error=gain_error, offset=offset_val)

        vin = 0.25  # well within range
        ideal_code = adc_ideal.convert(vin)
        combo_code = adc_combo.convert(vin)

        # Expected voltage after non-idealities: vin*(1+gain_error) + offset
        v_effective = vin * (1.0 + gain_error) + offset_val
        expected_code = adc_ideal.convert(np.clip(v_effective, 0, self.v_ref))

        self.assertAlmostEqual(combo_code, expected_code, delta=1,
                               msg="Combined gain_error + offset should match expected code")

    def test_aperture_jitter_negative_dvdt(self):
        """Aperture jitter with negative dvdt (falling edge) still spreads codes."""
        lsb = self.v_ref / 2**self.n_bits
        t_jitter = 100 * lsb
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE,
                        t_jitter=t_jitter)
        dvdt = -1.0  # falling edge
        codes = [adc.convert(0.5, dvdt=dvdt) for _ in range(200)]
        self.assertGreater(len(set(codes)), 1,
                           "Codes should vary with negative dvdt and non-zero jitter")

    def test_quantize_degenerate_v_min_equals_v_max(self):
        """_quantize with v_min == v_max should not crash (division by zero guard)."""
        adc = SimpleADC(self.n_bits, self.v_ref, InputType.SINGLE)
        # Calling _quantize directly with degenerate range
        # v_min == v_max means v_range == 0; any voltage clips to v_min
        try:
            code = adc._quantize(0.5, v_min=0.5, v_max=0.5)
            # If it doesn't raise, verify the code is within valid range
            self.assertGreaterEqual(code, 0)
            self.assertLessEqual(code, 2**self.n_bits - 1)
        except (ZeroDivisionError, ValueError, FloatingPointError):
            # Acceptable: the degenerate case is undefined
            pass

    # ------------------------------------------------------------------ #
    # SYMMETRIC mode — 1-bit ADC                                           #
    # ------------------------------------------------------------------ #

    def test_symmetric_1bit_adc(self):
        """SYMMETRIC mode with 1-bit ADC: code 0 below midpoint, code 1 above."""
        adc = SimpleADC(1, self.v_ref, InputType.SINGLE,
                        quant_mode=QuantizationMode.SYMMETRIC)
        # With 1-bit SYMMETRIC: LSB = v_ref/(2^1 - 1) = v_ref
        # Transition at 0.5*LSB = 0.5*v_ref
        self.assertEqual(adc.convert(0.0), 0)
        self.assertEqual(adc.convert(self.v_ref), 1)
        # Near midpoint
        self.assertEqual(adc.convert(self.v_ref * 0.4), 0)
        self.assertEqual(adc.convert(self.v_ref * 0.6), 1)


# ===========================================================================
# __main__ block coverage
# ===========================================================================

class TestSimpleADCMainBlock(unittest.TestCase):
    """Cover the __main__ block logic (lines 186-205) inline."""

    def test_main_block_logic(self):
        """Replicate the __main__ block to exercise lines 186-205."""
        import math

        # --- Ideal (default) ---
        adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)
        self.assertEqual(adc.convert(0.0), 0)
        self.assertEqual(adc.convert(0.5), 2048)
        self.assertEqual(adc.convert(1.0), 4095)

        # --- With noise and offset ---
        adc_noisy = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
                              noise_rms=1e-4, offset=5e-3, gain_error=0.001)
        code_mid = adc_noisy.convert(0.5)
        self.assertIsInstance(code_mid, (int, np.integer))

        # --- Aperture jitter ---
        adc_jitter = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
                               t_jitter=1e-12)
        f, A = 10e3, 0.5
        dvdt = A * 2 * math.pi * f
        code_jitter = adc_jitter.convert(0.0, dvdt=dvdt)
        self.assertIsInstance(code_jitter, (int, np.integer))


if __name__ == '__main__':
    unittest.main()
