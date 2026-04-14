"""
Tests for ResidueAmplifier component (used by pipelined ADC).
"""

import unittest
import math
import numpy as np
from pyDataconverter.components.residue_amplifier import ResidueAmplifier


class TestResidueAmplifierIdeal(unittest.TestCase):
    """settling_tau=0 collapses to an instantaneous ideal amp."""

    def test_ideal_amp_returns_target_plus_offset(self):
        # Under the pre-scaled convention the caller has already applied
        # gain to target, so amplify() returns target + offset directly.
        amp = ResidueAmplifier(gain=2.0, settling_tau=0.0)
        self.assertAlmostEqual(amp.amplify(target=0.6, initial_error=0.0, t_budget=1e-9), 0.6)

    def test_ideal_amp_ignores_initial_error(self):
        amp = ResidueAmplifier(gain=2.0, settling_tau=0.0)
        self.assertAlmostEqual(
            amp.amplify(target=0.3, initial_error=0.05, t_budget=1e-9), 0.3
        )

    def test_ideal_amp_includes_offset(self):
        amp = ResidueAmplifier(gain=4.0, offset=0.01, settling_tau=0.0)
        # target is the pre-scaled residue; amp returns target + offset.
        self.assertAlmostEqual(amp.amplify(0.4, 0.0, 1e-9), 0.4 + 0.01)


class TestResidueAmplifierSettling(unittest.TestCase):
    """With settling_tau > 0, exponential approach to target."""

    def test_zero_initial_error_returns_target(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        self.assertAlmostEqual(amp.amplify(0.3, 0.0, 2e-9), 0.3)

    def test_one_tau_settles_to_one_over_e(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        result = amp.amplify(target=0.0, initial_error=1.0, t_budget=1e-9)
        self.assertAlmostEqual(result, math.exp(-1.0), places=12)

    def test_infinite_budget_full_settling(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        self.assertAlmostEqual(
            amp.amplify(target=0.5, initial_error=100.0, t_budget=float('inf')),
            0.5,
        )

    def test_zero_budget_preserves_initial_error(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        self.assertAlmostEqual(
            amp.amplify(target=0.5, initial_error=100.0, t_budget=0.0),
            100.5,
        )

    def test_negative_budget_overshoots(self):
        amp = ResidueAmplifier(gain=1.0, settling_tau=1e-9)
        result = amp.amplify(target=0.0, initial_error=1.0, t_budget=-1e-9)
        self.assertAlmostEqual(result, math.exp(1.0), places=12)


class TestResidueAmplifierConstructor(unittest.TestCase):
    """Validation at __init__ time."""

    def test_zero_gain_raises(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=0.0)

    def test_negative_settling_tau_raises(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=2.0, settling_tau=-1e-9)

    def test_negative_slew_rate_raises(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=2.0, slew_rate=-1.0)

    def test_output_swing_ordering(self):
        with self.assertRaises(ValueError):
            ResidueAmplifier(gain=2.0, output_swing=(0.5, -0.5))

    def test_negative_gain_allowed(self):
        # Inverting amps are valid; construction must succeed. The amp does
        # not apply gain inside amplify() — the caller pre-multiplies and
        # would pass a negative target for an inverting stage.
        amp = ResidueAmplifier(gain=-2.0, settling_tau=0.0)
        self.assertEqual(amp.gain, -2.0)
        self.assertAlmostEqual(amp.amplify(-0.2, 0.0, 1e-9), -0.2)


class TestResidueAmplifierClipping(unittest.TestCase):
    """Output swing clipping."""

    def test_output_swing_clips_positive(self):
        amp = ResidueAmplifier(gain=10.0, settling_tau=0.0, output_swing=(-1.0, 1.0))
        self.assertEqual(amp.amplify(5.0, 0.0, 1e-9), 1.0)

    def test_output_swing_clips_negative(self):
        amp = ResidueAmplifier(gain=10.0, settling_tau=0.0, output_swing=(-1.0, 1.0))
        self.assertEqual(amp.amplify(-5.0, 0.0, 1e-9), -1.0)

    def test_no_output_swing_unclipped(self):
        amp = ResidueAmplifier(gain=1000.0, settling_tau=0.0)
        self.assertAlmostEqual(amp.amplify(500.0, 0.0, 1e-9), 500.0)
