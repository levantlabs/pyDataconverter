"""
Tests for voltage reference components (ReferenceBase, ReferenceLadder, ArbitraryReference).
"""

import numpy as np
import pytest
from pyDataconverter.components.reference import ReferenceLadder, ArbitraryReference


# ---------------------------------------------------------------------------
# ReferenceLadder
# ---------------------------------------------------------------------------

class TestReferenceLadder:

    def test_n_references(self):
        """2^n_bits - 1 taps."""
        for n in [3, 4, 8]:
            ladder = ReferenceLadder(n, 0.0, 1.0)
            assert ladder.n_references == 2**n - 1

    def test_ideal_spacing_single_ended(self):
        """Ideal thresholds are uniformly spaced within (v_min, v_max)."""
        ladder = ReferenceLadder(3, 0.0, 1.0)
        expected = np.linspace(0.0, 1.0, 9)[1:-1]
        np.testing.assert_allclose(ladder.voltages, expected)

    def test_ideal_spacing_differential(self):
        """Differential range -v_ref/2 to +v_ref/2 is supported via v_min/v_max."""
        ladder = ReferenceLadder(3, -0.5, 0.5)
        expected = np.linspace(-0.5, 0.5, 9)[1:-1]
        np.testing.assert_allclose(ladder.voltages, expected)

    def test_resistor_mismatch_changes_voltages(self):
        """Mismatch shifts static voltages away from ideal."""
        np.random.seed(0)
        ladder_ideal    = ReferenceLadder(4, 0.0, 1.0)
        np.random.seed(0)
        ladder_mismatch = ReferenceLadder(4, 0.0, 1.0, resistor_mismatch=0.05)
        assert not np.allclose(ladder_ideal.voltages, ladder_mismatch.voltages)

    def test_mismatch_is_fixed_after_construction(self):
        """Static voltages must not change between calls."""
        np.random.seed(42)
        ladder = ReferenceLadder(3, 0.0, 1.0, resistor_mismatch=0.01)
        v1 = ladder.voltages
        v2 = ladder.voltages
        np.testing.assert_array_equal(v1, v2)

    def test_no_noise_returns_static(self):
        """get_voltages() with noise_rms=0 returns the static array."""
        ladder = ReferenceLadder(3, 0.0, 1.0)
        np.testing.assert_array_equal(ladder.get_voltages(), ladder.voltages)

    def test_noise_varies_per_call(self):
        """get_voltages() with noise_rms>0 returns different arrays each call."""
        ladder = ReferenceLadder(3, 0.0, 1.0, noise_rms=1e-3)
        v1 = ladder.get_voltages()
        v2 = ladder.get_voltages()
        assert not np.allclose(v1, v2)

    def test_noise_does_not_change_static_voltages(self):
        """Calling get_voltages() must not mutate the stored static voltages."""
        ladder = ReferenceLadder(3, 0.0, 1.0, noise_rms=1e-3)
        static_before = ladder.voltages.copy()
        for _ in range(10):
            ladder.get_voltages()
        np.testing.assert_array_equal(ladder.voltages, static_before)

    def test_v_min_equals_v_max_raises(self):
        """ReferenceLadder with v_min == v_max raises ValueError."""
        with pytest.raises(ValueError):
            ReferenceLadder(3, 0.5, 0.5)

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            ReferenceLadder(3, 1.0, 0.0)   # v_max <= v_min

    def test_invalid_mismatch(self):
        with pytest.raises(ValueError):
            ReferenceLadder(3, 0.0, 1.0, resistor_mismatch=-0.01)

    def test_invalid_noise(self):
        with pytest.raises(ValueError):
            ReferenceLadder(3, 0.0, 1.0, noise_rms=-1e-4)


# ---------------------------------------------------------------------------
# ArbitraryReference
# ---------------------------------------------------------------------------

class TestArbitraryReference:

    def test_n_references(self):
        ref = ArbitraryReference([0.1, 0.3, 0.5, 0.7, 0.9])
        assert ref.n_references == 5

    def test_voltages_match_input(self):
        thresholds = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        ref = ArbitraryReference(thresholds)
        np.testing.assert_allclose(ref.voltages, thresholds)

    def test_no_noise_returns_static(self):
        ref = ArbitraryReference([0.25, 0.5, 0.75])
        np.testing.assert_array_equal(ref.get_voltages(), ref.voltages)

    def test_noise_varies_per_call(self):
        ref = ArbitraryReference([0.25, 0.5, 0.75], noise_rms=1e-3)
        assert not np.allclose(ref.get_voltages(), ref.get_voltages())

    def test_voltages_copy_is_returned(self):
        """Modifying the returned array must not affect stored voltages."""
        ref = ArbitraryReference([0.25, 0.5, 0.75])
        v = ref.voltages
        v[0] = 999.0
        assert ref.voltages[0] != 999.0

    def test_not_strictly_increasing(self):
        with pytest.raises(ValueError):
            ArbitraryReference([0.5, 0.3, 0.7])   # not increasing

    def test_duplicate_thresholds(self):
        with pytest.raises(ValueError):
            ArbitraryReference([0.25, 0.5, 0.5, 0.75])

    def test_empty_thresholds(self):
        with pytest.raises(ValueError):
            ArbitraryReference([])

    def test_single_element_threshold(self):
        """ArbitraryReference with single-element threshold [0.5] is valid."""
        ref = ArbitraryReference([0.5])
        assert ref.n_references == 1
        np.testing.assert_allclose(ref.voltages, [0.5])

    def test_nan_threshold_raises(self):
        """ArbitraryReference with NaN threshold raises ValueError."""
        with pytest.raises(ValueError):
            ArbitraryReference([0.25, float('nan'), 0.75])

    def test_inf_threshold_raises(self):
        """ArbitraryReference with Inf threshold raises ValueError."""
        with pytest.raises(ValueError):
            ArbitraryReference([0.25, float('inf'), 0.75])

    def test_negative_noise(self):
        with pytest.raises(ValueError):
            ArbitraryReference([0.25, 0.5, 0.75], noise_rms=-1e-4)
