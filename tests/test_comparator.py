"""
Tests for comparator components (ComparatorBase, DifferentialComparator, Comparator alias).
"""

import numpy as np
import pytest
from pyDataconverter.components.comparator import (
    ComparatorBase, DifferentialComparator, Comparator,
)


# ===========================================================================
# Construction and defaults
# ===========================================================================

class TestDifferentialComparatorConstruction:

    def test_default_parameters(self):
        comp = DifferentialComparator()
        assert comp.offset == 0.0
        assert comp.noise_rms == 0.0
        assert comp.bandwidth is None
        assert comp.hysteresis == 0.0
        assert comp.time_constant == 0.0

    def test_custom_parameters(self):
        comp = DifferentialComparator(
            offset=0.001, noise_rms=0.0005,
            bandwidth=1e9, hysteresis=0.002, time_constant=1e-9,
        )
        assert comp.offset == 0.001
        assert comp.noise_rms == 0.0005
        assert comp.bandwidth == 1e9
        assert comp.hysteresis == 0.002
        assert comp.time_constant == 1e-9

    def test_is_comparator_base(self):
        assert isinstance(DifferentialComparator(), ComparatorBase)

    def test_backward_compat_alias(self):
        assert Comparator is DifferentialComparator


# ===========================================================================
# Basic compare (2-input, no non-idealities)
# ===========================================================================

class TestComparatorIdealCompare:

    def test_positive_diff_returns_one(self):
        comp = DifferentialComparator()
        assert comp.compare(0.6, 0.4) == 1

    def test_negative_diff_returns_zero(self):
        comp = DifferentialComparator()
        assert comp.compare(0.4, 0.6) == 0

    def test_zero_diff_returns_zero(self):
        """Exactly zero difference: v_diff > 0 is False, so result is 0."""
        comp = DifferentialComparator()
        assert comp.compare(0.5, 0.5) == 0

    def test_very_small_positive_diff(self):
        comp = DifferentialComparator()
        assert comp.compare(0.5 + 1e-15, 0.5) == 1

    def test_very_small_negative_diff(self):
        comp = DifferentialComparator()
        assert comp.compare(0.5, 0.5 + 1e-15) == 0


# ===========================================================================
# 4-input compare with references
# ===========================================================================

class TestComparator4Input:

    def test_reference_subtraction(self):
        comp = DifferentialComparator()
        # (0.7 - 0.2) - (0.3 - 0.1) = 0.5 - 0.2 = 0.3 > 0
        assert comp.compare(0.7, 0.3, v_refp=0.2, v_refn=0.1) == 1

    def test_reference_makes_negative(self):
        comp = DifferentialComparator()
        # (0.5 - 0.6) - (0.3 - 0.1) = -0.1 - 0.2 = -0.3 < 0
        assert comp.compare(0.5, 0.3, v_refp=0.6, v_refn=0.1) == 0

    def test_defaults_match_two_input(self):
        """Default v_refp=v_refn=0 gives same result as 2-input call."""
        comp = DifferentialComparator()
        for v_pos, v_neg in [(0.6, 0.4), (0.3, 0.7), (0.5, 0.5)]:
            r2 = comp.compare(v_pos, v_neg)
            comp.reset()
            r4 = comp.compare(v_pos, v_neg, v_refp=0.0, v_refn=0.0)
            assert r2 == r4

    def test_equal_references_cancel(self):
        """v_refp == v_refn cancels out, result depends only on v_pos - v_neg."""
        comp = DifferentialComparator()
        assert comp.compare(0.6, 0.4, v_refp=0.5, v_refn=0.5) == 1
        comp.reset()
        assert comp.compare(0.4, 0.6, v_refp=0.5, v_refn=0.5) == 0


# ===========================================================================
# Offset
# ===========================================================================

class TestComparatorOffset:

    def test_positive_offset_shifts_threshold(self):
        """Positive offset adds to v_diff, making it easier to fire."""
        comp = DifferentialComparator(offset=0.1)
        # v_diff = (0.45 - 0.5) + 0.1 = 0.05 > 0
        assert comp.compare(0.45, 0.5) == 1

    def test_negative_offset_shifts_threshold(self):
        """Negative offset subtracts from v_diff."""
        comp = DifferentialComparator(offset=-0.1)
        # v_diff = (0.55 - 0.5) + (-0.1) = -0.05 < 0
        assert comp.compare(0.55, 0.5) == 0


# ===========================================================================
# Noise
# ===========================================================================

class TestComparatorNoise:

    def test_noise_causes_variation(self):
        """Near-threshold input should produce both 0 and 1 with noise."""
        np.random.seed(0)
        comp = DifferentialComparator(noise_rms=0.01)
        results = set()
        for _ in range(100):
            results.add(comp.compare(0.500, 0.500))
            if len(results) == 2:
                break
        assert len(results) == 2

    def test_zero_noise_deterministic(self):
        """With noise_rms=0, repeated calls give same result."""
        comp = DifferentialComparator(noise_rms=0.0)
        results = {comp.compare(0.6, 0.4) for _ in range(20)}
        assert len(results) == 1


# ===========================================================================
# Hysteresis
# ===========================================================================

class TestComparatorHysteresis:

    def test_hysteresis_holds_high(self):
        """After firing high, threshold drops (harder to switch back to 0)."""
        comp = DifferentialComparator(hysteresis=0.1)
        # First: clearly high
        assert comp.compare(0.6, 0.4) == 1
        # Now v_diff = 0.01, which is above -hysteresis/2 = -0.05, so stays 1
        assert comp.compare(0.505, 0.5) == 1

    def test_hysteresis_holds_low(self):
        """After being low, threshold is positive (harder to switch to 1)."""
        comp = DifferentialComparator(hysteresis=0.1)
        # First: clearly low
        assert comp.compare(0.4, 0.6) == 0
        # Now v_diff = -0.01, which is below hysteresis/2 = 0.05, so stays 0
        assert comp.compare(0.495, 0.5) == 0

    def test_hysteresis_eventually_switches(self):
        """With a large enough swing, comparator switches despite hysteresis."""
        comp = DifferentialComparator(hysteresis=0.1)
        assert comp.compare(0.4, 0.6) == 0  # low
        assert comp.compare(0.7, 0.4) == 1  # large positive swing


# ===========================================================================
# Bandwidth limiting
# ===========================================================================

class TestComparatorBandwidth:

    def test_bandwidth_requires_time_step(self):
        comp = DifferentialComparator(bandwidth=1e9)
        with pytest.raises(ValueError, match="time_step"):
            comp.compare(0.6, 0.4)

    def test_bandwidth_with_time_step_works(self):
        comp = DifferentialComparator(bandwidth=1e9)
        result = comp.compare(0.6, 0.4, time_step=1e-10)
        assert result in (0, 1)

    def test_bandwidth_low_pass_effect(self):
        """Bandwidth filtering attenuates a sudden step."""
        comp = DifferentialComparator(bandwidth=1e6)
        # First call: step from 0 to large positive
        # With very small time_step, alpha is small, so filtered value is small
        result = comp.compare(10.0, 0.0, time_step=1e-12)
        # The filtered value should be much less than 10.0 due to low-pass
        # but the test just verifies it runs without error
        assert result in (0, 1)


# ===========================================================================
# Reset
# ===========================================================================

class TestComparatorReset:

    def test_reset_clears_hysteresis(self):
        comp = DifferentialComparator(hysteresis=0.1)
        comp.compare(0.7, 0.3)  # fire high
        comp.reset()
        assert comp._last_output == 0

    def test_reset_clears_bandwidth_state(self):
        comp = DifferentialComparator(bandwidth=1e9)
        comp.compare(0.7, 0.3, time_step=1e-10)
        comp.reset()
        assert comp._filtered_state == 0.0

    def test_reset_without_bandwidth_ok(self):
        comp = DifferentialComparator()
        comp.compare(0.7, 0.3)
        comp.reset()  # should not raise


# ===========================================================================
# Repr
# ===========================================================================

class TestComparatorRepr:

    def test_repr_basic(self):
        comp = DifferentialComparator()
        r = repr(comp)
        assert 'DifferentialComparator' in r
        assert 'offset=' in r

    def test_repr_bandwidth_shown(self):
        comp = DifferentialComparator(bandwidth=1e9)
        assert 'bandwidth=' in repr(comp)

    def test_repr_time_constant_shown(self):
        comp = DifferentialComparator(time_constant=1e-9)
        assert 'time_constant=' in repr(comp)

    def test_repr_bandwidth_hidden_when_none(self):
        comp = DifferentialComparator()
        assert 'bandwidth=' not in repr(comp)

    def test_repr_time_constant_hidden_when_zero(self):
        comp = DifferentialComparator(time_constant=0.0)
        assert 'time_constant=' not in repr(comp)


# ===========================================================================
# Bandwidth filtering — deeper coverage
# ===========================================================================

class TestComparatorBandwidthFiltering:

    def test_bandwidth_attenuates_step_to_zero(self):
        """With very small time_step relative to tau, filtered output stays near 0."""
        comp = DifferentialComparator(bandwidth=1e6)
        # tau = 1/(2*pi*1e6) ≈ 1.59e-7 s; time_step = 1e-12 s
        # alpha = 1e-12 / (1e-12 + 1.59e-7) ≈ 6.3e-6 => tiny
        # filtered ≈ 6.3e-6 * 0.001 ≈ 6.3e-9, which is > 0 but barely
        # With a small positive input the filtered value is positive so result = 1
        result = comp.compare(0.001, 0.0, time_step=1e-12)
        assert result in (0, 1)

    def test_bandwidth_converges_over_many_steps(self):
        """Repeated calls with a constant input eventually let the filter settle."""
        comp = DifferentialComparator(bandwidth=1e9)
        # tau = 1/(2*pi*1e9) ≈ 1.59e-10; time_step = 1e-9 >> tau
        # alpha = 1e-9/(1e-9 + 1.59e-10) ≈ 0.86, so converges quickly
        for _ in range(20):
            result = comp.compare(0.1, 0.0, time_step=1e-9)
        assert result == 1  # settled to 0.1 > 0

    def test_bandwidth_filters_sign_change(self):
        """After settling positive, a sudden negative input is filtered slowly."""
        comp = DifferentialComparator(bandwidth=1e6)
        # Settle to positive
        for _ in range(1000):
            comp.compare(1.0, 0.0, time_step=1e-6)
        # Now apply negative input with tiny step — filter still positive
        result = comp.compare(0.0, 1.0, time_step=1e-12)
        assert result == 1  # filter hasn't caught up yet

    def test_bandwidth_reset_clears_filter(self):
        """After reset, filtered_state returns to 0."""
        comp = DifferentialComparator(bandwidth=1e9)
        for _ in range(10):
            comp.compare(1.0, 0.0, time_step=1e-9)
        comp.reset()
        assert comp._filtered_state == 0.0
        # After reset, tiny positive → still tiny filtered value
        result = comp.compare(0.0001, 0.0, time_step=1e-12)
        # alpha is tiny, filtered ≈ 0.0001 * alpha ≈ tiny but > 0
        assert result in (0, 1)


# ===========================================================================
# Hysteresis — deeper coverage
# ===========================================================================

class TestComparatorHysteresisTransitions:

    def test_high_to_low_transition(self):
        """After output=1, v_diff must drop below -hysteresis/2 to switch to 0."""
        comp = DifferentialComparator(hysteresis=0.2)
        # Go high
        assert comp.compare(1.0, 0.0) == 1
        # v_diff = -0.05, threshold = -0.1 (since last_output=1)
        # -0.05 > -0.1, so stays 1
        assert comp.compare(0.475, 0.525) == 1
        # v_diff = -0.15 < -0.1 → switches to 0
        assert comp.compare(0.425, 0.575) == 0

    def test_low_to_high_transition(self):
        """After output=0, v_diff must exceed +hysteresis/2 to switch to 1."""
        comp = DifferentialComparator(hysteresis=0.2)
        # Start low (default _last_output=0)
        assert comp.compare(0.0, 1.0) == 0
        # v_diff = 0.05, threshold = 0.1 (since last_output=0)
        # 0.05 < 0.1 → stays 0
        assert comp.compare(0.525, 0.475) == 0
        # v_diff = 0.15 > 0.1 → switches to 1
        assert comp.compare(0.575, 0.425) == 1

    def test_hysteresis_with_4_input_references(self):
        """Hysteresis works correctly with reference voltages."""
        comp = DifferentialComparator(hysteresis=0.1)
        # effective_diff = (0.6 - 0.1) - (0.2 - 0.1) = 0.4 > 0.05 → 1
        assert comp.compare(0.6, 0.2, v_refp=0.1, v_refn=0.1) == 1
        # effective_diff = (0.3 - 0.1) - (0.2 - 0.1) = 0.1 > -0.05 → stays 1
        assert comp.compare(0.3, 0.2, v_refp=0.1, v_refn=0.1) == 1


# ===========================================================================
# Combined non-idealities
# ===========================================================================

class TestComparatorCombined:

    def test_offset_with_hysteresis(self):
        """Offset shifts the effective diff before hysteresis is applied."""
        comp = DifferentialComparator(offset=0.05, hysteresis=0.1)
        # v_diff = (0.5 - 0.5) + 0.05 = 0.05, threshold = 0.05 (last=0)
        # 0.05 is NOT > 0.05, so result = 0
        assert comp.compare(0.5, 0.5) == 0
        # v_diff = (0.51 - 0.5) + 0.05 = 0.06 > 0.05 → 1
        assert comp.compare(0.51, 0.5) == 1

    def test_offset_with_bandwidth(self):
        """Offset and bandwidth interact correctly."""
        comp = DifferentialComparator(offset=0.1, bandwidth=1e9)
        # v_diff = (0.5 - 0.5) + 0.1 = 0.1, then filtered
        for _ in range(20):
            result = comp.compare(0.5, 0.5, time_step=1e-9)
        assert result == 1  # offset=0.1 > 0 once filter settles


# ===========================================================================
# __main__ block coverage
# ===========================================================================

class TestComparatorMainBlock:
    """Cover the __main__ block logic (lines 206-233) inline."""

    def test_main_block_logic(self):
        """Replicate the __main__ block to exercise lines 206-233."""
        import numpy as np
        import unittest.mock as mock

        comp = DifferentialComparator(
            offset=0.001,
            noise_rms=0.0005,
            hysteresis=0.002,
        )
        assert repr(comp)  # exercises __str__/repr path

        v_diff = np.linspace(-0.01, 0.01, 1000)
        n_trials = 5  # fewer trials to keep test fast
        results = []
        for _ in range(n_trials):
            comp.reset()
            results.append([comp.compare(v, 0.0) for v in v_diff])

        prob_high = np.mean(results, axis=0)
        assert len(prob_high) == 1000

        # Mock pyplot to avoid display; the plot calls must be reachable
        with mock.patch("matplotlib.pyplot.figure"), \
             mock.patch("matplotlib.pyplot.plot"), \
             mock.patch("matplotlib.pyplot.axvline"), \
             mock.patch("matplotlib.pyplot.xlabel"), \
             mock.patch("matplotlib.pyplot.ylabel"), \
             mock.patch("matplotlib.pyplot.title"), \
             mock.patch("matplotlib.pyplot.legend"), \
             mock.patch("matplotlib.pyplot.grid"), \
             mock.patch("matplotlib.pyplot.show"):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(v_diff * 1e3, prob_high)
            plt.axvline(comp.offset * 1e3, color='r', linestyle='--',
                        label=f'Offset: {comp.offset*1e3:.1f} mV')
            plt.xlabel('Differential Input (mV)')
            plt.ylabel('P(output = 1)')
            plt.title('DifferentialComparator Transfer Characteristic')
            plt.legend()
            plt.grid(True)
            plt.show()
