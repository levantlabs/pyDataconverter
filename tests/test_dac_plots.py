"""
Smoke tests for DAC visualization functions (dac_plots.py).

Each test creates a DAC, calls a plot function, and verifies it returns
a matplotlib Figure/Axes without raising.
"""

import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure

from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.visualizations.dac_plots import (
    plot_transfer_curve,
    plot_inl_dnl,
    plot_output_spectrum,
)


class TestDACPlots(unittest.TestCase):
    """Smoke tests for dac_plots visualization functions."""

    def setUp(self):
        self.n_bits = 8  # small for speed
        self.v_ref = 1.0
        self.dac_ideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE)
        self.dac_nonideal = SimpleDAC(self.n_bits, self.v_ref, OutputType.SINGLE,
                                      noise_rms=1e-4, offset=5e-3,
                                      gain_error=0.01)
        self.dac_diff = SimpleDAC(self.n_bits, self.v_ref, OutputType.DIFFERENTIAL)

    def tearDown(self):
        plt.close('all')

    # ------------------------------------------------------------------ #
    # plot_transfer_curve                                                  #
    # ------------------------------------------------------------------ #

    def test_transfer_curve_ideal(self):
        """plot_transfer_curve returns a Figure for an ideal DAC."""
        fig, (ax1, ax2) = plot_transfer_curve(self.dac_ideal)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_transfer_curve_nonideal(self):
        """plot_transfer_curve returns a Figure for a non-ideal DAC."""
        fig, (ax1, ax2) = plot_transfer_curve(self.dac_nonideal)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_transfer_curve_differential(self):
        """plot_transfer_curve works with a differential DAC."""
        fig, (ax1, ax2) = plot_transfer_curve(self.dac_diff)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_transfer_curve_custom_title(self):
        """plot_transfer_curve accepts a custom title."""
        fig, _ = plot_transfer_curve(self.dac_ideal, title="Custom Title")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    # ------------------------------------------------------------------ #
    # plot_inl_dnl                                                         #
    # ------------------------------------------------------------------ #

    def test_inl_dnl_ideal(self):
        """plot_inl_dnl returns a Figure for an ideal DAC."""
        fig, (ax1, ax2) = plot_inl_dnl(self.dac_ideal)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_inl_dnl_nonideal(self):
        """plot_inl_dnl returns a Figure for a non-ideal DAC."""
        fig, (ax1, ax2) = plot_inl_dnl(self.dac_nonideal)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_inl_dnl_differential(self):
        """plot_inl_dnl works with a differential DAC."""
        fig, (ax1, ax2) = plot_inl_dnl(self.dac_diff)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    # ------------------------------------------------------------------ #
    # plot_output_spectrum                                                 #
    # ------------------------------------------------------------------ #

    def test_output_spectrum_ideal(self):
        """plot_output_spectrum returns an Axes for an ideal DAC."""
        ax = plot_output_spectrum(self.dac_ideal, fs=1e6, f_sig=1e4,
                                  n_fft=256)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_output_spectrum_nonideal(self):
        """plot_output_spectrum returns an Axes for a non-ideal DAC."""
        ax = plot_output_spectrum(self.dac_nonideal, fs=1e6, f_sig=1e4,
                                  n_fft=256)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_output_spectrum_differential(self):
        """plot_output_spectrum works with a differential DAC."""
        ax = plot_output_spectrum(self.dac_diff, fs=1e6, f_sig=1e4,
                                  n_fft=256)
        self.assertIsInstance(ax, matplotlib.axes.Axes)


if __name__ == '__main__':
    unittest.main()
