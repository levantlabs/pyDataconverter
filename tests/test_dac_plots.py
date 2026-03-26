"""
Tests for DAC visualization functions (dac_plots.py) and plot_fft.

Covers:
  - plot_transfer_curve: smoke tests (ideal, non-ideal, differential)
  - plot_inl_dnl: smoke tests
  - plot_output_spectrum: new API (freqs/mags/fs), nyquist_zone 1/2/None, ZOH
  - plot_fft: min_freq / max_freq clipping behaviour
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure

from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.visualizations.dac_plots import (
    plot_transfer_curve,
    plot_inl_dnl,
    plot_output_spectrum,
)
from pyDataconverter.utils.visualizations.fft_plots import plot_fft
from pyDataconverter.utils.fft_analysis import compute_fft, FFTNormalization
from pyDataconverter.utils.signal_gen import generate_digital_sine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_spectrum(n_bits=8, v_ref=1.0, fs=1e6, f_sig=50e3, n_fft=512, oversample=1):
    """Return (freqs, mags, fs) for a simple DAC sine test."""
    dac = SimpleDAC(n_bits=n_bits, v_ref=v_ref, fs=fs, oversample=oversample)
    n_fin    = max(1, round(f_sig * n_fft / fs))
    f_actual = n_fin / n_fft * fs
    codes    = generate_digital_sine(n_bits, f_actual, fs,
                                     amplitude=0.9, offset=0.5,
                                     duration=n_fft / fs)
    _, voltages = dac.convert_sequence(codes)
    fs_fft = fs * oversample
    freqs, mags = compute_fft(voltages, fs_fft,
                               normalization=FFTNormalization.DBFS,
                               full_scale=v_ref)
    return freqs, mags, fs


# ---------------------------------------------------------------------------
# plot_transfer_curve
# ---------------------------------------------------------------------------

class TestPlotTransferCurve(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_ideal_returns_figure(self):
        dac = SimpleDAC(8, 1.0, OutputType.SINGLE)
        fig, _ = plot_transfer_curve(dac)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_nonideal_returns_figure(self):
        dac = SimpleDAC(8, 1.0, OutputType.SINGLE,
                        noise_rms=1e-4, offset=5e-3, gain_error=0.01)
        fig, _ = plot_transfer_curve(dac)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_differential_returns_figure(self):
        dac = SimpleDAC(8, 1.0, OutputType.DIFFERENTIAL)
        fig, _ = plot_transfer_curve(dac)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_custom_title(self):
        dac = SimpleDAC(8, 1.0)
        fig, _ = plot_transfer_curve(dac, title='My Title')
        self.assertIsInstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# plot_inl_dnl
# ---------------------------------------------------------------------------

class TestPlotInlDnl(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_ideal_returns_figure(self):
        dac = SimpleDAC(8, 1.0, OutputType.SINGLE)
        fig, _ = plot_inl_dnl(dac)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_nonideal_returns_figure(self):
        dac = SimpleDAC(8, 1.0, OutputType.SINGLE, gain_error=0.005, offset=2e-3)
        fig, _ = plot_inl_dnl(dac)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_differential_returns_figure(self):
        dac = SimpleDAC(8, 1.0, OutputType.DIFFERENTIAL)
        fig, _ = plot_inl_dnl(dac)
        self.assertIsInstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# plot_output_spectrum — new API
# ---------------------------------------------------------------------------

class TestPlotOutputSpectrum(unittest.TestCase):

    def setUp(self):
        self.freqs, self.mags, self.fs = _make_spectrum()

    def tearDown(self):
        plt.close('all')

    def test_returns_axes(self):
        """Default call returns a matplotlib Axes."""
        ax = plot_output_spectrum(self.freqs, self.mags, self.fs)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_zone1_xlim_max(self):
        """Zone 1: x-axis upper limit is fs/2."""
        ax = plot_output_spectrum(self.freqs, self.mags, self.fs, nyquist_zone=1)
        self.assertAlmostEqual(ax.get_xlim()[1],
                               self.fs / 2 / 1e3,   # kHz for 1 MHz fs
                               delta=1.0)

    def test_zone1_default(self):
        """nyquist_zone defaults to 1 — same result as explicit nyquist_zone=1."""
        ax_default  = plot_output_spectrum(self.freqs, self.mags, self.fs)
        ax_explicit = plot_output_spectrum(self.freqs, self.mags, self.fs, nyquist_zone=1)
        self.assertEqual(ax_default.get_xlim(), ax_explicit.get_xlim())

    def test_zone2_xlim(self):
        """Zone 2: x-axis spans [fs/2, fs]."""
        freqs, mags, fs = _make_spectrum(oversample=4)
        ax = plot_output_spectrum(freqs, mags, fs, nyquist_zone=2)
        xmin, xmax = ax.get_xlim()
        # max_freq=1 MHz triggers MHz unit, so values are in MHz
        self.assertAlmostEqual(xmin, fs / 2 / 1e6, delta=0.1)
        self.assertAlmostEqual(xmax, fs / 1e6,      delta=0.1)

    def test_all_zones_returns_axes(self):
        """nyquist_zone=None (all zones) returns an Axes."""
        freqs, mags, fs = _make_spectrum(oversample=4)
        ax = plot_output_spectrum(freqs, mags, fs, nyquist_zone=None)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_all_zones_xlim_max(self):
        """All-zones plot x-axis extends to fs_fft/2."""
        oversample = 4
        freqs, mags, fs = _make_spectrum(oversample=oversample)
        ax = plot_output_spectrum(freqs, mags, fs, nyquist_zone=None)
        fs_fft = fs * oversample
        self.assertAlmostEqual(ax.get_xlim()[1],
                               fs_fft / 2 / 1e6,   # MHz for 4 MHz fs_fft
                               delta=0.1)

    def test_custom_title_appears(self):
        """Custom title is set on the axes."""
        ax = plot_output_spectrum(self.freqs, self.mags, self.fs,
                                  title='Test Title')
        self.assertIn('Test Title', ax.get_title())

    def test_zoh_oversampled_zone1(self):
        """ZOH DAC (oversample=8), zone 1 — returns Axes without error."""
        freqs, mags, fs = _make_spectrum(oversample=8)
        ax = plot_output_spectrum(freqs, mags, fs, nyquist_zone=1)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_zoh_oversampled_all_zones(self):
        """ZOH DAC (oversample=8), all zones — returns Axes without error."""
        freqs, mags, fs = _make_spectrum(oversample=8)
        ax = plot_output_spectrum(freqs, mags, fs, nyquist_zone=None)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_enob_is_positive(self):
        """ENOB must be positive for an ideal 8-bit DAC sine input."""
        import pyDataconverter.utils.metrics as m_mod
        freqs, mags, fs = _make_spectrum(n_bits=8, n_fft=1024)
        zone_mask = freqs <= fs / 2
        fz = freqs[zone_mask]
        mz = mags[zone_mask]
        pos = fz > 0
        f0  = fz[pos][np.argmax(mz[pos])]
        res = m_mod.calculate_dac_dynamic_metrics(freqs=fz, mags=mz, fs=fs, f0=f0)
        self.assertGreater(res['ENOB'], 0,
                           f"ENOB should be positive, got {res['ENOB']:.2f}")


# ---------------------------------------------------------------------------
# plot_fft — min_freq / max_freq
# ---------------------------------------------------------------------------

class TestPlotFft(unittest.TestCase):

    def setUp(self):
        """Build a simple synthetic spectrum for all tests."""
        fs = 1e6
        N  = 512
        t  = np.arange(N) / fs
        sig = np.sin(2 * np.pi * 50e3 * t)
        self.freqs, self.mags = compute_fft(sig, fs,
                                             normalization=FFTNormalization.DBFS,
                                             full_scale=1.0)
        self.fs = fs

    def tearDown(self):
        plt.close('all')

    def test_default_starts_at_zero(self):
        """With no min_freq, x-axis starts at 0."""
        ax = plot_fft(self.freqs, self.mags)
        self.assertEqual(ax.get_xlim()[0], 0.0)

    def test_max_freq_clips_upper(self):
        """max_freq clips the upper x-axis limit."""
        ax = plot_fft(self.freqs, self.mags, max_freq=self.fs / 2)
        xmax = ax.get_xlim()[1]
        # x-axis is in kHz (fs/2 = 500 kHz), ceil → 500
        self.assertAlmostEqual(xmax, 500.0, delta=1.0)

    def test_min_freq_shifts_lower_bound(self):
        """min_freq shifts the x-axis lower bound away from zero."""
        min_f = self.fs / 4   # 250 kHz
        ax = plot_fft(self.freqs, self.mags,
                      min_freq=min_f, max_freq=self.fs / 2)
        xmin = ax.get_xlim()[0]
        self.assertGreater(xmin, 0.0)

    def test_min_freq_none_unchanged(self):
        """min_freq=None gives the same result as not passing it."""
        ax1 = plot_fft(self.freqs, self.mags)
        ax2 = plot_fft(self.freqs, self.mags, min_freq=None)
        self.assertEqual(ax1.get_xlim(), ax2.get_xlim())

    def test_nfft_in_title(self):
        """NFFT count is shown in the title."""
        ax = plot_fft(self.freqs, self.mags, title='Test')
        expected_nfft = len(self.freqs) * 2
        self.assertIn(f'NFFT={expected_nfft}', ax.get_title())

    def test_returns_axes(self):
        """plot_fft returns a matplotlib Axes."""
        ax = plot_fft(self.freqs, self.mags)
        self.assertIsInstance(ax, matplotlib.axes.Axes)


if __name__ == '__main__':
    unittest.main()
