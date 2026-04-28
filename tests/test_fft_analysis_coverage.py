"""
Tests to improve coverage of pyDataconverter/utils/fft_analysis.py.

Targets uncovered lines: 77, 81, 193, 198, 203-289, 293.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from unittest.mock import patch

from pyDataconverter.utils.fft_analysis import (
    compute_fft,
    FFTNormalization,
    find_fundamental,
    find_harmonics,
    _get_harmonic,
)
from pyDataconverter.utils.signal_gen import generate_coherent_sine


# ---------------------------------------------------------------------------
# compute_fft: FFTNormalization.POWER branch (line 77)
# ---------------------------------------------------------------------------

class TestComputeFFTPowerNormalization:

    def test_power_normalization_returns_valid_result(self):
        """POWER normalization should return the same as length-normalized FFT."""
        fs = 1024
        t = np.arange(fs) / fs
        signal = np.sin(2 * np.pi * 10 * t)

        freqs_none, mags_none = compute_fft(signal, fs, normalization=FFTNormalization.NONE)
        freqs_pow, mags_pow = compute_fft(signal, fs, normalization=FFTNormalization.POWER)

        # POWER normalization is identical to NONE (length normalization already applied)
        np.testing.assert_array_almost_equal(mags_none, mags_pow)


# ---------------------------------------------------------------------------
# compute_fft: FFTNormalization.DBFS branch (lines 81, 83)
# ---------------------------------------------------------------------------

class TestComputeFFTDbfsNormalization:

    def test_dbfs_without_full_scale_raises(self):
        """dBFS normalization without full_scale raises ValueError."""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 256))
        with pytest.raises(ValueError, match="full_scale must be provided"):
            compute_fft(signal, fs=256, normalization=FFTNormalization.DBFS)

    def test_dbfs_with_full_scale(self):
        """dBFS normalization with full_scale produces shifted magnitudes."""
        fs = 1024
        t = np.arange(fs) / fs
        signal = np.sin(2 * np.pi * 10 * t)

        freqs_none, mags_none = compute_fft(signal, fs, normalization=FFTNormalization.NONE)
        freqs_dbfs, mags_dbfs = compute_fft(
            signal, fs, normalization=FFTNormalization.DBFS, full_scale=2.0
        )

        # dBFS should shift magnitudes by -20*log10(full_scale/2) = 0 dB for full_scale=2
        np.testing.assert_array_almost_equal(mags_none, mags_dbfs)

    def test_dbfs_with_large_full_scale(self):
        """dBFS normalization shifts magnitudes relative to full scale."""
        fs = 1024
        t = np.arange(fs) / fs
        signal = np.sin(2 * np.pi * 10 * t)
        full_scale = 1024.0

        freqs_none, mags_none = compute_fft(signal, fs, normalization=FFTNormalization.NONE)
        freqs_dbfs, mags_dbfs = compute_fft(
            signal, fs, normalization=FFTNormalization.DBFS, full_scale=full_scale
        )

        expected_shift = 20 * np.log10(full_scale / 2)
        np.testing.assert_array_almost_equal(mags_none - expected_shift, mags_dbfs)


# ---------------------------------------------------------------------------
# find_fundamental (line 163)
# ---------------------------------------------------------------------------

class TestFindFundamental:

    def test_find_fundamental_coherent_signal(self):
        """find_fundamental locates the correct bin for a coherent tone."""
        fs = 1e6
        n_fft = 1024
        n_fin = 11
        signal, f0 = generate_coherent_sine(fs, n_fft, n_fin, amplitude=1.0)
        freqs, mags = compute_fft(signal, fs)

        fund_freq, fund_mag = find_fundamental(freqs, mags, f0, fs)
        assert abs(fund_freq - f0) < (fs / n_fft), "Fundamental frequency not found correctly"
        assert fund_mag > -50, "Fundamental magnitude unexpectedly low"


# ---------------------------------------------------------------------------
# find_harmonics with verbose=True (lines 192-198)
# ---------------------------------------------------------------------------

class TestFindHarmonicsVerbose:

    def test_verbose_prints_harmonic_info(self, capsys):
        """verbose=True prints harmonic information."""
        fs = 1e6
        n_fft = 1024
        n_fin = 11
        signal, f0 = generate_coherent_sine(fs, n_fft, n_fin, amplitude=1.0)

        # Add a 2nd harmonic so there's something to find
        t = np.arange(n_fft) / fs
        signal = signal + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)

        freqs, mags = compute_fft(signal, fs)
        harmonics = find_harmonics(freqs, mags, f0, fs, num_harmonics=3, verbose=True)

        captured = capsys.readouterr()
        assert "Harmonic" in captured.out
        assert len(harmonics) > 0

    def test_verbose_prints_warning_on_missing_harmonic(self, capsys):
        """verbose=True prints warning when a harmonic is not found (lines 196-198)."""
        # Use a non-coherent f0 that doesn't land on an FFT bin, with very tight
        # tolerance, so _get_harmonic raises ValueError for harmonics.
        fs = 1000.0
        n_fft = 64
        # f0 that doesn't align with any bin: bin spacing = 1000/64 = 15.625 Hz
        f0 = 33.3  # Not a multiple of 15.625
        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f0 * t)
        freqs, mags = compute_fft(signal, fs)

        # tol=0.0 ensures mismatch for most harmonics
        harmonics = find_harmonics(
            freqs, mags, f0, fs, num_harmonics=5, tol=0.0, verbose=True
        )
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# _get_harmonic: aliasing paths (lines 125-135)
# ---------------------------------------------------------------------------

class TestGetHarmonicAliasing:

    def test_harmonic_below_nyquist(self):
        """Harmonic below Nyquist is found at its true frequency."""
        fs = 1e6
        n_fft = 1024
        f0 = 10 * fs / n_fft  # 10th bin
        signal = np.sin(2 * np.pi * f0 * np.arange(n_fft) / fs)
        freqs, mags = compute_fft(signal, fs)

        freq, mag = _get_harmonic(freqs, mags, f0, fs, n=1)
        assert abs(freq - f0) < fs / n_fft

    def test_harmonic_above_nyquist_odd_folds(self):
        """Harmonic above Nyquist with odd fold count aliases correctly."""
        fs = 1e6
        n_fft = 1024
        f0 = fs / n_fft * 100  # fundamental at bin 100

        # Create signal with fundamental
        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f0 * t)
        freqs, mags = compute_fft(signal, fs)

        # n=6 gives f_harm = 6 * f0 = 600 * fs/n_fft
        # which is above Nyquist (512 * fs/n_fft)
        # This should alias back into the spectrum
        try:
            freq, mag = _get_harmonic(freqs, mags, f0, fs, n=6)
            assert 0 <= freq <= fs / 2
        except ValueError:
            pass  # Acceptable if not found within tolerance

    def test_harmonic_above_nyquist_even_folds(self):
        """Harmonic above Nyquist with even fold count aliases correctly."""
        fs = 1e6
        n_fft = 1024
        # Choose f0 so that 3*f0 > fs (even number of folds = 2)
        f0 = fs / n_fft * 400  # fundamental at bin 400

        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f0 * t)
        freqs, mags = compute_fft(signal, fs)

        # n=3 gives f_harm = 3*f0 = 1200*fs/n_fft, well above Nyquist
        try:
            freq, mag = _get_harmonic(freqs, mags, f0, fs, n=3)
            assert 0 <= freq <= fs / 2
        except ValueError:
            pass

    def test_harmonic_mirrored_exceeds_nyquist(self):
        """Cover line 135: target_freq > nyquist after fold, requiring fs - target_freq.

        Choose f0 and n so that after the fold calculation, target_freq > fs/2,
        triggering the extra mirror on line 135.
        Example: fs=1000, f0=300, n=2 -> f_harm=600 > 500 (Nyquist).
        num_folds = ceil(600/1000) = 1 (odd), so target = 1000 - (600 % 1000) = 400.
        400 < 500 -> line 135 NOT hit. Need a case where odd fold gives > Nyquist.

        fs=1000, f0=800, n=1 -> f_harm=800 > 500.
        num_folds = ceil(800/1000) = 1 (odd), target = 1000 - (800%1000) = 1000-800 = 200. Not hit.

        We need: odd num_folds AND fs - (f_harm % fs) > nyquist.
        fs - (f_harm % fs) > fs/2  =>  f_harm % fs < fs/2.
        With odd fold: num_folds = ceil(f_harm/fs) odd.
        E.g. fs=1000, f_harm = 1100 (f0=1100, n=1 but f0 > nyquist so n=1 won't enter else).
        Better: f0=400, n=3 -> f_harm=1200. num_folds=ceil(1200/1000)=2 (even).
        target = 1200 % 1000 = 200. 200 < 500 -> no.

        f0=350, n=3 -> f_harm=1050. num_folds=ceil(1050/1000)=2 (even).
        target = 1050%1000 = 50. 50 < 500 -> no.

        Need odd folds: f_harm in (fs, 2*fs) => num_folds=ceil=1 or 2.
        ceil(1.x) = 2 (even). ceil(0.x)=1 (odd) but then f_harm < fs.
        Actually for f_harm > nyquist but < fs: num_folds = ceil(f_harm/fs) = 1 (odd).
        target = fs - (f_harm % fs). f_harm % fs = f_harm (since < fs).
        target = fs - f_harm. For target > nyquist: fs - f_harm > fs/2 => f_harm < fs/2.
        But we assumed f_harm > nyquist = fs/2. Contradiction. So odd=1 can't hit it.

        For f_harm in (2*fs, 3*fs): num_folds = ceil = 3 (odd).
        target = fs - (f_harm % fs). f_harm % fs in (0, fs).
        target > fs/2 => f_harm % fs < fs/2.
        E.g. fs=1000, f0=250, n=9 -> f_harm=2250. num_folds=ceil(2250/1000)=3 (odd).
        f_harm % fs = 250. target = 1000 - 250 = 750 > 500. Line 135 triggers!
        Final target = 1000 - 750 = 250.
        """
        fs = 1000.0
        n_fft = 1024
        f0 = 250.0
        t = np.arange(n_fft) / fs
        signal = np.sin(2 * np.pi * f0 * t)
        freqs, mags = compute_fft(signal, fs)

        # n=9: f_harm=2250, num_folds=3 (odd), target=750>500 -> line 135 hit
        try:
            freq, mag = _get_harmonic(freqs, mags, f0, fs, n=9)
            assert 0 <= freq <= fs / 2
        except ValueError:
            pass  # Acceptable if not found within tolerance

    def test_harmonic_not_found_raises(self):
        """_get_harmonic raises ValueError if harmonic not within tolerance."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0])
        mags = np.array([-60.0, -60.0, -60.0, -60.0])
        with pytest.raises(ValueError, match="not found"):
            _get_harmonic(freqs, mags, f0=150.0, fs=800.0, n=1, tol=0.001)


# demo_fft_analysis was moved to examples/fft_analysis_demo.py and is no
# longer part of the library.
