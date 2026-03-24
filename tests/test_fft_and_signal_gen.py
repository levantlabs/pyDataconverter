"""
Tests for FFT analysis and signal generation utilities.
"""

import numpy as np
import pytest
from pyDataconverter.utils.fft_analysis import compute_fft
from pyDataconverter.utils.signal_gen import generate_coherent_sine, generate_sine


# ---------------------------------------------------------------------------
# compute_fft: invalid window raises ValueError
# ---------------------------------------------------------------------------

class TestComputeFFTWindowValidation:

    def test_invalid_window_name_raises(self):
        """compute_fft with invalid window name raises ValueError."""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1024))
        with pytest.raises(ValueError, match="Unknown window"):
            compute_fft(signal, fs=1024, window='invalid_window')

    def test_valid_window_does_not_raise(self):
        """compute_fft with valid window names works."""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1024))
        for win in ['hann', 'hamming', 'blackman']:
            freqs, mags = compute_fft(signal, fs=1024, window=win)
            assert len(freqs) > 0


# ---------------------------------------------------------------------------
# compute_fft: remove_dc=False preserves DC bin
# ---------------------------------------------------------------------------

class TestComputeFFTRemoveDC:

    def test_remove_dc_false_preserves_dc(self):
        """With remove_dc=False, a signal with DC offset shows large DC bin."""
        dc_offset = 5.0
        signal = dc_offset + 0.1 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1024))

        freqs_dc, mags_dc = compute_fft(signal, fs=1024, remove_dc=False)
        freqs_no_dc, mags_no_dc = compute_fft(signal, fs=1024, remove_dc=True)

        # DC bin (index 0) should be much larger without DC removal
        assert mags_dc[0] > mags_no_dc[0] + 20, \
            "DC bin should be much larger when remove_dc=False"


# ---------------------------------------------------------------------------
# compute_fft: zero input does not produce -inf
# ---------------------------------------------------------------------------

class TestComputeFFTZeroInput:

    def test_zero_input_no_negative_inf(self):
        """All-zero input should not produce -inf magnitudes (log(0) guard)."""
        signal = np.zeros(256)
        freqs, mags = compute_fft(signal, fs=1000, remove_dc=False)
        assert np.all(np.isfinite(mags)), "Zero input should not produce -inf magnitudes"


# ---------------------------------------------------------------------------
# generate_coherent_sine: n_fin > n_fft//2
# ---------------------------------------------------------------------------

class TestGenerateCoherentSine:

    def test_n_fin_above_nyquist(self):
        """generate_coherent_sine with n_fin > n_fft//2 should still return a signal."""
        fs = 1e6
        n_fft = 1024
        n_fin = n_fft // 2 + 10  # above Nyquist bin

        # Should either raise or return an aliased signal
        try:
            signal, f_in = generate_coherent_sine(fs, n_fft, n_fin)
            # If it returns, verify signal has correct length
            assert len(signal) == n_fft
            assert f_in == n_fin / n_fft * fs
        except (ValueError, RuntimeError):
            pass  # Raising is also acceptable

    def test_coherent_sine_length_matches_nfft(self):
        """Output signal length equals n_fft."""
        fs = 1e6
        n_fft = 512
        n_fin = 7
        signal, f_in = generate_coherent_sine(fs, n_fft, n_fin)
        assert len(signal) == n_fft
