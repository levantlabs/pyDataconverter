"""
Comprehensive tests for pyDataconverter.utils.signal_gen module.
"""

import numpy as np
import pytest
from pyDataconverter.utils.signal_gen import (
    convert_to_differential,
    generate_sine,
    generate_ramp,
    generate_step,
    generate_two_tone,
    generate_multitone,
    generate_imd_tones,
    generate_digital_ramp,
    generate_digital_step,
    generate_digital_sine,
    generate_digital_two_tone,
    generate_digital_multitone,
    generate_digital_imd_tones,
    generate_coherent_sine,
    generate_coherent_two_tone,
)
from pyDataconverter.utils.signal_gen import generate_chirp


# ---- generate_sine --------------------------------------------------------

class TestGenerateSine:

    def test_basic_length(self):
        sig = generate_sine(100, 1000, duration=0.01)
        assert len(sig) == 10

    def test_amplitude(self):
        sig = generate_sine(1.0, 1000, amplitude=2.5, duration=1.0)
        assert pytest.approx(np.max(sig), abs=0.01) == 2.5

    def test_offset(self):
        sig = generate_sine(1.0, 1000, amplitude=1.0, offset=3.0, duration=1.0)
        assert pytest.approx(np.mean(sig), abs=0.05) == 3.0

    def test_phase_shift(self):
        sig0 = generate_sine(1.0, 1000, phase=0.0, duration=1.0)
        sig90 = generate_sine(1.0, 1000, phase=np.pi / 2, duration=1.0)
        # At t=0, sin(0)=0, sin(pi/2)=1
        assert abs(sig0[0]) < 0.01
        assert pytest.approx(sig90[0], abs=0.01) == 1.0

    def test_zero_frequency_gives_dc(self):
        sig = generate_sine(0.0, 1000, amplitude=1.0, offset=2.0, duration=0.01)
        np.testing.assert_allclose(sig, 2.0, atol=1e-10)


# ---- generate_ramp --------------------------------------------------------

class TestGenerateRamp:

    def test_length(self):
        sig = generate_ramp(100)
        assert len(sig) == 100

    def test_endpoints(self):
        sig = generate_ramp(50, v_min=-1.0, v_max=1.0)
        assert pytest.approx(sig[0]) == -1.0
        assert pytest.approx(sig[-1]) == 1.0

    def test_monotonic(self):
        sig = generate_ramp(200, v_min=0.0, v_max=5.0)
        assert np.all(np.diff(sig) >= 0)

    def test_single_sample(self):
        sig = generate_ramp(1, v_min=0.0, v_max=1.0)
        assert len(sig) == 1


# ---- generate_step --------------------------------------------------------

class TestGenerateStep:

    def test_basic_step(self):
        # levels[0] is the initial level (set via signal init = zeros),
        # step_points are paired with levels[1:] via zip
        sig = generate_step(100, step_points=[50], levels=[0.0, 1.0])
        assert sig[0] == 0.0
        assert sig[49] == 0.0
        assert sig[50] == 1.0
        assert sig[99] == 1.0

    def test_multi_level(self):
        sig = generate_step(300, step_points=[100, 200], levels=[0.0, 1.0, 2.0])
        assert sig[0] == 0.0
        assert sig[99] == 0.0
        assert sig[100] == 1.0
        assert sig[200] == 2.0
        assert sig[299] == 2.0

    def test_output_length(self):
        sig = generate_step(500, step_points=[0, 250], levels=[0.0, 3.5])
        assert len(sig) == 500


# ---- generate_two_tone ----------------------------------------------------

class TestGenerateTwoTone:

    def test_length(self):
        sig = generate_two_tone(100, 200, 10000, duration=0.01)
        assert len(sig) == 100

    def test_superposition(self):
        fs = 10000
        dur = 1.0
        sig = generate_two_tone(100, 200, fs, amplitude1=1.0, amplitude2=1.0, duration=dur)
        t = np.arange(0, dur, 1 / fs)
        expected = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t)
        np.testing.assert_allclose(sig, expected, atol=1e-10)

    def test_phase_control(self):
        fs = 10000
        dur = 0.01
        sig = generate_two_tone(100, 200, fs, amplitude1=1.0, amplitude2=1.0,
                                phase1=np.pi / 2, phase2=0.0, duration=dur)
        # At t=0: sin(pi/2) + sin(0) = 1
        assert pytest.approx(sig[0], abs=0.01) == 1.0


# ---- generate_multitone ---------------------------------------------------

class TestGenerateMultitone:

    def test_length(self):
        sig = generate_multitone([100, 200, 300], 10000, duration=0.01)
        assert len(sig) == 100

    def test_default_amplitudes(self):
        sig = generate_multitone([100], 10000, duration=1.0)
        assert pytest.approx(np.max(np.abs(sig)), abs=0.01) == 1.0

    def test_custom_amplitudes_and_phases(self):
        sig = generate_multitone([100], 10000, amplitudes=[2.0], phases=[0.0], duration=1.0)
        assert pytest.approx(np.max(np.abs(sig)), abs=0.01) == 2.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            generate_multitone([100, 200], 10000, amplitudes=[1.0])

    def test_mismatched_phases_raises(self):
        with pytest.raises(ValueError, match="must match"):
            generate_multitone([100, 200], 10000, phases=[0.0])

    def test_single_tone_matches_sine(self):
        fs = 10000
        dur = 0.1
        sig_mt = generate_multitone([500], fs, amplitudes=[1.0], phases=[0.0], duration=dur)
        sig_sin = generate_sine(500, fs, amplitude=1.0, duration=dur)
        np.testing.assert_allclose(sig_mt, sig_sin, atol=1e-10)


# ---- generate_imd_tones ---------------------------------------------------

class TestGenerateImdTones:

    def test_returns_signal_and_dict(self):
        sig, freqs = generate_imd_tones(1e6, 1e3, 10e6)
        assert isinstance(sig, np.ndarray)
        assert isinstance(freqs, dict)

    def test_frequencies_correct(self):
        f1, delta_f = 1e6, 1e3
        _, freqs = generate_imd_tones(f1, delta_f, 10e6)
        assert freqs['f1'] == f1
        assert freqs['f2'] == f1 + delta_f
        assert freqs['imd2']['f1+f2'] == 2 * f1 + delta_f
        assert freqs['imd2']['f2-f1'] == delta_f

    def test_order3_products(self):
        f1, delta_f = 1e6, 1e3
        _, freqs = generate_imd_tones(f1, delta_f, 10e6, order=3)
        assert 'imd3' in freqs
        assert freqs['imd3']['2f1-f2'] == 2 * f1 - (f1 + delta_f)
        assert freqs['imd3']['2f2-f1'] == 2 * (f1 + delta_f) - f1

    def test_order2_no_imd3(self):
        _, freqs = generate_imd_tones(1e6, 1e3, 10e6, order=2)
        assert 'imd3' not in freqs
        assert 'imd2' in freqs

    def test_signal_length(self):
        sig, _ = generate_imd_tones(1e6, 1e3, 10e6, duration=0.001)
        assert len(sig) == 10000


# ---- generate_digital_ramp ------------------------------------------------

class TestGenerateDigitalRamp:

    def test_default_length(self):
        sig = generate_digital_ramp(8)
        assert len(sig) == 256

    def test_custom_length(self):
        sig = generate_digital_ramp(12, n_points=100)
        assert len(sig) == 100

    def test_range(self):
        sig = generate_digital_ramp(8)
        assert sig[0] == 0
        assert sig[-1] == 255

    def test_monotonic(self):
        sig = generate_digital_ramp(10, n_points=500)
        assert np.all(np.diff(sig) >= 0)

    def test_integer_type(self):
        sig = generate_digital_ramp(8)
        assert sig.dtype == int


# ---- generate_digital_step ------------------------------------------------

class TestGenerateDigitalStep:

    def test_basic(self):
        # step_points[-1] determines array length; step_points[1:] paired with levels[1:]
        sig = generate_digital_step(8, [0, 50, 100], [0, 128, 200])
        assert len(sig) == 100
        assert sig[0] == 0
        assert sig[50] == 128
        assert sig[99] == 128

    def test_output_length(self):
        sig = generate_digital_step(8, [0, 50, 100], [0, 100, 200])
        assert len(sig) == 100

    def test_level_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="less than"):
            generate_digital_step(8, [0, 50], [0, 256])

    def test_negative_level_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            generate_digital_step(8, [0, 50], [0, -1])


# ---- generate_digital_sine ------------------------------------------------

class TestGenerateDigitalSine:

    def test_output_range(self):
        sig = generate_digital_sine(12, 1e3, 1e6, duration=0.001)
        assert np.all(sig >= 0)
        assert np.all(sig <= 4095)

    def test_integer_output(self):
        sig = generate_digital_sine(8, 100, 10000, duration=0.01)
        assert sig.dtype == int

    def test_length(self):
        sig = generate_digital_sine(10, 100, 10000, duration=0.01)
        assert len(sig) == 100

    def test_centered_by_default(self):
        sig = generate_digital_sine(12, 100, 100000, duration=0.1)
        max_code = 4095
        mean_code = np.mean(sig)
        assert abs(mean_code - max_code * 0.5) < max_code * 0.05


# ---- generate_digital_two_tone --------------------------------------------

class TestGenerateDigitalTwoTone:

    def test_output_range(self):
        sig = generate_digital_two_tone(12, 1e3, 2e3, 1e6, duration=0.001)
        assert np.all(sig >= 0)
        assert np.all(sig <= 4095)

    def test_length(self):
        sig = generate_digital_two_tone(10, 100, 200, 10000, duration=0.01)
        assert len(sig) == 100

    def test_integer_output(self):
        sig = generate_digital_two_tone(8, 100, 200, 10000, duration=0.01)
        assert sig.dtype == int


# ---- generate_digital_multitone -------------------------------------------

class TestGenerateDigitalMultitone:

    def test_output_range(self):
        sig = generate_digital_multitone(12, [1e3, 2e3], 1e6, duration=0.001)
        assert np.all(sig >= 0)
        assert np.all(sig <= 4095)

    def test_default_amplitudes(self):
        sig = generate_digital_multitone(12, [1e3, 2e3, 3e3], 1e6, duration=0.001)
        # np.arange can produce slightly more samples due to float rounding
        assert len(sig) >= 1000

    def test_custom_amplitudes(self):
        sig = generate_digital_multitone(12, [1e3, 2e3], 1e6,
                                         amplitudes=[0.3, 0.3], duration=0.001)
        assert np.all(sig >= 0)
        assert np.all(sig <= 4095)

    def test_integer_output(self):
        sig = generate_digital_multitone(8, [100, 200], 10000, duration=0.01)
        assert sig.dtype == int


# ---- generate_digital_imd_tones -------------------------------------------

class TestGenerateDigitalImdTones:

    def test_returns_signal_and_dict(self):
        sig, freqs = generate_digital_imd_tones(12, 1e6, 1e3, 10e6, duration=0.001)
        assert isinstance(sig, np.ndarray)
        assert isinstance(freqs, dict)

    def test_frequencies(self):
        f1, delta_f = 1e6, 1e3
        _, freqs = generate_digital_imd_tones(12, f1, delta_f, 10e6)
        assert freqs['f1'] == f1
        assert freqs['f2'] == f1 + delta_f
        assert 'imd2' in freqs
        assert 'imd3' in freqs

    def test_signal_range(self):
        sig, _ = generate_digital_imd_tones(12, 1e6, 1e3, 10e6, duration=0.001)
        assert np.all(sig >= 0)
        assert np.all(sig <= 4095)

    def test_signal_integer(self):
        sig, _ = generate_digital_imd_tones(8, 1e3, 100, 100e3, duration=0.001)
        assert sig.dtype == int


# ---- generate_coherent_sine -----------------------------------------------

class TestGenerateCoherentSine:

    def test_length_matches_nfft(self):
        sig, _ = generate_coherent_sine(1e6, 1024, 7)
        assert len(sig) == 1024

    def test_frequency_calculation(self):
        fs, n_fft, n_fin = 1e6, 1024, 7
        _, f_in = generate_coherent_sine(fs, n_fft, n_fin)
        assert pytest.approx(f_in) == n_fin / n_fft * fs

    def test_amplitude_and_offset(self):
        sig, _ = generate_coherent_sine(1e6, 4096, 10, amplitude=2.0, offset=1.0)
        assert pytest.approx(np.max(sig), abs=0.05) == 3.0
        assert pytest.approx(np.min(sig), abs=0.05) == -1.0

    def test_spectral_purity(self):
        """Coherent sine should have energy concentrated at a single bin."""
        fs = 1e6
        n_fft = 1024
        n_fin = 7
        sig, _ = generate_coherent_sine(fs, n_fft, n_fin)
        spectrum = np.abs(np.fft.rfft(sig))
        peak_bin = np.argmax(spectrum)
        assert peak_bin == n_fin


# ---- generate_coherent_two_tone -------------------------------------------

class TestGenerateCoherentTwoTone:

    def test_length_matches_nfft(self):
        sig, _, _ = generate_coherent_two_tone(1e6, 1024, 7, 13)
        assert len(sig) == 1024

    def test_frequency_calculations(self):
        fs, n_fft = 1e6, 1024
        n_fin1, n_fin2 = 7, 13
        _, f1, f2 = generate_coherent_two_tone(fs, n_fft, n_fin1, n_fin2)
        assert pytest.approx(f1) == n_fin1 / n_fft * fs
        assert pytest.approx(f2) == n_fin2 / n_fft * fs

    def test_spectral_peaks(self):
        fs, n_fft = 1e6, 1024
        n_fin1, n_fin2 = 7, 13
        sig, _, _ = generate_coherent_two_tone(fs, n_fft, n_fin1, n_fin2)
        spectrum = np.abs(np.fft.rfft(sig))
        # The two largest bins should be at n_fin1 and n_fin2
        top2 = np.argsort(spectrum)[-2:]
        assert set(top2) == {n_fin1, n_fin2}

    def test_phase_control(self):
        fs, n_fft = 1e6, 1024
        sig1, _, _ = generate_coherent_two_tone(fs, n_fft, 7, 13, phase1=0.0, phase2=0.0)
        sig2, _, _ = generate_coherent_two_tone(fs, n_fft, 7, 13, phase1=np.pi, phase2=0.0)
        # Signals should differ
        assert not np.allclose(sig1, sig2)

    def test_amplitude_control(self):
        fs, n_fft = 1e6, 1024
        sig, _, _ = generate_coherent_two_tone(fs, n_fft, 7, 13,
                                               amplitude1=1.0, amplitude2=0.0)
        spectrum = np.abs(np.fft.rfft(sig))
        # Only bin 7 should have significant energy
        assert spectrum[7] > 100 * spectrum[13]


# ---- convert_to_differential -----------------------------------------------

class TestConvertToDifferential:

    def test_default_vcm(self):
        sig = np.array([1.0, -1.0, 0.5])
        v_pos, v_neg = convert_to_differential(sig)
        np.testing.assert_allclose(v_pos, [0.5, -0.5, 0.25])
        np.testing.assert_allclose(v_neg, [-0.5, 0.5, -0.25])

    def test_custom_vcm(self):
        sig = np.array([2.0, 0.0])
        v_pos, v_neg = convert_to_differential(sig, vcm=1.0)
        np.testing.assert_allclose(v_pos, [2.0, 1.0])
        np.testing.assert_allclose(v_neg, [0.0, 1.0])

    def test_differential_relationship(self):
        sig = np.random.randn(100)
        vcm = 0.5
        v_pos, v_neg = convert_to_differential(sig, vcm=vcm)
        # v_pos - v_neg should equal original signal
        np.testing.assert_allclose(v_pos - v_neg, sig, atol=1e-12)
        # Common mode should be vcm
        np.testing.assert_allclose((v_pos + v_neg) / 2, vcm, atol=1e-12)

    def test_zero_signal(self):
        sig = np.zeros(10)
        v_pos, v_neg = convert_to_differential(sig, vcm=2.0)
        np.testing.assert_allclose(v_pos, 2.0)
        np.testing.assert_allclose(v_neg, 2.0)


# ---- generate_chirp --------------------------------------------------------

def test_chirp_length():
    fs = 1e6
    n_samples = int(1e6 * 0.01)
    sig, t = generate_chirp(fs=fs, n_samples=n_samples, f_start=1e3, f_stop=100e3)
    assert len(sig) == n_samples

def test_chirp_amplitude():
    amp = 0.4
    fs = 1e6
    n_samples = int(1e6 * 0.005)
    sig, t = generate_chirp(fs=fs, n_samples=n_samples, f_start=1e3, f_stop=50e3, amplitude=amp)
    assert abs(np.max(np.abs(sig)) - amp) < 0.01

def test_chirp_offset():
    offset = 0.5
    fs = 1e6
    n_samples = int(1e6 * 0.005)
    sig, t = generate_chirp(fs=fs, n_samples=n_samples, f_start=1e3, f_stop=50e3, amplitude=0.1, offset=offset)
    assert abs(np.mean(sig) - offset) < 0.05

def test_chirp_default_amplitude():
    fs = 1e6
    n_samples = int(1e6 * 0.001)
    sig, t = generate_chirp(fs=fs, n_samples=n_samples, f_start=1e3, f_stop=10e3)
    assert abs(np.max(np.abs(sig)) - 1.0) < 0.01

def test_chirp_start_end_freqs():
    fs = 1e6
    duration = 0.01
    f_start, f_stop = 1e3, 100e3
    n_samples = int(fs * duration)
    sig, t = generate_chirp(fs=fs, n_samples=n_samples, f_start=f_start, f_stop=f_stop)
    n_total = len(sig)
    n_slice = n_total // 10  # 10% of signal

    # Dominant frequency in the first 10% of the signal
    freqs = np.fft.rfftfreq(n_slice, d=1.0 / fs)
    start_spectrum = np.abs(np.fft.rfft(sig[:n_slice]))
    f_dom_start = freqs[np.argmax(start_spectrum)]

    # Dominant frequency in the last 10% of the signal
    end_spectrum = np.abs(np.fft.rfft(sig[-n_slice:]))
    f_dom_end = freqs[np.argmax(end_spectrum)]

    midpoint = (f_start + f_stop) / 2
    assert f_dom_start < midpoint, f"Start freq {f_dom_start} should be below midpoint {midpoint}"
    assert f_dom_end > midpoint, f"End freq {f_dom_end} should be above midpoint {midpoint}"

def test_chirp_log_method():
    fs = 1e6
    n_samples = int(1e6 * 0.01)
    sig, t = generate_chirp(fs=fs, n_samples=n_samples, f_start=1e3, f_stop=100e3, method='logarithmic')
    assert len(sig) == n_samples


# ---- generate_prbs ---------------------------------------------------------

from pyDataconverter.utils.signal_gen import generate_prbs, apply_channel


def test_prbs_length():
    """Output has exactly n_samples samples."""
    sig = generate_prbs(order=7, n_samples=1000)
    assert len(sig) == 1000


def test_prbs_binary_values():
    """Default PRBS has only +amplitude and -amplitude values."""
    amp = 0.5
    sig = generate_prbs(order=7, n_samples=500, amplitude=amp)
    unique = np.unique(sig)
    assert set(unique).issubset({-amp, amp})


def test_prbs_with_offset():
    """Offset shifts all values."""
    sig = generate_prbs(order=7, n_samples=500, amplitude=0.5, offset=1.0)
    assert np.all(sig >= 0.4)
    assert np.all(sig <= 1.6)


def test_prbs_flat_spectrum():
    """PRBS has a roughly flat power spectrum (no dominant tone)."""
    sig = generate_prbs(order=10, n_samples=2**10 - 1, amplitude=1.0)
    fft_mag = np.abs(np.fft.rfft(sig - np.mean(sig)))
    # No single bin should dominate (no tone > 5x the mean)
    assert np.max(fft_mag[1:]) < 5 * np.mean(fft_mag[1:])


def test_prbs_reproducible_with_seed():
    """Same seed gives the same sequence."""
    s1 = generate_prbs(order=7, n_samples=200, seed=42)
    s2 = generate_prbs(order=7, n_samples=200, seed=42)
    np.testing.assert_array_equal(s1, s2)


# ---- apply_channel ---------------------------------------------------------


def test_apply_channel_length():
    """Output length matches input length (same-length convolution)."""
    sig = generate_prbs(order=7, n_samples=512)
    h = np.array([1.0, -0.5, 0.25])
    out = apply_channel(sig, h)
    assert len(out) == len(sig)


def test_apply_channel_identity():
    """Convolving with [1] returns the original signal."""
    sig = generate_prbs(order=7, n_samples=200)
    out = apply_channel(sig, np.array([1.0]))
    np.testing.assert_allclose(out, sig)


def test_apply_channel_lowpass():
    """A lowpass FIR reduces high-frequency content."""
    sig, _ = generate_chirp(fs=1e6, n_samples=int(1e6 * 0.01), f_start=1e3, f_stop=100e3)
    # Simple 3-tap moving average = lowpass
    h = np.ones(3) / 3.0
    out = apply_channel(sig, h)
    assert len(out) == len(sig)
    # High-frequency portion should have lower variance after filtering
    assert np.std(out[-100:]) <= np.std(sig[-100:]) + 0.01


# ---- generate_gaussian_noise -----------------------------------------------

from pyDataconverter.utils.signal_gen import generate_gaussian_noise, apply_window


def test_gaussian_noise_length():
    sig = generate_gaussian_noise(n_samples=1000)
    assert len(sig) == 1000


def test_gaussian_noise_statistics():
    """Zero mean, unit std by default."""
    rng = np.random.default_rng(0)
    sig = generate_gaussian_noise(n_samples=10000, std=1.0, rng=rng)
    assert abs(np.mean(sig)) < 0.05
    assert abs(np.std(sig) - 1.0) < 0.05


def test_gaussian_noise_std_param():
    rng = np.random.default_rng(1)
    std = 0.3
    sig = generate_gaussian_noise(n_samples=5000, std=std, rng=rng)
    assert abs(np.std(sig) - std) < 0.02


def test_gaussian_noise_offset():
    rng = np.random.default_rng(2)
    sig = generate_gaussian_noise(n_samples=5000, std=0.1, offset=2.5, rng=rng)
    assert abs(np.mean(sig) - 2.5) < 0.02


def test_gaussian_noise_reproducible():
    s1 = generate_gaussian_noise(100, std=1.0, rng=np.random.default_rng(7))
    s2 = generate_gaussian_noise(100, std=1.0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(s1, s2)


# ---- apply_window ----------------------------------------------------------


def test_apply_window_length():
    sig = np.ones(256)
    out = apply_window(sig, 'hann')
    assert len(out) == 256


def test_apply_window_hann_ends_zero():
    """Hann window tapers to zero at both ends."""
    sig = np.ones(256)
    out = apply_window(sig, 'hann')
    assert abs(out[0]) < 0.01
    assert abs(out[-1]) < 0.01


def test_apply_window_invalid_raises():
    with pytest.raises(ValueError, match="Unknown window"):
        apply_window(np.ones(64), 'not_a_window')


def test_apply_window_blackman():
    sig = np.ones(128)
    out = apply_window(sig, 'blackman')
    assert len(out) == 128
    assert abs(out[0]) < 0.01
