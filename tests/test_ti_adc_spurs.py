"""
Analytical mismatch spur tests for TimeInterleavedADC.

Each test constructs a TI-ADC with exactly one mismatch type active,
drives a coherent sine through it, runs an FFT on the output, and
asserts that specific tones appear at specific frequencies with
magnitudes matching closed-form formulas (Razavi, Murmann, Kester,
Gustavsson).

FFT normalisation used throughout: spec_norm = |FFT(codes)| / (n_fft / 2),
so spec_norm equals the amplitude of a coherent tone in CODE units (not in
fractional / V/V units). The closed-form spur amplitudes are therefore also
expressed in code units by dividing voltage amplitudes by lsb = v_ref / 2^n_bits.

The exact spur amplitude for a fixed mismatch array {d_k} is set by the
DFT coefficient DFT1 = sum_k d_k * exp(-j*2*pi*k/M):

  offset    -> spur at fs/M,        amplitude = (2/M) * |DFT1_off| / lsb
  gain      -> image at fs/M-f_in,  amplitude = |DFT1_g| * A_in / (M * lsb)
  skew      -> image at fs/M-f_in,  amplitude = |DFT1_tau| * 2*pi*f_in*A_in / (M * lsb)
  bandwidth -> image at fs/M-f_in,  grows with f_in (exact value depends on LPF order)

where A_in = amplitude in VOLTS of the input sine.

Tolerances:
  - offset / gain / timing-skew: +/- 2 dB
  - bandwidth: +/- 3 dB (first-order LPF approximation is itself approximate)

See docs/superpowers/specs/2026-04-14-ti-adc-design.md Appendix A for
the derivation of each formula.
"""

import unittest
import numpy as np

from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType


def _coherent_sine_params():
    """Standard coherent sine configuration used across spur tests."""
    return {
        "M": 4,
        "n_bits": 12,
        "v_ref": 1.0,
        "fs": 1e9,
        "n_fft": 2 ** 14,  # 16384 samples
        "n_fin_bins": 511,  # coprime to n_fft so bins fall cleanly
    }


def _run_coherent_sine(ti, amplitude: float, n_fft: int, n_fin_bins: int, fs: float,
                      use_waveform: bool, offset: float = 0.5):
    """Drive ti with a coherent sine, return (freqs, mag_db, codes).

    The sine is centred at ``offset`` (default v_ref/2 = 0.5) so that a
    unipolar SimpleADC sees every sample in its valid input range. The
    DC offset falls out of the FFT analysis (we subtract the mean before
    computing the spectrum), so the spur-magnitude formulas depend on
    amplitude only, not on offset.
    """
    t = np.arange(n_fft) / fs
    f_in = n_fin_bins * fs / n_fft
    v = offset + amplitude * np.sin(2 * np.pi * f_in * t)
    if use_waveform:
        codes = ti.convert_waveform(v, t)
    else:
        dvdt = 2 * np.pi * f_in * amplitude * np.cos(2 * np.pi * f_in * t)
        codes = np.array([int(ti.convert(float(v[i]), dvdt=float(dvdt[i])))
                          for i in range(n_fft)], dtype=int)
    # Spectrum
    codes_float = codes.astype(float) - np.mean(codes)
    spec = np.abs(np.fft.rfft(codes_float))
    # Normalise so spec_norm == amplitude in CODE units for a coherent tone
    spec_norm = spec / (n_fft / 2)
    mag_db = 20 * np.log10(spec_norm + 1e-30)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    return freqs, mag_db, codes


def _nearest_bin_db(freqs: np.ndarray, mag_db: np.ndarray, f_target: float) -> float:
    idx = int(np.argmin(np.abs(freqs - f_target)))
    return float(mag_db[idx])


def _dft1_magnitude(values: np.ndarray) -> float:
    """
    Magnitude of the first DFT coefficient of a length-M mismatch array.

    DFT1 = sum_{k=0}^{M-1} values[k] * exp(-j * 2*pi*k / M).

    This is the relevant mixing coefficient for TI-ADC mismatch spur theory:
    a mismatch pattern {d_k} creates an image whose amplitude is proportional
    to |DFT1({d_k})|, not to the statistical standard deviation.
    """
    M = len(values)
    return float(abs(sum(values[k] * np.exp(-1j * 2 * np.pi * k / M)
                         for k in range(M))))


class TestTIADCSpurs(unittest.TestCase):

    def test_offset_spur_at_fs_over_M(self):
        """Offset mismatch -> DC-independent tone at k*fs/M."""
        p = _coherent_sine_params()
        M = p["M"]
        lsb = p["v_ref"] / (2 ** p["n_bits"])
        # Explicit offset pattern with known DFT1. Using an explicit array
        # keeps the test reproducible and avoids RNG-dependent tolerances.
        offsets = np.array([1e-3, -1e-3, 0.5e-3, -0.5e-3])
        dft1_off = _dft1_magnitude(offsets)
        template = SimpleADC(n_bits=p["n_bits"], v_ref=p["v_ref"],
                              input_type=InputType.SINGLE)
        ti = TimeInterleavedADC(channels=M, sub_adc_template=template,
                                fs=p["fs"], offset=offsets)
        _, mag_db, _ = _run_coherent_sine(ti, amplitude=0.4,
                                            n_fft=p["n_fft"],
                                            n_fin_bins=p["n_fin_bins"],
                                            fs=p["fs"], use_waveform=False)
        freqs = np.fft.rfftfreq(p["n_fft"], 1.0 / p["fs"])
        f_spur = p["fs"] / M  # primary spur at fs/M
        spur_db = _nearest_bin_db(freqs, mag_db, f_spur)
        # Closed-form (code-domain): spur amplitude = (2/M) * |DFT1_off| / lsb
        expected_db = 20 * np.log10((2 / M) * dft1_off / lsb)
        self.assertAlmostEqual(spur_db, expected_db, delta=2.0,
            msg=f"offset spur at fs/M: expected ~{expected_db:.2f} dB, got {spur_db:.2f} dB")

    def test_gain_spur_at_fs_over_M_minus_fin(self):
        """Gain mismatch -> image at fs/M - f_in scaling with amplitude, NOT with f_in."""
        p = _coherent_sine_params()
        M = p["M"]
        lsb = p["v_ref"] / (2 ** p["n_bits"])
        gains = np.array([1e-3, -1e-3, 5e-4, -5e-4])
        dft1_g = _dft1_magnitude(gains)
        amplitude = 0.4
        template = SimpleADC(n_bits=p["n_bits"], v_ref=p["v_ref"],
                              input_type=InputType.SINGLE)
        ti = TimeInterleavedADC(channels=M, sub_adc_template=template,
                                fs=p["fs"], gain_error=gains)
        _, mag_db, _ = _run_coherent_sine(ti, amplitude=amplitude,
                                            n_fft=p["n_fft"],
                                            n_fin_bins=p["n_fin_bins"],
                                            fs=p["fs"], use_waveform=False)
        freqs = np.fft.rfftfreq(p["n_fft"], 1.0 / p["fs"])
        f_in = p["n_fin_bins"] * p["fs"] / p["n_fft"]
        f_image = p["fs"] / M - f_in
        spur_db = _nearest_bin_db(freqs, mag_db, f_image)
        # Closed-form (code-domain): image amplitude = |DFT1_g| * amplitude / (M * lsb)
        expected_db = 20 * np.log10(dft1_g * amplitude / (M * lsb))
        self.assertAlmostEqual(spur_db, expected_db, delta=2.0,
            msg=f"gain spur at fs/M - f_in: expected ~{expected_db:.2f} dB, got {spur_db:.2f} dB")

    def test_timing_skew_spur_scales_linearly_with_fin(self):
        """Timing-skew image scales linearly with f_in (distinguishes from gain)."""
        p = _coherent_sine_params()
        M = p["M"]
        lsb = p["v_ref"] / (2 ** p["n_bits"])
        skews = np.array([5e-13, -5e-13, 2e-13, -2e-13])  # 500 fs, 200 fs
        dft1_tau = _dft1_magnitude(skews)
        amplitude = 0.4
        template = SimpleADC(n_bits=p["n_bits"], v_ref=p["v_ref"],
                              input_type=InputType.SINGLE)

        # Run at two different input frequencies; the magnitude should scale linearly.
        # Use prime-ish bin numbers to keep coherence.
        spur_db_by_fin = {}
        for n_fin_bins in (257, 1021):
            ti = TimeInterleavedADC(channels=M, sub_adc_template=template,
                                    fs=p["fs"], timing_skew=skews)
            _, mag_db, _ = _run_coherent_sine(ti, amplitude=amplitude,
                                                n_fft=p["n_fft"],
                                                n_fin_bins=n_fin_bins,
                                                fs=p["fs"], use_waveform=False)
            freqs = np.fft.rfftfreq(p["n_fft"], 1.0 / p["fs"])
            f_in = n_fin_bins * p["fs"] / p["n_fft"]
            f_image = p["fs"] / M - f_in
            spur_db = _nearest_bin_db(freqs, mag_db, f_image)
            spur_db_by_fin[f_in] = spur_db

            # Closed-form (code-domain): image amplitude = |DFT1_tau| * 2*pi*f_in*A / (M*lsb)
            expected_db = 20 * np.log10(
                dft1_tau * 2 * np.pi * f_in * amplitude / (M * lsb))
            self.assertAlmostEqual(spur_db, expected_db, delta=2.0,
                msg=f"skew spur at f_in={f_in:.2e}: expected ~{expected_db:.2f} dB, got {spur_db:.2f} dB")

        # Verify linear scaling with f_in: a higher f_in should give a proportionally
        # higher spur. The difference in dB should match 20*log10(f_in_high / f_in_low).
        f_low, f_high = sorted(spur_db_by_fin.keys())
        measured_slope = spur_db_by_fin[f_high] - spur_db_by_fin[f_low]
        expected_slope = 20 * np.log10(f_high / f_low)
        self.assertAlmostEqual(measured_slope, expected_slope, delta=1.5,
            msg=f"skew spur slope: expected {expected_slope:.2f} dB, got {measured_slope:.2f} dB")

    def test_bandwidth_image_grows_with_fin(self):
        """Bandwidth mismatch image grows with f_in (requires convert_waveform)."""
        p = _coherent_sine_params()
        M = p["M"]
        # Mix of cutoffs spanning an octave; fs=1 GHz so Nyquist is 500 MHz.
        bandwidths = np.array([2e8, 3e8, 1.5e8, 2.5e8])
        template = SimpleADC(n_bits=p["n_bits"], v_ref=p["v_ref"],
                              input_type=InputType.SINGLE)

        spur_db_by_fin = {}
        for n_fin_bins in (257, 2049):
            ti = TimeInterleavedADC(channels=M, sub_adc_template=template,
                                    fs=p["fs"], bandwidth=bandwidths)
            _, mag_db, _ = _run_coherent_sine(ti, amplitude=0.4,
                                                n_fft=p["n_fft"],
                                                n_fin_bins=n_fin_bins,
                                                fs=p["fs"], use_waveform=True)
            freqs = np.fft.rfftfreq(p["n_fft"], 1.0 / p["fs"])
            f_in = n_fin_bins * p["fs"] / p["n_fft"]
            f_image = p["fs"] / M - f_in
            spur_db_by_fin[f_in] = _nearest_bin_db(freqs, mag_db, f_image)

        # The higher f_in should produce a LARGER spur (magnitude grows with
        # input frequency) than the lower f_in. Tolerance is generous because
        # the first-order LPF + FFT leakage make the exact magnitude hard to
        # pin down without extensive averaging.
        f_low, f_high = sorted(spur_db_by_fin.keys())
        self.assertGreater(
            spur_db_by_fin[f_high], spur_db_by_fin[f_low] + 3.0,
            msg=f"bandwidth spur should grow with f_in: low={spur_db_by_fin[f_low]:.2f}, "
                f"high={spur_db_by_fin[f_high]:.2f}")
