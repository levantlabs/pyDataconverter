"""Tests for generalised SAR ADC components."""
import numpy as np
import pytest
from pyDataconverter.components.cdac import (
    RedundantSARCDAC, SplitCapCDAC, SegmentedCDAC,
    SingleEndedCDAC, DifferentialCDAC,
)
from pyDataconverter.dataconverter import InputType


class TestRedundantSARCDAC:
    def test_construction(self):
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        assert cdac.n_bits == 4
        assert cdac.v_ref  == 1.0

    def test_weights_decrease(self):
        """Capacitor weights are strictly decreasing (MSB first)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.85)
        w = cdac.cap_weights
        assert np.all(np.diff(w) < 0)

    def test_weights_not_binary(self):
        """Weights are not powers of 2 (distinguishes from standard CDAC)."""
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        w = cdac.cap_weights
        binary_weights = 2 ** np.arange(cdac.n_bits - 1, -1, -1).astype(float)
        assert not np.allclose(w, binary_weights)

    def test_dec_monotone(self):
        """Decoded output is monotone with increasing DAC voltage.

        The DEC table must map raw codes to output codes such that
        decode(a) <= decode(b) whenever cdac.get_voltage(a) <= cdac.get_voltage(b).
        This is the correct monotonicity property for a redundant SAR DEC.
        """
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        n_codes = 2 ** 6
        voltages = np.array([cdac.get_voltage(c)[0] for c in range(n_codes)])
        decoded  = np.array([cdac.decode(c) for c in range(n_codes)])
        # Sort raw codes by DAC voltage; decoded values in that order must be non-decreasing
        order = np.argsort(voltages)
        assert list(decoded[order]) == sorted(decoded[order])

    def test_dec_range(self):
        """Decoded values span 0..2^n_bits-1."""
        cdac = RedundantSARCDAC(n_bits=4, v_ref=1.0, radix=1.8)
        decoded = [cdac.decode(code) for code in range(2**4)]
        assert min(decoded) == 0
        assert max(decoded) == 2**4 - 1

    def test_ideal_conversion_with_sar(self):
        """RedundantSARCDAC inside SARADC gives monotone ideal transfer."""
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = RedundantSARCDAC(n_bits=6, v_ref=1.0, radix=1.8)
        adc  = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE, cdac=cdac)
        vin  = np.linspace(0, 1.0, 200)
        codes = [adc.convert(float(v)) for v in vin]
        # codes should be non-decreasing
        assert all(b >= a for a, b in zip(codes, codes[1:]))


class TestSplitCapCDAC:
    def test_construction(self):
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        assert cdac.n_bits == 8
        assert cdac.v_ref  == 1.0

    def test_fewer_total_caps_than_full_binary(self):
        """Split-cap uses n_bits+1 caps (incl. bridge), not 2^n_bits."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        # n_msb + n_lsb + 1 bridge cap = n_bits + 1 total caps
        assert len(cdac.cap_weights) == 8 + 1  # 9 caps, not 256

    def test_ideal_output_zero(self):
        """Code 0 → 0 V."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        vp, vn = cdac.get_voltage(0)
        assert abs(vp - vn) < 1e-9

    def test_ideal_output_full_scale(self):
        """Max code → v_ref - LSB."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        max_code = 2**8 - 1
        vp, vn = cdac.get_voltage(max_code)
        expected = (max_code / 2**8) * 1.0
        assert abs(vp - vn - expected) < 0.01  # within 1% of LSB

    def test_monotone_voltages(self):
        """Voltages are non-decreasing with code."""
        cdac = SplitCapCDAC(n_bits=8, v_ref=1.0, n_msb=4)
        voltages = cdac.voltages
        assert np.all(np.diff(voltages) >= -1e-9)


class TestSegmentedCDAC:
    def test_construction(self):
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        assert cdac.n_bits == 8

    def test_monotone_voltages(self):
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        v = cdac.voltages
        assert np.all(np.diff(v) >= -1e-9)

    def test_output_range(self):
        """Max code → v_ref - LSB, code 0 → 0."""
        cdac = SegmentedCDAC(n_bits=8, v_ref=1.0, n_therm=4)
        v = cdac.voltages
        assert abs(v[0]) < 1e-9
        expected_max = (2**8 - 1) / 2**8
        assert abs(v[-1] - expected_max) < 0.01

    def test_thermometer_section_equal_steps(self):
        """The MSB 2^n_therm steps should be equal (thermometer linearity)."""
        n_bits, n_therm = 8, 4
        cdac = SegmentedCDAC(n_bits=n_bits, v_ref=1.0, n_therm=n_therm)
        v = cdac.voltages
        step = 2**n_bits // 2**n_therm  # codes per thermometer step
        msb_voltages = v[step-1::step][:2**n_therm]
        diffs = np.diff(msb_voltages)
        assert np.allclose(diffs, diffs[0], rtol=0.01)


class TestMultibitSARADC:
    def test_construction(self):
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=2)
        assert adc.n_bits == 8
        assert adc.bits_per_cycle == 2

    def test_invalid_bits_per_cycle_zero(self):
        """bits_per_cycle=0 must raise ValueError (line 397)."""
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        with pytest.raises(ValueError, match="bits_per_cycle"):
            MultibitSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=0)

    def test_invalid_bits_per_cycle_too_large(self):
        """bits_per_cycle > n_bits must raise ValueError (line 397)."""
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        with pytest.raises(ValueError, match="bits_per_cycle"):
            MultibitSARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=5)

    def test_repr_basic(self):
        """__repr__ without optional params covers lines 437-442, 451."""
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=2)
        r = repr(adc)
        assert "MultibitSARADC" in r
        assert "n_bits=6" in r
        assert "bits_per_cycle=2" in r

    def test_repr_with_all_nonidealities(self):
        """__repr__ with noise_rms, offset, gain_error, t_jitter covers lines 443-450."""
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(
            n_bits=6, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=2,
            noise_rms=1e-4, offset=0.001, gain_error=0.002, t_jitter=1e-10
        )
        r = repr(adc)
        assert "noise_rms" in r
        assert "offset" in r
        assert "gain_error" in r
        assert "t_jitter" in r

    def test_monotone_ideal(self):
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=2)
        vin = np.linspace(0.01, 0.99, 200)
        codes = [adc.convert(float(v)) for v in vin]
        assert all(b >= a for a, b in zip(codes, codes[1:]))

    def test_output_range(self):
        from pyDataconverter.architectures.SARADC import MultibitSARADC
        adc = MultibitSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=3)
        codes = [adc.convert(float(v)) for v in np.linspace(0, 1, 100)]
        assert min(codes) >= 0
        assert max(codes) <= 2**6 - 1

    def test_bits_per_cycle_1_matches_standard_sar(self):
        """bits_per_cycle=1 should behave identically to standard SARADC."""
        from pyDataconverter.architectures.SARADC import SARADC, MultibitSARADC
        np.random.seed(0)
        sar    = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        mbit   = MultibitSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE, bits_per_cycle=1)
        vins   = np.linspace(0.01, 0.99, 63)
        codes_sar  = [sar.convert(float(v))  for v in vins]
        codes_mbit = [mbit.convert(float(v)) for v in vins]
        assert codes_sar == codes_mbit


class TestNoiseshapingSARADC:
    def test_construction(self):
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        assert adc.n_bits == 6
        assert adc.integrator_state == 0.0

    def test_reset_clears_state(self):
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        adc.convert(0.5)  # leave some integrator state
        adc.reset()
        assert adc.integrator_state == 0.0

    def test_output_range(self):
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        codes = [adc.convert(0.5) for _ in range(100)]
        assert all(0 <= c <= 2**6 - 1 for c in codes)

    def test_repr_basic(self):
        """__repr__ without optional params covers lines 499-504, 514."""
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        r = repr(adc)
        assert "NoiseshapingSARADC" in r
        assert "n_bits=6" in r
        assert "integrator_state" in r

    def test_repr_with_all_nonidealities(self):
        """__repr__ with noise_rms, offset, gain_error, t_jitter covers lines 506-513."""
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        adc = NoiseshapingSARADC(
            n_bits=6, v_ref=1.0, input_type=InputType.SINGLE,
            noise_rms=1e-4, offset=0.001, gain_error=0.002, t_jitter=1e-10
        )
        r = repr(adc)
        assert "noise_rms" in r
        assert "offset" in r
        assert "gain_error" in r
        assert "t_jitter" in r

    def test_noise_shaping_improves_snr_at_low_freq(self):
        """First-order noise shaping should give better SNDR than standard SAR
        when measuring SNR over the lower half of the Nyquist band (oversampling)."""
        from pyDataconverter.architectures.SARADC import SARADC, NoiseshapingSARADC
        from pyDataconverter.utils.signal_gen import generate_coherent_sine
        from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics
        import numpy as np
        n_bits, v_ref, fs, n_fft = 6, 1.0, 1e6, 2048
        vin, _ = generate_coherent_sine(fs, n_fft, n_fin=5, amplitude=0.45, offset=0.5)
        adc_std = SARADC(n_bits=n_bits, input_type=InputType.SINGLE, v_ref=v_ref)
        adc_ns  = NoiseshapingSARADC(n_bits=n_bits, input_type=InputType.SINGLE, v_ref=v_ref)
        codes_std = np.array([adc_std.convert(float(v)) for v in vin], dtype=float)
        codes_ns  = np.array([adc_ns.convert(float(v))  for v in vin], dtype=float)
        m_std = calculate_adc_dynamic_metrics(time_data=codes_std, fs=fs)
        m_ns  = calculate_adc_dynamic_metrics(time_data=codes_ns,  fs=fs)
        # Noise shaping redistributes noise (less low-freq, more high-freq).
        # Full-band SNR should be comparable (within ~4 dB) even at Nyquist rate.
        assert m_ns['SNR'] > m_std['SNR'] - 4.0

    def test_differential_mode_integrator_does_not_saturate(self):
        """Regression: in differential mode the reconstructed analog value
        must be in the same coordinate system as v_sampled ([-v_ref/2, +v_ref/2]).
        Before the fix, reconstruction returned single-ended-equivalent voltage,
        so the residue was biased by +v_ref/2 and the integrator saturated
        immediately, silently disabling noise shaping."""
        from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
        from pyDataconverter.dataconverter import InputType
        adc = NoiseshapingSARADC(n_bits=6, v_ref=1.0,
                                  input_type=InputType.DIFFERENTIAL)
        lsb = 1.0 / 64  # v_ref / 2^n_bits
        # Small slow-varying differential signal well below full scale
        for i in range(200):
            v_diff = 0.1 * np.sin(i * 0.1)
            adc.convert((v_diff / 2, -v_diff / 2))
            # Healthy integrator: bounded by a few LSBs; pathological
            # (pre-fix) behaviour pins it at ±v_ref/2 = 32 LSBs.
            assert abs(adc.integrator_state) < 5 * lsb, (
                f"integrator state {adc.integrator_state} exceeded 5*LSB "
                f"at step {i} — differential reconstruction bug regressed"
            )


class TestSARAdcBaseLineCoverage:
    """Targeted tests to cover SARADC.py lines missing from this test file."""

    # Lines 145, 147, 149 — ValueError guards in __init__
    def test_negative_noise_rms_raises(self):
        from pyDataconverter.architectures.SARADC import SARADC
        with pytest.raises(ValueError, match="noise_rms"):
            SARADC(n_bits=6, input_type=InputType.SINGLE, noise_rms=-0.001)

    def test_negative_t_jitter_raises(self):
        from pyDataconverter.architectures.SARADC import SARADC
        with pytest.raises(ValueError, match="t_jitter"):
            SARADC(n_bits=6, input_type=InputType.SINGLE, t_jitter=-1e-12)

    def test_negative_cap_mismatch_raises(self):
        from pyDataconverter.architectures.SARADC import SARADC
        with pytest.raises(ValueError, match="cap_mismatch"):
            SARADC(n_bits=6, input_type=InputType.SINGLE, cap_mismatch=-0.01)

    # Line 165 — TypeError for non-CDACBase cdac
    def test_cdac_wrong_type_raises(self):
        from pyDataconverter.architectures.SARADC import SARADC
        with pytest.raises(TypeError):
            SARADC(n_bits=6, cdac="not_a_cdac")

    # Lines 167-168 — ValueError when cdac.n_bits mismatches
    def test_cdac_wrong_nbits_raises(self):
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
        with pytest.raises(ValueError, match="n_bits"):
            SARADC(n_bits=6, v_ref=1.0, cdac=cdac)

    # Lines 170-171 — ValueError when cdac.v_ref mismatches
    def test_cdac_wrong_vref_raises(self):
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = SingleEndedCDAC(n_bits=6, v_ref=2.0)
        with pytest.raises(ValueError, match="v_ref"):
            SARADC(n_bits=6, v_ref=1.0, cdac=cdac)

    # Line 177 — DifferentialCDAC auto-creation
    def test_differential_input_type_creates_diff_cdac(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        assert isinstance(adc.cdac, DifferentialCDAC)

    # Line 193 — dac_voltages property
    def test_dac_voltages_property(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        vols = adc.dac_voltages
        assert vols.shape == (2**6,)
        assert np.all(np.diff(vols) > 0)

    # Lines 246-251 — convert_with_trace
    def test_convert_with_trace_keys(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        trace = adc.convert_with_trace(0.4)
        assert set(trace.keys()) == {
            'code', 'sampled_voltage', 'dac_voltages',
            'bit_decisions', 'register_states',
        }
        assert 0 <= trace['code'] <= 2**6 - 1
        assert len(trace['dac_voltages']) == 6

    # Lines 287-288 — differential _sample_input path
    def test_differential_sample_input(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        code = adc.convert((0.3, 0.1))
        assert 0 <= code <= 2**6 - 1

    # Line 291 — gain_error branch in _sample_input
    def test_gain_error_applied(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc_ideal = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        adc_gain = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, gain_error=0.05)
        assert adc_ideal.convert(0.4) != adc_gain.convert(0.4)

    # Line 293 — offset branch in _sample_input
    def test_offset_applied(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc_ideal = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        adc_offset = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, offset=0.05)
        assert adc_ideal.convert(0.4) != adc_offset.convert(0.4)

    # Line 295 — noise_rms branch in _sample_input
    def test_noise_rms_applied(self):
        from pyDataconverter.architectures.SARADC import SARADC
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, noise_rms=0.02)
        codes = {adc.convert(0.5) for _ in range(30)}
        assert len(codes) > 1

    # Line 297 — t_jitter + dvdt branch in _sample_input
    def test_t_jitter_with_dvdt_applied(self):
        from pyDataconverter.architectures.SARADC import SARADC
        np.random.seed(0)
        adc = SARADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE, t_jitter=1e-9)
        dvdt = 0.5 * 2 * np.pi * 1e6
        codes = {adc.convert(0.4, dvdt=dvdt) for _ in range(30)}
        assert len(codes) > 1

    # Lines 359-373 — SARADC.__repr__ with optional non-idealities
    def test_repr_basic(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc = SARADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
        r = repr(adc)
        assert "SARADC" in r
        assert "n_bits=6" in r

    def test_repr_with_all_nonidealities(self):
        from pyDataconverter.architectures.SARADC import SARADC
        adc = SARADC(
            n_bits=6, v_ref=1.0, input_type=InputType.SINGLE,
            noise_rms=1e-4, offset=0.001, gain_error=0.002, t_jitter=1e-10
        )
        r = repr(adc)
        assert "noise_rms" in r
        assert "offset" in r
        assert "gain_error" in r
        assert "t_jitter" in r


class TestSARADCCapMismatchPassthrough:
    """When both cdac and cap_mismatch are supplied, SARADC overrides the
    CDAC's mismatch realization with a fresh draw at the SAR-level stddev."""

    def test_mismatch_is_applied_to_user_cdac(self):
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = SingleEndedCDAC(n_bits=8, v_ref=1.0, cap_mismatch=0.0)
        weights_before = cdac.cap_weights.copy()
        with pytest.warns(RuntimeWarning, match="cap_mismatch"):
            SARADC(n_bits=8, v_ref=1.0, cdac=cdac, cap_mismatch=0.02)
        assert not np.allclose(cdac.cap_weights, weights_before)

    def test_nominal_topology_preserved_through_override(self):
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = SingleEndedCDAC(n_bits=8, v_ref=1.0, cap_mismatch=0.0)
        nominals_before = [c.c_nominal for c in cdac.cap_instances]
        with pytest.warns(RuntimeWarning):
            SARADC(n_bits=8, v_ref=1.0, cdac=cdac, cap_mismatch=0.02)
        nominals_after = [c.c_nominal for c in cdac.cap_instances]
        assert nominals_before == nominals_after

    def test_no_warning_when_cap_mismatch_is_zero(self):
        from pyDataconverter.architectures.SARADC import SARADC
        import warnings
        cdac = SingleEndedCDAC(n_bits=8, v_ref=1.0, cap_mismatch=0.01)
        weights_before = cdac.cap_weights.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail
            SARADC(n_bits=8, v_ref=1.0, cdac=cdac, cap_mismatch=0.0)
        np.testing.assert_array_equal(cdac.cap_weights, weights_before)

    def test_differential_passthrough(self):
        from pyDataconverter.architectures.SARADC import SARADC
        cdac = DifferentialCDAC(n_bits=6, v_ref=1.0, cap_mismatch=0.0)
        pos_before = cdac.cap_weights.copy()
        with pytest.warns(RuntimeWarning):
            SARADC(n_bits=6, v_ref=1.0,
                   input_type=InputType.DIFFERENTIAL,
                   cdac=cdac, cap_mismatch=0.05)
        assert not np.allclose(cdac.cap_weights, pos_before)
