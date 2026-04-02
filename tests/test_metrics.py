"""
Test code for ADC metrics calculations
"""

import warnings
import numpy as np
import pytest
from pyDataconverter.utils.signal_gen import generate_sine
from pyDataconverter.utils.metrics import (
    calculate_adc_dynamic_metrics,
    calculate_adc_static_metrics,
    calculate_adc_static_metrics_histogram,
    is_monotonic,
    calculate_histogram,
    calculate_gain_offset_error,
)
from pyDataconverter.dataconverter import QuantizationMode


def test_gain_offset_error_ideal():
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    transitions = np.arange(1, 2**n_bits) * ideal_lsb
    result = calculate_gain_offset_error(transitions, n_bits, v_ref)
    assert abs(result['OffsetError']) < 1e-12
    assert abs(result['GainError']) < 1e-12

def test_gain_offset_error_with_offset():
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    shift = 0.01
    transitions = np.arange(1, 2**n_bits) * ideal_lsb + shift
    result = calculate_gain_offset_error(transitions, n_bits, v_ref)
    assert abs(result['OffsetError'] - shift) < 1e-12

def test_gain_offset_error_with_gain():
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    gain = 0.05
    ideal = np.arange(1, 2**n_bits) * ideal_lsb
    stretched = ideal[0] + (ideal - ideal[0]) * (1 + gain)
    result = calculate_gain_offset_error(stretched, n_bits, v_ref)
    assert abs(result['GainError'] - gain) < 1e-6

def test_static_metrics_uses_gain_offset_helper():
    import numpy as np
    from pyDataconverter.architectures.FlashADC import FlashADC
    adc = FlashADC(n_bits=6, v_ref=1.0, offset_std=0.002)
    vin = np.linspace(0, 1.0, 10000)
    codes = np.array([adc.convert(float(v)) for v in vin])
    m_static = calculate_adc_static_metrics(vin, codes, 6, 1.0)
    m_go = calculate_gain_offset_error(m_static['Transitions'], 6, 1.0)
    assert abs(m_static['Offset'] - m_go['OffsetError']) < 1e-12
    assert abs(m_static['GainError'] - m_go['GainError']) < 1e-12


# ---------------------------------------------------------------------------
# dBFS metrics tests
# ---------------------------------------------------------------------------

_RATIO_KEYS = ['SNR', 'SNDR', 'SFDR', 'THD']
_DBFS_KEYS  = [k + '_dBFS' for k in _RATIO_KEYS]


def _make_clean_signal():
    fs, NFFT, NFIN = 1e6, 1024, 11
    f0 = NFIN * fs / NFFT
    signal = generate_sine(f0, fs, amplitude=0.9, duration=NFFT / fs)
    return signal, fs, f0


def test_dbfs_keys_absent_without_full_scale():
    """dBFS variant keys must not be present when full_scale is not given."""
    signal, fs, f0 = _make_clean_signal()
    result = calculate_adc_dynamic_metrics(signal, fs, f0)
    for k in _DBFS_KEYS:
        assert k not in result, f"Key {k!r} should not be present without full_scale"


def test_dbfs_keys_present_with_full_scale():
    """dBFS variant keys must appear when full_scale is given."""
    signal, fs, f0 = _make_clean_signal()
    result = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=1.0)
    for k in _DBFS_KEYS:
        assert k in result, f"Key {k!r} missing from metrics dict"


def test_original_db_keys_still_present_with_full_scale():
    """Original dB ratio keys must be preserved when full_scale is given."""
    signal, fs, f0 = _make_clean_signal()
    result = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=1.0)
    for k in _RATIO_KEYS:
        assert k in result, f"Original key {k!r} missing when full_scale is set"


def test_dbfs_values_correct_formulas():
    """
    Verify the dBFS variants use the correct formulas relative to fund_mag_dBFS.

    - SFDR_dBFS = SFDR - fund_mag_dBFS   (spur distance from full scale)
    - SNR/SNDR/THD_dBFS = metric - fund_mag_dBFS  (fund_mag_dBFS is negative, so result > metric)

    For the time_data path, fund_mag_dBFS = fund_mag - 20*log10(full_scale/2) - 20*log10(N/2).
    The full_scale/2 factor accounts for a sine wave having peak amplitude = full_scale/2.
    """
    signal, fs, f0 = _make_clean_signal()
    NFFT = 1024
    full_scale = 2.0
    result = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=full_scale)

    level_correction = 20 * np.log10(full_scale / 2) + 20 * np.log10(NFFT / 2)
    fund_mag_dBFS = result["FundamentalMagnitude"] - level_correction

    assert abs(result["FundamentalMagnitude_dBFS"] - fund_mag_dBFS) < 1e-9

    # SFDR: spur below full scale
    assert abs(result["SFDR_dBFS"] - (result["SFDR"] - fund_mag_dBFS)) < 1e-9

    # SNR / SNDR / THD: fund_mag_dBFS is negative, subtracting it increases the value
    for k in ['SNR', 'SNDR', 'THD']:
        expected = result[k] - fund_mag_dBFS
        actual   = result[k + '_dBFS']
        assert abs(actual - expected) < 1e-9, (
            f"{k}_dBFS ({actual:.4f}) != {k} - fund_mag_dBFS ({expected:.4f})"
        )


def test_enob_unchanged_by_full_scale():
    """ENOB is derived from SNDR (a ratio) and must not change with full_scale."""
    signal, fs, f0 = _make_clean_signal()
    r_no_fs = calculate_adc_dynamic_metrics(signal, fs, f0)
    r_fs    = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=4.0)
    assert abs(r_no_fs['ENOB'] - r_fs['ENOB']) < 1e-9, "ENOB should be scale-independent"


# ---------------------------------------------------------------------------
# plot_fft metrics annotation tests
# ---------------------------------------------------------------------------

def test_plot_fft_metrics_annotation_dBFS(tmp_path):
    """plot_fft must display dBFS labels when SNR_dBFS is present in metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyDataconverter.utils.visualizations.fft_plots import plot_fft
    from pyDataconverter.utils.fft_analysis import compute_fft

    signal, fs, f0 = _make_clean_signal()
    freqs, mags = compute_fft(signal, fs)
    metrics = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=1.0)

    fig, ax = plt.subplots()
    plot_fft(freqs, mags, metrics=metrics, fig=fig, ax=ax)

    # Check annotation text contains 'dBFS'
    texts = [t.get_text() for t in ax.texts]
    assert any('dBFS' in t for t in texts), "Annotation should contain 'dBFS' label"
    plt.close(fig)


def test_plot_fft_metrics_annotation_db(tmp_path):
    """plot_fft must display dB labels when no dBFS keys are present."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyDataconverter.utils.visualizations.fft_plots import plot_fft
    from pyDataconverter.utils.fft_analysis import compute_fft

    signal, fs, f0 = _make_clean_signal()
    freqs, mags = compute_fft(signal, fs)
    metrics = calculate_adc_dynamic_metrics(signal, fs, f0)  # no full_scale

    fig, ax = plt.subplots()
    plot_fft(freqs, mags, metrics=metrics, fig=fig, ax=ax)

    texts = [t.get_text() for t in ax.texts]
    assert any('SNR' in t for t in texts), "Annotation should contain SNR"
    assert not any('dBFS' in t for t in texts), "Annotation should not contain 'dBFS' without full_scale"
    plt.close(fig)


def test_dynamic_metrics_raises_on_empty_input():
    """calculate_adc_dynamic_metrics raises on empty input."""
    with pytest.raises((ValueError, IndexError)):
        calculate_adc_dynamic_metrics(np.array([]), fs=1e6, f0=1e3)


def test_dynamic_metrics_all_zero_input():
    """calculate_adc_dynamic_metrics on all-zero input produces finite results (no -inf)."""
    signal = np.zeros(1024)
    # All-zero after DC removal is all zeros; should not produce -inf
    result = calculate_adc_dynamic_metrics(signal, fs=1e6, f0=1e3)
    assert np.isfinite(result["SNR"]), "SNR should be finite for all-zero input"
    assert np.isfinite(result["SNDR"]), "SNDR should be finite for all-zero input"


def test_static_metrics_skipped_codes():
    """calculate_adc_static_metrics handles ADC that skips codes."""
    n_bits = 3
    n_points = 1000
    input_voltages = np.linspace(0, 1, n_points)
    # Create codes that skip code 3 (goes directly from 2 to 4)
    ideal_codes = np.floor(input_voltages * 2**n_bits).astype(int)
    ideal_codes = np.clip(ideal_codes, 0, 2**n_bits - 1)
    # Replace code 3 with code 4
    output_codes = ideal_codes.copy()
    output_codes[output_codes == 3] = 4

    metrics = calculate_adc_static_metrics(input_voltages, output_codes, n_bits)
    # Should still return valid metrics
    assert 'DNL' in metrics
    assert 'INL' in metrics
    # MaxDNL should be > 0 due to skipped code
    assert metrics['MaxDNL'] > 0


def test_adc_static_metrics_array_lengths():
    """DNL has 2^N entries (one per code bin); INL has 2^N-1 (one per transition)."""
    n_bits = 4
    n_points = 2000
    input_voltages = np.linspace(0, 1, n_points)
    ideal_codes = np.floor(input_voltages * 2**n_bits).astype(int)
    ideal_codes = np.clip(ideal_codes, 0, 2**n_bits - 1)
    metrics = calculate_adc_static_metrics(input_voltages, ideal_codes, n_bits)
    assert len(metrics['DNL']) == 2**n_bits,   f"DNL length {len(metrics['DNL'])} != {2**n_bits}"
    assert len(metrics['INL']) == 2**n_bits - 1, f"INL length {len(metrics['INL'])} != {2**n_bits - 1}"


def test_adc_static_metrics_ideal_dnl_inl():
    """Ideal ADC (fine ramp) should have near-zero DNL and INL for all methods."""
    n_bits = 4
    n_points = 5000
    input_voltages = np.linspace(0, 1, n_points)
    ideal_codes = np.floor(input_voltages * 2**n_bits).astype(int)
    ideal_codes = np.clip(ideal_codes, 0, 2**n_bits - 1)
    for method in ('endpoint', 'best_fit', 'absolute'):
        metrics = calculate_adc_static_metrics(input_voltages, ideal_codes, n_bits,
                                               inl_method=method)
        assert metrics['MaxDNL'] < 0.05, f"[{method}] MaxDNL={metrics['MaxDNL']:.4f}"
        assert metrics['MaxINL'] < 0.05, f"[{method}] MaxINL={metrics['MaxINL']:.4f}"


def test_adc_static_metrics_inl_method_invalid():
    """Unknown inl_method should raise ValueError."""
    input_voltages = np.linspace(0, 1, 100)
    codes = np.arange(100) % 8
    with pytest.raises(ValueError, match="inl_method"):
        calculate_adc_static_metrics(input_voltages, codes, 3, inl_method='bad')


# ---------------------------------------------------------------------------
# calculate_adc_static_metrics_histogram tests
# ---------------------------------------------------------------------------

def _make_histogram_codes(n_bits=6, n_samples=200_000, amplitude_frac=1.0,
                           seed=42):
    """Generate codes from an ideal ADC driven by a sine wave."""
    rng = np.random.default_rng(seed)
    n_codes = 2 ** n_bits
    v_ref = 1.0
    amplitude = amplitude_frac * v_ref / 2
    offset    = v_ref / 2
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, 100 * np.pi, n_samples)
    vin = offset + amplitude * np.sin(t + phase)
    ideal_lsb = v_ref / n_codes
    codes = np.clip(np.floor(vin / ideal_lsb).astype(int), 0, n_codes - 1)
    return codes, n_bits, v_ref, amplitude


def test_histogram_metrics_return_keys():
    """Result must contain DNL, INL, MaxDNL, MaxINL and no other keys."""
    codes, n_bits, v_ref, amplitude = _make_histogram_codes()
    result = calculate_adc_static_metrics_histogram(codes, n_bits, v_ref, amplitude)
    assert set(result.keys()) == {"DNL", "INL", "MaxDNL", "MaxINL"}


def test_histogram_metrics_array_lengths():
    """DNL has 2^N entries; INL has 2^N-1 entries."""
    codes, n_bits, v_ref, amplitude = _make_histogram_codes()
    result = calculate_adc_static_metrics_histogram(codes, n_bits, v_ref, amplitude)
    assert len(result["DNL"]) == 2 ** n_bits
    assert len(result["INL"]) == 2 ** n_bits - 1


def test_histogram_metrics_ideal_adc_small_dnl_inl():
    """Ideal ADC with many samples should have small interior DNL and small INL.

    Edge codes (0 and 2^N-1) are excluded from the interior DNL check because
    the PDF singularity makes their estimates inherently less reliable.
    """
    codes, n_bits, v_ref, amplitude = _make_histogram_codes(n_bits=6,
                                                             n_samples=500_000)
    result = calculate_adc_static_metrics_histogram(codes, n_bits, v_ref, amplitude)
    interior_dnl = result["DNL"][1:-1]
    assert np.max(np.abs(interior_dnl)) < 0.10, \
        f"Interior MaxDNL={np.max(np.abs(interior_dnl)):.3f}"
    assert result["MaxINL"] < 0.15, f"MaxINL={result['MaxINL']:.3f}"


def test_histogram_metrics_endpoint_inl_zero_at_boundaries():
    """Endpoint INL must be zero at first and last transitions."""
    codes, n_bits, v_ref, amplitude = _make_histogram_codes()
    result = calculate_adc_static_metrics_histogram(codes, n_bits, v_ref,
                                                    amplitude,
                                                    inl_method='endpoint')
    assert abs(result["INL"][0])  < 1e-9
    assert abs(result["INL"][-1]) < 1e-9


def test_histogram_metrics_amplitude_warning():
    """Amplitude below 90 % of full scale must trigger a UserWarning."""
    codes, n_bits, v_ref, _ = _make_histogram_codes(amplitude_frac=0.5)
    low_amplitude = 0.5 * v_ref / 2   # 50 % of full scale — well below threshold
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_adc_static_metrics_histogram(codes, n_bits, v_ref, low_amplitude)
    assert any(issubclass(w.category, UserWarning) for w in caught), \
        "Expected a UserWarning for low amplitude"


def test_histogram_metrics_no_warning_at_full_scale():
    """Full-scale amplitude must not trigger any warning."""
    codes, n_bits, v_ref, amplitude = _make_histogram_codes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_adc_static_metrics_histogram(codes, n_bits, v_ref, amplitude)
    assert not any(issubclass(w.category, UserWarning) for w in caught), \
        "Unexpected UserWarning at full-scale amplitude"


def test_histogram_metrics_invalid_inl_method():
    """Unknown inl_method must raise ValueError."""
    codes, n_bits, v_ref, amplitude = _make_histogram_codes()
    with pytest.raises(ValueError):
        calculate_adc_static_metrics_histogram(codes, n_bits, v_ref, amplitude,
                                               inl_method='bad')


def test_histogram_metrics_invalid_amplitude():
    """amplitude <= 0 must raise ValueError."""
    codes, n_bits, v_ref, _ = _make_histogram_codes()
    with pytest.raises(ValueError):
        calculate_adc_static_metrics_histogram(codes, n_bits, v_ref,
                                               amplitude=-0.1)


def test_is_monotonic_with_skipped_codes():
    """is_monotonic returns False when a code is completely skipped.

    _calculate_code_edges fills missing transitions with duplicates,
    so np.diff(transitions) contains zeros, making is_monotonic False.
    """
    n_bits = 3
    n_points = 1000
    input_voltages = np.linspace(0, 1, n_points)
    # Create codes that skip code 3 entirely (0,1,2,4,5,6,7)
    codes = np.floor(input_voltages * 2**n_bits).astype(int)
    codes = np.clip(codes, 0, 2**n_bits - 1)
    # Replace code 3 with code 4 to simulate a skipped code
    codes[codes == 3] = 4

    result = is_monotonic(input_voltages, codes, n_bits)
    assert result == False, "Skipped code should cause non-monotonicity in transitions"


def test_histogram_all_same_code():
    """calculate_histogram with all-same-code input."""
    n_bits = 4
    codes = np.full(1000, 7)  # all code 7
    hist = calculate_histogram(codes, n_bits, input_type='uniform')
    # Only one code should have non-zero count
    nonzero_bins = np.sum(hist['bin_counts'] > 0)
    assert nonzero_bins == 1
    assert 7 in np.where(hist['bin_counts'] > 0)[0]


def test_histogram_sine_pdf_removal_near_edges():
    """Histogram PDF removal at normalized amplitude near ±1.0 does not crash or produce inf."""
    n_bits = 4
    n_points = 10000
    # Generate sine wave that hits edge codes
    t = np.linspace(0, 10 * np.pi, n_points)
    sine_wave = np.sin(t)
    sine_codes = np.round((sine_wave + 1) * (2**(n_bits - 1) - 0.5)).astype(int)
    sine_codes = np.clip(sine_codes, 0, 2**n_bits - 1)

    hist = calculate_histogram(sine_codes, n_bits, input_type='sine', remove_pdf=True)
    # Should not contain inf or nan
    assert np.all(np.isfinite(hist['bin_counts'])), "PDF-compensated histogram should have finite counts"


def test_plot_fft_show_metrics_false():
    """No annotation should appear when show_metrics=False."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyDataconverter.utils.visualizations.fft_plots import plot_fft
    from pyDataconverter.utils.fft_analysis import compute_fft

    signal, fs, f0 = _make_clean_signal()
    freqs, mags = compute_fft(signal, fs)
    metrics = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=1.0)

    fig, ax = plt.subplots()
    plot_fft(freqs, mags, metrics=metrics, show_metrics=False, fig=fig, ax=ax)
    assert len(ax.texts) == 0, "No annotation should be rendered when show_metrics=False"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Original integration tests (unchanged)
# ---------------------------------------------------------------------------

def test_dynamic_metrics():
    """Test dynamic metrics with known input signals"""

    # Test parameters
    fs = 1e6  # 1 MHz sampling
    NFFT = 1024  # FFT length
    duration = NFFT / fs

    # Generate a clean sine wave
    f0 = 11 * fs / NFFT  # Coherent sampling
    signal = generate_sine(f0, fs, amplitude=0.9, duration=duration)

    # Add some harmonics
    signal += 0.001 * generate_sine(2 * f0, fs, duration=duration)  # 2nd harmonic
    signal += 0.005 * generate_sine(3 * f0, fs, duration=duration)  # 3rd harmonic

    print("\nTest 1: Basic sine wave with harmonics")
    print("----------------------------------------")
    metrics = calculate_adc_dynamic_metrics(signal, fs, f0)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"{key}: {[f'{v:.2f}' for v in value]}")

    print("\nTest 2: Same signal with dBFS normalization")
    print("-------------------------------------------")
    metrics_dbfs = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=1.0)
    for key, value in metrics_dbfs.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"{key}: {[f'{v:.2f}' for v in value]}")


def test_static_metrics():
    """Test static metrics with ramp input"""
    print("\nTest 3: Static ADC metrics")
    print("-------------------------")

    # Create test data for a 3-bit ADC
    n_bits = 3
    n_points = 1000

    # Generate ideal ramp input
    input_voltages = np.linspace(0, 1, n_points)


    # Generate ideal output codes (0 to 7 for 3-bit ADC)
    ideal_codes = np.floor(input_voltages * (2 ** n_bits - 0.1)).astype(int)
    ideal_codes = np.clip(ideal_codes, 0, 2 ** n_bits - 1)
    output_codes = ideal_codes.copy()

    # Add some non-linearity
    transition_noise = 0.1 * ideal_codes + np.random.normal(0, 0.1, len(ideal_codes))
    output_codes = np.clip(ideal_codes + transition_noise, 0, 2 ** n_bits - 1).astype(int)

    # Calculate metrics
    metrics = calculate_adc_static_metrics(input_voltages, output_codes, n_bits)

    # Print results
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
        elif isinstance(value, np.ndarray):
            if len(value) < 10:  # Only print full array if it's small
                print(f"{key}: {[f'{v:.3f}' for v in value]}")
            else:
                print(f"{key}: Array of length {len(value)}")

    print("Test with monotonic ADC:")
    print(f"Is monotonic: {is_monotonic(input_voltages, ideal_codes, n_bits)}")
    print(f"Is monotonic: {is_monotonic(input_voltages, output_codes, n_bits)}")


def test_histogram():
    """Test histogram calculation with different input types"""
    print("\nTest Histogram Calculation")
    print("-------------------------")

    # Test parameters
    n_bits = 4
    n_points = 10000

    # Test 1: Uniform distribution
    print("\nTest with uniform distribution:")
    uniform_codes = np.random.randint(0, 2 ** n_bits, n_points)
    hist_uniform = calculate_histogram(uniform_codes, n_bits, input_type='uniform')

    # Test 2: Sine wave
    print("\nTest with sine wave input:")
    t = np.linspace(0, 10 * np.pi, n_points)
    sine_wave = np.sin(t)
    sine_codes = np.round((sine_wave + 1) * (2 ** (n_bits - 1) - 0.5)).astype(int)
    sine_codes = np.clip(sine_codes, 0, 2 ** n_bits - 1)

    # Calculate with and without PDF compensation
    hist_sine_raw = calculate_histogram(sine_codes, n_bits,
                                        input_type='sine', remove_pdf=False)
    hist_sine_comp = calculate_histogram(sine_codes, n_bits,
                                         input_type='sine', remove_pdf=True)

    # Plot results
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Uniform histogram
    ax1.bar(hist_uniform['bin_edges'], hist_uniform['bin_counts'],
            width=1.0, alpha=0.7)
    ax1.set_title('Uniform Input')
    ax1.set_xlabel('Code')
    ax1.set_ylabel('Normalized Counts')
    ax1.grid(True)

    # Raw sine histogram
    ax2.bar(hist_sine_raw['bin_edges'], hist_sine_raw['bin_counts'],
            width=1.0, alpha=0.7)
    ax2.set_title('Sine Input (Raw)')
    ax2.set_xlabel('Code')
    ax2.grid(True)

    # Compensated sine histogram
    ax3.bar(hist_sine_comp['bin_edges'], hist_sine_comp['bin_counts'],
            width=1.0, alpha=0.7)
    ax3.set_title('Sine Input (PDF Compensated)')
    ax3.set_xlabel('Code')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


def test_missing_codes_ideal_adc():
    """Ideal ADC has no missing codes."""
    from pyDataconverter.architectures.FlashADC import FlashADC
    import numpy as np
    adc = FlashADC(n_bits=6, v_ref=1.0, offset_std=0.0)
    vin = np.linspace(0, 1.0, 10000)
    codes = np.array([adc.convert(float(v)) for v in vin])
    m = calculate_adc_static_metrics(vin, codes, 6, 1.0)
    assert 'MissingCodes' in m
    assert len(m['MissingCodes']) == 0

def test_missing_codes_detected():
    """DNL <= -1 codes appear in MissingCodes."""
    import numpy as np
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    # Build a ramp where code 3 is missing (transition 2 == transition 3)
    transitions = np.arange(1, 2**n_bits, dtype=float) * ideal_lsb
    transitions[2] = transitions[3]  # code 3 missing
    vin = np.linspace(0, v_ref, 20000)
    codes = np.zeros(len(vin), dtype=int)
    for k, t in enumerate(transitions):
        codes[vin >= t] = k + 1
    m = calculate_adc_static_metrics(vin, codes, n_bits, v_ref)
    assert 'MissingCodes' in m
    assert len(m['MissingCodes']) > 0


def main():
    """Run all tests"""
    test_dynamic_metrics()
    test_static_metrics()
    test_histogram()


if __name__ == "__main__":
    main()