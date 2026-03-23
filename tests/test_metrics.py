"""
Test code for ADC metrics calculations
"""

import numpy as np
import pytest
from pyDataconverter.utils.signal_gen import generate_sine
from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics, calculate_adc_static_metrics, is_monotonic, calculate_histogram


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
    - SNR/SNDR/THD_dBFS = metric + fund_mag_dBFS  (referenced to full scale)

    For the time_data path, fund_mag_dBFS = fund_mag - 20*log10(full_scale) - 20*log10(N/2).
    """
    signal, fs, f0 = _make_clean_signal()
    NFFT = 1024
    full_scale = 2.0
    result = calculate_adc_dynamic_metrics(signal, fs, f0, full_scale=full_scale)

    level_correction = 20 * np.log10(full_scale) + 20 * np.log10(NFFT / 2)
    fund_mag_dBFS = result["FundamentalMagnitude"] - level_correction

    assert abs(result["FundamentalMagnitude_dBFS"] - fund_mag_dBFS) < 1e-9

    # SFDR: spur below full scale
    assert abs(result["SFDR_dBFS"] - (result["SFDR"] - fund_mag_dBFS)) < 1e-9

    # SNR / SNDR / THD: referenced to full scale
    for k in ['SNR', 'SNDR', 'THD']:
        expected = result[k] + fund_mag_dBFS
        actual   = result[k + '_dBFS']
        assert abs(actual - expected) < 1e-9, (
            f"{k}_dBFS ({actual:.4f}) != {k} + fund_mag_dBFS ({expected:.4f})"
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


def main():
    """Run all tests"""
    test_dynamic_metrics()
    test_static_metrics()
    test_histogram()


if __name__ == "__main__":
    main()