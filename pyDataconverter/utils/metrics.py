"""
Converter Performance Metrics
==============================

Functions for calculating ADC and DAC performance metrics.
Shared spectral metrics live in _calculate_dynamic_metrics; the public
calculate_adc_dynamic_metrics and calculate_dac_dynamic_metrics wrappers
call it and can add converter-specific metrics on top.
"""

import numpy as np
from typing import Dict, List, Optional
from .fft_analysis import compute_fft, find_harmonics, find_fundamental
from pyDataconverter.dataconverter import QuantizationMode


def _calculate_dynamic_metrics(freqs: np.ndarray,
                                mags: np.ndarray,
                                fs: float,
                                f0: float,
                                full_scale: float,
                                time_data: np.ndarray) -> Dict[str, float]:
    """
    Core FFT-based dynamic metrics shared by ADC and DAC calculations.

    Computes SNR, SNDR, SFDR, THD, ENOB, noise floor, and optional dBFS
    variants from a pre-computed FFT spectrum.

    Args:
        freqs: Frequency array in Hz.
        mags: Magnitude array in dB.
        fs: Sample/update rate in Hz.
        f0: Fundamental frequency in Hz (or None to auto-detect).
        full_scale: Full-scale value for dBFS conversion (or None for plain dB).
        time_data: Original time-domain data, used only for DC offset when provided.

    Returns:
        Dictionary of metrics.
    """
    bin_width = freqs[1] - freqs[0]

    fund_freq, fund_mag = find_fundamental(freqs, mags, f0, fs)
    harmonics = find_harmonics(freqs, mags, fund_freq, fs, num_harmonics=7)

    harmonic_pwr = sum(10 ** (h[1] / 10) for h in harmonics)
    fund_pwr = 10 ** (fund_mag / 10)
    thd = 10 * np.log10(max(harmonic_pwr, 1e-20) / fund_pwr)

    mask = np.abs(freqs - fund_freq) > bin_width
    max_spur = np.max(mags[mask])
    sfdr = fund_mag - max_spur

    exclude_freqs = np.array([fund_freq] + [h[0] for h in harmonics])
    mask = ~np.any(np.abs(freqs[np.newaxis, :] - exclude_freqs[:, np.newaxis]) <= bin_width, axis=0)

    noise_pwr = sum(10 ** (m / 10) for m in mags[mask])
    noise_floor = noise_pwr / (fs / 2)

    noise_pwr = max(float(noise_pwr), 1e-20)
    snr = 10 * np.log10(fund_pwr / noise_pwr)

    total_noise_and_dist_pwr = max(float(noise_pwr + harmonic_pwr), 1e-20)
    sndr = 10 * np.log10(fund_pwr / total_noise_and_dist_pwr)

    enob = (sndr - 1.76) / 6.02

    if time_data is not None:
        offset = np.mean(time_data)
    else:
        offset = 10 ** (mags[0] / 20)

    results = {
        "SNR": snr,
        "SNDR": sndr,
        "SFDR": sfdr,
        "THD": thd,
        "NoiseFloor": noise_floor,
        "ENOB": enob,
        "Offset": offset,
        "FundamentalFrequency": fund_freq,
        "FundamentalMagnitude": fund_mag,
        "HarmonicFreqs": [h[0] for h in harmonics],
        "HarmonicMags": [h[1] for h in harmonics],
    }

    if full_scale is not None:
        if time_data is not None:
            N = len(time_data)
            level_correction = 20 * np.log10(full_scale / 2) + 20 * np.log10(N / 2)
        else:
            level_correction = 0  # mags assumed already in dBFS

        fund_mag_dBFS = fund_mag - level_correction
        results["FundamentalMagnitude_dBFS"] = fund_mag_dBFS
        results["HarmonicMags_dBFS"] = [m - level_correction for m in results["HarmonicMags"]]
        results["SFDR_dBFS"]  = sfdr  - fund_mag_dBFS
        results["SNR_dBFS"]   = snr   - fund_mag_dBFS
        results["SNDR_dBFS"]  = sndr  - fund_mag_dBFS
        results["THD_dBFS"]   = thd   - fund_mag_dBFS

    return results


def calculate_adc_dynamic_metrics(time_data: np.ndarray = None,
                                   fs: float = None,
                                   f0: float = None,
                                   freqs: np.ndarray = None,
                                   mags: np.ndarray = None,
                                   full_scale: float = None) -> Dict[str, float]:
    """
    Calculate dynamic ADC metrics from either time-domain data or FFT data.

    Args:
        time_data: Input signal array (optional if freqs/mags provided).
        fs: Sampling frequency in Hz (required if time_data provided).
        f0: Fundamental frequency (if known).
        freqs: Frequency array from FFT (optional if time_data provided).
        mags: Magnitude array from FFT (optional if time_data provided).
        full_scale: Full-scale value for dBFS conversion. If None, results are in dB.

    Returns:
        Dictionary of metrics.

    Raises:
        ValueError: If neither time_data nor (freqs, mags) are provided.
    """
    if time_data is None and (freqs is None or mags is None):
        raise ValueError("Must provide either time_data or freqs/mags")

    if freqs is None or mags is None:
        freqs, mags = compute_fft(time_data, fs)

    return _calculate_dynamic_metrics(freqs, mags, fs, f0, full_scale, time_data)


def calculate_dac_dynamic_metrics(freqs: np.ndarray = None,
                                   mags: np.ndarray = None,
                                   fs: float = None,
                                   f0: float = None,
                                   full_scale: float = None) -> Dict[str, float]:
    """
    Calculate dynamic DAC metrics from FFT data.

    Args:
        freqs: Frequency array from FFT.
        mags: Magnitude array from FFT.
        fs: DAC update rate in Hz.
        f0: Fundamental frequency (if known).
        full_scale: Full-scale voltage for dBFS conversion. If None, results are in dB.

    Returns:
        Dictionary of metrics.

    Raises:
        ValueError: If freqs or mags are not provided.
    """
    if freqs is None or mags is None:
        raise ValueError("Must provide freqs and mags")

    return _calculate_dynamic_metrics(freqs, mags, fs, f0, full_scale, time_data=None)


def _calculate_code_edges(input_voltages: np.ndarray,
                          output_codes: np.ndarray,
                          n_bits: int) -> np.ndarray:
    """
    Internal function to calculate ADC transition levels from ramp input data.

    Args:
        input_voltages: Array of input voltages from ramp
        output_codes: Array of output codes from ADC
        n_bits: ADC resolution

    Returns:
        Array of transition voltages

    Notes:
        Assumes monotonic ramp input and sorted data
        This is an internal function used by calculate_static_metrics
    """
    n_codes = 2 ** n_bits - 1  # Number of transitions is 2^N - 1
    transitions = []

    # Find transitions (when code changes)
    code_changes = np.where(np.diff(output_codes))[0]

    # Calculate transition voltage as average between adjacent points
    for idx in code_changes:
        transitions.append((input_voltages[idx] + input_voltages[idx + 1]) / 2)

    # If we missed any transitions (due to missing codes), fill with interpolated values
    while len(transitions) < n_codes:
        transitions.append(transitions[-1])

    return np.array(transitions)

def calculate_adc_static_metrics(input_voltages: np.ndarray,
                             output_codes: np.ndarray,
                             n_bits: int,
                             v_ref: float = 1.0,
                             quant_mode: QuantizationMode = QuantizationMode.FLOOR) -> Dict[str, float]:
    """
    Calculate static ADC metrics from ramp test data.

    Args:
        input_voltages: Array of input voltages from ramp
        output_codes: Array of output codes from ADC
        n_bits: ADC resolution
        v_ref: Reference voltage
        quant_mode: Quantization mode (FLOOR or SYMMETRIC). Determines the
                    ideal LSB size and first/last transition positions used
                    for offset, gain error, and INL calculation.

    Returns:
        Dictionary of metrics including DNL, INL, offset, gain error

    Notes:
        Assumes input_voltages is a monotonic ramp
        Assumes output_codes are sorted
        INL is computed directly from transition positions vs. ideal (not
        via cumsum of DNL) so that missing codes do not accumulate error.
    """
    # Calculate transition levels
    transitions = _calculate_code_edges(input_voltages, output_codes, n_bits)

    if quant_mode == QuantizationMode.FLOOR:
        ideal_lsb    = v_ref / (2 ** n_bits)
        ideal_first  = ideal_lsb / 2
        ideal_last   = v_ref - ideal_lsb / 2
        # Ideal transition k is at (k + 0.5) * ideal_lsb (FLOOR convention)
        ideal_transitions = (np.arange(len(transitions)) + 0.5) * ideal_lsb
    else:  # SYMMETRIC
        ideal_lsb    = v_ref / (2 ** n_bits - 1)
        ideal_first  = ideal_lsb / 2
        ideal_last   = v_ref - ideal_lsb / 2
        ideal_transitions = (np.arange(len(transitions)) + 0.5) * ideal_lsb

    # Calculate DNL from actual bin widths
    actual_widths = np.diff(transitions)
    dnl = actual_widths / ideal_lsb - 1

    # Calculate INL directly from transition positions vs. ideal.
    # Using cumsum(DNL) causes missing-code -1 LSB entries to accumulate,
    # producing a monotonically drifting INL that does not reflect linearity.
    inl = (transitions[:-1] - ideal_transitions[:-1]) / ideal_lsb

    # Calculate offset (difference from ideal first transition)
    offset = transitions[0] - ideal_first

    # Calculate gain error
    gain_error = ((transitions[-1] - transitions[0]) -
                  (ideal_last - ideal_first)) / (ideal_last - ideal_first)

    return {
        "DNL": dnl,
        "INL": inl,
        "Offset": offset,
        "GainError": gain_error,
        "MaxDNL": np.max(np.abs(dnl)),
        "MaxINL": np.max(np.abs(inl)),
        "Transitions": transitions
    }


def is_monotonic(input_voltages: np.ndarray,
                 output_codes: np.ndarray,
                 n_bits: int) -> bool:
    """
    Check if ADC transfer function is monotonic.

    Args:
        input_voltages: Array of input voltages from ramp
        output_codes: Array of output codes from ADC
        n_bits: ADC resolution

    Returns:
        bool: True if ADC is monotonic

    Notes:
        ADC is monotonic if code transitions are strictly increasing
        (each transition voltage is higher than the previous one)
    """
    # Get transition levels
    transitions = _calculate_code_edges(input_voltages, output_codes, n_bits)

    # Check if transitions are strictly increasing
    return np.all(np.diff(transitions) > 0)


def calculate_histogram(codes: np.ndarray,
                        n_bits: int,
                        input_type: str = 'uniform',  # 'uniform' or 'sine'
                        normalize: bool = True,
                        remove_pdf: bool = True) -> Dict[str, np.ndarray]:
    """
    Calculate histogram of ADC output codes.

    Args:
        codes: Array of ADC output codes
        n_bits: ADC resolution
        input_type: Type of input signal ('uniform' or 'sine')
        normalize: If True, normalize histogram to sum to 1
        remove_pdf: If True and input_type is 'sine', compensate for sine wave PDF

    Returns:
        Dictionary containing:
            - bin_counts: Number of hits per code (compensated if using sine PDF)
            - bin_edges: Code values (0 to 2^n_bits - 1)
            - missing_codes: List of codes with zero hits
            - unused_range: Percentage of unused codes

    Notes:
        For sine wave input, the probability density is:
        P(x) = 1/(π√(A² - x²)) where A is amplitude
    """
    if input_type not in ['uniform', 'sine']:
        raise ValueError("input_type must be either 'uniform' or 'sine'")

    n_codes = 2 ** n_bits

    # Calculate raw histogram
    bin_counts, bin_edges = np.histogram(codes,
                                         bins=n_codes,
                                         range=(0, n_codes))

    # Remove sine wave PDF if requested
    if input_type == 'sine' and remove_pdf:
        # Convert code numbers to normalized amplitude (-1 to 1)
        code_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        normalized_amp = (code_centers - n_codes / 2) / (n_codes / 2)

        # Calculate ideal sine wave PDF (avoiding division by zero)
        valid_mask = np.abs(normalized_amp) < 0.999  # Avoid edges
        pdf = np.zeros_like(normalized_amp)
        pdf[valid_mask] = 1 / (np.pi * np.sqrt(1 - normalized_amp[valid_mask] ** 2))

        # Remove PDF from histogram where counts exist and PDF is valid
        nonzero_mask = bin_counts > 0
        valid_pdf = (pdf > 0) & nonzero_mask
        bin_counts[valid_pdf] = bin_counts[valid_pdf] / pdf[valid_pdf]

    # Normalize if requested
    if normalize:
        nonzero_mask = bin_counts > 0
        bin_counts[nonzero_mask] = bin_counts[nonzero_mask] / np.sum(bin_counts[nonzero_mask])

    # Find missing codes
    missing_codes = np.where(bin_counts == 0)[0]

    # Calculate percentage of unused range
    used_codes = np.where(bin_counts > 0)[0]
    if len(used_codes) > 0:
        code_range = used_codes[-1] - used_codes[0] + 1
        unused_range = 100 * (1 - code_range / n_codes)
    else:
        unused_range = 100.0

    return {
        "bin_counts": bin_counts,
        "bin_edges": bin_edges[:-1],
        "missing_codes": missing_codes,
        "unused_range": unused_range
    }


