"""
ADC Performance Metrics
=======================

Functions for calculating ADC static and dynamic performance metrics.
"""

import numpy as np
from typing import Dict, List, Optional
from ..fft_analysis import compute_fft
from ...dataconverter import QuantizationMode
from ._shared import _calculate_dynamic_metrics


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
    transitions = _calculate_code_edges(input_voltages, output_codes, n_bits)
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
