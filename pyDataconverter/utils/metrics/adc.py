"""
ADC Performance Metrics
=======================

Functions for calculating ADC static and dynamic performance metrics.
"""

import warnings
import numpy as np
from typing import Dict, List, Optional
from ..fft_analysis import compute_fft
from ...dataconverter import QuantizationMode
from ._dynamic import _calculate_dynamic_metrics
from ._shared import _fit_reference_line


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
                             quant_mode: QuantizationMode = QuantizationMode.FLOOR,
                             inl_method: str = 'endpoint') -> Dict[str, float]:
    """
    Calculate static ADC metrics from ramp test data.

    Args:
        input_voltages: Array of input voltages from ramp.
        output_codes: Array of output codes from ADC.
        n_bits: ADC resolution.
        v_ref: Reference voltage.
        quant_mode: Quantization mode (FLOOR or SYMMETRIC).
            FLOOR: ideal_lsb = v_ref / 2^N, transitions at integer multiples
            of ideal_lsb.
            SYMMETRIC: ideal_lsb = v_ref / (2^N - 1), transitions at half-
            integer multiples of ideal_lsb (midpoints between output levels).
        inl_method: Method for computing INL ('endpoint', 'best_fit', or
            'absolute').
            'endpoint'  — remove gain and offset by fitting a line through
                          the first and last actual transitions. First and
                          last INL entries are 0 by definition.
            'best_fit'  — least-squares line through all transitions.
                          Minimises RMS INL; first/last entries are not
                          forced to zero.
            'absolute'  — compare each transition directly to the
                          mode-specific ideal position, with no gain or
                          offset correction.

    Returns:
        Dictionary with keys:
            DNL        : np.ndarray, length 2^N. One entry per code bin.
                         Code 0 bin spans [0, T[0]]; code 2^N-1 bin spans
                         [T[-1], v_ref].
            INL        : np.ndarray, length 2^N-1. One entry per transition.
            Offset     : float, deviation of T[0] from ideal (V).
            GainError  : float, fractional gain error.
            MaxDNL     : float, max |DNL| (LSB).
            MaxINL     : float, max |INL| (LSB).
            Transitions: np.ndarray of measured transition voltages.

    Notes:
        Assumes input_voltages is a monotonic ramp from 0 to v_ref.
        Missing codes are represented as duplicate transitions, giving
        DNL = -1 for the missing code and a wider adjacent bin; INL is
        computed directly from transition positions so missing-code -1 LSB
        entries do not accumulate.
    """
    transitions = _calculate_code_edges(input_voltages, output_codes, n_bits)
    k = np.arange(len(transitions))  # 0 .. 2^N-2

    if quant_mode == QuantizationMode.FLOOR:
        ideal_lsb = v_ref / (2 ** n_bits)
        # Transition k is between code k and code k+1, at (k+1)*ideal_lsb
        ideal_transitions = (k + 1) * ideal_lsb
        ideal_first = ideal_lsb            # T[0] ideal
        ideal_last  = (2**n_bits - 1) * ideal_lsb  # T[-1] ideal
    else:  # SYMMETRIC
        ideal_lsb = v_ref / (2 ** n_bits - 1)
        # Output level for code k is k*ideal_lsb; transition at midpoint
        ideal_transitions = (k + 0.5) * ideal_lsb
        ideal_first = 0.5 * ideal_lsb
        ideal_last  = (2**n_bits - 1.5) * ideal_lsb

    # --- DNL ---
    # Include the first (0→T[0]) and last (T[-1]→v_ref) code bins.
    # Previously np.diff(transitions) silently dropped both endpoint codes.
    bin_widths = np.diff(np.concatenate([[0.0], transitions, [v_ref]]))
    dnl = bin_widths / ideal_lsb - 1  # length 2^N

    # --- INL ---
    # Compute directly from transition positions (not cumsum(DNL)) so that
    # missing-code -1 LSB entries do not accumulate into neighbouring codes.
    if inl_method == 'absolute':
        inl = (transitions - ideal_transitions) / ideal_lsb
    elif inl_method in ('endpoint', 'best_fit'):
        line = _fit_reference_line(k.astype(float), transitions, inl_method)
        inl = (transitions - line) / ideal_lsb
    else:
        raise ValueError("inl_method must be 'endpoint', 'best_fit', or 'absolute'")

    # --- Offset and gain error (mode-independent structure) ---
    ideal_span = ideal_last - ideal_first
    offset     = transitions[0] - ideal_first
    gain_error = ((transitions[-1] - transitions[0]) - ideal_span) / ideal_span

    return {
        "DNL": dnl,
        "INL": inl,
        "Offset": offset,
        "GainError": gain_error,
        "MaxDNL": float(np.max(np.abs(dnl))),
        "MaxINL": float(np.max(np.abs(inl))),
        "Transitions": transitions,
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


def calculate_adc_static_metrics_histogram(
        codes: np.ndarray,
        n_bits: int,
        v_ref: float = 1.0,
        amplitude: Optional[float] = None,
        offset: Optional[float] = None,
        inl_method: str = 'endpoint') -> Dict[str, object]:
    """
    Calculate ADC static metrics (DNL, INL) from a sine-wave histogram test.

    Rather than sweeping a ramp and locating transition voltages, this method
    applies a sine wave and measures how many samples fall in each code bin.
    After compensating for the sine-wave probability density function (PDF),
    the normalised bin counts are proportional to bin widths, giving DNL
    directly.  INL is recovered via cumulative summation of DNL and then
    corrected with the chosen reference line.

    Compared to the ramp method:
      - No missing-code problem in the cumsum: every bin receives at least
        some hits under a full-scale sine, so cumulative errors do not arise
        from completely absent codes.
      - Does NOT return Offset, GainError, or Transitions because the method
        measures relative widths only, not absolute transition positions.

    Args:
        codes: 1-D array of ADC output codes recorded while a sine wave was
               applied.  Should contain enough samples (≥ 100 per code bin)
               for reliable statistics.
        n_bits: ADC resolution.
        v_ref: Reference voltage (V).  Defines the full input range [0, v_ref]
               and the ideal LSB = v_ref / 2^n_bits.
        amplitude: Peak amplitude of the sine wave (V).  Defaults to v_ref/2
                   (full-scale single-ended input).  Must be > 0.
        offset: DC offset of the sine wave (V).  Defaults to v_ref/2 (centred
                in the input range).
        inl_method: Reference line used to correct the cumulative INL.
            'endpoint' (default) — line through the first and last
                cumulative-sum values; corrected INL is zero at both ends.
            'best_fit' — least-squares line through all cumulative-sum values;
                minimises RMS INL.

    Returns:
        Dictionary with keys:
            DNL    : np.ndarray, length 2^N.  One entry per code bin.
                     Codes outside the sine amplitude are set to -1
                     (missing-code convention).  The two outermost codes
                     (code 0 and code 2^N-1) have inherently less reliable
                     DNL estimates due to the open-ended nature of the edge
                     bins near the PDF singularity.
            INL    : np.ndarray, length 2^N-1.  One entry per transition,
                     after endpoint/best_fit correction.
            MaxDNL : float, max |DNL| in LSB.
            MaxINL : float, max |INL| in LSB.

    Warns:
        UserWarning: If amplitude < 90 % of full-scale (v_ref/2).  Codes
            outside the sine range will be missing and INL estimates near
            the endpoints will be unreliable.

    Raises:
        ValueError: If inl_method is not 'endpoint' or 'best_fit'.
        ValueError: If amplitude <= 0.
    """
    codes = np.asarray(codes)
    n_codes = 2 ** n_bits
    ideal_lsb = v_ref / n_codes

    if amplitude is None:
        amplitude = v_ref / 2
    if amplitude <= 0:
        raise ValueError("amplitude must be > 0")
    if offset is None:
        offset = v_ref / 2

    # --- Amplitude check ---
    full_scale_amp = v_ref / 2
    if amplitude < 0.9 * full_scale_amp:
        warnings.warn(
            f"Sine amplitude ({amplitude:.4g} V) is less than 90 % of full "
            f"scale ({full_scale_amp:.4g} V). Codes outside the sine range "
            "will be reported as missing (DNL = -1) and INL near the "
            "endpoints will be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # --- Raw histogram (counts per code bin, no PDF compensation) ---
    raw_counts, _ = np.histogram(codes, bins=n_codes, range=(0, n_codes))
    raw_counts = raw_counts.astype(float)

    # --- Sine PDF compensation ---
    # PDF of a sine with amplitude A centred at `offset`, evaluated at the
    # voltage at the centre of each code bin:
    #   P(v) = 1 / (π * sqrt(A² - (v - offset)²))
    # Normalised position u = (v - offset) / A; PDF singularity at |u| = 1.
    code_centers_v = (np.arange(n_codes) + 0.5) * ideal_lsb
    u = (code_centers_v - offset) / amplitude

    in_range = np.abs(u) < 0.999          # exclude near-singular edge bins
    pdf = np.zeros(n_codes)
    pdf[in_range] = 1.0 / (np.pi * np.sqrt(1.0 - u[in_range] ** 2))

    # Compensate counts; codes outside the amplitude or with zero counts → -1
    usable = in_range & (raw_counts > 0)
    comp = np.full(n_codes, np.nan)
    comp[usable] = raw_counts[usable] / pdf[usable]

    # Exclude the outermost codes from the normalisation reference.
    # The PDF diverges near the amplitude limits so the two edge bins are
    # open-ended and systematically over-counted; including them in the mean
    # biases every interior DNL value by ~0.01 LSB.
    interior = np.ones(n_codes, dtype=bool)
    interior[[0, -1]] = False
    norm_mask = in_range & interior & np.isfinite(comp)
    mean_comp = (np.nanmean(comp[norm_mask]) if norm_mask.any()
                 else np.nanmean(comp[in_range]))

    dnl = np.where(np.isfinite(comp), comp / mean_comp - 1.0, -1.0)

    # --- INL via cumulative sum of DNL ---
    # cumsum(DNL)[k] = INL at transition k (between code k and k+1),
    # referenced to the absolute ideal position.  Length 2^N-1 (drop last).
    inl_raw = np.cumsum(dnl)[:-1]

    k = np.arange(len(inl_raw), dtype=float)
    line = _fit_reference_line(k, inl_raw, inl_method)
    inl = inl_raw - line

    return {
        "DNL":    dnl,
        "INL":    inl,
        "MaxDNL": float(np.max(np.abs(dnl))),
        "MaxINL": float(np.max(np.abs(inl))),
    }
