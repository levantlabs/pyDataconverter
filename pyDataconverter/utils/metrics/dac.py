"""
DAC Performance Metrics
=======================

Functions for calculating DAC static and dynamic performance metrics.
"""

import numpy as np
from typing import Dict, Optional
from ..fft_analysis import compute_fft, find_fundamental, find_harmonics
from ...dataconverter import DACBase, OutputType
from ._dynamic import _calculate_dynamic_metrics
from ._shared import _fit_reference_line


def calculate_dac_static_metrics(dac: DACBase,
                                 n_points: Optional[int] = None,
                                 inl_method: str = 'endpoint',
                                 ) -> Dict[str, object]:
    """
    Compute static linearity metrics by sweeping codes through a DAC.

    Drives every code (or a subset) through *dac*, records the output
    voltage, and computes DNL, INL, offset, and gain error.

    Parameters
    ----------
    dac : DACBase
        A DAC instance to characterize.  Must expose ``n_bits``, ``v_ref``,
        ``output_type``, and ``convert(code)``.
    n_points : int, optional
        Number of evenly-spaced codes to sweep.  When *None* (default),
        all ``2**n_bits`` codes are used.  Must be >= 2 if provided.
    inl_method : str, optional
        Method for computing INL (default ``'endpoint'``).
        ``'endpoint'`` — fit a line through the first and last measured
            voltages (IEEE 1057).  First and last INL entries are 0 by
            definition.
        ``'best_fit'`` — least-squares line through all measured voltages.
            Minimises RMS INL; first/last entries are not forced to zero.

    Returns
    -------
    dict
        DNL : np.ndarray
            Differential non-linearity per code step, in LSB units.
            Length is ``len(Codes) - 1``.
        INL : np.ndarray
            Integral non-linearity, in LSB units.
            Length equals ``len(Codes)``.
        MaxDNL : float
            Maximum absolute DNL value (LSB).
        MaxINL : float
            Maximum absolute INL value (LSB).
        Offset : float
            DC offset voltage at code 0 (V).
        GainError : float
            Fractional gain error (dimensionless).  0.01 means +1 %.
        Codes : np.ndarray
            The digital codes that were swept.
        Voltages : np.ndarray
            The measured output voltage for each code.

    Raises
    ------
    TypeError
        If *dac* is not a ``DACBase`` instance.
    ValueError
        If *n_points* is less than 2.
    ValueError
        If *inl_method* is not ``'endpoint'`` or ``'best_fit'``.

    Notes
    -----
    - LSB is defined as ``v_ref / (2**n_bits - 1)`` so that the maximum
      code maps exactly to ``v_ref``.
    - For differential DACs the output voltage is ``v_pos - v_neg``.

    Examples
    --------
    >>> from pyDataconverter.architectures.SimpleDAC import SimpleDAC
    >>> dac = SimpleDAC(n_bits=8, v_ref=1.0)
    >>> result = calculate_dac_static_metrics(dac)
    >>> result['MaxDNL']
    0.0
    >>> result['Offset']
    0.0
    """
    if not isinstance(dac, DACBase):
        raise TypeError("dac must be a DACBase instance")
    if n_points is not None and n_points < 2:
        raise ValueError("n_points must be >= 2")
    if inl_method not in ('endpoint', 'best_fit'):
        raise ValueError("inl_method must be 'endpoint' or 'best_fit'")

    max_code = 2 ** dac.n_bits - 1
    if n_points is None:
        codes = np.arange(0, max_code + 1)
    else:
        codes = np.unique(np.linspace(0, max_code, n_points).astype(int))

    voltages = np.empty(len(codes), dtype=float)
    for i, code in enumerate(codes):
        result = dac.convert(int(code))
        if dac.output_type == OutputType.DIFFERENTIAL:
            v_pos, v_neg = result
            voltages[i] = v_pos - v_neg
        else:
            voltages[i] = result

    lsb = dac.v_ref / (2 ** dac.n_bits - 1)

    line = _fit_reference_line(codes.astype(float), voltages, inl_method)

    step_voltages = np.diff(voltages)
    dnl = step_voltages / (lsb * np.diff(codes)) - 1
    inl = (voltages - line) / lsb
    offset = voltages[0] - 0.0

    ideal_span = (codes[-1] - codes[0]) * lsb
    gain_error = (voltages[-1] - voltages[0] - ideal_span) / ideal_span

    return {
        "DNL": dnl,
        "INL": inl,
        "MaxDNL": float(np.max(np.abs(dnl))),
        "MaxINL": float(np.max(np.abs(inl))),
        "Offset": float(offset),
        "GainError": float(gain_error),
        "Codes": codes,
        "Voltages": voltages,
    }


def _calculate_dac_dynamic_metrics_from_fft(freqs: np.ndarray = None,
                                             mags: np.ndarray = None,
                                             fs: float = None,
                                             f0: float = None,
                                             full_scale: float = None) -> Dict[str, float]:
    """
    Calculate dynamic DAC metrics from a pre-computed FFT (no zone filtering).

    This is an internal helper used by plot_output_spectrum, which already
    holds a zone-filtered FFT and does not have the original voltage array.
    For the full pipeline (voltages → FFT → zone selection → metrics), use
    calculate_dac_dynamic_metrics instead.

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


def calculate_dac_dynamic_metrics(voltages: np.ndarray,
                                  fs: float,
                                  fs_update: Optional[float] = None,
                                  nyquist_zone: int = 1,
                                  window: str = 'hann',
                                  full_scale: Optional[float] = None
                                  ) -> Dict[str, object]:
    """
    Compute dynamic performance metrics from captured DAC output.

    Performs an FFT on *voltages* (sampled at *fs*) and calculates SNR,
    SNDR, SFDR, THD, and ENOB within the requested Nyquist zone of the
    DAC update rate.

    Parameters
    ----------
    voltages : np.ndarray
        1-D array of captured DAC output samples (V), sampled at rate *fs*.
    fs : float
        Capture sampling rate (Hz).
    fs_update : float, optional
        DAC update rate (Hz).  Defaults to *fs* when *None* (no
        oversampling).  Required when the capture samples a DAC at a
        higher rate than its update clock.
    nyquist_zone : int, optional
        Which Nyquist zone of the DAC to analyse (default 1).
        Zone *i* covers ``[(i-1)*fs_update/2, i*fs_update/2)``.
        Zone 1 is baseband, zone 2 is the first image band, etc.
    window : str, optional
        Window function name passed to ``compute_fft`` (default ``'hann'``).
    full_scale : float, optional
        Full-scale voltage for dBFS normalisation.  When *None*, metrics
        are in dB.  When set, additional ``_dBFS`` keys are included.

    Returns
    -------
    dict
        SNR : float
        SNDR : float
        SFDR : float
        THD : float
        ENOB : float
        FundamentalFrequency : float
        FundamentalMagnitude : float
        ZoneBandHz : tuple of float
        NyquistZone : int

        When *full_scale* is not None:
        SNR_dBFS, SNDR_dBFS, SFDR_dBFS, THD_dBFS, FundamentalMagnitude_dBFS

    Raises
    ------
    ValueError
        If ``fs < nyquist_zone * fs_update``.
    ValueError
        If ``nyquist_zone < 1``.
    ValueError
        If *voltages* is not a 1-D array or is empty.
    """
    voltages = np.asarray(voltages)
    if voltages.ndim != 1 or voltages.size == 0:
        raise ValueError("voltages must be a non-empty 1-D array")
    if nyquist_zone < 1:
        raise ValueError("nyquist_zone must be >= 1")
    if fs_update is None:
        fs_update = fs
    if fs < nyquist_zone * fs_update:
        raise ValueError(
            f"fs ({fs}) must be >= nyquist_zone * fs_update "
            f"({nyquist_zone * fs_update}) to capture zone {nyquist_zone}"
        )

    freqs, mags = compute_fft(voltages, fs, window=window)

    f_low = (nyquist_zone - 1) * fs_update / 2
    f_high = nyquist_zone * fs_update / 2

    zone_mask = (freqs >= f_low) & (freqs < f_high)
    zone_freqs = freqs[zone_mask]
    zone_mags = mags[zone_mask]

    bin_width = freqs[1] - freqs[0]

    fund_idx_in_zone = np.argmax(zone_mags)
    fund_freq = zone_freqs[fund_idx_in_zone]
    fund_mag = zone_mags[fund_idx_in_zone]
    fund_pwr = 10 ** (fund_mag / 10)

    in_zone_harmonics = []
    nyquist = fs / 2
    for n in range(2, 9):
        f_harm = n * fund_freq
        f_alias = f_harm % fs
        if f_alias > nyquist:
            f_alias = fs - f_alias
        if f_low <= f_alias < f_high:
            idx = np.argmin(np.abs(zone_freqs - f_alias))
            if abs(zone_freqs[idx] - f_alias) <= 0.5 * bin_width:
                in_zone_harmonics.append((zone_freqs[idx], zone_mags[idx]))

    harmonic_pwr = sum(10 ** (h[1] / 10) for h in in_zone_harmonics)

    exclude_freqs = np.array([fund_freq] + [h[0] for h in in_zone_harmonics])
    noise_mask = ~np.any(
        np.abs(zone_freqs[np.newaxis, :] - exclude_freqs[:, np.newaxis]) <= bin_width,
        axis=0
    )

    noise_pwr = sum(10 ** (m / 10) for m in zone_mags[noise_mask])
    noise_pwr = max(float(noise_pwr), 1e-20)

    snr = 10 * np.log10(fund_pwr / noise_pwr)

    harmonic_pwr = max(float(harmonic_pwr), 1e-20)
    thd = 10 * np.log10(harmonic_pwr / fund_pwr)

    total_noise_and_dist_pwr = max(float(noise_pwr + harmonic_pwr), 1e-20)
    sndr = 10 * np.log10(fund_pwr / total_noise_and_dist_pwr)

    spur_mask = np.abs(zone_freqs - fund_freq) > bin_width
    if np.any(spur_mask):
        max_spur = np.max(zone_mags[spur_mask])
        sfdr = fund_mag - max_spur
    else:
        sfdr = float('inf')

    enob = (sndr - 1.76) / 6.02

    results = {
        "SNR": float(snr),
        "SNDR": float(sndr),
        "SFDR": float(sfdr),
        "THD": float(thd),
        "ENOB": float(enob),
        "FundamentalFrequency": float(fund_freq),
        "FundamentalMagnitude": float(fund_mag),
        "ZoneBandHz": (float(f_low), float(f_high)),
        "NyquistZone": nyquist_zone,
    }

    if full_scale is not None:
        N = len(voltages)
        level_correction = 20 * np.log10(full_scale / 2) + 20 * np.log10(N / 2)
        fund_mag_dBFS = fund_mag - level_correction

        results["FundamentalMagnitude_dBFS"] = float(fund_mag_dBFS)
        results["SFDR_dBFS"] = float(sfdr - fund_mag_dBFS)
        results["SNR_dBFS"] = float(snr - fund_mag_dBFS)
        results["SNDR_dBFS"] = float(sndr - fund_mag_dBFS)
        results["THD_dBFS"] = float(thd - fund_mag_dBFS)

    return results
