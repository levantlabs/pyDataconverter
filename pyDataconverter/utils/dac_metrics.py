"""
DAC Performance Metrics
======================

Functions for calculating DAC static and dynamic performance metrics.

Design Spec
-----------

### Conventions

- **LSB**: ``v_ref / (2**n_bits - 1)``  — max code maps exactly to ``v_ref``.
  This matches ``DACBase.lsb`` (see ``dataconverter.py``).
- **Differential DAC output**: always use ``v_pos - v_neg`` to collapse a
  differential pair to a single voltage before any metric computation.
- **Output type detection**: read ``dac.output_type`` (``OutputType.SINGLE``
  vs ``OutputType.DIFFERENTIAL``) to decide how to interpret ``dac.convert()``
  return values.

### Function 1 — ``calculate_dac_static_metrics``

**Purpose**: Sweep digital codes through a DAC instance and compute static
linearity metrics from the resulting transfer function.

**Algorithm steps**:

1. Determine the set of codes to sweep:
   - If ``n_points is None``: sweep every code from 0 to ``2**n_bits - 1``.
   - If ``n_points`` is given: select ``n_points`` evenly-spaced codes across
     the full range (``np.linspace(0, 2**n_bits - 1, n_points).astype(int)``),
     then ``np.unique`` to remove duplicates.
2. For each code call ``dac.convert(code)``.
   - ``OutputType.SINGLE`` → voltage is the returned float.
   - ``OutputType.DIFFERENTIAL`` → voltage is ``v_pos - v_neg``.
3. Compute the **ideal LSB**: ``lsb = dac.v_ref / (2**dac.n_bits - 1)``.
4. Build the **ideal straight line** (endpoint fit, IEEE 1057):
   ``ideal[i] = voltages[0] + (voltages[-1] - voltages[0]) *
   (codes[i] - codes[0]) / (codes[-1] - codes[0])``
5. Compute **step widths** from consecutive voltage differences:
   ``step_voltages = np.diff(voltages)``
   ``ideal_step = lsb * np.diff(codes)``  (accounts for non-unit code spacing
   when ``n_points`` is used).
6. **DNL** (per step): ``dnl = step_voltages / (lsb * np.diff(codes)) - 1``
   Each entry tells how much the actual step deviates from the ideal step,
   in LSB units.
7. **INL** (per code, endpoint-corrected):
   ``inl = (voltages - ideal) / lsb``
   Length equals number of codes; first and last entries are 0 by definition
   of the endpoint fit.
8. **Offset**: ``voltages[0] - 0.0``  (code 0 should produce 0 V ideally).
9. **Gain error** (fractional):
   ``(voltages[-1] - voltages[0] - dac.v_ref) / dac.v_ref``
   A positive value means the span is larger than ideal.
10. Return dict with keys:
    ``DNL, INL, MaxDNL, MaxINL, Offset, GainError, Codes, Voltages``

**Error conditions**:
- ``dac`` must be a ``DACBase`` instance (raise ``TypeError``).
- ``n_points``, if given, must be ``>= 2`` (raise ``ValueError``).

### Function 2 — ``calculate_dac_dynamic_metrics``

**Purpose**: Compute frequency-domain (dynamic) performance metrics from a
captured DAC output waveform, with support for Nyquist-zone selection.

**Parameters**:
- ``voltages``: 1-D array of captured DAC output samples (volts), sampled at
  rate ``fs``.
- ``fs``: Capture sampling rate (Hz).
- ``fs_update``: DAC update rate (Hz).  Defaults to ``fs`` when ``None``
  (i.e. no oversampling — capture rate equals DAC rate).
- ``nyquist_zone``: integer >= 1.  Zone *i* covers the frequency band
  ``[(i-1)*fs_update/2,  i*fs_update/2)``.
  Zone 1 is baseband (0 to ``fs_update/2``).
  Zone 2 is the first image band (``fs_update/2`` to ``fs_update``), etc.
- ``window``: window function name passed to ``compute_fft`` (default
  ``'hann'``).
- ``full_scale``: full-scale voltage for dBFS normalization.  When ``None``,
  absolute-dB metrics are returned; when set, additional ``_dBFS`` keys are
  included (mirrors ``calculate_adc_dynamic_metrics`` pattern).

**Algorithm steps**:

1. Default ``fs_update = fs`` when ``None``.
2. **Validate**: ``fs >= nyquist_zone * fs_update``, else ``ValueError``.
   The zone's upper edge is ``nyquist_zone * fs_update / 2``; to be visible
   in the FFT the capture rate must satisfy ``fs/2 >= zone_upper``.
3. Compute FFT via ``compute_fft(voltages, fs, window=window)``.
4. Define the **zone band**:
   ``f_low  = (nyquist_zone - 1) * fs_update / 2``
   ``f_high = nyquist_zone * fs_update / 2``
5. Build a boolean mask selecting only FFT bins within ``[f_low, f_high)``.
6. **Find fundamental** within the zone: peak bin in the masked spectrum
   (use ``find_fundamental`` with zone-restricted search).
7. **Find harmonics** (up to 7): for each harmonic order *n*, compute the
   expected harmonic frequency ``n * f_fund``, fold it into ``[0, fs/2]``
   using aliasing rules, and check whether it falls in the zone.  Only
   harmonics landing inside the zone contribute to in-zone THD.
8. Compute **in-zone metrics** using the zone-masked spectrum:
   - **SNR**: fundamental power / total in-zone noise power (excluding
     fundamental and in-zone harmonics).
   - **THD**: in-zone harmonic power / fundamental power.
   - **SNDR**: fundamental / (noise + distortion).
   - **SFDR**: fundamental magnitude − worst in-zone spur.
   - **ENOB**: ``(SNDR − 1.76) / 6.02``.
9. If ``full_scale`` is provided, add ``_dBFS`` keys following the same
   convention as ``calculate_adc_dynamic_metrics``.
10. Return dict with keys:
    ``SNR, SNDR, SFDR, THD, ENOB, FundamentalFrequency,
    FundamentalMagnitude, ZoneBandHz, NyquistZone``
    plus ``SNR_dBFS, SNDR_dBFS, SFDR_dBFS, THD_dBFS,
    FundamentalMagnitude_dBFS`` when ``full_scale`` is not ``None``.

**Error conditions**:
- ``ValueError`` if ``fs < nyquist_zone * fs_update``.
- ``ValueError`` if ``nyquist_zone < 1``.
- ``ValueError`` if ``voltages`` is empty or 1-D check fails.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
from .fft_analysis import compute_fft, find_harmonics, find_fundamental
from ..dataconverter import DACBase, OutputType


def calculate_dac_static_metrics(dac: DACBase,
                                 n_points: Optional[int] = None
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

    Returns
    -------
    dict
        DNL : np.ndarray
            Differential non-linearity per code step, in LSB units.
            Length is ``len(Codes) - 1``.
        INL : np.ndarray
            Integral non-linearity (endpoint-fit), in LSB units.
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

    Notes
    -----
    - LSB is defined as ``v_ref / (2**n_bits - 1)`` so that the maximum
      code maps exactly to ``v_ref``.
    - For differential DACs the output voltage is ``v_pos - v_neg``.
    - INL uses an endpoint-fit reference line per IEEE 1057.

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
    # Validate inputs
    if not isinstance(dac, DACBase):
        raise TypeError("dac must be a DACBase instance")
    if n_points is not None and n_points < 2:
        raise ValueError("n_points must be >= 2")

    # Step 1: Determine codes to sweep
    max_code = 2 ** dac.n_bits - 1
    if n_points is None:
        codes = np.arange(0, max_code + 1)
    else:
        codes = np.unique(np.linspace(0, max_code, n_points).astype(int))

    # Step 2: Sweep codes through DAC
    voltages = np.empty(len(codes), dtype=float)
    for i, code in enumerate(codes):
        result = dac.convert(int(code))
        if dac.output_type == OutputType.DIFFERENTIAL:
            v_pos, v_neg = result
            voltages[i] = v_pos - v_neg
        else:
            voltages[i] = result

    # Step 3: Ideal LSB
    lsb = dac.v_ref / (2 ** dac.n_bits - 1)

    # Step 4: Ideal straight line (endpoint fit, IEEE 1057)
    ideal = voltages[0] + (voltages[-1] - voltages[0]) * (codes - codes[0]) / (codes[-1] - codes[0])

    # Step 5: Step widths
    step_voltages = np.diff(voltages)

    # Step 6: DNL
    dnl = step_voltages / (lsb * np.diff(codes)) - 1

    # Step 7: INL (endpoint-corrected)
    inl = (voltages - ideal) / lsb

    # Step 8: Offset
    offset = voltages[0] - 0.0

    # Step 9: Gain error (fractional)
    # Use the ideal span based on the actual code range swept, not dac.v_ref
    # directly, so the formula remains correct for any subset of codes.
    ideal_span = (codes[-1] - codes[0]) * lsb
    gain_error = (voltages[-1] - voltages[0] - ideal_span) / ideal_span

    # Step 10: Return results
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
            Signal-to-noise ratio (dB), fundamental vs in-zone noise.
        SNDR : float
            Signal-to-noise-and-distortion ratio (dB).
        SFDR : float
            Spurious-free dynamic range (dB), fundamental vs worst spur.
        THD : float
            Total harmonic distortion (dB), in-zone harmonics only.
        ENOB : float
            Effective number of bits, ``(SNDR - 1.76) / 6.02``.
        FundamentalFrequency : float
            Detected fundamental frequency (Hz).
        FundamentalMagnitude : float
            Fundamental magnitude (dB or dBFS).
        ZoneBandHz : tuple of float
            ``(f_low, f_high)`` edges of the analysed Nyquist zone (Hz).
        NyquistZone : int
            The zone number that was analysed.

        When *full_scale* is not None, the following keys are added:

        SNR_dBFS : float
        SNDR_dBFS : float
        SFDR_dBFS : float
        THD_dBFS : float
        FundamentalMagnitude_dBFS : float

    Raises
    ------
    ValueError
        If ``fs < nyquist_zone * fs_update`` (requested zone is not
        visible in the captured spectrum).
    ValueError
        If ``nyquist_zone < 1``.
    ValueError
        If *voltages* is not a 1-D array or is empty.

    Notes
    -----
    - Nyquist zone *i* spans ``[(i-1)*fs_update/2, i*fs_update/2)``.
      For a DAC with update rate *fs_update*, the baseband output lives in
      zone 1.  Images (sinc-weighted replicas) appear in higher zones.
    - Only harmonics that alias *into* the selected zone contribute to
      the in-zone THD and noise calculations.
    - The dBFS convention matches ``calculate_adc_dynamic_metrics`` in
      ``metrics.py``.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1e6
    >>> t = np.arange(1024) / fs
    >>> voltages = 0.5 * np.sin(2 * np.pi * 10e3 * t)
    >>> result = calculate_dac_dynamic_metrics(voltages, fs)
    >>> result['NyquistZone']
    1
    >>> result['ENOB'] > 0
    True
    """
    # Validate inputs
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

    # Step 3: Compute FFT
    freqs, mags = compute_fft(voltages, fs, window=window)

    # Step 4: Define zone band
    f_low = (nyquist_zone - 1) * fs_update / 2
    f_high = nyquist_zone * fs_update / 2

    # Step 5: Build zone mask
    zone_mask = (freqs >= f_low) & (freqs < f_high)
    zone_freqs = freqs[zone_mask]
    zone_mags = mags[zone_mask]

    bin_width = freqs[1] - freqs[0]

    # Step 6: Find fundamental within zone (peak bin in masked spectrum)
    fund_idx_in_zone = np.argmax(zone_mags)
    fund_freq = zone_freqs[fund_idx_in_zone]
    fund_mag = zone_mags[fund_idx_in_zone]
    fund_pwr = 10 ** (fund_mag / 10)

    # Step 7: Find harmonics that land in the zone
    in_zone_harmonics = []
    nyquist = fs / 2
    for n in range(2, 9):  # harmonics 2 through 8
        f_harm = n * fund_freq
        # Fold into [0, fs/2] using aliasing rules
        # Reduce modulo fs, then fold
        f_alias = f_harm % fs
        if f_alias > nyquist:
            f_alias = fs - f_alias
        # Check if this harmonic falls within the zone
        if f_low <= f_alias < f_high:
            # Find nearest bin in zone
            idx = np.argmin(np.abs(zone_freqs - f_alias))
            if abs(zone_freqs[idx] - f_alias) <= 0.5 * bin_width:
                in_zone_harmonics.append((zone_freqs[idx], zone_mags[idx]))

    # Step 8: Compute in-zone metrics
    # Harmonic power (in-zone only)
    harmonic_pwr = sum(10 ** (h[1] / 10) for h in in_zone_harmonics)

    # Build exclusion mask for noise calculation (exclude fundamental and in-zone harmonics)
    exclude_freqs = np.array([fund_freq] + [h[0] for h in in_zone_harmonics])
    noise_mask = ~np.any(
        np.abs(zone_freqs[np.newaxis, :] - exclude_freqs[:, np.newaxis]) <= bin_width,
        axis=0
    )

    noise_pwr = sum(10 ** (m / 10) for m in zone_mags[noise_mask])
    noise_pwr = max(float(noise_pwr), 1e-20)

    # SNR
    snr = 10 * np.log10(fund_pwr / noise_pwr)

    # THD
    harmonic_pwr = max(float(harmonic_pwr), 1e-20)
    thd = 10 * np.log10(harmonic_pwr / fund_pwr)

    # SNDR
    total_noise_and_dist_pwr = max(float(noise_pwr + harmonic_pwr), 1e-20)
    sndr = 10 * np.log10(fund_pwr / total_noise_and_dist_pwr)

    # SFDR: fundamental magnitude - worst in-zone spur
    spur_mask = np.abs(zone_freqs - fund_freq) > bin_width
    if np.any(spur_mask):
        max_spur = np.max(zone_mags[spur_mask])
        sfdr = fund_mag - max_spur
    else:
        sfdr = float('inf')

    # ENOB
    enob = (sndr - 1.76) / 6.02

    # Build results
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

    # Step 9: dBFS keys if full_scale provided
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
