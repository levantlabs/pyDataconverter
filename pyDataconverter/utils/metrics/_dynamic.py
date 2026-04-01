"""
Internal spectral (dynamic) metrics helper.
"""

import numpy as np
from typing import Dict
from ..fft_analysis import find_fundamental, find_harmonics


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
