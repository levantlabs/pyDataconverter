"""
Characterization Utilities
==========================

High-level convenience functions that drive an ADC through a sweep and
return a scalar figure-of-merit (DR, ERBW).

These functions accept any object that implements ``.convert(float) -> int``
so they work with FlashADC, SARADC, or any custom model.
"""

import numpy as np
from typing import Dict, Optional

from .signal_gen import generate_coherent_sine
from .metrics import (
    calculate_adc_dynamic_metrics,
    calculate_dynamic_range_from_curve,
    calculate_erbw_from_curve,
)


def measure_dynamic_range(
        adc,
        n_bits: int,
        v_ref: float,
        fs: float,
        n_fft: int,
        n_fin: int,
        n_amplitudes: int = 20,
        amplitude_range_dBFS: tuple = (-80.0, -1.0),
) -> Dict:
    """
    Measure ADC dynamic range by sweeping input amplitude.

    Generates coherent sine waves at logarithmically-spaced amplitudes
    from ``amplitude_range_dBFS[0]`` to ``amplitude_range_dBFS[1]`` dBFS,
    converts each through ``adc``, measures SNR, then calls
    ``calculate_dynamic_range_from_curve`` to find where SNR = 0 dB.

    Args:
        adc: ADC model.  Must implement ``.convert(float) -> int``.
        n_bits: ADC resolution (bits).
        v_ref: Full-scale reference voltage (V).
        fs: Sampling rate (Hz).
        n_fft: FFT / record length (samples per measurement).
        n_fin: Input frequency bin number (integer, coherent with n_fft).
               Actual frequency = n_fin / n_fft * fs.
        n_amplitudes: Number of amplitude steps in the sweep (default 20).
        amplitude_range_dBFS: (low_dBFS, high_dBFS) sweep range.
            Default (-80, -1) covers most of the dynamic range for a
            well-behaved ADC.

    Returns:
        Dict with keys:
            DR_dB          : Dynamic range in dB.
            Amplitudes_dBFS: np.ndarray of sweep amplitudes (dBFS).
            SNR_values     : np.ndarray of measured SNR (dB) at each step.
            AmplitudeAtSNR0_dBFS: Amplitude where SNR interpolates to 0 dB.
    """
    full_scale_amp = v_ref / 2.0
    offset         = v_ref / 2.0

    amplitudes_dBFS = np.linspace(amplitude_range_dBFS[0],
                                   amplitude_range_dBFS[1],
                                   n_amplitudes)
    amplitudes_v = full_scale_amp * 10 ** (amplitudes_dBFS / 20.0)

    snr_values = np.zeros(n_amplitudes)
    for i, amp in enumerate(amplitudes_v):
        vin, _ = generate_coherent_sine(fs, n_fft, n_fin,
                                        amplitude=amp, offset=offset)
        codes = np.array([adc.convert(float(v)) for v in vin], dtype=float)
        m = calculate_adc_dynamic_metrics(time_data=codes, fs=fs)
        snr_values[i] = m['SNR']
        if not np.isfinite(snr_values[i]):
            import warnings
            warnings.warn(
                f"non-finite SNR ({snr_values[i]}) at amplitude "
                f"{amplitudes_dBFS[i]:.2f} dBFS — the FFT produced a "
                f"degenerate spectrum (all-zero output, clipping, or "
                f"missing fundamental). The sweep result at this point "
                f"will not be meaningful.",
                RuntimeWarning,
                stacklevel=2,
            )

    dr_result = calculate_dynamic_range_from_curve(amplitudes_dBFS, snr_values)

    return {
        'DR_dB':               dr_result['DR_dB'],
        'AmplitudeAtSNR0_dBFS': dr_result['AmplitudeAtSNR0_dB'],
        'Amplitudes_dBFS':     amplitudes_dBFS,
        'SNR_values':          snr_values,
    }


def measure_erbw(
        adc,
        n_bits: int,
        v_ref: float,
        fs: float,
        n_fft: int,
        freq_range_hz: tuple,
        n_frequencies: int = 20,
        amplitude_dBFS: float = -3.0,
) -> Dict:
    """
    Measure ADC effective resolution bandwidth (ERBW).

    Generates coherent sine waves at logarithmically-spaced frequencies
    from ``freq_range_hz[0]`` to ``freq_range_hz[1]``, converts each
    through ``adc``, measures ENOB, then calls
    ``calculate_erbw_from_curve`` to find where ENOB drops by 0.5 bits.

    Each frequency uses the nearest coherent bin to the target (to avoid
    spectral leakage), so the actual frequency array may differ slightly
    from the requested one.

    Args:
        adc: ADC model.  Must implement ``.convert(float) -> int``.
        n_bits: ADC resolution (bits).
        v_ref: Full-scale reference voltage (V).
        fs: Sampling rate (Hz).
        n_fft: FFT / record length (samples per measurement).
        freq_range_hz: (f_low, f_high) frequency sweep range in Hz.
        n_frequencies: Number of frequency steps (default 20).
        amplitude_dBFS: Input amplitude in dBFS (default -3 dBFS ≈ full scale).

    Returns:
        Dict with keys:
            ERBW_Hz      : Effective resolution bandwidth in Hz.
            ENOB_ref     : ENOB at the lowest measured frequency.
            Frequencies_Hz: np.ndarray of actual measurement frequencies.
            ENOB_values  : np.ndarray of ENOB at each frequency.
    """
    full_scale_amp = v_ref / 2.0
    amplitude      = full_scale_amp * 10 ** (amplitude_dBFS / 20.0)
    offset         = v_ref / 2.0

    f_low, f_high = freq_range_hz
    target_freqs = np.geomspace(f_low, f_high, n_frequencies)

    # Snap each target to the nearest coherent bin
    actual_freqs = np.zeros(n_frequencies)
    enob_values  = np.zeros(n_frequencies)

    for i, f_target in enumerate(target_freqs):
        n_fin = max(1, round(f_target / fs * n_fft))
        n_fin = min(n_fin, n_fft // 2 - 1)
        actual_freq = n_fin / n_fft * fs
        actual_freqs[i] = actual_freq

        vin, _ = generate_coherent_sine(fs, n_fft, n_fin,
                                        amplitude=amplitude, offset=offset)
        codes = np.array([adc.convert(float(v)) for v in vin], dtype=float)
        m = calculate_adc_dynamic_metrics(time_data=codes, fs=fs)
        enob_values[i] = m['ENOB']

    erbw_result = calculate_erbw_from_curve(actual_freqs, enob_values)

    return {
        'ERBW_Hz':       erbw_result['ERBW_Hz'],
        'ENOB_ref':      erbw_result['ENOB_ref'],
        'Frequencies_Hz': actual_freqs,
        'ENOB_values':   enob_values,
    }
