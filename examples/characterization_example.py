"""
Characterization Example
========================

Demonstrates the high-level characterization utilities:

  1. ``measure_dynamic_range`` — sweeps input amplitude and finds where SNR
     drops to 0 dB (the noise floor).  Reports DR in dB.

  2. ``measure_erbw`` — sweeps input frequency and finds where ENOB drops
     by 0.5 bits from the low-frequency reference.  Reports ERBW in Hz.

Both functions are applied to an 8-bit FlashADC.

A two-panel figure shows:
  - Left:  SNR vs input amplitude (dBFS)  — DR curve
  - Right: ENOB vs input frequency (Hz)   — ERBW curve

Usage
-----
    python examples/characterization_example.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.utils.characterization import measure_dynamic_range, measure_erbw


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_BITS  = 8
V_REF   = 1.0
FS      = 10e6       # sampling rate (Hz)
N_FFT   = 256        # FFT / record length per measurement
N_FIN   = 13         # coherent frequency bin for DR sweep

# DR sweep
N_AMPLITUDES     = 12
AMPLITUDE_RANGE  = (-70.0, -1.0)    # dBFS

# ERBW sweep
N_FREQUENCIES    = 12
FREQ_RANGE       = (1e4, 4.5e6)     # Hz  (up to near Nyquist = fs/2)
AMPLITUDE_DBFS   = -3.0             # near full scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Characterization of {N_BITS}-bit FlashADC')
    print(f'  fs = {FS/1e6:.1f} MHz,  N_FFT = {N_FFT}')
    print()

    adc = FlashADC(n_bits=N_BITS, v_ref=V_REF)

    # ------------------------------------------------------------------
    # 1. Dynamic range sweep
    # ------------------------------------------------------------------
    print('--- measure_dynamic_range ---')
    dr_result = measure_dynamic_range(
        adc,
        n_bits=N_BITS,
        v_ref=V_REF,
        fs=FS,
        n_fft=N_FFT,
        n_fin=N_FIN,
        n_amplitudes=N_AMPLITUDES,
        amplitude_range_dBFS=AMPLITUDE_RANGE,
    )

    print(f'  DR       = {dr_result["DR_dB"]:.2f} dB')
    print(f'  SNR=0 dB at amplitude = {dr_result["AmplitudeAtSNR0_dBFS"]:.2f} dBFS')
    print(f'  Ideal DR (6.02N + 1.76) = {6.02 * N_BITS + 1.76:.1f} dB')

    # ------------------------------------------------------------------
    # 2. ERBW sweep
    # ------------------------------------------------------------------
    print('\n--- measure_erbw ---')
    # Re-create a fresh ADC (same ideal model, no state)
    adc2 = FlashADC(n_bits=N_BITS, v_ref=V_REF)

    erbw_result = measure_erbw(
        adc2,
        n_bits=N_BITS,
        v_ref=V_REF,
        fs=FS,
        n_fft=N_FFT,
        freq_range_hz=FREQ_RANGE,
        n_frequencies=N_FREQUENCIES,
        amplitude_dBFS=AMPLITUDE_DBFS,
    )

    nyquist = FS / 2
    print(f'  ERBW     = {erbw_result["ERBW_Hz"] / 1e6:.3f} MHz')
    print(f'  ENOB_ref = {erbw_result["ENOB_ref"]:.2f} bits  '
          f'(at {erbw_result["Frequencies_Hz"][0] / 1e3:.1f} kHz)')
    print(f'  Nyquist  = {nyquist / 1e6:.1f} MHz')
    if erbw_result['ERBW_Hz'] > 0:
        print(f'  ERBW / Nyquist = {erbw_result["ERBW_Hz"] / nyquist:.2%}')

    # ------------------------------------------------------------------
    # Figure: DR curve + ERBW curve side by side
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{N_BITS}-bit FlashADC — Characterization Sweep',
                 fontweight='bold')

    # --- Panel 1: SNR vs amplitude (DR curve) ---
    amps   = dr_result['Amplitudes_dBFS']
    snrs   = dr_result['SNR_values']

    ax1.plot(amps, snrs, color='steelblue', linewidth=2, marker='o',
             markersize=5, label='Measured SNR')
    ax1.axhline(0, color='tomato', linewidth=1.2, linestyle='--', label='SNR = 0 dB')
    ax1.axvline(dr_result['AmplitudeAtSNR0_dBFS'], color='tomato', linewidth=1.2,
                linestyle=':', alpha=0.7)

    dr_val = dr_result['DR_dB']
    ax1.annotate(f'DR = {dr_val:.1f} dB',
                 xy=(dr_result['AmplitudeAtSNR0_dBFS'], 0),
                 xytext=(amps[2], max(snrs) * 0.6),
                 fontsize=9, color='tomato',
                 arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2))

    ax1.set_xlabel('Input Amplitude (dBFS)')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Dynamic Range Sweep')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Panel 2: ENOB vs frequency (ERBW curve) ---
    freqs  = erbw_result['Frequencies_Hz']
    enobs  = erbw_result['ENOB_values']
    enob0  = erbw_result['ENOB_ref']
    erbw   = erbw_result['ERBW_Hz']

    ax2.semilogx(freqs / 1e6, enobs, color='seagreen', linewidth=2, marker='s',
                 markersize=5, label='Measured ENOB')
    ax2.axhline(enob0 - 0.5, color='tomato', linewidth=1.2, linestyle='--',
                label=f'ENOB_ref − 0.5 = {enob0 - 0.5:.2f} bits')

    if erbw > 0:
        ax2.axvline(erbw / 1e6, color='tomato', linewidth=1.2, linestyle=':',
                    alpha=0.8, label=f'ERBW = {erbw/1e6:.3f} MHz')

    ax2.set_xlabel('Input Frequency (MHz)')
    ax2.set_ylabel('ENOB (bits)')
    ax2.set_title('Effective Resolution Bandwidth Sweep')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4, which='both')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('characterization_demo.png', dpi=100, bbox_inches='tight')
    print('\n  Figure saved: characterization_demo.png')
    plt.close(fig)
    print('\nDone.')


if __name__ == '__main__':
    main()
