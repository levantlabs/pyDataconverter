"""
SimpleDAC Usage Examples
========================

Demonstrates the SimpleDAC with various configurations:
  1. Ideal transfer curve — single-ended and differential
  2. Non-idealities — gain error, offset, and noise effects on transfer curve
  3. INL / DNL — static linearity with and without non-idealities
  4. Output spectrum — FFT of a sinusoidal code sequence (ideal vs noisy)
  5. Non-ideality sweep — SNR vs noise_rms for different resolutions
  6. ZOH oversampling — time-domain staircase, full multi-zone spectrum, first and third Nyquist zones
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.visualizations.dac_plots import (
    plot_transfer_curve,
    plot_inl_dnl,
    plot_output_spectrum,
)
from pyDataconverter.utils.signal_gen import generate_digital_sine
from pyDataconverter.utils.fft_analysis import compute_fft, FFTNormalization
import pyDataconverter.utils.metrics as metrics


# ---------------------------------------------------------------------------
# 1. Ideal transfer curve — single-ended and differential
# ---------------------------------------------------------------------------

def demo_ideal_transfer():
    """Sweep all codes through ideal 8-bit DACs (single-ended & differential)."""
    print("--- 1. Ideal transfer curve (8-bit) ---")

    n_bits = 8
    v_ref = 1.0

    dac_se = SimpleDAC(n_bits=n_bits, v_ref=v_ref, output_type=OutputType.SINGLE)
    dac_diff = SimpleDAC(n_bits=n_bits, v_ref=v_ref, output_type=OutputType.DIFFERENTIAL)

    codes = np.arange(2 ** n_bits)
    v_se = np.array([dac_se.convert(int(c)) for c in codes])
    v_diff = np.array([dac_diff.convert(int(c)) for c in codes], dtype=object)
    v_pos = np.array([v[0] for v in v_diff])
    v_neg = np.array([v[1] for v in v_diff])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{n_bits}-bit SimpleDAC — Ideal Transfer Curve', fontweight='bold')

    # Single-ended
    ax1.plot(codes, v_se, color='steelblue', linewidth=1.5)
    ax1.set_xlabel('Input Code')
    ax1.set_ylabel('Output Voltage (V)')
    ax1.set_title('Single-Ended')
    ax1.grid(True, alpha=0.4)

    # Differential
    ax2.plot(codes, v_pos, color='steelblue', linewidth=1.5, label='V+')
    ax2.plot(codes, v_neg, color='tomato', linewidth=1.5, label='V−')
    ax2.plot(codes, v_pos - v_neg, color='seagreen', linewidth=1.5,
             linestyle='--', label='V+ − V−')
    ax2.set_xlabel('Input Code')
    ax2.set_ylabel('Output Voltage (V)')
    ax2.set_title('Differential')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Non-idealities — effect on transfer curve
# ---------------------------------------------------------------------------

def demo_nonideality_transfer():
    """
    Compare ideal vs non-ideal 8-bit DACs using the plot_transfer_curve helper.
    Shows gain error, offset, and noise individually and combined.
    """
    print("--- 2. Non-ideality effects on transfer curve ---")

    n_bits = 8
    v_ref = 1.0

    cases = [
        ("Ideal",                  dict()),
        ("Gain error +2 %",       dict(gain_error=0.02)),
        ("Offset +10 mV",         dict(offset=0.010)),
        ("Noise 1 mV rms",        dict(noise_rms=1e-3)),
        ("Combined",              dict(gain_error=0.02, offset=0.010, noise_rms=1e-3)),
    ]

    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 4 * len(cases)),
                             sharex=True)
    fig.suptitle(f'{n_bits}-bit SimpleDAC — Non-Ideality Comparison',
                 fontweight='bold', y=1.0)

    for row, (label, params) in enumerate(cases):
        np.random.seed(42)  # reproducible noise
        dac = SimpleDAC(n_bits=n_bits, v_ref=v_ref, output_type=OutputType.SINGLE,
                        **params)

        # Use the library plot helper but capture its figure
        sub_fig, (ax_top, ax_bot) = plot_transfer_curve(dac, title=label)
        plt.close(sub_fig)  # we'll redraw into our grid

        # Re-create the data for the combined figure
        n_codes = 2 ** n_bits
        codes = np.arange(n_codes)
        ideal_dac = SimpleDAC(n_bits, v_ref, OutputType.SINGLE)
        ideal_v = np.array([ideal_dac.convert(int(c)) for c in codes])

        np.random.seed(42)
        actual_v = np.array([dac.convert(int(c)) for c in codes])
        lsb = v_ref / (n_codes - 1)
        error_lsb = (actual_v - ideal_v) / lsb

        ax_t = axes[row, 0]
        ax_t.plot(codes, ideal_v, color='gray', linestyle='--', linewidth=1.0,
                  label='Ideal')
        ax_t.plot(codes, actual_v, color='steelblue', linewidth=1.5,
                  label='Actual')
        ax_t.set_ylabel('V_out (V)')
        ax_t.set_title(label)
        ax_t.legend(loc='upper left', fontsize=8)
        ax_t.grid(True, alpha=0.4)

        ax_b = axes[row, 1]
        ax_b.plot(codes, error_lsb, color='steelblue', linewidth=1.0)
        ax_b.axhline(0.0, color='black', linewidth=0.7)
        ax_b.axhline(0.5, color='gray', linewidth=0.7, linestyle=':')
        ax_b.axhline(-0.5, color='gray', linewidth=0.7, linestyle=':')
        ax_b.set_ylabel('Error (LSB)')
        ax_b.set_title(f'{label} — Error')
        ax_b.grid(True, alpha=0.4)

    axes[-1, 0].set_xlabel('Input Code')
    axes[-1, 1].set_xlabel('Input Code')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. INL / DNL — static linearity
# ---------------------------------------------------------------------------

def demo_inl_dnl():
    """Plot INL and DNL for an ideal DAC and one with gain error + offset."""
    print("--- 3. INL / DNL ---")

    n_bits = 8
    v_ref = 1.0

    # Ideal
    dac_ideal = SimpleDAC(n_bits=n_bits, v_ref=v_ref)
    fig_ideal, _ = plot_inl_dnl(dac_ideal, title='Ideal 8-bit DAC')
    plt.show()
    plt.close(fig_ideal)

    # With gain error and offset
    dac_nonideal = SimpleDAC(n_bits=n_bits, v_ref=v_ref,
                             gain_error=0.005, offset=2e-3)
    fig_ni, _ = plot_inl_dnl(dac_nonideal,
                             title='8-bit DAC (gain_error=0.5 %, offset=2 mV)')
    plt.show()
    plt.close(fig_ni)


# ---------------------------------------------------------------------------
# 4. Output spectrum — ideal vs noisy
# ---------------------------------------------------------------------------

def demo_output_spectrum():
    """
    Drive a 10-bit DAC with a sinusoidal code sequence and plot the FFT.
    Compares an ideal DAC to one with noise, offset, and gain error.
    """
    print("--- 4. Output spectrum (10-bit DAC) ---")

    n_bits = 10
    v_ref  = 1.0
    fs     = 1e6     # DAC update rate
    f_sig  = 50e3
    n_fft  = 4096

    n_fin    = max(1, round(f_sig * n_fft / fs))
    f_actual = n_fin / n_fft * fs
    codes    = generate_digital_sine(n_bits, f_actual, fs,
                                     amplitude=0.9, offset=0.5, duration=n_fft / fs)

    # --- Ideal ---
    dac_ideal = SimpleDAC(n_bits=n_bits, v_ref=v_ref, fs=fs)
    _, voltages1 = dac_ideal.convert_sequence(codes)
    freqs1, mags1 = compute_fft(voltages1, fs, normalization=FFTNormalization.DBFS,
                                 full_scale=v_ref)
    ax1 = plot_output_spectrum(freqs1, mags1, fs,
                               title='Ideal 10-bit DAC — Output Spectrum')
    plt.show()
    plt.close(ax1.get_figure())

    # --- With non-idealities ---
    np.random.seed(0)
    dac_noisy = SimpleDAC(n_bits=n_bits, v_ref=v_ref, fs=fs,
                          noise_rms=0.5e-3, offset=5e-3, gain_error=0.01)
    _, voltages2 = dac_noisy.convert_sequence(codes)
    freqs2, mags2 = compute_fft(voltages2, fs, normalization=FFTNormalization.DBFS,
                                 full_scale=v_ref)
    ax2 = plot_output_spectrum(freqs2, mags2, fs,
                               title='10-bit DAC (noise=0.5mV, offset=5mV, gain=+1%) — Spectrum')
    plt.show()
    plt.close(ax2.get_figure())


# ---------------------------------------------------------------------------
# 5. Non-ideality sweep — SNR vs noise_rms for different resolutions
# ---------------------------------------------------------------------------

def demo_snr_vs_noise():
    """
    Sweep output noise_rms and measure SNR for 8-, 10-, and 12-bit DACs.
    Shows how noise degrades dynamic performance as resolution increases.
    """
    print("--- 5. SNR vs output noise (8/10/12-bit) ---")

    v_ref = 1.0
    fs = 1e6
    n_fft = 4096
    f_sig = 50e3

    # Snap to coherent bin
    n_fin = max(1, round(f_sig * n_fft / fs))
    f_actual = n_fin / n_fft * fs

    bit_configs = [8, 10, 12]
    noise_levels = np.logspace(-5, -2, 15)  # 10 uV to 10 mV

    colors = {'8': 'steelblue', '10': 'tomato', '12': 'seagreen'}

    fig, ax = plt.subplots(figsize=(10, 6))

    for n_bits in bit_configs:
        snrs = []
        for noise_rms in noise_levels:
            np.random.seed(7)
            dac = SimpleDAC(n_bits=n_bits, v_ref=v_ref, noise_rms=noise_rms)

            duration = n_fft / fs
            codes = generate_digital_sine(n_bits, f_actual, fs,
                                          amplitude=0.9, offset=0.5,
                                          duration=duration)
            voltages = np.array([dac.convert(int(c)) for c in codes])

            freqs, mags = compute_fft(voltages, fs, window='hann',
                                      normalization=FFTNormalization.DBFS,
                                      full_scale=v_ref)
            res = metrics.calculate_adc_dynamic_metrics(
                freqs=freqs, mags=mags, fs=fs, f0=f_actual)
            snrs.append(res['SNR'])

        ax.semilogx(noise_levels * 1e3, snrs, marker='o', markersize=4,
                     linewidth=1.8, color=colors[str(n_bits)],
                     label=f'{n_bits}-bit')

    # Ideal quantization SNR reference lines
    for n_bits in bit_configs:
        ideal_snr = 6.02 * n_bits + 1.76
        ax.axhline(ideal_snr, color=colors[str(n_bits)],
                    linewidth=0.8, linestyle=':', alpha=0.6)
        ax.text(noise_levels[0] * 1e3, ideal_snr + 1,
                f'{n_bits}-bit ideal ({ideal_snr:.1f} dB)',
                fontsize=7, color=colors[str(n_bits)])

    ax.set_xlabel('Output Noise RMS (mV)')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SimpleDAC: SNR vs Output Noise for Different Resolutions')
    ax.legend()
    ax.grid(True, alpha=0.4, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. ZOH oversampling
# ---------------------------------------------------------------------------

def demo_zoh_oversampling():
    """
    Demonstrates ZOH (zero-order hold) oversampling with four plots:
      1. Zoomed time-domain staircase vs ideal analog sine (with ZOH sample markers)
      2. Full oversampled spectrum — all Nyquist zones with sinc envelope
      3. First Nyquist zone only (clipped at fs/2, with metrics)
      4. Third Nyquist zone — shows attenuated image beyond 2× update rate
    """
    print("--- 6. ZOH oversampling (10-bit, 8× ZOH) ---")

    n_bits     = 10
    v_ref      = 1.0
    fs         = 1e6
    oversample = 8
    f_sig      = 50e3
    n_fft      = 1024

    dac = SimpleDAC(n_bits=n_bits, v_ref=v_ref, fs=fs, oversample=oversample)

    # Snap to a coherent FFT bin and generate the ZOH waveform
    n_fin    = max(1, round(f_sig * n_fft / fs))
    f_actual = n_fin / n_fft * fs
    codes    = generate_digital_sine(n_bits, f_actual, fs,
                                     amplitude=0.9, offset=0.5, duration=n_fft / fs)
    t, voltages = dac.convert_sequence(codes)

    # --- Plot 1: zoomed time-domain staircase ---
    n_cycles  = 2
    t_zoom    = n_cycles / f_actual
    zoom      = t <= t_zoom
    t_fine    = np.linspace(0, t_zoom, 4000)
    v_ideal   = (0.9 / 2 * np.sin(2 * np.pi * f_actual * t_fine) + 0.5) * v_ref
    n_updates = int(t_zoom * fs) + 1
    t_updates = np.arange(n_updates) / fs
    v_updates = np.array([dac.convert(int(codes[k])) for k in range(n_updates)])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_fine * 1e6, v_ideal, color='gray', linewidth=1.2,
            linestyle='--', label='Ideal analog')
    ax.plot(t[zoom] * 1e6, voltages[zoom], color='steelblue', linewidth=1.5,
            drawstyle='steps-post', label=f'ZOH output (×{oversample})')
    ax.scatter(t[zoom] * 1e6, voltages[zoom], color='steelblue', s=10,
               zorder=4, alpha=0.6, label=f'ZOH samples (×{oversample})')
    ax.scatter(t_updates * 1e6, v_updates, color='tomato', s=40, zorder=5,
               label='DAC update')
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Output Voltage (V)')
    ax.set_title(f'{n_bits}-bit DAC — ZOH Time Domain  '
                 f'({n_cycles} cycles, f={f_actual/1e3:.1f} kHz, '
                 f'fs={fs/1e6:.0f} MHz, ×{oversample} ZOH)')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # compute_fft at the oversampled rate — voltages has oversample * n_fft points
    fs_out = fs * oversample
    freqs, mags = compute_fft(voltages, fs_out, normalization=FFTNormalization.DBFS,
                               full_scale=v_ref)

    # --- Plot 2: full spectrum — all Nyquist zones with sinc envelope ---
    ax2 = plot_output_spectrum(freqs, mags, fs, nyquist_zone=None,
                               title=f'{n_bits}-bit DAC ZOH Spectrum')
    plt.show()
    plt.close(ax2.get_figure())

    # --- Plot 3: first Nyquist zone with metrics ---
    ax3 = plot_output_spectrum(freqs, mags, fs,
                               title=f'{n_bits}-bit DAC ZOH — First Nyquist Zone')
    plt.show()
    plt.close(ax3.get_figure())

    # --- Plot 4: third Nyquist zone (image at fs + f_sig, attenuated by ZOH sinc) ---
    ax4 = plot_output_spectrum(freqs, mags, fs, nyquist_zone=3,
                               title=f'{n_bits}-bit DAC ZOH — Third Nyquist Zone')
    plt.show()
    plt.close(ax4.get_figure())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    demo_ideal_transfer()
    demo_nonideality_transfer()
    demo_inl_dnl()
    demo_output_spectrum()
    demo_snr_vs_noise()
    demo_zoh_oversampling()
    print("\nAll examples complete.")


if __name__ == '__main__':
    main()
