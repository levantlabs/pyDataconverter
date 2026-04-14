"""
Pipelined ADC Example
=====================

Constructs the canonical 12-bit pipelined ADC (3-bit first stage + 1026-level
backend flash) in the bipolar configuration that is bit-exact to the
adc_book reference, runs a coherent sine through it, reports SNR/SNDR/SFDR/
ENOB, and plots the output time-series and spectrum.

The bipolar setup: sub-ADC thresholds span [-v_ref/2, +v_ref/2] via an
explicit ArbitraryReference, and the sub-DAC uses offset=-v_ref/2 to
produce bipolar output levels matching the reference's
dacout[code] = code*lsb - v_ref/2. See
docs/superpowers/specs/2026-04-13-pipelined-adc-design.md Appendix A
and tests/test_pipelined_adc_vs_reference.py for the full derivation.

Run with:

    python examples/pipelined_adc_example.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly; comment out for interactive use
import matplotlib.pyplot as plt

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.architectures.PipelinedADC import PipelineStage, PipelinedADC
from pyDataconverter.components.residue_amplifier import ResidueAmplifier
from pyDataconverter.components.reference import ArbitraryReference
from pyDataconverter.dataconverter import InputType, OutputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine
from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics


def _compute_spectrum(codes: np.ndarray, fs: float):
    """FFT spectrum of a code sequence, in dB, independent of metric dict keys."""
    x = codes - np.mean(codes)
    spec = np.abs(np.fft.rfft(x))
    spec_db = 20 * np.log10(spec + 1e-30)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, spec_db


def build_canonical_pipelined_adc(fs: float = 500e6) -> PipelinedADC:
    """
    12-bit pipelined ADC matching the adc_book canonical configuration.

    Uses bipolar sub-ADC thresholds (via ArbitraryReference) and a
    bipolar-shifted SimpleDAC (via offset=-v_ref/2) so the construction
    matches the reference implementation exactly.
    """
    v_ref = 1.0

    # Bipolar stage0 sub-ADC: 8 thresholds at exactly the reference's spacing.
    # Reference formula (N=8, FSR=1):
    #   ref = arange(N)/(N-1)*(FSR-LSB) - (FSR/2-LSB/2)
    N_stage0 = 8
    lsb0 = v_ref / N_stage0
    stage0_thresholds = np.arange(N_stage0)/(N_stage0-1)*(v_ref-lsb0) - (v_ref/2-lsb0/2)
    stage0_ref = ArbitraryReference(stage0_thresholds, noise_rms=0.0)

    stage0_sub_adc = FlashADC(
        n_bits=3, v_ref=v_ref, input_type=InputType.SINGLE,
        n_comparators=N_stage0, reference=stage0_ref,
    )
    stage0_sub_dac = SimpleDAC(
        n_bits=3, n_levels=N_stage0 + 1, v_ref=v_ref,
        output_type=OutputType.SINGLE, offset=-v_ref / 2,
    )
    stage0_amp = ResidueAmplifier(gain=4.0, settling_tau=0.0)
    stage0 = PipelineStage(
        sub_adc=stage0_sub_adc, sub_dac=stage0_sub_dac,
        residue_amp=stage0_amp, fs=fs, code_offset=-1,
    )

    # Bipolar backend sub-ADC: 1026 thresholds (reference uses N=1026).
    N_backend = 1026
    lsb_be = v_ref / N_backend
    backend_thresholds = np.arange(N_backend)/(N_backend-1)*(v_ref-lsb_be) - (v_ref/2-lsb_be/2)
    backend_ref = ArbitraryReference(backend_thresholds, noise_rms=0.0)
    backend = FlashADC(
        n_bits=10, v_ref=v_ref, input_type=InputType.SINGLE,
        n_comparators=N_backend, reference=backend_ref,
    )

    return PipelinedADC(
        n_bits=12, v_ref=v_ref, input_type=InputType.SINGLE,
        stages=[stage0], backend=backend,
        backend_H=512, backend_code_offset=0, fs=fs,
    )


def main():
    fs = 500e6
    n_fft = 2 ** 14
    adc = build_canonical_pipelined_adc(fs=fs)

    # Bipolar coherent sine: centered at 0, amplitude 0.45 (well within
    # the bipolar sub-ADC's [-0.5, +0.5] range).
    vin, _ = generate_coherent_sine(fs, n_fft, n_fin=127,
                                    amplitude=0.45, offset=0.0)
    codes = np.array([adc.convert(float(v)) for v in vin], dtype=float)

    metrics = calculate_adc_dynamic_metrics(time_data=codes, fs=fs)
    print("Pipelined ADC — canonical 12-bit bipolar configuration")
    print(f"  fs       = {fs/1e6:.1f} MHz")
    print(f"  n_fft    = {n_fft}")
    print(f"  SNR      = {metrics['SNR']:.2f} dB")
    print(f"  SNDR     = {metrics['SNDR']:.2f} dB")
    print(f"  SFDR     = {metrics['SFDR']:.2f} dB")
    print(f"  ENOB     = {metrics['ENOB']:.2f} bits")

    # Plot 1: first 256 output samples
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(codes[:256], color="#4a9eff")
    axes[0].set_xlabel("sample")
    axes[0].set_ylabel("output code")
    axes[0].set_title("First 256 ADC output samples")
    axes[0].grid(True, linestyle=":")

    # Plot 2: output spectrum (locally computed)
    fft_freqs, fft_mags_db = _compute_spectrum(codes, fs)
    axes[1].plot(fft_freqs / 1e6, fft_mags_db, color="#f4a261")
    axes[1].set_xlabel("frequency (MHz)")
    axes[1].set_ylabel("magnitude (dB)")
    axes[1].set_title(
        f"Output spectrum, SNDR={metrics['SNDR']:.1f} dB, "
        f"ENOB={metrics['ENOB']:.2f} b")
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig("pipelined_adc_spectrum.png", dpi=150)
    print("\nWrote pipelined_adc_spectrum.png")


if __name__ == "__main__":
    main()
