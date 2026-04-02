"""
Voltage-Mode DAC Example
========================

Demonstrates three voltage-mode DAC architectures:

  1. ResistorStringDAC  — Kelvin divider; inherently monotonic
  2. R2RDAC             — R-2R ladder; compact but MSB-sensitive
  3. SegmentedResistorDAC — thermometer coarse + R-2R fine; best of both worlds

For each architecture we show:
  - Ideal (no mismatch) DNL and INL
  - With 1 % resistor mismatch (Monte Carlo, single run)

A single matplotlib figure compares all three INL curves.

Usage
-----
    python examples/voltage_mode_dac_example.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
from pyDataconverter.architectures.R2RDAC import R2RDAC
from pyDataconverter.architectures.SegmentedResistorDAC import SegmentedResistorDAC
from pyDataconverter.utils.metrics import calculate_dac_static_metrics


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_metrics(dac):
    """Return DNL and INL arrays for a DAC using endpoint INL method."""
    m = calculate_dac_static_metrics(dac, inl_method='endpoint')
    return m['DNL'], m['INL']


def _print_summary(label, dnl, inl):
    print(f'  {label}')
    print(f'    Max |DNL| = {np.max(np.abs(dnl)):.4f} LSB  '
          f'Max |INL| = {np.max(np.abs(inl)):.4f} LSB')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_bits     = 8
    v_ref      = 1.0
    r_mismatch = 0.01    # 1 % resistor mismatch (realistic silicon)
    n_therm    = 4       # thermometer bits for SegmentedResistorDAC
    seed       = 7       # fixed seed for reproducibility

    print(f'Voltage-Mode DAC Comparison — {n_bits}-bit, v_ref={v_ref} V')
    print(f'Resistor mismatch: {r_mismatch*100:.0f} %  (seed={seed})')
    print()

    # ------------------------------------------------------------------
    # 1. ResistorStringDAC
    # ------------------------------------------------------------------
    print('--- ResistorStringDAC ---')
    dac_rs_ideal   = ResistorStringDAC(n_bits=n_bits, v_ref=v_ref, r_mismatch=0.0)
    dac_rs_mismatch = ResistorStringDAC(n_bits=n_bits, v_ref=v_ref,
                                        r_mismatch=r_mismatch, seed=seed)

    dnl_rs_ideal, inl_rs_ideal     = _get_metrics(dac_rs_ideal)
    dnl_rs_mm,    inl_rs_mm        = _get_metrics(dac_rs_mismatch)

    _print_summary('Ideal', dnl_rs_ideal, inl_rs_ideal)
    _print_summary(f'Mismatch {r_mismatch*100:.0f}%', dnl_rs_mm, inl_rs_mm)

    # Spot check output voltages
    lsb = v_ref / 2 ** n_bits
    v0  = dac_rs_ideal.convert(0)
    v_fs = dac_rs_ideal.convert(2 ** n_bits - 1)
    print(f'  Ideal: code 0 → {v0:.6f} V,  code {2**n_bits-1} → {v_fs:.6f} V  '
          f'(expected {0:.6f} and {v_ref - lsb:.6f})')

    # ------------------------------------------------------------------
    # 2. R2RDAC
    # ------------------------------------------------------------------
    print('\n--- R2RDAC ---')
    dac_r2r_ideal   = R2RDAC(n_bits=n_bits, v_ref=v_ref, r_mismatch=0.0, r2_mismatch=0.0)
    dac_r2r_mismatch = R2RDAC(n_bits=n_bits, v_ref=v_ref,
                               r_mismatch=r_mismatch, r2_mismatch=r_mismatch, seed=seed)

    dnl_r2r_ideal, inl_r2r_ideal = _get_metrics(dac_r2r_ideal)
    dnl_r2r_mm,    inl_r2r_mm    = _get_metrics(dac_r2r_mismatch)

    _print_summary('Ideal', dnl_r2r_ideal, inl_r2r_ideal)
    _print_summary(f'Mismatch {r_mismatch*100:.0f}%', dnl_r2r_mm, inl_r2r_mm)

    v0_r2r  = dac_r2r_ideal.convert(0)
    vfs_r2r = dac_r2r_ideal.convert(2 ** n_bits - 1)
    print(f'  Ideal: code 0 → {v0_r2r:.6f} V,  code {2**n_bits-1} → {vfs_r2r:.6f} V')

    # ------------------------------------------------------------------
    # 3. SegmentedResistorDAC
    # ------------------------------------------------------------------
    print('\n--- SegmentedResistorDAC ---')
    dac_seg_ideal   = SegmentedResistorDAC(n_bits=n_bits, v_ref=v_ref, n_therm=n_therm,
                                            r_mismatch=0.0)
    dac_seg_mismatch = SegmentedResistorDAC(n_bits=n_bits, v_ref=v_ref, n_therm=n_therm,
                                             r_mismatch=r_mismatch, seed=seed)

    dnl_seg_ideal, inl_seg_ideal = _get_metrics(dac_seg_ideal)
    dnl_seg_mm,    inl_seg_mm    = _get_metrics(dac_seg_mismatch)

    _print_summary('Ideal', dnl_seg_ideal, inl_seg_ideal)
    _print_summary(f'Mismatch {r_mismatch*100:.0f}%  (n_therm={n_therm})', dnl_seg_mm, inl_seg_mm)

    # Transfer curve
    n_codes = 2 ** n_bits
    codes   = np.arange(n_codes)
    v_out_seg = np.array([dac_seg_mismatch.convert(int(c)) for c in codes])

    # ------------------------------------------------------------------
    # Figure: INL comparison + SegmentedDAC transfer curve
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'Voltage-Mode DAC Comparison — {n_bits}-bit, {r_mismatch*100:.0f}% resistor mismatch',
        fontweight='bold'
    )

    inl_x = np.arange(len(inl_rs_ideal))

    # Panel 1: INL comparison (ideal)
    axes[0].plot(inl_x, inl_rs_ideal,  color='steelblue', linewidth=1.4, label='ResistorString')
    axes[0].plot(inl_x, inl_r2r_ideal, color='tomato',    linewidth=1.4, label='R2R')
    axes[0].plot(inl_x, inl_seg_ideal, color='seagreen',  linewidth=1.4, label='Segmented')
    axes[0].axhline(0, color='black', linewidth=0.7, linestyle='--')
    axes[0].set_title('INL — Ideal (no mismatch)')
    axes[0].set_xlabel('Code')
    axes[0].set_ylabel('INL (LSB)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: INL comparison (with mismatch)
    axes[1].plot(inl_x, inl_rs_mm,  color='steelblue', linewidth=1.4,
                 label=f'ResistorString  max={np.max(np.abs(inl_rs_mm)):.2f} LSB')
    axes[1].plot(inl_x, inl_r2r_mm, color='tomato',    linewidth=1.4,
                 label=f'R2R  max={np.max(np.abs(inl_r2r_mm)):.2f} LSB')
    axes[1].plot(inl_x, inl_seg_mm, color='seagreen',  linewidth=1.4,
                 label=f'Segmented  max={np.max(np.abs(inl_seg_mm)):.2f} LSB')
    axes[1].axhline(0, color='black', linewidth=0.7, linestyle='--')
    axes[1].set_title(f'INL — {r_mismatch*100:.0f}% resistor mismatch')
    axes[1].set_xlabel('Code')
    axes[1].set_ylabel('INL (LSB)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: SegmentedResistorDAC transfer curve with mismatch
    ideal_v = codes / n_codes * v_ref
    axes[2].step(codes, v_out_seg, where='post', color='seagreen', linewidth=1.4,
                 label=f'Segmented (n_therm={n_therm})')
    axes[2].plot(codes, ideal_v, color='black', linewidth=0.8, linestyle='--',
                 alpha=0.6, label='Ideal linear')
    axes[2].set_title(f'SegmentedResistorDAC Transfer Curve\n(n_therm={n_therm}, mismatch={r_mismatch*100:.0f}%)')
    axes[2].set_xlabel('Code')
    axes[2].set_ylabel('Output Voltage (V)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('voltage_mode_dac_demo.png', dpi=100, bbox_inches='tight')
    print('\n  Figure saved: voltage_mode_dac_demo.png')
    plt.close(fig)
    print('\nDone.')


if __name__ == '__main__':
    main()
