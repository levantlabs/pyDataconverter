"""
SimpleDAC demo — ideal, noisy, and differential conversion examples.

Run::

    python examples/simpledac_demo.py

Prints zero-/mid-/full-scale outputs for an ideal 12-bit DAC, then a
midscale value with noise + offset + gain error, and finally a
differential midscale conversion showing v_pos / v_neg / v_diff.
"""

from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType


def main() -> None:
    # --- Ideal (default) ---
    dac = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.SINGLE)
    print("Ideal:")
    print(f"  Zero-scale:  {dac.convert(0)}")
    print(f"  Mid-scale:   {dac.convert(2048)}")
    print(f"  Full-scale:  {dac.convert(4095)}")

    # --- With offset and gain error ---
    dac_noisy = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.SINGLE,
                          noise_rms=1e-4, offset=5e-3, gain_error=0.001)
    print("\nWith noise + offset + gain error:")
    print(f"  Mid-scale: {dac_noisy.convert(2048)}")

    # --- Differential ---
    dac_diff = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.DIFFERENTIAL)
    v_pos, v_neg = dac_diff.convert(2048)
    print(f"\nDifferential mid-scale: v_pos={v_pos:.3f}V, "
          f"v_neg={v_neg:.3f}V, diff={v_pos - v_neg:.3f}V")


if __name__ == "__main__":
    main()
