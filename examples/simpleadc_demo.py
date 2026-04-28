"""
SimpleADC demo — ideal, noisy, and jittered conversion examples.

Run::

    python examples/simpleadc_demo.py

Prints the output code for zero-/mid-/full-scale inputs on an ideal
12-bit FLOOR ADC, then re-runs midscale with thermal noise + offset +
gain error, then with aperture jitter at the peak slope of a 10 kHz
sine.
"""

import math

from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType


def main() -> None:
    # --- Ideal (default) ---
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)
    print("Ideal FLOOR:")
    print(f"  Zero-scale:  {adc.convert(0.0)}")    # expect 0
    print(f"  Mid-scale:   {adc.convert(0.5)}")    # expect 2048
    print(f"  Full-scale:  {adc.convert(1.0)}")    # expect 4095

    # --- With noise and offset ---
    adc_noisy = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
                          noise_rms=1e-4, offset=5e-3, gain_error=0.001)
    print("\nWith noise + offset + gain error:")
    print(f"  Mid-scale: {adc_noisy.convert(0.5)}")

    # --- Aperture jitter ---
    adc_jitter = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
                           t_jitter=1e-12)
    f, A = 10e3, 0.5
    dvdt = A * 2 * math.pi * f  # peak slope of a 10 kHz sine at zero-crossing
    print("\nWith aperture jitter (10 kHz sine, peak slope):")
    print(f"  {adc_jitter.convert(0.0, dvdt=dvdt)}")


if __name__ == "__main__":
    main()
