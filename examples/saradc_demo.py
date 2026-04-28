"""
SARADC demo — transfer functions and a worked conversion trace.

Run::

    python examples/saradc_demo.py

Plots an ideal 4-bit single-ended SAR transfer function, prints a
cycle-by-cycle conversion trace for vin = 0.37 V, and plots a
4-bit differential SAR transfer function for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.dataconverter import InputType


def main() -> None:
    # --- Ideal 4-bit SAR, single-ended ---
    adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)
    v_in = np.linspace(0, 1, 1000)
    codes = [adc.convert(v) for v in v_in]

    plt.figure(figsize=(10, 4))
    plt.plot(v_in, codes, 'b.', markersize=1)
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Output Code')
    plt.title('Ideal 4-bit SAR ADC Transfer Function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Conversion trace ---
    trace = adc.convert_with_trace(0.37)
    print("SAR conversion trace for vin = 0.37 V:")
    print(f"  Sampled voltage : {trace['sampled_voltage']:.4f} V")
    for k, (v_dac, bit, reg) in enumerate(
        zip(trace['dac_voltages'], trace['bit_decisions'],
            trace['register_states'][1:])
    ):
        print(f"  Bit {adc.n_bits - 1 - k}: "
              f"v_dac={v_dac:.4f} V  decision={bit}  "
              f"register={reg:0{adc.n_bits}b}")
    print(f"  Final code: {trace['code']}")

    # --- Differential SAR ---
    adc_diff = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
    v_diff = np.linspace(-0.5, 0.5, 1000)
    codes_diff = [adc_diff.convert((v / 2, -v / 2)) for v in v_diff]

    plt.figure(figsize=(10, 4))
    plt.plot(v_diff, codes_diff, 'r.', markersize=1)
    plt.xlabel('Differential Input (V)')
    plt.ylabel('Output Code')
    plt.title('Ideal 4-bit Differential SAR ADC Transfer Function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
