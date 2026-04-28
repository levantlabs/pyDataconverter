"""
FlashADC demo — transfer functions, time-domain response, and visualisations.

Run::

    python examples/flashadc_demo.py

Builds two FlashADCs with different non-idealities, plots the static
transfer function, time-domain response to a 1 kHz sine, and runs the
static + animated visualisation helpers from
``pyDataconverter.utils.visualizations.visualize_FlashADC``.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.dataconverter import InputType


def main() -> None:
    # --- Static transfer function: 3-bit Flash with non-idealities ---
    adc = FlashADC(
        n_bits=3,
        v_ref=1.0,
        input_type=InputType.SINGLE,
        comparator_params={
            'noise_rms': 0.001,   # 1 mV RMS noise
            'hysteresis': 0.002,  # 2 mV hysteresis
        },
        offset_std=0.002,         # 2 mV comparator-offset stddev
        resistor_mismatch=0.01,   # 1 % resistor mismatch
    )

    v_in = np.linspace(0, 1, 1000)
    codes = [adc.convert(v) for v in v_in]

    plt.figure(figsize=(10, 6))
    plt.plot(v_in, codes, 'b.', markersize=1)
    plt.grid(True)
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Output Code')
    plt.title('Flash ADC Transfer Function')

    ideal_codes = np.floor(v_in * 2 ** adc.n_bits).astype(int)
    ideal_codes = np.clip(ideal_codes, 0, 2 ** adc.n_bits - 1)
    plt.plot(v_in, ideal_codes, 'r--', alpha=0.5, label='Ideal')
    plt.legend()
    plt.show()

    # --- Time-domain response to a 1 kHz sine ---
    t = np.linspace(0, 1e-3, 1000)  # 1 ms
    f = 1e3                         # 1 kHz
    v_in = 0.5 + 0.4 * np.sin(2 * np.pi * f * t)
    codes = [adc.convert(v) for v in v_in]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t * 1e3, v_in)
    plt.grid(True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Voltage (V)')
    plt.title('Flash ADC Time Domain Response')

    plt.subplot(2, 1, 2)
    plt.plot(t * 1e3, codes)
    plt.grid(True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Output Code')

    plt.tight_layout()
    plt.show()

    # --- 4-bit Flash for visualisation helpers ---
    adc = FlashADC(
        n_bits=4,
        v_ref=1.0,
        input_type=InputType.SINGLE,
        comparator_params={
            'offset': 0.001,      # 1 mV offset
            'noise_rms': 0.0005,  # 0.5 mV noise
        },
    )

    from pyDataconverter.utils.visualizations.visualize_FlashADC import (
        visualize_flash_adc, animate_flash_adc,
    )
    visualize_flash_adc(adc, input_voltage=0.4)

    t = np.linspace(0, 2 * np.pi, 50)
    input_voltages = 0.5 + 0.4 * np.sin(t)
    animate_flash_adc(adc, input_voltages, interval=0.1)


if __name__ == "__main__":
    main()
