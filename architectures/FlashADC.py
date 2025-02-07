"""
Flash ADC Module
===============

This module provides a Flash ADC implementation with configurable non-idealities.

Classes:
    FlashADC: Flash ADC implementation inheriting from ADCBase

Version History:
---------------
1.0.0 (2024-02-07):
    - Initial release
    - Basic Flash ADC implementation
    - Integration with Comparator component
    - Support for offset, noise, and resistor mismatch

Notes:
------
The Flash ADC model includes:
    - Configurable number of bits
    - Individual comparator non-idealities
    - Resistor ladder mismatch
    - Reference noise modeling
Future versions may include:
    - Temperature effects
    - Dynamic timing effects
    - Power consumption modeling
"""

from typing import Optional, Type
import numpy as np
from pyDataconverter.dataconverter import ADCBase, InputType
from pyDataconverter.components.comparator import Comparator


class FlashADC(ADCBase):
    """
    Flash ADC implementation with configurable non-idealities.

    Attributes:
        Inherits all attributes from ADCBase, plus:
        n_comparators: Number of comparators (2^n_bits - 1)
        comparators: List of comparator instances
        reference_voltages: Reference voltages from resistor ladder
        reference_noise: RMS noise of reference ladder
    """

    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 input_type: InputType = InputType.SINGLE,
                 comparator_type: Type[Comparator] = Comparator,
                 comparator_params: Optional[dict] = None,
                 offset_std: float = 0.0,
                 reference_noise: float = 0.0,
                 resistor_mismatch: float = 0.0):
        """
        Initialize Flash ADC.

        Args:
            n_bits: Resolution in bits
            v_ref: Reference voltage
            input_type: Input type (SINGLE or DIFFERENTIAL)
            comparator_type: Comparator class to use
            comparator_params: Parameters for comparator initialization
            offset_std: Standard deviation of comparator offsets
            reference_noise: RMS noise of reference ladder
            resistor_mismatch: Standard deviation of resistor mismatch
        """
        # Initialize parent class
        super().__init__(n_bits, v_ref, input_type)

        # Number of comparators
        self.n_comparators = 2 ** n_bits - 1

        # Initialize comparator parameters
        if comparator_params is None:
            comparator_params = {}

        # Generate random offsets if specified
        if offset_std > 0:
            offsets = np.random.normal(0, offset_std, self.n_comparators)
        else:
            offsets = np.zeros(self.n_comparators)

        # Create comparators
        self.comparators = []
        for i in range(self.n_comparators):
            params = comparator_params.copy()
            params['offset'] = offsets[i]
            self.comparators.append(comparator_type(**params))

        # Generate reference voltages with mismatch
        ideal_refs = np.linspace(0, v_ref, self.n_comparators + 2)[1:-1]
        if resistor_mismatch > 0:
            mismatch = np.random.normal(0, resistor_mismatch, self.n_comparators)
            self.reference_voltages = ideal_refs * (1 + mismatch)
        else:
            self.reference_voltages = ideal_refs

        self.reference_noise = reference_noise

    def _convert_input(self, analog_input: float) -> int:
        """
        Convert analog input to digital output.

        Args:
            analog_input: Input voltage

        Returns:
            Digital output code
        """
        # Add reference noise if specified
        if self.reference_noise > 0:
            ref_noise = np.random.normal(0, self.reference_noise, self.n_comparators)
            comp_refs = self.reference_voltages + ref_noise
        else:
            comp_refs = self.reference_voltages

        # Run comparisons
        thermometer_code = np.array([
            comp.compare(analog_input, ref)
            for comp, ref in zip(self.comparators, comp_refs)
        ])

        # Convert thermometer code to binary
        code = np.sum(thermometer_code)

        # Clip to valid range
        return np.clip(code, 0, 2 ** self.n_bits - 1)

    def reset(self):
        """Reset all comparators to initial state."""
        for comp in self.comparators:
            comp.reset()





if __name__ == "__main__":
    """Example usage of the Flash ADC"""
    import matplotlib.pyplot as plt

    # Create a 3-bit Flash ADC with some non-idealities
    adc = FlashADC(
        n_bits=3,
        v_ref=1.0,
        comparator_params={
            'noise_rms': 0.001,  # 1mV RMS noise
            'hysteresis': 0.002  # 2mV hysteresis
        },
        offset_std=0.002,  # 2mV offset standard deviation
        resistor_mismatch=0.01  # 1% resistor mismatch
    )

    # Test with ramp input
    v_in = np.linspace(0, 1, 1000)
    codes = [adc.convert(v) for v in v_in]

    # Plot transfer function
    plt.figure(figsize=(10, 6))
    plt.plot(v_in, codes, 'b.', markersize=1)
    plt.grid(True)
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Output Code')
    plt.title('Flash ADC Transfer Function')

    # Add ideal transfer function
    ideal_codes = np.floor(v_in * 2 ** adc.n_bits).astype(int)
    ideal_codes = np.clip(ideal_codes, 0, 2 ** adc.n_bits - 1)
    plt.plot(v_in, ideal_codes, 'r--', alpha=0.5, label='Ideal')
    plt.legend()

    plt.show()

    # Test with sine wave input
    t = np.linspace(0, 1e-3, 1000)  # 1ms
    f = 1e3  # 1kHz
    v_in = 0.5 + 0.4 * np.sin(2 * np.pi * f * t)  # 0.5V offset, 0.4V amplitude
    codes = [adc.convert(v) for v in v_in]

    # Plot time domain behavior
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

    # Create a 3-bit Flash ADC with some non-idealities
    adc = FlashADC(
        n_bits=4,
        v_ref=1.0,
        comparator_params={
            'offset': 0.001,  # 1mV offset
            'noise_rms': 0.0005  # 0.5mV noise
        }
    )





    # Static visualization
    from pyDataconverter.utils.visualizations.visualize_FlashADC import visualize_flash_adc, animate_flash_adc
    visualize_flash_adc(adc, input_voltage=0.4)

    # Animation example
    t = np.linspace(0, 2 * np.pi, 50)
    input_voltages = 0.5 + 0.4 * np.sin(t)  # Sine wave centered at 0.5V
    animate_flash_adc(adc, input_voltages, interval=0.1)