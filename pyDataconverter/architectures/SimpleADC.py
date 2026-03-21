"""
Simple ADC Implementation
========================

This module provides a simple, ideal ADC implementation.

Classes:
    SimpleADC: Basic ADC implementation with ideal quantization

Version History:
---------------
1.0.0 (2025-02-01):
    - Initial release
    - Basic quantization implementation
"""

from pyDataconverter.dataconverter import ADCBase, InputType
from typing import Union, Tuple
import numpy as np

class SimpleADC(ADCBase):
    """
    Simple ADC implementation with ideal quantization.

    This ADC performs ideal quantization without any non-idealities.
    Useful as a reference implementation or for basic simulation.

    Attributes:
        Inherits all attributes from ADCBase
    """

    def _convert_input(self, analog_input: Union[float, Tuple[float, float]]) -> int:
        """
        Perform ideal quantization based on input type.

        Args:
            analog_input: Single voltage or tuple of voltages for differential

        Returns:
            int: Digital output code
        """
        if self.input_type == InputType.SINGLE:
            voltage = analog_input
            # Clip to reference range
            voltage = np.clip(voltage, 0, self.v_ref)

            # Ideal quantization for single-ended
            code = int(voltage * (2**self.n_bits - 1) / self.v_ref)

        else:  # DIFFERENTIAL
            v_pos, v_neg = analog_input
            # Different processing for differential inputs
            # Example: might handle common mode differently
            vdiff = v_pos - v_neg
            vdiff = np.clip(vdiff, -self.v_ref/2, self.v_ref/2)

            # Ideal quantization for differential
            mid_scale = 2**(self.n_bits - 1)
            voltage_scale = vdiff / (self.v_ref/2)
            code = int(mid_scale * (1 + voltage_scale))

        # Clip to valid code range
        return np.clip(code, 0, 2**self.n_bits - 1)

# Example usage if run directly
if __name__ == "__main__":
    # Create ADC instance
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)

    # Test some conversions
    print(f"Mid-scale conversion: {adc.convert(0.5)}")
    print(f"Full-scale conversion: {adc.convert(1.0)}")
    print(f"Zero-scale conversion: {adc.convert(0.0)}")

    # Test differential mode
    adc_diff = SimpleADC(n_bits=12, v_ref=2.0, input_type=InputType.DIFFERENTIAL)
    print(f"Differential conversion 1: {adc_diff.convert((0.5, 0.5))}")  # Should give mid-scale
    print(f"Differential conversion 2: {adc_diff.convert((0.02, 0.98))}")  # Near low scale
    print(f"Differential conversion 3: {adc_diff.convert((0.98, 0.02))}")  # Near high scale