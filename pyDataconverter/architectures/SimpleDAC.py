"""
Simple DAC Implementation
========================

This module provides a simple, ideal DAC implementation.

Classes:
SimpleDAC: Basic DAC implementation with ideal conversion

Version History:
---------------
1.0.0 (2025-02-06):
- Initial release
- Basic conversion implementation
"""

from pyDataconverter.dataconverter import DACBase, OutputType
from typing import Union, Tuple
import numpy as np


class SimpleDAC(DACBase):
    """
    Simple DAC implementation with ideal conversion.

    This DAC performs ideal conversion without any non-idealities.
    Useful as a reference implementation or for basic simulation.

    Attributes:
        Inherits all attributes from DACBase
    """

    def _convert_input(self, digital_input: int) -> Union[float, Tuple[float, float]]:
        """
        Perform ideal conversion of input code to voltage.

        Args:
            digital_input: Pre-validated input code

        Returns:
            float or tuple: Output voltage(s)
                - Single-ended: returns float voltage
                - Differential: returns tuple of (v_pos, v_neg)
        """
        # Calculate normalized voltage (0 to 1)
        voltage = digital_input * self.lsb

        if self.output_type == OutputType.SINGLE:
            return voltage
        else:  # DIFFERENTIAL
            # For differential, center around v_ref/2
            # Full range goes from -v_ref/2 to +v_ref/2
            v_diff = voltage - self.v_ref / 2
            v_pos = v_diff / 2 + self.v_ref / 2
            v_neg = -v_diff / 2 + self.v_ref / 2
            return (v_pos, v_neg)


# Example usage if run directly
if __name__ == "__main__":
    # Create DAC instance
    dac = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.SINGLE)

    # Test some conversions
    print(f"Zero-scale conversion: {dac.convert(0)}")
    print(f"Mid-scale conversion: {dac.convert(2048)}")  # 2^11 for 12-bit
    print(f"Full-scale conversion: {dac.convert(4095)}")  # 2^12 - 1

    # Test differential mode
    dac_diff = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.DIFFERENTIAL)
    v_pos, v_neg = dac_diff.convert(2048)  # Mid-scale
    print(f"Differential mid-scale: v_pos={v_pos:.3f}V, v_neg={v_neg:.3f}V, diff={v_pos - v_neg:.3f}V")