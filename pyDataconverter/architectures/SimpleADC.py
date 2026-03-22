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
1.1.0 (2026-03-22):
    - Added QuantizationMode support (FLOOR and SYMMETRIC)
    - Applied quantization mode to both single-ended and differential inputs
    - Fixed differential quantization formula
"""

from pyDataconverter.dataconverter import ADCBase, InputType, QuantizationMode
from typing import Union, Tuple
import numpy as np

class SimpleADC(ADCBase):
    """
    Simple ADC implementation with ideal quantization.

    This ADC performs ideal quantization without any non-idealities.
    Useful as a reference implementation or for basic simulation.

    Attributes:
        quant_mode (QuantizationMode): Selects the quantization model.
            FLOOR (default): Standard hardware ADC model. Equal-width bins,
                LSB = v_ref / 2^N. Matches IEEE 1241 and industry datasheets.
            SYMMETRIC: DSP model. Half-width bins at code 0 and code 2^N-1,
                full-width elsewhere. LSB = v_ref / (2^N - 1). Zero-mean
                quantization error, useful for noise analysis.
        Inherits all other attributes from ADCBase.
    """

    def __init__(self, n_bits: int, v_ref: float = 1.0,
                 input_type: InputType = InputType.DIFFERENTIAL,
                 quant_mode: QuantizationMode = QuantizationMode.FLOOR):
        super().__init__(n_bits, v_ref, input_type)
        if not isinstance(quant_mode, QuantizationMode):
            raise TypeError("quant_mode must be a QuantizationMode enum")
        self.quant_mode = quant_mode

    def _quantize(self, voltage: float, v_min: float, v_max: float) -> int:
        """
        Apply the selected quantization mode to a voltage in [v_min, v_max].

        FLOOR mode:  code = floor(v * 2^N / v_range)
            All bins are exactly 1 LSB = v_range / 2^N wide.

        SYMMETRIC mode: code = floor(v * (2^N - 1) / v_range + 0.5)
            End bins (code 0 and 2^N-1) are LSB/2 wide; all others are 1 LSB.
            LSB = v_range / (2^N - 1).
        """
        v_range = v_max - v_min
        v = np.clip(voltage, v_min, v_max) - v_min  # shift to [0, v_range]

        if self.quant_mode == QuantizationMode.FLOOR:
            code = int(v * 2**self.n_bits / v_range)
        else:  # SYMMETRIC
            code = int(v * (2**self.n_bits - 1) / v_range + 0.5)

        return int(np.clip(code, 0, 2**self.n_bits - 1))

    def _convert_input(self, analog_input: Union[float, Tuple[float, float]]) -> int:
        """
        Perform ideal quantization based on input type.

        Args:
            analog_input: Single voltage for single-ended, or (v_pos, v_neg)
                          tuple for differential.

        Returns:
            int: Digital output code in range [0, 2^n_bits - 1]
        """
        if self.input_type == InputType.SINGLE:
            return self._quantize(analog_input, v_min=0, v_max=self.v_ref)

        else:  # DIFFERENTIAL
            v_pos, v_neg = analog_input
            vdiff = v_pos - v_neg
            # Differential input range: [-v_ref/2, +v_ref/2]
            return self._quantize(vdiff, v_min=-self.v_ref / 2, v_max=self.v_ref / 2)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n_bits={self.n_bits}, v_ref={self.v_ref}, "
                f"input_type={self.input_type.name}, quant_mode={self.quant_mode.name})")


# Example usage if run directly
if __name__ == "__main__":
    # --- Single-ended, FLOOR mode (default) ---
    adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)
    print("Single-ended FLOOR:")
    print(f"  Zero-scale:  {adc.convert(0.0)}")    # expect 0
    print(f"  Mid-scale:   {adc.convert(0.5)}")    # expect 2048
    print(f"  Full-scale:  {adc.convert(1.0)}")    # expect 4095

    # --- Single-ended, SYMMETRIC mode ---
    adc_sym = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
                        quant_mode=QuantizationMode.SYMMETRIC)
    print("\nSingle-ended SYMMETRIC:")
    print(f"  Zero-scale:  {adc_sym.convert(0.0)}")    # expect 0
    print(f"  Mid-scale:   {adc_sym.convert(0.5)}")    # expect 2048
    print(f"  Full-scale:  {adc_sym.convert(1.0)}")    # expect 4095

    # --- Differential, FLOOR mode ---
    adc_diff = SimpleADC(n_bits=12, v_ref=2.0, input_type=InputType.DIFFERENTIAL)
    print("\nDifferential FLOOR:")
    print(f"  Mid-scale  (0.5, 0.5): {adc_diff.convert((0.5, 0.5))}")   # expect 2048
    print(f"  Near-low   (0.02, 0.98): {adc_diff.convert((0.02, 0.98))}")
    print(f"  Near-high  (0.98, 0.02): {adc_diff.convert((0.98, 0.02))}")
