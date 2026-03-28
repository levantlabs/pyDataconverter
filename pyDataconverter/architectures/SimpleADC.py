"""
Simple ADC Implementation
========================

This module provides a simple ADC implementation with optional first-order
non-idealities.

Classes:
    SimpleADC: ADC with ideal quantization and optional thermal noise,
               offset, gain error, and aperture jitter.

Version History:
---------------
1.0.0 (2025-02-01):
    - Initial release
    - Basic quantization implementation
1.1.0 (2026-03-22):
    - Added QuantizationMode support (FLOOR and SYMMETRIC)
    - Applied quantization mode to both single-ended and differential inputs
    - Fixed differential quantization formula
1.2.0 (2026-03-23):
    - Added optional non-idealities: thermal noise, offset, gain error,
      aperture jitter
    - Overrode convert() to accept dvdt for aperture jitter modelling
"""

from pyDataconverter.dataconverter import ADCBase, InputType, QuantizationMode
from typing import Union, Tuple, Optional
import numpy as np


class SimpleADC(ADCBase):
    """
    ADC with ideal quantization and optional first-order non-idealities.

    All non-ideality parameters default to 0 (disabled). Set any of them to
    model a realistic converter.

    Attributes:
        quant_mode (QuantizationMode): FLOOR or SYMMETRIC quantization model.
        noise_rms (float): Input-referred RMS thermal noise voltage (V).
                           Adds N(0, noise_rms) to the input each sample.
        offset (float): Input-referred DC offset voltage (V).
                        Shifts every code by this fixed amount.
        gain_error (float): Fractional gain error (dimensionless).
                            0.01 = +1 %. Scales the input as v*(1+gain_error).
        t_jitter (float): RMS aperture jitter (s).
                          Requires dvdt to be passed to convert(); ignored
                          otherwise.  Adds dvdt * N(0, t_jitter) to input.
    """

    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 input_type: InputType = InputType.DIFFERENTIAL,
                 quant_mode: QuantizationMode = QuantizationMode.FLOOR,
                 noise_rms: float = 0.0,
                 offset: float = 0.0,
                 gain_error: float = 0.0,
                 t_jitter: float = 0.0):
        super().__init__(n_bits, v_ref, input_type)

        if not isinstance(quant_mode, QuantizationMode):
            raise TypeError("quant_mode must be a QuantizationMode enum")
        if noise_rms < 0:
            raise ValueError("noise_rms must be >= 0")
        if t_jitter < 0:
            raise ValueError("t_jitter must be >= 0")

        self.quant_mode  = quant_mode
        self.noise_rms   = noise_rms
        self.offset      = offset
        self.gain_error  = gain_error
        self.t_jitter    = t_jitter
        # self._dvdt is inherited from ADCBase

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def convert(self,
                vin: Union[float, Tuple[float, float]],
                dvdt: float = 0.0) -> int:
        """
        Convert one analog sample to a digital code.

        Args:
            vin:   Voltage (single-ended) or (v_pos, v_neg) tuple (differential).
            dvdt:  Signal slope at the sampling instant (V/s). Only used when
                   t_jitter > 0.  Pass this for accurate aperture jitter
                   modelling; if omitted the jitter term is zero.

        Returns:
            int: Output code in [0, 2^n_bits - 1].
        """
        return super().convert(vin, dvdt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_nonidealities(self, v: float) -> float:
        """
        Apply all enabled non-idealities to an input voltage, in order:
        gain error → offset → thermal noise → aperture jitter.

        Args:
            v: Input voltage (V), already reduced to a single scalar
               (differential voltage for diff inputs).

        Returns:
            float: Modified voltage.
        """
        if self.gain_error:
            v = v * (1.0 + self.gain_error)
        if self.offset:
            v = v + self.offset
        if self.noise_rms:
            v = v + np.random.normal(0.0, self.noise_rms)
        if self.t_jitter and self._dvdt:
            v = v + self._dvdt * np.random.normal(0.0, self.t_jitter)
        return v

    def _quantize(self, voltage: float, v_min: float, v_max: float) -> int:
        """
        Apply the selected quantization mode to a voltage in [v_min, v_max].

        FLOOR mode:  code = floor(v * 2^N / v_range)
            All bins are exactly 1 LSB = v_range / 2^N wide.

        SYMMETRIC mode: code = floor(v * (2^N - 1) / v_range + 0.5)
            End bins (code 0 and 2^N-1) are LSB/2 wide; all others are 1 LSB.
            LSB = v_range / (2^N - 1).
        """
        if v_max <= v_min:
            raise ValueError(f"v_max ({v_max}) must be greater than v_min ({v_min})")
        v_range = v_max - v_min
        v = np.clip(voltage, v_min, v_max) - v_min  # shift to [0, v_range]

        if self.quant_mode == QuantizationMode.FLOOR:
            code = int(np.floor(v * 2**self.n_bits / v_range))
        else:  # SYMMETRIC
            code = int(np.floor(v * (2**self.n_bits - 1) / v_range + 0.5))

        return int(np.clip(code, 0, 2**self.n_bits - 1))

    def _convert_input(self, analog_input: Union[float, Tuple[float, float]]) -> int:
        """
        Apply non-idealities then quantize.

        Args:
            analog_input: Pre-validated voltage or (v_pos, v_neg) tuple.

        Returns:
            int: Digital output code in [0, 2^n_bits - 1].
        """
        if self.input_type == InputType.SINGLE:
            v = self._apply_nonidealities(float(analog_input))
            return self._quantize(v, v_min=0, v_max=self.v_ref)
        else:  # DIFFERENTIAL
            v_pos, v_neg = analog_input
            vdiff = self._apply_nonidealities(v_pos - v_neg)
            return self._quantize(vdiff, v_min=-self.v_ref / 2, v_max=self.v_ref / 2)

    def __repr__(self) -> str:
        parts = [
            f"n_bits={self.n_bits}",
            f"v_ref={self.v_ref}",
            f"input_type={self.input_type.name}",
            f"quant_mode={self.quant_mode.name}",
        ]
        if self.noise_rms:
            parts.append(f"noise_rms={self.noise_rms}")
        if self.offset:
            parts.append(f"offset={self.offset}")
        if self.gain_error:
            parts.append(f"gain_error={self.gain_error}")
        if self.t_jitter:
            parts.append(f"t_jitter={self.t_jitter}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# Example usage if run directly
if __name__ == "__main__":
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
    import math
    dvdt = A * 2 * math.pi * f  # peak slope of a 10 kHz sine at zero-crossing
    print("\nWith aperture jitter (10 kHz sine, peak slope):")
    print(f"  {adc_jitter.convert(0.0, dvdt=dvdt)}")
