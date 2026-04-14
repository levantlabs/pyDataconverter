"""
Simple DAC Implementation
========================

This module provides a simple DAC implementation with optional first-order
non-idealities.

Classes:
    SimpleDAC: DAC with ideal conversion and optional noise, offset, and
               gain error.

Version History:
---------------
1.0.0 (2025-02-06):
    - Initial release
    - Basic conversion implementation
1.1.0 (2026-03-24):
    - Added optional non-idealities: noise_rms, offset, gain_error
    - Added _apply_nonidealities() method mirroring SimpleADC pattern
"""

from pyDataconverter.dataconverter import DACBase, OutputType
from typing import Union, Tuple
import numpy as np


class SimpleDAC(DACBase):
    """
    DAC with ideal conversion and optional first-order non-idealities.

    All non-ideality parameters default to 0 (disabled). Set any of them to
    model a realistic converter.

    Unlike SimpleADC (which converts one sample at a time and has no time
    axis), SimpleDAC supports sequence output via convert_sequence(), which
    requires a sample rate and optional oversampling factor.  These are set
    at construction because they define the DAC operating point.

    Attributes:
        noise_rms (float):   Output-referred RMS noise voltage (V).
                             Adds N(0, noise_rms) to the output each conversion.
        offset (float):      Output DC offset voltage (V).
                             Shifts every output by this fixed amount.
        gain_error (float):  Fractional gain error (dimensionless).
                             0.01 = +1 %. Scales the output as v*(1+gain_error).
        fs (float):          DAC update rate in Hz. Used by convert_sequence()
                             to compute the time axis. Default 1.0 Hz.
        oversample (int):    Number of output samples per input code (zero-order
                             hold factor). Used by convert_sequence() to
                             upsample the output waveform. Default 1 (no
                             oversampling).
    """

    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 output_type: OutputType = OutputType.SINGLE,
                 noise_rms: float = 0.0,
                 offset: float = 0.0,
                 gain_error: float = 0.0,
                 fs: float = 1.0,
                 oversample: int = 1,
                 n_levels: int = None):
        super().__init__(n_bits, v_ref, output_type, n_levels=n_levels)

        if noise_rms < 0:
            raise ValueError("noise_rms must be >= 0")
        if oversample < 1:
            raise ValueError("oversample must be >= 1")

        self.noise_rms  = noise_rms
        self.offset     = offset
        self.gain_error = gain_error
        self.fs         = fs
        self.oversample = oversample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_nonidealities(self, v: float) -> float:
        """
        Apply all enabled non-idealities to an output voltage, in order:
        gain error -> offset -> noise.

        Args:
            v: Ideal output voltage (V).

        Returns:
            float: Modified voltage.
        """
        if self.gain_error:
            v = v * (1.0 + self.gain_error)
        if self.offset:
            v = v + self.offset
        if self.noise_rms:
            v = v + np.random.normal(0.0, self.noise_rms)
        return v

    def _convert_input(self, digital_input: int) -> Union[float, Tuple[float, float]]:
        """
        Compute ideal voltage, apply non-idealities, then format output.

        Args:
            digital_input: Pre-validated input code

        Returns:
            float or tuple: Output voltage(s)
                - Single-ended: returns float voltage
                - Differential: returns tuple of (v_pos, v_neg)
        """
        # Calculate ideal voltage
        voltage = digital_input * self.lsb

        # Apply non-idealities before single/differential split
        voltage = self._apply_nonidealities(voltage)

        if self.output_type == OutputType.SINGLE:
            return voltage
        else:  # DIFFERENTIAL
            # For differential, center around v_ref/2
            # Full range goes from -v_ref/2 to +v_ref/2
            v_diff = 2 * voltage - self.v_ref
            v_pos = v_diff / 2 + self.v_ref / 2
            v_neg = -v_diff / 2 + self.v_ref / 2
            return (v_pos, v_neg)

    def convert_sequence(self, codes: np.ndarray) -> Union[
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Convert an array of digital codes to a time-domain ZOH waveform.

        Args:
            codes: Array of integer DAC codes.

        Returns:
            Single-ended: (t, voltages)
            Differential:  (t, v_pos, v_neg)
        """
        max_code = (1 << self.n_bits) - 1
        codes = np.clip(codes, 0, max_code)

        voltages = codes.astype(float) * self.lsb

        if self.gain_error:
            voltages *= (1.0 + self.gain_error)
        if self.offset:
            voltages += self.offset

        voltages = np.repeat(voltages, self.oversample)

        if self.noise_rms:
            voltages = voltages + np.random.normal(0.0, self.noise_rms, len(voltages))

        t = np.arange(len(voltages)) / (self.fs * self.oversample)

        if self.output_type == OutputType.SINGLE:
            return (t, voltages)
        else:
            v_diff = 2 * voltages - self.v_ref
            v_pos = v_diff / 2 + self.v_ref / 2
            v_neg = -v_diff / 2 + self.v_ref / 2
            return (t, v_pos, v_neg)

    def __repr__(self) -> str:
        parts = [
            f"n_bits={self.n_bits}",
            f"v_ref={self.v_ref}",
            f"output_type={self.output_type.name}",
        ]
        if self.noise_rms:
            parts.append(f"noise_rms={self.noise_rms}")
        if self.offset:
            parts.append(f"offset={self.offset}")
        if self.gain_error:
            parts.append(f"gain_error={self.gain_error}")
        if self.fs != 1.0:
            parts.append(f"fs={self.fs}")
        if self.oversample != 1:
            parts.append(f"oversample={self.oversample}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# Example usage if run directly
if __name__ == "__main__":
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
    print(f"\nDifferential mid-scale: v_pos={v_pos:.3f}V, v_neg={v_neg:.3f}V, diff={v_pos - v_neg:.3f}V")
