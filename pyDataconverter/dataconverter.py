"""
Data Converter Base Classes
===========================

This module provides interfaces for both ADC and DAC implementations.

Classes:
    ADCBase: abstract class for all ADC implementations
    DACBase: abstract class for all DAC implementations

Version History:
2025-01-31: First pass wrapper
2025-02-06: Added DACBase abstract class
2026-03-22: Added QuantizationMode enum
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Tuple
import numpy as np


class InputType(Enum):
    """Defines differential or single ended inputs for ADCs"""
    SINGLE = 'single'
    DIFFERENTIAL='differential'


class QuantizationMode(Enum):
    """
    Defines the quantization model used by an ADC.

    FLOOR:
        Standard hardware ADC model, consistent with IEEE 1241 and industry
        datasheets (Analog Devices, TI, etc.).
        - LSB = v_ref / 2^N
        - All code bins are exactly 1 LSB wide
        - Quantization error ranges from 0 to +LSB (always positive bias)
        - Formula: code = floor(vin * 2^N / v_ref)

    SYMMETRIC:
        DSP / signal processing model. Suitable for quantization noise analysis
        where zero-mean error is desired.
        - LSB = v_ref / (2^N - 1)
        - The first (code 0) and last (code 2^N-1) bins are half-width (LSB/2)
        - All middle bins are exactly 1 LSB wide
        - Quantization error is symmetric: -LSB/2 to +LSB/2 (zero mean)
        - Formula: code = floor(vin * (2^N - 1) / v_ref + 0.5)
    """
    FLOOR = 'floor'
    SYMMETRIC = 'symmetric'

class ADCBase(ABC):
    """Abstract base class for all ADC architectures"""

    def __init__(self, n_bits: int, v_ref: float=1.0, input_type: InputType=InputType.DIFFERENTIAL):
        if not isinstance(n_bits, int):
            raise TypeError('n_bits must be an integer.')
        if n_bits < 1 or n_bits > 32:
            raise ValueError('n_bits must be between 1 and 32.')
        self.n_bits = n_bits
        # Validate v_ref
        if not isinstance(v_ref, (int, float)):
            raise TypeError("v_ref must be a number")
        if v_ref <= 0:
            raise ValueError("v_ref must be positive")

        self.v_ref = v_ref
        if not isinstance(input_type, InputType):
                raise TypeError("input_type must be of an InputType enum")
        self.input_type = input_type
        self._dvdt = 0.0

    def _validate_input(self, vin: Union[float, Tuple[float, float]]) -> None:
        """Validate vin against the configured input_type."""
        if self.input_type == InputType.SINGLE:
            if not isinstance(vin, (int, float)):
                raise TypeError("Single-ended input must be a number.")
        elif self.input_type == InputType.DIFFERENTIAL:
            if not isinstance(vin, tuple) or len(vin) != 2:
                raise TypeError("Differential input must be a tuple of (positive, negative).")

    def convert(self, vin: Union[float, Tuple[float, float]], dvdt: float = 0.0):
        """Convert analog input to digital output
        Can either be a single value, if single ended, or a tuple if differential.
        Check input type validation before running conversion.

        Args:
            vin:  Voltage (single-ended) or (v_pos, v_neg) tuple (differential).
            dvdt: Signal slope at the sampling instant (V/s). Used by
                  subclasses that model aperture jitter."""
        self._validate_input(vin)
        self._dvdt = float(dvdt)
        return self._convert_input(vin) #Pass this on to a abstract function

    @abstractmethod
    def _convert_input(self, vin:  Union[float, Tuple[float, float]]):
        "Architecture specific conversion. "
        pass

    def __repr__(self) -> str:
        """String representation of the ADC"""
        return f"{self.__class__.__name__}(n_bits={self.n_bits}, v_ref={self.v_ref}, input_type={self.input_type.name})"





class OutputType(Enum):
    """Defines single or differential output for DAC"""
    SINGLE = 'single'
    DIFFERENTIAL = 'differential'


class DACBase(ABC):
    """
    Abstract base class for DAC implementations.

    Attributes:
        n_bits (int): DAC resolution
        v_ref (float): Reference voltage
        output_type (OutputType): Single-ended or differential
        lsb (float): Least significant bit size
    """

    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 output_type: OutputType = OutputType.SINGLE,
                 n_levels: int = None):
        # Validate n_bits
        if not isinstance(n_bits, int):
            raise TypeError("n_bits must be an integer")
        if n_bits < 1 or n_bits > 32:
            raise ValueError("n_bits must be between 1 and 32")

        # Validate v_ref
        if not isinstance(v_ref, (int, float)):
            raise TypeError("v_ref must be a number")
        if v_ref <= 0:
            raise ValueError("v_ref must be positive")

        # Validate output_type
        if not isinstance(output_type, OutputType):
            raise TypeError("output_type must be an OutputType enum")

        # Resolve n_levels: explicit override wins, else default to 2^n_bits.
        if n_levels is None:
            n_levels = 2 ** n_bits
        else:
            if not isinstance(n_levels, int) or isinstance(n_levels, bool):
                raise TypeError("n_levels must be an integer")
            if n_levels < 2:
                raise ValueError(f"n_levels must be >= 2, got {n_levels}")

        # Assign attributes
        self.n_bits = n_bits
        self.v_ref = v_ref
        self.output_type = output_type
        self.n_levels = n_levels
        self.lsb = v_ref / (n_levels - 1)

    def convert(self, digital_input: int) -> Union[float, Tuple[float, float]]:
        """
        Convert digital input to analog output.

        Args:
            digital_input: Input code (must be between 0 and 2^n_bits - 1)

        Returns:
            Single voltage or tuple of (pos, neg) for differential

        Raises:
            ValueError: If input code is out of range
        """
        # Validate input range
        if not isinstance(digital_input, (int, np.integer)):
            raise TypeError("Digital input must be an integer")
        if digital_input < 0 or digital_input >= self.n_levels:
            raise ValueError(f"Digital input must be between 0 and {self.n_levels - 1}")

        # Call architecture-specific conversion
        return self._convert_input(digital_input)

    @abstractmethod
    def _convert_input(self, digital_input: int) -> Union[float, Tuple[float, float]]:
        """
        Architecture-specific conversion implementation.

        Args:
            digital_input: Pre-validated input code

        Returns:
            Single voltage or tuple of voltages for differential
        """
        pass

    def __repr__(self) -> str:
        """String representation of the DAC"""
        return f"{self.__class__.__name__}(n_bits={self.n_bits}, v_ref={self.v_ref}, output_type={self.output_type.name})"
