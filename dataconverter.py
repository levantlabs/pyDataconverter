"""
Data Converter Base Closses
===========================

This module provides interfaces for both ADC and DAC implementations.

Classes:
    ADCBase: abstract class for all ADC implementations
    DACBase: abstract class for all DAC implementations

Version History:
2025-01-31: First pass wrapper
2025-02-06: Added DACBase abstract class
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Tuple
import numpy as np


class InputType(Enum):
    """Defines differential or single ended inputs for ADCs"""
    SINGLE = 'single'
    DIFFERENTIAL='differential'

class ADCBase(ABC):
    """Abstract base class for all ADC architectures"""

    def __init__(self, n_bits: int, v_ref: float=1.0, input_type: InputType=InputType.DIFFERENTIAL):
        if not isinstance(n_bits, int):
            raise TypeError('n_bits must be an integer.')
        if n_bits < 1:
            raise ValueError('n_bits must be larger than 0.')
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

    def convert(self, vin: Union[float, Tuple[float, float]]):
        """Convert analog input to digital output
        Can either be a single value, if single ended, or a tuple if differential.
        Check input type validation before running conversion"""
        if self.input_type == InputType.SINGLE:
            if not isinstance(vin, (int, float)):
                raise TypeError("Single-ended input must be a number.")
        elif self.input_type == InputType.DIFFERENTIAL:
            if not isinstance(vin, tuple) or len(vin) != 2:
                raise TypeError("Differential input must be a tuple of (positive, negative).")

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
    DIFFERENTIAL = 'diff'


class DACBase(ABC):
    """
    Abstract base class for DAC implementations.

    Attributes:
        n_bits (int): DAC resolution
        v_ref (float): Reference voltage
        output_type (OutputType): Single-ended or differential
        lsb (float): Least significant bit size
    """

    def __init__(self, n_bits: int, v_ref: float = 1.0, output_type: OutputType = OutputType.SINGLE):
        # Validate n_bits
        if not isinstance(n_bits, int):
            raise TypeError("n_bits must be an integer")
        if n_bits <= 0 or n_bits > 32:
            raise ValueError("n_bits must be between 1 and 32")

        # Validate v_ref
        if not isinstance(v_ref, (int, float)):
            raise TypeError("v_ref must be a number")
        if v_ref <= 0:
            raise ValueError("v_ref must be positive")

        # Validate output_type
        if not isinstance(output_type, OutputType):
            raise TypeError("output_type must be an OutputType enum")

        # Assign attributes
        self.n_bits = n_bits
        self.v_ref = v_ref
        self.output_type = output_type
        self.lsb = v_ref / (2 ** n_bits)

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
        if digital_input < 0 or digital_input >= 2 ** self.n_bits:
            raise ValueError(f"Digital input must be between 0 and {2 ** self.n_bits - 1}")

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
