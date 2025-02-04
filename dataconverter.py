"""
Data Converter Base Closses
===========================

This module provides interfaces for both ADC and DAC implementations.

Classes:
    ADCBase: abstract class for all ADC implementations
    DACBase: abstract class for all DAC implementations

Version History:
2025-01-31: First pass wrapper
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Tuple


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



