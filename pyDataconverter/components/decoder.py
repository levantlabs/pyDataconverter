"""
DAC Decoder Components
======================

Decoders translate a compact N-bit digital code into control signals that
drive a physical DAC element array.  This is the DAC-side counterpart of the
ADC Encoder (e.g. EncoderType in FlashADC), which compresses comparator
outputs down to a binary code.

    ADC side:  physical comparators → Encoder  → compact N-bit code
    DAC side:  compact N-bit code   → Decoder  → element control signals

All decoders share the same return convention from decode():

    (therm_index: int, binary_bits: np.ndarray)

    therm_index   — number of thermometer unit elements to switch on
                    (integer in [0, 2^n_therm_bits − 1])
    binary_bits   — bit array for the binary sub-array, MSB first
                    (length == n_binary_bits; empty array when n_binary_bits == 0)

This uniform return type means the DAC never needs to know which decoder
variant it holds — it always unpacks the same two values.

Classes:
    DecoderBase:         Abstract base defining the decode() interface.
    BinaryDecoder:       Passthrough — all bits drive a binary-weighted array.
    ThermometerDecoder:  Full unary decode — code drives 2^N − 1 unit elements.
    SegmentedDecoder:    MSBs thermometer-decoded, LSBs binary (general case).

Version History:
---------------
1.0.0 (2026-03-25):
    - Initial release
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class DecoderBase(ABC):
    """
    Abstract base class for all DAC decoders.

    A decoder takes an N-bit integer code and returns the control signals
    needed to drive the physical element array:

        (therm_index, binary_bits) = decoder.decode(code)

    All subclasses must implement decode() and expose n_bits and n_therm_bits.
    n_binary_bits is derived and need not be overridden.
    """

    @property
    @abstractmethod
    def n_bits(self) -> int:
        """Total DAC resolution in bits."""

    @property
    @abstractmethod
    def n_therm_bits(self) -> int:
        """Number of MSBs decoded to thermometer (unary) control signals."""

    @property
    def n_binary_bits(self) -> int:
        """Number of LSBs that remain as binary control signals."""
        return self.n_bits - self.n_therm_bits

    @abstractmethod
    def decode(self, code: int) -> Tuple[int, np.ndarray]:
        """
        Decode a digital code into thermometer and binary control signals.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            Tuple[int, np.ndarray]:
                therm_index  — integer in [0, 2^n_therm_bits − 1].  Represents
                               how many unit thermometer elements to switch on.
                binary_bits  — 1-D array of length n_binary_bits (dtype float),
                               MSB first.  Empty array when n_binary_bits == 0.

        Raises:
            ValueError: If code is outside [0, 2^n_bits − 1].
        """

    def _validate_code(self, code: int) -> None:
        if not isinstance(code, (int, np.integer)):
            raise TypeError(f"code must be an integer, got {type(code).__name__}")
        if code < 0 or code >= 2 ** self.n_bits:
            raise ValueError(
                f"code {code} is out of range [0, {2 ** self.n_bits - 1}]"
            )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"n_bits={self.n_bits}, "
                f"n_therm_bits={self.n_therm_bits}, "
                f"n_binary_bits={self.n_binary_bits})")


class BinaryDecoder(DecoderBase):
    """
    Binary (passthrough) decoder.

    The input code is returned as a bit array with no thermometer expansion.
    All bits drive a binary-weighted element array directly.

        n_therm_bits  = 0
        n_binary_bits = n_bits

    This is the degenerate case of SegmentedDecoder with n_therm_bits=0.

    Example (n_bits=4, code=0b1011=11):
        therm_index = 0
        binary_bits = [1., 0., 1., 1.]
    """

    def __init__(self, n_bits: int):
        """
        Args:
            n_bits: DAC resolution in bits (1–32).

        Raises:
            ValueError: If n_bits is out of range.
        """
        if not isinstance(n_bits, int) or n_bits < 1 or n_bits > 32:
            raise ValueError("n_bits must be an integer in [1, 32]")
        self._n_bits = n_bits

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def n_therm_bits(self) -> int:
        return 0

    def decode(self, code: int) -> Tuple[int, np.ndarray]:
        """
        Return (0, binary_bits) for the given code.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            Tuple[int, np.ndarray]: (0, bit array of length n_bits, MSB first).
        """
        self._validate_code(code)
        bits = np.array(
            [(code >> (self._n_bits - 1 - k)) & 1 for k in range(self._n_bits)],
            dtype=float,
        )
        return 0, bits


class ThermometerDecoder(DecoderBase):
    """
    Full thermometer (unary) decoder.

    The N-bit input code selects how many of the 2^N − 1 unit elements are
    switched on.  No binary sub-array exists.

        n_therm_bits  = n_bits
        n_binary_bits = 0

    This is the degenerate case of SegmentedDecoder with n_therm_bits=n_bits.

    Properties:
        n_elements: Number of unit thermometer elements (= 2^n_bits − 1).

    Example (n_bits=3, code=5):
        therm_index = 5   (switch on 5 of 7 unit elements)
        binary_bits = []  (empty — no binary sub-array)
    """

    def __init__(self, n_bits: int):
        """
        Args:
            n_bits: DAC resolution in bits (1–16).
                    Limited to 16 to keep element counts tractable
                    (2^16 − 1 = 65535 unit elements).

        Raises:
            ValueError: If n_bits is out of range.
        """
        if not isinstance(n_bits, int) or n_bits < 1 or n_bits > 16:
            raise ValueError(
                "n_bits must be an integer in [1, 16] for a full thermometer decoder "
                "(2^n_bits − 1 unit elements would be created)"
            )
        self._n_bits = n_bits

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def n_therm_bits(self) -> int:
        return self._n_bits

    @property
    def n_elements(self) -> int:
        """Number of unit thermometer elements (2^n_bits − 1)."""
        return 2 ** self._n_bits - 1

    def decode(self, code: int) -> Tuple[int, np.ndarray]:
        """
        Return (code, []) for the given code.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            Tuple[int, np.ndarray]: (code, empty array).
        """
        self._validate_code(code)
        return code, np.array([], dtype=float)


class SegmentedDecoder(DecoderBase):
    """
    Segmented decoder — thermometer MSBs, binary LSBs.

    Splits the N-bit code into:
        upper n_therm_bits  → thermometer index (integer)
        lower n_binary_bits → binary bit array (MSB first)

    This is the most general case:
        n_therm_bits = 0        → equivalent to BinaryDecoder
        n_therm_bits = n_bits   → equivalent to ThermometerDecoder

    Typical usage: upper 4–6 bits thermometer, remaining bits binary.

    Example (n_bits=6, n_therm_bits=4, code=0b101101=45):
        upper 4 bits → 0b1011 = 11   → therm_index = 11
        lower 2 bits → 0b01          → binary_bits = [0., 1.]

    Attributes:
        n_therm_elements: Number of thermometer unit elements (2^n_therm_bits − 1).
    """

    def __init__(self, n_bits: int, n_therm_bits: int):
        """
        Args:
            n_bits:       Total DAC resolution (1–32).
            n_therm_bits: Number of MSBs to decode as thermometer (0–n_bits).
                          Thermometer segment creates 2^n_therm_bits − 1 unit
                          elements; limited to n_therm_bits ≤ 16 for tractability.

        Raises:
            ValueError: If arguments are invalid.
        """
        if not isinstance(n_bits, int) or n_bits < 1 or n_bits > 32:
            raise ValueError("n_bits must be an integer in [1, 32]")
        if not isinstance(n_therm_bits, int) or n_therm_bits < 0 or n_therm_bits > n_bits:
            raise ValueError(
                f"n_therm_bits must be an integer in [0, n_bits={n_bits}]"
            )
        if n_therm_bits > 16:
            raise ValueError(
                "n_therm_bits > 16 would create more than 65535 unit elements"
            )
        self._n_bits       = n_bits
        self._n_therm_bits = n_therm_bits

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def n_therm_bits(self) -> int:
        return self._n_therm_bits

    @property
    def n_therm_elements(self) -> int:
        """Number of thermometer unit elements (2^n_therm_bits − 1)."""
        return max(0, 2 ** self._n_therm_bits - 1)

    def decode(self, code: int) -> Tuple[int, np.ndarray]:
        """
        Split code into thermometer index and binary bit array.

        Args:
            code: Integer in [0, 2^n_bits − 1].

        Returns:
            Tuple[int, np.ndarray]:
                therm_index  — upper n_therm_bits as an integer.
                binary_bits  — lower n_binary_bits as a float bit array, MSB
                               first.  Empty when n_binary_bits == 0.
        """
        self._validate_code(code)
        n_bin = self.n_binary_bits

        therm_index = code >> n_bin

        if n_bin > 0:
            binary_code = code & ((1 << n_bin) - 1)
            binary_bits = np.array(
                [(binary_code >> (n_bin - 1 - k)) & 1 for k in range(n_bin)],
                dtype=float,
            )
        else:
            binary_bits = np.array([], dtype=float)

        return therm_index, binary_bits
