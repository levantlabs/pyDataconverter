"""
Internal bit-packing helpers.

Centralises the MSB-first bit extraction used by decoders, CDACs, and the
R-2R ladder. The previous pattern (``[(code >> (n-1-k)) & 1 for k in range(n)]``)
is O(n) Python bytecode per call; ``np.unpackbits`` does the same work in C
with a fixed overhead independent of ``n_bits``, which is measurable on
per-sample CDAC hot paths for 10+ bit converters.
"""

import numpy as np


def code_to_bits_msb_first(code: int, n_bits: int, dtype=np.int8) -> np.ndarray:
    """
    Unpack an integer code into an MSB-first bit array of length ``n_bits``.

    Args:
        code: Integer in ``[0, 2**n_bits - 1]``.
        n_bits: Output bit width.
        dtype: NumPy dtype for the returned array (default ``np.int8``).
            Callers that need float arrays for downstream arithmetic can
            pass ``dtype=float``.

    Returns:
        ``np.ndarray`` of shape ``(n_bits,)``: ``bits[0]`` is the MSB,
        ``bits[-1]`` is the LSB.
    """
    byte_len = (n_bits + 7) // 8
    byte_buf = int(code).to_bytes(byte_len, "big")
    bits = np.unpackbits(np.frombuffer(byte_buf, dtype=np.uint8))
    # Discard any leading pad bits introduced by byte-boundary rounding.
    return bits[-n_bits:].astype(dtype, copy=False)
