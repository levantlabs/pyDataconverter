"""
Shared static-metrics helpers used by both adc.py and dac.py.
"""

import numpy as np


def _fit_reference_line(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """
    Fit a straight reference line to (x, y) data.

    Used by both ADC and DAC INL calculations to produce the ideal reference
    against which deviations are measured.

    Unit-agnostic: x and y can be in any consistent units (volts, LSB, etc.);
    the returned line is in the same units as y.

    Args:
        x: Independent variable (e.g. transition index, code value).
           Must have at least 2 elements and x[-1] != x[0].
        y: Dependent variable (e.g. transition voltages, output voltages,
           or cumulative INL already in LSB).  Same length as x.
        method: One of:
            'endpoint' — line through (x[0], y[0]) and (x[-1], y[-1]).
                         Corrected INL is zero at both endpoints by construction.
            'best_fit' — least-squares linear fit through all (x, y) pairs.
                         Minimises RMS INL; endpoints are not forced to zero.

    Returns:
        np.ndarray of fitted line values evaluated at each x, in the same
        units as y.

    Raises:
        ValueError: If method is not 'endpoint' or 'best_fit'.
    """
    if method == 'endpoint':
        return y[0] + (y[-1] - y[0]) * (x - x[0]) / (x[-1] - x[0])
    elif method == 'best_fit':
        coeffs = np.polyfit(x, y, 1)
        return np.polyval(coeffs, x)
    else:
        raise ValueError(f"inl_method {method!r} is not recognised. "
                         "Use 'endpoint' or 'best_fit'.")
