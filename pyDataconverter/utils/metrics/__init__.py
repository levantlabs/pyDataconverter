"""
Converter Performance Metrics
==============================

Functions for calculating ADC and DAC performance metrics.

Public API (import as ``from pyDataconverter.utils import metrics`` then
call ``metrics.XYZ``):

ADC:
    calculate_gain_offset_error
    calculate_adc_dynamic_metrics
    calculate_adc_static_metrics
    calculate_adc_static_metrics_histogram
    is_monotonic
    calculate_histogram

DAC:
    calculate_dac_static_metrics
    calculate_dac_dynamic_metrics
"""

from .adc import (
    calculate_gain_offset_error,
    calculate_adc_dynamic_metrics,
    calculate_adc_static_metrics,
    calculate_adc_static_metrics_histogram,
    is_monotonic,
    calculate_histogram,
    calculate_adc_iip3,
)
from .dac import (
    calculate_dac_static_metrics,
    calculate_dac_dynamic_metrics,
    _calculate_dac_dynamic_metrics_from_fft,
)

__all__ = [
    "calculate_gain_offset_error",
    "calculate_adc_dynamic_metrics",
    "calculate_adc_static_metrics",
    "calculate_adc_static_metrics_histogram",
    "is_monotonic",
    "calculate_histogram",
    "calculate_adc_iip3",
    "calculate_dac_static_metrics",
    "calculate_dac_dynamic_metrics",
    "_calculate_dac_dynamic_metrics_from_fft",
]
