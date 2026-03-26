adc_plots.py Code Review
========================

Date: 2026-03-25
Location: pyDataconverter/utils/visualizations/adc_plots.py

This is a clean, focused module. No major bugs found.

POTENTIAL IMPROVEMENTS
----------------------
1. plot_transfer_function hardcoded error limits (line 118)
   - `ax2.set_ylim(-0.75, 0.75)` assumes error is within ±0.5 LSB
   - For ADCs with large DNL/INL, error can exceed these bounds
   - Should auto-scale based on actual error range or make configurable

2. plot_transfer_function ramp validation (line 84)
   - No validation that v_min < v_max
   - No validation that v range is appropriate for ADC

3. plot_transfer_function assumes single-ended ADC
   - Docstring says "any single-ended ADC"
   - But doesn't validate input_type or raise error for differential
   - Could silently produce incorrect results

4. plot_conversion no validation
   - No checks that x and input_signal have same length
   - No checks that output_codes matches length

5. plot_conversion output_codes type annotation (line 21)
   - `output_codes` parameter has no type hint
   - Should be np.ndarray for type consistency

6. Consider adding figure size parameters
   - Hardcoded figsize=(10, 8) for both functions
   - Could accept optional fig_size parameter

7. Missing input validation for ADC methods
   - Assumes adc.convert() exists and is callable
   - No hasattr checks

8. Consider returning metrics dict
   - Could compute and return DNL/INL for the ramp

SUMMARY
-------
Simple, focused module with good basic plots. Main opportunities are
validation improvements and making plot parameters configurable.
