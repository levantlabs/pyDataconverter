metrics.py Code Review
=====================

Date: 2026-03-25
Location: pyDataconverter/utils/metrics.py

BUGS
----
1. Typo in function call (line 159)
   - References `time_data=None` but this is a parameter named `time_data` in
     _calculate_dynamic_metrics, not a local variable
   - Should be `time_data=None` passed directly, but the variable doesn't exist
     in calculate_dac_dynamic_metrics scope

2. _calculate_dynamic_metrics offset calculation ambiguity (lines 66-69)
   - Uses time_data mean for offset OR falls back to DC bin magnitude
   - These are not equivalent; DC bin from FFT already has window scaling applied

3. Division by zero protection uses magic number (line 46, 58, 61)
   - Uses 1e-20 as floor to avoid log10(0)
   - Could use np.finfo(float).tiny instead

POTENTIAL IMPROVEMENTS
----------------------
1. Level correction formula (line 88)
   - 20 * log10(full_scale / 2) + 20 * log10(N / 2) is unusual
   - Standard dBFS uses full_scale/2 as normalization, not this formula

2. find_fundamental tolerance check (line 48)
   - Uses bin_width > freq tolerance, but harmonic check uses different logic
   - Inconsistent tolerance handling across functions

3. Missing input validation
   - No checks for NaN/Inf in input arrays
   - No validation that fs > 0

4. calculate_histogram hardcoded magic numbers (line 317)
   - Uses 0.999 for amplitude clipping threshold
   - Should be configurable or documented

5. calculate_histogram sine PDF removal is approximate (lines 311-324)
   - PDF division can amplify noise in low-count bins
   - No handling for zero PDF (edge codes)

6. _calculate_code_edges could miss transitions (lines 191-193)
   - While loop fills missing transitions with last value
   - This creates incorrect transition estimates

7. Consider using numpy arrays consistently throughout
   - calculate_adc_static_metrics mixes list operations with np.diff

8. add return type hints
   - Functions lack type annotations for return values

SUMMARY
-------
Good overall structure with clear separation of concerns. Main issues are
the time_data reference bug, inconsistent tolerance handling, and approximate
histogram PDF correction.
