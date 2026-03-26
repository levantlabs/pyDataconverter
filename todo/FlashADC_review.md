FlashADC.py Code Review
=======================

Date: 2026-03-25

BUGS
----
1. Noise non-determinism (lines 149, reference.py:108, 123)
   - Uses np.random.normal() without a seed mechanism in multiple places
   - Comparator offsets drawn at construction are non-reproducible
   - Reference ladder mismatch and noise are non-reproducible

2. XOR encoder edge case (lines 184-195)
   - np.bitwise_or.reduce() on an empty array raises ValueError
   - But this is protected by len(active) == 0 check (line 189)
   - However, the logic could be simplified

3. comparator_params passed to each comparator (line 154-156)
   - 'offset' key is reserved and overwritten per-comparator
   - But no warning if user passes offset in comparator_params
   - Silent override could confuse users

4. reference_voltages property (line 161)
   - For DIFFERENTIAL mode, multiplies by 2 but this creates aliasing
   - The property is a convenience alias but the name suggests static values
   - Should be called static_reference_voltages or similar to clarify

POTENTIAL IMPROVEMENTS
----------------------
1. Add seed support
   - Add a seed parameter to __init__ for reproducible simulations
   - Should seed: comparator offsets, reference mismatch

2. Thermometer encoding not exposed publicly
   - _encode() is internal but could be useful for debugging
   - Consider adding a public method to get thermometer code

3. No check for hysteresis on comparators
   - If hysteresis is enabled, reset() must be called between samples
   - Could add a warning or documentation about this requirement

4. Comparator instantiation is inefficient for high n_bits
   - Creates 2^N - 1 comparator objects (e.g., 4095 for 12-bit)
   - Consider a vectorized approach for performance

5. reference_voltages docstring misleading (line 160)
   - Says "no noise" but for DIFFERENTIAL mode the multiplication by 2
     is not the same as what's used internally

6. Missing validation of comparator_type
   - Should check it's a subclass of ComparatorBase

7. Dead code in __main__ (lines 304-312)
   - Second FlashADC instance created but never used

8. Consider adding a convert_batch() method
   - Vectorized conversion for performance with large datasets

9. TODO noted but not implemented (lines 32-35)
   - Bandwidth modeling not passed through to comparator

SUMMARY
-------
FlashADC is a comprehensive implementation with good encoder options. Main issues
are reproducibility, inefficient object creation for high resolutions, and some
confusing property naming. The XOR encoder has good documentation.
