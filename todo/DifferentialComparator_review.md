DifferentialComparator.py Code Review
=====================================

Date: 2026-03-25

BUGS
----
1. Noise non-determinism (line 167)
   - Uses np.random.normal() without a seed mechanism
   - Simulations are non-reproducible

2. Hysteresis state not properly isolated per comparison
   - _last_output persists across conversions unless reset() is called
   - In Flash ADC, comparators are reused without explicit reset
   - Could cause incorrect hysteresis behavior in sequential samples

3. Bandwidth filter initialization issue (lines 129-131)
   - _last_input initialized to 0.0 when bandwidth is set
   - First sample will have incorrect filtering (cold start)

4. time_constant parameter unused (line 111, 126)
   - Listed in __init__ but never used anywhere
   - Dead code that could confuse users

POTENTIAL IMPROVEMENTS
----------------------
1. Add seed support
   - Add a seed parameter for reproducible noise

2. Make noise application optional per comparison
   - Currently noise is always added if noise_rms > 0
   - Some use cases need deterministic operation

3. Hysteresis documentation could be clearer
   - Clarify that reset() must be called between independent samples
   - Add note about Flash ADC usage pattern

4. Consider adding a compare_no_noise() method
   - Useful for testing and deterministic scenarios

5. bandwidth parameter validation
   - Should check bandwidth > 0 if provided

6. _last_input should be initialized to v_diff, not 0
   - For proper first-order filter behavior

SUMMARY
-------
DifferentialComparator is well-structured with proper abstract base class.
Main issues are reproducibility, uninitialized filter state, and unused
time_constant parameter.
