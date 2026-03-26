SimpleADC.py Code Review
=========================

Date: 2026-03-25

BUGS
----
1. Noise non-determinism (lines 128, 130)
   - Uses np.random.normal() without a seed mechanism
   - Simulations are non-reproducible, problematic for testing/debugging

2. Missing type validation for offset and gain_error
   - No checks that offset and gain_error are numeric types
   - Inconsistent with noise_rms and t_jitter which have validation

3. _convert_input returns int but not clipped in base validation
   - The base class ADCBase.convert() doesn't validate return type
   - A subclass could theoretically return an out-of-range integer

POTENTIAL IMPROVEMENTS
----------------------
1. Add seed support
   - Add a seed parameter to __init__ for reproducible noise simulations

2. Redundant input validation
   - convert() duplicates validation from ADCBase.convert()
   - Could call super().convert() or remove duplicate checks

3. Jitter-only condition at line 129-130 is confusing
   - if self.t_jitter and self._dvdt: applies jitter only if BOTH are set
   - This is correct but could be clearer with separate if statements

4. Missing __repr__ for non-ideality parameters when enabled
   - Currently only shows quant_mode even if noise/offset/gain/jitter are set
   - __repr__ correctly includes these when non-zero, but could be more complete

5. dvdt parameter could be validated
   - Should check if dvdt is a number (int/float)
   - Currently accepts any type passed to convert()

6. Consider adding a convert_sequence() method
   - Like SimpleDAC has, for batch processing
   - Would allow vectorized noise application

SUMMARY
-------
SimpleADC is well-designed with good separation of concerns (_apply_nonidealities,
_quantize). Main issues are reproducibility and some redundant validation.
