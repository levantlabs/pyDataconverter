SimpleDAC.py Code Review
=========================

Location: pyDataconverter/architectures/SimpleDAC.py

Date: 2026-03-25

BUGS
----
1. Noise non-determinism (lines 85, 142)
   - Uses np.random.normal() without a seed mechanism
   - Simulations are non-reproducible, problematic for testing/debugging

2. Inconsistency between convert() and convert_sequence()
   - convert() applies noise via _apply_nonidealities() per code
   - convert_sequence() applies noise after ZOH repetition, so noise is
     per-oversampled-point rather than per-code

3. Missing input validation
   - No validation for offset (should check type)
   - No validation for gain_error (should check if it's a number)

POTENTIAL IMPROVEMENTS
----------------------
1. Add seed support
   - Add a seed parameter to __init__ for reproducible noise

2. Differential output math (lines 111-114)
   - The calculation v_diff = 2 * voltage - self.v_ref gives ±v_ref/2 range
   - This is correct for rail-to-rail but the docstring doesn't clarify this

3. Redundant differential calculations
   - Lines 149-151 duplicate lines 111-114
   - Could be extracted to a helper method

4. __repr__ missing output_type
   - Only shows n_bits, v_ref, and non-ideality params
   - output_type is a key attribute that should be displayed

5. Missing parameter validation
   - offset and gain_error could benefit from type checking like noise_rms has

6. Consider adding output clipping
   - Output can exceed rails due to gain_error + offset
   - May want to clip to [0, v_ref]

CROSS-CUTTING ISSUES (All Architectures)
-----------------------------------------
1. **No seed support** - All components use np.random.normal() without seed
   - Every architecture has this issue
   - Makes simulations non-reproducible

2. **LSP/Type errors in base class**
   - ADCBase._convert_input returns None (abstract method with pass)
   - Should return Union[float, Tuple[float, float]] for ADC, Union[float, Tuple] for DAC
   - FlashADC and SimpleADC have return type mismatches

3. **Inconsistent validation patterns**
   - Some parameters have type/range validation, others don't

SUMMARY
-------
The core DAC logic is sound. The main issues are reproducibility (seed support)
and minor code duplication in differential calculations.
