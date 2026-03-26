ReferenceLadder.py Code Review
=============================

Date: 2026-03-25

BUGS
----
1. Noise non-determinism (lines 108, 123-124)
   - Uses np.random.normal() without a seed mechanism
   - Mismatch and noise are non-reproducible

2. Resistor mismatch formula concern (lines 107-109)
   - Uses multiplicative mismatch: ideal * (1 + mismatch)
   - For small resistor values near ground, this could produce negative voltages
   - No check that resulting voltages stay valid

POTENTIAL IMPROVEMENTS
----------------------
1. Add seed support
   - Add a seed parameter for reproducible mismatch and noise

2. Add validation for resulting voltages
   - Check that mismatch doesn't produce invalid reference values

3. Consider additive mismatch model as alternative
   - ideal + mismatch (in volts) may be more intuitive
   - Or support both with a parameter

4. get_voltages() always copies array
   - Even when noise_rms=0, returns a copy via .copy()
   - Minor performance issue for large arrays

5. Missing type hints for n_bits in __init__
   - Should validate n_bits is positive integer

SUMMARY
-------
ReferenceLadder is straightforward with good validation. Main issue is
reproducibility. The mismatch formula could produce edge cases.
