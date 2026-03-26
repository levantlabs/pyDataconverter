CDAC (Capacitive DAC) Code Review
================================

Date: 2026-03-25
Location: pyDataconverter/components/cdac.py

This is a well-documented module with clear architecture. No major bugs found.

POTENTIAL IMPROVEMENTS
----------------------
1. Voltages property inefficiency (lines 116-118)
   - Calls get_voltage() for every code in a loop
   - For 16-bit DAC, that's 65536 calls
   - Could be vectorized for performance

2. SingleEndedCDAC._code_to_bits inefficiency (lines 237-241)
   - Creates numpy array with Python loop over bits
   - Could use bitwise operations on full array

3. DifferentialCDAC._code_to_bits duplicated (lines 379-382)
   - Exact copy of SingleEndedCDAC._code_to_bits
   - Could be moved to CDACBase as shared method

4. Missing cap_weights_neg property in SingleEndedCDAC
   - SingleEndedCDAC has cap_weights but DifferentialCDAC also has cap_weights_neg
   - Inconsistent interface for single-ended variant

5. Capacitor mismatch documentation (lines 34-40)
   - Notes say mismatch is "multiplicative Gaussian"
   - But 3-sigma rule not enforced - extreme values could produce negative caps
   - Could add clipping: np.clip(mismatch, -0.5, 0.5) or similar

6. get_voltage always returns tuple (lines 203-217, 353-373)
   - Even for SingleEndedCDAC which returns (v_dac, 0.0)
   - Consistent but SingleEndedCDAC could return single float for clarity

7. Missing __repr__ for DifferentialCDAC
   - Inherits from CDACBase but SingleEndedCDAC has custom __repr__
   - DifferentialCDAC should also show mismatch info

8. Consider adding cap_mismatch validation range
   - No check that mismatch std is reasonable (< 0.5 or similar)
   - Extreme values could produce invalid capacitor values

SUMMARY
-------
Well-designed capacitive DAC module with good documentation. Main opportunities
are vectorization for performance and code deduplication.
