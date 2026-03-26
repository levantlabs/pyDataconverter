fft_analysis.py Code Review
=========================

Date: 2026-03-25
Location: pyDataconverter/utils/fft_analysis.py

BUGS
----
1. demo_fft_analysis undefined variable (line 233)
   - Uses `duration` variable that is never defined
   - Should be `NFFT / fs` or similar

2. demo_fft_analysis undefined variable (line 277)
   - Uses `duration` in generate_sine calls, never defined in that scope

POTENTIAL IMPROVEMENTS
----------------------
1. compute_fft n_fft calculation is wrong (line 58)
   - `n_fft = len(freqs) * 2` is incorrect
   - FFT length should be len(time_data), not derived from freqs
   - freqs comes from np.fft.fftfreq which gives N unique freqs for N-point FFT

2. Inconsistent tolerance units (line 135 vs find_harmonics)
   - _get_harmonic uses tol * freq_spacing for comparison
   - find_harmonics passes tol=0.1 (fraction), but comment says "fraction of
     frequency spacing" - the default is actually in Hz if interpreted literally

3. Missing input validation
   - No checks for fs <= 0
   - No checks for empty time_data
   - No checks for invalid window names before getattr

4. No return type hints

5. Aliasing calculation could be simplified (lines 124-131)
   - Complex modulo/folding logic for harmonic frequencies
   - Could use np.fft.rfftfreq with appropriate handling

6. DC bin handling (line 72)
   - Adds 1e-20 to avoid log10(0) but DC bin is excluded anyway (freqs >= 0 mask)
   - This guard is unnecessary for positive frequencies

7. Consider separating demo function from module
   - demo_fft_analysis() is substantial and could be in a separate examples file

SUMMARY
-------
Core FFT computation is sound but demo code has bugs. The n_fft calculation
appears incorrect but may not affect plots since it's only used in title.
