fft_plots.py Code Review
=======================

Date: 2026-03-25
Location: pyDataconverter/utils/visualizations/fft_plots.py

This is a clean visualization module. No major bugs found.

POTENTIAL IMPROVEMENTS
----------------------
1. n_fft calculation misleading (line 58)
   - `n_fft = len(freqs) * 2`
   - This is incorrect - n_fft should be the original FFT size
   - len(freqs) for rfft is N/2+1, so n_fft would be N+2
   - Should be passed as a parameter or stored differently

2. Missing input validation
   - No checks for empty freqs/mags arrays
   - No checks for fs <= 0

3. Frequency unit selection edge case (lines 84-85)
   - `plot_max = scaled_max if scaled_max > 1 else 1.0`
   - This forces minimum x-axis of 1 Hz/kHz/MHz even if data is less
   - Could be confusing for narrowband analysis

4. Metrics annotation box positioning (line 137)
   - Uses 0.98, 0.97 (near top-right)
   - Could overlap with data if SNR is very high
   - Consider making position configurable

5. min_db default (line 91)
   - `min_db = max_db - 120` assumes -120 dB floor
   - For some signals (e.g., high resolution), -120 dBFS may not show noise floor
   - Consider making this configurable

6. y-axis label fixed as dBFS (line 108)
   - Always says "dBFS" even when normalization=NONE
   - Should be "dB" or configurable when not using full-scale normalization

7. Return type annotation missing
   - Function returns ax but no type hint

8. Consider accepting numpy arrays directly
   - freqs/mags assumed to be numpy arrays but not enforced

SUMMARY
-------
Clean plotting module with good styling and metrics annotation. Main issue
is the incorrect n_fft calculation which only affects plot titles.
