signal_gen.py Code Review
=========================

Date: 2026-03-25
Location: pyDataconverter/utils/signal_gen.py

This module is well-structured with good docstrings. No major bugs found.

POTENTIAL IMPROVEMENTS
----------------------
1. generate_sine time array edge case (line 46)
   - Uses `np.arange(0, duration, 1/sampling_rate)`
   - Due to floating point, last sample may be slightly less than duration
   - Could use `np.linspace(0, duration, int(duration * sampling_rate) + 1)`

2. generate_digital_sine clipping behavior (line 293)
   - Uses np.round before np.clip
   - Round is correct but comment says "90% of full scale" but uses 0.5 offset
   - Formula: amplitude/2 * sin(...) + offset
   - For amplitude=0.9, offset=0.5, range is [0.05, 0.95], never reaches full scale

3. generate_digital_two_tone similar issue (lines 325-326)
   - Same amplitude/offset handling issue

4. generate_imd_tones inconsistent naming (line 186)
   - Docstring example calls it `generate_imd_test` but function is `generate_imd_tones`

5. generate_multitone phase handling (line 159)
   - Uses np.outer for vectorization which is good
   - But if len(frequencies) is large, memory usage could be high
   - Consider generator approach for large n_tones

6. Missing type hints throughout

7. generate_coherent_sine could validate n_fin range (line 448)
   - Should check 1 <= n_fin < n_fft/2 as docstring states
   - Currently silently produces incorrect results if violated

8. generate_digital_ramp dtype inconsistency (line 232)
   - Returns int dtype when using np.linspace with dtype=int
   - But np.linspace(0, max_code, n_points, dtype=int) can produce duplicates

9. Consider adding seed parameter for reproducibility
   - These functions are deterministic, but users may want reproducible test signals

SUMMARY
-------
Solid signal generation module with good coverage of test signals. Main issues
are minor floating-point edge cases and some parameter validation gaps.
