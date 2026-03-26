dac_plots.py Code Review
=======================

Date: 2026-03-25
Location: pyDataconverter/utils/visualizations/dac_plots.py

BUGS
----
1. plot_output_spectrum fs_fft calculation (line 199)
   - `fs_fft = len(freqs) * 2 * (freqs[1] - freqs[0])`
   - For rfft output, len(freqs) = N/2+1, so this gives N+2 bins
   - The fs calculation is incorrect and will be wrong for small FFTs

2. _get_voltage function (lines 26-31)
   - Handles tuple but doesn't check tuple length
   - If dac.convert returns wrong format, error is confusing

POTENTIAL IMPROVEMENTS
----------------------
1. plot_transfer_curve creates new SimpleDAC (line 54)
   - Always creates ideal reference even if one is passed
   - Could accept optional ideal reference parameter

2. plot_inl_dnl bar width calculation (line 139-140)
   - bar_width and bar_lw both equal 1.0 and 0.5 for all cases
   - Conditional logic never changes values (same for <=8 and >8)
   - Dead code that could be simplified

3. plot_output_spectrum zone handling (lines 246-258)
   - When nyquist_zone is 1, f_lo becomes 0 but then None is passed to plot_fft
   - Inconsistent: code handles zone 1 specially but doesn't for other zones

4. plot_output_spectrum metrics recalculation (line 251-253)
   - Recomputes metrics if not provided, but metrics.calculate_dac_dynamic_metrics
     is called with zone-restricted freqs
   - fs_fft is incorrect (see bug #1), affecting metrics calculation

5. plot_output_spectrum full spectrum mode
   - Uses sinc envelope which assumes ZOH reconstruction
   - Could add parameter to disable sinc overlay

6. Missing input validation
   - No checks for fs <= 0
   - No checks for nyquist_zone < 1

7. Consider return type hints

8. plt.show() should not be called in plotting functions
   - Functions should return fig/ax and let caller decide when to show
   - Current design is correct but mixed with tight_layout()

SUMMARY
-------
Good visualization module with comprehensive plots. Main issues are the
incorrect fs_fft calculation affecting metrics and dead code in bar width.
