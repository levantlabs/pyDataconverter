visualize_FlashADC.py Code Review
================================

Date: 2026-03-25
Location: pyDataconverter/utils/visualizations/visualize_FlashADC.py

BUGS
----
1. animate_flash_adc missing required keyword arguments (line 270-271)
   - Calls `_draw_static(..., ladder_x=ladder_x, comp_x=comp_x)`
   - But function signature is `_draw_static(adc, ax, show_comparator_details,
     ladder_x, comp_x)`
   - Missing positional argument `show_comparator_details`

2. animate_flash_adc missing return (line 281-282)
   - FuncAnimation created but not returned or assigned
   - Animation object goes out of scope immediately
   - Animation may not display or be garbage collected

3. plt.show() called in visualize function (lines 226, 244, 283)
   - Standard matplotlib issue - blocking call
   - For interactive use in notebooks/scripts this blocks execution

POTENTIAL IMPROVEMENTS
----------------------
1. _effective_thresholds accesses public property (line 21)
   - reference_voltages property multiplies by 2 for differential
   - But this is a convenience alias, not the internal representation
   - Logic may be confusing when debugging

2. _eval_comparators uses static references (line 45)
   - Comment says "static (noiseless)" but reference_voltages may have noise
   - Inconsistent with FlashADC which uses reference.get_voltages() for dynamic refs
   - Visualization shows ideal thresholds, not actual thresholds

3. Visual scaling uses arbitrary constants (lines 83, 135, 200)
   - tri_r, xlim, figsize use hardcoded values
   - May not scale well for very high n_bits

4. _y_limits uses thresholds[-1] - thresholds[0] (line 60)
   - Assumes thresholds are sorted/centered around zero
   - For differential mode this is true, but for single-ended it depends on ladder

5. Input validation missing
   - No checks that input_voltage is in valid range
   - Could pass values outside ADC range

6. animate_flash_adc function signature inconsistent (line 249)
   - Takes `interval` but FuncAnimation uses `interval * 1000`
   - Documentation says "seconds" but FuncAnimation expects milliseconds

7. visualize_flash_adc return type inconsistent
   - Returns (fig, ax) tuple
   - But docstring says just `ax`

8. Consider separating visualization from matplotlib
   - Heavy matplotlib coupling makes testing difficult
   - Could use abstract interface for different backends

SUMMARY
-------
Useful visualization tool but has bugs in animate function (missing argument,
missing return). Also shows static thresholds when actual may have noise.
