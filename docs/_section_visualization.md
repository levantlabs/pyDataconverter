## Visualization

---

### `plot_conversion`

*`pyDataconverter.utils.visualizations.adc_plots`*

Plot ADC input signal and output codes on a shared x-axis as a two-panel figure.

Creates a vertically stacked pair of subplots: the top panel shows the analog input
signal and the bottom panel shows the corresponding digital output codes. The x-axis
is shared between both panels for easy time-domain comparison.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| x | `np.ndarray` | — | X-axis values (e.g. time array or voltage array). |
| input_signal | `np.ndarray` | — | Analog input signal array. |
| output_codes | array-like | — | Digital output code array. |
| title | `str` | — | Plot title prefix for the input signal subplot. |
| xlabel | `str` | `'Time (s)'` | X-axis label. |
| ylabel | `str` | `'Voltage (V)'` | Y-axis label for the input signal subplot. |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[Figure, tuple[Axes, Axes]]` | Matplotlib figure and a tuple of the two axes `(ax1, ax2)`. |

**Examples**

```python
import numpy as np
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.utils.visualizations.adc_plots import plot_conversion

adc = SimpleADC(n_bits=8, v_ref=1.0)
t = np.linspace(0, 1e-3, 500)
signal = 0.5 + 0.4 * np.sin(2 * np.pi * 1e3 * t)
codes = np.array([adc.convert(v) for v in signal])
fig, (ax1, ax2) = plot_conversion(t, signal, codes, title='1 kHz Sine')
```

**See Also**

- `plot_transfer_function` — sweep-based transfer function and quantization error plot

---

### `plot_transfer_function`

*`pyDataconverter.utils.visualizations.adc_plots`*

Sweep an ADC with a voltage ramp and plot the transfer function with quantization error.

Generates a linear voltage ramp from `v_min` to `v_max`, converts each sample through
the ADC, and produces a two-panel figure. The top panel shows output code vs. input
voltage (transfer function) and the bottom panel shows the quantization error in LSBs.
The error calculation adapts to the ADC's quantization mode: FLOOR mode uses bin
midpoints `(code + 0.5) * LSB`, while SYMMETRIC mode uses `code * LSB` directly.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| adc | ADCBase subclass | — | ADC instance with a `convert(float)` method and `n_bits` attribute. Must be configured for single-ended input. |
| v_min | `float` | — | Minimum sweep voltage (V). |
| v_max | `float` | — | Maximum sweep voltage (V). |
| n_points | `int` | `10000` | Number of ramp points. |
| title | `Optional[str]` | `None` | Plot title. Defaults to the ADC's `__repr__`. |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[Figure, tuple[Axes, Axes]]` | Matplotlib figure and a tuple of the two axes `(ax1, ax2)`. |

**Notes**

- The LSB value is displayed in the plot title in millivolts.
- Quantization error is bounded to approximately [-0.5, +0.5] LSB for an ideal ADC.
- The y-axis of the error subplot is fixed to [-0.75, +0.75] LSB for consistent visual comparison.

**Examples**

```python
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.utils.visualizations.adc_plots import plot_transfer_function

adc = SimpleADC(n_bits=8, v_ref=1.0)
fig, (ax_tf, ax_err) = plot_transfer_function(adc, v_min=0.0, v_max=1.0)
```

**See Also**

- `plot_conversion` — time-domain input/output plot
- `SimpleADC` — a concrete ADC to use with this function
- `QuantizationMode` — affects how quantization error is computed

---

### `plot_transfer_curve`

*`pyDataconverter.utils.visualizations.dac_plots`*

Sweep all codes through a DAC and plot the transfer curve with output error.

Compares the actual DAC output (including any non-idealities) against an ideal
reference DAC of the same resolution and reference voltage. Produces a two-panel
figure: the top panel shows ideal vs. actual output voltage, and the bottom panel
shows the output error in LSBs. For differential DACs, the differential voltage
`(v_pos - v_neg)` is plotted.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| dac | DACBase subclass | — | DAC instance with a `convert(int)` method and `n_bits` attribute. |
| title | `Optional[str]` | `None` | Plot title. Defaults to the DAC's `__repr__`. |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[Figure, tuple[Axes, Axes]]` | Matplotlib figure and a tuple of the two axes `(ax1, ax2)`. |

**Notes**

- The ideal reference is a `SimpleDAC` with the same `n_bits` and `v_ref`, configured as single-ended with no non-idealities.
- The LSB value is displayed in the plot title in millivolts.
- Error y-axis is fixed to [-0.75, +0.75] LSB for visual consistency.

**Examples**

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.visualizations.dac_plots import plot_transfer_curve

dac = SimpleDAC(n_bits=8, v_ref=1.0, output_type=OutputType.SINGLE)
fig, (ax_tf, ax_err) = plot_transfer_curve(dac)
```

**See Also**

- `plot_inl_dnl` — INL/DNL bar charts for the same DAC
- `SimpleDAC` — a concrete DAC to use with this function

---

### `plot_inl_dnl`

*`pyDataconverter.utils.visualizations.dac_plots`*

Compute and plot INL and DNL bar charts for a DAC.

DNL (Differential Non-Linearity) measures the deviation of each code step from the
ideal 1-LSB step size. INL (Integral Non-Linearity) measures the cumulative deviation
from the ideal transfer function, with an endpoint-fit correction that removes offset
and gain error per IEEE 1057. For differential DACs, the differential voltage
`(v_pos - v_neg)` is used.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| dac | DACBase subclass | — | DAC instance with a `convert(int)` method and `n_bits` attribute. |
| title | `Optional[str]` | `None` | Plot title prefix. Defaults to the DAC's `__repr__`. |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[Figure, tuple[Axes, Axes]]` | Matplotlib figure and a tuple of the two axes `(ax1, ax2)`. |

**Notes**

- DNL is plotted for codes 1 through 2^N - 1 (one fewer point than INL, since DNL measures step differences).
- INL uses an endpoint-fit correction: a straight line from INL[0] to INL[2^N - 1] is subtracted.
- Reference lines are drawn at +/-0.5 LSB (gray) and +/-1.0 LSB (red) on the DNL plot.
- Max |DNL| and max |INL| values are displayed in the subplot titles.

**Examples**

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.visualizations.dac_plots import plot_inl_dnl

dac = SimpleDAC(n_bits=8, v_ref=1.0, output_type=OutputType.SINGLE,
                gain_error=0.002, offset_error=0.001)
fig, (ax_dnl, ax_inl) = plot_inl_dnl(dac)
```

**See Also**

- `plot_transfer_curve` — full transfer curve with error plot
- `SimpleDAC` — a concrete DAC to use with this function

---

### `plot_output_spectrum`

*`pyDataconverter.utils.visualizations.dac_plots`*

Plot the output spectrum of a DAC driven by a sinusoidal code sequence.

Generates a digital sine wave, converts each code through the DAC, then computes and
plots the FFT of the output voltage. Coherent sampling is used: `f_sig` is snapped to
the nearest FFT bin to eliminate spectral leakage.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| dac | DACBase subclass | — | DAC instance. |
| fs | `float` | — | DAC update rate (Hz). |
| f_sig | `float` | — | Desired signal frequency (Hz). |
| n_fft | `int` | `4096` | FFT size. |
| window | `str` | `'hann'` | Window function for FFT. |
| title | `Optional[str]` | `None` | Plot title. Defaults to the DAC's `__repr__`. |
| metrics | `dict` | `None` | Pre-computed metrics dict to display in an annotation box. |

**Returns**

| Type | Description |
|------|-------------|
| `Axes` | The matplotlib axis used for the plot. |

**Notes**

- The signal frequency is snapped to the nearest FFT bin for coherent sampling: `f_actual = round(f_sig * n_fft / fs) / n_fft * fs`.
- The digital sine is generated with amplitude 0.9 and offset 0.5 (normalized to the code range).
- When `fs >> f_sig`, oversampling spreads the noise floor across a wider bandwidth, lowering in-band noise density.
- For differential DACs, the differential voltage `(v_pos - v_neg)` is used.
- The FFT is computed with dBFS normalization using the DAC's `v_ref` as full scale.

**Examples**

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType
from pyDataconverter.utils.visualizations.dac_plots import plot_output_spectrum

dac = SimpleDAC(n_bits=10, v_ref=1.0, output_type=OutputType.SINGLE)
ax = plot_output_spectrum(dac, fs=1e6, f_sig=10e3)
```

**See Also**

- `plot_fft` — the underlying FFT plotting function
- `generate_digital_sine` — generates the digital sine code sequence
- `compute_fft` — computes the FFT used for the spectrum

---

### `plot_fft`

*`pyDataconverter.utils.visualizations.fft_plots`*

Plot an FFT spectrum with automatic frequency unit selection and optional metrics annotation.

Renders a frequency-domain magnitude plot with a filled-under-curve style. Frequency
units are automatically selected (Hz, kHz, MHz, GHz) based on the maximum frequency.
Infinite or NaN values (e.g. from log10 of zero at DC) are clipped to the plot floor
so they don't compress the y-axis scale.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| freqs | `np.ndarray` | — | Frequency array (Hz). |
| mags | `np.ndarray` | — | Magnitude array (dB or dBFS). |
| title | `str` | `"FFT Spectrum"` | Plot title. |
| max_freq | `float` | `None` | Maximum frequency to display (Hz). If `None`, shows full spectrum. |
| min_db | `float` | `None` | Floor of the y-axis (dB). Defaults to `max_db - 120`. |
| max_db | `float` | `None` | Ceiling of the y-axis (dB). Defaults to `0`. |
| metrics | `dict` | `None` | Performance metrics dict from `calculate_adc_dynamic_metrics`. If provided and `show_metrics` is `True`, key metrics are displayed as an annotation box. |
| show_metrics | `bool` | `True` | Whether to render the metrics annotation box. Set to `False` to suppress even if `metrics` is provided. |
| metrics_dbfs | `bool` | `False` | If `True`, labels SNR/SNDR/SFDR/THD as dBFS instead of dB. Use when metrics were computed with dBFS normalization. |
| fig | `Figure` | `None` | Existing matplotlib figure to plot on. |
| ax | `Axes` | `None` | Existing matplotlib axis to plot on. |

**Returns**

| Type | Description |
|------|-------------|
| `Axes` | The matplotlib axis used for the plot. |

**Notes**

- When both `fig` and `ax` are `None`, a new figure of size (12, 5) is created.
- The y-axis defaults to the range [max_db - 120, max_db], i.e. [-120, 0] dBFS by default.
- Horizontal gridlines are drawn every 20 dB; vertical gridlines are disabled.
- Top and right spines are removed for a cleaner appearance.
- The metrics annotation box displays SNR, SNDR, SFDR, THD, and ENOB when available in the `metrics` dict.
- If `metrics_dbfs` is `True`, the function looks for keys with a `_dBFS` suffix (e.g. `SNR_dBFS`).

**Examples**

```python
import numpy as np
from pyDataconverter.utils.fft_analysis import compute_fft, FFTNormalization
from pyDataconverter.utils.visualizations.fft_plots import plot_fft

fs = 1e6
t = np.arange(4096) / fs
signal = 0.5 * np.sin(2 * np.pi * 10e3 * t)
freqs, mags = compute_fft(signal, fs, normalization=FFTNormalization.DBFS)
ax = plot_fft(freqs, mags, title='10 kHz Sine Spectrum', max_freq=fs / 2)
```

**See Also**

- `compute_fft` — computes the FFT data to pass to this function
- `calculate_adc_dynamic_metrics` — computes the metrics dict for the annotation box
- `plot_output_spectrum` — DAC-specific wrapper that calls this function

---

### `visualize_flash_adc`

*`pyDataconverter.utils.visualizations.visualize_FlashADC`*

Visualize the Flash ADC comparator bank as a schematic diagram.

Draws the reference voltage ladder, comparator triangles, and wiring for a Flash ADC.
In static mode, an optional input voltage is shown with comparator outputs highlighted
(green = fired, red = not fired) and the resulting digital code annotated. In
interactive mode, a slider allows real-time exploration of the full input range.
Supports both single-ended and differential Flash ADCs.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| adc | `FlashADC` | — | FlashADC instance to visualize. |
| input_voltage | `Optional[Union[float, Tuple[float, float]]]` | `None` | Input voltage for static snapshot. For single-ended ADCs, pass a scalar. For differential, pass a `(v_pos, v_neg)` tuple or a scalar `v_diff` (split as +/-v_diff/2). Ignored when `interactive=True`. |
| show_comparator_details | `bool` | `True` | Annotate each comparator with its offset voltage (in mV). |
| interactive | `bool` | `False` | If `True`, add a slider widget for real-time input voltage exploration. |
| fig | `Figure` | `None` | Existing matplotlib figure. |
| ax | `Axes` | `None` | Existing matplotlib axis. |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[Figure, Axes]` | Matplotlib figure and axis. |

**Notes**

- In interactive mode, the slider range spans the full ADC input range: [0, v_ref] for single-ended or [-v_ref/2, +v_ref/2] for differential.
- For differential ADCs, each comparator threshold is displayed as the effective differential threshold along with the individual ladder tap voltages.
- The figure height scales with the number of comparators (minimum 6 inches for static, 7 for interactive).
- When `fig` and `ax` are provided (static mode only), the function draws into the existing axes without calling `plt.show()`.

**Examples**

```python
from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.visualizations.visualize_FlashADC import visualize_flash_adc

adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE)
fig, ax = visualize_flash_adc(adc, input_voltage=0.6)
```

**See Also**

- `animate_flash_adc` — animate the visualization over a sequence of input voltages
- `FlashADC` — the Flash ADC architecture this function visualizes

---

### `animate_flash_adc`

*`pyDataconverter.utils.visualizations.visualize_FlashADC`*

Animate Flash ADC operation over a sequence of input voltages.

Draws the static reference ladder and comparator bank once, then updates only the
dynamic elements (input voltage line, comparator output dots, code annotation) each
frame using `matplotlib.animation.FuncAnimation`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| adc | `FlashADC` | — | FlashADC instance to animate. |
| input_voltages | `np.ndarray` | — | 1-D array of input voltages. For single-ended ADCs, pass scalars. For differential, pass `(v_pos, v_neg)` tuples. |
| interval | `float` | `0.1` | Time between frames (seconds). |

**Returns**

| Type | Description |
|------|-------------|
| `None` | Displays the animation window; does not return a value. |

**Notes**

- The animation is non-blitting (`blit=False`) for compatibility across matplotlib backends.
- The interval is converted to milliseconds internally for `FuncAnimation`.
- The function calls `plt.show()`, which blocks until the animation window is closed.

**Examples**

```python
import numpy as np
from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.visualizations.visualize_FlashADC import animate_flash_adc

adc = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE)
voltages = np.linspace(0, 1.0, 50)
animate_flash_adc(adc, voltages, interval=0.05)
```

**See Also**

- `visualize_flash_adc` — static or interactive (slider) visualization
- `FlashADC` — the Flash ADC architecture this function animates
