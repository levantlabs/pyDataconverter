## Utilities

### Signal Generation

---

### `convert_to_differential`

*`pyDataconverter.utils.signal_gen`*

Convert a single-ended signal to a differential pair with a specified common-mode voltage.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| signal | np.ndarray | — | Input signal array. |
| vcm | float | 0.0 | Common-mode voltage (V). |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[np.ndarray, np.ndarray] | `(v_pos, v_neg)` differential signal arrays. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_sine, convert_to_differential

sine = generate_sine(frequency=1e3, sampling_rate=1e6, amplitude=0.5)
v_pos, v_neg = convert_to_differential(sine, vcm=0.5)
```

---

### `generate_sine`

*`pyDataconverter.utils.signal_gen`*

Generate a sinusoidal test signal.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| frequency | float | — | Signal frequency (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitude | float | 1.0 | Peak amplitude (V). |
| offset | float | 0.0 | DC offset (V). |
| duration | float | 1.0 | Signal duration (s). |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Signal array of length `int(duration * sampling_rate)`. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_sine

signal = generate_sine(frequency=1e3, sampling_rate=1e6, amplitude=0.5, duration=0.01)
print(len(signal))  # 10000
```

---

### `generate_ramp`

*`pyDataconverter.utils.signal_gen`*

Generate a linearly-spaced ramp signal.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| samples | int | — | Number of samples. |
| v_min | float | 0.0 | Minimum voltage (V). |
| v_max | float | 1.0 | Maximum voltage (V). |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Linearly-spaced array from `v_min` to `v_max`. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_ramp

ramp = generate_ramp(samples=1024, v_min=0.0, v_max=3.3)
print(ramp[0], ramp[-1])  # 0.0 3.3
```

---

### `generate_step`

*`pyDataconverter.utils.signal_gen`*

Generate a multi-level step signal.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| samples | int | — | Total number of samples. |
| step_points | List[int] | — | Sample indices where steps occur. |
| levels | List[float] | — | Voltage levels for each step. The first element sets the initial level; subsequent levels take effect at the corresponding `step_points` index. |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Step signal array. |

**Notes**

- The signal is initialised to zero; `levels[0]` is not applied (only `levels[1:]` are used with `step_points`).

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_step

signal = generate_step(samples=1000, step_points=[0, 250, 500, 750], levels=[0.0, 0.5, 1.0, 1.5])
```

---

### `generate_two_tone`

*`pyDataconverter.utils.signal_gen`*

Generate a two-tone test signal for intermodulation and linearity testing.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| f1 | float | — | First tone frequency (Hz). |
| f2 | float | — | Second tone frequency (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitude1 | float | 0.5 | First tone amplitude (V). |
| amplitude2 | float | 0.5 | Second tone amplitude (V). |
| phase1 | float | 0.0 | First tone phase (rad). |
| phase2 | float | 0.0 | Second tone phase (rad). |
| duration | float | 1.0 | Signal duration (s). |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Sum of the two tones. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_two_tone

signal = generate_two_tone(f1=1e3, f2=1.1e3, sampling_rate=1e6, duration=0.01)
```

---

### `generate_multitone`

*`pyDataconverter.utils.signal_gen`*

Generate a multitone test signal composed of an arbitrary number of sinusoids.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| frequencies | List[float] | — | Tone frequencies (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitudes | List[float] | None | Per-tone amplitudes (V). Defaults to `1/N` for each tone. |
| phases | List[float] | None | Per-tone phases (rad). Defaults to 0.0 for each tone. |
| duration | float | 1.0 | Signal duration (s). |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Summed multitone signal. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If lengths of `frequencies`, `amplitudes`, and `phases` do not match. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_multitone

signal = generate_multitone(frequencies=[1e3, 2e3, 3e3], sampling_rate=1e6, duration=0.01)
```

---

### `generate_imd_tones`

*`pyDataconverter.utils.signal_gen`*

Generate an Intermodulation Distortion (IMD) test signal and compute expected IMD product frequencies.

Two tones are generated at `f1` and `f1 + delta_f`, each with half the specified amplitude. The function also returns a dictionary of expected 2nd- and (optionally) 3rd-order IMD product frequencies.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| f1 | float | — | First tone frequency (Hz). |
| delta_f | float | — | Frequency spacing between tones (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitude | float | 0.5 | Total amplitude, split equally between tones (V). |
| duration | float | 1.0 | Signal duration (s). |
| order | int | 3 | IMD order to calculate (2 or 3). |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[np.ndarray, dict] | `(signal, imd_freqs)` where `imd_freqs` contains keys `'f1'`, `'f2'`, `'imd2'`, and optionally `'imd3'`. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_imd_tones

signal, imd_freqs = generate_imd_tones(f1=1e6, delta_f=1e3, sampling_rate=10e6)
print(imd_freqs['imd3']['2f1-f2'])  # 999000.0
```

---

### `generate_digital_ramp`

*`pyDataconverter.utils.signal_gen`*

Generate a digital ramp signal as an array of integer codes.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | DAC resolution in bits. |
| n_points | int | None | Number of points. If None, generates all `2^n_bits` codes. |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Array of integer codes from 0 to `2^n_bits - 1`. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_digital_ramp

codes = generate_digital_ramp(n_bits=8)
print(len(codes))  # 256
```

---

### `generate_digital_step`

*`pyDataconverter.utils.signal_gen`*

Generate a digital step signal as an array of integer codes.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | DAC resolution in bits. |
| step_points | List[int] | — | Sample indices where steps occur. |
| levels | List[int] | — | Digital codes for each step. |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Array of integer codes. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If any level exceeds `2^n_bits - 1` or is negative. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_digital_step

codes = generate_digital_step(n_bits=12, step_points=[0, 200, 400], levels=[0, 1000, 2000])
```

---

### `generate_digital_sine`

*`pyDataconverter.utils.signal_gen`*

Generate a digital sine wave as integer DAC codes.

The analog sine is scaled by `amplitude` (fraction of full scale), offset by `offset`, and then quantized and clipped to the valid code range `[0, 2^n_bits - 1]`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | DAC resolution in bits. |
| frequency | float | — | Signal frequency (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitude | float | 0.9 | Amplitude as fraction of full scale. |
| offset | float | 0.5 | DC offset as fraction of full scale. |
| duration | float | 1.0 | Signal duration (s). |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Array of integer codes. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_digital_sine

codes = generate_digital_sine(n_bits=12, frequency=1e3, sampling_rate=1e6, duration=0.005)
```

---

### `generate_digital_two_tone`

*`pyDataconverter.utils.signal_gen`*

Generate a digital two-tone signal as integer DAC codes.

The signal is centered at code mid-scale (0.5 offset) and clipped to the valid range.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | DAC resolution in bits. |
| f1 | float | — | First tone frequency (Hz). |
| f2 | float | — | Second tone frequency (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitude1 | float | 0.45 | First tone amplitude as fraction of full scale. |
| amplitude2 | float | 0.45 | Second tone amplitude as fraction of full scale. |
| duration | float | 1.0 | Signal duration (s). |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Array of integer codes. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_digital_two_tone

codes = generate_digital_two_tone(n_bits=12, f1=1e3, f2=1.1e3, sampling_rate=1e6, duration=0.01)
```

---

### `generate_digital_multitone`

*`pyDataconverter.utils.signal_gen`*

Generate a digital multitone signal as integer DAC codes.

If amplitudes are not provided, each tone gets `0.9 / N` of full scale. The signal is centered at mid-scale and clipped to the valid range.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | DAC resolution in bits. |
| frequencies | List[float] | — | Tone frequencies (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitudes | List[float] | None | Per-tone amplitudes as fraction of full scale. Defaults to `0.9 / N` per tone. |
| duration | float | 1.0 | Signal duration (s). |

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Array of integer codes. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_digital_multitone

codes = generate_digital_multitone(n_bits=12, frequencies=[1e3, 2e3, 3e3], sampling_rate=1e6)
```

---

### `generate_digital_imd_tones`

*`pyDataconverter.utils.signal_gen`*

Generate a digital IMD test signal as integer DAC codes along with expected IMD product frequencies.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | DAC resolution in bits. |
| f1 | float | — | First tone frequency (Hz). |
| delta_f | float | — | Frequency spacing (Hz). |
| sampling_rate | float | — | Sampling rate (Hz). |
| amplitude | float | 0.9 | Total amplitude as fraction of full scale. |
| duration | float | 1.0 | Signal duration (s). |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[np.ndarray, dict] | `(digital_codes, imd_freqs)` where `imd_freqs` contains 2nd- and 3rd-order product frequencies. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_digital_imd_tones

codes, imd_freqs = generate_digital_imd_tones(n_bits=12, f1=1e6, delta_f=1e3, sampling_rate=10e6)
```

---

### `generate_coherent_sine`

*`pyDataconverter.utils.signal_gen`*

Generate a coherent sine wave for FFT-based ADC testing.

Coherent sampling places the signal at an exact FFT bin frequency (`f_in = n_fin / n_fft * fs`), eliminating spectral leakage so that dynamic metrics (SNR, SFDR, THD, ENOB) can be measured accurately from a single FFT window.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| sampling_rate | float | — | Sampling rate (Hz). |
| n_fft | int | — | FFT size. Sets duration to `n_fft / sampling_rate`. |
| n_fin | int | — | Input frequency bin number (1 ≤ `n_fin` < `n_fft / 2`). |
| amplitude | float | 1.0 | Peak amplitude (V). |
| offset | float | 0.0 | DC offset (V). |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[np.ndarray, float] | `(signal, f_in)` where `f_in` is the input frequency in Hz. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_coherent_sine

signal, f_in = generate_coherent_sine(sampling_rate=1e6, n_fft=1024, n_fin=11)
print(f_in)  # 10742.1875
```

**See Also**

- `generate_coherent_two_tone` — two-tone variant for IMD measurements.

---

### `generate_coherent_two_tone`

*`pyDataconverter.utils.signal_gen`*

Generate a coherent two-tone signal for FFT-based ADC testing.

Both tones land on exact FFT bin frequencies, eliminating spectral leakage. Useful for IMD and two-tone intermodulation measurements.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| sampling_rate | float | — | Sampling rate (Hz). |
| n_fft | int | — | FFT size. Duration = `n_fft / sampling_rate`. |
| n_fin1 | int | — | First tone frequency bin number. |
| n_fin2 | int | — | Second tone frequency bin number. |
| amplitude1 | float | 0.5 | First tone peak amplitude (V). |
| amplitude2 | float | 0.5 | Second tone peak amplitude (V). |
| phase1 | float | 0.0 | First tone initial phase (rad). |
| phase2 | float | 0.0 | Second tone initial phase (rad). |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[np.ndarray, float, float] | `(signal, f1, f2)` where `f1` and `f2` are the tone frequencies in Hz. |

**Examples**

```python
from pyDataconverter.utils.signal_gen import generate_coherent_two_tone

signal, f1, f2 = generate_coherent_two_tone(sampling_rate=1e6, n_fft=1024, n_fin1=11, n_fin2=23)
```

**See Also**

- `generate_coherent_sine` — single-tone variant.

---

### FFT Analysis

---

### `FFTNormalization`

*`pyDataconverter.utils.fft_analysis`*

Enumeration of FFT normalization modes for `compute_fft`.

**Members**

| Name | Value | Description |
|------|-------|-------------|
| NONE | `'none'` | Raw FFT magnitudes in dB. |
| POWER | `'power'` | Normalize by FFT length (subtract `20·log10(N)`). |
| DBFS | `'dbfs'` | Normalize to full scale (dBFS). Requires `full_scale` parameter. |

**Examples**

```python
from pyDataconverter.utils.fft_analysis import FFTNormalization, compute_fft

freqs, mags = compute_fft(signal, fs=1e6, normalization=FFTNormalization.DBFS, full_scale=2.0)
```

---

### `compute_fft`

*`pyDataconverter.utils.fft_analysis`*

Compute the FFT of a time-domain signal with optional windowing and normalization.

Returns only positive frequencies. DC is optionally removed before the transform.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| time_data | np.ndarray | — | Input signal array. |
| fs | float | — | Sampling frequency (Hz). |
| window | str | None | Window type (`'hann'`, `'hamming'`, `'blackman'`, `'bartlett'`, `'kaiser'`, `'flattop'`, `'boxcar'`, `'tukey'`, `'cosine'`, `'exponential'`). None for rectangular. |
| remove_dc | bool | True | If True, subtract the mean before the FFT. |
| normalization | FFTNormalization | FFTNormalization.NONE | Normalization mode. |
| full_scale | float | None | Full-scale value, required when `normalization` is `DBFS`. |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[np.ndarray, np.ndarray] | `(frequencies, magnitudes_db)` for positive frequencies only. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If `normalization` is `DBFS` and `full_scale` is None. |
| ValueError | If `window` is not in the allowed set. |

**Notes**

- Magnitudes are in dB (`20·log10(|X| + 1e-20)`).
- POWER normalization subtracts `20·log10(N)`.
- DBFS normalization subtracts `20·log10(full_scale) + 20·log10(N/2)`.

**Examples**

```python
from pyDataconverter.utils.fft_analysis import compute_fft
from pyDataconverter.utils.signal_gen import generate_sine

signal = generate_sine(frequency=1e3, sampling_rate=1e6, duration=0.01)
freqs, mags = compute_fft(signal, fs=1e6, window='hann')
```

---

### `find_fundamental`

*`pyDataconverter.utils.fft_analysis`*

Find the fundamental frequency and its magnitude in FFT data.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| freqs | np.ndarray | — | Frequency array from FFT. |
| mags | np.ndarray | — | Magnitude array in dB. |
| f0 | float | — | Expected fundamental frequency (Hz). |
| fs | float | — | Sampling frequency (Hz). |
| tol | float | 0.1 | Frequency tolerance as fraction of bin spacing. |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[float, float] | `(frequency, magnitude)` of the fundamental. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If the fundamental is not found within tolerance. |

**Examples**

```python
from pyDataconverter.utils.fft_analysis import compute_fft, find_fundamental

freqs, mags = compute_fft(signal, fs=1e6)
freq, mag = find_fundamental(freqs, mags, f0=1e3, fs=1e6)
```

**See Also**

- `find_harmonics` — find harmonic frequencies above the fundamental.

---

### `find_harmonics`

*`pyDataconverter.utils.fft_analysis`*

Find harmonic frequencies and their magnitudes, excluding the fundamental.

Handles aliased harmonics that fold back into the Nyquist band.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| freqs | np.ndarray | — | Frequency array from FFT. |
| mags | np.ndarray | — | Magnitude array in dB. |
| f0 | float | — | Fundamental frequency (Hz). |
| fs | float | — | Sampling frequency (Hz). |
| num_harmonics | int | 5 | Number of harmonics to find (excluding fundamental). |
| tol | float | 0.1 | Frequency tolerance as fraction of bin spacing. |
| verbose | bool | False | If True, print harmonic details. |

**Returns**

| Type | Description |
|------|-------------|
| List[Tuple[float, float]] | List of `(frequency, magnitude)` tuples for each harmonic found. |

**Notes**

- Harmonics that alias above Nyquist are folded back correctly.
- If a harmonic is not found within tolerance it is silently skipped (unless `verbose=True`).

**Examples**

```python
from pyDataconverter.utils.fft_analysis import compute_fft, find_harmonics

freqs, mags = compute_fft(signal, fs=1e6)
harmonics = find_harmonics(freqs, mags, f0=1e3, fs=1e6, num_harmonics=5)
for freq, mag in harmonics:
    print(f"{freq:.0f} Hz: {mag:.1f} dB")
```

**See Also**

- `find_fundamental` — locate the fundamental tone.

---

### `demo_fft_analysis`

*`pyDataconverter.utils.fft_analysis`*

Demonstrate FFT analysis functions with various test signals.

Generates four visualisation demos: single tone with harmonics, window function comparison, spectral leakage, and frequency resolution. Intended for interactive / educational use.

**Notes**

- Requires matplotlib for display.
- Calls `plt.show()` — will block in non-interactive environments.

---

### Metrics

---

### `calculate_adc_dynamic_metrics`

*`pyDataconverter.utils.metrics`*

Calculate dynamic ADC performance metrics from either time-domain data or pre-computed FFT data.

Computes SNR, SNDR, SFDR, THD, ENOB, noise floor, and fundamental/harmonic information. When `full_scale` is provided, additional dBFS-referenced variants are included.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| time_data | np.ndarray | None | Input signal array (optional if `freqs`/`mags` provided). |
| fs | float | None | Sampling frequency (Hz). Required if `time_data` is provided. |
| f0 | float | None | Fundamental frequency (Hz). |
| freqs | np.ndarray | None | Frequency array from FFT (optional if `time_data` provided). |
| mags | np.ndarray | None | Magnitude array from FFT (optional if `time_data` provided). |
| full_scale | float | None | Full-scale value for dBFS conversion. If None, results are in dB. |

**Returns**

| Type | Description |
|------|-------------|
| Dict[str, float] | Dictionary with keys: `"SNR"`, `"SNDR"`, `"SFDR"`, `"THD"`, `"NoiseFloor"`, `"ENOB"`, `"Offset"`, `"FundamentalFrequency"`, `"FundamentalMagnitude"`, `"HarmonicFreqs"` (list), `"HarmonicMags"` (list). When `full_scale` is set, additional keys `"*_dBFS"` are included. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If neither `time_data` nor both `freqs` and `mags` are provided. |

**Notes**

- Harmonics up to the 7th are used for THD calculation.
- ENOB is derived as `(SNDR - 1.76) / 6.02`.
- When using the `freqs`/`mags` path with `full_scale`, magnitudes are assumed to already be in dBFS.
- When using the `time_data` path with `full_scale`, the dBFS level correction is applied automatically.

**Examples**

```python
from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics
from pyDataconverter.utils.signal_gen import generate_coherent_sine

signal, f_in = generate_coherent_sine(sampling_rate=1e6, n_fft=1024, n_fin=11)
metrics = calculate_adc_dynamic_metrics(time_data=signal, fs=1e6, f0=f_in)
print(f"SNR: {metrics['SNR']:.1f} dB, ENOB: {metrics['ENOB']:.1f} bits")
```

**See Also**

- `calculate_adc_static_metrics` — static linearity metrics from ramp data.
- `compute_fft` — pre-compute FFT data for the `freqs`/`mags` path.

---

### `calculate_adc_static_metrics`

*`pyDataconverter.utils.metrics`*

Calculate static ADC linearity metrics (DNL, INL, offset, gain error) from ramp test data.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| input_voltages | np.ndarray | — | Monotonic ramp of input voltages. |
| output_codes | np.ndarray | — | Corresponding ADC output codes. |
| n_bits | int | — | ADC resolution in bits. |
| v_ref | float | 1.0 | Reference voltage (V). |

**Returns**

| Type | Description |
|------|-------------|
| Dict[str, float] | Dictionary with keys: `"DNL"` (array), `"INL"` (array), `"Offset"`, `"GainError"`, `"MaxDNL"`, `"MaxINL"`, `"Transitions"` (array). |

**Notes**

- Assumes `input_voltages` is a monotonic ramp and `output_codes` are sorted.
- DNL and INL are in units of LSB.
- Missing codes are handled by repeating the last known transition.

**Examples**

```python
from pyDataconverter.utils.metrics import calculate_adc_static_metrics
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.utils.signal_gen import generate_ramp
import numpy as np

adc = SimpleADC(n_bits=8, v_ref=1.0)
ramp = generate_ramp(samples=10000, v_min=0.0, v_max=1.0)
codes = np.array([adc.convert(v) for v in ramp])
metrics = calculate_adc_static_metrics(ramp, codes, n_bits=8, v_ref=1.0)
print(f"Max DNL: {metrics['MaxDNL']:.3f} LSB, Max INL: {metrics['MaxINL']:.3f} LSB")
```

**See Also**

- `calculate_adc_dynamic_metrics` — frequency-domain dynamic metrics.
- `is_monotonic` — quick monotonicity check.

---

### `is_monotonic`

*`pyDataconverter.utils.metrics`*

Check whether an ADC transfer function is monotonic.

An ADC is monotonic if every code transition voltage is strictly greater than the previous one.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| input_voltages | np.ndarray | — | Monotonic ramp of input voltages. |
| output_codes | np.ndarray | — | Corresponding ADC output codes. |
| n_bits | int | — | ADC resolution in bits. |

**Returns**

| Type | Description |
|------|-------------|
| bool | True if the ADC is monotonic. |

**Examples**

```python
from pyDataconverter.utils.metrics import is_monotonic

result = is_monotonic(input_voltages=ramp, output_codes=codes, n_bits=8)
print(result)  # True
```

**See Also**

- `calculate_adc_static_metrics` — full static analysis including DNL/INL.

---

### `calculate_histogram`

*`pyDataconverter.utils.metrics`*

Calculate a histogram of ADC output codes with optional sine-wave PDF compensation.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| codes | np.ndarray | — | Array of ADC output codes. |
| n_bits | int | — | ADC resolution in bits. |
| input_type | str | `'uniform'` | Input signal type (`'uniform'` or `'sine'`). |
| normalize | bool | True | If True, normalize so non-zero bins sum to 1. |
| remove_pdf | bool | True | If True and `input_type` is `'sine'`, compensate for sine-wave PDF. |

**Returns**

| Type | Description |
|------|-------------|
| Dict[str, np.ndarray] | Dictionary with keys: `"bin_counts"` (hit count per code), `"bin_edges"` (code values), `"missing_codes"` (codes with zero hits), `"unused_range"` (percentage of unused codes). |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If `input_type` is not `'uniform'` or `'sine'`. |

**Notes**

- For sine-wave input, the probability density is `P(x) = 1 / (π·√(A² - x²))`.
- Edge bins (|normalised amplitude| ≥ 0.999) are excluded from PDF compensation to avoid division by zero.

**Examples**

```python
from pyDataconverter.utils.metrics import calculate_histogram
import numpy as np

codes = np.random.randint(0, 256, size=10000)
hist = calculate_histogram(codes, n_bits=8, input_type='uniform')
print(f"Missing codes: {len(hist['missing_codes'])}")
```

**See Also**

- `calculate_adc_static_metrics` — DNL/INL from ramp data.
