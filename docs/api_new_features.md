# API Reference: New Features

This document covers all new public APIs added in the `feature-roadmap` branch.

---

## 1. Signal Generation (`pyDataconverter/utils/signal_gen.py`)

### `generate_chirp(fs, n_samples, f_start, f_stop, amplitude, offset, method, phi)`

Generate a swept-frequency (chirp) signal using `scipy.signal.chirp`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fs` | `float` | — | Sampling rate (Hz). |
| `n_samples` | `int` | — | Number of output samples. |
| `f_start` | `float` | — | Start frequency (Hz). |
| `f_stop` | `float` | — | Stop frequency (Hz). |
| `amplitude` | `float` | `1.0` | Peak amplitude (V). |
| `offset` | `float` | `0.0` | DC offset (V). |
| `method` | `str` | `'linear'` | Sweep method: `'linear'` or `'logarithmic'`. |
| `phi` | `float` | `0.0` | Phase offset (degrees), passed to `scipy.signal.chirp`. |

**Returns**

| Type | Description |
|------|-------------|
| `Tuple[np.ndarray, np.ndarray]` | `(signal, time)` — both of length `n_samples`. |

**Example**

```python
from pyDataconverter.utils.signal_gen import generate_chirp
sig, t = generate_chirp(fs=1e6, n_samples=4096, f_start=1e3, f_stop=100e3)
```

---

### `generate_prbs(order, n_samples, amplitude, offset, seed)`

Generate a Pseudo-Random Binary Sequence (PRBS) using a maximal-length LFSR.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | `int` | — | LFSR order (2–20). Period = 2^order − 1. |
| `n_samples` | `int` | — | Output length (tiled/truncated to fit). |
| `amplitude` | `float` | `1.0` | Half-range; output values are `±amplitude`. |
| `offset` | `float` | `0.0` | DC offset (V). |
| `seed` | `int \| None` | `None` | Seed for LFSR initial state. `None` uses all-ones (deterministic). |

**Returns**

| Type | Description |
|------|-------------|
| `np.ndarray` | Signal of length `n_samples` with values in `{offset ± amplitude}`. |

**Example**

```python
from pyDataconverter.utils.signal_gen import generate_prbs
seq = generate_prbs(order=7, n_samples=1000, amplitude=0.5, seed=42)
```

---

### `apply_channel(signal, h)`

Apply a channel impulse response to a signal via linear convolution, truncated to the input length.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | `np.ndarray` | — | Input signal. |
| `h` | `np.ndarray` | — | Channel impulse response (FIR coefficients). |

**Returns**

| Type | Description |
|------|-------------|
| `np.ndarray` | Filtered signal, same length as `signal`. |

**Example**

```python
from pyDataconverter.utils.signal_gen import apply_channel
import numpy as np
h = np.array([0.5, 0.3, 0.2])  # simple 3-tap FIR
out = apply_channel(signal, h)
```

---

### `generate_gaussian_noise(n_samples, std, offset, rng)`

Generate Gaussian white noise.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | `int` | — | Number of samples. |
| `std` | `float` | `1.0` | Standard deviation (V). |
| `offset` | `float` | `0.0` | DC offset (V). |
| `rng` | `np.random.Generator \| None` | `None` | Numpy Generator for reproducibility. `None` uses global random state. |

**Returns**

| Type | Description |
|------|-------------|
| `np.ndarray` | Noise array of length `n_samples`. |

**Example**

```python
from pyDataconverter.utils.signal_gen import generate_gaussian_noise
import numpy as np
rng = np.random.default_rng(0)
noise = generate_gaussian_noise(n_samples=4096, std=0.001, rng=rng)
```

---

### `apply_window(signal, window_type)`

Apply a named window function (scipy) to a signal element-wise.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | `np.ndarray` | — | Input signal. |
| `window_type` | `str` | — | Window name: `'hann'`, `'hamming'`, `'blackman'`, `'bartlett'`, `'kaiser'`, `'flattop'`, `'boxcar'`, `'tukey'`, `'cosine'`, `'exponential'`. |

**Returns**

| Type | Description |
|------|-------------|
| `np.ndarray` | Windowed signal, same length as `signal`. |

**Example**

```python
from pyDataconverter.utils.signal_gen import apply_window
windowed = apply_window(signal, 'hann')
```

---

## 2. SAR Variants (`pyDataconverter/components/cdac.py`, `pyDataconverter/architectures/SARADC.py`)

### `RedundantSARCDAC(n_bits, v_ref, radix, cap_mismatch)`

C-DAC with sub-binary radix weights for SAR ADCs with digital error correction (DEC).

Weights follow `r^(N-1-k)` for `k=0..N-1` (MSB first), where `r < 2`. A lookup DEC table is built at construction.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | ADC resolution. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `radix` | `float` | `1.85` | Sub-binary radix `r` in `(1.0, 2.0)`. Typical: 1.8–1.9. |
| `cap_mismatch` | `float` | `0.0` | Std of multiplicative capacitor mismatch. |

**Returns**

Returns a `RedundantSARCDAC` instance. Use `.decode(raw_code)` to apply error correction.

| Method | Description |
|--------|-------------|
| `decode(raw_code) -> int` | Map raw SAR register value to corrected output code via DEC table. |
| `get_voltage(code) -> (float, float)` | Return `(v_dac, 0.0)` for a given code. |

**Example**

```python
from pyDataconverter.components.cdac import RedundantSARCDAC
cdac = RedundantSARCDAC(n_bits=10, v_ref=1.0, radix=1.85)
raw = 0b1001100101
corrected = cdac.decode(raw)
```

---

### `SplitCapCDAC(n_bits, v_ref, n_msb, cap_mismatch)`

Split-capacitor C-DAC with a bridge between coarse (MSB) and fine (LSB) arrays. Uses `n_bits + 1` physical capacitors instead of the `2^n_bits` required by a full binary array.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | Total ADC resolution. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `n_msb` | `int \| None` | `n_bits // 2` | Number of MSB capacitors. Must be in `[1, n_bits-1]`. |
| `cap_mismatch` | `float` | `0.0` | Std of multiplicative capacitor mismatch. |

**Returns**

Returns a `SplitCapCDAC` instance (subclass of `SingleEndedCDAC`).

| Method | Description |
|--------|-------------|
| `get_voltage(code) -> (float, float)` | Return `(v_dac, 0.0)` for a given code. |
| `voltages` property | Array of effective DAC thresholds for all `2^n_bits` codes. |

**Example**

```python
from pyDataconverter.components.cdac import SplitCapCDAC
cdac = SplitCapCDAC(n_bits=12, v_ref=1.0, n_msb=6, cap_mismatch=0.001)
v_dac, _ = cdac.get_voltage(2048)
```

---

### `SegmentedCDAC(n_bits, v_ref, n_therm, cap_mismatch)`

Segmented C-DAC: thermometer-coded MSBs plus binary-weighted LSBs in a single flat weight vector.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | Total ADC resolution. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `n_therm` | `int` | `4` | Number of MSBs as thermometer (1 ≤ n_therm < n_bits). |
| `cap_mismatch` | `float` | `0.0` | Std of multiplicative mismatch applied to all caps. |

**Returns**

Returns a `SegmentedCDAC` instance (implements `CDACBase`).

| Method | Description |
|--------|-------------|
| `get_voltage(code) -> (float, float)` | Decode `code` into thermometer + binary parts, return `(v_dac, 0.0)`. |
| `voltages` property | Array of effective DAC thresholds for all `2^n_bits` codes. |

**Example**

```python
from pyDataconverter.components.cdac import SegmentedCDAC
cdac = SegmentedCDAC(n_bits=12, v_ref=1.0, n_therm=4, cap_mismatch=0.001)
```

---

### `MultibitSARADC(n_bits, v_ref, bits_per_cycle, **kwargs)`

SAR ADC that resolves multiple bits per cycle using a flash sub-ADC, reducing the number of conversion cycles to `ceil(N / bits_per_cycle)`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | Total ADC resolution. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `bits_per_cycle` | `int` | `2` | Bits resolved per SAR cycle. Must be in `[1, n_bits]`. |
| `**kwargs` | — | — | All other `SARADC` parameters (`cap_mismatch`, `noise_rms`, `offset`, `gain_error`, `t_jitter`, `cdac`, `input_type`, etc.). |

**Returns**

Returns a `MultibitSARADC` instance (subclass of `SARADC`). Inherits `.convert()`, `.convert_with_trace()`, and `.reset()`.

**Example**

```python
from pyDataconverter.architectures.SARADC import MultibitSARADC
adc = MultibitSARADC(n_bits=12, v_ref=1.0, bits_per_cycle=2)
code = adc.convert(0.37)
```

---

### `NoiseshapingSARADC(n_bits, v_ref, **kwargs)`

First-order noise-shaping SAR ADC. Accumulates the quantisation residue in an integrator and feeds it back to the next sample, shaping quantisation noise toward higher frequencies.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | ADC resolution. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `**kwargs` | — | — | All other `SARADC` parameters (`cap_mismatch`, `noise_rms`, `offset`, `gain_error`, `t_jitter`, `cdac`, `input_type`, etc.). |

**Returns**

Returns a `NoiseshapingSARADC` instance. `.reset()` also clears the integrator state.

| Attribute | Description |
|-----------|-------------|
| `integrator_state` | Current integrator output (V). Clipped to `±v_ref/2` to prevent runaway. |

**Example**

```python
from pyDataconverter.architectures.SARADC import NoiseshapingSARADC
adc = NoiseshapingSARADC(n_bits=12, v_ref=1.0)
codes = [adc.convert(v) for v in input_signal]
```

---

## 3. Voltage-Mode DACs (`pyDataconverter/architectures/`)

### `ResistorStringDAC(n_bits, v_ref, r_unit, r_mismatch, seed)`

Resistor string (Kelvin divider) DAC: `2^N` equal resistors in series between `V_ref` and GND. Tap voltages are pre-computed at construction so `convert()` is O(1). Inherently monotonic for any mismatch.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | DAC resolution (1–32). |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `r_unit` | `float` | `1e3` | Nominal unit resistor (Ω). |
| `r_mismatch` | `float` | `0.0` | Std of multiplicative resistor mismatch (e.g. `0.01` = 1%). |
| `seed` | `int \| None` | `None` | Random seed for mismatch draw. |

**Returns**

Returns a `ResistorStringDAC` instance (subclass of `DACBase`). Use `.convert(code) -> float`.

| Attribute | Description |
|-----------|-------------|
| `r_values` | Actual resistor values with mismatch, ordered GND → V_ref (length `2^N`). |

**Example**

```python
from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
dac = ResistorStringDAC(n_bits=8, v_ref=1.0, r_mismatch=0.005, seed=0)
v_out = dac.convert(128)  # midscale
```

---

### `R2RDAC(n_bits, v_ref, r_unit, r_mismatch, r2_mismatch, seed)`

R-2R ladder voltage-mode DAC solved via nodal (MNA) analysis. Separate mismatch parameters for horizontal R arms and vertical 2R switch arms. All `2^N` code voltages are pre-computed at construction.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | DAC resolution (1–32). |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `r_unit` | `float` | `1e3` | Nominal R value (Ω); 2R arms use `2 * r_unit`. |
| `r_mismatch` | `float` | `0.0` | Std of multiplicative mismatch for R (horizontal) arms. |
| `r2_mismatch` | `float` | `0.0` | Std of multiplicative mismatch for 2R (vertical) arms. |
| `seed` | `int \| None` | `None` | Random seed for mismatch draw. |

**Returns**

Returns an `R2RDAC` instance (subclass of `DACBase`). Use `.convert(code) -> float`.

| Attribute | Description |
|-----------|-------------|
| `r_values` | Actual R-arm values, length `n_bits - 1`. |
| `r2_values` | Actual 2R-arm values, length `n_bits`. |

**Example**

```python
from pyDataconverter.architectures.R2RDAC import R2RDAC
dac = R2RDAC(n_bits=10, v_ref=1.0, r_mismatch=0.005, r2_mismatch=0.005, seed=0)
v_out = dac.convert(512)
```

---

### `SegmentedResistorDAC(n_bits, v_ref, n_therm, r_unit, r_mismatch, seed)`

Segmented DAC combining a thermometer-coded coarse resistor string (top `n_therm` bits) with a binary R-2R fine sub-DAC (lower `n_bits - n_therm` bits). The fine sub-DAC spans one coarse LSB.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bits` | `int` | — | Total DAC resolution (2–32). |
| `v_ref` | `float` | `1.0` | Full-scale reference voltage (V). |
| `n_therm` | `int` | `4` | Number of MSBs for the thermometer coarse string (1 ≤ n_therm ≤ n_bits-1). |
| `r_unit` | `float` | `1e3` | Nominal unit resistor (Ω). |
| `r_mismatch` | `float` | `0.0` | Std of multiplicative mismatch applied independently to coarse and fine stages. |
| `seed` | `int \| None` | `None` | Random seed for reproducible mismatch draws. |

**Returns**

Returns a `SegmentedResistorDAC` instance (subclass of `DACBase`). Use `.convert(code) -> float`.

| Attribute | Description |
|-----------|-------------|
| `n_therm` | Number of MSBs in the coarse stage. |
| `n_fine` | Number of LSBs in the fine R-2R stage (`= n_bits - n_therm`). |

**Example**

```python
from pyDataconverter.architectures.SegmentedResistorDAC import SegmentedResistorDAC
dac = SegmentedResistorDAC(n_bits=12, v_ref=1.0, n_therm=4, r_mismatch=0.005, seed=0)
v_out = dac.convert(2048)
```

---

## 4. Metrics (`pyDataconverter/utils/metrics/adc.py`)

### `calculate_gain_offset_error(transitions, n_bits, v_ref, quant_mode)`

Calculate offset error and gain error from measured ADC transition voltages.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transitions` | `np.ndarray` | — | Measured transition voltages, length `2^n_bits - 1`. |
| `n_bits` | `int` | — | ADC resolution. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). |
| `quant_mode` | `QuantizationMode` | `FLOOR` | `FLOOR` or `SYMMETRIC` quantisation convention. |

**Returns**

| Key | Type | Description |
|-----|------|-------------|
| `OffsetError` | `float` | Deviation of first transition from ideal (V). |
| `GainError` | `float` | Fractional gain error (dimensionless). |

**Example**

```python
from pyDataconverter.utils.metrics.adc import calculate_gain_offset_error
result = calculate_gain_offset_error(transitions, n_bits=12, v_ref=1.0)
print(result['OffsetError'], result['GainError'])
```

---

### `calculate_adc_iip3(time_data, fs, f1, f2, full_scale)`

Calculate IIP3 and OIP3 from a two-tone ADC output using the standard intercept formula.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_data` | `np.ndarray` | — | ADC output codes (or voltages) as 1-D float array. |
| `fs` | `float` | — | Sampling rate (Hz). |
| `f1` | `float` | — | First tone frequency (Hz). |
| `f2` | `float` | — | Second tone frequency (Hz). |
| `full_scale` | `float \| None` | `None` | Full-scale value for dBFS conversion. `None` → results in dB. |

**Returns**

| Key | Type | Description |
|-----|------|-------------|
| `IIP3_dB` | `float` | Input-referred third-order intercept point (dB). |
| `OIP3_dB` | `float` | Output-referred third-order intercept point (dB). |
| `IM3_dB` | `float` | Average IM3 product level relative to input tones (dBc). |
| `P_in_dB` | `float` | Average input tone power (dB). |
| `P_im3_dB` | `float` | Average IM3 product power (dB). |

**Example**

```python
from pyDataconverter.utils.metrics.adc import calculate_adc_iip3
result = calculate_adc_iip3(codes, fs=1e6, f1=99e3, f2=101e3)
print(f"IIP3 = {result['IIP3_dB']:.1f} dB")
```

---

### `calculate_dynamic_range_from_curve(amplitudes_dB, snr_values)`

Estimate dynamic range from a measured SNR vs. amplitude sweep.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `amplitudes_dB` | `np.ndarray` | — | Input amplitude sweep (dBFS or dB). Largest value is taken as full scale. |
| `snr_values` | `np.ndarray` | — | SNR (dB) at each amplitude point. |

**Returns**

| Key | Type | Description |
|-----|------|-------------|
| `DR_dB` | `float` | Dynamic range (dB): span from full scale to amplitude where SNR = 0 dB. |
| `AmplitudeAtSNR0_dB` | `float` | Amplitude (dBFS) where SNR interpolates to 0 dB. |

**Example**

```python
from pyDataconverter.utils.metrics.adc import calculate_dynamic_range_from_curve
result = calculate_dynamic_range_from_curve(amplitudes_dBFS, snr_values)
print(f"DR = {result['DR_dB']:.1f} dB")
```

---

### `calculate_erbw_from_curve(frequencies, enob_values, enob_ref)`

Estimate effective resolution bandwidth (ERBW) from an ENOB vs. frequency sweep. ERBW is where ENOB drops 0.5 bits below the low-frequency reference.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequencies` | `np.ndarray` | — | Frequency array (Hz), sorted ascending. |
| `enob_values` | `np.ndarray` | — | ENOB at each frequency. |
| `enob_ref` | `float \| None` | `None` | Reference ENOB. Defaults to `enob_values[0]`. |

**Returns**

| Key | Type | Description |
|-----|------|-------------|
| `ERBW_Hz` | `float` | Effective resolution bandwidth (Hz). |
| `ENOB_ref` | `float` | Reference ENOB used. |

**Example**

```python
from pyDataconverter.utils.metrics.adc import calculate_erbw_from_curve
result = calculate_erbw_from_curve(freqs, enob_values)
print(f"ERBW = {result['ERBW_Hz']/1e6:.2f} MHz")
```

---

## 5. Characterization (`pyDataconverter/utils/characterization.py`)

### `measure_dynamic_range(adc, n_bits, v_ref, fs, n_fft, n_fin, n_amplitudes, amplitude_range_dBFS)`

Measure ADC dynamic range by sweeping coherent sine amplitudes and measuring SNR at each step.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adc` | any | — | ADC model implementing `.convert(float) -> int`. |
| `n_bits` | `int` | — | ADC resolution (bits). |
| `v_ref` | `float` | — | Full-scale reference voltage (V). |
| `fs` | `float` | — | Sampling rate (Hz). |
| `n_fft` | `int` | — | FFT / record length (samples per measurement). |
| `n_fin` | `int` | — | Input frequency bin number (coherent with `n_fft`). |
| `n_amplitudes` | `int` | `20` | Number of amplitude steps in the sweep. |
| `amplitude_range_dBFS` | `tuple` | `(-80.0, -1.0)` | `(low_dBFS, high_dBFS)` sweep range. |

**Returns**

| Key | Type | Description |
|-----|------|-------------|
| `DR_dB` | `float` | Dynamic range (dB). |
| `AmplitudeAtSNR0_dBFS` | `float` | Amplitude where SNR = 0 dB, relative to full-scale (dBFS). |
| `AmplitudeAtSNR0_dB` | `float` | Same amplitude converted to absolute dBV-style units (dBFS + 20·log₁₀(v_ref/2)). |
| `Amplitudes_dBFS` | `np.ndarray` | Sweep amplitudes used (dBFS). |
| `SNR_values` | `np.ndarray` | Measured SNR (dB) at each amplitude step. |

**Example**

```python
from pyDataconverter.utils.characterization import measure_dynamic_range
from pyDataconverter.architectures.SARADC import SARADC
adc = SARADC(n_bits=12, v_ref=1.0)
result = measure_dynamic_range(adc, n_bits=12, v_ref=1.0, fs=1e6, n_fft=4096, n_fin=17)
print(f"DR = {result['DR_dB']:.1f} dB")
```

---

### `measure_erbw(adc, n_bits, v_ref, fs, n_fft, freq_range_hz, n_frequencies, amplitude_dBFS)`

Measure ADC effective resolution bandwidth (ERBW) by sweeping coherent sine frequencies and measuring ENOB at each step.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adc` | any | — | ADC model implementing `.convert(float) -> int`. |
| `n_bits` | `int` | — | ADC resolution (bits). |
| `v_ref` | `float` | — | Full-scale reference voltage (V). |
| `fs` | `float` | — | Sampling rate (Hz). |
| `n_fft` | `int` | — | FFT / record length (samples per measurement). |
| `freq_range_hz` | `tuple` | — | `(f_low, f_high)` frequency sweep range (Hz). |
| `n_frequencies` | `int` | `20` | Number of frequency steps (log-spaced). |
| `amplitude_dBFS` | `float` | `-3.0` | Input amplitude (dBFS). |

**Returns**

| Key | Type | Description |
|-----|------|-------------|
| `ERBW_Hz` | `float` | Effective resolution bandwidth (Hz). |
| `ENOB_ref` | `float` | ENOB at the lowest measured frequency. |
| `Frequencies_Hz` | `np.ndarray` | Actual (coherent-snapped) measurement frequencies (Hz). |
| `ENOB_values` | `np.ndarray` | ENOB at each frequency step. |

**Example**

```python
from pyDataconverter.utils.characterization import measure_erbw
from pyDataconverter.architectures.SARADC import SARADC
adc = SARADC(n_bits=12, v_ref=1.0)
result = measure_erbw(adc, n_bits=12, v_ref=1.0, fs=100e6, n_fft=4096,
                      freq_range_hz=(1e3, 40e6))
print(f"ERBW = {result['ERBW_Hz']/1e6:.1f} MHz")
```
