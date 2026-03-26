# pyDataconverter API Reference

> Version: 0.02
> Last updated: 2026-03-25 (updated)

## Contents

1. [Core](#core)
   - [ADC Base Classes](#adc-base-classes)
   - [DAC Base Classes](#dac-base-classes)
2. [Architectures](#architectures)
   - [SimpleADC](#simpleadc)
   - [SimpleDAC](#simpledac)
   - [FlashADC](#flashadc)
   - [SARADC](#saradc)
   - [CurrentSteeringDAC](#currentsteeringdac)
3. [Components](#components)
   - [Comparator](#comparator)
   - [Reference](#reference)
   - [CDAC](#cdac)
   - [Decoder](#decoder)
   - [Capacitor](#capacitor)
   - [Current Source](#current-source)
4. [Utilities](#utilities)
   - [Signal Generation](#signal-generation)
   - [FFT Analysis](#fft-analysis)
   - [Metrics](#metrics)
   - [DAC Metrics](#dac-metrics)
5. [Visualization](#visualization)
   - [ADC Plots](#adc-plots)
   - [DAC Plots](#dac-plots)
   - [FFT Plots](#fft-plots)
   - [Flash ADC Visualization](#flash-adc-visualization)
   - [SAR ADC Visualization](#sar-adc-visualization)

---

## Core

### ADC Base Classes

---

### `InputType`

*`pyDataconverter.dataconverter.InputType`*

Enum defining the input configuration for ADC architectures.

**Members**

| Name | Value | Description |
|------|-------|-------------|
| `SINGLE` | `'single'` | Single-ended input; accepts a scalar voltage. |
| `DIFFERENTIAL` | `'differential'` | Differential input; accepts a tuple of (positive, negative) voltages. |

**Examples**

```python
from pyDataconverter.dataconverter import InputType

input_cfg = InputType.SINGLE
print(input_cfg.value)  # 'single'
```

**See Also**

- `ADCBase` — uses `InputType` to determine input validation behaviour

---

### `QuantizationMode`

*`pyDataconverter.dataconverter.QuantizationMode`*

Enum defining the quantization model used by an ADC.

**Members**

| Name | Value | Description |
|------|-------|-------------|
| `FLOOR` | `'floor'` | Standard hardware ADC model consistent with IEEE 1241. All code bins are exactly 1 LSB wide; quantization error ranges from 0 to +LSB. Formula: `code = floor(vin * 2^N / v_ref)`. |
| `SYMMETRIC` | `'symmetric'` | DSP / signal-processing model with zero-mean quantization error. First and last bins are half-width (LSB/2); all middle bins are 1 LSB wide. Formula: `code = floor(vin * (2^N - 1) / v_ref + 0.5)`. |

**Notes**

- `FLOOR` mode uses `LSB = v_ref / 2^N`, giving a quantization error of 0 to +LSB (positive bias).
- `SYMMETRIC` mode uses `LSB = v_ref / (2^N - 1)`, giving a quantization error of -LSB/2 to +LSB/2 (zero mean), which is preferable for quantization noise analysis.

**Examples**

```python
from pyDataconverter.dataconverter import QuantizationMode

mode = QuantizationMode.FLOOR
print(mode.value)  # 'floor'
```

**See Also**

- `SimpleADC` — accepts `QuantizationMode` as a constructor parameter

---

### `ADCBase`

*`pyDataconverter.dataconverter.ADCBase`*

Abstract base class for all ADC architectures.

`ADCBase` provides input validation and a public `convert` method that delegates to architecture-specific implementations via the abstract `_convert_input` method. It cannot be instantiated directly.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_bits` | `int` | — | ADC resolution in bits. Must be between 1 and 32 inclusive. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). Must be positive. |
| `input_type` | `InputType` | `InputType.DIFFERENTIAL` | Input configuration: single-ended or differential. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `n_bits` | `int` | ADC resolution in bits. |
| `v_ref` | `float` | Reference voltage. |
| `input_type` | `InputType` | Input configuration. |

**Methods**

#### `convert(vin)`

Convert an analog input to a digital output code.

| Name | Type | Description |
|------|------|-------------|
| `vin` | `float \| tuple[float, float]` | Analog input voltage. A scalar for single-ended mode, or a `(positive, negative)` tuple for differential mode. |

**Returns**

| Type | Description |
|------|-------------|
| `int` | Digital output code determined by the architecture-specific implementation. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `TypeError` | If `n_bits` is not an integer. |
| `TypeError` | If `v_ref` is not a number. |
| `TypeError` | If `input_type` is not an `InputType` enum member. |
| `TypeError` | If `vin` is not a number in single-ended mode, or not a 2-tuple in differential mode. |
| `ValueError` | If `n_bits` is not between 1 and 32. |
| `ValueError` | If `v_ref` is not positive. |

**Notes**

- This is an abstract base class; subclass it and implement `_convert_input` to create a concrete ADC.
- Input type validation is performed in `convert` before delegating to `_convert_input`.

**Examples**

```python
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType

# ADCBase cannot be instantiated directly; use a concrete subclass
adc = SimpleADC(n_bits=8, v_ref=3.3, input_type=InputType.SINGLE)
code = adc.convert(1.65)
print(code)  # midscale code
```

**See Also**

- `SimpleADC` — concrete single-ended / differential ADC implementation
- `FlashADC` — concrete flash ADC implementation
- `InputType` — enum for input configuration

---

### DAC Base Classes

---

### `OutputType`

*`pyDataconverter.dataconverter.OutputType`*

Enum defining the output configuration for DAC architectures.

**Members**

| Name | Value | Description |
|------|-------|-------------|
| `SINGLE` | `'single'` | Single-ended output; produces a scalar voltage. |
| `DIFFERENTIAL` | `'diff'` | Differential output; produces a `(positive, negative)` voltage tuple. |

**Examples**

```python
from pyDataconverter.dataconverter import OutputType

output_cfg = OutputType.DIFFERENTIAL
print(output_cfg.value)  # 'diff'
```

**See Also**

- `DACBase` — uses `OutputType` to determine output format

---

### `DACBase`

*`pyDataconverter.dataconverter.DACBase`*

Abstract base class for all DAC architectures.

`DACBase` provides input code validation and a public `convert` method that delegates to architecture-specific implementations via the abstract `_convert_input` method. It cannot be instantiated directly.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_bits` | `int` | — | DAC resolution in bits. Must be between 1 and 32 inclusive. |
| `v_ref` | `float` | `1.0` | Reference voltage (V). Must be positive. |
| `output_type` | `OutputType` | `OutputType.SINGLE` | Output configuration: single-ended or differential. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `n_bits` | `int` | DAC resolution in bits. |
| `v_ref` | `float` | Reference voltage. |
| `output_type` | `OutputType` | Output configuration. |
| `lsb` | `float` | Least significant bit size, computed as `v_ref / (2^n_bits - 1)`. |

**Methods**

#### `convert(digital_input)`

Convert a digital input code to an analog output voltage.

| Name | Type | Description |
|------|------|-------------|
| `digital_input` | `int` | Digital input code. Must be between 0 and `2^n_bits - 1` inclusive. |

**Returns**

| Type | Description |
|------|-------------|
| `float \| tuple[float, float]` | Output voltage. A scalar for single-ended mode, or a `(positive, negative)` tuple for differential mode. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `TypeError` | If `n_bits` is not an integer. |
| `TypeError` | If `v_ref` is not a number. |
| `TypeError` | If `output_type` is not an `OutputType` enum member. |
| `TypeError` | If `digital_input` is not an integer. |
| `ValueError` | If `n_bits` is not between 1 and 32. |
| `ValueError` | If `v_ref` is not positive. |
| `ValueError` | If `digital_input` is out of range `[0, 2^n_bits - 1]`. |

**Notes**

- This is an abstract base class; subclass it and implement `_convert_input` to create a concrete DAC.
- The `lsb` attribute uses the formula `v_ref / (2^n_bits - 1)`, mapping code 0 to 0 V and the maximum code to `v_ref`.
- `digital_input` accepts both Python `int` and NumPy integer types.

**Examples**

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType

# DACBase cannot be instantiated directly; use a concrete subclass
dac = SimpleDAC(n_bits=10, v_ref=2.5, output_type=OutputType.SINGLE)
voltage = dac.convert(512)
print(f"{voltage:.4f}")  # approximately mid-scale voltage
```

**See Also**

- `SimpleDAC` — concrete DAC implementation with non-ideality support
- `OutputType` — enum for output configuration

---

## Architectures

---

### `SimpleADC`

*`pyDataconverter.architectures.SimpleADC`*

ADC with ideal quantization and optional first-order non-idealities.

SimpleADC inherits from `ADCBase` and implements a straightforward quantization
model with two selectable modes (FLOOR and SYMMETRIC). It optionally applies
thermal noise, DC offset, gain error, and aperture jitter to the input before
quantization. All non-ideality parameters default to zero (disabled).

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | Resolution in bits (1–32). |
| v_ref | float | `1.0` | Reference voltage (V). |
| input_type | InputType | `InputType.DIFFERENTIAL` | Single-ended or differential input mode. |
| quant_mode | QuantizationMode | `QuantizationMode.FLOOR` | Quantization model — FLOOR (standard hardware) or SYMMETRIC (DSP / zero-mean error). |
| noise_rms | float | `0.0` | Input-referred RMS thermal noise voltage (V). Must be >= 0. |
| offset | float | `0.0` | Input-referred DC offset voltage (V). |
| gain_error | float | `0.0` | Fractional gain error (dimensionless). 0.01 = +1 %. |
| t_jitter | float | `0.0` | RMS aperture jitter (s). Must be >= 0. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| n_bits | int | Resolution in bits. |
| v_ref | float | Reference voltage (V). |
| input_type | InputType | Input mode. |
| quant_mode | QuantizationMode | Active quantization model. |
| noise_rms | float | Input-referred RMS thermal noise voltage (V). |
| offset | float | Input-referred DC offset voltage (V). |
| gain_error | float | Fractional gain error. |
| t_jitter | float | RMS aperture jitter (s). |

**Methods**

#### `convert(vin, dvdt=0.0)`

Convert one analog sample to a digital code.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| vin | float \| tuple[float, float] | — | Voltage (single-ended) or (v_pos, v_neg) tuple (differential). |
| dvdt | float | `0.0` | Signal slope at the sampling instant (V/s). Only used when `t_jitter > 0`. |

**Returns**

| Type | Description |
|------|-------------|
| int | Output code in [0, 2^n_bits − 1]. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| TypeError | `quant_mode` is not a `QuantizationMode` enum. |
| TypeError | Single-ended input is not a number, or differential input is not a 2-tuple. |
| ValueError | `noise_rms` is negative. |
| ValueError | `t_jitter` is negative. |

**Notes**

- Non-idealities are applied in the order: gain error → offset → thermal noise → aperture jitter.
- Aperture jitter only contributes when both `t_jitter > 0` and `dvdt != 0`.
- FLOOR mode: all bins are exactly 1 LSB wide; quantization error is [0, +LSB).
- SYMMETRIC mode: end bins are LSB/2 wide; quantization error is [−LSB/2, +LSB/2].
- The input is clipped to the valid voltage range before quantization.

**Examples**

```python
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType

adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)
code = adc.convert(0.5)
print(code)  # 2048
```

**See Also**

- `ADCBase` — abstract base class providing the `convert()` / `_convert_input()` contract
- `FlashADC` — parallel comparator-based ADC architecture
- `QuantizationMode` — enum selecting FLOOR vs SYMMETRIC quantization

---

### `SimpleDAC`

*`pyDataconverter.architectures.SimpleDAC`*

DAC with ideal conversion and optional first-order non-idealities.

SimpleDAC inherits from `DACBase` and converts an integer code to an analog
voltage. It optionally applies gain error, DC offset, and output noise. All
non-ideality parameters default to zero (disabled).

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | Resolution in bits (1–32). |
| v_ref | float | `1.0` | Reference voltage (V). |
| output_type | OutputType | `OutputType.SINGLE` | Single-ended or differential output mode. |
| noise_rms | float | `0.0` | Output-referred RMS noise voltage (V). Must be >= 0. |
| offset | float | `0.0` | Output DC offset voltage (V). |
| gain_error | float | `0.0` | Fractional gain error (dimensionless). 0.01 = +1 %. |
| fs | float | `1.0` | Sample rate (Hz). Used by `convert_sequence` to generate the time axis. |
| oversample | int | `1` | Zero-order-hold oversampling factor. Each code is repeated this many times in the output of `convert_sequence`. Must be >= 1. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| n_bits | int | Resolution in bits. |
| v_ref | float | Reference voltage (V). |
| output_type | OutputType | Output mode. |
| lsb | float | Least significant bit size (V), equal to `v_ref / (2^n_bits − 1)`. |
| noise_rms | float | Output-referred RMS noise voltage (V). |
| offset | float | Output DC offset voltage (V). |
| gain_error | float | Fractional gain error. |
| fs | float | Sample rate (Hz). |
| oversample | int | Zero-order-hold oversampling factor. |

**Methods**

#### `convert_sequence(codes)`

Convert an array of digital codes to a time-domain zero-order-hold (ZOH) waveform.

Each code is held for `oversample` output samples. The time axis is spaced at
`1 / (fs * oversample)` seconds. Non-idealities (gain error, offset, noise) are
applied to the full oversampled waveform. Out-of-range codes are clipped to
`[0, 2^n_bits − 1]`.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| codes | numpy.ndarray | — | 1-D array of integer DAC codes. |

**Returns**

| Type | Description |
|------|-------------|
| tuple[numpy.ndarray, numpy.ndarray] | `(t, voltages)` — time vector and voltage waveform (single-ended mode). |
| tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | `(t, v_pos, v_neg)` — time vector and differential voltages (differential mode). |

#### `convert(digital_input)`

Convert a single digital code to an analog output voltage (inherited from `DACBase`).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| digital_input | int | — | Input code in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| float | Output voltage (single-ended mode). |
| tuple[float, float] | (v_pos, v_neg) voltages (differential mode). |

**Raises**

| Exception | Condition |
|-----------|-----------|
| TypeError | `digital_input` is not an integer (raised by `convert`). |
| ValueError | `digital_input` is outside the valid code range (raised by `convert`). |
| ValueError | `noise_rms` is negative (raised at construction). |
| ValueError | `oversample` is less than 1 (raised at construction). |

**Notes**

- Non-idealities are applied in the order: gain error → offset → noise.
- In differential mode the output is centered around `v_ref / 2`: `v_pos = v_diff/2 + v_ref/2`, `v_neg = −v_diff/2 + v_ref/2`.
- The ideal voltage is computed as `digital_input * lsb`.
- `convert_sequence` applies noise *after* oversampling so each held sample receives independent noise.
- `convert_sequence` clips out-of-range codes silently rather than raising an error.

**Examples**

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
import numpy as np

# Single-sample conversion
dac = SimpleDAC(n_bits=12, v_ref=1.0)
voltage = dac.convert(2048)
print(f"{voltage:.4f}")  # 0.5001

# ZOH waveform with oversampling
dac = SimpleDAC(n_bits=8, v_ref=3.3, fs=1000.0, oversample=4)
codes = np.array([0, 128, 255])
t, v = dac.convert_sequence(codes)
print(len(t))  # 12  (3 codes × 4 oversample)
```

**See Also**

- `DACBase` — abstract base class providing the `convert()` / `_convert_input()` contract
- `OutputType` — enum selecting single-ended vs differential output

---

### `EncoderType`

*`pyDataconverter.architectures.FlashADC`*

Thermometer-to-binary encoding strategy for Flash ADC.

| Member | Value | Description |
|--------|-------|-------------|
| `COUNT_ONES` | `'count_ones'` | Counts asserted comparator outputs. Robust to bubble errors: each bubble reduces the code by 1 rather than producing a sparkle error. |
| `XOR` | `'xor'` | XORs adjacent thermometer bits to produce a one-hot intermediate, then maps to binary via OR gates. Mirrors a standard hardware ROM encoder. Bubble errors produce sparkle codes. |

**Examples**

```python
from pyDataconverter.architectures.FlashADC import EncoderType

enc = EncoderType.COUNT_ONES
print(enc.value)  # 'count_ones'
```

---

### `FlashADC`

*`pyDataconverter.architectures.FlashADC`*

Flash (parallel) ADC implementation with configurable non-idealities and encoder.

FlashADC inherits from `ADCBase` and models a parallel comparator architecture.
Each of the 2^N − 1 comparators compares the input against a reference tap from
a resistor ladder. The thermometer code is then converted to binary using the
selected encoder strategy. Per-comparator offsets, reference ladder mismatch,
reference noise, and comparator-level non-idealities are all configurable.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | Resolution in bits (1–32). |
| v_ref | float | `1.0` | Reference voltage (V). |
| input_type | InputType | `InputType.SINGLE` | Single-ended or differential input mode. |
| comparator_type | Type[ComparatorBase] | `DifferentialComparator` | Comparator class to instantiate for each stage. |
| comparator_params | dict \| None | `None` | Keyword arguments passed to every comparator (e.g. `noise_rms`, `hysteresis`). The `'offset'` key is reserved for per-comparator offsets set via `offset_std`. |
| offset_std | float | `0.0` | Standard deviation of comparator input-referred offsets (V). Drawn once at construction from N(0, offset_std). |
| reference | ReferenceBase \| None | `None` | Voltage reference instance. If provided, `reference_noise` and `resistor_mismatch` are ignored. Must have exactly 2^n_bits − 1 taps. |
| reference_noise | float | `0.0` | RMS dynamic noise for the default `ReferenceLadder` (V). Ignored when `reference` is provided. |
| resistor_mismatch | float | `0.0` | Resistor mismatch standard deviation for the default `ReferenceLadder`. Ignored when `reference` is provided. |
| encoder_type | EncoderType | `EncoderType.COUNT_ONES` | Thermometer-to-binary encoding strategy. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| n_bits | int | Resolution in bits. |
| v_ref | float | Reference voltage (V). |
| input_type | InputType | Input mode. |
| n_comparators | int | Number of comparators (2^n_bits − 1). |
| comparators | list[ComparatorBase] | Per-comparator instances with individual offsets. |
| reference | ReferenceBase | Voltage reference generator (ladder or user-supplied). |
| encoder_type | EncoderType | Active encoding strategy. |

**Methods**

#### `convert(vin)`

Convert one analog sample to a digital code (inherited from `ADCBase`).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| vin | float \| tuple[float, float] | — | Voltage (single-ended) or (v_pos, v_neg) tuple (differential). |

**Returns**

| Type | Description |
|------|-------------|
| int | Output code in [0, 2^n_bits − 1]. |

#### `reset()`

Reset all comparator states (hysteresis history, bandwidth filter).

**Properties**

#### `reference_voltages`

| Type | Description |
|------|-------------|
| numpy.ndarray | Static reference voltages (no noise). Alias for `reference.voltages`. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| TypeError | `encoder_type` is not an `EncoderType` enum. |
| TypeError | `reference` is provided but is not a `ReferenceBase` instance. |
| TypeError | Single-ended input is not a number, or differential input is not a 2-tuple. |
| ValueError | `reference` tap count does not match 2^n_bits − 1. |

**Notes**

- In differential mode the reference ladder spans [−v_ref/4, +v_ref/4] so that the effective differential thresholds cover [−v_ref/2, +v_ref/2].
- In single-ended mode the ladder spans [0, v_ref].
- The `COUNT_ONES` encoder is robust to bubble errors; `XOR` mirrors hardware but produces sparkle codes on bubbles.
- Per-comparator offsets are drawn once at construction and held fixed for the lifetime of the instance.
- Bandwidth modelling via `Comparator.compare(time_step=...)` is supported by the comparator interface but not yet wired through `FlashADC`.

**Examples**

```python
from pyDataconverter.architectures.FlashADC import FlashADC

adc = FlashADC(n_bits=3, v_ref=1.0)
code = adc.convert(0.4)
print(code)  # 3
```

**See Also**

- `ADCBase` — abstract base class providing the `convert()` / `_convert_input()` contract
- `EncoderType` — enum selecting COUNT_ONES vs XOR encoding
- `ComparatorBase` / `DifferentialComparator` — comparator component used in each stage
- `ReferenceBase` / `ReferenceLadder` — voltage reference component
- `SimpleADC` — simpler ADC model without explicit comparator architecture

---

## Components

### Comparator

---

### `ComparatorBase`

*`pyDataconverter.components.comparator`*

Abstract base class defining the interface for all comparator models.

All comparator implementations must subclass `ComparatorBase` and implement `compare()` with
the 4-input differential signature and `reset()` for clearing internal state. This class
cannot be instantiated directly.

**Methods**

| Name | Description |
|------|-------------|
| `compare(v_pos, v_neg, v_refp=0.0, v_refn=0.0, time_step=None)` | Compare two differential inputs against an optional differential reference. Returns `1` if `(v_pos - v_refp) - (v_neg - v_refn) > threshold`, else `0`. |
| `reset()` | Reset internal state (hysteresis history, bandwidth filter). |

**Notes**

- The 4-input `compare()` signature allows architectures like Flash ADC to pass signal and reference rails separately.
- When `v_refp = v_refn = 0` (defaults), the comparator reduces to a pure differential comparator (`v_pos - v_neg`).

**See Also**

- `DifferentialComparator` — concrete implementation with offset, noise, bandwidth, and hysteresis non-idealities

---

### `DifferentialComparator`

*`pyDataconverter.components.comparator`*

Differential latch comparator with optional parallel reference injection and configurable non-idealities.

Computes an effective input difference as `(v_pos - v_refp) - (v_neg - v_refn) + offset`, then applies
bandwidth limiting (first-order low-pass), input-referred noise, and hysteresis in that order. With default
reference voltages (`v_refp = v_refn = 0`) the comparator behaves as a classic differential comparator.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| offset | float | 0.0 | DC input-referred offset voltage (V). |
| noise_rms | float | 0.0 | RMS input-referred noise voltage (V). |
| bandwidth | float or None | None | -3 dB bandwidth (Hz). `None` for infinite bandwidth. |
| hysteresis | float | 0.0 | Hysteresis window voltage (V), symmetric around the threshold. |
| time_constant | float | 0.0 | Reserved for future temporal modelling (s). |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| offset | float | DC input-referred offset voltage (V). |
| noise_rms | float | RMS input-referred noise voltage (V). |
| bandwidth | float or None | -3 dB bandwidth (Hz), or `None` for infinite. |
| hysteresis | float | Hysteresis window voltage (V). |
| time_constant | float | Time constant for temporal behaviour (s). |

**Methods**

#### `compare(v_pos, v_neg, v_refp=0.0, v_refn=0.0, time_step=None)`

Compare `(v_pos - v_refp)` against `(v_neg - v_refn)` with non-idealities applied.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| v_pos | float | — | Positive signal input. |
| v_neg | float | — | Negative signal input. |
| v_refp | float | 0.0 | Positive reference voltage. |
| v_refn | float | 0.0 | Negative reference voltage. |
| time_step | float or None | None | Time step for bandwidth calculations. Required when `bandwidth` is set. |

**Returns**

| Type | Description |
|------|-------------|
| int | `1` if effective input > threshold, else `0`. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If `bandwidth` is set but `time_step` is `None`. |

#### `reset()`

Reset hysteresis history and bandwidth filter state to initial values.

**Notes**

- Non-ideality application order: offset -> bandwidth limiting -> noise -> hysteresis.
- Bandwidth limiting uses a first-order IIR low-pass filter with time constant `tau = 1 / (2 * pi * bandwidth)`.
- Noise is drawn from a zero-mean Gaussian distribution with standard deviation `noise_rms` on every call.
- Hysteresis shifts the decision threshold by `+/- hysteresis/2` depending on the previous output.
- The backward-compatibility alias `Comparator = DifferentialComparator` is provided at module level.

**Examples**

```python
from pyDataconverter.components.comparator import DifferentialComparator

# Ideal comparator
comp = DifferentialComparator()
print(comp.compare(0.6, 0.4))  # 1

# Comparator with non-idealities
comp = DifferentialComparator(offset=0.001, noise_rms=0.0005, hysteresis=0.002)
comp.reset()
print(comp.compare(0.5, 0.0))  # 1
```

**See Also**

- `ComparatorBase` — abstract base class defining the comparator interface
- `FlashADC` — uses `DifferentialComparator` as the default comparator in its comparator bank

---

### Reference

---

### `ReferenceBase`

*`pyDataconverter.components.reference`*

Abstract base class for voltage reference generators.

A reference generator owns a set of threshold voltages used by the comparator bank in a Flash ADC.
Static non-idealities (e.g. resistor mismatch) are fixed at construction; dynamic noise is redrawn
on every call to `get_voltages()`. This class cannot be instantiated directly.

**Properties**

| Name | Type | Description |
|------|------|-------------|
| n_references | int | Number of reference voltages (= 2^N - 1 for an N-bit Flash ADC). |
| voltages | np.ndarray | Static (noiseless) reference voltages. |

**Methods**

| Name | Description |
|------|-------------|
| `get_voltages()` | Return reference voltages for one conversion, with dynamic noise redrawn each call. |

**See Also**

- `ReferenceLadder` — uniform resistor-ladder implementation
- `ArbitraryReference` — user-supplied threshold array implementation

---

### `ReferenceLadder`

*`pyDataconverter.components.reference`*

Uniform resistor-ladder voltage reference generator.

Generates `2^n_bits - 1` equally-spaced threshold voltages between `v_min` and `v_max`, with optional
static resistor mismatch (drawn once at construction) and per-sample dynamic noise (redrawn on every
call to `get_voltages()`).

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | ADC resolution; determines the number of taps (`2^n_bits - 1`). |
| v_min | float | — | Bottom of the reference range (V). |
| v_max | float | — | Top of the reference range (V). |
| resistor_mismatch | float | 0.0 | Standard deviation of multiplicative resistor mismatch (e.g. 0.01 = 1%). Drawn once at construction. |
| noise_rms | float | 0.0 | RMS dynamic noise added to every reference voltage on each call to `get_voltages()` (V). |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| n_references | int | Number of reference taps (`2^n_bits - 1`). |
| voltages | np.ndarray | Static thresholds including mismatch (no noise). Returns a copy. |
| noise_rms | float | RMS dynamic noise applied per conversion (V). |

**Methods**

#### `get_voltages()`

Return reference voltages for one conversion with dynamic noise applied.

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Array of length `n_references` with static mismatch and dynamic noise applied. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If `v_max <= v_min`. |
| ValueError | If `resistor_mismatch < 0`. |
| ValueError | If `noise_rms < 0`. |

**Notes**

- Ideal thresholds are computed via `np.linspace(v_min, v_max, n_references + 2)[1:-1]`, excluding the endpoints.
- Resistor mismatch is multiplicative: `threshold_i *= (1 + N(0, mismatch))`, drawn once at construction.
- Dynamic noise is additive: `threshold_i += N(0, noise_rms)`, redrawn on each `get_voltages()` call.
- The `voltages` property always returns a copy to prevent external mutation.

**Examples**

```python
from pyDataconverter.components.reference import ReferenceLadder

# 3-bit ideal reference ladder (7 taps from 0 V to 1 V)
ref = ReferenceLadder(n_bits=3, v_min=0.0, v_max=1.0)
print(ref.n_references)      # 7
print(ref.get_voltages())    # array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
```

**See Also**

- `ReferenceBase` — abstract base class
- `ArbitraryReference` — for non-uniform reference profiles
- `FlashADC` — consumes `ReferenceLadder` as its default reference generator

---

### `ArbitraryReference`

*`pyDataconverter.components.reference`*

User-defined voltage reference accepting an explicit array of threshold voltages.

Allows non-uniform spacing or any custom reference profile. Optional dynamic noise can be added
per conversion via `get_voltages()`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| thresholds | array-like | — | Array of reference voltages. Must be a non-empty, strictly increasing 1-D sequence of finite values. |
| noise_rms | float | 0.0 | RMS dynamic noise added on each `get_voltages()` call (V). |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| n_references | int | Number of thresholds. |
| voltages | np.ndarray | Static threshold array (no noise). Returns a copy. |
| noise_rms | float | RMS dynamic noise applied per conversion (V). |

**Methods**

#### `get_voltages()`

Return reference voltages for one conversion with dynamic noise applied.

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Array of length `n_references` with dynamic noise applied. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | If `thresholds` is not a non-empty 1-D array. |
| ValueError | If `thresholds` contains NaN or Inf values. |
| ValueError | If `thresholds` is not strictly increasing. |
| ValueError | If `noise_rms < 0`. |

**Examples**

```python
from pyDataconverter.components.reference import ArbitraryReference

# Custom non-uniform thresholds
ref = ArbitraryReference(thresholds=[0.1, 0.3, 0.4, 0.8])
print(ref.n_references)      # 4
print(ref.get_voltages())    # array([0.1, 0.3, 0.4, 0.8])
```

**See Also**

- `ReferenceBase` — abstract base class
- `ReferenceLadder` — for uniform resistor-ladder references

---

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

---

### DAC Metrics

Functions for calculating DAC-specific static and dynamic performance metrics.

---

### `calculate_dac_static_metrics`

*`pyDataconverter.utils.dac_metrics`*

Compute static linearity metrics by sweeping digital codes through a DAC instance.

Drives every code (or an evenly-spaced subset) through the DAC, records the output
voltage, and derives DNL, INL, offset, and gain error.  For differential DACs the
output is collapsed to a single voltage via `v_pos - v_neg` before computation.
The INL reference line uses an endpoint fit per IEEE 1057.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| dac | `DACBase` | — | A DAC instance to characterize. Must expose `n_bits`, `v_ref`, `output_type`, and `convert(code)`. |
| n_points | `int` or `None` | `None` | Number of evenly-spaced codes to sweep. When `None`, all `2**n_bits` codes are used. Must be >= 2 if provided. |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | Dictionary with the following keys: |

| Key | Type | Description |
|-----|------|-------------|
| `DNL` | `np.ndarray` | Differential non-linearity per code step (LSB). Length is `len(Codes) - 1`. |
| `INL` | `np.ndarray` | Integral non-linearity, endpoint-fit (LSB). Length equals `len(Codes)`. |
| `MaxDNL` | `float` | Maximum absolute DNL value (LSB). |
| `MaxINL` | `float` | Maximum absolute INL value (LSB). |
| `Offset` | `float` | DC offset voltage at code 0 (V). |
| `GainError` | `float` | Fractional gain error (dimensionless). 0.01 means +1 %. |
| `Codes` | `np.ndarray` | The digital codes that were swept. |
| `Voltages` | `np.ndarray` | The measured output voltage for each code. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `TypeError` | If `dac` is not a `DACBase` instance. |
| `ValueError` | If `n_points` is less than 2. |

**Notes**

- LSB is defined as `v_ref / (2**n_bits - 1)` so that the maximum code maps exactly to `v_ref`.
- For differential DACs the output voltage is `v_pos - v_neg`.
- INL uses an endpoint-fit reference line per IEEE 1057.

**Examples**

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.utils.dac_metrics import calculate_dac_static_metrics

dac = SimpleDAC(n_bits=8, v_ref=1.0)
result = calculate_dac_static_metrics(dac)
print(result['MaxDNL'])   # 0.0
print(result['Offset'])   # 0.0
```

**See Also**

- `calculate_adc_static_metrics` — equivalent static metrics for ADCs.
- `DACBase` — abstract base class that `dac` must implement.

---

### `calculate_dac_dynamic_metrics`

*`pyDataconverter.utils.dac_metrics`*

Compute frequency-domain performance metrics from a captured DAC output waveform.

Performs an FFT on the captured voltage samples and calculates SNR, SNDR, SFDR,
THD, and ENOB within the requested Nyquist zone of the DAC update rate.  Supports
multi-zone analysis for characterizing DAC images in higher Nyquist zones.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| voltages | `np.ndarray` | — | 1-D array of captured DAC output samples (V), sampled at rate `fs`. |
| fs | `float` | — | Capture sampling rate (Hz). |
| fs_update | `float` or `None` | `None` | DAC update rate (Hz). Defaults to `fs` when `None` (no oversampling). |
| nyquist_zone | `int` | `1` | Which Nyquist zone to analyse. Zone *i* covers `[(i-1)*fs_update/2, i*fs_update/2)`. |
| window | `str` | `'hann'` | Window function name passed to `compute_fft`. |
| full_scale | `float` or `None` | `None` | Full-scale voltage for dBFS normalisation. When `None`, metrics are in absolute dB. |

**Returns**

| Type | Description |
|------|-------------|
| `dict` | Dictionary with the following keys: |

| Key | Type | Description |
|-----|------|-------------|
| `SNR` | `float` | Signal-to-noise ratio (dB), fundamental vs in-zone noise. |
| `SNDR` | `float` | Signal-to-noise-and-distortion ratio (dB). |
| `SFDR` | `float` | Spurious-free dynamic range (dB), fundamental vs worst spur. |
| `THD` | `float` | Total harmonic distortion (dB), in-zone harmonics only. |
| `ENOB` | `float` | Effective number of bits, `(SNDR - 1.76) / 6.02`. |
| `FundamentalFrequency` | `float` | Detected fundamental frequency (Hz). |
| `FundamentalMagnitude` | `float` | Fundamental magnitude (dB or dBFS). |
| `ZoneBandHz` | `tuple[float, float]` | `(f_low, f_high)` edges of the analysed Nyquist zone (Hz). |
| `NyquistZone` | `int` | The zone number that was analysed. |

When `full_scale` is not `None`, the following additional keys are included:

| Key | Type | Description |
|-----|------|-------------|
| `SNR_dBFS` | `float` | SNR referenced to full scale. |
| `SNDR_dBFS` | `float` | SNDR referenced to full scale. |
| `SFDR_dBFS` | `float` | SFDR referenced to full scale. |
| `THD_dBFS` | `float` | THD referenced to full scale. |
| `FundamentalMagnitude_dBFS` | `float` | Fundamental magnitude referenced to full scale. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `fs < nyquist_zone * fs_update` (requested zone is not visible in the captured spectrum). |
| `ValueError` | If `nyquist_zone < 1`. |
| `ValueError` | If `voltages` is not a non-empty 1-D array. |

**Notes**

- Nyquist zone *i* spans `[(i-1)*fs_update/2, i*fs_update/2)`. For a DAC with update rate `fs_update`, the baseband output lives in zone 1. Images (sinc-weighted replicas) appear in higher zones.
- Only harmonics that alias *into* the selected zone contribute to the in-zone THD and noise calculations.
- The dBFS convention matches `calculate_adc_dynamic_metrics` in `metrics.py`.

**Examples**

```python
import numpy as np
from pyDataconverter.utils.dac_metrics import calculate_dac_dynamic_metrics

fs = 1e6
t = np.arange(1024) / fs
voltages = 0.5 * np.sin(2 * np.pi * 10e3 * t)
result = calculate_dac_dynamic_metrics(voltages, fs)
print(result['NyquistZone'])  # 1
print(result['ENOB'] > 0)    # True
```

**See Also**

- `calculate_adc_dynamic_metrics` — equivalent dynamic metrics for ADCs.
- `compute_fft` — the FFT engine used internally.

---

## Visualization

### ADC Plots

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

### DAC Plots

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
                gain_error=0.002, offset=0.001)
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

### FFT Plots

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

### Flash ADC Visualization

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

---

## SARADC

---

### `SARADC`

*`pyDataconverter.architectures.SARADC`*

Successive Approximation Register ADC with C-DAC component and configurable non-idealities.

Models the actual bit-cycling algorithm using a single comparator and a binary-weighted C-DAC.
The SAR performs N comparison cycles per conversion (one per bit, MSB first): it tentatively
sets each bit, queries the C-DAC for the corresponding trial voltage, resets the comparator,
and retains the bit if the comparator output is 1. A custom C-DAC can be injected to study
non-standard capacitor arrays. This class inherits from `ADCBase`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | ADC resolution in bits. |
| v_ref | float | `1.0` | Reference voltage (V). |
| input_type | `InputType` | `InputType.SINGLE` | `SINGLE` for single-ended; `DIFFERENTIAL` for differential input. |
| comparator_type | `Type[ComparatorBase]` | `DifferentialComparator` | Comparator class to instantiate. |
| comparator_params | dict or None | `None` | Keyword arguments forwarded to the comparator constructor (e.g. `noise_rms`, `offset`, `hysteresis`). |
| cdac | `CDACBase` or None | `None` | Pre-constructed C-DAC instance. If provided, `cap_mismatch` is ignored. Its `n_bits` and `v_ref` must match the ADC. If `None`, a `SingleEndedCDAC` or `DifferentialCDAC` is auto-created from `input_type`. |
| cap_mismatch | float | `0.0` | Standard deviation of multiplicative capacitor mismatch (dimensionless, e.g. `0.001` = 0.1 %). Ignored when `cdac` is provided. |
| noise_rms | float | `0.0` | Input-referred RMS sampling noise (V). Represents kT/C or other front-end noise fixed at the sampling instant. Applied once per conversion before the bit loop. |
| offset | float | `0.0` | Input-referred DC offset voltage (V). |
| gain_error | float | `0.0` | Fractional gain error (dimensionless, e.g. `0.01` = +1 %). Scales the sampled input as `v * (1 + gain_error)`. |
| t_jitter | float | `0.0` | RMS aperture jitter (s). Only active when `dvdt != 0` is passed to `convert()`. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| comparator | `ComparatorBase` | Single comparator instance used for all N evaluations per conversion. |
| cdac | `CDACBase` | Capacitive DAC component. Auto-created from `input_type` if not supplied. |
| noise_rms | float | Input-referred RMS sampling noise / kT/C (V). |
| offset | float | Input-referred DC offset (V). |
| gain_error | float | Fractional gain error (dimensionless). |
| t_jitter | float | RMS aperture jitter (s). |

**Methods**

#### `convert(vin, dvdt=0.0)`

Convert one analog sample to a digital code.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| vin | float or Tuple[float, float] | — | Voltage for single-ended input, or `(v_pos, v_neg)` tuple for differential input. |
| dvdt | float | `0.0` | Signal slope at the sampling instant (V/s). Only used when `t_jitter > 0`. |

**Returns**

| Type | Description |
|------|-------------|
| int | Output code in [0, 2^n_bits − 1]. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| TypeError | Single-ended input is not a number, or differential input is not a 2-tuple. |

#### `convert_with_trace(vin, dvdt=0.0)`

Convert and return the cycle-by-cycle SAR trace for visualisation or debugging.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| vin | float or Tuple[float, float] | — | Voltage (single-ended) or `(v_pos, v_neg)` tuple (differential). |
| dvdt | float | `0.0` | Signal slope for aperture jitter modelling (V/s). |

**Returns**

| Type | Description |
|------|-------------|
| dict | Dictionary with keys: `'code'` (int) — final output code; `'sampled_voltage'` (float) — held input after non-idealities; `'dac_voltages'` (list[float]) — N effective trial thresholds, one per bit cycle; `'bit_decisions'` (list[int]) — N comparator outputs (0 or 1); `'register_states'` (list[int]) — N + 1 register values, initial 0 followed by the register after each bit decision. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| TypeError | Single-ended input is not a number, or differential input is not a 2-tuple. |

#### `reset()`

Reset comparator state (hysteresis history, bandwidth filter).

#### `dac_voltages` *(property)*

Effective DAC threshold for every code (`v_refp − v_refn`). Analogous to `FlashADC.reference_voltages`. Delegates to `cdac.voltages`.

**Returns**

| Type | Description |
|------|-------------|
| np.ndarray | Shape `(2^n_bits,)`. Entry k is the effective DAC threshold for code k. |

**Notes**

- Non-idealities are applied to the sampled voltage in this order: gain error (multiplicative) → offset (additive) → sampling noise / kT/C (additive) → aperture jitter (additive).
- The SAR binary search naturally implements FLOOR quantisation: `code = floor(vin * 2^N / v_ref)` for single-ended, `code = floor((v_diff + v_ref/2) * 2^N / v_ref)` for differential.
- In single-ended mode, `v_sampled` is a voltage in [0, v_ref]; in differential mode it is the differential voltage `v_pos − v_neg` in [−v_ref/2, +v_ref/2].
- When `cdac` is injected, `cap_mismatch` is ignored and the caller is responsible for ensuring the C-DAC's `n_bits` and `v_ref` match the ADC.
- The comparator is reset before every individual bit comparison to clear hysteresis and latch state.

**Examples**

```python
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.dataconverter import InputType

# Ideal 4-bit single-ended SAR ADC
adc = SARADC(n_bits=4, v_ref=1.0)
code = adc.convert(0.37)
print(code)  # 5

# Cycle-by-cycle trace
trace = adc.convert_with_trace(0.37)
print(trace['bit_decisions'])   # e.g. [0, 1, 0, 1]
print(trace['code'])            # 5
```

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | `noise_rms < 0`, `t_jitter < 0`, or `cap_mismatch < 0`. |
| TypeError | `cdac` is provided but is not a `CDACBase` instance. |
| ValueError | Injected `cdac.n_bits` or `cdac.v_ref` does not match the ADC. |

**See Also**

- `ADCBase` — abstract base class providing the `convert()` / `_convert_input()` contract
- `CDACBase` / `SingleEndedCDAC` / `DifferentialCDAC` — capacitive DAC components
- `ComparatorBase` / `DifferentialComparator` — comparator used in each bit cycle
- `InputType` — enum selecting single-ended vs. differential input mode
- `visualize_sar_adc` — static and interactive visualisation of SAR operation
- `animate_sar_conversion` — bit-by-bit animation of a single conversion

---

## CurrentSteeringDAC

---

### `CurrentSteeringDAC`

*`pyDataconverter.architectures.CurrentSteeringDAC`*

Current-steering DAC with configurable binary, thermometer, or segmented topology.

`CurrentSteeringDAC` inherits from `DACBase` and models a current-steering architecture
where all current sources are always conducting and are steered between positive and negative
output rails rather than switched on and off. Segmentation is controlled by `n_therm_bits`:
setting it to `0` gives a fully binary-weighted DAC, setting it equal to `n_bits` gives a
full thermometer DAC, and any value in between gives a segmented DAC with thermometer MSBs
and binary LSBs. A custom decoder or current source array can be injected at construction
time for advanced use cases.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_bits` | `int` | — | DAC resolution in bits (1–32). |
| `v_ref` | `float` | `1.0` | Reference voltage (V, > 0). Stored for compatibility with `DACBase`; output voltage is determined by `i_unit` and `r_load`. |
| `n_therm_bits` | `int` | `0` | Number of MSBs decoded as thermometer (0 = fully binary, `n_bits` = fully thermometer). |
| `i_unit` | `float` | `100e-6` | Nominal unit current source value (A, > 0). |
| `r_load` | `float` | `1000.0` | Load resistance (Ω, > 0) for I→V conversion. |
| `current_mismatch` | `float` | `0.0` | Standard deviation of multiplicative mismatch per source (≥ 0). Each source draws independently at construction. |
| `output_type` | `OutputType` | `OutputType.DIFFERENTIAL` | `OutputType.SINGLE` or `OutputType.DIFFERENTIAL`. |
| `decoder` | `DecoderBase` or `None` | `None` | Optional pre-built `DecoderBase` instance. If `None`, a `SegmentedDecoder` is created from `n_bits` and `n_therm_bits`. |
| `current_array` | `CurrentSourceArray` or `None` | `None` | Optional pre-built `CurrentSourceArray`. If `None`, one is created automatically. |
| `source_class` | `Type[UnitCurrentSourceBase]` | `IdealCurrentSource` | `UnitCurrentSourceBase` subclass to use when building the current array automatically. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `n_therm_bits` | `int` | Number of MSBs decoded as thermometer. |
| `n_binary_bits` | `int` | Number of LSBs remaining as binary (`n_bits − n_therm_bits`). |
| `i_unit` | `float` | Nominal unit current (A). |
| `r_load` | `float` | Load resistance (Ω). |
| `current_mismatch` | `float` | Standard deviation of multiplicative mismatch used at construction. |
| `i_total` | `float` | Total current from all sources (A). Constant regardless of code. |
| `dac_currents` | `np.ndarray` | Shape `(2^n_bits,)`. Ideal positive-rail output current for every code. |
| `decoder` | `DecoderBase` | Decoder instance used for code splitting. |
| `current_array` | `CurrentSourceArray` | Element array instance. |

**Methods**

#### `convert(digital_input)`

Convert a digital code to an analog output voltage (inherited from `DACBase`).

| Name | Type | Description |
|------|------|-------------|
| `digital_input` | `int` | Input code in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| `float` | Output voltage `i_selected * r_load` in single-ended mode. |
| `Tuple[float, float]` | `(v_pos, v_neg)` = `(i_pos * r_load, i_neg * r_load)` in differential mode. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `n_bits` is not in [1, 32], or any numeric parameter is out of range. |
| `ValueError` | `n_therm_bits` is not in [0, `n_bits`]. |
| `ValueError` | `decoder.n_bits` or `current_array.n_bits` does not match `n_bits`. |
| `TypeError` | `decoder` is not a `DecoderBase` instance. |
| `TypeError` | `current_array` is not a `CurrentSourceArray` instance. |

**Notes**

- All sources always conduct; they are steered between rails rather than switched off. This keeps `i_total` constant regardless of code.
- Single-ended output: `v_out = i_selected * r_load`.
- Differential output: `v_pos = i_pos * r_load`, `v_neg = i_neg * r_load`, where `i_neg = i_total − i_selected`. The differential voltage is `v_diff = (2 * i_selected − i_total) * r_load`.
- The thermometer segment nominal current is scaled to `2^n_binary_bits * i_unit` to guarantee monotonicity at every segment boundary rollover.
- Mismatch is static (drawn once at construction) and independent per source.

**Examples**

```python
from pyDataconverter.architectures.CurrentSteeringDAC import CurrentSteeringDAC
from pyDataconverter.dataconverter import OutputType

# 6-bit segmented DAC: 4 thermometer MSBs, 2 binary LSBs
dac = CurrentSteeringDAC(n_bits=6, n_therm_bits=4, i_unit=100e-6, r_load=1000.0)
v_pos, v_neg = dac.convert(32)  # midscale code
print(v_pos - v_neg)  # ≈ 0.0 V differential

# Single-ended binary DAC
dac = CurrentSteeringDAC(n_bits=8, n_therm_bits=0, output_type=OutputType.SINGLE)
v_out = dac.convert(128)
print(f"{v_out:.4f}")  # positive rail voltage
```

**See Also**

- `DACBase` — abstract base class providing the `convert()` / `_convert_input()` contract
- `DecoderBase` / `SegmentedDecoder` — decoder used to split codes into thermometer and binary signals
- `CurrentSourceArray` — element array holding thermometer and binary current sources
- `UnitCurrentSourceBase` / `IdealCurrentSource` — unit current source models
- `OutputType` — enum selecting single-ended vs. differential output

---

## CDAC

---

### `CDACBase`

*`pyDataconverter.components.cdac`*

Abstract base class defining the interface for all capacitive DAC models.

A C-DAC owns a set of capacitor weights and converts a digital code to a pair of reference
voltages `(v_refp, v_refn)` that the comparator evaluates against the held input signal. The
effective decision threshold for the comparator is `v_refp − v_refn`. This class cannot be
instantiated directly; subclasses must implement `n_bits`, `v_ref`, `cap_weights`, `cap_total`,
and `get_voltage`.

**Methods and Properties**

| Name | Description |
|------|-------------|
| `n_bits` *(abstract property)* | ADC resolution; number of bit capacitors. |
| `v_ref` *(abstract property)* | Reference voltage (V). |
| `cap_weights` *(abstract property)* | Actual capacitor weights (with mismatch applied), MSB first. Length equals `n_bits`. |
| `cap_total` *(abstract property)* | Total effective capacitance including the termination cap. For `DifferentialCDAC` this refers to the positive array. |
| `get_voltage(code)` *(abstract)* | Return `(v_refp, v_refn)` for the given integer code. The pair maps directly onto the `DifferentialComparator` 4-input signature. |
| `voltages` *(property)* | `np.ndarray` of shape `(2^n_bits,)`. Effective output voltage `v_refp − v_refn` for every code. |

**Notes**

- The termination capacitor of 1.0 unit is included in `cap_total`, giving `cap_total = (2^N − 1) + 1 = 2^N` for ideal binary weights. This ensures `get_voltage(code)` returns exactly `code / 2^N * v_ref` (FLOOR quantisation) with no mismatch.
- `voltages` iterates over all codes and is primarily used by `SARADC.dac_voltages`.

**See Also**

- `SingleEndedCDAC` — concrete C-DAC for single-ended SAR ADCs
- `DifferentialCDAC` — concrete C-DAC with complementary arrays for differential SAR ADCs
- `SARADC` — the architecture that uses a C-DAC instance internally

---

### `SingleEndedCDAC`

*`pyDataconverter.components.cdac`*

Binary-weighted capacitive DAC for single-ended SAR ADCs.

The positive reference output (`v_refp`) is the weighted sum of selected capacitors normalised
to `v_ref`. The negative reference output (`v_refn`) is always 0 V (ground). Capacitor mismatch
is drawn once at construction from a multiplicative Gaussian distribution and held fixed,
producing static DNL/INL errors.

Ideal output (no mismatch): `v_dac = code / 2^n_bits * v_ref`.
Output range: [0, `v_ref − LSB`] where `LSB = v_ref / 2^n_bits`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | Resolution in bits. Must be in [1, 32]. |
| v_ref | float | `1.0` | Reference voltage (V). Must be positive. |
| cap_weights | np.ndarray or None | `None` | Nominal capacitor weights, MSB first. Length must equal `n_bits`; all values must be positive. If `None`, binary weights `[2^(N-1), …, 2, 1]` are used. |
| cap_mismatch | float | `0.0` | Standard deviation of multiplicative capacitor mismatch (dimensionless, e.g. `0.001` = 0.1 %). Each capacitor instance draws its own mismatch independently at construction. |
| cap_class | `Type[UnitCapacitorBase]` | `IdealCapacitor` | `UnitCapacitorBase` subclass to instantiate for each capacitor in the array. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| cap_weights | np.ndarray | Actual capacitor values (with mismatch applied), MSB first, in units of C_unit. |
| cap_total | float | `sum(cap_weights)` + termination cap (1.0). |
| cap_mismatch | float | Standard deviation of multiplicative mismatch used at construction. |
| cap_instances | `List[UnitCapacitorBase]` | Individual capacitor objects, MSB first. Length equals `n_bits`. Allows per-element inspection or replacement. |

**Methods**

#### `get_voltage(code)`

Return `(v_dac, 0.0)` for the given code.

Formula: `v_dac = dot(bits, cap_weights) / cap_total * v_ref`

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| code | int | — | Integer in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[float, float] | `(v_dac, 0.0)` — positive DAC voltage and ground reference. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | `n_bits` is not an integer in [1, 32]. |
| ValueError | `v_ref` is not a positive number. |
| ValueError | `cap_mismatch < 0`. |
| ValueError | `cap_weights` has wrong length or contains non-positive values. |

**Notes**

- The termination capacitor (1.0 unit) is added to `cap_total` so ideal binary weights give `v_dac = code / 2^N * v_ref`.
- Mismatch is applied multiplicatively: `actual_weight = nominal_weight * (1 + N(0, cap_mismatch))`.
- A copy of `cap_weights` is returned by the property to prevent external mutation.

**Examples**

```python
from pyDataconverter.components.cdac import SingleEndedCDAC

cdac = SingleEndedCDAC(n_bits=4, v_ref=1.0)
v_refp, v_refn = cdac.get_voltage(8)  # code 8 → midscale
print(v_refp)   # 0.5
print(v_refn)   # 0.0
```

**See Also**

- `CDACBase` — abstract base class defining the C-DAC interface
- `DifferentialCDAC` — complementary differential C-DAC
- `SARADC` — uses `SingleEndedCDAC` when `input_type=InputType.SINGLE`
- `UnitCapacitorBase` / `IdealCapacitor` — unit capacitor models used inside the array

---

### `DifferentialCDAC`

*`pyDataconverter.components.cdac`*

Binary-weighted capacitive DAC with complementary arrays for differential SAR ADCs.

Both a positive and a negative capacitor array are modelled. They switch complementarily:
a bit set to 1 connects the positive capacitor to `v_ref/2` and the negative capacitor to
0 V; a bit set to 0 does the opposite. The effective differential threshold is
`v_dacp − v_dacn`. Independent mismatch is drawn for each array at construction, which is
required to model realistic differential nonlinearity (correlated mismatch would cancel).

Ideal outputs (no mismatch):
- `v_dacp = code / 2^n_bits * v_ref / 2`
- `v_dacn = (1 − code / 2^n_bits) * v_ref / 2`
- `v_dac_diff = (2 * code / 2^n_bits − 1) * v_ref / 2`

Output range of `v_dac_diff`: [−v_ref/2, ≈ +v_ref/2].

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| n_bits | int | — | Resolution in bits. Must be in [1, 32]. |
| v_ref | float | `1.0` | Reference voltage (V). Must be positive. |
| cap_weights | np.ndarray or None | `None` | Nominal capacitor weights applied to both arrays before mismatch, MSB first, all positive. If `None`, binary weights `[2^(N-1), …, 2, 1]` are used. |
| cap_mismatch | float | `0.0` | Standard deviation of multiplicative mismatch (dimensionless). Each capacitor in both arrays draws its own mismatch independently at construction. |
| cap_class | `Type[UnitCapacitorBase]` | `IdealCapacitor` | `UnitCapacitorBase` subclass to instantiate for each capacitor in both arrays. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| cap_weights | np.ndarray | Positive-side capacitor weights (with mismatch), MSB first. |
| cap_weights_neg | np.ndarray | Negative-side capacitor weights (with mismatch), MSB first. |
| cap_total | float | Positive-side total capacitance including the termination cap. |
| cap_total_neg | float | Negative-side total capacitance including the termination cap. |
| cap_mismatch | float | Standard deviation of multiplicative mismatch used at construction. |
| cap_instances | `List[UnitCapacitorBase]` | Positive-side capacitor objects, MSB first. Length equals `n_bits`. |
| cap_instances_neg | `List[UnitCapacitorBase]` | Negative-side capacitor objects, MSB first. Length equals `n_bits`. |

**Methods**

#### `get_voltage(code)`

Return `(v_dacp, v_dacn)` for the given code.

Formulas:
- `v_dacp = dot(bits, cap_weights_pos) / cap_total_pos * v_ref / 2`
- `v_dacn = (cap_total_neg − dot(bits, cap_weights_neg)) / cap_total_neg * v_ref / 2`

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| code | int | — | Integer in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[float, float] | `(v_dacp, v_dacn)` — positive and negative rail DAC voltages. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| ValueError | `n_bits` is not an integer in [1, 32]. |
| ValueError | `v_ref` is not a positive number. |
| ValueError | `cap_mismatch < 0`. |
| ValueError | `cap_weights` has wrong length or contains non-positive values. |

**Notes**

- Independent mismatch draws for the positive and negative arrays are essential for modelling realistic differential DNL/INL. Correlated (identical) mismatch cancels in the differential output.
- `cap_total` and `cap_total_neg` include a termination capacitor of 1.0 unit each.
- Copies are returned by the `cap_weights` and `cap_weights_neg` properties to prevent external mutation.

**Examples**

```python
from pyDataconverter.components.cdac import DifferentialCDAC

cdac = DifferentialCDAC(n_bits=4, v_ref=1.0)
v_dacp, v_dacn = cdac.get_voltage(8)  # code 8 → midscale
print(v_dacp - v_dacn)  # ≈ 0.0 (midscale differential)
```

**See Also**

- `CDACBase` — abstract base class defining the C-DAC interface
- `SingleEndedCDAC` — single-ended variant
- `SARADC` — uses `DifferentialCDAC` when `input_type=InputType.DIFFERENTIAL`
- `UnitCapacitorBase` / `IdealCapacitor` — unit capacitor models used inside both arrays

---

## Decoder

---

### `DecoderBase`

*`pyDataconverter.components.decoder`*

Abstract base class for all DAC decoders.

A decoder translates a compact N-bit digital code into control signals that drive a physical
DAC element array. All subclasses must implement `decode()` and expose `n_bits` and
`n_therm_bits`. The `n_binary_bits` property is derived and need not be overridden. This class
cannot be instantiated directly.

The uniform return convention from `decode()` is:

```
(therm_index: int, binary_bits: np.ndarray)
```

- `therm_index` — integer in [0, 2^n_therm_bits − 1]; number of thermometer unit elements to switch on.
- `binary_bits` — float array of length `n_binary_bits`, MSB first; empty when `n_binary_bits == 0`.

**Properties**

| Name | Type | Description |
|------|------|-------------|
| `n_bits` *(abstract)* | `int` | Total DAC resolution in bits. |
| `n_therm_bits` *(abstract)* | `int` | Number of MSBs decoded to thermometer (unary) control signals. |
| `n_binary_bits` | `int` | Number of LSBs that remain as binary control signals (`n_bits − n_therm_bits`). |

**Methods**

#### `decode(code)`

Decode a digital code into thermometer and binary control signals.

| Name | Type | Description |
|------|------|-------------|
| `code` | `int` | Integer in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| `Tuple[int, np.ndarray]` | `(therm_index, binary_bits)` — thermometer index integer and binary bit array. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `code` is outside [0, 2^n_bits − 1]. |
| `TypeError` | `code` is not an integer. |

**Notes**

- The uniform return type means a DAC never needs to know which decoder variant it holds — it always unpacks the same `(therm_index, binary_bits)` pair.

**See Also**

- `BinaryDecoder` — passthrough; all bits drive a binary-weighted array
- `ThermometerDecoder` — full unary decode; code drives 2^N − 1 unit elements
- `SegmentedDecoder` — MSBs thermometer-decoded, LSBs binary (general case)

---

### `BinaryDecoder`

*`pyDataconverter.components.decoder`*

Binary (passthrough) decoder; all bits drive a binary-weighted element array directly.

The input code is returned as a bit array with no thermometer expansion. `n_therm_bits` is
fixed at 0 and `n_binary_bits` equals `n_bits`. This is the degenerate case of
`SegmentedDecoder` with `n_therm_bits=0`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_bits` | `int` | — | DAC resolution in bits (1–32). |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `n_bits` is not an integer in [1, 32]. |

**Methods**

#### `decode(code)`

Return `(0, binary_bits)` for the given code.

| Name | Type | Description |
|------|------|-------------|
| `code` | `int` | Integer in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| `Tuple[int, np.ndarray]` | `(0, bit_array)` — therm_index is always 0; `bit_array` has length `n_bits`, MSB first. |

**Examples**

```python
from pyDataconverter.components.decoder import BinaryDecoder

dec = BinaryDecoder(n_bits=4)
therm_idx, bits = dec.decode(11)  # 0b1011
print(therm_idx)   # 0
print(bits)        # [1. 0. 1. 1.]
```

**See Also**

- `DecoderBase` — abstract base class
- `SegmentedDecoder` — general case; `BinaryDecoder` is equivalent to `SegmentedDecoder(n_bits, 0)`

---

### `ThermometerDecoder`

*`pyDataconverter.components.decoder`*

Full thermometer (unary) decoder; the N-bit input code selects how many of the 2^N − 1 unit elements are switched on.

No binary sub-array exists: `n_therm_bits` equals `n_bits` and `n_binary_bits` is 0. This is
the degenerate case of `SegmentedDecoder` with `n_therm_bits=n_bits`. The maximum `n_bits` is
16, limiting the element count to 65535.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_bits` | `int` | — | DAC resolution in bits (1–16). Limited to 16 to keep element counts tractable (2^16 − 1 = 65535 unit elements). |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `n_elements` | `int` | Number of unit thermometer elements (2^n_bits − 1). |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `n_bits` is not an integer in [1, 16]. |

**Methods**

#### `decode(code)`

Return `(code, [])` for the given code.

| Name | Type | Description |
|------|------|-------------|
| `code` | `int` | Integer in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| `Tuple[int, np.ndarray]` | `(code, empty_array)` — therm_index equals the input code; binary_bits is an empty array. |

**Examples**

```python
from pyDataconverter.components.decoder import ThermometerDecoder

dec = ThermometerDecoder(n_bits=3)
print(dec.n_elements)         # 7
therm_idx, bits = dec.decode(5)
print(therm_idx)  # 5
print(bits)       # []
```

**See Also**

- `DecoderBase` — abstract base class
- `SegmentedDecoder` — general case; `ThermometerDecoder` is equivalent to `SegmentedDecoder(n_bits, n_bits)`

---

### `SegmentedDecoder`

*`pyDataconverter.components.decoder`*

Segmented decoder — thermometer MSBs, binary LSBs; the most general decoder form.

Splits the N-bit code into an upper `n_therm_bits` portion decoded as a thermometer index and
a lower `n_binary_bits = n_bits − n_therm_bits` portion kept as a binary bit array. When
`n_therm_bits=0` this reduces to `BinaryDecoder`; when `n_therm_bits=n_bits` it reduces to
`ThermometerDecoder`. Typical segmented DAC usage applies 4–6 thermometer bits to the MSBs.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_bits` | `int` | — | Total DAC resolution (1–32). |
| `n_therm_bits` | `int` | — | Number of MSBs decoded as thermometer (0–`n_bits`). The thermometer segment creates 2^n_therm_bits − 1 unit elements; limited to ≤ 16 for tractability. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `n_therm_elements` | `int` | Number of thermometer unit elements (2^n_therm_bits − 1). |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `n_bits` is not an integer in [1, 32]. |
| `ValueError` | `n_therm_bits` is not in [0, `n_bits`] or exceeds 16. |

**Methods**

#### `decode(code)`

Split code into thermometer index and binary bit array.

| Name | Type | Description |
|------|------|-------------|
| `code` | `int` | Integer in [0, 2^n_bits − 1]. |

**Returns**

| Type | Description |
|------|-------------|
| `Tuple[int, np.ndarray]` | `(therm_index, binary_bits)` — upper `n_therm_bits` as integer; lower `n_binary_bits` as float bit array, MSB first. Empty array when `n_binary_bits == 0`. |

**Examples**

```python
from pyDataconverter.components.decoder import SegmentedDecoder

# 6-bit code, 4 thermometer MSBs, 2 binary LSBs
dec = SegmentedDecoder(n_bits=6, n_therm_bits=4)
therm_idx, bits = dec.decode(45)  # 0b101101
print(therm_idx)  # 11  (upper 4 bits = 0b1011)
print(bits)       # [0. 1.]  (lower 2 bits = 0b01)
```

**See Also**

- `DecoderBase` — abstract base class
- `BinaryDecoder` — degenerate case with `n_therm_bits=0`
- `ThermometerDecoder` — degenerate case with `n_therm_bits=n_bits`
- `CurrentSteeringDAC` — uses `SegmentedDecoder` internally

---

## Capacitor

---

### `UnitCapacitorBase`

*`pyDataconverter.components.capacitor`*

Abstract base class for unit capacitor models used in capacitive DAC arrays.

Defines the interface used by C-DAC arrays. Each instance represents a single physical
capacitor. Subclasses model different physical effects (leakage, voltage dependence, etc.)
while keeping the array and DAC code unchanged. This class cannot be instantiated directly.

**Properties**

| Name | Type | Description |
|------|------|-------------|
| `c_nominal` *(abstract)* | `float` | Designed (nominal) capacitance in farads or normalised units. |
| `capacitance` *(abstract)* | `float` | Actual capacitance used in simulation. For ideal capacitors this equals `c_nominal * (1 + mismatch_draw)`. |

**See Also**

- `IdealCapacitor` — concrete implementation with static Gaussian mismatch
- `SingleEndedCDAC` / `DifferentialCDAC` — hold arrays of `UnitCapacitorBase` instances

---

### `IdealCapacitor`

*`pyDataconverter.components.capacitor`*

Ideal capacitor with static Gaussian mismatch drawn once at construction.

The actual capacitance is `capacitance = c_nominal * (1 + ε)` where `ε ~ N(0, mismatch)` is
drawn once and held fixed for the lifetime of the object, modelling process-induced static
spread. The mismatch draw is zero when `mismatch=0.0` (default), giving an exactly ideal
capacitor.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `c_nominal` | `float` | `1.0` | Designed capacitance (> 0). May be in any consistent unit (F, fF, or normalised to a unit cap). |
| `mismatch` | `float` | `0.0` | Standard deviation of multiplicative mismatch (≥ 0). A single draw `ε ~ N(0, mismatch)` is applied at construction: `actual = c_nominal * (1 + ε)`. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `c_nominal` | `float` | Designed (nominal) capacitance. |
| `mismatch` | `float` | Standard deviation of mismatch used at construction. |
| `capacitance` | `float` | Actual capacitance after mismatch draw. Fixed for the lifetime of the instance. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `c_nominal` is not a positive number. |
| `ValueError` | `mismatch` is negative. |

**Notes**

- Mismatch is static: the draw happens once at `__init__` and `capacitance` never changes.
- To simulate a fresh mismatch draw, create a new `IdealCapacitor` instance.
- Passing `mismatch=0.0` (default) gives `capacitance == c_nominal` exactly, with no random draw.

**Examples**

```python
from pyDataconverter.components.capacitor import IdealCapacitor

# Ideal capacitor (no mismatch)
cap = IdealCapacitor(c_nominal=1.0)
print(cap.capacitance)  # 1.0

# Capacitor with 1 % mismatch
cap = IdealCapacitor(c_nominal=1.0, mismatch=0.01)
print(abs(cap.capacitance - 1.0) < 0.1)  # True (within several sigma)
```

**See Also**

- `UnitCapacitorBase` — abstract base class defining the interface
- `SingleEndedCDAC` / `DifferentialCDAC` — pass `cap_class=IdealCapacitor` by default

---

## Current Source

---

### `UnitCurrentSourceBase`

*`pyDataconverter.components.current_source`*

Abstract base class for unit current source models used in current-steering DAC arrays.

Defines the interface used by `CurrentSourceArray`. Each instance represents one physical
current source cell. Subclasses model different physical implementations (ideal, cascode,
regulated cascode) while keeping the array and DAC code unchanged. This class cannot be
instantiated directly.

**Properties**

| Name | Type | Description |
|------|------|-------------|
| `i_nominal` *(abstract)* | `float` | Designed (nominal) current in amperes. |
| `current` *(abstract)* | `float` | Actual current used in simulation. For ideal sources this equals `i_nominal * (1 + mismatch_draw)`. |

**See Also**

- `IdealCurrentSource` — concrete implementation with static Gaussian mismatch
- `CurrentSourceArray` — holds arrays of `UnitCurrentSourceBase` instances

---

### `IdealCurrentSource`

*`pyDataconverter.components.current_source`*

Ideal current source with static Gaussian mismatch drawn once at construction.

The actual current is `current = i_nominal * (1 + ε)` where `ε ~ N(0, mismatch)` is drawn
once and held fixed, modelling process-induced static spread (threshold voltage variation,
etc.). The mismatch draw is zero when `mismatch=0.0` (default).

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `i_nominal` | `float` | `100e-6` | Designed current in amperes (> 0). |
| `mismatch` | `float` | `0.0` | Standard deviation of multiplicative mismatch (≥ 0). A single draw `ε ~ N(0, mismatch)` is applied at construction: `actual = i_nominal * (1 + ε)`. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `i_nominal` | `float` | Designed (nominal) current (A). |
| `mismatch` | `float` | Standard deviation of mismatch used at construction. |
| `current` | `float` | Actual current after mismatch draw. Fixed for the lifetime of the instance. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `i_nominal` is not a positive number. |
| `ValueError` | `mismatch` is negative. |

**Notes**

- Mismatch is static: the draw happens once at `__init__` and `current` never changes.
- Passing `mismatch=0.0` (default) gives `current == i_nominal` exactly.

**Examples**

```python
from pyDataconverter.components.current_source import IdealCurrentSource

src = IdealCurrentSource(i_nominal=100e-6)
print(src.current)  # 0.0001 A

src_mm = IdealCurrentSource(i_nominal=100e-6, mismatch=0.01)
print(abs(src_mm.current - 100e-6) < 10e-6)  # True (within several sigma)
```

**See Also**

- `UnitCurrentSourceBase` — abstract base class
- `CurrentSourceArray` — creates and holds arrays of `IdealCurrentSource` by default

---

### `CurrentSourceArray`

*`pyDataconverter.components.current_source`*

Array of current sources for a current-steering DAC, comprising a thermometer segment and a binary segment.

Holds two segments that mirror the decoder output. The thermometer segment contains
`2^n_therm_bits − 1` unit sources each with nominal current `2^n_binary_bits * i_unit`
(scaled to guarantee monotonicity at segment boundaries). The binary segment contains
`n_binary_bits` sources with binary-weighted nominals: source `j` has
`i_nominal = 2^j * i_unit` where `j=0` is the LSB.

All sources always conduct and are steered between positive and negative output rails,
keeping `i_total` constant regardless of code. `get_current()` returns `(i_selected, i_total)`
so the DAC can compute differential or single-ended output voltages.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_therm_bits` | `int` | — | Number of thermometer MSBs. Creates `2^n_therm_bits − 1` unit sources. May be 0. |
| `n_binary_bits` | `int` | — | Number of binary LSBs. Creates `n_binary_bits` binary-weighted sources. May be 0. |
| `i_unit` | `float` | `100e-6` | Nominal unit current (A, > 0). |
| `current_mismatch` | `float` | `0.0` | Standard deviation of multiplicative mismatch for every source (≥ 0). Each source draws independently. |
| `source_class` | `Type[UnitCurrentSourceBase]` | `IdealCurrentSource` | `UnitCurrentSourceBase` subclass to instantiate. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `therm_sources` | `List[UnitCurrentSourceBase]` | Thermometer unit sources in switching order (index 0 switches on first for code = 1). |
| `binary_sources` | `List[UnitCurrentSourceBase]` | Binary-weighted sources; index 0 = LSB weight (1×i_unit), index `n_binary_bits−1` = MSB weight. |
| `i_unit` | `float` | Nominal unit current (A). |
| `current_mismatch` | `float` | Standard deviation of multiplicative mismatch used at construction. |
| `n_therm_bits` | `int` | Number of thermometer MSB bits. |
| `n_binary_bits` | `int` | Number of binary LSB bits. |
| `n_bits` | `int` | Total DAC resolution (`n_therm_bits + n_binary_bits`). |
| `i_total` | `float` | Total current conducted by all sources (A). Constant regardless of code. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `n_therm_bits` or `n_binary_bits` is negative, or both are 0. |
| `ValueError` | `i_unit` is not positive or `current_mismatch` is negative. |
| `TypeError` | `source_class` is not a `UnitCurrentSourceBase` subclass. |

**Methods**

#### `get_current(therm_index, binary_bits)`

Compute DAC output current for given decoder control signals.

Sums the currents of the selected thermometer elements and the asserted binary-weighted
elements. Thermometer elements 0 through `therm_index − 1` are steered to the positive rail.
Binary elements are steered to the positive rail when the corresponding bit is 1.

| Name | Type | Description |
|------|------|-------------|
| `therm_index` | `int` | Number of thermometer unit elements steered to the positive output (integer in [0, n_therm_elements]). |
| `binary_bits` | `np.ndarray` | Bit array for the binary segment, MSB first (length `n_binary_bits`). Bit = 1 steers source to positive output. |

**Returns**

| Type | Description |
|------|-------------|
| `Tuple[float, float]` | `(i_selected, i_total)` — current steered to positive rail and total current from all sources. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `therm_index` is outside [0, n_therm_elements] or `binary_bits` has wrong length. |

**Notes**

- `i_total` is constant (precomputed at construction) because all sources always conduct.
- The DAC computes single-ended or differential output: `v_se = i_selected * r_load`; `v_diff = (2 * i_selected − i_total) * r_load`.

**Examples**

```python
from pyDataconverter.components.current_source import CurrentSourceArray

arr = CurrentSourceArray(n_therm_bits=3, n_binary_bits=2, i_unit=100e-6)
import numpy as np
i_sel, i_total = arr.get_current(therm_index=4, binary_bits=np.array([1., 0.]))
print(f"i_selected={i_sel*1e6:.1f} uA, i_total={i_total*1e6:.1f} uA")
```

**See Also**

- `UnitCurrentSourceBase` / `IdealCurrentSource` — unit current source models
- `DecoderBase` — produces `(therm_index, binary_bits)` consumed by `get_current()`
- `CurrentSteeringDAC` — owns a `CurrentSourceArray` and a `DecoderBase`

---

## SAR ADC Visualization

---

### `visualize_sar_adc`

*`pyDataconverter.utils.visualizations.visualize_SARADC`*

Visualise SAR ADC operation for a given input voltage with a two-panel static or interactive plot.

The left panel shows the binary search funnel: a narrowing voltage interval one bar per bit cycle
(MSB to LSB), with green shading for the kept half, red for the discarded half, an orange tick for
the trial DAC voltage, and a blue dashed line for the held input. The right panel shows the C-DAC
capacitor state: each bit's voltage contribution as a bar coloured by decision (green = 1, grey = 0,
blue = undecided) with an orange dashed line for the running DAC output.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| adc | `SARADC` | — | SARADC instance to visualise. |
| input_voltage | float, Tuple, or None | `None` | Input voltage. Scalar for single-ended; `(v_pos, v_neg)` tuple for differential. Ignored when `interactive=True`. |
| interactive | bool | `False` | When `True`, adds a Matplotlib slider for real-time exploration of the full input range. |
| fig | `matplotlib.figure.Figure` or None | `None` | Existing figure to draw into. If `None`, a new figure is created. |
| axes | Tuple or None | `None` | Existing `(ax_funnel, ax_cdac)` axes pair. If `None`, new axes are created. |

**Returns**

| Type | Description |
|------|-------------|
| Tuple[Figure, Tuple[Axes, Axes]] | `(fig, (ax_funnel, ax_cdac))` — the figure and its two axes. |

**Notes**

- When `interactive=True`, a `matplotlib.widgets.Slider` is added and `plt.show()` is called internally.
- When `interactive=False` and new axes are created, `plt.tight_layout()` and `plt.show()` are called automatically.
- When `fig` and `axes` are both supplied, neither `tight_layout` nor `show` is called, allowing the caller to manage the figure lifecycle.
- For a differential SARADC, a scalar `input_voltage` is interpreted as the differential voltage `v_diff`; internally it is split into `(v_diff/2, -v_diff/2)`.

**Examples**

```python
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.utils.visualizations.visualize_SARADC import visualize_sar_adc

adc = SARADC(n_bits=4, v_ref=1.0)
fig, axes = visualize_sar_adc(adc, input_voltage=0.37)
```

**See Also**

- `animate_sar_conversion` — bit-by-bit animation of a single conversion
- `animate_sar_adc` — sweep animation across a sequence of input voltages
- `SARADC` — the SAR ADC architecture being visualised

---

### `animate_sar_conversion`

*`pyDataconverter.utils.visualizations.visualize_SARADC`*

Animate a single SAR conversion, revealing one bit cycle per frame.

Frame k shows the first k bit decisions; remaining cycles appear as placeholder blocks.
The most recently decided bit is highlighted in orange so the viewer can follow the binary
search step by step. Both panels (funnel and C-DAC state) update each frame.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| adc | `SARADC` | — | SARADC instance. |
| input_voltage | float or Tuple | — | Input voltage. Scalar for single-ended; `(v_pos, v_neg)` tuple for differential. |
| interval | float | `0.7` | Seconds between frames. |

**Returns**

| Type | Description |
|------|-------------|
| `FuncAnimation` | Animation object. Keep a reference to prevent garbage collection. |

**Notes**

- The animation loops with `repeat=True`. Close the window to stop.
- `plt.show()` is called internally; the function blocks until the window is closed.
- The animation runs for exactly `n_bits` frames (one per bit decision).

**Examples**

```python
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.utils.visualizations.visualize_SARADC import animate_sar_conversion

adc = SARADC(n_bits=4, v_ref=1.0)
anim = animate_sar_conversion(adc, input_voltage=0.62, interval=0.8)
```

**See Also**

- `visualize_sar_adc` — static or interactive snapshot
- `animate_sar_adc` — sweep animation across multiple input voltages

---

### `animate_sar_adc`

*`pyDataconverter.utils.visualizations.visualize_SARADC`*

Animate SAR ADC operation across a sequence of input voltages.

Each frame shows the complete binary search trace for one input voltage, making it easy to
observe how the search pattern shifts as the input sweeps. Both panels (funnel and C-DAC
state) are redrawn fully on each frame.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| adc | `SARADC` | — | SARADC instance. |
| input_voltages | array-like | — | 1-D array of scalars for single-ended input, or list of `(v_pos, v_neg)` tuples for differential input. |
| interval | float | `0.12` | Seconds between frames. |

**Returns**

| Type | Description |
|------|-------------|
| `FuncAnimation` | Animation object. Keep a reference to prevent garbage collection. |

**Notes**

- The animation loops with `repeat=True`. Close the window to stop.
- `plt.show()` is called internally; the function blocks until the window is closed.
- The number of frames equals `len(input_voltages)`.

**Examples**

```python
import numpy as np
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.utils.visualizations.visualize_SARADC import animate_sar_adc

adc = SARADC(n_bits=4, v_ref=1.0)
voltages = 0.5 + 0.45 * np.sin(np.linspace(0, 2 * np.pi, 40))
anim = animate_sar_adc(adc, input_voltages=voltages, interval=0.1)
```

**See Also**

- `visualize_sar_adc` — static or interactive snapshot
- `animate_sar_conversion` — bit-by-bit animation of a single conversion
- `SARADC` — the SAR ADC architecture being animated
