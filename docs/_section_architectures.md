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

**Methods**

#### `convert(digital_input)`

Convert a digital code to an analog output voltage.

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
| TypeError | `digital_input` is not an integer. |
| ValueError | `digital_input` is outside the valid code range. |
| ValueError | `noise_rms` is negative (raised at construction). |

**Notes**

- Non-idealities are applied in the order: gain error → offset → noise.
- In differential mode the output is centered around `v_ref / 2`: `v_pos = v_diff/2 + v_ref/2`, `v_neg = −v_diff/2 + v_ref/2`.
- The ideal voltage is computed as `digital_input * lsb`.

**Examples**

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC

dac = SimpleDAC(n_bits=12, v_ref=1.0)
voltage = dac.convert(2048)
print(f"{voltage:.4f}")  # 0.5001
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
