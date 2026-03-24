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
