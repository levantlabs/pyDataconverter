## Components

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
