# pyDataconverter Quick Start Guide

Get up and running in minutes. This guide covers the three converter architectures, signal generation, performance analysis, and visualization.

---

## Installation

```bash
pip install pyDataconverter
```

Or from source:

```bash
git clone <repo-url>
cd pyDataconverter
pip install -e .
```

---

## Contents

1. [Core Concepts](#core-concepts)
2. [SimpleADC](#simpleadc)
3. [SimpleDAC](#simpledac)
4. [FlashADC](#flashadc)
5. [SARADC](#saradc)
6. [CurrentSteeringDAC](#currentsteeringdac)
7. [Signal Generation](#signal-generation)
8. [Analysis](#analysis)
   - [Static Metrics — ADC](#static-metrics--adc)
   - [Dynamic Metrics — ADC / DAC](#dynamic-metrics--adc--dac)
9. [Visualization](#visualization)
   - [ADC Plots](#adc-plots)
   - [DAC Plots](#dac-plots)
   - [Flash ADC Visualization](#flash-adc-visualization)
   - [SAR ADC Visualization](#sar-adc-visualization)

---

## Core Concepts

| Concept | Description |
|---|---|
| `n_bits` | Resolution. Codes span `[0, 2^n_bits - 1]`. |
| `v_ref` | Full-scale reference voltage (V). Default `1.0`. |
| `lsb` | Step size = `v_ref / (2^n_bits - 1)`. Max code maps exactly to `v_ref`. |
| `InputType` | `SINGLE` (0 → v_ref) or `DIFFERENTIAL` (−v_ref/2 → +v_ref/2). |
| `OutputType` | `SINGLE` (float) or `DIFFERENTIAL` (v_pos, v_neg tuple). |

---

## SimpleADC

A straightforward quantizer with optional first-order non-idealities.

### Ideal conversion

```python
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType, QuantizationMode

# 12-bit, 1 V reference, single-ended input
adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)

print(adc.convert(0.0))    # 0
print(adc.convert(0.5))    # 2048
print(adc.convert(1.0))    # 4095
```

### Quantization modes

```python
# FLOOR (default): uniform bins, error in [0, -1] LSB
adc_floor = SimpleADC(12, quant_mode=QuantizationMode.FLOOR)

# SYMMETRIC: mid-tread, error in [-0.5, +0.5] LSB
adc_sym = SimpleADC(12, quant_mode=QuantizationMode.SYMMETRIC)
```

### Non-idealities

```python
adc = SimpleADC(
    n_bits=12,
    v_ref=1.0,
    input_type=InputType.SINGLE,
    noise_rms=200e-6,   # 200 µV RMS thermal noise
    offset=5e-3,        # +5 mV DC offset
    gain_error=0.002,   # +0.2% gain error
    t_jitter=1e-12,     # 1 ps RMS aperture jitter
)

import math
f_sig = 10e3
dvdt = 0.5 * 2 * math.pi * f_sig  # slope at zero-crossing for 0.5 V sine
code = adc.convert(0.0, dvdt=dvdt)
```

Non-idealities are applied in order: **gain error → offset → thermal noise → aperture jitter**.

### Differential input

```python
adc_diff = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

# Input range: v_pos - v_neg in [-0.5 V, +0.5 V]
code = adc_diff.convert((0.75, 0.25))   # v_diff = +0.5 V → code 4095
code = adc_diff.convert((0.5,  0.5))    # v_diff =  0.0 V → code 2048
code = adc_diff.convert((0.25, 0.75))   # v_diff = -0.5 V → code 0
```

### Convert an array

```python
import numpy as np
from pyDataconverter.utils.signal_gen import generate_sine

signal = generate_sine(frequency=1e3, sampling_rate=1e6, amplitude=0.4, offset=0.5)
codes = np.array([adc.convert(v) for v in signal])
```

---

## SimpleDAC

Ideal digital-to-analog conversion with optional output non-idealities.

### Ideal conversion

```python
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.dataconverter import OutputType

dac = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.SINGLE)

print(dac.convert(0))       # 0.0 V
print(dac.convert(2048))    # ~0.5 V
print(dac.convert(4095))    # 1.0 V
```

### Non-idealities

```python
dac = SimpleDAC(
    n_bits=12,
    v_ref=1.0,
    noise_rms=100e-6,   # 100 µV RMS output noise
    offset=2e-3,        # +2 mV DC offset
    gain_error=0.001,   # +0.1% gain error
)

voltage = dac.convert(2048)
```

Non-idealities are applied in order: **gain error → offset → noise**.

### Differential output

```python
dac_diff = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.DIFFERENTIAL)

v_pos, v_neg = dac_diff.convert(4095)   # full scale:  v_diff = +1.0 V
v_pos, v_neg = dac_diff.convert(2048)   # mid scale:   v_diff ≈  0.0 V
v_pos, v_neg = dac_diff.convert(0)      # zero scale:  v_diff = -1.0 V

v_diff = v_pos - v_neg  # differential output spans [-v_ref, +v_ref]
```

---

## FlashADC

Parallel comparator bank with a thermometer-to-binary encoder. Supports per-comparator offset, noise, hysteresis, resistor ladder mismatch, and reference noise.

### Ideal (no non-idealities)

```python
from pyDataconverter.architectures.FlashADC import FlashADC, EncoderType
from pyDataconverter.dataconverter import InputType

adc = FlashADC(n_bits=4, v_ref=1.0)

code = adc.convert(0.5)   # → 8  (mid-scale for a 4-bit ADC)
```

### With non-idealities

```python
adc = FlashADC(
    n_bits=4,
    v_ref=1.0,
    comparator_params={
        'noise_rms': 1e-3,    # 1 mV RMS comparator noise
        'hysteresis': 2e-3,   # 2 mV hysteresis
    },
    offset_std=1e-3,          # 1 mV std per-comparator offset (drawn at init)
    resistor_mismatch=0.01,   # 1% resistor ladder mismatch
    reference_noise=0.5e-3,   # 0.5 mV RMS reference noise
)
```

### Encoder types

```python
# COUNT_ONES (default): robust to bubble errors
adc_co = FlashADC(n_bits=4, encoder_type=EncoderType.COUNT_ONES)

# XOR: mirrors hardware ROM encoder; bubble errors become sparkle codes
adc_xor = FlashADC(n_bits=4, encoder_type=EncoderType.XOR)
```

### Differential input

```python
adc_diff = FlashADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

# Input range: v_pos - v_neg in [-0.5 V, +0.5 V]
code = adc_diff.convert((0.6, 0.4))   # v_diff = +0.2 V
```

---

## SARADC

A structural SAR ADC model that performs N binary-search comparisons per conversion using a binary-weighted C-DAC. Supports capacitor mismatch, comparator noise/offset/hysteresis, input-referred noise, gain error, and aperture jitter.

### Ideal conversion

```python
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.dataconverter import InputType

# 12-bit, 1 V reference, single-ended input
adc = SARADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)

print(adc.convert(0.0))    # 0
print(adc.convert(0.5))    # ~2048
print(adc.convert(1.0))    # 4095
```

### With non-idealities

```python
adc = SARADC(
    n_bits=12,
    v_ref=1.0,
    cap_mismatch=0.001,         # 0.1% capacitor mismatch (static DNL/INL)
    comparator_params={
        'noise_rms': 0.5e-3,    # 0.5 mV RMS comparator noise
        'offset':    1e-3,      # 1 mV comparator offset
        'hysteresis': 0.5e-3,   # 0.5 mV hysteresis
    },
    noise_rms=100e-6,           # 100 µV kT/C sampling noise
    offset=2e-3,                # 2 mV input-referred offset
    gain_error=0.002,           # +0.2% gain error
    t_jitter=1e-12,             # 1 ps aperture jitter
)
```

### Cycle-by-cycle trace

`convert_with_trace()` returns the final code plus a dict with every bit decision and C-DAC voltage:

```python
code, trace = adc.convert_with_trace(0.37)

print(f"Code: {code}")
for k, (bit, v_dac) in enumerate(zip(trace['bits'], trace['v_dac'])):
    print(f"  Bit {k}: trial_v_dac={v_dac:.4f} V  →  bit={bit}")
```

### Differential input

```python
adc_diff = SARADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

# Input range: v_diff = v_pos - v_neg in [-v_ref/2, +v_ref/2]
code = adc_diff.convert((0.75, 0.25))   # v_diff = +0.5 V
code = adc_diff.convert((0.5,  0.5))    # v_diff =  0.0 V
```

---

## CurrentSteeringDAC

A current-steering DAC that supports binary, thermometer, and segmented topologies. All current sources are always conducting; they are steered between positive and negative output rails by a decoder.

Segmentation is controlled by `n_therm_bits`:
- `n_therm_bits = 0` — fully binary-weighted (default)
- `n_therm_bits = n_bits` — fully thermometer (unary)
- `0 < n_therm_bits < n_bits` — segmented (MSBs thermometer, LSBs binary)

### Binary mode (default)

```python
from pyDataconverter.architectures.CurrentSteeringDAC import CurrentSteeringDAC
from pyDataconverter.dataconverter import OutputType

# 8-bit binary DAC, 100 µA unit current, 1 kΩ load → full-scale ~10 mV per LSB
dac = CurrentSteeringDAC(
    n_bits=8,
    n_therm_bits=0,       # fully binary
    i_unit=100e-6,
    r_load=1000.0,
    output_type=OutputType.SINGLE,
)

v_out = dac.convert(128)   # ~half-scale voltage
```

### Thermometer mode

```python
# 4-bit fully thermometer — 15 unit sources
dac_therm = CurrentSteeringDAC(
    n_bits=4,
    n_therm_bits=4,       # fully thermometer
    i_unit=100e-6,
    r_load=1000.0,
    output_type=OutputType.SINGLE,
)
```

### Segmented mode

```python
# 8-bit segmented: top 4 bits thermometer, bottom 4 bits binary
dac_seg = CurrentSteeringDAC(
    n_bits=8,
    n_therm_bits=4,
    i_unit=100e-6,
    r_load=1000.0,
    output_type=OutputType.SINGLE,
)
```

### Differential output

```python
dac_diff = CurrentSteeringDAC(
    n_bits=8,
    n_therm_bits=4,
    i_unit=100e-6,
    r_load=1000.0,
    output_type=OutputType.DIFFERENTIAL,
)

v_pos, v_neg = dac_diff.convert(255)   # full scale
v_pos, v_neg = dac_diff.convert(128)   # mid scale
v_diff = v_pos - v_neg
```

### With current source mismatch

```python
dac = CurrentSteeringDAC(
    n_bits=10,
    n_therm_bits=4,
    i_unit=50e-6,
    r_load=2000.0,
    current_mismatch=0.005,   # 0.5% mismatch per source (static DNL/INL)
    output_type=OutputType.DIFFERENTIAL,
)
```

---

## Signal Generation

```python
from pyDataconverter.utils.signal_gen import (
    generate_sine,
    generate_coherent_sine,
    generate_ramp,
    generate_two_tone,
    generate_imd_tones,
    convert_to_differential,
)
import numpy as np

fs = 1e6   # 1 MHz sampling rate

# --- Simple sine ---
sig = generate_sine(frequency=10e3, sampling_rate=fs, amplitude=0.4, offset=0.5)

# --- Coherent sine (no spectral leakage — use this for FFT metrics) ---
n_fft = 4096
sig, f_in = generate_coherent_sine(sampling_rate=fs, n_fft=n_fft, n_fin=97,
                                    amplitude=0.4, offset=0.5)
# f_in = 97 / 4096 * 1e6 ≈ 23.7 kHz, exactly on an FFT bin

# --- Ramp (for static linearity testing) ---
ramp = generate_ramp(samples=10000, v_min=0.0, v_max=1.0)

# --- Two-tone (for IMD testing) ---
two_tone = generate_two_tone(f1=9.9e3, f2=10.1e3, sampling_rate=fs,
                              amplitude1=0.2, amplitude2=0.2)

# --- IMD signal with expected product frequencies ---
imd_sig, imd_freqs = generate_imd_tones(f1=100e3, delta_f=1e3,
                                         sampling_rate=fs, amplitude=0.4)
print(imd_freqs['imd3'])   # {'2f1-f2': 99000.0, '2f2-f1': 101000.0, ...}

# --- Single-ended to differential ---
v_pos, v_neg = convert_to_differential(sig, vcm=0.5)
```

---

## Analysis

### Static Metrics — ADC

Measures offset, gain error, DNL, and INL from a ramp sweep.

```python
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_ramp
from pyDataconverter.utils.metrics import calculate_adc_static_metrics
import numpy as np

adc = SimpleADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE,
                offset=5e-3, gain_error=0.01)

ramp = generate_ramp(samples=50000, v_min=0.0, v_max=1.0)
codes = np.array([adc.convert(v) for v in ramp])

metrics = calculate_adc_static_metrics(ramp, codes, n_bits=8, v_ref=1.0)

print(f"Offset:      {metrics['Offset']*1000:.2f} mV")
print(f"Gain error:  {metrics['GainError']*100:.3f} %")
print(f"Max DNL:     {metrics['MaxDNL']:.3f} LSB")
print(f"Max INL:     {metrics['MaxINL']:.3f} LSB")
```

### Dynamic Metrics — ADC / DAC

Measures SNR, SNDR, SFDR, THD, and ENOB from an FFT. Use coherent sampling to avoid spectral leakage.

```python
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine
from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics
import numpy as np

fs    = 1e6
n_fft = 4096
adc   = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE,
                  noise_rms=100e-6)

sig, f_in = generate_coherent_sine(sampling_rate=fs, n_fft=n_fft, n_fin=97,
                                    amplitude=0.4, offset=0.5)
codes = np.array([adc.convert(v) for v in sig])

# Normalize codes to voltage for FFT
voltages = codes / (2**12 - 1) * adc.v_ref

metrics = calculate_adc_dynamic_metrics(time_data=voltages, fs=fs,
                                         full_scale=adc.v_ref)

print(f"SNR:   {metrics['SNR']:.1f} dB")
print(f"SNDR:  {metrics['SNDR']:.1f} dB")
print(f"SFDR:  {metrics['SFDR']:.1f} dB")
print(f"THD:   {metrics['THD']:.1f} dB")
print(f"ENOB:  {metrics['ENOB']:.2f} bits")
```

---

## Visualization

### ADC Plots

```python
import matplotlib.pyplot as plt
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.visualizations.adc_plots import (
    plot_transfer_function,
    plot_conversion,
)
from pyDataconverter.utils.signal_gen import generate_sine
import numpy as np

adc = SimpleADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE,
                offset=10e-3, gain_error=0.02)

# Transfer function + quantization error (ramp sweep)
fig, axes = plot_transfer_function(adc, v_min=0.0, v_max=1.0)
plt.show()

# Time-domain: input signal vs output codes
fs = 100e3
sig = generate_sine(frequency=1e3, sampling_rate=fs, amplitude=0.4, offset=0.5,
                    duration=2e-3)
t   = np.arange(len(sig)) / fs
codes = np.array([adc.convert(v) for v in sig])

fig, axes = plot_conversion(x=t, input_signal=sig, output_codes=codes,
                             title='SimpleADC 6-bit', xlabel='Time (s)')
plt.show()
```

### DAC Plots

```python
import matplotlib.pyplot as plt
from pyDataconverter.architectures.SimpleDAC import SimpleDAC
from pyDataconverter.utils.visualizations.dac_plots import (
    plot_transfer_curve,
    plot_inl_dnl,
    plot_output_spectrum,
)

dac = SimpleDAC(n_bits=8, v_ref=1.0, offset=5e-3, gain_error=0.01,
                noise_rms=200e-6)

# Transfer curve: ideal vs actual + error in LSB
fig, axes = plot_transfer_curve(dac)
plt.show()

# INL / DNL bar charts
fig, axes = plot_inl_dnl(dac)
plt.show()

# Output spectrum (coherent sine drive)
fs = 1e6
ax = plot_output_spectrum(dac, fs=fs, f_sig=10e3, n_fft=4096)
plt.show()
```

### Flash ADC Visualization

Static snapshot and animated sweep of the comparator bank.

```python
import numpy as np
from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.utils.visualizations.visualize_FlashADC import (
    visualize_flash_adc,
    animate_flash_adc,
)

adc = FlashADC(n_bits=3, v_ref=1.0, offset_std=5e-3,
               comparator_params={'noise_rms': 2e-3})

# Static snapshot at a single input voltage
visualize_flash_adc(adc, input_voltage=0.6)

# Animated sweep
t = np.linspace(0, 2 * np.pi, 60)
voltages = 0.5 + 0.45 * np.sin(t)
animate_flash_adc(adc, voltages, interval=0.08)
```

### SAR ADC Visualization

Static snapshot of the bit-cycling funnel and animated conversion.

```python
import numpy as np
from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.visualizations.visualize_SARADC import (
    visualize_sar_adc,
    animate_sar_conversion,
    animate_sar_adc,
)

adc = SARADC(n_bits=4, v_ref=1.0, cap_mismatch=0.005,
             comparator_params={'noise_rms': 1e-3})

# Static snapshot at a single input voltage
visualize_sar_adc(adc, input_voltage=0.37)

# Interactive mode: slider to step through bit cycles
visualize_sar_adc(adc, interactive=True)

# Animated single conversion (shows each bit decision in turn)
animate_sar_conversion(adc, input_voltage=0.62)

# Animated sequence of conversions (e.g., a sine wave)
t = np.linspace(0, 2 * np.pi, 40)
v_in = 0.5 + 0.45 * np.sin(t)
animate_sar_adc(adc, input_voltages=v_in)
```

---

## Putting It All Together

A complete ADC characterization flow in ~30 lines:

```python
import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_ramp, generate_coherent_sine
from pyDataconverter.utils.metrics import (
    calculate_adc_static_metrics,
    calculate_adc_dynamic_metrics,
)
from pyDataconverter.utils.visualizations.adc_plots import plot_transfer_function

# 1. Build the converter
adc = SimpleADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                noise_rms=150e-6, offset=3e-3, gain_error=0.005)

# 2. Static characterization
ramp  = generate_ramp(samples=100000, v_min=0.0, v_max=1.0)
codes = np.array([adc.convert(v) for v in ramp])
static = calculate_adc_static_metrics(ramp, codes, n_bits=10, v_ref=1.0)
print(f"Max DNL: {static['MaxDNL']:.3f} LSB  |  Max INL: {static['MaxINL']:.3f} LSB")

# 3. Dynamic characterization
fs, n_fft = 1e6, 8192
sig, f_in = generate_coherent_sine(fs, n_fft, n_fin=113, amplitude=0.45, offset=0.5)
out = np.array([adc.convert(v) for v in sig])
voltages = out / (2**10 - 1) * adc.v_ref
dyn = calculate_adc_dynamic_metrics(time_data=voltages, fs=fs, full_scale=adc.v_ref)
print(f"SNR: {dyn['SNR']:.1f} dB  |  ENOB: {dyn['ENOB']:.2f} bits  |  SFDR: {dyn['SFDR']:.1f} dB")

# 4. Plot transfer function
fig, axes = plot_transfer_function(adc, v_min=0.0, v_max=1.0)
plt.show()
```

---

## Next Steps

- See `examples/` for complete runnable scripts:
  - `simple_adc_example.py` — ADC non-ideality sweeps
  - `simple_dac_example.py` — DAC non-ideality and spectrum examples
  - `flash_adc_example.py` — Flash ADC with non-idealities and animation
  - `sar_adc_example.py` — SAR ADC with C-DAC mismatch, static/dynamic metrics, and visualization
- See `docs/api_reference.md` for full parameter documentation on every class and function.
