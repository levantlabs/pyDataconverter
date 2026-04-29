# pyDataconverter

A Python toolbox for modeling and analyzing Data Converters (ADCs and DACs) and their performance metrics.

## Features

- **Signal Generation** — sine waves, ramps, multi-tone, coherent, IMD tones, PRBS, and digital equivalents; single-ended and differential
- **ADC Architectures** — SimpleADC, FlashADC, SAR (including Multibit and Noise-shaping), PipelinedADC, TimeInterleavedADC
- **DAC Architectures** — SimpleDAC, CurrentSteeringDAC, R2RDAC, ResistorStringDAC, SegmentedResistorDAC
- **Component Models** — comparator, capacitive DAC, capacitor, reference ladder, current source, residue amplifier, decoder
- **Performance Metrics** — dynamic (SNR, SNDR, SFDR, THD, ENOB) and static (DNL, INL, gain/offset error, monotonicity) for both ADCs and DACs
- **FFT Tools** — spectral analysis, harmonic detection, coherent frequency generation, windowing
- **Visualization** — transfer functions, DNL/INL plots, FFT spectra, and interactive architecture animators

## Installation

```bash
git clone https://github.com/levantlabs/pyDataconverter.git
cd pyDataconverter
pip install -e .
```

Or install from PyPI:

```bash
pip install pyDataconverter
```

## Quick Start

```python
import numpy as np
from pyDataconverter.architectures import SimpleADC
from pyDataconverter.dataconverter import InputType
from pyDataconverter.utils.signal_gen import generate_coherent_sine
import pyDataconverter.utils.metrics as metrics

# 12-bit differential ADC
adc = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)

# Coherent sine wave (avoids spectral leakage)
fs = 1e6
NFFT = 1024
sine, f_in = generate_coherent_sine(fs, NFFT, n_cycles=11, amplitude=0.45)

# Convert
codes = [adc.convert((v, -v)) for v in sine]

# Dynamic metrics
result = metrics.calculate_adc_dynamic_metrics(
    time_data=np.array(codes), fs=fs, f0=f_in
)
print(f"SNDR: {result['SNDR']:.1f} dB")
print(f"SFDR: {result['SFDR']:.1f} dB")
print(f"ENOB: {result['ENOB']:.2f} bits")
```

## ADC Architectures

### SimpleADC

Ideal quantizer with optional non-idealities: thermal noise, offset, gain error, and aperture jitter.

```python
from pyDataconverter.architectures import SimpleADC
from pyDataconverter.dataconverter import InputType, QuantizationMode

adc = SimpleADC(
    n_bits=12,
    v_ref=1.0,
    input_type=InputType.DIFFERENTIAL,
    noise_rms=50e-6,       # 50 µV input-referred noise
    offset=1e-3,           # 1 mV offset
    gain_error=0.001,      # 0.1% gain error
    t_jitter=1e-12,        # 1 ps aperture jitter
    quantization_mode=QuantizationMode.FLOOR,
)
code = adc.convert((v_pos, v_neg))
```

### SAR ADC

Successive-approximation ADC using a capacitive DAC and comparator. Supports single-ended and differential topologies.

```python
from pyDataconverter.architectures import SARADC
from pyDataconverter.components.cdac import DifferentialCDAC
from pyDataconverter.dataconverter import InputType

cdac = DifferentialCDAC(n_bits=10)
adc = SARADC(n_bits=10, v_ref=1.0, input_type=InputType.DIFFERENTIAL, cdac=cdac)
code = adc.convert((v_pos, v_neg))

# Inspect the bit-by-bit conversion trace
trace = adc.last_trace
```

Also available: `MultibitSARADC` and `NoiseshapingSARADC`.

### FlashADC

Parallel comparator bank — one comparator per threshold level.

```python
from pyDataconverter.architectures import FlashADC

adc = FlashADC(n_bits=4, v_ref=1.0)
code = adc.convert(v_in)
```

### PipelinedADC

Multi-stage pipelined converter. Each stage uses a sub-ADC, sub-DAC, and residue amplifier.

```python
from pyDataconverter.architectures import PipelinedADC, SimpleADC, SimpleDAC
from pyDataconverter.architectures.PipelinedADC import PipelineStage
from pyDataconverter.components.residue_amplifier import ResidueAmplifier

stage = PipelineStage(
    sub_adc=SimpleADC(n_bits=3, v_ref=1.0),
    sub_dac=SimpleDAC(n_bits=3, v_ref=1.0),
    residue_amp=ResidueAmplifier(gain=8.0),
    fs=100e6,
    H=8,
)
backend = SimpleADC(n_bits=6, v_ref=1.0)
adc = PipelinedADC(n_bits=12, v_ref=1.0, stages=[stage], backend=backend, backend_H=1, fs=100e6)
code = adc.convert(v_in)
```

### TimeInterleavedADC

M interleaved channels with per-channel offset, gain, timing-skew, and bandwidth mismatch.

```python
from pyDataconverter.architectures import SimpleADC, TimeInterleavedADC
from pyDataconverter.dataconverter import InputType

template = SimpleADC(n_bits=12, v_ref=1.0, input_type=InputType.SINGLE)
ti_adc = TimeInterleavedADC(
    channels=4,
    sub_adc_template=template,
    fs=1e9,
    offset=0.002,        # 2 mV stddev per channel
    timing_skew=5e-12,   # 5 ps stddev per channel
    seed=42,
)
codes = ti_adc.convert_waveform(v_dense, t_dense)
```

## DAC Architectures

### SimpleDAC

Ideal voltage-mode DAC with optional gain error, offset, and output noise.

```python
from pyDataconverter.architectures import SimpleDAC
from pyDataconverter.dataconverter import OutputType

dac = SimpleDAC(n_bits=12, v_ref=1.0, output_type=OutputType.SINGLE)
voltage = dac.convert(code)
```

### CurrentSteeringDAC

Current-mode DAC with a programmable current mirror array and per-source mismatch.

```python
from pyDataconverter.architectures import CurrentSteeringDAC

dac = CurrentSteeringDAC(n_bits=10, i_ref=1e-3, mismatch=0.001)
current = dac.convert(code)
```

### R2RDAC

Binary-weighted R-2R resistor ladder DAC.

```python
from pyDataconverter.architectures import R2RDAC

dac = R2RDAC(n_bits=10, v_ref=1.0, mismatch=0.001)
voltage = dac.convert(code)
```

### ResistorStringDAC

Single resistor chain with tap points. Inherently monotonic.

```python
from pyDataconverter.architectures import ResistorStringDAC

dac = ResistorStringDAC(n_bits=8, v_ref=1.0, mismatch=0.002)
voltage = dac.convert(code)
```

### SegmentedResistorDAC

Coarse + fine resistor string hybrid for reduced component count.

```python
from pyDataconverter.architectures import SegmentedResistorDAC

dac = SegmentedResistorDAC(n_bits=10, n_coarse_bits=5, v_ref=1.0, mismatch=0.001)
voltage = dac.convert(code)
```

## Component Models

Low-level building blocks used inside the architecture models, also usable directly.

| Component | Class | Key parameters |
|-----------|-------|----------------|
| Comparator | `DifferentialComparator` | threshold, hysteresis, aperture delay, noise |
| Capacitive DAC | `SingleEndedCDAC`, `DifferentialCDAC` | n_bits, unit capacitor size, mismatch |
| Capacitor | `Capacitor` | nominal value, unit mismatch stddev |
| Reference ladder | `ReferenceLadder` | v_min, v_max, n taps, mismatch |
| Current source | `CurrentSource` | i_nominal, mismatch |
| Residue amplifier | `ResidueAmplifier` | gain, bandwidth, noise |
| Decoder | `Decoder` | thermometer or binary |

```python
from pyDataconverter.components.comparator import DifferentialComparator
from pyDataconverter.components.cdac import DifferentialCDAC

comp = DifferentialComparator(threshold=0.0, hysteresis=1e-3, noise_rms=100e-6)
result = comp.compare(v_pos, v_neg)

cdac = DifferentialCDAC(n_bits=10, mismatch=0.002, seed=42)
cdac.apply_mismatch()          # draw new mismatch for Monte Carlo
voltage = cdac.get_voltage(code)
```

## Performance Metrics

### ADC Metrics

```python
import pyDataconverter.utils.metrics as metrics

# Dynamic (FFT-based)
result = metrics.calculate_adc_dynamic_metrics(
    time_data=np.array(codes), fs=1e6, f0=10e3, full_scale=4096
)
# result keys: SNR, SNDR, SFDR, THD, SINAD, ENOB

# Static (ramp-based)
static = metrics.calculate_adc_static_metrics(
    input_voltages=ramp_voltages, output_codes=codes, n_bits=12
)
# static keys: DNL, INL, offset_error, gain_error, is_monotonic

# Gain and offset error only
gain_err, offset_err = metrics.calculate_gain_offset_error(
    input_voltages=ramp_voltages, output_codes=codes, n_bits=12
)
```

### DAC Metrics

```python
# Dynamic
result = metrics.calculate_dac_dynamic_metrics(
    voltages=output_voltages, fs=1e6, full_scale=1.0
)
# result keys: SNR, SNDR, SFDR, THD, SINAD, ENOB

# Static
static = metrics.calculate_dac_static_metrics(dac=dac)
# static keys: DNL, INL, offset_error, gain_error, is_monotonic
```

## Signal Generation

```python
from pyDataconverter.utils.signal_gen import (
    generate_sine,
    generate_coherent_sine,
    generate_two_tone,
    generate_coherent_two_tone,
    generate_ramp,
    generate_imd_tones,
    generate_multitone,
    generate_digital_sine,
    generate_digital_ramp,
    convert_to_differential,
)

# Analog signals
sine = generate_sine(frequency=10e3, sampling_rate=1e6, amplitude=0.9)
sine, f_in = generate_coherent_sine(fs=1e6, NFFT=1024, n_cycles=11, amplitude=0.45)
two_tone = generate_two_tone(f1=9.9e3, f2=10.1e3, sampling_rate=1e6)
ramp = generate_ramp(v_min=-0.5, v_max=0.5, n_points=4096)

# Differential
v_pos, v_neg = convert_to_differential(sine, vcm=0.5)

# Digital equivalents
codes = generate_digital_sine(n_bits=12, f_in=10e3, fs=1e6)
codes = generate_digital_ramp(n_bits=12, n_points=4096)
```

## FFT Analysis

```python
from pyDataconverter.utils.fft_analysis import compute_fft, FFTNormalization, find_harmonics

freqs, mags = compute_fft(signal, fs=1e6, normalization=FFTNormalization.DBFS, full_scale=4096)
harmonics = find_harmonics(freqs, mags, f0=10e3, n_harmonics=5)
```

## Visualization

```python
# ADC plots
from pyDataconverter.utils.visualizations.adc_plots import plot_transfer_function, plot_conversion

plot_transfer_function(adc)
plot_conversion(t, v_in, codes)

# DAC plots
from pyDataconverter.utils.visualizations.dac_plots import (
    plot_transfer_curve, plot_inl_dnl, plot_output_spectrum
)

plot_transfer_curve(dac)
plot_inl_dnl(dac)
plot_output_spectrum(freqs, mags, fs=1e6, f0=10e3)

# FFT plot
from pyDataconverter.utils.visualizations.fft_plots import plot_fft

plot_fft(freqs, mags, title="ADC Output Spectrum")

# Interactive architecture visualizers
from pyDataconverter.utils.visualizations.visualize_SARADC import animate_sar_adc
from pyDataconverter.utils.visualizations.visualize_FlashADC import animate_flash_adc

animate_sar_adc(adc, v_in=0.3)
animate_flash_adc(adc, v_in=0.3)
```

## Examples

The `examples/` directory contains 15 scripts covering:

| Script | What it demonstrates |
|--------|---------------------|
| `simple_adc_example.py` | SimpleADC non-idealities, FFT, dynamic metrics |
| `simple_dac_example.py` | SimpleDAC transfer curve, spectrum, ZOH oversampling |
| `sar_adc_example.py` | SARADC transfer function, C-DAC mismatch, architecture visualizer |
| `flash_adc_example.py` | FlashADC operation and comparator-level detail |
| `pipelined_adc_example.py` | PipelinedADC construction and performance |
| `ti_adc_example.py` | TimeInterleavedADC mismatch spurs |
| `characterization_example.py` | Full ADC/DAC characterization flows |
| `dnl_inl_*.py` | Static linearity analysis |

## Project Structure

```
pyDataconverter/
├── pyDataconverter/
│   ├── dataconverter.py          # ADCBase, DACBase, InputType, QuantizationMode
│   ├── architectures/            # ADC and DAC implementations
│   │   ├── SimpleADC.py
│   │   ├── SimpleDAC.py
│   │   ├── FlashADC.py
│   │   ├── SARADC.py
│   │   ├── PipelinedADC.py
│   │   ├── TimeInterleavedADC.py
│   │   ├── CurrentSteeringDAC.py
│   │   ├── R2RDAC.py
│   │   ├── ResistorStringDAC.py
│   │   └── SegmentedResistorDAC.py
│   ├── components/               # Circuit-level component models
│   │   ├── comparator.py
│   │   ├── cdac.py
│   │   ├── capacitor.py
│   │   ├── reference.py
│   │   ├── current_source.py
│   │   ├── residue_amplifier.py
│   │   └── decoder.py
│   └── utils/
│       ├── signal_gen.py         # Signal generation
│       ├── fft_analysis.py       # FFT and spectral tools
│       ├── characterization.py   # Characterization flows
│       ├── nodal_solver.py       # Circuit network solver
│       ├── metrics/              # ADC and DAC performance metrics
│       └── visualizations/       # Plotting and interactive visualizers
├── examples/                     # 15 example scripts
├── tests/                        # Test suite (~973 tests)
└── docs/                         # API reference and design documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-3.0 license — see the LICENSE file for details.
