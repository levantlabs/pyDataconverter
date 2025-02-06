# pyDataconverter

A Python toolbox for modeling and analyzing Data Converters (ADCs and DACs) and their performance metrics.

## Features

- Signal Generation
  - Sine waves, ramps, and multi-tone signals
  - Support for both single-ended and differential signals
  - Configurable signal parameters (frequency, amplitude, etc.)

- Data Converter Modeling
  - Base ADC class with extensible architecture
  - Support for various ADC architectures (SAR, Pipeline, etc.)
  - Configurable parameters (resolution, reference voltage, etc.)

- Analysis Tools
  - Dynamic performance metrics (SNR, SFDR, THD, etc.)
  - Static performance metrics (DNL, INL, etc.)
  - Spectral analysis with FFT tools
  - Histogram analysis for both uniform and sine wave inputs

## Installation

```bash
git clone https://github.com/yourusername/pyDataconverter.git
cd pyDataconverter
pip install -e .
```
This can also be installed as a python module with ```pip install pyDataconverter```

## Quick Start

```python
from pyDataconverter import SimpleADC
from pyDataconverter.utils.signal_gen import generate_sine
from pyDataconverter.utils.metrics import calculate_dynamic_metrics

# Create ADC instance
adc = SimpleADC(n_bits=12, v_ref=1.0)

# Generate test signal
fs = 1e6  # 1 MHz sampling rate
f_in = 10e3  # 10 kHz input
signal = generate_sine(f_in, fs, amplitude=0.9)

# Convert signal
output = [adc.convert(v) for v in signal]

# Calculate metrics
metrics = calculate_dynamic_metrics(output, fs, f_in)
print(f"SNDR: {metrics['SNDR']:.1f} dB")
print(f"SFDR: {metrics['SFDR']:.1f} dB")
print(f"THD: {metrics['THD']:.1f} dB")
```

## Documentation

### Signal Generation

```python
# Generate single tone
signal = generate_sine(frequency=10e3, 
                      sampling_rate=1e6, 
                      amplitude=0.9)

# Generate two-tone signal
signal = generate_two_tone(f1=10e3, 
                          f2=11e3,
                          sampling_rate=1e6)

# Generate differential signals
v_pos, v_neg = convert_to_differential(signal, vcm=0.5)
```

### Analysis Functions

#### Dynamic Metrics
```python
metrics = calculate_dynamic_metrics(
    time_data=output,
    fs=1e6,
    f0=10e3,
    full_scale=1.0  # Optional: for dBFS results
)
```

#### Static Metrics
```python
metrics = calculate_static_metrics(
    input_voltages=ramp_in,
    output_codes=codes,
    n_bits=12
)
```

#### Histogram Analysis
```python
hist = calculate_histogram(
    codes=output_codes,
    n_bits=12,
    input_type='sine',
    remove_pdf=True
)
```

## Project Structure

```
pyDataconverter/
├── __init__.py
├── architectures/     # Data converter architecture implementations
├── examples/     # Example files that illustrate usage
├── utils/            # Utility functions
└── analysis/         # Analysis tools

```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
