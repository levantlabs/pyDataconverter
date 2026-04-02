# Characterization Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `pyDataconverter/utils/characterization.py` with `measure_dynamic_range` and `measure_erbw` — high-level convenience functions that drive an ADC through an amplitude or frequency sweep and return the DR or ERBW.

**Architecture:** This module sits between the architectures layer and the metrics layer. It uses duck typing (any object with a `.convert(float) -> int` method works). It calls `generate_coherent_sine` to produce the stimulus, calls `calculate_adc_dynamic_metrics` for each sweep point, and calls `calculate_dynamic_range_from_curve` / `calculate_erbw_from_curve` (from the metrics enhancements plan) to extract the scalar result.

**Dependency:** Requires `calculate_dynamic_range_from_curve` and `calculate_erbw_from_curve` from `2026-04-01-metrics-enhancements.md` to be implemented first.

**Tech Stack:** numpy, existing `generate_coherent_sine` from `signal_gen.py`, existing metrics functions.

---

## File Map

| File | Change |
|---|---|
| `pyDataconverter/utils/characterization.py` | New file: `measure_dynamic_range`, `measure_erbw` |
| `pyDataconverter/utils/__init__.py` | Re-export the two new functions (or leave bare — check existing pattern) |
| `tests/test_characterization.py` | New test file |

---

## Task 1: Create `measure_dynamic_range`

**Files:**
- Create: `pyDataconverter/utils/characterization.py`
- Create: `tests/test_characterization.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_characterization.py`:

```python
"""Tests for pyDataconverter.utils.characterization."""
import numpy as np
import pytest
from pyDataconverter.architectures.FlashADC import FlashADC


def _make_adc(n_bits=8, offset_std=0.0):
    return FlashADC(n_bits=n_bits, v_ref=1.0, offset_std=offset_std)


def test_measure_dynamic_range_returns_dict():
    from pyDataconverter.utils.characterization import measure_dynamic_range
    adc = _make_adc(n_bits=8)
    result = measure_dynamic_range(adc, n_bits=8, v_ref=1.0, fs=1e6,
                                   n_fft=512, n_fin=13)
    assert isinstance(result, dict)
    assert 'DR_dB' in result
    assert 'Amplitudes_dBFS' in result
    assert 'SNR_values' in result


def test_measure_dynamic_range_ideal_adc():
    """8-bit ideal ADC should have DR close to 8*6.02 + 1.76 ≈ 49.9 dB."""
    from pyDataconverter.utils.characterization import measure_dynamic_range
    adc = _make_adc(n_bits=8)
    result = measure_dynamic_range(adc, n_bits=8, v_ref=1.0, fs=1e6,
                                   n_fft=1024, n_fin=13, n_amplitudes=15)
    # DR for an ideal N-bit ADC is approximately 6.02*N + 1.76 dB
    expected_dr = 6.02 * 8 + 1.76
    assert abs(result['DR_dB'] - expected_dr) < 5.0  # within 5 dB


def test_measure_dynamic_range_array_lengths():
    """Amplitudes_dBFS and SNR_values have n_amplitudes entries."""
    from pyDataconverter.utils.characterization import measure_dynamic_range
    adc = _make_adc(n_bits=6)
    n_amp = 10
    result = measure_dynamic_range(adc, n_bits=6, v_ref=1.0, fs=1e6,
                                   n_fft=512, n_fin=7, n_amplitudes=n_amp)
    assert len(result['Amplitudes_dBFS']) == n_amp
    assert len(result['SNR_values']) == n_amp


def test_measure_dynamic_range_duck_typing():
    """Any object with .convert(float)->int works as the ADC."""
    from pyDataconverter.utils.characterization import measure_dynamic_range

    class SimpleQuantizer:
        def __init__(self, n_bits, v_ref):
            self.n_bits = n_bits
            self.v_ref  = v_ref
        def convert(self, v):
            code = int(v / self.v_ref * 2**self.n_bits)
            return max(0, min(2**self.n_bits - 1, code))

    adc = SimpleQuantizer(n_bits=6, v_ref=1.0)
    result = measure_dynamic_range(adc, n_bits=6, v_ref=1.0, fs=1e6,
                                   n_fft=512, n_fin=7, n_amplitudes=8)
    assert 'DR_dB' in result
    assert result['DR_dB'] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lazarus/python-dev/pyDataconverter
python -m pytest tests/test_characterization.py -v
```

Expected: ModuleNotFoundError for `characterization`.

- [ ] **Step 3: Implement `measure_dynamic_range`**

Create `pyDataconverter/utils/characterization.py`:

```python
"""
Characterization Utilities
==========================

High-level convenience functions that drive an ADC through a sweep and
return a scalar figure-of-merit (DR, ERBW).

These functions accept any object that implements ``.convert(float) -> int``
so they work with FlashADC, SARADC, or any custom model.
"""

import numpy as np
from typing import Dict, Optional

from .signal_gen import generate_coherent_sine
from .metrics import (
    calculate_adc_dynamic_metrics,
    calculate_dynamic_range_from_curve,
    calculate_erbw_from_curve,
)


def measure_dynamic_range(
        adc,
        n_bits: int,
        v_ref: float,
        fs: float,
        n_fft: int,
        n_fin: int,
        n_amplitudes: int = 20,
        amplitude_range_dBFS: tuple = (-80.0, -1.0),
) -> Dict:
    """
    Measure ADC dynamic range by sweeping input amplitude.

    Generates coherent sine waves at logarithmically-spaced amplitudes
    from ``amplitude_range_dBFS[0]`` to ``amplitude_range_dBFS[1]`` dBFS,
    converts each through ``adc``, measures SNR, then calls
    ``calculate_dynamic_range_from_curve`` to find where SNR = 0 dB.

    Args:
        adc: ADC model.  Must implement ``.convert(float) -> int``.
        n_bits: ADC resolution (bits).
        v_ref: Full-scale reference voltage (V).
        fs: Sampling rate (Hz).
        n_fft: FFT / record length (samples per measurement).
        n_fin: Input frequency bin number (integer, coherent with n_fft).
               Actual frequency = n_fin / n_fft * fs.
        n_amplitudes: Number of amplitude steps in the sweep (default 20).
        amplitude_range_dBFS: (low_dBFS, high_dBFS) sweep range.
            Default (-80, -1) covers most of the dynamic range for a
            well-behaved ADC.

    Returns:
        Dict with keys:
            DR_dB          : Dynamic range in dB.
            Amplitudes_dBFS: np.ndarray of sweep amplitudes (dBFS).
            SNR_values     : np.ndarray of measured SNR (dB) at each step.
            AmplitudeAtSNR0_dBFS: Amplitude where SNR interpolates to 0 dB.
    """
    full_scale_amp = v_ref / 2.0
    offset         = v_ref / 2.0

    amplitudes_dBFS = np.linspace(amplitude_range_dBFS[0],
                                   amplitude_range_dBFS[1],
                                   n_amplitudes)
    amplitudes_v = full_scale_amp * 10 ** (amplitudes_dBFS / 20.0)

    snr_values = np.zeros(n_amplitudes)
    for i, amp in enumerate(amplitudes_v):
        vin, _ = generate_coherent_sine(fs, n_fft, n_fin,
                                        amplitude=amp, offset=offset)
        codes = np.array([adc.convert(float(v)) for v in vin], dtype=float)
        m = calculate_adc_dynamic_metrics(time_data=codes, fs=fs)
        snr_values[i] = m['SNR']

    dr_result = calculate_dynamic_range_from_curve(amplitudes_dBFS, snr_values)

    return {
        'DR_dB':               dr_result['DR_dB'],
        'AmplitudeAtSNR0_dBFS': dr_result['AmplitudeAtSNR0_dB'],
        'Amplitudes_dBFS':     amplitudes_dBFS,
        'SNR_values':          snr_values,
    }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_characterization.py::test_measure_dynamic_range_returns_dict tests/test_characterization.py::test_measure_dynamic_range_array_lengths tests/test_characterization.py::test_measure_dynamic_range_duck_typing -v
```

Expected: 3 passed. (Skip the ideal_adc test for now — it's slow.)

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/utils/characterization.py tests/test_characterization.py
git commit -m "feat: add characterization module with measure_dynamic_range"
```

---

## Task 2: Add `measure_erbw`

**Files:**
- Modify: `pyDataconverter/utils/characterization.py`
- Modify: `tests/test_characterization.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_characterization.py`:

```python
def test_measure_erbw_returns_dict():
    from pyDataconverter.utils.characterization import measure_erbw
    adc = _make_adc(n_bits=8)
    result = measure_erbw(adc, n_bits=8, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=8)
    assert isinstance(result, dict)
    assert 'ERBW_Hz' in result
    assert 'Frequencies_Hz' in result
    assert 'ENOB_values' in result


def test_measure_erbw_array_lengths():
    from pyDataconverter.utils.characterization import measure_erbw
    adc = _make_adc(n_bits=6)
    n_freq = 7
    result = measure_erbw(adc, n_bits=6, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=n_freq)
    assert len(result['Frequencies_Hz']) == n_freq
    assert len(result['ENOB_values']) == n_freq


def test_measure_erbw_positive():
    """ERBW is a positive finite frequency."""
    from pyDataconverter.utils.characterization import measure_erbw
    adc = _make_adc(n_bits=8)
    result = measure_erbw(adc, n_bits=8, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=8)
    assert np.isfinite(result['ERBW_Hz'])
    assert result['ERBW_Hz'] > 0


def test_measure_erbw_duck_typing():
    """measure_erbw works with any object having .convert(float)->int."""
    from pyDataconverter.utils.characterization import measure_erbw

    class SimpleQuantizer:
        def __init__(self, n_bits, v_ref):
            self.n_bits = n_bits
            self.v_ref  = v_ref
        def convert(self, v):
            code = int(v / self.v_ref * 2**self.n_bits)
            return max(0, min(2**self.n_bits - 1, code))

    adc = SimpleQuantizer(n_bits=6, v_ref=1.0)
    result = measure_erbw(adc, n_bits=6, v_ref=1.0,
                          fs=10e6, n_fft=512,
                          freq_range_hz=(1e3, 4e6), n_frequencies=6)
    assert 'ERBW_Hz' in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_characterization.py::test_measure_erbw_returns_dict tests/test_characterization.py::test_measure_erbw_array_lengths tests/test_characterization.py::test_measure_erbw_positive tests/test_characterization.py::test_measure_erbw_duck_typing -v
```

Expected: AttributeError — `measure_erbw` not yet defined.

- [ ] **Step 3: Implement `measure_erbw`**

Add to `pyDataconverter/utils/characterization.py`:

```python
def measure_erbw(
        adc,
        n_bits: int,
        v_ref: float,
        fs: float,
        n_fft: int,
        freq_range_hz: tuple,
        n_frequencies: int = 20,
        amplitude_dBFS: float = -3.0,
) -> Dict:
    """
    Measure ADC effective resolution bandwidth (ERBW).

    Generates coherent sine waves at logarithmically-spaced frequencies
    from ``freq_range_hz[0]`` to ``freq_range_hz[1]``, converts each
    through ``adc``, measures ENOB, then calls
    ``calculate_erbw_from_curve`` to find where ENOB drops by 0.5 bits.

    Each frequency uses the nearest coherent bin to the target (to avoid
    spectral leakage), so the actual frequency array may differ slightly
    from the requested one.

    Args:
        adc: ADC model.  Must implement ``.convert(float) -> int``.
        n_bits: ADC resolution (bits).
        v_ref: Full-scale reference voltage (V).
        fs: Sampling rate (Hz).
        n_fft: FFT / record length (samples per measurement).
        freq_range_hz: (f_low, f_high) frequency sweep range in Hz.
        n_frequencies: Number of frequency steps (default 20).
        amplitude_dBFS: Input amplitude in dBFS (default -3 dBFS ≈ full scale).

    Returns:
        Dict with keys:
            ERBW_Hz      : Effective resolution bandwidth in Hz.
            ENOB_ref     : ENOB at the lowest measured frequency.
            Frequencies_Hz: np.ndarray of actual measurement frequencies.
            ENOB_values  : np.ndarray of ENOB at each frequency.
    """
    full_scale_amp = v_ref / 2.0
    amplitude      = full_scale_amp * 10 ** (amplitude_dBFS / 20.0)
    offset         = v_ref / 2.0

    f_low, f_high = freq_range_hz
    target_freqs = np.geomspace(f_low, f_high, n_frequencies)

    # Snap each target to the nearest coherent bin
    actual_freqs = np.zeros(n_frequencies)
    enob_values  = np.zeros(n_frequencies)

    for i, f_target in enumerate(target_freqs):
        n_fin = max(1, round(f_target / fs * n_fft))
        n_fin = min(n_fin, n_fft // 2 - 1)
        actual_freq = n_fin / n_fft * fs
        actual_freqs[i] = actual_freq

        vin, _ = generate_coherent_sine(fs, n_fft, n_fin,
                                        amplitude=amplitude, offset=offset)
        codes = np.array([adc.convert(float(v)) for v in vin], dtype=float)
        m = calculate_adc_dynamic_metrics(time_data=codes, fs=fs)
        enob_values[i] = m['ENOB']

    erbw_result = calculate_erbw_from_curve(actual_freqs, enob_values)

    return {
        'ERBW_Hz':       erbw_result['ERBW_Hz'],
        'ENOB_ref':      erbw_result['ENOB_ref'],
        'Frequencies_Hz': actual_freqs,
        'ENOB_values':   enob_values,
    }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_characterization.py::test_measure_erbw_returns_dict tests/test_characterization.py::test_measure_erbw_array_lengths tests/test_characterization.py::test_measure_erbw_positive tests/test_characterization.py::test_measure_erbw_duck_typing -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyDataconverter/utils/characterization.py tests/test_characterization.py
git commit -m "feat: add measure_erbw to characterization module"
```
