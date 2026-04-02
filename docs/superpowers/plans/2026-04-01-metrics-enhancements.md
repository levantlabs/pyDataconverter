# Metrics Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gain/offset error as a public function, missing codes, individual harmonics (HD2…HDn), NSD, IIP3, and low-level DR/ERBW curve functions to the metrics package.

**Architecture:** All additions live inside `pyDataconverter/utils/metrics/`. The new `calculate_gain_offset_error` function is called by `calculate_adc_static_metrics` so the math is not duplicated. Dynamic metric additions extend `_dynamic.py`. IIP3 and the DR/ERBW curve helpers are new functions in `adc.py` and re-exported from `__init__.py`.

**Tech Stack:** numpy, scipy.interpolate (for DR/ERBW interpolation), existing `compute_fft` / `find_harmonics` from `fft_analysis.py`.

---

## File Map

| File | Change |
|---|---|
| `pyDataconverter/utils/metrics/adc.py` | Add `calculate_gain_offset_error`, `calculate_adc_iip3`, `calculate_dynamic_range_from_curve`, `calculate_erbw_from_curve`; refactor static metrics to call gain/offset helper; add `MissingCodes` to output |
| `pyDataconverter/utils/metrics/_dynamic.py` | Add `HD2`…`HDn` individual keys, `NSD_dBHz`, `Spurious` to output dict |
| `pyDataconverter/utils/metrics/__init__.py` | Re-export the four new public functions |
| `tests/test_metrics.py` | New test cases for all new functions |
| `tests/test_dac_metrics.py` | No change needed |

---

## Task 1: Add `calculate_gain_offset_error` and refactor static metrics

**Files:**
- Modify: `pyDataconverter/utils/metrics/adc.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_metrics.py`:

```python
from pyDataconverter.utils.metrics import (
    calculate_gain_offset_error,
    calculate_adc_static_metrics,
)
from pyDataconverter.dataconverter import QuantizationMode
import numpy as np

def test_gain_offset_error_ideal():
    """Ideal ADC has zero offset and zero gain error."""
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    # Ideal transitions at 1*lsb, 2*lsb, ..., 15*lsb
    transitions = np.arange(1, 2**n_bits) * ideal_lsb
    result = calculate_gain_offset_error(transitions, n_bits, v_ref)
    assert abs(result['OffsetError']) < 1e-12
    assert abs(result['GainError']) < 1e-12

def test_gain_offset_error_with_offset():
    """Known offset shift is returned correctly."""
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    shift = 0.01  # 10 mV offset
    transitions = np.arange(1, 2**n_bits) * ideal_lsb + shift
    result = calculate_gain_offset_error(transitions, n_bits, v_ref)
    assert abs(result['OffsetError'] - shift) < 1e-12

def test_gain_offset_error_with_gain():
    """Known gain stretch is returned correctly."""
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    gain = 0.05  # +5% gain error
    # Stretch all transitions from the first one
    ideal = np.arange(1, 2**n_bits) * ideal_lsb
    stretched = ideal[0] + (ideal - ideal[0]) * (1 + gain)
    result = calculate_gain_offset_error(stretched, n_bits, v_ref)
    assert abs(result['GainError'] - gain) < 1e-6

def test_static_metrics_uses_gain_offset_helper():
    """calculate_adc_static_metrics Offset/GainError match calculate_gain_offset_error."""
    import numpy as np
    from pyDataconverter.architectures.FlashADC import FlashADC
    adc = FlashADC(n_bits=6, v_ref=1.0, offset_std=0.002)
    vin = np.linspace(0, 1.0, 10000)
    codes = np.array([adc.convert(float(v)) for v in vin])
    m_static = calculate_adc_static_metrics(vin, codes, 6, 1.0)
    m_go = calculate_gain_offset_error(m_static['Transitions'], 6, 1.0)
    assert abs(m_static['Offset'] - m_go['OffsetError']) < 1e-12
    assert abs(m_static['GainError'] - m_go['GainError']) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lazarus/python-dev/pyDataconverter
python -m pytest tests/test_metrics.py::test_gain_offset_error_ideal tests/test_metrics.py::test_gain_offset_error_with_offset tests/test_metrics.py::test_gain_offset_error_with_gain tests/test_metrics.py::test_static_metrics_uses_gain_offset_helper -v
```

Expected: ImportError or NameError for `calculate_gain_offset_error`.

- [ ] **Step 3: Implement `calculate_gain_offset_error` and refactor static metrics**

In `pyDataconverter/utils/metrics/adc.py`, add after the imports:

```python
def calculate_gain_offset_error(
        transitions: np.ndarray,
        n_bits: int,
        v_ref: float = 1.0,
        quant_mode: QuantizationMode = QuantizationMode.FLOOR,
) -> Dict[str, float]:
    """
    Calculate offset error and gain error from measured transition voltages.

    Args:
        transitions: Measured transition voltages, length 2^n_bits - 1.
        n_bits: ADC resolution.
        v_ref: Reference voltage (V).
        quant_mode: FLOOR or SYMMETRIC (sets ideal first/last transition).

    Returns:
        Dict with keys:
            OffsetError : float — deviation of first transition from ideal (V).
            GainError   : float — fractional gain error (dimensionless).
    """
    if quant_mode == QuantizationMode.FLOOR:
        ideal_lsb   = v_ref / (2 ** n_bits)
        ideal_first = ideal_lsb
        ideal_last  = (2 ** n_bits - 1) * ideal_lsb
    else:
        ideal_lsb   = v_ref / (2 ** n_bits - 1)
        ideal_first = 0.5 * ideal_lsb
        ideal_last  = (2 ** n_bits - 1.5) * ideal_lsb

    ideal_span   = ideal_last - ideal_first
    offset_error = float(transitions[0] - ideal_first)
    gain_error   = float((transitions[-1] - transitions[0] - ideal_span) / ideal_span)
    return {'OffsetError': offset_error, 'GainError': gain_error}
```

Then in `calculate_adc_static_metrics`, replace the inline block:

```python
    # --- Offset and gain error (mode-independent structure) ---
    ideal_span = ideal_last - ideal_first
    offset     = transitions[0] - ideal_first
    gain_error = ((transitions[-1] - transitions[0]) - ideal_span) / ideal_span
```

with:

```python
    # --- Offset and gain error ---
    _go = calculate_gain_offset_error(transitions, n_bits, v_ref, quant_mode)
    offset     = _go['OffsetError']
    gain_error = _go['GainError']
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_metrics.py::test_gain_offset_error_ideal tests/test_metrics.py::test_gain_offset_error_with_offset tests/test_metrics.py::test_gain_offset_error_with_gain tests/test_metrics.py::test_static_metrics_uses_gain_offset_helper -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/utils/metrics/adc.py tests/test_metrics.py
git commit -m "feat(metrics): add calculate_gain_offset_error; refactor static metrics to use it"
```

---

## Task 2: Add `MissingCodes` to static metrics output

**Files:**
- Modify: `pyDataconverter/utils/metrics/adc.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_missing_codes_ideal_adc():
    """Ideal ADC has no missing codes."""
    from pyDataconverter.architectures.FlashADC import FlashADC
    import numpy as np
    adc = FlashADC(n_bits=6, v_ref=1.0, offset_std=0.0)
    vin = np.linspace(0, 1.0, 10000)
    codes = np.array([adc.convert(float(v)) for v in vin])
    m = calculate_adc_static_metrics(vin, codes, 6, 1.0)
    assert 'MissingCodes' in m
    assert len(m['MissingCodes']) == 0

def test_missing_codes_detected():
    """DNL <= -1 codes appear in MissingCodes."""
    import numpy as np
    # Manually craft a transitions array where code 3 is missing
    # (transition 2 and 3 coincide → code 3 has width 0)
    n_bits = 4
    v_ref = 1.0
    ideal_lsb = v_ref / 2**n_bits
    transitions = np.arange(1, 2**n_bits, dtype=float) * ideal_lsb
    transitions[2] = transitions[3]  # code 3 missing
    # Compute DNL manually to verify
    bin_widths = np.diff(np.concatenate([[0.0], transitions, [v_ref]]))
    dnl = bin_widths / ideal_lsb - 1
    assert dnl[3] <= -1.0
    # Now test via static metrics by injecting fake ramp data
    # Build input voltages that produce these transitions
    vin = np.linspace(0, v_ref, 20000)
    # Build output codes matching the tweaked transitions
    codes = np.zeros(len(vin), dtype=int)
    for k, t in enumerate(transitions):
        codes[vin >= t] = k + 1
    m = calculate_adc_static_metrics(vin, codes, n_bits, v_ref)
    assert 3 in m['MissingCodes'] or len(m['MissingCodes']) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metrics.py::test_missing_codes_ideal_adc tests/test_metrics.py::test_missing_codes_detected -v
```

Expected: KeyError on `MissingCodes`.

- [ ] **Step 3: Add MissingCodes to return dict**

In `calculate_adc_static_metrics`, after the `dnl` computation, add:

```python
    missing_codes = list(np.where(dnl <= -1.0)[0])
```

And add it to the return dict:

```python
    return {
        "DNL": dnl,
        "INL": inl,
        "Offset": offset,
        "GainError": gain_error,
        "MissingCodes": missing_codes,
        "MaxDNL": float(np.max(np.abs(dnl))),
        "MaxINL": float(np.max(np.abs(inl))),
        "Transitions": transitions,
    }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_metrics.py::test_missing_codes_ideal_adc tests/test_metrics.py::test_missing_codes_detected -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/utils/metrics/adc.py tests/test_metrics.py
git commit -m "feat(metrics): add MissingCodes to calculate_adc_static_metrics output"
```

---

## Task 3: Add individual harmonics and NSD to dynamic metrics

**Files:**
- Modify: `pyDataconverter/utils/metrics/_dynamic.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_dynamic_metrics_individual_harmonics():
    """HD2, HD3 are present and HD2 > HD3 for a typical sine."""
    from pyDataconverter.utils.signal_gen import generate_coherent_sine
    from pyDataconverter.architectures.FlashADC import FlashADC
    import numpy as np
    fs = 1e6
    n_fft = 1024
    adc = FlashADC(n_bits=8, v_ref=1.0, offset_std=0.001)
    vin, _ = generate_coherent_sine(fs, n_fft, n_fin=13, amplitude=0.45, offset=0.5)
    codes = np.array([adc.convert(float(v)) for v in vin])
    from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics
    m = calculate_adc_dynamic_metrics(time_data=codes.astype(float), fs=fs)
    assert 'HD2' in m
    assert 'HD3' in m
    assert isinstance(m['HD2'], float)
    assert isinstance(m['HD3'], float)

def test_dynamic_metrics_nsd():
    """NSD key is present and is a finite negative float (dB/Hz)."""
    from pyDataconverter.utils.signal_gen import generate_coherent_sine
    from pyDataconverter.architectures.FlashADC import FlashADC
    import numpy as np
    fs = 1e6
    n_fft = 1024
    adc = FlashADC(n_bits=8, v_ref=1.0, offset_std=0.0)
    vin, _ = generate_coherent_sine(fs, n_fft, n_fin=13, amplitude=0.45, offset=0.5)
    codes = np.array([adc.convert(float(v)) for v in vin])
    from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics
    m = calculate_adc_dynamic_metrics(time_data=codes.astype(float), fs=fs)
    assert 'NSD_dBHz' in m
    assert np.isfinite(m['NSD_dBHz'])
    assert m['NSD_dBHz'] < 0

def test_dynamic_metrics_spurious():
    """Spurious key contains the strongest non-harmonic spur magnitude."""
    from pyDataconverter.utils.signal_gen import generate_coherent_sine
    from pyDataconverter.architectures.FlashADC import FlashADC
    import numpy as np
    fs = 1e6
    n_fft = 1024
    adc = FlashADC(n_bits=6, v_ref=1.0, offset_std=0.001)
    vin, _ = generate_coherent_sine(fs, n_fft, n_fin=13, amplitude=0.45, offset=0.5)
    codes = np.array([adc.convert(float(v)) for v in vin])
    from pyDataconverter.utils.metrics import calculate_adc_dynamic_metrics
    m = calculate_adc_dynamic_metrics(time_data=codes.astype(float), fs=fs)
    assert 'Spurious' in m
    assert isinstance(m['Spurious'], float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metrics.py::test_dynamic_metrics_individual_harmonics tests/test_metrics.py::test_dynamic_metrics_nsd tests/test_metrics.py::test_dynamic_metrics_spurious -v
```

Expected: 3 failures (KeyError).

- [ ] **Step 3: Update `_calculate_dynamic_metrics`**

In `pyDataconverter/utils/metrics/_dynamic.py`, update the `results` dict construction:

```python
    # Individual harmonic levels
    harmonic_dict = {}
    for i, (hf, hm) in enumerate(harmonics, start=2):
        harmonic_dict[f"HD{i}"]      = float(hm)
        harmonic_dict[f"HD{i}_freq"] = float(hf)

    # Non-harmonic spurious: strongest bin excluding fundamental and harmonics
    exclude_freqs = np.array([fund_freq] + [h[0] for h in harmonics])
    non_harmonic_mask = ~np.any(
        np.abs(freqs[np.newaxis, :] - exclude_freqs[:, np.newaxis]) <= bin_width,
        axis=0,
    )
    spurious = float(np.max(mags[non_harmonic_mask])) if non_harmonic_mask.any() else float(fund_mag)

    # NSD: noise power normalised to 1 Hz bandwidth
    n_noise_bins = int(np.sum(mask))
    if n_noise_bins > 0 and bin_width > 0:
        nsd_dBHz = 10 * np.log10(max(noise_pwr, 1e-20) / n_noise_bins) - 10 * np.log10(bin_width)
    else:
        nsd_dBHz = float('-inf')

    results = {
        "SNR": snr,
        "SNDR": sndr,
        "SFDR": sfdr,
        "THD": thd,
        "NSD_dBHz": nsd_dBHz,
        "NoiseFloor": noise_floor,
        "Spurious": spurious,
        "ENOB": enob,
        "Offset": offset,
        "FundamentalFrequency": fund_freq,
        "FundamentalMagnitude": fund_mag,
        "HarmonicFreqs": [h[0] for h in harmonics],
        "HarmonicMags":  [h[1] for h in harmonics],
        **harmonic_dict,
    }
```

Note: the `mask` variable used for noise power (excluding fundamental and harmonics) is already computed earlier in the function. The `non_harmonic_mask` for `Spurious` is a separate mask that excludes harmonics but still includes noise bins — that variable already exists in the original code as `mask`.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_metrics.py::test_dynamic_metrics_individual_harmonics tests/test_metrics.py::test_dynamic_metrics_nsd tests/test_metrics.py::test_dynamic_metrics_spurious -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add pyDataconverter/utils/metrics/_dynamic.py tests/test_metrics.py
git commit -m "feat(metrics): add HD2..HDn, NSD_dBHz, Spurious to dynamic metrics output"
```

---

## Task 4: Add `calculate_adc_iip3`

**Files:**
- Modify: `pyDataconverter/utils/metrics/adc.py`
- Modify: `pyDataconverter/utils/metrics/__init__.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_calculate_adc_iip3_ideal():
    """IIP3 for an ideal ADC is very high (no IM3 products)."""
    from pyDataconverter.architectures.FlashADC import FlashADC
    from pyDataconverter.utils.signal_gen import generate_two_tone
    from pyDataconverter.utils.metrics import calculate_adc_iip3
    import numpy as np
    fs = 10e6
    adc = FlashADC(n_bits=8, v_ref=1.0, offset_std=0.0)
    f1, f2 = 1e6, 1.1e6
    vin = generate_two_tone(f1, f2, fs, amplitude1=0.2, amplitude2=0.2,
                            duration=512/fs) + 0.5
    codes = np.array([adc.convert(float(v)) for v in vin])
    m = calculate_adc_iip3(codes.astype(float), fs, f1, f2)
    assert 'IIP3_dB' in m
    assert 'OIP3_dB' in m
    assert 'IM3_dB' in m
    assert np.isfinite(m['IIP3_dB'])

def test_calculate_adc_iip3_nonlinear():
    """Nonlinear ADC has finite, lower IIP3 than ideal."""
    from pyDataconverter.architectures.FlashADC import FlashADC
    from pyDataconverter.utils.signal_gen import generate_two_tone
    from pyDataconverter.utils.metrics import calculate_adc_iip3
    import numpy as np
    fs = 10e6
    # Higher offset_std means more nonlinearity → lower IIP3
    adc_ideal = FlashADC(n_bits=6, v_ref=1.0, offset_std=0.0)
    adc_noisy = FlashADC(n_bits=6, v_ref=1.0, offset_std=0.05)
    f1, f2 = 1e6, 1.1e6
    vin = generate_two_tone(f1, f2, fs, amplitude1=0.15, amplitude2=0.15,
                            duration=512/fs) + 0.5
    codes_ideal = np.array([adc_ideal.convert(float(v)) for v in vin])
    codes_noisy = np.array([adc_noisy.convert(float(v)) for v in vin])
    m_ideal = calculate_adc_iip3(codes_ideal.astype(float), fs, f1, f2)
    m_noisy = calculate_adc_iip3(codes_noisy.astype(float), fs, f1, f2)
    assert m_ideal['IIP3_dB'] > m_noisy['IIP3_dB']
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metrics.py::test_calculate_adc_iip3_ideal tests/test_metrics.py::test_calculate_adc_iip3_nonlinear -v
```

Expected: ImportError for `calculate_adc_iip3`.

- [ ] **Step 3: Implement `calculate_adc_iip3` in `adc.py`**

```python
def calculate_adc_iip3(
        time_data: np.ndarray,
        fs: float,
        f1: float,
        f2: float,
        full_scale: float = None,
) -> Dict[str, float]:
    """
    Calculate IIP3 and OIP3 from a two-tone ADC output.

    Uses the standard IIP3 formula:
        IIP3 = P_in + (P_in - P_im3) / 2

    where P_in is the average power of the two tones and P_im3 is the
    average power of the two third-order IM products (2f1-f2, 2f2-f1).
    All powers in dB (or dBFS if full_scale provided).

    Args:
        time_data: ADC output codes (or voltages) as a 1-D float array.
        fs: Sampling rate (Hz).
        f1: First tone frequency (Hz).
        f2: Second tone frequency (Hz).
        full_scale: Full-scale value for dBFS (optional).

    Returns:
        Dict with keys:
            IIP3_dB  : Input-referred third-order intercept point (dB).
            OIP3_dB  : Output-referred third-order intercept point (dB).
            IM3_dB   : Average IM3 product level relative to input tones (dBc).
            P_in_dB  : Average input tone power (dB).
            P_im3_dB : Average IM3 product power (dB).
    """
    freqs, mags = compute_fft(time_data, fs)
    bin_width = freqs[1] - freqs[0]

    def _peak_mag(f_target):
        idx = np.argmin(np.abs(freqs - f_target))
        # Take the max in a ±2-bin window to handle small frequency errors
        lo = max(0, idx - 2)
        hi = min(len(mags) - 1, idx + 2)
        return float(np.max(mags[lo:hi + 1]))

    p1 = _peak_mag(f1)
    p2 = _peak_mag(f2)
    p_in = (p1 + p2) / 2.0

    f_im3_low  = abs(2 * f1 - f2)
    f_im3_high = abs(2 * f2 - f1)
    im3_low  = _peak_mag(f_im3_low)
    im3_high = _peak_mag(f_im3_high)
    p_im3 = (im3_low + im3_high) / 2.0

    iip3 = p_in + (p_in - p_im3) / 2.0
    oip3 = iip3 + p_in

    return {
        'IIP3_dB':  iip3,
        'OIP3_dB':  oip3,
        'IM3_dB':   p_im3 - p_in,
        'P_in_dB':  p_in,
        'P_im3_dB': p_im3,
    }
```

- [ ] **Step 4: Re-export from `__init__.py`**

In `pyDataconverter/utils/metrics/__init__.py`, add `calculate_adc_iip3` to the import and `__all__`:

```python
from .adc import (
    calculate_adc_dynamic_metrics,
    calculate_adc_static_metrics,
    calculate_adc_static_metrics_histogram,
    calculate_gain_offset_error,
    calculate_adc_iip3,
    is_monotonic,
    calculate_histogram,
)
```

And add `"calculate_gain_offset_error"` and `"calculate_adc_iip3"` to `__all__`.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_metrics.py::test_calculate_adc_iip3_ideal tests/test_metrics.py::test_calculate_adc_iip3_nonlinear -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyDataconverter/utils/metrics/adc.py pyDataconverter/utils/metrics/__init__.py tests/test_metrics.py
git commit -m "feat(metrics): add calculate_adc_iip3 (IIP3/OIP3 from two-tone test)"
```

---

## Task 5: Add `calculate_dynamic_range_from_curve` and `calculate_erbw_from_curve`

**Files:**
- Modify: `pyDataconverter/utils/metrics/adc.py`
- Modify: `pyDataconverter/utils/metrics/__init__.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_dynamic_range_from_curve_basic():
    """DR is the span from full scale to the amplitude where SNR = 0 dB."""
    from pyDataconverter.utils.metrics import calculate_dynamic_range_from_curve
    import numpy as np
    # SNR rises linearly with amplitude (in dB, slope ~1 dB/dB)
    amplitudes_db = np.linspace(-60, 0, 61)
    snr_values    = amplitudes_db + 50   # SNR = 0 at -50 dBFS
    m = calculate_dynamic_range_from_curve(amplitudes_db, snr_values)
    assert 'DR_dB' in m
    # DR should be ~50 dB
    assert abs(m['DR_dB'] - 50.0) < 1.0

def test_erbw_from_curve_basic():
    """ERBW is the frequency where ENOB drops by 0.5 bits below reference."""
    from pyDataconverter.utils.metrics import calculate_erbw_from_curve
    import numpy as np
    freqs = np.array([1e3, 10e3, 100e3, 1e6, 10e6])
    # ENOB drops from 8.0 at low freq to below 7.5 at 1 MHz
    enob  = np.array([8.0, 8.0, 7.8, 7.4, 6.9])
    m = calculate_erbw_from_curve(freqs, enob)
    assert 'ERBW_Hz' in m
    assert 100e3 < m['ERBW_Hz'] < 1e6

def test_erbw_custom_reference():
    """Custom enob_ref shifts the -0.5 bit threshold."""
    from pyDataconverter.utils.metrics import calculate_erbw_from_curve
    import numpy as np
    freqs = np.linspace(1e3, 10e6, 50)
    enob  = 8.0 - np.linspace(0, 3.0, 50)
    m_default = calculate_erbw_from_curve(freqs, enob)
    m_custom  = calculate_erbw_from_curve(freqs, enob, enob_ref=7.0)
    # Custom ref is lower → ERBW at custom ref is higher frequency
    assert m_custom['ERBW_Hz'] > m_default['ERBW_Hz']
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metrics.py::test_dynamic_range_from_curve_basic tests/test_metrics.py::test_erbw_from_curve_basic tests/test_metrics.py::test_erbw_custom_reference -v
```

Expected: ImportError.

- [ ] **Step 3: Implement both functions in `adc.py`**

Add after `calculate_adc_iip3`:

```python
def calculate_dynamic_range_from_curve(
        amplitudes_db: np.ndarray,
        snr_values: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate dynamic range from a measured SNR vs amplitude curve.

    Dynamic range is defined as the ratio (in dB) between the full-scale
    amplitude and the amplitude at which SNR = 0 dB.

    Args:
        amplitudes_db: Input amplitude sweep in dBFS (or dB), sorted
            ascending.  The largest value is taken as full scale.
        snr_values: SNR (dB) at each amplitude.  Same length as
            amplitudes_db.

    Returns:
        Dict with keys:
            DR_dB              : Dynamic range in dB.
            AmplitudeAtSNR0_dB : Amplitude (same units as amplitudes_db)
                                 where SNR interpolates to 0 dB.
    """
    from scipy.interpolate import interp1d
    amplitudes_db = np.asarray(amplitudes_db, dtype=float)
    snr_values    = np.asarray(snr_values,    dtype=float)

    idx = np.argsort(amplitudes_db)
    amp = amplitudes_db[idx]
    snr = snr_values[idx]

    # Interpolate amplitude as a function of SNR to find SNR=0 crossing
    f = interp1d(snr, amp, kind='linear', bounds_error=False,
                 fill_value=(amp[0], amp[-1]))
    amp_at_snr0 = float(f(0.0))
    dr = float(amp[-1] - amp_at_snr0)

    return {
        'DR_dB':              dr,
        'AmplitudeAtSNR0_dB': amp_at_snr0,
    }


def calculate_erbw_from_curve(
        frequencies: np.ndarray,
        enob_values: np.ndarray,
        enob_ref: float = None,
) -> Dict[str, float]:
    """
    Estimate effective resolution bandwidth (ERBW) from ENOB vs frequency.

    ERBW is the frequency at which ENOB degrades by 0.5 bits below the
    low-frequency reference ENOB.

    Args:
        frequencies: Frequency array (Hz), sorted ascending.
        enob_values: ENOB at each frequency.  Same length as frequencies.
        enob_ref: Reference ENOB.  Defaults to enob_values[0].

    Returns:
        Dict with keys:
            ERBW_Hz  : Effective resolution bandwidth in Hz.
            ENOB_ref : Reference ENOB used for the -0.5 bit threshold.
    """
    from scipy.interpolate import interp1d
    frequencies = np.asarray(frequencies, dtype=float)
    enob_values = np.asarray(enob_values, dtype=float)

    if enob_ref is None:
        enob_ref = float(enob_values[0])

    target = enob_ref - 0.5

    # Interpolate frequency as a function of ENOB (ENOB is decreasing)
    # Flip so we interpolate on a monotone increasing sequence of ENOB
    f = interp1d(enob_values[::-1], frequencies[::-1], kind='linear',
                 bounds_error=False, fill_value=(frequencies[0], frequencies[-1]))
    erbw = float(f(target))

    return {
        'ERBW_Hz':  erbw,
        'ENOB_ref': enob_ref,
    }
```

- [ ] **Step 4: Re-export from `__init__.py`**

Add `calculate_dynamic_range_from_curve` and `calculate_erbw_from_curve` to the import block and `__all__` in `pyDataconverter/utils/metrics/__init__.py`.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_metrics.py::test_dynamic_range_from_curve_basic tests/test_metrics.py::test_erbw_from_curve_basic tests/test_metrics.py::test_erbw_custom_reference -v
```

Expected: 3 passed.

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pyDataconverter/utils/metrics/adc.py pyDataconverter/utils/metrics/__init__.py tests/test_metrics.py
git commit -m "feat(metrics): add calculate_dynamic_range_from_curve and calculate_erbw_from_curve"
```
