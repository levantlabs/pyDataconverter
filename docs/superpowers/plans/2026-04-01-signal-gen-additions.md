# Signal Generation Additions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `generate_chirp`, `generate_prbs`, `apply_channel`, `generate_gaussian_noise`, and `apply_window` to `pyDataconverter/utils/signal_gen.py`.

**Architecture:** All additions are pure functions in the existing `signal_gen.py` file. No new files needed. `apply_window` wraps `scipy.signal.windows` to support the same window names already used in `fft_analysis.py`.

**Tech Stack:** numpy, scipy.signal.

---

## File Map

| File | Change |
|---|---|
| `pyDataconverter/utils/signal_gen.py` | Add 5 new functions |
| `tests/test_signal_gen.py` | Add tests for each new function |

---

## Task 1: Add `generate_chirp`

**Files:**
- Modify: `pyDataconverter/utils/signal_gen.py`
- Modify: `tests/test_signal_gen.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_signal_gen.py`:

```python
from pyDataconverter.utils.signal_gen import generate_chirp
import numpy as np

def test_chirp_length():
    """Output length matches sampling_rate * duration."""
    sig = generate_chirp(f_start=1e3, f_stop=100e3,
                         sampling_rate=1e6, duration=0.01)
    assert len(sig) == int(1e6 * 0.01)

def test_chirp_amplitude():
    """Peak amplitude matches the amplitude parameter."""
    amp = 0.4
    sig = generate_chirp(f_start=1e3, f_stop=50e3,
                         sampling_rate=1e6, amplitude=amp, duration=0.005)
    assert abs(np.max(np.abs(sig)) - amp) < 0.01

def test_chirp_offset():
    """DC offset shifts mean by the offset parameter."""
    offset = 0.5
    sig = generate_chirp(f_start=1e3, f_stop=50e3, sampling_rate=1e6,
                         amplitude=0.1, offset=offset, duration=0.005)
    assert abs(np.mean(sig) - offset) < 0.05

def test_chirp_start_end_freqs():
    """Instantaneous frequency starts at f_start and ends near f_stop."""
    fs = 1e6
    duration = 0.01
    f_start, f_stop = 1e3, 100e3
    sig = generate_chirp(f_start, f_stop, fs, duration=duration)
    # Check start frequency via zero-crossing rate over first 1 ms
    n_start = int(fs * 0.001)
    crossings = np.where(np.diff(np.sign(sig[:n_start])))[0]
    if len(crossings) > 1:
        f_est = fs / (2 * np.mean(np.diff(crossings)))
        assert 0.5 * f_start < f_est < 2 * f_start

def test_chirp_default_amplitude():
    """Default amplitude is 1.0."""
    sig = generate_chirp(1e3, 10e3, 1e6, duration=0.001)
    assert abs(np.max(np.abs(sig)) - 1.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lazarus/python-dev/pyDataconverter
python -m pytest tests/test_signal_gen.py::test_chirp_length tests/test_signal_gen.py::test_chirp_amplitude tests/test_signal_gen.py::test_chirp_offset tests/test_signal_gen.py::test_chirp_start_end_freqs tests/test_signal_gen.py::test_chirp_default_amplitude -v
```

Expected: ImportError for `generate_chirp`.

- [ ] **Step 3: Implement `generate_chirp`**

Add to `pyDataconverter/utils/signal_gen.py` (after `generate_sine`):

```python
def generate_chirp(f_start: float,
                   f_stop: float,
                   sampling_rate: float,
                   amplitude: float = 1.0,
                   offset: float = 0.0,
                   duration: float = 1.0,
                   method: str = 'linear') -> np.ndarray:
    """
    Generate a swept-frequency (chirp) signal.

    Args:
        f_start: Start frequency in Hz.
        f_stop: Stop frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        amplitude: Peak amplitude in volts (default 1.0).
        offset: DC offset in volts (default 0.0).
        duration: Signal duration in seconds (default 1.0).
        method: Frequency sweep method — 'linear' (default) or 'logarithmic'.

    Returns:
        Signal array of length int(sampling_rate * duration).
    """
    from scipy.signal import chirp as scipy_chirp
    t = np.arange(0, duration, 1 / sampling_rate)
    sig = scipy_chirp(t, f0=f_start, f1=f_stop, t1=duration, method=method)
    return amplitude * sig + offset
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_signal_gen.py::test_chirp_length tests/test_signal_gen.py::test_chirp_amplitude tests/test_signal_gen.py::test_chirp_offset tests/test_signal_gen.py::test_chirp_start_end_freqs tests/test_signal_gen.py::test_chirp_default_amplitude -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/utils/signal_gen.py tests/test_signal_gen.py
git commit -m "feat(signal_gen): add generate_chirp"
```

---

## Task 2: Add `generate_prbs` and `apply_channel`

**Files:**
- Modify: `pyDataconverter/utils/signal_gen.py`
- Modify: `tests/test_signal_gen.py`

- [ ] **Step 1: Write the failing tests**

```python
from pyDataconverter.utils.signal_gen import generate_prbs, apply_channel

def test_prbs_length():
    """Output has exactly n_samples samples."""
    sig = generate_prbs(order=7, n_samples=1000)
    assert len(sig) == 1000

def test_prbs_binary_values():
    """Default PRBS has only +amplitude and -amplitude values."""
    amp = 0.5
    sig = generate_prbs(order=7, n_samples=500, amplitude=amp)
    unique = np.unique(sig)
    assert set(unique).issubset({-amp, amp})

def test_prbs_with_offset():
    """Offset shifts all values."""
    sig = generate_prbs(order=7, n_samples=500, amplitude=0.5, offset=1.0)
    assert np.all(sig >= 0.4)
    assert np.all(sig <= 1.6)

def test_prbs_flat_spectrum():
    """PRBS has a roughly flat power spectrum (no dominant tone)."""
    sig = generate_prbs(order=10, n_samples=2**10 - 1, amplitude=1.0)
    fft_mag = np.abs(np.fft.rfft(sig - np.mean(sig)))
    # No single bin should dominate (no tone > 5x the mean)
    assert np.max(fft_mag[1:]) < 5 * np.mean(fft_mag[1:])

def test_prbs_reproducible_with_seed():
    """Same seed gives the same sequence."""
    s1 = generate_prbs(order=7, n_samples=200, seed=42)
    s2 = generate_prbs(order=7, n_samples=200, seed=42)
    np.testing.assert_array_equal(s1, s2)

def test_apply_channel_length():
    """Output length matches input length (same-length convolution)."""
    sig = generate_prbs(order=7, n_samples=512)
    h = np.array([1.0, -0.5, 0.25])
    out = apply_channel(sig, h)
    assert len(out) == len(sig)

def test_apply_channel_identity():
    """Convolving with [1] returns the original signal."""
    sig = generate_prbs(order=7, n_samples=200)
    out = apply_channel(sig, np.array([1.0]))
    np.testing.assert_allclose(out, sig)

def test_apply_channel_lowpass():
    """A lowpass FIR reduces high-frequency content."""
    from pyDataconverter.utils.signal_gen import generate_chirp
    sig = generate_chirp(1e3, 100e3, 1e6, duration=0.01)
    # Simple 3-tap moving average = lowpass
    h = np.ones(3) / 3.0
    out = apply_channel(sig, h)
    assert len(out) == len(sig)
    # High-frequency portion should have lower variance after filtering
    assert np.std(out[-100:]) <= np.std(sig[-100:]) + 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_signal_gen.py::test_prbs_length tests/test_signal_gen.py::test_prbs_binary_values tests/test_signal_gen.py::test_prbs_flat_spectrum tests/test_signal_gen.py::test_apply_channel_length tests/test_signal_gen.py::test_apply_channel_identity -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `generate_prbs` and `apply_channel`**

Add to `pyDataconverter/utils/signal_gen.py`:

```python
def generate_prbs(order: int,
                  n_samples: int,
                  amplitude: float = 1.0,
                  offset: float = 0.0,
                  seed: int = None) -> np.ndarray:
    """
    Generate a Pseudo-Random Binary Sequence (PRBS).

    Uses a maximal-length linear feedback shift register (LFSR) to produce
    a balanced ±amplitude binary sequence with a flat power spectrum.

    Args:
        order: LFSR order (2–20).  The full period length is 2^order − 1.
        n_samples: Number of output samples.  The PRBS is tiled or truncated
            to this length.
        amplitude: Half-range of the output (default 1.0).  Output values
            are +amplitude and −amplitude.
        offset: DC offset in volts (default 0.0).
        seed: Optional integer seed for the initial LFSR state.  If None a
            fixed all-ones state is used, giving a deterministic sequence.

    Returns:
        Signal array of length n_samples with values {offset±amplitude}.
    """
    # Standard maximal-length LFSR taps (Fibonacci form)
    _TAPS = {
        2: [2, 1], 3: [3, 2], 4: [4, 3], 5: [5, 3], 6: [6, 5],
        7: [7, 6], 8: [8, 6, 5, 4], 9: [9, 5], 10: [10, 7],
        11: [11, 9], 12: [12, 11, 10, 4], 13: [13, 12, 11, 8],
        14: [14, 13, 12, 2], 15: [15, 14], 16: [16, 15, 13, 4],
        17: [17, 14], 18: [18, 11], 19: [19, 18, 17, 14], 20: [20, 17],
    }
    if order not in _TAPS:
        raise ValueError(f"order must be between 2 and 20, got {order}")

    taps = _TAPS[order]
    period = 2 ** order - 1

    rng = np.random.default_rng(seed)
    state = int(rng.integers(1, 2**order)) if seed is not None else (2**order - 1)

    bits = np.zeros(period, dtype=np.int8)
    for i in range(period):
        bits[i] = state & 1
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (tap - 1)) & 1
        state = ((state >> 1) | (feedback << (order - 1))) & ((1 << order) - 1)

    # Tile to n_samples
    reps  = (n_samples + period - 1) // period
    tiled = np.tile(bits, reps)[:n_samples]

    # Map 0/1 → -amplitude/+amplitude
    return tiled.astype(float) * 2 * amplitude - amplitude + offset


def apply_channel(signal: np.ndarray,
                  h: np.ndarray) -> np.ndarray:
    """
    Apply a channel impulse response to a signal (linear convolution).

    Output is truncated to the same length as the input so the return
    array can be fed directly to an ADC.

    Args:
        signal: Input signal array.
        h: Channel impulse response (FIR filter coefficients).

    Returns:
        Signal array of the same length as ``signal``.
    """
    from scipy.signal import fftconvolve
    out = fftconvolve(signal, h, mode='full')
    return out[:len(signal)]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_signal_gen.py::test_prbs_length tests/test_signal_gen.py::test_prbs_binary_values tests/test_signal_gen.py::test_prbs_with_offset tests/test_signal_gen.py::test_prbs_flat_spectrum tests/test_signal_gen.py::test_prbs_reproducible_with_seed tests/test_signal_gen.py::test_apply_channel_length tests/test_signal_gen.py::test_apply_channel_identity tests/test_signal_gen.py::test_apply_channel_lowpass -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add pyDataconverter/utils/signal_gen.py tests/test_signal_gen.py
git commit -m "feat(signal_gen): add generate_prbs and apply_channel"
```

---

## Task 3: Add `generate_gaussian_noise` and `apply_window`

**Files:**
- Modify: `pyDataconverter/utils/signal_gen.py`
- Modify: `tests/test_signal_gen.py`

- [ ] **Step 1: Write the failing tests**

```python
from pyDataconverter.utils.signal_gen import generate_gaussian_noise, apply_window

def test_gaussian_noise_length():
    sig = generate_gaussian_noise(n_samples=1000)
    assert len(sig) == 1000

def test_gaussian_noise_statistics():
    """Zero mean, unit std by default."""
    rng = np.random.default_rng(0)
    sig = generate_gaussian_noise(n_samples=10000, std=1.0, rng=rng)
    assert abs(np.mean(sig)) < 0.05
    assert abs(np.std(sig) - 1.0) < 0.05

def test_gaussian_noise_std_param():
    rng = np.random.default_rng(1)
    std = 0.3
    sig = generate_gaussian_noise(n_samples=5000, std=std, rng=rng)
    assert abs(np.std(sig) - std) < 0.02

def test_gaussian_noise_offset():
    rng = np.random.default_rng(2)
    sig = generate_gaussian_noise(n_samples=5000, std=0.1, offset=2.5, rng=rng)
    assert abs(np.mean(sig) - 2.5) < 0.02

def test_gaussian_noise_reproducible():
    s1 = generate_gaussian_noise(100, std=1.0, rng=np.random.default_rng(7))
    s2 = generate_gaussian_noise(100, std=1.0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(s1, s2)

def test_apply_window_length():
    sig = np.ones(256)
    out = apply_window(sig, 'hann')
    assert len(out) == 256

def test_apply_window_hann_ends_zero():
    """Hann window tapers to zero at both ends."""
    sig = np.ones(256)
    out = apply_window(sig, 'hann')
    assert abs(out[0]) < 0.01
    assert abs(out[-1]) < 0.01

def test_apply_window_invalid_raises():
    with pytest.raises(ValueError, match="Unknown window"):
        apply_window(np.ones(64), 'not_a_window')

def test_apply_window_blackman():
    sig = np.ones(128)
    out = apply_window(sig, 'blackman')
    assert len(out) == 128
    assert abs(out[0]) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_signal_gen.py::test_gaussian_noise_length tests/test_signal_gen.py::test_gaussian_noise_statistics tests/test_signal_gen.py::test_apply_window_length tests/test_signal_gen.py::test_apply_window_hann_ends_zero tests/test_signal_gen.py::test_apply_window_invalid_raises -v
```

Expected: ImportError.

- [ ] **Step 3: Implement both functions**

Add to `pyDataconverter/utils/signal_gen.py`:

```python
def generate_gaussian_noise(n_samples: int,
                             std: float = 1.0,
                             offset: float = 0.0,
                             rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate Gaussian white noise.

    Args:
        n_samples: Number of samples.
        std: Standard deviation in volts (default 1.0).
        offset: DC offset in volts (default 0.0).
        rng: Optional numpy Generator for reproducibility.  If None, uses
             the global numpy random state.

    Returns:
        Signal array of length n_samples.
    """
    if rng is None:
        return np.random.normal(loc=offset, scale=std, size=n_samples)
    return rng.normal(loc=offset, scale=std, size=n_samples)


def apply_window(signal: np.ndarray, window_type: str) -> np.ndarray:
    """
    Apply a window function to a signal.

    Supports the same window names used by ``compute_fft``:
    'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'flattop',
    'boxcar', 'tukey', 'cosine', 'exponential'.

    Args:
        signal: Input signal array.
        window_type: Name of the window (case-sensitive, scipy.signal.windows).

    Returns:
        Windowed signal array of the same length as ``signal``.

    Raises:
        ValueError: If window_type is not a recognised window name.
    """
    _ALLOWED_WINDOWS = {
        'hann', 'hamming', 'blackman', 'bartlett', 'kaiser',
        'flattop', 'boxcar', 'tukey', 'cosine', 'exponential',
    }
    if window_type not in _ALLOWED_WINDOWS:
        raise ValueError(
            f"Unknown window '{window_type}'. "
            f"Allowed: {sorted(_ALLOWED_WINDOWS)}")
    from scipy.signal import windows as sp_windows
    w = getattr(sp_windows, window_type)(len(signal))
    return signal * w
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_signal_gen.py::test_gaussian_noise_length tests/test_signal_gen.py::test_gaussian_noise_statistics tests/test_signal_gen.py::test_gaussian_noise_std_param tests/test_signal_gen.py::test_gaussian_noise_offset tests/test_signal_gen.py::test_gaussian_noise_reproducible tests/test_signal_gen.py::test_apply_window_length tests/test_signal_gen.py::test_apply_window_hann_ends_zero tests/test_signal_gen.py::test_apply_window_invalid_raises tests/test_signal_gen.py::test_apply_window_blackman -v
```

Expected: 9 passed.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyDataconverter/utils/signal_gen.py tests/test_signal_gen.py
git commit -m "feat(signal_gen): add generate_gaussian_noise and apply_window"
```
