# Time-Interleaved ADC (TI-ADC) — Phase 1 Design

**Date:** 2026-04-14
**Status:** Design complete, pending user review. Implementation has not started.
**Scope:** Phase 1 of pyDataconverter's time-interleaved ADC support. Follows the Pipelined ADC Phase 1 work (merged 2026-04-13) which established the `PipelinedADC`/`PipelineStage`/`ResidueAmplifier` composition pattern that TI-ADC extends.

---

## 1. Context and motivation

pyDataconverter currently implements six ADC families: `SimpleADC`, `FlashADC`, `SARADC` (with `MultibitSARADC` and `NoiseshapingSARADC` variants), and `PipelinedADC` (with `PipelineStage`). Time-interleaved ADCs — the only way to reach multi-GS/s sampling rates in CMOS, and the dominant architecture at the high-speed end of the market — are missing. This design adds a composable `TimeInterleavedADC(ADCBase)` class that:

1. **Wraps any ADCBase as a channel template.** Every channel in a given TI-ADC is a deep copy of the same sub-ADC; per-channel mismatch is injected at call time rather than baked into each sub-ADC instance. This keeps the class's constructor surface small and matches how `FlashADC` already uses `comparator_type` to replicate its comparator bank.
2. **Is itself an `ADCBase`**, so it can be used as a sub-ADC inside another architecture — for example, as the backend of a `PipelinedADC`, as a sub-ADC inside a `PipelineStage`, or as a sub-ADC inside another `TimeInterleavedADC` for hierarchical interleaving.
3. **Supports hierarchical interleaving** via multiple sampling stages with different interleaving factors — physically how real multi-GS/s TI-ADCs are built. A classmethod helper builds the tree from a list of per-level interleaving factors; manual nesting via the flat constructor is also supported.
4. **Models the four inter-channel mismatches** that matter most in practice: offset, gain error, timing skew, and bandwidth. Each generates a predictable spectral signature that textbook formulas describe; analytical spur validation is the Phase 1 correctness criterion (TI-ADC has no external bit-exact reference oracle to match against, unlike the adc_book reference for pipelined ADC).
5. **Ships with a new `ADCBase.convert_waveform(v_dense, t_dense)` method** (default implementation on the base class) that takes a dense waveform and extracts output codes with numerically derived `dvdt`. Every existing ADC inherits it without modification. TI-ADC overrides it to apply per-channel `scipy.signal.lfilter` LPFs for bandwidth mismatch — the one mismatch type that cannot be expressed via pointwise `convert(v, dvdt=...)`.

The deliverable is a single class plus tests, a small extension to `ADCBase`, one example script, and documentation. No cross-cutting refactors.

---

## 2. Architecture overview

### 2.1 Class shape

```
                ┌────────────────────────────────────────────────────────────┐
  v_in  ───────▶│                TimeInterleavedADC(ADCBase)                 │
  dvdt  ───────▶│                                                            │
                │  channel_counter ──┐                                        │
                │                    ▼                                        │
                │    ┌────────────────┐    ┌────────────────┐                 │
                │    │ channel 0      │    │ channel 1      │  ... M copies   │
                │    │ offset[0]      │    │ offset[1]      │                 │
                │    │ gain_error[0]  │    │ gain_error[1]  │                 │
                │    │ timing_skew[0] │    │ timing_skew[1] │                 │
                │    │ bandwidth[0]   │    │ bandwidth[1]   │                 │
                │    │ sub_adc (copy) │    │ sub_adc (copy) │                 │
                │    └────────────────┘    └────────────────┘                 │
                │                                                             │
                └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                                  int output code
```

A `TimeInterleavedADC` holds `M` channels (`M ≥ 2`). Each channel stores:

- A deep-copied `sub_adc` instance (the template, copied at construction).
- Per-channel `offset` (V, input-referred), `gain_error` (fractional, dimensionless), `timing_skew` (seconds), and `bandwidth` (Hz, first-order LPF cutoff).

The four mismatch arrays are stored at the `TimeInterleavedADC` level (not on each sub-ADC), so the sub-ADCs themselves remain identical — the template pattern is preserved.

### 2.2 Stateful channel counter

The class maintains a single integer `_counter` that advances by one per `convert()` call. Each call routes to `channels[counter % M]`. `reset()` restores `_counter = 0`. `convert_waveform(v_dense, t_dense)` advances the counter by `len(v_dense)` at the end of the call, so back-to-back invocations pick up where the previous call left off. A read-only property `last_channel` exposes `(counter − 1) % M` after the most recent `convert()` for debugging and per-channel characterization.

### 2.3 Call-path asymmetry (bandwidth only)

There are two call paths, `convert()` and `convert_waveform()`. Offset, gain, and timing skew work via both paths — they are all pointwise first-order operations that compose naturally with the existing `convert(v, dvdt=...)` API. Bandwidth mismatch requires the full dense waveform (it is a convolution with a per-channel LPF impulse response, which cannot be expressed pointwise) and is only available via `convert_waveform()`. If a user constructs a `TimeInterleavedADC` with any nonzero bandwidth and calls the pointwise `convert()`, a `RuntimeError` explains the asymmetry and directs them to `convert_waveform()`. This is a deliberate tradeoff: rather than silently produce wrong results, the class fails loudly at the API boundary.

### 2.4 Hierarchical construction

Since `TimeInterleavedADC` inherits from `ADCBase`, a `TimeInterleavedADC` instance can be passed as another `TimeInterleavedADC`'s `sub_adc_template`, giving multi-level interleaving trees. Two user-facing shapes:

**Flat constructor** (simple case, manually nested for hierarchy):

```python
inner = TimeInterleavedADC(channels=2, sub_adc_template=FlashADC(n_bits=10, v_ref=1.0))
outer = TimeInterleavedADC(channels=4, sub_adc_template=inner)
# 4 × 2 = 8 total channels
```

**Classmethod helper** (explicit, takes a level list):

```python
ti = TimeInterleavedADC.hierarchical(
    channels_per_level=[4, 2],
    sub_adc_template=FlashADC(n_bits=10, v_ref=1.0),
    offset_std_per_level=[1e-3, 0.5e-3],
    gain_std_per_level=[0.001, 0.0005],
    fs=8e9,
)
# Same tree as above, plus mismatches applied per level:
# outer level σ_offset=1 mV, inner level σ_offset=0.5 mV; etc.
```

Both produce `TimeInterleavedADC` instances with identical runtime behaviour. The classmethod is sugar over the flat constructor. It is documented in the class docstring alongside a manual-nesting example so users know they have both options.

### 2.5 Composability as a sub-component

A `TimeInterleavedADC` can be passed as `PipelineStage.sub_adc`, `PipelinedADC.backend`, or any other slot that expects an `ADCBase`. Its `convert(v, dvdt=...)` call is the only contract that matters at the boundary. The class is thus drop-in wherever pyDataconverter's existing ADC classes are accepted — a user building a pipelined ADC with a TI-ADC backend writes exactly the same `PipelinedADC(..., backend=TimeInterleavedADC(...))` call they would write for any other backend.

### 2.6 Output semantics

`convert()` returns one `int` per call: the raw code from the currently active channel, passthrough, no digital post-processing. The sub-ADC template determines `n_bits`, `v_ref`, and `input_type`; these are inherited by the `TimeInterleavedADC` and must not be overridden at construction. `convert_waveform()` returns a 1-D `np.ndarray[int]` of length `len(v_dense)`. The output stream is the interleaved sequence of raw channel codes — exactly what a real TI-ADC produces at its digital output, including all mismatch artefacts.

A `split_by_channel(codes)` helper method reshapes a 1-D code array into a 2-D array of shape `(M, len(codes) // M)` so callers who want per-channel analysis (e.g., plotting channel 0's transfer function vs. channel 1's) have a first-class accessor. The helper does not maintain state — it operates on whatever array the caller passes in.

---

## 3. Components

### 3.1 NEW — files created

| Path | Role |
|---|---|
| `pyDataconverter/architectures/TimeInterleavedADC.py` | `TimeInterleavedADC(ADCBase)` class, `hierarchical` classmethod, `split_by_channel` method, `last_channel` property. Overrides `convert_waveform` to apply per-channel `scipy.signal.lfilter` LPFs when bandwidth mismatch is active. |
| `tests/test_ti_adc.py` | Construction validation, channel rotation, `split_by_channel`, hierarchical construction, composability with `PipelinedADC`, and the default `ADCBase.convert_waveform` path exercised on non-TI subclasses. |
| `tests/test_ti_adc_spurs.py` | Analytical spur validation: offset, gain, timing-skew, and bandwidth mismatch each generate closed-form spectral signatures; tests assert the expected spurs appear with expected magnitudes (±2 dB tolerance, ±3 dB for bandwidth). |
| `examples/ti_adc_example.py` | 4-channel 10-bit TI-ADC driven by a coherent sine, plots the output spectrum showing mismatch spurs. Follows the style of `examples/pipelined_adc_example.py`. |

### 3.2 EXTENDED — surgical additions to existing files

All changes are strictly additive. Defaults preserve existing behaviour.

| Path | Change |
|---|---|
| `pyDataconverter/dataconverter.py` | Add `convert_waveform(v_dense, t_dense)` method to `ADCBase`. Default implementation: `dvdt_dense = np.gradient(v_dense, t_dense)`; loop over samples calling `self.convert(v_dense[i], dvdt=dvdt_dense[i])`; return the code array. Every `ADCBase` subclass inherits it. TI-ADC overrides it to handle bandwidth mismatch. Other subclasses use the default unchanged. |
| `pyDataconverter/architectures/__init__.py` | Export `TimeInterleavedADC`. |
| `docs/api_reference.md` | New section for `TimeInterleavedADC` with constructor parameters, hierarchical classmethod, `split_by_channel`, `last_channel`, and the four mismatch parameters. Brief mention of `ADCBase.convert_waveform` in the base class entry. |
| `todo/adc_architectures.md` | Flip TI-ADC entry from `IN DESIGN (2026-04-13)` to `PHASE 1 IMPLEMENTED (2026-04-14)`. Note Phase 2 items (second-order skew correction, digital calibration, explicit SampleAndHold, characterization helpers). |

### 3.3 REUSED — unchanged

Everything in `pyDataconverter/components/` (no component needs to be aware of TI-ADC); every other architecture class (they use the default `convert_waveform`); `utils/signal_gen` (for coherent sine generation in tests and example); `utils/metrics` (for FFT/SNDR analysis in tests and example); `utils/_bits`; the existing test suite; the pipelined ADC classes from Phase 1 (they gain a new legitimate use case as a sub-ADC template inside TI-ADC, but require no modification).

---

## 4. Data flow per conversion

### 4.1 Pointwise path — `convert(v_in, dvdt=0.0)`

The base class `ADCBase.convert` sets `self._dvdt = float(dvdt)` and calls `self._convert_input(v_in)`. TI-ADC's `_convert_input`:

```python
def _convert_input(self, analog_input):
    # Bandwidth mismatch cannot be modelled pointwise. Fail loudly.
    if np.any(self.bandwidth != 0):
        raise RuntimeError(
            "bandwidth mismatch requires convert_waveform(); use that "
            "method or disable bandwidth to use the pointwise convert().")

    k = self._counter % self.M
    offset_k = float(self.offset[k])
    gain_k   = float(self.gain_error[k])
    skew_k   = float(self.timing_skew[k])

    # Input type resolution (single-ended vs differential)
    if self.input_type == InputType.DIFFERENTIAL:
        v_pos, v_neg = analog_input
        v = float(v_pos) - float(v_neg)
    else:
        v = float(analog_input)

    # First-order per-channel correction
    v_eff = v * (1.0 + gain_k) + offset_k + self._dvdt * skew_k

    # Repackage for the sub-ADC's expected input_type
    if self.input_type == InputType.DIFFERENTIAL:
        sub_input = (v_eff / 2 + self.v_ref / 2, -v_eff / 2 + self.v_ref / 2)
    else:
        sub_input = v_eff

    raw_code = self.channels[k].convert(sub_input, dvdt=self._dvdt)

    self._last_channel = k
    self._counter += 1
    return int(raw_code)
```

Key points:
- The mismatch corrections are applied to a single scalar `v` (differential inputs are converted to a single `v_diff` for the arithmetic, then repackaged back into a `(v_pos, v_neg)` tuple before the sub-ADC call).
- `dvdt` is forwarded to the sub-ADC as well, so any aperture jitter the sub-ADC already models still operates on top of the TI-ADC's timing skew. These are distinct physical effects: TI-ADC timing skew is a constant-per-channel phase error; aperture jitter is a random-per-sample draw. Both coexist cleanly.
- `_counter` advances by exactly one per call.

### 4.2 Waveform path — `convert_waveform(v_dense, t_dense)`

TI-ADC overrides the default inherited from `ADCBase`:

```python
def convert_waveform(self, v_dense, t_dense):
    v_dense = np.asarray(v_dense, dtype=float)
    t_dense = np.asarray(t_dense, dtype=float)
    if v_dense.shape != t_dense.shape or v_dense.ndim != 1:
        raise ValueError("v_dense and t_dense must be 1-D arrays of the same length")

    N = len(v_dense)
    dvdt_dense = np.gradient(v_dense, t_dense)

    # Per-channel bandwidth filtering (when active)
    if np.any(self.bandwidth != 0):
        dt = float(t_dense[1] - t_dense[0])
        fs_dense = 1.0 / dt
        v_per_channel = np.empty((self.M, N))
        for k in range(self.M):
            bw_k = float(self.bandwidth[k])
            if bw_k > 0:
                # First-order Butterworth LPF at the per-channel cutoff
                b, a = scipy.signal.butter(1, bw_k / (fs_dense / 2), btype='low')
                v_per_channel[k] = scipy.signal.lfilter(b, a, v_dense)
            else:
                v_per_channel[k] = v_dense
    else:
        v_per_channel = None  # sentinel: use v_dense directly

    codes = np.empty(N, dtype=int)
    for i in range(N):
        k = (self._counter + i) % self.M
        v_in = v_per_channel[k, i] if v_per_channel is not None else v_dense[i]

        offset_k = float(self.offset[k])
        gain_k   = float(self.gain_error[k])
        skew_k   = float(self.timing_skew[k])

        v_eff = v_in * (1.0 + gain_k) + offset_k + dvdt_dense[i] * skew_k

        if self.input_type == InputType.DIFFERENTIAL:
            sub_input = (v_eff / 2 + self.v_ref / 2, -v_eff / 2 + self.v_ref / 2)
        else:
            sub_input = v_eff

        codes[i] = int(self.channels[k].convert(sub_input, dvdt=dvdt_dense[i]))

    self._counter += N
    self._last_channel = (self._counter - 1) % self.M
    return codes
```

Non-TI subclasses inherit the default `ADCBase.convert_waveform`:

```python
def convert_waveform(self, v_dense, t_dense):
    v_dense = np.asarray(v_dense, dtype=float)
    t_dense = np.asarray(t_dense, dtype=float)
    if v_dense.shape != t_dense.shape or v_dense.ndim != 1:
        raise ValueError("v_dense and t_dense must be 1-D arrays of the same length")
    dvdt_dense = np.gradient(v_dense, t_dense)
    codes = np.empty(len(v_dense), dtype=int)
    for i in range(len(v_dense)):
        codes[i] = int(self.convert(float(v_dense[i]), dvdt=float(dvdt_dense[i])))
    return codes
```

The default gives every existing ADC (`SimpleADC`, `FlashADC`, `SARADC`, `PipelinedADC`) a waveform API for free. Users with dense waveforms (noise bursts, arbitrary signals, chirps) can feed them in without hand-computing `dvdt`. The default is a pointwise loop with numerically-derived `dvdt` — mathematically equivalent to the caller computing `dvdt` themselves via finite differences and calling `convert(v[i], dvdt=dvdt[i])`.

### 4.3 Hierarchical classmethod

```python
@classmethod
def hierarchical(cls,
                 channels_per_level,
                 sub_adc_template,
                 fs,
                 offset_std_per_level=None,
                 gain_std_per_level=None,
                 timing_skew_std_per_level=None,
                 bandwidth_std_per_level=None,
                 seed=None):
    """Build a multi-level interleaving tree from a list of per-level factors."""
    if not channels_per_level:
        raise ValueError("channels_per_level must have at least one entry")
    for i, m in enumerate(channels_per_level):
        if not isinstance(m, int) or m < 2:
            raise ValueError(f"channels_per_level[{i}]={m}, must be integer >= 2")

    L = len(channels_per_level)
    # Normalise per-level stddev lists to length L (default to zero)
    offset_levels = _resolve_per_level(offset_std_per_level, L)
    gain_levels   = _resolve_per_level(gain_std_per_level, L)
    skew_levels   = _resolve_per_level(timing_skew_std_per_level, L)
    bw_levels     = _resolve_per_level(bandwidth_std_per_level, L)

    # Start from the innermost level and walk outward.
    # channels_per_level[0] is the OUTERMOST factor; [-1] is the innermost.
    current_template = sub_adc_template
    fs_at_level = fs
    for level_index in reversed(range(L)):
        M_level = channels_per_level[level_index]
        current_template = cls(
            channels=M_level,
            sub_adc_template=current_template,
            offset=offset_levels[level_index],
            gain_error=gain_levels[level_index],
            timing_skew=skew_levels[level_index],
            bandwidth=bw_levels[level_index],
            fs=fs_at_level,
            seed=seed,
        )
        fs_at_level = fs_at_level / M_level  # next-inner level sees the slower rate

    return current_template
```

The convention is that `channels_per_level[0]` is the outermost (fastest) level. `fs` is the aggregate sample rate of the top-level TI-ADC; each inner level's effective rate is computed by dividing by the outer factors.

### 4.4 `split_by_channel(codes)`

```python
def split_by_channel(self, codes):
    codes = np.asarray(codes, dtype=int)
    if codes.ndim != 1:
        raise ValueError(f"codes must be 1-D, got shape {codes.shape}")
    if len(codes) % self.M != 0:
        raise ValueError(
            f"len(codes)={len(codes)} is not a multiple of M={self.M}; "
            "pass an array whose length is an integer multiple of the "
            "channel count.")
    N_per_channel = len(codes) // self.M
    return codes.reshape(N_per_channel, self.M).T  # shape (M, N_per_channel)
```

Row `k` of the returned array is channel `k`'s code sequence. `split_by_channel` does not consult the TI-ADC's internal state — it operates on whatever array the caller passes. This makes it stateless and composable with test fixtures.

---

## 5. Error handling and validation

All validation fires at `__init__` time or at `convert()` time; there is no lazy validation. Error messages name the offending parameter and value.

### 5.1 `TimeInterleavedADC.__init__`

| Parameter | Rule | Error |
|---|---|---|
| `channels` (M) | integer ≥ 2 | `TypeError` / `ValueError` |
| `sub_adc_template` | instance of `ADCBase` | `TypeError` |
| `offset` | scalar (non-negative if interpreted as stddev) OR 1-D array of length M | `TypeError` / `ValueError` with observed shape |
| `gain_error` | scalar (non-negative if interpreted as stddev) OR 1-D array of length M | `TypeError` / `ValueError` |
| `timing_skew` | scalar (non-negative if interpreted as stddev) OR 1-D array of length M | `TypeError` / `ValueError` |
| `bandwidth` | scalar (non-negative if interpreted as stddev) OR 1-D array of length M | `TypeError` / `ValueError` |
| `fs` | positive float (aggregate sample rate) | `ValueError` |
| `seed` | int or `None` (for RNG in the stddev case) | `TypeError` |
| `n_bits`, `v_ref`, `input_type` | MUST NOT be passed — inherited from template | `TypeError` with a message directing the user to configure these on the template |

The mismatch parameters accept both scalars and arrays. Scalars are interpreted as the stddev of `N(0, σ)` per-channel random draws; arrays are interpreted as explicit per-channel values (the array's length must equal `M`). A scalar of 0 or `None` disables that mismatch type. When the stddev path is taken, `np.random.default_rng(seed)` is used so that constructions with the same `seed` produce identical mismatch realizations — essential for reproducible tests.

`n_bits`, `v_ref`, and `input_type` are forbidden as constructor arguments — they are determined by the template and must not be overridden. The class raises if any of these names appear in the constructor's `**kwargs`, protecting against silent mismatches between the template and the wrapper.

### 5.2 `TimeInterleavedADC._convert_input` (runtime)

- `np.any(self.bandwidth != 0)` → `RuntimeError("bandwidth mismatch requires convert_waveform(); use that method or disable bandwidth to use the pointwise convert().")`. This is the deliberate API asymmetry.
- Input type check for `analog_input` is inherited from `ADCBase.convert`.

### 5.3 `TimeInterleavedADC.split_by_channel`

- `codes.ndim != 1` → `ValueError` naming the observed shape.
- `len(codes) % self.M != 0` → `ValueError` naming both `len(codes)` and `self.M`.

### 5.4 `TimeInterleavedADC.hierarchical`

- `len(channels_per_level) < 1` → `ValueError`.
- Any `channels_per_level[i]` non-integer or `< 2` → `ValueError` naming the offending index and value.
- Any mismatch-stddev list whose length does not match `len(channels_per_level)` → `ValueError`.

### 5.5 `ADCBase.convert_waveform` (default)

- `v_dense.shape != t_dense.shape` → `ValueError`.
- `v_dense.ndim != 1` → `ValueError`.
- Everything else: propagates from the per-sample `convert()` calls.

### 5.6 Deliberately not guarded

- **Sample-rate mismatches** between `fs` and the spacing in `t_dense` during `convert_waveform`. If the user passes a `t_dense` with an irregular grid, the `np.gradient` call handles it correctly but the per-channel LPF filtering assumes a regular sample rate derived from `t_dense[1] - t_dense[0]`. Document this in the docstring; do not enforce regularity at runtime. Users doing irregular-grid simulations are advanced users who should know the assumption.
- **Per-channel LPF cutoff exceeding Nyquist.** If `bandwidth[k] > fs_dense/2` the `butter` call will clip or raise; let the scipy error propagate.
- **Negative timing skew.** Negative values are physically meaningful (a channel that samples early rather than late). Do not reject them. Only raise if the user passes a non-number type.

---

## 6. Testing strategy

Three layers, same structure as the Pipelined ADC Phase 1 test suite:

### 6.1 Layer 1 — self-consistency tests (`tests/test_ti_adc.py::TestTIADCIdeal`)

For each of several sub-ADC templates (`SimpleADC`, `FlashADC`, `SARADC`), construct:

```python
template = FlashADC(n_bits=6, v_ref=1.0, input_type=InputType.SINGLE)
ti = TimeInterleavedADC(channels=4, sub_adc_template=template, fs=1e9)
# All four mismatches default to zero
```

and assert that:

1. For every input `v` in a dense sweep, `ti.convert(v)` returns exactly the same code as `template.convert(v)` (bit-exact, no tolerance). The rotation state advances but the output value is identical because every channel is an ideal copy of the template.
2. `ti.convert_waveform(v_dense, t_dense)` returns an array where every element equals the corresponding `template.convert(v_dense[i], dvdt=dvdt_dense[i])` call — again bit-exact.
3. Hierarchical construction via `TimeInterleavedADC.hierarchical(channels_per_level=[4, 2], sub_adc_template=template, fs=1e9)` with zero mismatches produces the same output sequence as `template` alone for any input.
4. `adc.reset()` restores `_counter` to 0.

This layer is the TI-ADC equivalent of the pipelined ADC's bit-exact comparison tests: a binary correctness check with no wiggle room.

### 6.2 Layer 2 — analytical spur matching (`tests/test_ti_adc_spurs.py`)

Each test drives a coherent sine through a TI-ADC with exactly one mismatch type active, runs an FFT on the output, and asserts that specific tones appear at specific frequencies with magnitudes matching closed-form formulas.

**Test 1: offset mismatch spurs.**

Setup: `M=4`, explicit `offset=[1e-3, -1e-3, 0.5e-3, -0.5e-3]` (all other mismatches zero). Drive a coherent sine at `f_in = 5·fs/M/NFFT` bins from DC. Assert that tones appear at `k·fs/M` for `k = 1, 2, 3` with magnitudes matching the closed-form formula `20·log10(σ_offset·√(2/M)/v_ref)`. Tolerance ±2 dB (finite-FFT leakage). Also assert no unexpected tones at other image locations.

**Test 2: gain mismatch spurs.**

Setup: `M=4`, explicit `gain_error=[0.001, -0.001, 0.0005, -0.0005]`. Drive a coherent sine at `f_in`. Assert image at `fs/M − f_in` at magnitude `20·log10(σ_gain·amplitude/(2·v_ref))`. Tolerance ±2 dB.

**Test 3: timing-skew spurs (slope-linear).**

Setup: `M=4`, explicit `timing_skew=[1e-12, -1e-12, 0.5e-12, -0.5e-12]`. Drive two coherent sines at different frequencies `f1` and `f2 = 2·f1`. Assert that the image at `fs/M − f_in` has magnitude `20·log10(2π·σ_τ·f_in·amplitude/v_ref)` at both frequencies, AND that the magnitude scales linearly with `f_in` (within 1 dB). The linearity-with-frequency check is what distinguishes timing skew from gain mismatch — both produce the same spur location but only skew scales with `f_in`.

**Test 4: bandwidth mismatch spurs (frequency-dependent).**

Setup: `M=4`, explicit `bandwidth=[1e9, 2e9, 0.5e9, 1.5e9]` (per-channel LPF cutoffs in Hz). Drive coherent sines at two input frequencies `f_low` and `f_high`. Run via `convert_waveform` (pointwise `convert` raises). Assert that the resulting images at `fs/M ± f_in` grow with `f_in` following the first-order LPF magnitude-response difference across channels (i.e., the spur should be negligible at DC and grow with `f_in` as the channels diverge in their LPF droop). Tolerance ±3 dB.

**Test 5: `convert()` raises when `bandwidth != 0`.**

Construct a TI-ADC with nonzero bandwidth. Assert `adc.convert(0.5)` raises `RuntimeError` with a message mentioning `convert_waveform`.

### 6.3 Layer 3 — API and composition tests (`tests/test_ti_adc.py`)

**Constructor validation** (one test per rule in §5.1): `channels < 2`, non-`ADCBase` template, wrong-length offset array, non-int `channels`, wrong-type `fs`, forbidden `n_bits` kwarg, forbidden `v_ref` kwarg, forbidden `input_type` kwarg.

**Channel rotation state**: call `convert()` 8 times on an `M=4` TI-ADC, inspect `last_channel` after each to verify the rotation `[0, 1, 2, 3, 0, 1, 2, 3]`. Call `reset()`, verify `last_channel` is `None` (or the most recent valid value, whichever the spec settles on — see Open Question 1 below).

**`split_by_channel`**: construct a known 1-D array of length 12, call `split_by_channel` on an `M=4` TI-ADC, verify the returned 2-D array has shape `(4, 3)` and the right per-row content. Test that `len % M != 0` raises.

**Hierarchical construction**: build `hierarchical(channels_per_level=[4, 2], ...)`, inspect `ti.channels[0].sub_adc_template.channels[0]` to verify the tree structure. Verify that an 8-sample `convert_waveform` rotates through all 8 leaf channels in the right order.

**Composability with `PipelinedADC`**: construct a `PipelinedADC(backend=TimeInterleavedADC(...))`. Run a short sweep. Assert it produces valid codes and doesn't raise. (This is a smoke test — we're not asserting bit-exact results, just that the composition is mechanically valid.)

**`ADCBase.convert_waveform` default**: for `SimpleADC`, `FlashADC`, `SARADC` (non-TI subclasses), construct a short dense waveform and call `convert_waveform(v_dense, t_dense)`. Assert the returned array equals a manual loop `[adc.convert(v, dvdt=dvdt_dense[i]) for i, v in enumerate(v_dense)]`. Assert `v_dense.shape != t_dense.shape` raises.

**Coverage target**: ≥ 95% statement coverage on the new `TimeInterleavedADC` class; 100% coverage of the new lines in `dataconverter.py` for `convert_waveform`.

---

## 7. Phase boundaries

### 7.1 Phase 1 — this spec, ships as one commit series

Deliverables:

1. `ADCBase.convert_waveform` default implementation.
2. `TimeInterleavedADC(ADCBase)` with flat constructor, `hierarchical` classmethod, `split_by_channel`, `last_channel` property.
3. TI-ADC `convert_waveform` override applying per-channel LPFs for bandwidth mismatch.
4. The four mismatch types: offset, gain_error, timing_skew, bandwidth. Each accepts a scalar (stddev) or array (explicit) argument.
5. Construction validation covering every rule in §5.1.
6. Runtime validation: pointwise `convert` raises when bandwidth is active.
7. Tests from §6 (three layers).
8. `examples/ti_adc_example.py`.
9. API reference update + roadmap status flip.

Acceptance criteria:

1. All Layer 1 self-consistency tests pass (ideal TI-ADC = template alone for every input).
2. All Layer 2 analytical spur tests pass within their stated tolerances (±2 dB for offset/gain/skew, ±3 dB for bandwidth).
3. All Layer 3 construction/API/composition tests pass.
4. Existing test suite still passes (898 → 898 + N new tests, no regressions).
5. `examples/ti_adc_example.py` runs end-to-end and produces a spectrum plot with visible mismatch spurs.
6. `docs/api_reference.md` documents every new public API element.
7. `todo/adc_architectures.md` marks TI-ADC `PHASE 1 IMPLEMENTED`.

### 7.2 Phase 2 — deferred, tracked in roadmap

- **Second-order-correct timing skew** via cubic-spline interpolation at `t + τ_k` in `convert_waveform`. Phase 1 uses first-order Taylor (`v + dvdt · τ_k`), which is accurate when `τ·f_in ≪ 1` but degrades at very-high-frequency inputs. Captures the `(τ·f_in)²` second-order term.
- **Digital calibration engine** for offset/gain/skew/bandwidth estimation and correction. Needs its own research-grade design pass (estimation algorithms are non-trivial). The TI-ADC would accept optional `offset_correction`, `gain_correction`, etc. arrays at construction that are applied digitally at the output before returning codes.
- **Bandwidth mismatch via stateful per-channel LPF in the pointwise path**. Removes the API asymmetry by letting `convert()` work for all four mismatch types. Requires per-channel filter state carried across sample calls, which is the kind of inter-sample state deferred in `todo/parking_lot.md`. Separate design pass.
- **Explicit `SampleAndHold` component** (shared with Pipelined ADC Phase 2). A per-channel S&H would let users model front-end bandwidth, input impedance, and droop explicitly; currently TI-ADC treats each sub-ADC's input as ideal and applies mismatches as input-referred corrections.
- **TI-ADC characterization helpers** in `utils/characterization.py`: per-channel INL sweeps, mismatch-spur extraction from output spectra, channel-by-channel transfer-function plotting.
- **Heterogeneous channel templates**: explicit list of `M` sub-ADC instances at construction, instead of a single template. Would support experiments like "channel 0 is a SAR, channels 1-3 are Flashes". Separate brainstorming session.

### 7.3 Deliberately NOT in scope — any phase of TI-ADC work

- Dynamic clock phase adjustment (calibration loops that mutate `timing_skew` during conversion).
- Channel power gating / low-power modes.
- Input signal pre-distortion.
- Foreground vs. background calibration distinctions (that's the digital calibration engine's concern).
- Multi-core / parallel per-channel execution. TI-ADC is inherently sequential in its output stream; any parallelism is an implementation detail that does not change the externally observable behavior.

---

## Appendix A — Mismatch spur formulas

The following closed-form formulas come from the TI-ADC literature (Razavi, Murmann, Kester, Gustavsson). They are what Layer 2 of the test suite validates against. Full derivations are beyond the scope of this spec; the test file should include a comment block citing these results.

### A.1 Offset mismatch

Per-channel DC offsets produce DC-independent tones at every multiple of the per-channel sample rate:

- **Location**: `f_spur = k · fs / M` for `k = 1, 2, ..., M−1`.
- **Magnitude** (normalised to full-scale): for explicit offsets with zero mean and variance `σ_off²`, the magnitude of the `k=1` spur is approximately `σ_off · √(2/M) / v_ref` in linear units, or `20·log10(σ_off·√(2/M)/v_ref)` in dBFS.
- **Independent of input frequency and amplitude.** This is the signature that distinguishes offset mismatch from the other three.

### A.2 Gain mismatch

Per-channel gain errors modulate the input signal, producing images near every multiple of the per-channel sample rate, each at the same distance from the carrier as the input frequency:

- **Location**: `f_image = k · fs / M ± f_in` for `k = 1, 2, ..., M−1`.
- **Magnitude**: for explicit gain errors with variance `σ_g²`, the magnitude of the `k=1, −f_in` image is approximately `σ_g · amplitude / (2·v_ref)` in linear units.
- **Scales linearly with input amplitude, independent of input frequency.** The amplitude-linearity distinguishes it from offset (amplitude-independent) and skew (also amplitude-linear but additionally frequency-linear).

### A.3 Timing skew

Per-channel sampling clock phase errors cause each channel to sample the input at a slightly different time. For a sine input `v_in(t) = A·sin(2π·f_in·t)`, the first-order Taylor correction is `v_k(t) = v_in(t) + (2π·f_in·A)·cos(2π·f_in·t)·τ_k`.

- **Location**: same as gain mismatch — `f_image = k · fs / M ± f_in`.
- **Magnitude**: for explicit skews with variance `σ_τ²`, the magnitude of the `k=1, −f_in` image is approximately `2π · σ_τ · f_in · amplitude / v_ref` in linear units.
- **Scales linearly with both input frequency AND amplitude.** The frequency scaling is what distinguishes it from pure gain mismatch — gain images don't grow with `f_in`, skew images do.

### A.4 Bandwidth mismatch

Per-channel analog bandwidth differences cause each channel to see a frequency-dependent amplitude and phase distortion of the input. For a first-order LPF at cutoff `BW_k`, the channel's transfer function is `H_k(f) = 1 / (1 + j·f/BW_k)`. Different `BW_k` across channels produces frequency-dependent amplitude and phase variations that appear as images at the same locations as gain and skew, but with magnitudes that follow the LPF transfer-function difference.

- **Location**: same as gain and skew — `f_image = k · fs / M ± f_in`.
- **Magnitude**: at input frequency `f_in`, each channel's transfer function differs from the others by an amount determined by `1/√(1 + (f_in/BW_k)²) − 1/√(1 + (f_in/BW_avg)²)`. The resulting image magnitude depends on the variance of these differences across channels, and grows as `f_in → BW_avg`. Closed-form formulas are more involved than the offset/gain/skew cases; the Phase 1 test accepts a looser ±3 dB tolerance.
- **Negligible at DC, grows with `f_in`, saturates when `f_in ≫ BW`.** The frequency-dependent signature distinguishes it from all three other mismatch types.

---

## Appendix B — Open design questions to resolve during implementation

These are small, local decisions that don't affect the overall shape and can be settled by the implementer:

1. **`last_channel` value immediately after construction or after `reset()`**: should it be `None`, `-1`, or raise `AttributeError`? Choose whichever is most consistent with `SARADC`'s existing debug properties. (Suggest: `None`, so a test can assert `adc.last_channel is None`.)
2. **Exact `scipy.signal.butter` arguments** for the per-channel LPFs: whether to use `btype='low'`, `output='ba'` vs `output='sos'`, and whether to expose the LPF order as a kwarg. (Suggest: `btype='low'`, `output='ba'`, order fixed at 1 for Phase 1 — document that Phase 2 will add `bandwidth_order`.)
3. **Whether `convert_waveform` should cache `v_per_channel` between calls** when the waveform is unchanged. (Suggest: no — keep it stateless. Premature optimization.)
4. **Whether `split_by_channel` should handle a 2-D input and transpose it back** (i.e., round-trip support). (Suggest: no — accept 1-D only, raise on anything else.)

None of these blocks the implementation plan. They can be settled by the implementer using the suggestions as defaults and flagged in the plan if any turns out to conflict with a test.
