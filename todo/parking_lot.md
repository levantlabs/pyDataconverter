# Parking Lot — Deferred Ideas

Ideas and redesigns we've consciously decided not to do *right now*, but
that we want to preserve the thinking on so we don't re-derive it later.
Anything here is a candidate for a future focused session.

---

## 1. Batch `convert_batch(analog_inputs)` API for ADC/DAC characterization

**Why deferred (2026-04-13):** simpler to keep per-sample `.convert()` as
the only API until the memory-effects story is properly thought through.
A naive vectorisation would silently give wrong results for any converter
that has inter-sample state (hysteresis, bandwidth LPF, noise-shaping
integrator), and quietly papering over that with a sequential fallback
hides a real architectural question.

### The core distinction

- **Intra-sample state** is thrown away at the end of each conversion
  (SAR register, bit-cycling DAC voltage, temporary `v_input`). Safe to
  vectorise across samples by flipping the loop order: outer loop walks
  bit positions (≤ n_bits iterations), inner loop is a NumPy op over all
  N samples at once. 100-1000x speedup typical for SimpleADC/FlashADC,
  10-50x for SARs.
- **Inter-sample state** makes sample N+1 depend on residues of sample N.
  Serially dependent — cannot be vectorised across samples regardless of
  implementation language. This is a physical constraint, not a code
  smell.

### Inter-sample state already in the codebase

| Component | State field | Mechanism | Location |
|---|---|---|---|
| `DifferentialComparator` | `_last_output` | Hysteresis: threshold is `±h/2` depending on previous decision. | `comparator.py:170-176` |
| `DifferentialComparator` | `_filtered_state` | First-order LPF: `v[n] = (1-α)·v[n-1] + α·v_new[n]`. Linear recurrence; inside a SAR feedback loop this can't even use `scipy.signal.lfilter` because the input depends on the filter output one bit at a time. |  `comparator.py:161-163` |
| `NoiseshapingSARADC` | `integrator_state` | Residue from sample N feeds sample N+1's SAR input. | `SARADC.py:480-509` |

### Memory effects the code does NOT model yet but real converters have

These matter because sooner or later someone will want them, and the
batch API needs to not paint itself into a corner:

1. **1/f (flicker) noise** on references, comparators, current sources.
   Power spectrum goes as 1/f; correlations extend over thousands of
   samples. A per-sample independent `N(0, σ)` is simply wrong.
2. **Clock phase noise driving aperture jitter.** Real phase noise has a
   spectral shape; the current model treats jitter as white.
3. **DAC glitch-energy tails** — a DAC that just transitioned from code A
   to code B has ringing/settling energy that bleeds into sample N+1 if
   the sample period is comparable to the settling time.
4. **Reference supply bounce** — each conversion draws a transient from
   v_ref; v_ref sags/rings between conversions.
5. **Capacitor dielectric absorption** (slow voltage "memory" on the
   sampling cap), **thermal self-heating** (code-dependent chip
   temperature shifts reference), **metastability recovery** in
   comparators. Specialised but real.

### Three categories a well-designed batch API has to handle

| Category | Example | Vectorisation strategy |
|---|---|---|
| Stateless per sample | gain_error, offset, independent Gaussian noise, plain quantisation, Flash bank without hysteresis/BW | Full vectorisation. All arithmetic in one NumPy expression. |
| Signal-independent stochastic trajectory | 1/f noise, phase-noise-driven jitter, thermal drift | Pre-generate the full noise trajectory of length N *before* the conversion loop, then add as a vector perturbation. The noise doesn't depend on the signal, so the trajectory is known up front. |
| Signal-dependent sequential state | Hysteresis, bandwidth LPF inside a SAR loop, noise-shaping integrator, DAC glitch tails, metastability | Cannot vectorise. Options: (a) fall back to per-sample loop, (b) detect-and-refuse with a clear error, (c) specialised kernel only where the recurrence is linear and decoupled from any feedback loop (rare). |

Interesting edge case: the comparator bandwidth LPF **outside** a
feedback loop (as a front-end filter on the input signal) IS a linear
recurrence you can vectorise via `scipy.signal.lfilter`. But inside a
SAR feedback loop, the filter input depends on the comparator output one
bit at a time — same recurrence, completely different tractability
depending on whether it's in a feedback path.

### Proposed API contract (when we pick this up)

```python
class ADCBase:
    def convert_batch(
        self,
        analog_inputs: np.ndarray,
        *,
        reset_state: bool = True,   # clear inter-sample state at entry?
        strict: bool = False,       # raise if batching would silently
                                    # serialise the stateful path?
    ) -> np.ndarray:
        """
        Vectorised conversion. Behaviour depends on enabled non-idealities:
        - Fully stateless: fast path (single NumPy expression).
        - State-carrying: falls back to per-sample loop. If strict=True,
          raises NotImplementedError instead of silently serialising.
        Noise trajectories are drawn once up-front so the RNG consumption
        is identical to N sequential .convert() calls with the same seed.
        """
```

**The single most important contract property:** `convert_batch(inputs, seed=42)`
and `[convert(x, seed=42) for x in inputs]` MUST produce identical
outputs. If they don't, batching becomes unusable for regression testing
(every test that compares outputs becomes batch-dependent). Requires
careful use of `Generator.spawn()` or equivalent sub-stream allocation.

### Three-stage roll-out (when we pick this up)

1. Draft the `convert_batch` contract (signature, state policy, RNG
   policy, error modes) as a separate markdown design doc. Get alignment
   before any code change.
2. Pilot on `SimpleADC` — fully stateless today, clean proof-of-concept,
   measurable speedup. Establishes the pattern.
3. Extend to `FlashADC` and the non-shaping `SARADC` once the pattern
   holds.

`NoiseshapingSARADC` stays sequential forever. Document in the class
docstring.

---
