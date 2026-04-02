# pyDataconverter — Roadmap Brainstorm

> Draft for discussion. Add comments inline.

My comments are prefaced by MEC: 

---

## 1. Additional Metrics

### ADC Static
- Gain error and offset error as explicit scalar outputs (currently implicit in INL). MEC: let's do this.  Then the INL code should call this so that we don't have replicated code
- Missing codes (codes where DNL ≤ −1 LSB).  MEC:  let's do this.  
- Transition noise — standard deviation of each transition voltage over repeated sweeps:  MEC:  is this a real data converter metric?
- Code width histogram uniformity (chi-squared test against ideal).  MEC: is this a real data converter metric

### ADC Dynamic
- Individual harmonic levels (HD2, HD3, HD4 …) rather than aggregate THD only: MEC:  let's do this.  let's also generalize to tthe nth harmonic, and then have metrics for non-HD23 harmonics
- Two-tone IIP3 / OIP3 — third-order intercept point from an IMD measurement.  MEC:  let's do this
- Noise spectral density (NSD) in dBFS/√Hz — useful for oversampling contexts: MEC: let's do this
- ERBW (effective resolution bandwidth) — frequency at which ENOB degrades by 0.5 bits; requires sweeping input frequency.  MEC:  let's do this
MEC: we also need dynamic range, which requires and amplitude sweep

### DAC
- Settling time — samples to settle within ½ LSB of final code (needs step-response simulation).  MEC: let's skip this for now, add it to a later list
- Glitch energy — glitch impulse area at major-carry transitions (e.g. 0111→1000):  MEC: let's skip this for now, this requires some timing modeling.  Add it to a later list
- Code-dependent output impedance: MEC:  let's skip this

---

## 2. Additional Signal Types

- **Chirp / swept sine** — sweep frequency over time; essential for ERBW measurement.  MEC:  please add
- **PRBS** (pseudo-random binary sequence) — flat-spectrum broadband stimulus for noise floor and memory-effect testing.  MEC: let's add.  There should be a way to convolve with a channel too
- **Windowed sine** — explicit Hann, Blackman-Harris, flat-top window support in `signal_gen`.  MEC: let's add
- **Burst / step** — clean single-step between two codes; for settling time measurement. MEC: let's skip for now, add it to a later list
- **Noise-modulated sine** — sine + additive Gaussian noise; for aperture jitter sensitivity. MEC: no need, this is just noise.  But there should be a noise generator available
- **Incremental staircase** — steps up one LSB at a time; for transition noise measurement.  MEC:  not needed

---

## 3. Generalising the SAR ADC

Replace the binary weight vector `[2^(N−1), …, 2, 1]` with an arbitrary weight vector. This enables:

- **Redundant SAR** (radix < 2, e.g. 1.8 or 1.85 per bit) — overlapping decision regions allow digital error correction of comparator metastability; widely used in production
- **Split-capacitor array** — bridge capacitor between coarse and fine sub-arrays; reduces total capacitance ~2× with no accuracy loss
- **Segmented SAR** — thermometer-coded MSB section + binary LSB section
- **Multi-bit per cycle** — replace single comparator with a small flash sub-ADC to resolve multiple bits per cycle
- **Noise-shaping SAR** — integrates residue voltage and feeds it forward, shaping quantisation noise out of band

The CDAC already accepts a capacitor array; the main work is making the successive-approximation logic weight-aware and adding a digital error correction layer.

MEC:  let's do all of 3

---

## 4. Voltage-Mode DACs

### Resistor String DAC
- 2^N equal resistors in series between V_ref and GND; switches select the appropriate tap
- Inherently monotonic (DNL > −1 LSB always)
- Model: draw each resistor from `R(1 + ε)`, ε ~ N(0, σ²); solve the divider; INL comes from cumulative mismatch
- Key parameter: resistor matching σ (e.g. 0.1–0.5%)

### R-2R Ladder DAC
- Only 2N resistors; each bit switches its node between V_ref and GND
- Voltage-output (Thévenin) and current-summing (virtual-ground) variants
- Model: full ladder as resistor network with per-element mismatch, solved via nodal analysis
- R vs 2R mismatch have distinct INL signatures — worth modelling separately
- Output impedance is code-dependent

### Segmented Resistor String
- Coarse thermometer section (resistor string for MSBs) + fine binary section (R-2R or sub-string for LSBs)
- Reduces switch count vs. full string; improves glitch behaviour

### Shared Infrastructure
All voltage-mode DACs need a **resistor network solver** (nodal analysis) as the core simulation primitive, analogous to charge redistribution in the CDAC.

MEC:  let's do all of 4

---

## 5. Suggested Priorities

| Priority | Item | Rationale |
|---|---|---|
| 1 | Resistor string DAC | Simplest voltage-mode model; immediately useful |
| 2 | R-2R ladder DAC | Most common architecture; requires nodal solver |
| 3 | Redundant SAR | High practical value; contained change to existing SAR |
| 4 | Settling time + glitch energy metrics | Completes DAC characterisation story |
| 5 | Chirp + ERBW | Rounds out dynamic metrics |

MEC:  I am good with this order, please feel free to use a full team to implement all.  Don't forget to create additional tests for these, to vet the design decisions, to have a code reviewer, and finally to update the documentation

---

*Add comments / edits below or inline above.*
