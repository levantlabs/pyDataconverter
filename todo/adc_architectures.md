ADC Architecture Roadmap
========================

Date: 2026-04-13.

Current state: five ADC families are implemented in `pyDataconverter/architectures/`
— `SimpleADC` (behavioural), `FlashADC` (full reference ladder + comparator bank),
`SARADC` (configurable CDAC backend), `MultibitSARADC` (multi-bit per cycle via
flash sub-ADC), and `NoiseshapingSARADC` (first-order Nyquist-rate shaping).

This file lists ADC architectures that are not yet modelled, in rough order of
value-to-add. Entries with STATUS: IN DESIGN are currently being scoped in a
brainstorming session; see `docs/superpowers/specs/` for the resulting design
docs.

---

## 1. Pipelined ADC  [HIGH] — STATUS: IN DESIGN (2026-04-13)

**What it is:** A cascade of N stages. Each stage resolves a small number of bits
(typically 1, 1.5, 2, or 2.5 bits) using a sub-ADC, reconstructs the quantised
value through an internal DAC, subtracts it from the held input to produce a
residue, and amplifies the residue to full scale before passing it to the next
stage. A digital backend aligns the per-stage bits and applies redundancy-based
error correction.

**Why model it:** The dominant architecture for 10–14 bit ADCs at 50–500 MS/s
(telecom, instrumentation, imaging). Rich modelling content around stage gain
error, capacitor mismatch, amplifier settling, and inter-stage crosstalk. Also
forces us to build reusable sub-components (MDAC, residue amplifier,
sample-and-hold) that unlock several other architectures.

**Key non-idealities to add:**
- Residue amplifier finite gain → closed-loop gain error per stage
- Residue amplifier slew rate and settling time (bandwidth-limited)
- Residue amplifier offset
- MDAC capacitor mismatch (per-stage, independent)
- Comparator offsets inside the sub-ADC (absorbed by redundancy if the stage has
  overlap, otherwise directly visible as INL)
- Inter-stage timing skew (if clock phases are staggered)

**Components needed (some new, some existing):**
- **`ResidueAmplifier`** (new) — models gain, slew, settling, offset. Input is a
  held voltage; output is `G * (v_in - v_dac)` with the above imperfections.
- **`MDAC`** (new) — composes a sample-and-hold, a sub-DAC (likely a reuse of
  `SingleEndedCDAC` or `DifferentialCDAC`), a subtractor, and a `ResidueAmplifier`.
  Exposes `convert_stage(v_in) -> (stage_code, v_residue)`.
- **`PipelineStage`** (new) — composes a sub-ADC (likely `FlashADC` or a 1.5-bit
  custom comparator pair) and an `MDAC`. Clean unit, one stage = one pipeline
  cell.
- **`PipelinedADC`** (new, extends `ADCBase`) — cascades N `PipelineStage`
  instances plus an optional final flash stage and a digital alignment / error
  correction backend.

**Bits-per-stage decision:** to be resolved in the design session. Options
include 1, 1.5, 2, and 2.5 bits per stage, or a heterogeneous cascade.

---

## 2. Time-Interleaved ADC (TI-ADC)  [HIGH] — STATUS: IN DESIGN (2026-04-13)

**What it is:** M parallel sub-ADCs sampling the same input at staggered clock
phases. Sample N is routed to sub-ADC (N mod M); the output is reassembled in
order. A perfect TI-ADC behaves as a single ADC at M times the per-channel
rate. Imperfect channels generate characteristic mismatch spurs.

**Why model it:** The only way to reach multi-GS/s in CMOS. Orthogonal to the
other architectures — you can interleave Flash, SAR, or pipelined sub-ADCs.
The interesting modelling content lives entirely in the channel mismatches and
their spectral signatures, which are hard to study any other way.

**Key non-idealities to add (all inter-channel):**
- Offset mismatch → spur at `fs/M` and its multiples (DC-independent)
- Gain mismatch → images at `±fin` around `k·fs/M` for each channel
- Timing skew (sampling clock phase error) → images at `k·fs/M ± fin`, scaled by
  `fin`
- Bandwidth mismatch → frequency-dependent images (sub-ADC input BW differs)

**Components / design decisions:**
- **Composition over inheritance.** The TI wrapper should accept any
  `ADCBase` instance as a channel template and replicate it M times; it should
  not require a dedicated TI subclass per architecture.
- **`TimeInterleavedADC`** (new, extends `ADCBase`) — holds M sub-ADCs, a
  channel mux, per-channel mismatch parameters, and a digital recombiner.
- **Calibration support:** expose hooks for per-channel offset, gain, and
  timing correction that can be driven either from a configuration dict or
  from a calibration routine run offline.

---

## 3. Sigma-Delta (ΔΣ) ADC  [HIGH]

**What it is:** Oversampled loop with noise shaping. A low-resolution (often
1-bit) quantizer inside a feedback loop with an integrator chain; the loop
shapes quantisation noise away from the signal band. A decimation filter on the
output removes out-of-band noise and reduces the sample rate to Nyquist.

**Why model it:** Dominant architecture for audio, instrumentation, and
precision measurement. Conceptual cousin of `NoiseshapingSARADC` but a
distinct architecture — `NoiseshapingSARADC` is a Nyquist-rate first-order
shaped SAR, not a proper ΔΣ.

**Key components:**
- Loop filter (integrator chain — first, second, third order)
- Multi-bit or 1-bit quantizer
- Feedback DAC (shares infrastructure with the planned Sigma-Delta DAC)
- Decimation filter (CIC + droop compensation + FIR) — often more code than the
  modulator itself
- Optional: dynamic element matching on the inner DAC

**Recommended starting point:** First-order modulator with 1-bit inner DAC and a
CIC-only decimation filter. Add second-order, higher-OSR, and multi-bit
variants as follow-ons.

---

## 4. Integrating / Dual-Slope ADC  [MEDIUM]

**What it is:** Converts the input voltage into a time interval by integrating
it onto a capacitor for a fixed time, then discharging with a known reference
current; the digital code is proportional to the discharge duration.

**Why model it:** Slow but extremely linear. Standard in handheld DMMs and
electrochemistry instruments. Pedagogically valuable because it makes the
time-domain integration explicit, and it's a great test case for 1/f noise
rejection (integration acts as a low-pass).

**Key non-idealities:**
- Integrator capacitor dielectric absorption
- Reference current drift during the discharge phase
- Comparator offset (zero-crossing detector)
- Input noise averaged over the integration window (hence the 1/f rejection)

---

## 5. Folding (and Folding-and-Interpolating) ADC  [MEDIUM-LOW]

**What it is:** A hybrid between flash and SAR. Analog "folding" circuits
generate a piecewise-linear, high-frequency-output waveform from the input,
which a smaller flash then quantises. Far fewer comparators than full flash for
the same resolution.

**Why model it:** Historically common in high-speed oscilloscope front-ends;
now niche. Interesting because the folding amplifier is an unusual analog
primitive that doesn't appear anywhere else in the converter family.

---

## 6. Subranging / Two-Step ADC  [LOW]

**What it is:** Coarse flash gives the MSBs, residue is fed to a fine flash for
the LSBs. Conceptual ancestor of the pipelined ADC — functionally a two-stage
pipeline with no inter-stage amplification.

**Why model it:** If the pipelined ADC is built cleanly, subranging is nearly a
degenerate case (two stages, unity inter-stage gain). Low incremental effort
once pipelined exists.

---

## 7. Cyclic / Algorithmic ADC  [LOW]

**What it is:** A pipelined ADC where the same physical stage is reused for
every bit via a feedback loop. Outputs one bit per clock cycle. Compact, slow,
occasionally seen in smart-sensor front-ends and low-power IoT.

**Why model it:** Close relative of pipelined; trivial to derive once the
pipeline stage component exists.

---

## 8. VCO-based / TDC-based ADC  [MEDIUM — deep-submicron interest]

**What it is:** Converts voltage to frequency (VCO-based) or phase (TDC-based)
and then quantises in the time domain. Benefits from fast CMOS time resolution
and doesn't need precision analog building blocks, making it attractive in
deep-submicron processes where voltage headroom is tight.

**Why model it:** Modern architecture with very different non-idealities — VCO
phase noise, TDC INL, strongly non-linear voltage-to-frequency transfer. Would
stretch the codebase beyond "classical" converter modelling.

**Key non-idealities:**
- VCO tuning non-linearity
- VCO phase noise (1/f² and thermal)
- TDC quantisation and DNL
- Counter rollover handling

---

## Cross-cutting building blocks needed for new ADCs

These are shared infrastructure that several of the above architectures depend
on. Building them once means subsequent architectures become compositional.

### Sample-and-hold — STATUS: PARTIAL
The SAR family currently rolls S&H into `_sample_input`. Pipelined and TI
architectures need an explicit `SampleAndHold` component with bandwidth,
acquisition time, and droop parameters.

### Residue amplifier — STATUS: NOT STARTED
Required by pipelined. Models finite open-loop gain → closed-loop gain error,
slew rate, settling time, offset. Reusable for subranging and cyclic.

### Multiplying DAC (MDAC) — STATUS: NOT STARTED
Required by pipelined. Composes S&H, sub-DAC, subtractor, and residue amp into
one pipeline-stage building block. Standalone it also implements a
configurable analog multiplier.

### Loop filter / integrator chain — STATUS: NOT STARTED
Required by sigma-delta. CIFB, CRFB, and MASH topologies.

### Decimation filter — STATUS: NOT STARTED
Required by sigma-delta. CIC + droop compensation + FIR. Standalone, also
useful as a DSP primitive.

### Glitch model — STATUS: NOT STARTED
Already listed on the DAC cross-cutting backlog. Needed by pipelined and
current-steering-based sub-DACs for SFDR-vs-frequency studies.

### Dynamic element matching (DEM) — STATUS: NOT STARTED
Already listed on the DAC cross-cutting backlog. Reusable in current-steering,
sigma-delta, and thermometer-segment architectures.

### Batch conversion API (`convert_batch`) — STATUS: DEFERRED
Not started. Full rationale and proposed contract in
`todo/parking_lot.md` §1. The sub-ADC inside a pipelined stage would benefit;
TI-ADC could distribute a batch across channels. Worth considering during
design, even if implementation is deferred.
