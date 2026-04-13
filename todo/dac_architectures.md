DAC Architecture Roadmap
========================

Date: 2026-03-25 — last updated 2026-04-13.

Current state (as of 2026-04-13): R-2R, resistor string, thermometer / current-
steering (binary, thermometer, segmented), segmented-resistor, and switched-cap
CDAC are all implemented. Sigma-Delta and PWM DACs are still open. See per-item
STATUS blocks below.

---

## 1. R-2R Ladder DAC  [HIGH] — STATUS: DONE (`architectures/R2RDAC.py`)

**What it is:** Binary-weighted resistor network.  Each bit drives a node in the
ladder; the output voltage is the Thevenin sum.  The SimpleDAC already models the
ideal transfer function; this would add the physical resistor network explicitly.

**Why model it:** Pedagogically fundamental.  Resistor mismatch is the dominant
non-ideality and is easy to reason about.

**Key non-idealities to add:**
- Resistor mismatch (R and 2R independently mismatched)
- Finite output impedance (varies with code — affects settling into a load)
- Loading effect from finite termination resistance

**Outputs:** Voltage (single-ended or differential)

**Relationship to SimpleDAC:** Could be a subclass or a separate architecture
that exposes the resistor network explicitly.

---

## 2. String DAC (Resistor String)  [HIGH] — STATUS: DONE (`architectures/ResistorStringDAC.py`)

**What it is:** A chain of 2^N equal resistors between V_ref and GND.  A MUX
selects the tap corresponding to the input code.

**Why model it:** Guaranteed DNL ≤ 0.5 LSB by construction (monotone by design),
widely used for low-to-medium resolution (DAC in ADC reference, trim DACs).
Very different character from binary-weighted.

**Key non-idealities to add:**
- Resistor mismatch (each segment independently)
- MUX switch resistance (code-dependent output impedance)
- INL from accumulated mismatch (random walk)

**Outputs:** Voltage

---

## 3. Thermometer (Unary) DAC  [HIGH] — STATUS: DONE (folded into `CurrentSteeringDAC` via `n_therm_bits == n_bits`; decoder in `components/decoder.py`)

**What it is:** 2^N − 1 identical unit elements (resistors, caps, or current
sources).  For code k, exactly k elements are switched on.

**Why model it:** Guaranteed monotonicity, DNL ≤ 1 LSB, no major-carry glitch.
Basis for the MSB segment of segmented DACs.

**Key non-idealities to add:**
- Unit element mismatch (each element has an independent error)
- Element selection order (thermometric vs. randomised / DEM — dynamic element
  matching could be a separate flag)
- Switching glitch (code-dependent, but smaller than binary because only one
  element changes at a time for unit-step inputs)

**Outputs:** Voltage or current (parameterisable)

---

## 4. Current-Steering DAC  [HIGH] — STATUS: DONE (`architectures/CurrentSteeringDAC.py` with `CurrentSourceArray`; static mismatch modelled; glitch energy / DEM still on the cross-cutting backlog)

**What it is:** Array of current sources (binary-weighted, thermometer, or
segmented) steered between differential output nodes by differential switches.

**Why model it:** The dominant high-speed DAC architecture in RF/comms (12–16 bit,
1 GSPS+).  Has a rich set of dynamic non-idealities not present in resistive DACs.

**Key non-idealities to add:**
- Current source mismatch (static DNL/INL)
- Output impedance (finite, code-dependent — degrades SFDR at high frequency)
- Switch timing skew (causes glitch energy)
- Glitch energy / glitch impulse area (parameterisable per transition type)
- Substrate/supply coupling (optional, advanced)
- Dynamic element matching (DEM) mode flag

**Outputs:** Differential current (can be converted to voltage via load resistor
parameter)

**Metrics of interest:** SFDR vs. frequency, glitch energy, settling time

---

## 5. Segmented DAC  [MEDIUM-HIGH] — STATUS: DONE (`architectures/SegmentedResistorDAC.py` for the coarse-string + R-2R variant; the segmented-current-steering variant is also covered by `CurrentSteeringDAC`)

**What it is:** MSBs decoded to thermometer, LSBs remain binary.  Typical split
is upper 4–6 bits thermometer, lower bits binary.  Balances area and linearity.

**Why model it:** The most common practical architecture for 10–16 bit DACs.
Combines the monotonicity of thermometer with the area efficiency of binary.
Capturing the "segment boundary" glitch (occurs at major thermometer transitions)
is a key modelling goal.

**Key non-idealities to add:**
- Mismatch within thermometer segment
- Mismatch within binary segment
- Segment boundary glitch (larger at MSB-group rollovers)
- Encoder / decoder errors at segment boundary

**Parameters:** n_therm_bits (number of MSBs decoded to thermometer), n_binary_bits

---

## 6. Switched-Capacitor DAC  [MEDIUM] — STATUS: DONE (`components/cdac.py`: `SingleEndedCDAC`, `DifferentialCDAC`, `RedundantSARCDAC`, `SplitCapCDAC`, `SegmentedCDAC`; used by the SAR family and also instantiable standalone)

**What it is:** Binary-weighted or thermometer capacitor array switched between
V_ref and GND.  Already partially modelled via CDAC for the SAR ADC.

**Why model it:** Natural extension of the existing CDAC component; relevant for
pipeline ADC sub-DACs, audio DACs, and charge-redistribution DACs.

**Key non-idealities to add:**
- Capacitor mismatch (already in CDAC — reuse)
- kT/C noise (sampling noise per capacitor)
- Charge injection from switches (code-dependent offset)
- Incomplete settling (bandwidth × capacitance limit)

**Relationship to existing code:** Wrap or extend SingleEndedCDAC /
DifferentialCDAC rather than starting from scratch.

---

## 7. Sigma-Delta DAC  [MEDIUM] — STATUS: OPEN (not started; shares infrastructure with the planned Sigma-Delta ADC in `todo/adc_architectures.md`)

**What it is:** Oversampling + noise-shaping modulator driving a low-resolution
(often 1-bit) inner DAC.  Reconstruction filter removes out-of-band quantisation
noise.

**Why model it:** Dominant architecture for high-resolution audio and measurement
DACs.  Very different design space from Nyquist DACs.

**Key components to model:**
- Oversampling ratio (OSR)
- Noise-shaping filter order and type (CIFB, CRFB, etc.) — could start with
  first- and second-order loops
- Inner DAC element mismatch (critical for multi-bit inner DAC)
- Reconstruction / interpolation filter (CIC + FIR cascade)
- Idle tones (pattern noise at DC)

**Note:** This is the most complex entry on this list.  Suggest modelling
first-order Σ∆ with a 1-bit inner DAC as a starting point.

---

## 8. PWM DAC  [LOW] — STATUS: OPEN (not started)

**What it is:** Digital input sets the duty cycle of a square wave; a low-pass
filter extracts the DC average.

**Why model it:** Common in embedded/microcontroller applications (motor control,
LED dimming).  Simple but worth having for completeness.

**Key non-idealities to add:**
- Finite rise/fall time of the switch
- Ripple at the filter output (depends on filter order and switching frequency)
- Dead-time / minimum pulse width (limits usable code range)

---

## Cross-cutting component work (pre-requisites)

### Decoder hierarchy — STATUS: DONE (`components/decoder.py`)
- `DecoderBase`, `BinaryDecoder`, `ThermometerDecoder`, `SegmentedDecoder`
  all implemented. Symmetric counterpart to `EncoderType` on the ADC side.

### Unit current source hierarchy — STATUS: PARTIAL (`components/current_source.py`)
- `UnitCurrentSourceBase`, `IdealCurrentSource`, `CurrentSourceArray` implemented.
- Future subclasses (`CascodeCurrentSource`, `RegulatedCascodeCurrentSource`) still open.

### Unit capacitor hierarchy — STATUS: DONE (`components/capacitor.py`)
- `UnitCapacitorBase` and `IdealCapacitor` implemented.
- `LeakyCapacitor` / `NonLinearCapacitor` subclasses still open.

---

## Cross-cutting work needed for all new DACs

1. **Glitch model** — A general GlitchModel component that can be attached to any
   DAC and injects a code-transition-dependent impulse into the output waveform.
   Parameterised by glitch area (V·s) and time constant.

2. **Output impedance model** — Code-dependent Thevenin impedance so that DAC
   output settling into a load can be simulated.

3. **Dynamic Element Matching (DEM)** — A mixin or strategy class that randomises
   which unit elements are selected for a given code.  Converts mismatch-induced
   harmonic distortion into white-noise-like error floor.

4. **Visualizer for each architecture** — Following the pattern of
   visualize_FlashADC and visualize_SARADC, each new DAC should have a
   visualizer showing its unit-element structure and switching state.

5. **Example file** — Following the pattern of sar_adc_example.py and
   flash_adc_example.py.

6. **Tests** — Unit and integration tests following the existing pytest structure.

7. **API documentation** — Entries in docs/api_reference.md and
   docs/documentation_log.json.
