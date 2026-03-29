# Code Review: Completed Fixes

Issues resolved from the pyDataconverter code review (original review date: 2026-03-28).

---

## `pyDataconverter/dataconverter.py`

### 1.2 Inconsistency: ADCBase / DACBase `n_bits` check style
- **Type:** Logic inconsistency
- **Description:** `ADCBase` used `n_bits < 1`; `DACBase` used `n_bits <= 0`. Equivalent but inconsistent style.
- **Resolution:** Changed `DACBase` to `n_bits < 1` to match `ADCBase`. *(commit ec182d0)*

### 1.3 Naming: `OutputType.DIFFERENTIAL` value was `'diff'`
- **Type:** Naming
- **Description:** `InputType.DIFFERENTIAL` had value `'differential'`; `OutputType.DIFFERENTIAL` had value `'diff'`. Inconsistency risks bugs if enum values are used for serialization or display.
- **Resolution:** Changed `OutputType.DIFFERENTIAL` value to `'differential'`. Updated corresponding test. *(commit ec182d0)*

---

## `pyDataconverter/architectures/SimpleADC.py`

### 2.1 & 2.2 Bug: `int()` used instead of `floor()` in `_quantize()`
- **Type:** Bug
- **Description:** `int()` truncates toward zero, not toward negative infinity. For negative intermediate values (possible with noise/offset on differential inputs), `int(-0.3) == 0` but `floor(-0.3) == -1`. Diverges from documented FLOOR semantics.
- **Resolution:** Replaced `int(...)` with `int(np.floor(...))` for both FLOOR and SYMMETRIC modes in `_quantize()`. *(commit ec182d0)*

### 2.3 Complexity: Duplicated input validation between `ADCBase` and `SimpleADC`
- **Type:** Complexity
- **Description:** `SimpleADC.convert()` re-implemented the input-type validation from `ADCBase.convert()`. If `ADCBase` validation changed, `SimpleADC` would drift.
- **Resolution:** Extracted `_validate_input()` helper into `ADCBase`. Added `dvdt` parameter to `ADCBase.convert()` and `self._dvdt` to `ADCBase.__init__`. `SimpleADC.convert()` now calls `super().convert(vin, dvdt)`. *(commit ec182d0)*

---

## `pyDataconverter/architectures/SARADC.py`

### 4.1 Design: `_dvdt` stored as instance state duplicated in `SimpleADC` and `SARADC`
- **Type:** Design
- **Description:** Both `SimpleADC` and `SARADC` maintained their own `_dvdt = 0.0` and duplicate validation blocks in `convert()` / `convert_with_trace()`.
- **Resolution:** Moved `_dvdt` init and `dvdt` parameter to `ADCBase`. Replaced duplicate validation in `SARADC.convert()` and `convert_with_trace()` with `self._validate_input(vin)`. *(commit ec182d0)*

---

## `pyDataconverter/architectures/SimpleDAC.py`

### 5.1 Naming: `fs` and `oversample` undocumented in class docstring
- **Type:** Naming
- **Description:** `fs` and `oversample` were accepted in `__init__` but not documented. Note: `SimpleADC` has no `fs` at all — it converts single samples with no time axis. `SimpleDAC` holds `fs`/`oversample` in `__init__` because `convert_sequence()` needs them to build the ZOH time axis.
- **Resolution:** Added full docstring entries for `fs` and `oversample` in `SimpleDAC`, including the contrast with `SimpleADC`. *(commit pending)*

---

## `pyDataconverter/components/comparator.py`

### 7.1 Naming: `_last_input` stored the filter *output*, not input
- **Type:** Naming
- **Description:** The IIR bandwidth filter state variable was named `_last_input` but actually stored the filtered output (assigned after the filter equation). Misleading name.
- **Resolution:** Renamed `_last_input` → `_filtered_state` everywhere (`__init__`, `compare()`, `reset()`). Updated corresponding test. *(commit pending)*

---

## `pyDataconverter/components/capacitor.py`

### 10.1 Logic: Mismatch can produce negative capacitance
- **Type:** Logic error
- **Description:** With large mismatch (e.g., 0.5 σ), `np.random.normal()` can return ε < -1, making `c_nominal * (1 + ε)` negative. A negative capacitance is physically meaningless and would invert DAC behavior.
- **Resolution:** Clamped to zero: `max(0.0, c_nominal * (1 + ε))`. Keeps normal distribution as requested. *(commit pending)*

---

## `pyDataconverter/components/cdac.py`

### 11.1 Performance: `voltages` property called `get_voltage()` twice per code
- **Type:** Performance
- **Description:** `self.get_voltage(c)[0] - self.get_voltage(c)[1]` called `get_voltage()` twice per code. For a 16-bit DAC this is 131,072 calls instead of 65,536.
- **Resolution:** Changed to `vp, vn = self.get_voltage(c)` with a single call per code. *(commit pending)*

---

## `pyDataconverter/components/current_source.py`

### 12.1 Logic: Mismatch can produce negative current
- **Type:** Logic error
- **Description:** Same as `capacitor.py` — large mismatch values can produce `i_nominal * (1 + ε)` where `ε < -1`, resulting in negative current and nonsensical DAC outputs.
- **Resolution:** Clamped to zero: `max(0.0, i_nominal * (1 + ε))`. *(commit pending)*

---

## `pyDataconverter/utils/metrics.py`

### 13.3 Logic: `calculate_adc_static_metrics` assumed FLOOR quantization
- **Type:** Logic error
- **Description:** `ideal_first` and `ideal_last` were hardcoded for FLOOR mode. SYMMETRIC mode has a different LSB size (`v_ref / (2^N - 1)`) so the ideal transition positions differ.
- **Resolution:** Added optional `quant_mode: QuantizationMode = QuantizationMode.FLOOR` parameter. Adjusts `ideal_lsb`, `ideal_transitions`, `ideal_first`, and `ideal_last` accordingly. *(commit pending)*

### 13.4 Bug: Missing codes caused INL to accumulate via `cumsum(DNL)`
- **Type:** Bug
- **Description:** When codes were missing, `_calculate_code_edges` filled in duplicate transition values (correctly giving DNL = -1 per missing code). However, `np.cumsum(dnl)` then accumulated these -1 entries, producing an increasingly negative INL that did not reflect actual linearity error.
- **Resolution:** INL is now computed directly as `(transitions - ideal_transitions) / ideal_lsb` rather than `cumsum(DNL)`. DNL = -1 for missing codes is preserved. *(commit pending)*
