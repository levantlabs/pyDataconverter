"""
SAR ADC Module
==============

This module provides a Successive Approximation Register (SAR) ADC model with
configurable non-idealities.

Classes:
    SARADC: SAR ADC implementation inheriting from ADCBase.

Version History:
---------------
1.0.0 (2026-03-25):
    - Initial release
    - Structural bit-cycling model with binary-weighted C-DAC component
    - Single-ended and differential input support via SingleEndedCDAC /
      DifferentialCDAC
    - Input-referred non-idealities: sampling noise (kT/C), offset, gain
      error, aperture jitter
    - Comparator non-idealities via DifferentialComparator
    - convert_with_trace() for cycle-by-cycle inspection

Notes:
------
Algorithm
---------
For each conversion the SAR ADC performs N comparison cycles (one per bit):

  1. Sample-and-Hold
     The input is sampled once.  Non-idealities (gain error, offset, kT/C
     noise, aperture jitter) are applied at this stage and held fixed for the
     duration of the bit-cycling.

  2. Bit loop (MSB → LSB)
     For bit k (k = 0 is MSB):
       a. Set bit k high  →  trial_code
       b. Query the C-DAC: (v_refp, v_refn) = cdac.get_voltage(trial_code)
       c. Reset the comparator (clears latch / hysteresis state)
       d. Compare: compare(v_sampled, 0.0, v_refp, v_refn)
          effective_diff = (v_sampled − v_refp) − (0 − v_refn)
                         = v_sampled − (v_refp − v_refn)
       e. Retain bit k iff effective_diff > 0

  3. Output the final register value as the digital code.

Input conventions
-----------------
Single-ended:
  v_sampled is the raw input voltage in [0, v_ref].
  cdac.get_voltage() returns (v_dac, 0.0).
  Comparator sees v_sampled − v_dac.

Differential:
  v_sampled is the differential voltage v_pos − v_neg in [−v_ref/2, +v_ref/2].
  cdac.get_voltage() returns (v_dacp, v_dacn).
  Comparator sees (v_diff − v_dacp) + v_dacn = v_diff − v_dac_diff.
  Non-idealities (offset, noise, etc.) are applied to the differential
  voltage before the bit loop.

Quantisation
------------
The binary search naturally implements FLOOR quantisation:
  code = floor(vin * 2^N / v_ref)   (single-ended)
  code = floor((v_diff + v_ref/2) * 2^N / v_ref)   (differential)

Component dependencies
----------------------
Comparator : pyDataconverter.components.comparator.DifferentialComparator
C-DAC      : pyDataconverter.components.cdac.SingleEndedCDAC  (SINGLE)
             pyDataconverter.components.cdac.DifferentialCDAC (DIFFERENTIAL)
"""

from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from pyDataconverter.dataconverter import ADCBase, InputType
from pyDataconverter.components.comparator import ComparatorBase, DifferentialComparator
from pyDataconverter.components.cdac import CDACBase, SingleEndedCDAC, DifferentialCDAC


class SARADC(ADCBase):
    """
    SAR ADC with C-DAC component and configurable non-idealities.

    Models the actual bit-cycling algorithm using a single comparator and a
    binary-weighted C-DAC.  A custom C-DAC can be injected to study
    non-standard capacitor arrays.

    Attributes:
        Inherits all attributes from ADCBase, plus:
        comparator (ComparatorBase): Single comparator instance used for all N
            evaluations per conversion.
        cdac (CDACBase): Capacitive DAC component.  Auto-created from
            input_type if not supplied.
        noise_rms (float): Input-referred RMS sampling noise / kT/C (V).
        offset (float): Input-referred DC offset (V).
        gain_error (float): Fractional gain error (dimensionless).
        t_jitter (float): RMS aperture jitter (s).
    """

    def __init__(
        self,
        n_bits: int,
        v_ref: float = 1.0,
        input_type: InputType = InputType.SINGLE,
        comparator_type: Type[ComparatorBase] = DifferentialComparator,
        comparator_params: Optional[dict] = None,
        cdac: Optional[CDACBase] = None,
        cap_mismatch: float = 0.0,
        noise_rms: float = 0.0,
        offset: float = 0.0,
        gain_error: float = 0.0,
        t_jitter: float = 0.0,
    ):
        """
        Initialize the SAR ADC.

        Args:
            n_bits: Resolution in bits.
            v_ref: Reference voltage (V).
            input_type: SINGLE or DIFFERENTIAL.
            comparator_type: Comparator class to instantiate.
            comparator_params: Keyword arguments forwarded to the comparator
                constructor (e.g. noise_rms, offset, hysteresis).
            cdac: Pre-constructed CDACBase instance.  If provided, cap_mismatch
                is ignored and the supplied C-DAC is used directly.  Its n_bits
                and v_ref must match those of the ADC.  If None, a
                SingleEndedCDAC or DifferentialCDAC is auto-created based on
                input_type.
            cap_mismatch: Standard deviation of multiplicative capacitor
                mismatch (dimensionless, e.g. 0.001 = 0.1 %).  Ignored when
                cdac is provided.
            noise_rms: Input-referred RMS sampling noise (V).  Represents kT/C
                or other front-end noise fixed at the sampling instant.
                Applied once per conversion before the bit loop.
            offset: Input-referred DC offset voltage (V).
            gain_error: Fractional gain error (dimensionless, e.g. 0.01 = +1 %).
                Scales the sampled input as v * (1 + gain_error).
            t_jitter: RMS aperture jitter (s).  Only active when dvdt != 0 is
                passed to convert().
        """
        super().__init__(n_bits, v_ref, input_type)

        if noise_rms < 0:
            raise ValueError("noise_rms must be >= 0")
        if t_jitter < 0:
            raise ValueError("t_jitter must be >= 0")
        if cap_mismatch < 0:
            raise ValueError("cap_mismatch must be >= 0")

        self.noise_rms  = noise_rms
        self.offset     = offset
        self.gain_error = gain_error
        self.t_jitter   = t_jitter
        # self._dvdt is inherited from ADCBase

        # Comparator
        if comparator_params is None:
            comparator_params = {}
        self.comparator = comparator_type(**comparator_params)

        # C-DAC
        if cdac is not None:
            if not isinstance(cdac, CDACBase):
                raise TypeError("cdac must be a CDACBase instance")
            if cdac.n_bits != n_bits:
                raise ValueError(
                    f"cdac.n_bits={cdac.n_bits} does not match n_bits={n_bits}")
            if cdac.v_ref != v_ref:
                raise ValueError(
                    f"cdac.v_ref={cdac.v_ref} does not match v_ref={v_ref}")
            self.cdac = cdac
        else:
            if input_type == InputType.SINGLE:
                self.cdac = SingleEndedCDAC(n_bits, v_ref, cap_mismatch=cap_mismatch)
            else:
                self.cdac = DifferentialCDAC(n_bits, v_ref, cap_mismatch=cap_mismatch)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def dac_voltages(self) -> np.ndarray:
        """
        Effective DAC threshold for every code (v_refp − v_refn).

        Analogous to FlashADC.reference_voltages.  Delegates to cdac.voltages.

        Returns:
            np.ndarray: Shape (2^n_bits,).
        """
        return self.cdac.voltages

    def convert(
        self,
        vin: Union[float, Tuple[float, float]],
        dvdt: float = 0.0,
    ) -> int:
        """
        Convert one analog sample to a digital code.

        Args:
            vin: Voltage (single-ended) or (v_pos, v_neg) tuple (differential).
            dvdt: Signal slope at the sampling instant (V/s).  Only used when
                t_jitter > 0.

        Returns:
            int: Output code in [0, 2^n_bits − 1].
        """
        self._validate_input(vin)
        self._dvdt = float(dvdt)
        return self._convert_input(vin)

    def convert_with_trace(
        self,
        vin: Union[float, Tuple[float, float]],
        dvdt: float = 0.0,
    ) -> Dict:
        """
        Convert and return the cycle-by-cycle SAR trace.

        Useful for visualising or debugging the bit-cycling operation.

        Args:
            vin: Voltage (single-ended) or (v_pos, v_neg) tuple (differential).
            dvdt: Signal slope for aperture jitter modelling (V/s).

        Returns:
            dict with keys:
                'code' (int):
                    Final output code in [0, 2^n_bits − 1].
                'sampled_voltage' (float):
                    Held input after non-idealities.  Single-ended: voltage in
                    [0, v_ref].  Differential: v_diff = v_pos − v_neg in
                    [−v_ref/2, +v_ref/2].
                'dac_voltages' (list[float]):
                    N effective trial thresholds (v_refp − v_refn), one per
                    bit cycle.
                'bit_decisions' (list[int]):
                    N comparator outputs (0 or 1), one per bit cycle.
                'register_states' (list[int]):
                    N + 1 register values — initial 0 followed by the register
                    value after each bit decision.
        """
        self._validate_input(vin)
        self._dvdt = float(dvdt)
        v_sampled = self._sample_input(vin)
        code, dac_vs, decisions, reg_states = self._run_sar(v_sampled)

        return {
            'code':            code,
            'sampled_voltage': v_sampled,
            'dac_voltages':    dac_vs,
            'bit_decisions':   decisions,
            'register_states': reg_states,
        }

    def reset(self):
        """Reset comparator state (hysteresis history, bandwidth filter)."""
        self.comparator.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_input(self, analog_input: Union[float, Tuple[float, float]]) -> float:
        """
        Sample and hold the input, applying input-referred non-idealities.

        Returns a scalar in all cases:
            Single-ended  → raw voltage v ∈ ℝ (naturally clamped by SAR).
            Differential  → differential voltage v_diff = v_pos − v_neg ∈ ℝ.

        Non-idealities applied in order:
            1. Gain error (multiplicative)
            2. Offset (additive)
            3. Sampling noise / kT/C (additive, one draw per conversion)
            4. Aperture jitter (additive, proportional to _dvdt)

        Returns:
            float: Scalar sampled voltage.
        """
        if self.input_type == InputType.SINGLE:
            v = float(analog_input)
        else:
            v_pos, v_neg = analog_input
            v = v_pos - v_neg

        if self.gain_error:
            v = v * (1.0 + self.gain_error)
        if self.offset:
            v = v + self.offset
        if self.noise_rms:
            v = v + np.random.normal(0.0, self.noise_rms)
        if self.t_jitter and self._dvdt:
            v = v + self._dvdt * np.random.normal(0.0, self.t_jitter)

        return float(v)

    def _run_sar(
        self, v_sampled: float
    ) -> Tuple[int, List[float], List[int], List[int]]:
        """
        Execute the SAR bit-cycling algorithm.

        For each bit k (MSB first):
            1. Tentatively set bit k  →  trial_code
            2. Query the C-DAC: (v_refp, v_refn) = cdac.get_voltage(trial_code)
            3. Reset comparator (clears hysteresis / latch state)
            4. Decision = compare(v_sampled, 0.0, v_refp, v_refn)
               effective_diff = v_sampled − (v_refp − v_refn)
            5. Retain bit k iff decision == 1

        Args:
            v_sampled: Scalar held input (raw for single-ended, differential
                voltage for differential input).

        Returns:
            Tuple of (code, dac_voltages, bit_decisions, register_states).
        """
        register = 0
        dac_voltages:    List[float] = []
        bit_decisions:   List[int]   = []
        register_states: List[int]   = [0]

        for k in range(self.n_bits):
            trial_code       = register | (1 << (self.n_bits - 1 - k))
            v_refp, v_refn   = self.cdac.get_voltage(trial_code)

            self.comparator.reset()
            decision = self.comparator.compare(v_sampled, 0.0, v_refp, v_refn)

            dac_voltages.append(v_refp - v_refn)
            bit_decisions.append(decision)

            if decision:
                register = trial_code

            register_states.append(register)

        return register, dac_voltages, bit_decisions, register_states

    def _convert_input(self, analog_input: Union[float, Tuple[float, float]]) -> int:
        """
        Sample the input and run the SAR algorithm.

        Args:
            analog_input: Pre-validated voltage or (v_pos, v_neg) tuple.

        Returns:
            int: Output code in [0, 2^n_bits − 1].
        """
        v_sampled = self._sample_input(analog_input)
        code, _, _, _ = self._run_sar(v_sampled)
        return code

    def __repr__(self) -> str:
        parts = [
            f"n_bits={self.n_bits}",
            f"v_ref={self.v_ref}",
            f"input_type={self.input_type.name}",
            f"cdac={self.cdac!r}",
        ]
        if self.noise_rms:
            parts.append(f"noise_rms={self.noise_rms}")
        if self.offset:
            parts.append(f"offset={self.offset}")
        if self.gain_error:
            parts.append(f"gain_error={self.gain_error}")
        if self.t_jitter:
            parts.append(f"t_jitter={self.t_jitter}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class MultibitSARADC(SARADC):
    """
    SAR ADC that resolves multiple bits per cycle using a flash sub-ADC.

    Each cycle uses a (2^bits_per_cycle − 1)-comparator flash to determine
    the next bits_per_cycle bits simultaneously.  With N total bits and M
    bits per cycle, the conversion requires ceil(N / M) cycles.

    The flash sub-ADC is idealised (no offset, no noise) unless a comparator
    with non-idealities is injected via comparator_params.

    Args:
        n_bits: Total resolution.
        v_ref: Reference voltage (V).
        bits_per_cycle: Number of bits resolved per SAR cycle (default 2).
        All other args forwarded to SARADC.__init__.
    """

    def __init__(self, n_bits: int, v_ref: float = 1.0,
                 bits_per_cycle: int = 2, **kwargs):
        if bits_per_cycle < 1 or bits_per_cycle > n_bits:
            raise ValueError("bits_per_cycle must be in [1, n_bits]")
        self.bits_per_cycle = bits_per_cycle
        super().__init__(n_bits, v_ref, **kwargs)

    def _run_sar(
        self, v_sampled: float
    ) -> Tuple[int, List[float], List[int], List[int]]:
        """Override: use flash sub-ADC to resolve bits_per_cycle bits at once."""
        import math
        register        = 0
        dac_voltages:    List[float] = []
        bit_decisions:   List[int]   = []
        register_states: List[int]   = [0]

        n_cycles = math.ceil(self.n_bits / self.bits_per_cycle)

        for cycle in range(n_cycles):
            msb_position = self.n_bits - cycle * self.bits_per_cycle
            lsb_position = max(0, msb_position - self.bits_per_cycle)
            n_sub = msb_position - lsb_position  # bits to resolve this cycle

            n_levels = 2 ** n_sub
            best_code = 0
            for sub_code in range(n_levels - 1, -1, -1):
                trial_code = register | (sub_code << lsb_position)
                v_refp, v_refn = self.cdac.get_voltage(trial_code)
                self.comparator.reset()
                decision = self.comparator.compare(v_sampled, 0.0, v_refp, v_refn)
                dac_voltages.append(v_refp - v_refn)
                bit_decisions.append(decision)
                if decision:
                    best_code = sub_code
                    break

            register = register | (best_code << lsb_position)
            register_states.append(register)

        return register, dac_voltages, bit_decisions, register_states

    def __repr__(self) -> str:
        parts = [
            f"n_bits={self.n_bits}",
            f"v_ref={self.v_ref}",
            f"bits_per_cycle={self.bits_per_cycle}",
            f"input_type={self.input_type.name}",
        ]
        if self.noise_rms:
            parts.append(f"noise_rms={self.noise_rms}")
        if self.offset:
            parts.append(f"offset={self.offset}")
        if self.gain_error:
            parts.append(f"gain_error={self.gain_error}")
        if self.t_jitter:
            parts.append(f"t_jitter={self.t_jitter}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Ideal 4-bit SAR, single-ended ---
    adc = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)
    v_in = np.linspace(0, 1, 1000)
    codes = [adc.convert(v) for v in v_in]

    plt.figure(figsize=(10, 4))
    plt.plot(v_in, codes, 'b.', markersize=1)
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Output Code')
    plt.title('Ideal 4-bit SAR ADC Transfer Function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Conversion trace ---
    trace = adc.convert_with_trace(0.37)
    print("SAR conversion trace for vin = 0.37 V:")
    print(f"  Sampled voltage : {trace['sampled_voltage']:.4f} V")
    for k, (v_dac, bit, reg) in enumerate(
        zip(trace['dac_voltages'], trace['bit_decisions'],
            trace['register_states'][1:])
    ):
        print(f"  Bit {adc.n_bits - 1 - k}: "
              f"v_dac={v_dac:.4f} V  decision={bit}  "
              f"register={reg:0{adc.n_bits}b}")
    print(f"  Final code: {trace['code']}")

    # --- Differential SAR ---
    adc_diff = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
    v_diff = np.linspace(-0.5, 0.5, 1000)
    codes_diff = [adc_diff.convert((v / 2, -v / 2)) for v in v_diff]

    plt.figure(figsize=(10, 4))
    plt.plot(v_diff, codes_diff, 'r.', markersize=1)
    plt.xlabel('Differential Input (V)')
    plt.ylabel('Output Code')
    plt.title('Ideal 4-bit Differential SAR ADC Transfer Function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
