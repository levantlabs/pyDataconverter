"""
Flash ADC Module
===============

This module provides a Flash ADC implementation with configurable non-idealities
and encoder types.

Classes:
    EncoderType: Enum defining the thermometer-to-binary encoding strategy.
    FlashADC: Flash ADC implementation inheriting from ADCBase.

Version History:
---------------
1.0.0 (2024-02-07):
    - Initial release
    - Basic Flash ADC implementation
    - Integration with Comparator component
    - Support for offset, noise, and resistor mismatch
1.1.0 (2026-03-23):
    - Added EncoderType enum (COUNT_ONES, XOR)
    - Added differential input support
    - Refactored encoder into _encode() method

Notes:
------
The Flash ADC model includes:
    - Configurable number of bits
    - Individual comparator non-idealities (via Comparator class)
    - Resistor ladder mismatch
    - Reference noise modeling
    - Choice of thermometer-to-binary encoder

Quantisation:
    FlashADC is a *structural* model — its output comes from a bank of
    comparators firing against uniformly-spaced ReferenceLadder taps.
    Quantisation is FLOOR-by-construction (LSB = v_ref / 2^N) and is not
    configurable.  The ``QuantizationMode`` enum from
    ``pyDataconverter.dataconverter`` does not apply here; to compute
    zero-mean (SYMMETRIC) metrics on a FlashADC's output, pass
    ``quant_mode=QuantizationMode.SYMMETRIC`` to the relevant function in
    ``pyDataconverter.utils.metrics``.

TODO:
    - Bandwidth modeling: Comparator.compare() accepts time_step for bandwidth
      limiting, but FlashADC does not yet pass it.
"""

from enum import Enum
from typing import Optional, Type, Union, Tuple
import numpy as np
from pyDataconverter.dataconverter import ADCBase, InputType
from pyDataconverter.components.comparator import ComparatorBase, DifferentialComparator
from pyDataconverter.components.reference import ReferenceBase, ReferenceLadder, ArbitraryReference


class EncoderType(Enum):
    """
    Thermometer-to-binary encoding strategy for Flash ADC.

    COUNT_ONES:
        Counts the total number of asserted comparator outputs.
        Equivalent to a Wallace tree counter in hardware.
        Robust to bubble errors: each bubble reduces the code by 1
        rather than producing a large sparkle error.
        Formula: code = sum(thermometer)

    XOR:
        XORs adjacent thermometer bits to produce a one-hot intermediate X,
        then maps to binary via OR gates:
            bit k of output = OR of X[i] for all i where bit k is set in (i+1)
        Mirrors the standard hardware ROM encoder.
        With a valid thermometer code the result equals COUNT_ONES.
        With bubble errors, multiple XOR bits fire and the OR logic produces
        sparkle codes (large, unpredictable errors).
    """
    COUNT_ONES = 'count_ones'
    XOR        = 'xor'


class FlashADC(ADCBase):
    """
    Flash ADC implementation with configurable non-idealities and encoder.

    Reference ladder convention
    ---------------------------
    Single-ended mode: the reference ladder spans ``[0, v_ref]`` and each
    tap ``t[i]`` is the threshold at comparator ``i`` directly.

    Differential mode: the ladder spans the narrower single-ended range
    ``[-v_ref/4, +v_ref/4]``.  Each comparator ``i`` is driven by a
    symmetric tap pair — ``v_refp = t[i]`` (ascending) and
    ``v_refn = t[n-1-i]`` (descending) — so the effective differential
    threshold is ``v_refp - v_refn = 2 * t[i]``, which spans the full
    ``[-v_ref/2, +v_ref/2]`` differential input range (``v_ref``
    peak-to-peak).  Accordingly, the ``reference_voltages`` property
    returns the ×2-scaled effective thresholds so callers inspecting the
    ladder see the voltages the comparators actually apply.  See the
    block comments in ``__init__`` and ``_convert_input`` for the
    derivation.

    Attributes:
        Inherits all attributes from ADCBase, plus:
        n_comparators (int): Number of comparators. Defaults to
            2^n_bits - 1; override via the n_comparators constructor kwarg
            for arbitrary (including even) counts.
        comparators (list): Per-comparator Comparator instances.
        reference (ReferenceBase): Voltage reference generator.
        encoder_type (EncoderType): Thermometer-to-binary encoding strategy.
    """

    def __init__(self,
                 n_bits: int,
                 v_ref: float = 1.0,
                 input_type: InputType = InputType.DIFFERENTIAL,
                 comparator_type: Type[ComparatorBase] = DifferentialComparator,
                 comparator_params: Optional[dict] = None,
                 offset_std: float = 0.0,
                 reference: Optional[ReferenceBase] = None,
                 reference_noise: float = 0.0,
                 resistor_mismatch: float = 0.0,
                 encoder_type: EncoderType = EncoderType.COUNT_ONES,
                 n_comparators: Optional[int] = None,
                 seed: Optional[int] = None):
        """
        Initialize Flash ADC.

        Args:
            n_bits: Resolution in bits.
            v_ref: Reference voltage.
            input_type: SINGLE or DIFFERENTIAL.
            comparator_type: Comparator class to instantiate for each stage.
            comparator_params: Shared keyword arguments passed to every
                comparator (e.g. noise_rms, hysteresis). The 'offset' key
                is reserved — per-comparator offsets are set via offset_std.
            offset_std: Standard deviation of comparator input-referred offsets
                (V). Drawn once at construction and held fixed.
            reference: Voltage reference instance (ReferenceBase subclass).
                If provided, reference_noise and resistor_mismatch are ignored
                and the reference component fully controls the thresholds.
                If None, a default ReferenceLadder is built from v_ref,
                input_type, resistor_mismatch, and reference_noise.
            reference_noise: RMS dynamic noise for the default ReferenceLadder
                (ignored when reference is provided).
            resistor_mismatch: Resistor mismatch std for the default
                ReferenceLadder (ignored when reference is provided, and
                also ignored when n_comparators overrides the default
                2**n_bits - 1 — the ArbitraryReference used in that path
                does not model resistor mismatch).
            encoder_type: Thermometer-to-binary encoding strategy.
            n_comparators: Number of comparators. If None, defaults to 2**n_bits - 1.
                Can be any positive integer to override the default derivation.
            seed: Optional integer seed for the construction-time random
                draws (per-comparator offsets and the default ladder's
                resistor mismatch).  ``None`` (default) uses OS entropy
                (non-deterministic); an integer makes both draws
                reproducible.  When a user-provided ``reference`` is
                passed, this seed only controls the offset draw — the
                reference's own randomness is the caller's responsibility.
                Per-conversion noise (sampling noise, jitter, comparator
                noise, reference noise) continues to use ``np.random``
                global state.
        """
        super().__init__(n_bits, v_ref, input_type)
        self.seed = seed

        if not isinstance(encoder_type, EncoderType):
            raise TypeError("encoder_type must be an EncoderType enum")

        # Resolve n_comparators: explicit override wins, else derived from n_bits.
        if n_comparators is None:
            self.n_comparators = 2 ** n_bits - 1
        else:
            if not isinstance(n_comparators, int) or isinstance(n_comparators, bool):
                raise TypeError(
                    f"n_comparators must be an integer, got {type(n_comparators).__name__}")
            if n_comparators < 1:
                raise ValueError(f"n_comparators must be >= 1, got {n_comparators}")
            self.n_comparators = n_comparators

        self.encoder_type = encoder_type

        # Reference generator
        if reference is not None:
            if not isinstance(reference, ReferenceBase):
                raise TypeError("reference must be a ReferenceBase instance")
            if reference.n_references != self.n_comparators:
                raise ValueError(
                    f"reference has {reference.n_references} taps but "
                    f"this FlashADC has n_comparators={self.n_comparators}")
            self.reference = reference
        else:
            # Build a default ladder whose length matches n_comparators.
            # Power-of-2 case: use ReferenceLadder (preserves existing
            # behaviour for backward compatibility, including resistor
            # mismatch modelling). Non-power-of-2 case: use ArbitraryReference
            # with linearly-spaced bin-midpoint thresholds — this is the
            # appropriate spacing for an arbitrary count, but note that the
            # two formulas produce numerically different thresholds even at
            # n_comparators == 2**n_bits - 1, which is why the guard must be
            # exact (do not mix the two paths for the same count).
            v_min = -v_ref / 4 if input_type == InputType.DIFFERENTIAL else 0.0
            v_max =  v_ref / 4 if input_type == InputType.DIFFERENTIAL else v_ref
            if self.n_comparators == 2 ** n_bits - 1:
                self.reference = ReferenceLadder(n_bits, v_min, v_max,
                                                 resistor_mismatch=resistor_mismatch,
                                                 noise_rms=reference_noise,
                                                 seed=seed)
            else:
                lsb = (v_max - v_min) / self.n_comparators
                thresholds = v_min + lsb * (np.arange(self.n_comparators) + 0.5)
                self.reference = ArbitraryReference(thresholds, noise_rms=reference_noise)

        # Comparator bank
        if comparator_params is None:
            comparator_params = {}

        # Per-comparator offset draw uses a separate RNG seeded from `seed`
        # so it is independent of the ladder's resistor-mismatch draw but
        # deterministic when seed is set.  Stream the seed through
        # SeedSequence to avoid correlation between the two draws.
        if offset_std > 0:
            ss = np.random.SeedSequence(seed)
            offset_rng = np.random.default_rng(ss.spawn(2)[1])
            offsets = offset_rng.normal(0, offset_std, self.n_comparators)
        else:
            offsets = np.zeros(self.n_comparators)

        self.comparators = []
        for i in range(self.n_comparators):
            params = comparator_params.copy()
            params['offset'] = offsets[i]
            self.comparators.append(comparator_type(**params))

        self._last_v_sampled = 0.0

    @property
    def reference_voltages(self) -> np.ndarray:
        """
        Static reference voltages (no noise).

        In differential mode the underlying ladder stores single-ended tap
        voltages in ``[-v_ref/4, +v_ref/4]``, but the effective differential
        threshold at comparator ``i`` is ``v_refp[i] - v_refn[i] = 2 *
        tap_voltage[i]`` (see the block comment in ``__init__``). This
        accessor returns the ×2-scaled values so callers inspecting the
        ladder see the same thresholds the comparators actually apply.
        """
        return self.reference.voltages*2 if self.input_type == InputType.DIFFERENTIAL else self.reference.voltages

    def _encode(self, thermometer: np.ndarray) -> int:
        """
        Convert thermometer code to binary output code.

        COUNT_ONES: code = sum of asserted bits.

        XOR: XOR adjacent thermometer bits (appending a virtual 0 at the top)
             to produce a one-hot intermediate X, then for each output bit k
             OR together all X[i] where bit k is set in (i+1).
             With a bubble, multiple X[i] bits fire and the OR produces a
             sparkle code.

        Args:
            thermometer: Boolean/int array of length n_comparators.

        Returns:
            int: Binary output code in [0, 2^n_bits - 1].
        """
        if self.encoder_type == EncoderType.COUNT_ONES:
            return int(np.sum(thermometer))

        # XOR encoder — vectorized
        extended = np.concatenate([thermometer, [0]])
        X = thermometer ^ extended[1:]          # 2^N - 1 one-hot-like bits

        active = np.where(X)[0]
        if len(active) == 0:
            return 0
        # values[i] = i + 1 for each active XOR bit
        values = active + 1
        # OR together all active values to produce the binary code
        code = int(np.bitwise_or.reduce(values))
        return code

    def _convert_input(self, analog_input) -> int:
        """
        Run the comparator bank and encode the thermometer code.

        Args:
            analog_input: Single voltage (single-ended) or (v_pos, v_neg)
                          tuple (differential).

        Returns:
            int: Output code in [0, n_comparators] (which equals
                2^n_bits - 1 in the default configuration).
        """
        # Cache the effective single-ended input for metastability reporting
        if self.input_type == InputType.DIFFERENTIAL:
            v_pos_in, v_neg_in = analog_input
            self._last_v_sampled = float(v_pos_in) - float(v_neg_in)
        else:
            self._last_v_sampled = float(analog_input)

        comp_refs = self.reference.get_voltages()
        n = self.n_comparators

        thermometer = np.empty(n, dtype=int)
        if self.input_type == InputType.DIFFERENTIAL:
            v_pos, v_neg = analog_input
            # v_refp[i] comes from the bottom of the ladder (ascending index);
            # v_refn[i] comes from the opposite (top) end (descending index).
            # Effective threshold[i] = v_refp[i] − v_refn[i] = 2 × comp_refs[i],
            # which spans [−v_ref/2, +v_ref/2] because the ladder spans [−v_ref/4, +v_ref/4].
            for i, comp in enumerate(self.comparators):
                thermometer[i] = comp.compare(v_pos, v_neg, comp_refs[i], comp_refs[n - 1 - i])
        else:
            vin = float(analog_input)
            # Single-ended: reference injected as v_refp; v_neg and v_refn are 0.
            for i, (comp, ref) in enumerate(zip(self.comparators, comp_refs)):
                thermometer[i] = comp.compare(vin, 0.0, ref, 0.0)

        return int(np.clip(self._encode(thermometer), 0, self.n_comparators))

    def reset(self):
        """Reset all comparator states (hysteresis history, bandwidth filter)."""
        for comp in self.comparators:
            comp.reset()

    def last_conversion_time(self) -> float:
        """
        Regeneration time of the slowest comparator from the most recent
        convert() call, in seconds.

        Aggregates ``last_regen_time`` across the comparator bank via
        ``max()``. Returns 0.0 when every comparator has ``tau_regen=0``
        (metastability modelling disabled).
        """
        return max((c.last_regen_time for c in self.comparators), default=0.0)

    def last_metastable_sign(self) -> int:
        """
        Sign of the initial-condition error a residue amp would see at the
        start of its settling window, for the most recent convert() call.

        Returns:
            +1 if the sub-ADC's nearest threshold is above the last input,
            -1 if the nearest threshold is below the last input,
             0 if metastability modelling is disabled (tau_regen=0 on every
               comparator) — in which case downstream pipelined ADCs treat
               the stage as operating ideally.

        Internally uses ``self.reference.voltages`` (not ``get_voltages()``)
        so the sign reflects the static ladder geometry, independent of any
        dynamic reference noise realisation.
        """
        # Aggregate flag: if no comparator has regen enabled, return 0.
        if not any(c.tau_regen > 0 for c in self.comparators):
            return 0
        # Use the property (not self.reference.voltages) so differential-input
        # mode sees the 2x-scaled effective thresholds in [-v_ref/2, +v_ref/2]
        # rather than the raw ladder voltages in [-v_ref/4, +v_ref/4]. The
        # property remains noise-free (it reads .voltages under the hood).
        thresholds = self.reference_voltages
        if len(thresholds) == 0:
            return 0
        diffs = np.abs(thresholds - self._last_v_sampled)
        i_nearest = int(np.argmin(diffs))
        # +1 if threshold > v_sampled, else -1 (strictly greater — ties go to -1)
        return 1 if (thresholds[i_nearest] - self._last_v_sampled) > 0 else -1

    def __repr__(self) -> str:
        parts = [
            f"n_bits={self.n_bits}",
            f"v_ref={self.v_ref}",
            f"input_type={self.input_type.name}",
            f"encoder_type={self.encoder_type.name}",
            f"reference={self.reference!r}",
        ]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return f"{self.__class__.__name__}({', '.join(parts)})"





if __name__ == "__main__":
    """Example usage of the Flash ADC"""
    import matplotlib.pyplot as plt

    # Create a 3-bit Flash ADC with some non-idealities
    adc = FlashADC(
        n_bits=3,
        v_ref=1.0,
        input_type=InputType.SINGLE,
        comparator_params={
            'noise_rms': 0.001,  # 1mV RMS noise
            'hysteresis': 0.002  # 2mV hysteresis
        },
        offset_std=0.002,  # 2mV offset standard deviation
        resistor_mismatch=0.01  # 1% resistor mismatch
    )

    # Test with ramp input
    v_in = np.linspace(0, 1, 1000)
    codes = [adc.convert(v) for v in v_in]

    # Plot transfer function
    plt.figure(figsize=(10, 6))
    plt.plot(v_in, codes, 'b.', markersize=1)
    plt.grid(True)
    plt.xlabel('Input Voltage (V)')
    plt.ylabel('Output Code')
    plt.title('Flash ADC Transfer Function')

    # Add ideal transfer function
    ideal_codes = np.floor(v_in * 2 ** adc.n_bits).astype(int)
    ideal_codes = np.clip(ideal_codes, 0, 2 ** adc.n_bits - 1)
    plt.plot(v_in, ideal_codes, 'r--', alpha=0.5, label='Ideal')
    plt.legend()

    plt.show()

    # Test with sine wave input
    t = np.linspace(0, 1e-3, 1000)  # 1ms
    f = 1e3  # 1kHz
    v_in = 0.5 + 0.4 * np.sin(2 * np.pi * f * t)  # 0.5V offset, 0.4V amplitude
    codes = [adc.convert(v) for v in v_in]

    # Plot time domain behavior
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t * 1e3, v_in)
    plt.grid(True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Voltage (V)')
    plt.title('Flash ADC Time Domain Response')

    plt.subplot(2, 1, 2)
    plt.plot(t * 1e3, codes)
    plt.grid(True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Output Code')

    plt.tight_layout()
    plt.show()

    # Create a 3-bit Flash ADC with some non-idealities
    adc = FlashADC(
        n_bits=4,
        v_ref=1.0,
        input_type=InputType.SINGLE,
        comparator_params={
            'offset': 0.001,  # 1mV offset
            'noise_rms': 0.0005  # 0.5mV noise
        }
    )





    # Static visualization
    from pyDataconverter.utils.visualizations.visualize_FlashADC import visualize_flash_adc, animate_flash_adc
    visualize_flash_adc(adc, input_voltage=0.4)

    # Animation example
    t = np.linspace(0, 2 * np.pi, 50)
    input_voltages = 0.5 + 0.4 * np.sin(t)  # Sine wave centered at 0.5V
    animate_flash_adc(adc, input_voltages, interval=0.1)
