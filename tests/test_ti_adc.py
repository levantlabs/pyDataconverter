"""
Tests for TimeInterleavedADC.

Construction validation in this file; pointwise conversion tests,
split_by_channel, and composition tests land in later commits.
"""

import unittest
import numpy as np

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.architectures.SimpleADC import SimpleADC
from pyDataconverter.architectures.TimeInterleavedADC import TimeInterleavedADC
from pyDataconverter.dataconverter import InputType


def _make_template(n_bits=6, input_type=InputType.SINGLE):
    return FlashADC(n_bits=n_bits, v_ref=1.0, input_type=input_type)


class TestTIADCConstruction(unittest.TestCase):
    """Basic constructor validation and attribute inheritance from template."""

    def test_minimal_valid_construction(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        self.assertEqual(ti.M, 4)
        self.assertEqual(ti.n_bits, 6)
        self.assertEqual(ti.v_ref, 1.0)
        self.assertEqual(ti.input_type, InputType.SINGLE)
        self.assertEqual(ti.fs, 1e9)
        self.assertEqual(len(ti.channels), 4)
        # Mismatch arrays default to length-M zero arrays
        np.testing.assert_array_equal(ti.offset, np.zeros(4))
        np.testing.assert_array_equal(ti.gain_error, np.zeros(4))
        np.testing.assert_array_equal(ti.timing_skew, np.zeros(4))
        np.testing.assert_array_equal(ti.bandwidth, np.zeros(4))

    def test_channels_less_than_two_raises(self):
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=1, sub_adc_template=_make_template(), fs=1e9)
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=0, sub_adc_template=_make_template(), fs=1e9)

    def test_channels_not_int_raises(self):
        with self.assertRaises(TypeError):
            TimeInterleavedADC(channels=4.0, sub_adc_template=_make_template(), fs=1e9)

    def test_template_not_adcbase_raises(self):
        with self.assertRaises(TypeError):
            TimeInterleavedADC(channels=4, sub_adc_template="not an ADCBase", fs=1e9)

    def test_fs_non_positive_raises(self):
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=0.0)
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=-1e9)

    def test_scalar_offset_interpreted_as_stddev(self):
        # Scalar mismatch => random draw of length M with std approximately equal
        # to the scalar value (for reasonable seeds; just check the shape and
        # that nonzero values are present).
        ti = TimeInterleavedADC(channels=64, sub_adc_template=_make_template(),
                                fs=1e9, offset=1e-3, seed=42)
        self.assertEqual(ti.offset.shape, (64,))
        # Not all zero
        self.assertGreater(np.std(ti.offset), 0.0)

    def test_explicit_array_offset(self):
        explicit = np.array([1e-3, -1e-3, 0.5e-3, -0.5e-3])
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(),
                                fs=1e9, offset=explicit)
        np.testing.assert_array_equal(ti.offset, explicit)

    def test_explicit_array_wrong_length_raises(self):
        explicit = np.array([1e-3, -1e-3, 0.5e-3])  # length 3, M=4
        with self.assertRaises(ValueError):
            TimeInterleavedADC(channels=4, sub_adc_template=_make_template(),
                               fs=1e9, offset=explicit)

    def test_same_seed_gives_same_mismatch(self):
        ti1 = TimeInterleavedADC(channels=8, sub_adc_template=_make_template(),
                                 fs=1e9, offset=1e-3, gain_error=0.001, seed=123)
        ti2 = TimeInterleavedADC(channels=8, sub_adc_template=_make_template(),
                                 fs=1e9, offset=1e-3, gain_error=0.001, seed=123)
        np.testing.assert_array_equal(ti1.offset, ti2.offset)
        np.testing.assert_array_equal(ti1.gain_error, ti2.gain_error)

    def test_forbidden_kwargs_raise(self):
        # n_bits, v_ref, input_type must be inherited from the template, not
        # overridden at the TI-ADC level. Catching these at construction
        # prevents silent mismatches between wrapper and template.
        for kw in ("n_bits", "v_ref", "input_type"):
            with self.assertRaises(TypeError, msg=f"{kw} must raise"):
                TimeInterleavedADC(
                    channels=4,
                    sub_adc_template=_make_template(),
                    fs=1e9,
                    **{kw: 8 if kw == "n_bits" else (1.0 if kw == "v_ref" else InputType.SINGLE)},
                )

    def test_channels_are_deep_copies(self):
        # Each channel should be an independent copy so mutating one
        # (e.g. changing an attribute) does not affect the others.
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        self.assertEqual(len(ti.channels), 4)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertIsNot(ti.channels[i], ti.channels[j])
        # Not the same object as the template either
        tmpl = _make_template()
        ti2 = TimeInterleavedADC(channels=4, sub_adc_template=tmpl, fs=1e9)
        for ch in ti2.channels:
            self.assertIsNot(ch, tmpl)


class TestTIADCPointwise(unittest.TestCase):
    """Pointwise convert() path — the core data flow."""

    def test_ideal_ti_adc_matches_template(self):
        """Zero mismatch => TI-ADC output identical to template output sample-by-sample."""
        template = SimpleADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(n_bits=8),
                                fs=1e9)
        # Use a NEW template for the reference since each sub-ADC is a deep
        # copy and we want a fresh one to compare against (avoids any state).
        ref_template = _make_template(n_bits=8)
        for v in np.linspace(0, 1.0, 101):
            vf = float(v)
            expected = ref_template.convert(vf)
            actual = ti.convert(vf)
            self.assertEqual(actual, expected,
                             f"mismatch at v={v}: expected {expected}, got {actual}")

    def test_channel_counter_rotates_on_each_call(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        expected_sequence = [0, 1, 2, 3, 0, 1, 2, 3]
        for expected_k in expected_sequence:
            ti.convert(0.5)
            self.assertEqual(ti.last_channel, expected_k)

    def test_offset_mismatch_shifts_output(self):
        """Explicit offset array produces predictable per-channel code shifts."""
        # 4-bit ADC has LSB = 1/15 ~ 0.067. Use an offset array whose values are
        # large enough to flip several codes.
        explicit = np.array([0.0, 0.1, -0.1, 0.05])
        ti = TimeInterleavedADC(
            channels=4,
            sub_adc_template=SimpleADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE),
            fs=1e9,
            offset=explicit,
        )
        ref = SimpleADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)
        v = 0.5
        for k in range(4):
            expected = ref.convert(v + explicit[k])
            actual = ti.convert(v)
            self.assertEqual(actual, expected,
                             f"channel {k}: expected {expected}, got {actual}")

    def test_gain_error_scales_output(self):
        """Explicit gain_error array scales the input per channel before quantisation."""
        explicit = np.array([0.0, 0.1, -0.1, 0.05])  # fractional gain errors
        ti = TimeInterleavedADC(
            channels=4,
            sub_adc_template=SimpleADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE),
            fs=1e9,
            gain_error=explicit,
        )
        ref = SimpleADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)
        v = 0.5
        for k in range(4):
            expected = ref.convert(v * (1 + explicit[k]))
            actual = ti.convert(v)
            self.assertEqual(actual, expected,
                             f"channel {k}: expected {expected}, got {actual}")

    def test_timing_skew_applies_dvdt_correction(self):
        """Explicit timing_skew array times dvdt produces an input-referred shift."""
        explicit = np.array([0.0, 1e-12, -1e-12, 2e-12])  # per-channel skews
        ti = TimeInterleavedADC(
            channels=4,
            sub_adc_template=SimpleADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE),
            fs=1e9,
            timing_skew=explicit,
        )
        ref = SimpleADC(n_bits=8, v_ref=1.0, input_type=InputType.SINGLE)
        v = 0.5
        dvdt = 1e9  # 1 V/ns
        for k in range(4):
            expected = ref.convert(v + dvdt * explicit[k])
            actual = ti.convert(v, dvdt=dvdt)
            self.assertEqual(actual, expected,
                             f"channel {k}: expected {expected}, got {actual}")

    def test_all_three_mismatches_compose(self):
        """offset + gain + skew applied together should sum as v*(1+g) + offset + dvdt*skew."""
        explicit_off = np.array([0.05, 0.0, 0.0, 0.0])
        explicit_gain = np.array([0.0, 0.05, 0.0, 0.0])
        explicit_skew = np.array([0.0, 0.0, 1e-11, 0.0])
        ti = TimeInterleavedADC(
            channels=4,
            sub_adc_template=SimpleADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE),
            fs=1e9,
            offset=explicit_off,
            gain_error=explicit_gain,
            timing_skew=explicit_skew,
        )
        ref = SimpleADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE)
        v = 0.4
        dvdt = 5e8
        for k in range(4):
            v_eff = v * (1 + explicit_gain[k]) + explicit_off[k] + dvdt * explicit_skew[k]
            expected = ref.convert(v_eff)
            actual = ti.convert(v, dvdt=dvdt)
            self.assertEqual(actual, expected,
                             f"channel {k}: v_eff={v_eff}, expected {expected}, got {actual}")


class TestTIADCHelpers(unittest.TestCase):
    """last_channel property, reset(), split_by_channel helper."""

    def test_last_channel_none_before_any_convert(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        self.assertIsNone(ti.last_channel)

    def test_last_channel_advances_with_convert(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        expected_sequence = [0, 1, 2, 3, 0, 1]
        for k in expected_sequence:
            ti.convert(0.5)
            self.assertEqual(ti.last_channel, k)

    def test_reset_restores_counter_and_last_channel(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        ti.convert(0.5)
        ti.convert(0.5)
        self.assertEqual(ti.last_channel, 1)
        ti.reset()
        self.assertIsNone(ti.last_channel)
        ti.convert(0.5)
        self.assertEqual(ti.last_channel, 0)

    def test_split_by_channel_reshape_and_content(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        codes = np.array([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        result = ti.split_by_channel(codes)
        self.assertEqual(result.shape, (4, 3))
        # Row k should be codes[k::M]
        np.testing.assert_array_equal(result[0], np.array([10, 20, 30]))
        np.testing.assert_array_equal(result[1], np.array([11, 21, 31]))
        np.testing.assert_array_equal(result[2], np.array([12, 22, 32]))
        np.testing.assert_array_equal(result[3], np.array([13, 23, 33]))

    def test_split_by_channel_wrong_length_raises(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        codes = np.array([1, 2, 3, 4, 5, 6, 7])  # not a multiple of 4
        with self.assertRaises(ValueError):
            ti.split_by_channel(codes)

    def test_split_by_channel_non_1d_raises(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        codes = np.zeros((4, 4), dtype=int)
        with self.assertRaises(ValueError):
            ti.split_by_channel(codes)


class TestTIADCWaveform(unittest.TestCase):
    """convert_waveform path: bandwidth mismatch + pointwise-vs-waveform parity."""

    def test_ideal_waveform_matches_pointwise(self):
        """With zero mismatch, convert_waveform and a pointwise loop agree."""
        ti_a = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(n_bits=8),
                                   fs=1e9)
        ti_b = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(n_bits=8),
                                   fs=1e9)
        t = np.linspace(0, 1e-6, 128)
        v = 0.5 + 0.4 * np.sin(2 * np.pi * 2e6 * t)
        dvdt = np.gradient(v, t)
        expected = np.array([ti_a.convert(float(v[i]), dvdt=float(dvdt[i]))
                             for i in range(len(v))], dtype=int)
        actual = ti_b.convert_waveform(v, t)
        np.testing.assert_array_equal(actual, expected)

    def test_pointwise_raises_when_bandwidth_nonzero(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(),
                                fs=1e9, bandwidth=np.array([1e8, 2e8, 1.5e8, 1e8]))
        with self.assertRaises(RuntimeError) as ctx:
            ti.convert(0.5)
        self.assertIn("convert_waveform", str(ctx.exception))

    def test_waveform_runs_with_bandwidth_active(self):
        """Smoke test: nonzero bandwidth exercises the LPF branch without raising."""
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(n_bits=10),
                                fs=1e9, bandwidth=np.array([1e8, 2e8, 1.5e8, 1e8]))
        t = np.linspace(0, 1e-6, 64)
        v = 0.5 + 0.3 * np.sin(2 * np.pi * 5e7 * t)
        codes = ti.convert_waveform(v, t)
        self.assertEqual(len(codes), 64)
        self.assertEqual(codes.dtype.kind, "i")
        # Every code should be in the valid sub-ADC range
        self.assertTrue(np.all(codes >= 0))
        self.assertTrue(np.all(codes <= 2**10 - 1))

    def test_waveform_advances_channel_counter(self):
        ti = TimeInterleavedADC(channels=4, sub_adc_template=_make_template(), fs=1e9)
        t = np.linspace(0, 1e-6, 8)
        v = 0.5 * np.ones_like(t)
        _ = ti.convert_waveform(v, t)
        # After 8 samples on M=4, last_channel should be (8-1) % 4 = 3.
        self.assertEqual(ti.last_channel, 3)


class TestTIADCHierarchical(unittest.TestCase):
    """hierarchical classmethod for multi-level interleaving trees."""

    def test_two_level_tree_has_correct_structure(self):
        """channels_per_level=[4, 2] → outer M=4, each channel is an M=2 TI-ADC."""
        ti = TimeInterleavedADC.hierarchical(
            channels_per_level=[4, 2],
            sub_adc_template=_make_template(n_bits=6),
            fs=8e9,
        )
        self.assertIsInstance(ti, TimeInterleavedADC)
        self.assertEqual(ti.M, 4)
        for k in range(4):
            inner = ti.channels[k]
            self.assertIsInstance(inner, TimeInterleavedADC,
                                  f"channel {k} should be a TimeInterleavedADC")
            self.assertEqual(inner.M, 2)
            self.assertEqual(inner.fs, 8e9 / 4)

    def test_ideal_hierarchy_matches_template(self):
        """Zero mismatches at every level → hierarchy produces template output."""
        ti = TimeInterleavedADC.hierarchical(
            channels_per_level=[4, 2],
            sub_adc_template=_make_template(n_bits=8),
            fs=8e9,
        )
        ref = _make_template(n_bits=8)
        for v in np.linspace(0, 1.0, 50):
            self.assertEqual(ti.convert(float(v)), ref.convert(float(v)))

    def test_per_level_mismatch_lists_are_applied(self):
        """per-level offset_std lists populate offsets at the right level."""
        ti = TimeInterleavedADC.hierarchical(
            channels_per_level=[4, 2],
            sub_adc_template=_make_template(n_bits=8),
            fs=8e9,
            offset_std_per_level=[1e-3, 0.5e-3],
            seed=42,
        )
        self.assertEqual(ti.offset.shape, (4,))
        self.assertGreater(np.std(ti.offset), 0.0)
        # Each inner TI-ADC should also have offsets (drawn from the inner stddev)
        for inner in ti.channels:
            self.assertEqual(inner.offset.shape, (2,))
            self.assertGreater(np.std(inner.offset), 0.0)

    def test_channels_per_level_must_be_nonempty(self):
        with self.assertRaises(ValueError):
            TimeInterleavedADC.hierarchical(
                channels_per_level=[],
                sub_adc_template=_make_template(),
                fs=1e9,
            )

    def test_channels_per_level_entry_lt_two_raises(self):
        with self.assertRaises(ValueError):
            TimeInterleavedADC.hierarchical(
                channels_per_level=[4, 1],
                sub_adc_template=_make_template(),
                fs=1e9,
            )

    def test_per_level_list_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            TimeInterleavedADC.hierarchical(
                channels_per_level=[4, 2],
                sub_adc_template=_make_template(),
                fs=1e9,
                offset_std_per_level=[1e-3],  # length 1, expected 2
            )


class TestTIADCComposition(unittest.TestCase):
    """TI-ADC composes with other architectures (its whole reason for being)."""

    def test_ti_adc_as_pipelined_backend(self):
        """A TimeInterleavedADC can serve as a PipelinedADC backend."""
        # Importing here to avoid circular-ish cost at module load
        from pyDataconverter.architectures.PipelinedADC import (
            PipelineStage, PipelinedADC)
        from pyDataconverter.components.residue_amplifier import ResidueAmplifier
        from pyDataconverter.architectures.SimpleDAC import SimpleDAC
        from pyDataconverter.dataconverter import OutputType

        # Small stage0 + a TI-ADC backend built from a FlashADC template.
        stage_sub_adc = FlashADC(n_bits=3, v_ref=1.0,
                                  input_type=InputType.SINGLE, n_comparators=8)
        stage_sub_dac = SimpleDAC(n_bits=3, n_levels=9, v_ref=1.0,
                                   output_type=OutputType.SINGLE)
        stage_amp = ResidueAmplifier(gain=4.0, settling_tau=0.0)
        stage = PipelineStage(sub_adc=stage_sub_adc, sub_dac=stage_sub_dac,
                              residue_amp=stage_amp, fs=1e9, code_offset=0)

        backend_template = FlashADC(n_bits=8, v_ref=1.0,
                                     input_type=InputType.SINGLE)
        ti_backend = TimeInterleavedADC(channels=4,
                                         sub_adc_template=backend_template,
                                         fs=1e9)

        adc = PipelinedADC(n_bits=10, v_ref=1.0, input_type=InputType.SINGLE,
                            stages=[stage], backend=ti_backend,
                            backend_H=255, backend_code_offset=0, fs=1e9)

        # Smoke test: a few inputs produce valid codes without raising.
        for v in (0.1, 0.25, 0.5, 0.75, 0.9):
            code = adc.convert(v)
            self.assertIsInstance(code, int)
            self.assertGreaterEqual(code, 0)
            self.assertLessEqual(code, 2 ** 10 - 1)

    def test_ti_adc_inside_ti_adc_manual_nesting(self):
        """Manual nested construction also builds a valid hierarchy (same as classmethod)."""
        inner = TimeInterleavedADC(channels=2, sub_adc_template=_make_template(n_bits=6),
                                    fs=1e9)
        outer = TimeInterleavedADC(channels=4, sub_adc_template=inner, fs=4e9)
        self.assertEqual(outer.M, 4)
        # Each outer channel is a TimeInterleavedADC (deep copy of inner)
        for ch in outer.channels:
            self.assertIsInstance(ch, TimeInterleavedADC)
            self.assertEqual(ch.M, 2)
        # An ideal nested TI-ADC with zero mismatches should equal the template
        ref = _make_template(n_bits=6)
        for v in np.linspace(0, 1.0, 20):
            self.assertEqual(outer.convert(float(v)), ref.convert(float(v)))
