"""Tests for voltage-mode DAC models and the nodal solver."""
import numpy as np
import pytest


class TestNodalSolver:
    def test_voltage_divider(self):
        """Two equal resistors between V_ref and GND gives V_ref/2 at midpoint."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        # Nodes: 0=GND (fixed 0V), 1=midpoint (unknown), 2=V_ref (fixed 1V)
        # R between node 0 and 1, R between node 1 and 2
        nodes = 3
        resistors = [(0, 1, 1e3), (1, 2, 1e3)]  # (node_a, node_b, R_ohms)
        fixed = {0: 0.0, 2: 1.0}                 # node: voltage
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert abs(voltages[1] - 0.5) < 1e-9

    def test_unequal_divider(self):
        """1kΩ / 3kΩ divider gives 0.75 V at midpoint."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        nodes = 3
        resistors = [(0, 1, 3e3), (1, 2, 1e3)]
        fixed = {0: 0.0, 2: 1.0}
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert abs(voltages[1] - 0.75) < 1e-9

    def test_three_node_network(self):
        """Star network: three 1kΩ resistors meeting at a centre node."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        # Nodes: 0=fixed 0V, 1=fixed 1V, 2=fixed 0.5V, 3=centre (unknown)
        nodes = 4
        resistors = [(0, 3, 1e3), (1, 3, 1e3), (2, 3, 1e3)]
        fixed = {0: 0.0, 1: 1.0, 2: 0.5}
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert abs(voltages[3] - 0.5) < 1e-9

    def test_returns_all_node_voltages(self):
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        nodes = 4
        resistors = [(0, 1, 1e3), (1, 2, 1e3), (2, 3, 1e3)]
        fixed = {0: 0.0, 3: 1.0}
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert len(voltages) == nodes
        assert abs(voltages[1] - 1/3) < 1e-9
        assert abs(voltages[2] - 2/3) < 1e-9


class TestResistorStringDAC:
    def test_construction(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0, r_unit=1e3)
        assert dac.n_bits == 4
        assert dac.v_ref == 1.0

    def test_ideal_output_zero(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0)
        assert abs(dac.convert(0)) < 1e-9

    def test_ideal_output_full_scale(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0)
        max_code = 2**4 - 1
        expected = max_code / 2**4 * 1.0
        assert abs(dac.convert(max_code) - expected) < 1e-9

    def test_ideal_monotone(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=6, v_ref=1.0)
        voltages = [dac.convert(c) for c in range(2**6)]
        assert all(b >= a for a, b in zip(voltages, voltages[1:]))

    def test_ideal_dnl_inl_small(self):
        """Ideal (no mismatch) DAC has uniform steps and near-zero INL.

        Note: DACBase defines lsb = v_ref/(2^N-1) while the resistor string
        produces steps of v_ref/2^N.  This causes a systematic offset in DNL
        of ~1/2^N (≈0.016 LSB for N=6) when measured by calculate_dac_static_metrics.
        The INL, however, is zero because all steps are perfectly uniform.
        """
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        dac = ResistorStringDAC(n_bits=6, v_ref=1.0)
        m = calculate_dac_static_metrics(dac)
        # DNL is systematically ~1/2^N due to LSB definition mismatch; all
        # steps are identical so the spread around the mean is < 0.001 LSB.
        dnl = m['DNL']
        assert (dnl.max() - dnl.min()) < 0.001, (
            "Ideal DAC should have identical step sizes (zero DNL spread)"
        )
        assert m['MaxINL'] < 0.001

    def test_mismatch_increases_inl(self):
        """Non-zero mismatch produces larger INL than ideal."""
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        dac_ideal    = ResistorStringDAC(n_bits=6, v_ref=1.0, r_mismatch=0.0)
        dac_mismatch = ResistorStringDAC(n_bits=6, v_ref=1.0, r_mismatch=0.05,
                                         seed=42)
        m_ideal    = calculate_dac_static_metrics(dac_ideal)
        m_mismatch = calculate_dac_static_metrics(dac_mismatch)
        assert m_mismatch['MaxINL'] > m_ideal['MaxINL']

    def test_repr(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0, r_unit=1e3, r_mismatch=0.01)
        r = repr(dac)
        assert 'ResistorStringDAC' in r
        assert 'n_bits=4' in r

    def test_out_of_range_code_raises(self):
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac = ResistorStringDAC(n_bits=4, v_ref=1.0)
        with pytest.raises((ValueError, TypeError)):
            dac.convert(2**4)

    def test_seed_reproducible(self):
        """Same seed gives identical r_values."""
        from pyDataconverter.architectures.ResistorStringDAC import ResistorStringDAC
        dac1 = ResistorStringDAC(n_bits=4, v_ref=1.0, r_mismatch=0.05, seed=7)
        dac2 = ResistorStringDAC(n_bits=4, v_ref=1.0, r_mismatch=0.05, seed=7)
        np.testing.assert_array_equal(dac1.r_values, dac2.r_values)


class TestR2RDAC:
    def test_construction(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=4, v_ref=1.0)
        assert dac.n_bits == 4

    def test_ideal_output_zero(self):
        """Code 0: all bits low → output 0 V."""
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        assert abs(dac.convert(0)) < 1e-9

    def test_ideal_output_msb_only(self):
        """MSB only (code = 2^(N-1)) → V_ref/2."""
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        assert abs(dac.convert(2**5) - 0.5) < 1e-6

    def test_ideal_monotone(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        voltages = [dac.convert(c) for c in range(2**6)]
        assert all(b >= a - 1e-9 for a, b in zip(voltages, voltages[1:]))

    def test_ideal_dnl_inl_small(self):
        """Ideal (no mismatch) R-2R DAC has uniform steps and near-zero INL.

        Note: DACBase defines lsb = v_ref/(2^N-1) while the R-2R network
        produces steps of v_ref/2^N.  This causes a systematic DNL offset of
        ~1/2^N (≈0.016 LSB for N=6).  All steps are identical, so the spread
        around the mean is zero, and INL is zero.
        """
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        dac = R2RDAC(n_bits=6, v_ref=1.0)
        m = calculate_dac_static_metrics(dac)
        # All steps are identical → DNL spread is zero
        dnl = m['DNL']
        assert (dnl.max() - dnl.min()) < 0.001, (
            "Ideal R-2R DAC should have identical step sizes (zero DNL spread)"
        )
        assert m['MaxINL'] < 0.001

    def test_r_mismatch_independent_r_and_2r(self):
        """Separate R and 2R mismatch parameters can be set."""
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=6, v_ref=1.0, r_mismatch=0.02, r2_mismatch=0.03,
                     seed=0)
        assert dac.r_mismatch == 0.02
        assert dac.r2_mismatch == 0.03

    def test_mismatch_increases_inl(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        from pyDataconverter.utils.metrics import calculate_dac_static_metrics
        dac_ideal    = R2RDAC(n_bits=6, v_ref=1.0)
        dac_mismatch = R2RDAC(n_bits=6, v_ref=1.0, r_mismatch=0.05,
                              r2_mismatch=0.05, seed=99)
        m_ideal    = calculate_dac_static_metrics(dac_ideal)
        m_mismatch = calculate_dac_static_metrics(dac_mismatch)
        assert m_mismatch['MaxINL'] > m_ideal['MaxINL']

    def test_seed_reproducible(self):
        """Same seed gives identical r_values and r2_values."""
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac1 = R2RDAC(n_bits=4, v_ref=1.0, r_mismatch=0.05, r2_mismatch=0.03, seed=42)
        dac2 = R2RDAC(n_bits=4, v_ref=1.0, r_mismatch=0.05, r2_mismatch=0.03, seed=42)
        np.testing.assert_array_equal(dac1.r_values, dac2.r_values)
        np.testing.assert_array_equal(dac1.r2_values, dac2.r2_values)

    def test_repr(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=4, v_ref=1.0, r_unit=1e3, r_mismatch=0.01, r2_mismatch=0.02)
        r = repr(dac)
        assert 'R2RDAC' in r
        assert 'n_bits=4' in r

    def test_out_of_range_code_raises(self):
        from pyDataconverter.architectures.R2RDAC import R2RDAC
        dac = R2RDAC(n_bits=4, v_ref=1.0)
        with pytest.raises((ValueError, TypeError)):
            dac.convert(2**4)
