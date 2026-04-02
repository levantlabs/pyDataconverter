"""Tests for the nodal solver (MNA resistor network solver)."""
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

    def test_fixed_nodes_correct(self):
        """Fixed nodes must appear in output with their prescribed values."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        nodes = 3
        resistors = [(0, 1, 1e3), (1, 2, 1e3)]
        fixed = {0: 0.0, 2: 1.0}
        voltages = solve_resistor_network(nodes, resistors, fixed)
        assert abs(voltages[0] - 0.0) < 1e-12
        assert abs(voltages[2] - 1.0) < 1e-12

    def test_singular_raises_value_error(self):
        """A disconnected network with no fixed voltage raises ValueError."""
        from pyDataconverter.utils.nodal_solver import solve_resistor_network
        nodes = 3
        resistors = [(0, 1, 1e3)]  # node 2 is isolated
        fixed = {}  # no fixed voltages — singular
        with pytest.raises(ValueError):
            solve_resistor_network(nodes, resistors, fixed)
