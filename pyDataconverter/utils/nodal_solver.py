"""
Nodal Solver
============

Modified Nodal Analysis (MNA) solver for resistor networks.

Solves for all unknown node voltages given a set of resistors and
fixed (source) node voltages, using numpy's linear algebra solver.
"""

import numpy as np
from typing import Dict, List, Tuple


def solve_resistor_network(
        n_nodes: int,
        resistors: List[Tuple[int, int, float]],
        fixed_voltages: Dict[int, float],
) -> np.ndarray:
    """
    Solve for all node voltages in a resistor network.

    Uses Modified Nodal Analysis: builds the conductance matrix G and
    solves G · V = I, with fixed-voltage nodes handled via row substitution.

    Args:
        n_nodes: Total number of nodes (0-indexed).
        resistors: List of (node_a, node_b, resistance_ohms) tuples.
            Each resistor connects node_a to node_b with the given resistance.
        fixed_voltages: Dict mapping node index → fixed voltage (V).
            Must include at least one node (e.g. ground node = 0 V).

    Returns:
        np.ndarray of shape (n_nodes,): solved voltage at every node.
        Fixed-voltage nodes have their prescribed values; unknown nodes
        are solved by the linear system.

    Raises:
        ValueError: If the system is singular (disconnected network or
            insufficient fixed nodes).
    """
    G = np.zeros((n_nodes, n_nodes))
    I = np.zeros(n_nodes)

    # Build conductance matrix from resistors
    for na, nb, r in resistors:
        g = 1.0 / r
        G[na, na] += g
        G[nb, nb] += g
        G[na, nb] -= g
        G[nb, na] -= g

    # Apply fixed voltages via row substitution
    for node, voltage in fixed_voltages.items():
        G[node, :] = 0.0
        G[node, node] = 1.0
        I[node] = voltage

    try:
        voltages = np.linalg.solve(G, I)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Resistor network is singular — check that all nodes are "
            "reachable and at least one voltage is fixed."
        ) from exc

    return voltages
