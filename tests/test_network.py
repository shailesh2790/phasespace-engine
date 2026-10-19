"""Sanity tests for the deterministic Weymouth network solver.

These are the "sanity-check against known values" step from the Phase 0
plan: the solver is the ground truth for everything downstream (training
data + the always-shown audit companion), so it must be trusted first.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phasespace.physics.network import DEMAND, SOURCE, Edge, Network, Node, solve_network
from phasespace.physics.weymouth import GasProperties, weymouth_flow


def test_single_pipe_matches_closed_form_weymouth():
    """One source, one demand node, one pipe: the solved downstream pressure
    must reproduce the closed-form Weymouth equation exactly (this is just
    Newton-Raphson finding the root of a single equation with a known
    analytic answer).
    """
    gas = GasProperties()
    p1 = 800.0
    diameter_in = 6.0
    length_mi = 5.0
    demand = 12_000_000.0  # scf/day drawn at the downstream node (~36% of pipe capacity)

    net = Network(gas=gas)
    net.add_node(Node(id=0, kind=SOURCE, pressure_psia=p1))
    net.add_node(Node(id=1, kind=DEMAND, demand_scfd=demand))
    net.add_edge(Edge(a=0, b=1, diameter_in=diameter_in, length_mi=length_mi))

    result = solve_network(net)
    assert result.converged
    p2 = result.pressures[1]

    q_check = weymouth_flow(p1, p2, diameter_in, length_mi, gas)
    assert math.isclose(q_check, demand, rel_tol=1e-3)


def test_symmetric_branch_splits_flow_evenly():
    """One source feeding two identical parallel pipes into two identical
    demand nodes: by symmetry, both branches must carry exactly half the
    total demand and reach the same downstream pressure.
    """
    gas = GasProperties()
    p1 = 900.0
    diameter_in = 8.0
    length_mi = 3.0
    demand_each = 25_000_000.0  # scf/day, ~24% of each branch's capacity

    net = Network(gas=gas)
    net.add_node(Node(id=0, kind=SOURCE, pressure_psia=p1))
    net.add_node(Node(id=1, kind=DEMAND, demand_scfd=demand_each))
    net.add_node(Node(id=2, kind=DEMAND, demand_scfd=demand_each))
    net.add_edge(Edge(a=0, b=1, diameter_in=diameter_in, length_mi=length_mi))
    net.add_edge(Edge(a=0, b=2, diameter_in=diameter_in, length_mi=length_mi))

    result = solve_network(net)
    assert result.converged
    assert math.isclose(result.pressures[1], result.pressures[2], rel_tol=1e-6)
    assert math.isclose(result.flows[(0, 1)], demand_each, rel_tol=1e-3)
    assert math.isclose(result.flows[(0, 2)], demand_each, rel_tol=1e-3)


def test_series_pipes_conserve_mass():
    """Source -> junction -> demand through two pipes in series: the flow in
    both pipes must equal the single downstream demand (mass conservation
    at the junction, which has zero demand of its own).
    """
    gas = GasProperties()
    p1 = 1000.0
    demand = 15_000_000.0  # scf/day, ~32% of the bottleneck (second) pipe's capacity

    net = Network(gas=gas)
    net.add_node(Node(id=0, kind=SOURCE, pressure_psia=p1))
    net.add_node(Node(id=1, kind=DEMAND, demand_scfd=0.0))  # pure junction
    net.add_node(Node(id=2, kind=DEMAND, demand_scfd=demand))
    net.add_edge(Edge(a=0, b=1, diameter_in=10.0, length_mi=2.0))
    net.add_edge(Edge(a=1, b=2, diameter_in=6.0, length_mi=4.0))

    result = solve_network(net)
    assert result.converged
    assert math.isclose(result.flows[(0, 1)], demand, rel_tol=1e-3)
    assert math.isclose(result.flows[(1, 2)], demand, rel_tol=1e-3)
    # Pressure must strictly decrease along the flow path.
    assert p1 > result.pressures[1] > result.pressures[2]


def test_loop_network_converges_and_conserves_mass_at_every_node():
    """A small looped network (source -> two junctions in parallel paths that
    reconnect -> demand) is the first genuinely non-trivial topology: no
    closed form, but mass balance at every node must still hold exactly.
    """
    gas = GasProperties()
    net = Network(gas=gas)
    net.add_node(Node(id=0, kind=SOURCE, pressure_psia=850.0))
    net.add_node(Node(id=1, kind=DEMAND, demand_scfd=0.0))
    net.add_node(Node(id=2, kind=DEMAND, demand_scfd=0.0))
    net.add_node(Node(id=3, kind=DEMAND, demand_scfd=30_000_000.0))  # scf/day, within both path capacities
    net.add_edge(Edge(a=0, b=1, diameter_in=8.0, length_mi=2.0))
    net.add_edge(Edge(a=0, b=2, diameter_in=6.0, length_mi=2.5))
    net.add_edge(Edge(a=1, b=3, diameter_in=8.0, length_mi=2.0))
    net.add_edge(Edge(a=2, b=3, diameter_in=6.0, length_mi=2.5))
    net.add_edge(Edge(a=1, b=2, diameter_in=4.0, length_mi=1.0))  # the loop tie

    result = solve_network(net)
    assert result.converged

    for node_id, node in net.nodes.items():
        if node.kind != DEMAND:
            continue
        balance = node.demand_scfd
        for e in net.edges:
            if e.a == node_id:
                balance += result.flows[(e.a, e.b)]
            if e.b == node_id:
                balance -= result.flows[(e.a, e.b)]
        assert math.isclose(balance, 0.0, abs_tol=100.0)  # scf/day, well below the solver's own tolerance
