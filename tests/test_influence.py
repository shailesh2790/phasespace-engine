"""Tests for "if this well changes, who else is affected".

Three kinds of check.

Against independent physics: on a single pipe the answer has a closed form --
the Weymouth equation at the new flow -- so the differenced solves can be
checked against arithmetic rather than against themselves. Interference has to
be mutual, and two wells on separate pipes from a fixed-pressure source cannot
influence each other at all.

Against the reporting rule: the gauge threshold must only ever shrink the
affected set, the perturbed node must not be counted as news about itself, and
a non-converged solve must be declared rather than differenced silently.

And against the caller's expectations: the input network is not modified, and
inputs that cannot mean anything are refused rather than answered.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.components import GasPipe
from phasespace.sim.fluids import GasProperties
from phasespace.sim.influence import (
    DEFAULT_GAUGE_RESOLUTION_PSI,
    influence_matrix,
    influence_of_demand,
)
from phasespace.sim.solver import FLOW, GAS, PRESSURE, Network, Node, solve

GAS_65 = GasProperties(gravity=0.65, temperature_r=520.0)


def gathering_field():
    """A separator, a trunk line, a header, and three wells on it.

    W-far sits behind W-B, so it is the node that should feel W-B's changes
    most and the separator's least.
    """
    net = Network(phase=GAS, fluid=GAS_65)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0, name="Separator"))
    net.add_node(Node(1, FLOW, name="Header"))
    net.add_node(Node(2, FLOW, withdrawal=8e6, name="W-A"))
    net.add_node(Node(3, FLOW, withdrawal=8e6, name="W-B"))
    net.add_node(Node(4, FLOW, withdrawal=2e6, name="W-far"))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=8.0, length_mi=5.0))
    net.add_branch(GasPipe(a=1, b=2, diameter_in=6.0, length_mi=3.0))
    net.add_branch(GasPipe(a=1, b=3, diameter_in=6.0, length_mi=3.0))
    net.add_branch(GasPipe(a=3, b=4, diameter_in=4.0, length_mi=1.0))
    return net


def independent_wells():
    """Two wells each on their own pipe straight off a fixed-pressure source.

    There is no shared pipe, so there is no mechanism by which one can affect
    the other. Any coupling reported here would be a bug in the differencing.
    """
    net = Network(phase=GAS, fluid=GAS_65)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0, name="Separator"))
    net.add_node(Node(1, FLOW, withdrawal=10e6, name="W-1"))
    net.add_node(Node(2, FLOW, withdrawal=10e6, name="W-2"))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=6.0, length_mi=4.0))
    net.add_branch(GasPipe(a=0, b=2, diameter_in=6.0, length_mi=4.0))
    return net


GRID_SIDE = 6


def looped_grid(side=GRID_SIDE, diameter_in=6.0):
    """A looped gathering grid fed from one source at its centre.

    Every node has two to four ways home, which is what makes it absorb a
    local change locally. The pipe size matters: at 4 inches this grid is
    tight enough that one well's change moves the whole field, and at 6 it is
    loose enough that nothing else notices.
    """
    net = Network(phase=GAS, fluid=GAS_65)
    centre = (side // 2) * side + (side // 2)
    for i in range(side * side):
        if i == centre:
            net.add_node(Node(i, PRESSURE, pressure_psia=900.0, name="Sep"))
        else:
            net.add_node(Node(i, FLOW, withdrawal=1.5e6, name="N%d" % i))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                net.add_branch(GasPipe(a=i, b=i + 1,
                                       diameter_in=diameter_in, length_mi=1.0))
            if r + 1 < side:
                net.add_branch(GasPipe(a=i, b=i + side,
                                       diameter_in=diameter_in, length_mi=1.0))
    return net


# ------------------------------------------------- against the closed form --


def test_a_single_pipe_delta_matches_the_weymouth_equation():
    """One source, one well, one pipe. The pressure the well sits at after a
    20 percent increase is not a matter of opinion -- it is the Weymouth
    equation at the new rate, and the reported delta must reproduce it."""
    net = Network(phase=GAS, fluid=GAS_65)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0, name="Sep"))
    net.add_node(Node(1, FLOW, withdrawal=12e6, name="W-1"))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0))

    report = influence_of_demand(net, 1, change_fraction=0.20)
    entry = report.affected[0]
    assert entry.is_source

    # Solve the same pipe at the raised rate independently.
    check = Network(phase=GAS, fluid=GAS_65)
    check.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    check.add_node(Node(1, FLOW, withdrawal=12e6 * 1.20))
    check.add_branch(GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0))
    independent = solve(check)

    assert entry.changed_psia == pytest.approx(independent.pressures[1], rel=1e-6)
    # Taking more out of a pipe lowers the pressure at the far end.
    assert entry.delta_psi < 0.0


def test_taking_more_gas_lowers_pressure_and_taking_less_raises_it():
    net = gathering_field()
    up = influence_of_demand(net, 3, change_fraction=1.0)
    down = influence_of_demand(net, 3, change_fraction=-0.5)

    assert all(e.delta_psi < 0.0 for e in up.affected), [
        (e.name, e.delta_psi) for e in up.affected]
    assert all(e.delta_psi > 0.0 for e in down.affected), [
        (e.name, e.delta_psi) for e in down.affected]


def test_wells_on_separate_pipes_do_not_influence_each_other():
    """No shared pipe means no coupling, and the fixed-pressure source holds
    the only common node. Checked below gauge resolution, at the solver's own
    tolerance, because the physical answer here is exactly zero."""
    net = independent_wells()
    report = influence_of_demand(net, 1, change_fraction=1.0,
                                 gauge_resolution_psi=1e-6)
    assert [e.name for e in report.others] == []
    assert report.worst is None
    assert report.sparsity == 0.0


def test_interference_is_mutual():
    """If W-A's rate moves W-B, then W-B's rate moves W-A. They share the
    header and the trunk line; influence through a shared restriction cannot
    run one way only."""
    net = gathering_field()
    a_on_b = influence_of_demand(net, 2, change_fraction=2.0,
                                 gauge_resolution_psi=0.5)
    b_on_a = influence_of_demand(net, 3, change_fraction=2.0,
                                 gauge_resolution_psi=0.5)

    assert 3 in [e.node for e in a_on_b.others]
    assert 2 in [e.node for e in b_on_a.others]


def test_the_well_behind_another_feels_it_most():
    """W-far takes its gas through W-B's node. When W-B's rate changes, W-far
    must move at least as much as anything upstream of the header does --
    it carries the header's change plus the change in its own feeder."""
    net = gathering_field()
    report = influence_of_demand(net, 3, change_fraction=2.0,
                                 gauge_resolution_psi=0.5)
    by_node = {e.node: e.magnitude for e in report.others}
    assert by_node[4] >= by_node[1], by_node


# ------------------------------------------------- against the gauge rule ---


def test_a_coarser_gauge_can_only_shrink_the_affected_set():
    """The threshold is a reporting decision, not a physical one, so it must
    be monotone: nothing becomes affected by being measured more crudely."""
    net = gathering_field()
    previous = None
    for resolution in (0.01, 0.5, 2.0, 50.0, 5000.0):
        report = influence_of_demand(net, 3, change_fraction=2.0,
                                     gauge_resolution_psi=resolution)
        nodes = {e.node for e in report.affected}
        if previous is not None:
            assert nodes <= previous, (resolution, nodes, previous)
        previous = nodes


def test_a_gauge_coarser_than_the_whole_response_reports_nothing_moved():
    net = gathering_field()
    report = influence_of_demand(net, 2, change_fraction=0.05,
                                 gauge_resolution_psi=5000.0)
    assert report.affected == []
    assert report.sparsity == 0.0
    assert "invisible on the rest of the field's instruments" in report.notes[0]


def test_the_perturbed_node_is_not_counted_as_news_about_itself():
    """Of course the well you turned up moved. It is kept in the list because
    it is worth seeing, and kept out of the sparsity measure and the headline
    because the question was who *else* is affected."""
    net = gathering_field()
    report = influence_of_demand(net, 2, change_fraction=0.30)

    sources = [e for e in report.affected if e.is_source]
    assert len(sources) == 1 and sources[0].node == 2
    assert all(e.node != 2 for e in report.others)
    assert report.worst is None or report.worst.node != 2


def test_affected_nodes_are_ordered_worst_first():
    net = gathering_field()
    report = influence_of_demand(net, 3, change_fraction=2.0,
                                 gauge_resolution_psi=0.1)
    magnitudes = [e.magnitude for e in report.affected]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_fixed_pressure_nodes_are_not_reported_at_all():
    """A separator held at 900 psia is held there by definition. Listing it as
    unaffected would be reporting the boundary condition back as a finding."""
    net = gathering_field()
    report = influence_of_demand(net, 2, change_fraction=0.50)
    assert 0 not in [e.node for e in report.affected]
    assert report.total_nodes == 4  # five nodes, one of them fixed


def test_counts_and_sparsity_agree_with_the_lists():
    net = gathering_field()
    report = influence_of_demand(net, 3, change_fraction=2.0,
                                 gauge_resolution_psi=0.5)
    assert report.total_nodes == len(report.affected) + report.unaffected_count
    assert report.sparsity == pytest.approx(
        len(report.others) / (report.total_nodes - 1))


def test_a_field_that_cannot_deliver_is_declared_not_differenced():
    """Two solves that did not converge still produce two sets of numbers, and
    differencing them produces a confident-looking answer about nothing. The
    report has to say so."""
    net = gathering_field()
    report = influence_of_demand(net, 3, change_fraction=400.0)
    assert not report.reliable
    assert "did not converge" in report.notes[0]


# ----------------------------------------------------- the caller's network --


def test_the_input_network_is_left_alone():
    """The perturbation goes on a copy. A sensitivity run that quietly edits
    the user's case is a data-loss bug wearing an analysis hat."""
    net = gathering_field()
    before = {i: n.withdrawal for i, n in net.nodes.items()}
    influence_of_demand(net, 2, change_fraction=0.75)
    assert {i: n.withdrawal for i, n in net.nodes.items()} == before


def test_running_it_twice_gives_the_same_answer():
    net = gathering_field()
    first = influence_of_demand(net, 2, change_fraction=0.75)
    second = influence_of_demand(net, 2, change_fraction=0.75)
    assert [(e.node, e.delta_psi) for e in first.affected] == \
           [(e.node, e.delta_psi) for e in second.affected]


def test_nonsense_requests_are_refused():
    net = gathering_field()
    with pytest.raises(ValueError, match="not in the network"):
        influence_of_demand(net, 99)
    with pytest.raises(ValueError, match="fixed-pressure node"):
        influence_of_demand(net, 0)
    with pytest.raises(ValueError, match="withdraws nothing"):
        influence_of_demand(net, 1)  # the header is a pure junction
    with pytest.raises(ValueError, match="gauge resolution"):
        influence_of_demand(net, 2, gauge_resolution_psi=0.0)


def test_the_default_gauge_is_a_real_transducer():
    """2 psi, which is roughly 0.25 percent of a 1,000 psi span. The point of
    the threshold is lost if it drifts to a number no instrument matches."""
    assert 0.5 <= DEFAULT_GAUGE_RESOLUTION_PSI <= 5.0


# ------------------------------------------------------------ the matrix ----


def test_the_matrix_covers_every_well_that_has_a_rate():
    """Junctions and fixed-pressure nodes have no demand to perturb, so they
    are not rows."""
    net = gathering_field()
    matrix = influence_matrix(net, change_fraction=0.50)
    assert sorted(matrix) == [2, 3, 4]
    assert all(r.source_node == node for node, r in matrix.items())


def test_the_matrix_agrees_with_the_single_runs():
    net = gathering_field()
    matrix = influence_matrix(net, nodes=[3], change_fraction=2.0,
                              gauge_resolution_psi=0.5)
    single = influence_of_demand(net, 3, change_fraction=2.0,
                                 gauge_resolution_psi=0.5)
    assert [(e.node, e.delta_psi) for e in matrix[3].affected] == \
           [(e.node, e.delta_psi) for e in single.affected]


def test_a_looped_grid_absorbs_locally_and_a_radial_system_does_not():
    """Topology decides who is affected, and that is the engineering content.

    The same 20 percent change is felt by every well on the radial gathering
    system -- they all pull through one trunk line, so the trunk's extra drop
    reaches all of them -- and by nothing at all on a looped grid, where the
    surrounding loops redistribute the extra flow before it is worth reading
    on a gauge.

    Written the other way round first, as "most of a large field does not
    respond". That is true of the 400-node randomised grid it was measured on
    and false of a uniform one: there, every path home shares the load, and
    the response goes from nothing to everything over a narrow range of pipe
    size with no regime in between. What survives measurement is the contrast
    below, not a number about field size.
    """
    radial = influence_of_demand(gathering_field(), 2, change_fraction=0.20)
    assert len(radial.others) == radial.total_nodes - 1

    looped = influence_of_demand(looped_grid(), 0, change_fraction=0.20)
    assert looped.reliable
    assert looped.others == [], [(e.name, e.delta_psi) for e in looped.others]
    assert looped.sparsity == 0.0


def test_influence_falls_off_with_distance():
    """Below gauge resolution the response is still there, and it is local:
    a node's immediate neighbours in the grid move more than the nodes three
    hops away. This is the property that makes "who else" a short list rather
    than a copy of the network."""
    report = influence_of_demand(looped_grid(), 0, change_fraction=1.0,
                                 gauge_resolution_psi=1e-4)
    moved = {e.node: e.magnitude for e in report.others}
    side = GRID_SIDE
    adjacent = [1, side]                      # one hop from the perturbed corner
    distant = [4 * side + 4, 3 * side + 4]    # the far side of the grid
    assert min(moved[n] for n in adjacent) > max(moved[n] for n in distant), moved
