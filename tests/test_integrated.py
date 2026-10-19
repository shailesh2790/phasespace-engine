"""Tests for solving wells and the network together.

Two kinds of test here, and the second matters more.

The first kind is degeneracy: when the network cannot influence the wells --
one well, or a line so large it drops no pressure -- the coupled answer must
reduce *exactly* to standalone nodal analysis. That is the strongest available
check, because it compares against a function that was written years earlier
for a different purpose and is separately tested.

The second kind asks whether coupling was worth doing at all. If the coupled
field rate matched the sum of uncoupled well rates, this module would be
ceremony. It does not match, and the gap is the product: wells interfere
through the gathering system, and a set of calculators cannot see it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.components import LiquidPipe
from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.integrated import (
    CoupledWell,
    IntegratedResult,
    attach_wells,
    solve_integrated,
)
from phasespace.sim.solver import FLOW, LIQUID, Network, Node, PRESSURE
from phasespace.sim.units import f_to_r
from phasespace.sim.wells import ProductivityIndexIPR, Tubing, nodal_analysis

OIL = LiquidProperties(api_gravity=32.0, temperature_r=f_to_r(180.0),
                       bubble_point_psia=2200.0, water_cut=0.2)
GAS = GasProperties(gravity=0.75, temperature_r=f_to_r(180.0))
SEPARATOR_PSIA = 150.0


def tubing() -> Tubing:
    return Tubing(depth_ft=8000.0, inner_diameter_in=2.441,
                  wellhead_temperature_f=110.0, bottomhole_temperature_f=180.0)


def well(name: str, node: int, pi: float = 1.5) -> CoupledWell:
    return CoupledWell(name=name, node=node,
                       ipr=ProductivityIndexIPR(3000.0, pi), tubing=tubing(),
                       gas_oil_ratio_scf_stb=600.0, water_cut=0.2,
                       oil=OIL, gas=GAS)


def field_case(count: int = 3, diameter_in: float = 4.0,
               length_mi: float = 1.5, separator: float = SEPARATOR_PSIA,
               trunk_in: float = 6.0, trunk_mi: float = 3.0):
    """A gathering system: wells onto a common header, header down a trunk.

    The shared trunk is the whole point and an earlier version of this helper
    did not have one -- every well had its own line straight to a
    fixed-pressure separator, so no well could affect any other and the
    interference tests were checking a topology in which interference is
    impossible. A star onto a fixed pressure is not a gathering system.

    Node 0 is the separator, node 1 the header, wells from node 2 up.
    """
    net = Network(phase=LIQUID, fluid=OIL)
    net.add_node(Node(0, PRESSURE, pressure_psia=separator, name="separator"))
    net.add_node(Node(1, FLOW, withdrawal=0.0, name="header"))
    net.add_branch(LiquidPipe(a=1, b=0, diameter_in=trunk_in,
                              length_mi=trunk_mi, name="trunk"))
    wells = []
    for i in range(count):
        node = 2 + i
        net.add_node(Node(node, FLOW, withdrawal=0.0, name="W%d" % (i + 1)))
        net.add_branch(LiquidPipe(a=node, b=1, diameter_in=diameter_in,
                                  length_mi=length_mi, name="FL-%d" % (i + 1)))
        wells.append(well("W%d" % (i + 1), node, pi=1.5))
    return net, wells


# ------------------------------------------------------------- degeneracy --


def standalone_rate() -> float:
    """The same well, at the separator, by the standalone path."""
    return nodal_analysis(
        ipr=ProductivityIndexIPR(3000.0, 1.5), tubing=tubing(),
        wellhead_pressure_psia=SEPARATOR_PSIA, gas_oil_ratio_scf_stb=600.0,
        water_cut=0.2, oil=OIL, gas=GAS).liquid_rate_stb_d


def test_one_well_on_a_huge_line_reduces_to_nodal_analysis():
    """The load-bearing test. With a line so large it drops essentially no
    pressure, the well sees the separator directly and the coupled answer must
    be the standalone one -- computed by a function written years earlier, for
    a different purpose, and separately tested.

    The tolerance is the cost of interpolation, not slack. The coupled solve
    samples each well's deliverability onto a curve and interpolates, because
    re-running a full nodal analysis every pass took two minutes on four
    wells. At the default resolution that is about 1% low, and the next test
    shows it converging away.
    """
    net, wells = field_case(count=1, diameter_in=24.0, length_mi=0.05,
                            trunk_in=24.0, trunk_mi=0.05)
    result = solve_integrated(net, wells)

    assert result.converged
    assert result.wells[0].liquid_rate_stb_d == pytest.approx(
        standalone_rate(), rel=0.015)
    assert result.wells[0].wellhead_psia == pytest.approx(SEPARATOR_PSIA, rel=0.02)


def test_refining_the_curve_converges_on_the_standalone_answer():
    """That the interpolation is a faithful approximation rather than a
    biased one, which a single loose tolerance could never show.

    Error against the standalone path falls monotonically with curve
    resolution -- about 1.0%, 0.22%, 0.08%, 0.03% -- so the default trades a
    known, shrinking error for a hundredfold speed-up rather than hiding one.
    """
    reference = standalone_rate()
    errors = []
    for points in (14, 20, 30, 44):
        net, wells = field_case(count=1, diameter_in=24.0, length_mi=0.05,
                                trunk_in=24.0, trunk_mi=0.05)
        rate = solve_integrated(net, wells,
                                curve_points=points).wells[0].liquid_rate_stb_d
        errors.append(abs(rate - reference) / reference)

    assert all(a > b for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-3, "%.4f%% at the finest resolution" % (100 * errors[-1])


def test_identical_wells_on_identical_lines_produce_identically():
    """Symmetry is exact, so any asymmetry in the answer is a bug in the
    coupling rather than physics."""
    net, wells = field_case(count=3)
    result = solve_integrated(net, wells)

    rates = [w.liquid_rate_stb_d for w in result.wells]
    assert max(rates) - min(rates) < 1e-6 * max(rates), rates
    pressures = [w.wellhead_psia for w in result.wells]
    assert max(pressures) - min(pressures) < 1e-6 * max(pressures)


def test_mass_balance_closes_at_the_separator():
    """Everything the wells make has to arrive. A coupling that leaked would
    still converge and still look plausible."""
    net, wells = field_case(count=3)
    result = solve_integrated(net, wells)

    produced = sum(w.liquid_rate_stb_d for w in result.wells)
    assert result.total_liquid_stb_d == pytest.approx(produced, rel=1e-12)
    assert produced > 0.0


# ------------------------------------------------- why coupling is needed --


def test_coupling_changes_the_answer():
    """The justification for the whole module.

    Solved separately, each well is computed against the separator pressure as
    though the gathering system were not there. Coupled, each well raises the
    pressure the others work against. If these agreed, this module would be
    ceremony.
    """
    net, wells = field_case(count=4, trunk_in=3.5, trunk_mi=5.0)
    coupled = solve_integrated(net, wells)

    uncoupled = sum(
        nodal_analysis(ipr=w.ipr, tubing=w.tubing,
                       wellhead_pressure_psia=SEPARATOR_PSIA,
                       gas_oil_ratio_scf_stb=w.gas_oil_ratio_scf_stb,
                       water_cut=w.water_cut, oil=w.oil,
                       gas=w.gas).liquid_rate_stb_d
        for w in wells)

    assert coupled.converged
    assert coupled.total_liquid_stb_d < uncoupled, (
        "coupled %.0f vs uncoupled %.0f" % (coupled.total_liquid_stb_d, uncoupled))
    shortfall = (uncoupled - coupled.total_liquid_stb_d) / uncoupled
    assert shortfall > 0.02, (
        "coupling changed the field rate by only %.2f%%; on a restricted "
        "gathering system it should be materially more" % (100 * shortfall))


def test_wells_interfere_through_the_gathering_system():
    """Adding a well to a shared line reduces what the existing wells make.
    Nothing in a single-well calculation can produce this result, and it is the
    question an operator actually asks before drilling an infill."""
    small, wells_small = field_case(count=2, trunk_in=3.5, trunk_mi=5.0)
    large, wells_large = field_case(count=4, trunk_in=3.5, trunk_mi=5.0)

    first = solve_integrated(small, wells_small).wells[0].liquid_rate_stb_d
    later = solve_integrated(large, wells_large).wells[0].liquid_rate_stb_d
    assert later < first


def test_raising_separator_pressure_costs_production():
    """Backpressure sensitivity, with the sign that matters commercially: this
    is the calculation behind every compression project."""
    rates = []
    for separator in (100.0, 200.0, 350.0):
        net, wells = field_case(count=3, separator=separator)
        rates.append(solve_integrated(net, wells).total_liquid_stb_d)
    assert rates[0] > rates[1] > rates[2]


def test_a_bigger_flowline_recovers_production():
    """The other half of the same question: debottlenecking. A larger gathering
    line delivers more from the same wells and the same reservoir."""
    narrow, wells_n = field_case(count=3, trunk_in=2.5, trunk_mi=5.0)
    wide, wells_w = field_case(count=3, trunk_in=8.0, trunk_mi=5.0)
    assert (solve_integrated(wide, wells_w).total_liquid_stb_d
            > solve_integrated(narrow, wells_n).total_liquid_stb_d)


# ------------------------------------------------------------ reporting ----


def test_the_binding_constraint_is_named():
    """A rate without its limiting constraint is not actionable: separator
    pressure is a compression project, tubing is a workover, and inflow is
    neither."""
    net, wells = field_case(count=3)
    result = solve_integrated(net, wells)
    assert result.binding_constraints
    for state in result.wells:
        assert state.constraint


def test_back_pressure_is_identified_when_the_network_dominates():
    """On a badly restricted system the constraint should name the network,
    not the tubing."""
    net, wells = field_case(count=4, trunk_in=3.0, trunk_mi=5.0,
                            separator=400.0)
    result = solve_integrated(net, wells)
    flowing = [w for w in result.wells if w.flowing]
    assert flowing, "the case is too restricted to test anything: every well died"
    named = " ".join(w.constraint for w in flowing)
    assert "back pressure" in named, named
    # And it quantifies the prize, because "back pressure is binding" is not
    # an argument for a compressor; a percentage of the field is.
    assert "%" in named


# ----------------------------------------------------- non-smooth wells ----


def test_a_well_below_its_minimum_rate_is_shut_and_reported():
    """The commonest operating rule there is, and the commonest source of
    non-smoothness in the solve."""
    net, wells = field_case(count=2)
    wells[1].min_rate_stb_d = 1e6      # impossible: it must shut
    result = solve_integrated(net, wells)

    shut = [w for w in result.wells if not w.flowing]
    assert len(shut) == 1
    assert shut[0].name == "W2"
    assert "minimum rate" in shut[0].constraint


def test_a_well_above_its_wellhead_limit_does_not_flow():
    net, wells = field_case(count=2)
    wells[0].max_wellhead_psia = 10.0   # below the separator, so never flows
    result = solve_integrated(net, wells)
    assert not result.wells[0].flowing
    assert "above the well" in result.wells[0].constraint


def test_shutting_a_well_leaves_more_for_the_others():
    """Conservation with a switch in it: the wells that keep flowing must pick
    up some of what the shut well was taking from the system."""
    net, wells = field_case(count=3, trunk_in=3.5, trunk_mi=5.0)
    before = solve_integrated(net, wells).wells[0].liquid_rate_stb_d

    net2, wells2 = field_case(count=3, trunk_in=3.5, trunk_mi=5.0)
    wells2[2].min_rate_stb_d = 1e6
    after = solve_integrated(net2, wells2)
    assert after.wells[0].liquid_rate_stb_d > before
    assert not after.wells[2].flowing


# --------------------------------------------------------------- guards ----


def test_a_well_on_a_fixed_pressure_node_is_refused():
    """That is the uncoupled calculation this module replaces: a well
    delivering into a pressure held constant regardless of what it produces."""
    net = Network(phase=LIQUID, fluid=OIL)
    net.add_node(Node(0, PRESSURE, pressure_psia=SEPARATOR_PSIA))
    with pytest.raises(ValueError, match="fixed-pressure node"):
        solve_integrated(net, [well("W1", 0)])


def test_a_well_on_a_missing_node_is_refused():
    net, wells = field_case(count=1)
    wells.append(well("ghost", 99))
    with pytest.raises(ValueError, match="not in the network"):
        solve_integrated(net, wells)


def test_no_wells_is_refused():
    net, _ = field_case(count=1)
    with pytest.raises(ValueError, match="no wells"):
        solve_integrated(net, [])


def test_damping_is_validated():
    net, wells = field_case(count=1)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="damping"):
            solve_integrated(net, wells, initial_damping=bad)


def test_attach_wells_adds_only_the_missing_nodes():
    net = Network(phase=LIQUID, fluid=OIL)
    net.add_node(Node(0, PRESSURE, pressure_psia=SEPARATOR_PSIA))
    wells = [well("W1", 1), well("W2", 2)]
    attach_wells(net, wells)
    assert set(net.nodes) == {0, 1, 2}
    attach_wells(net, wells)
    assert set(net.nodes) == {0, 1, 2}


def test_an_empty_result_reports_nothing_rather_than_zero():
    blank = IntegratedResult()
    assert blank.total_liquid_stb_d == 0.0
    assert blank.flowing_wells == []
    assert blank.binding_constraints == {}


# -------------------------------------------------- an impossible system ---


def test_a_field_that_cannot_flow_says_so_rather_than_converging():
    """The failure this module got wrong first, and the one that matters most.

    On a trunk far too small, the network pressure climbs above the wells'
    own reservoir pressure. The first version reported every well flowing at
    its last rate, against a wellhead pressure higher than the reservoir
    supplying it, and marked the solve converged -- because rate change is
    zero once every well is frozen, so the convergence test passed by
    construction.

    A frozen well is an unresolved discontinuity, not a solution.
    """
    net, wells = field_case(count=4, trunk_in=2.0, trunk_mi=6.0,
                            separator=150.0)
    result = solve_integrated(net, wells)

    assert not result.converged
    assert result.total_liquid_stb_d == 0.0
    assert all(not w.flowing for w in result.wells)
    for state in result.wells:
        assert "cannot flow" in state.constraint
    assert any("frozen" in note for note in result.notes)


def test_no_well_is_ever_reported_producing_above_its_reservoir():
    """The invariant behind that test, stated on its own so it cannot regress
    quietly: a flowing well's bottomhole pressure must be below the pressure
    driving it."""
    for trunk in (2.0, 3.0, 6.0):
        net, wells = field_case(count=3, trunk_in=trunk, trunk_mi=5.0)
        for state in solve_integrated(net, wells).wells:
            if state.flowing:
                assert state.bottomhole_psia < 3000.0, (
                    "%s reported flowing at %.0f psia bottomhole against a "
                    "3000 psia reservoir" % (state.name, state.bottomhole_psia))
