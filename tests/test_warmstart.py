"""The warm start, and the one property that makes it safe.

Newton converges to the same root from any starting point it converges
from, so a guess may change the iteration count and must change nothing
else. Every test here is ultimately about that: the answers are compared,
not just the speed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.components import GasPipe
from phasespace.sim.fluids import GasProperties
from phasespace.sim.solver import FLOW, PRESSURE, Network, Node, solve
from phasespace.sim.warmstart import (
    benchmark,
    linearised_pressures,
    warm_solve,
)


def star(wells=12, seed=0):
    """Wells into a header into a separator: a gathering system."""
    rng = np.random.default_rng(seed)
    net = Network(fluid=GasProperties(gravity=0.65))
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0, name="separator"))
    net.add_node(Node(1, FLOW, name="header"))
    net.add_branch(GasPipe(a=1, b=0, diameter_in=12.0, length_mi=6.0))
    for well in range(wells):
        net.add_node(Node(2 + well, PRESSURE,
                          pressure_psia=float(rng.uniform(900.0, 1400.0)),
                          name="w%d" % well))
        net.add_branch(GasPipe(a=2 + well, b=1,
                               diameter_in=float(rng.uniform(3.0, 8.0)),
                               length_mi=float(rng.uniform(0.5, 6.0))))
    return net


def trunk(nodes=40, total=4.0e7, seed=0):
    rng = np.random.default_rng(seed)
    net = Network(fluid=GasProperties(gravity=0.65))
    net.add_node(Node(0, PRESSURE, pressure_psia=1200.0, name="inlet"))
    for i in range(1, nodes):
        net.add_node(Node(i, FLOW, withdrawal=total / (nodes - 1)))
        net.add_branch(GasPipe(a=i - 1, b=i, diameter_in=20.0,
                               length_mi=float(rng.uniform(0.3, 1.2))))
    return net


# ------------------------------------------------- the answer is unchanged --


@pytest.mark.parametrize("builder", [star, trunk])
def test_the_warm_answer_is_the_cold_answer(builder):
    """The property the whole idea rests on. A starting point cannot move
    the root, and if this ever fails the warm start is changing the
    problem rather than the path to it."""
    net = builder()
    cold = solve(net)
    warm = solve(net, initial_pressures=linearised_pressures(net))
    assert cold.converged and warm.converged
    for node, pressure in cold.pressures.items():
        assert warm.pressures[node] == pytest.approx(pressure, rel=1e-9)


def test_a_wild_guess_still_reaches_the_same_answer():
    """Newton is allowed to be slow from a bad start; it is not allowed to
    land somewhere else. This is what makes a learned guess safe without
    the learned model being any good."""
    net = star()
    cold = solve(net)
    silly = {nid: 5.0 for nid in net.unknown_node_ids()}
    warm = solve(net, initial_pressures=silly)
    assert warm.converged
    for node, pressure in cold.pressures.items():
        assert warm.pressures[node] == pytest.approx(pressure, rel=1e-9)


def test_an_impossible_guess_is_ignored_rather_than_used():
    """A predictor that returns a negative or non-finite pressure would
    otherwise put the first Newton step where the correlations are
    undefined."""
    net = star()
    cold = solve(net)
    poisoned = {nid: value for nid, value in
                zip(net.unknown_node_ids(), [-50.0, float("nan"), 0.0])}
    warm = solve(net, initial_pressures=poisoned)
    assert warm.converged
    for node, pressure in cold.pressures.items():
        assert warm.pressures[node] == pytest.approx(pressure, rel=1e-9)


def test_an_unknown_node_id_in_the_guess_is_skipped():
    net = star()
    warm = solve(net, initial_pressures={9999: 700.0})
    assert warm.converged


# --------------------------------------------------------------- the guess --


def test_the_guess_respects_the_boundary_conditions():
    """A field that ignores the fixed pressures would be no better than
    the constant it replaces."""
    net = star()
    guess = linearised_pressures(net)
    assert guess, "a star network should produce a guess"
    header = guess[1]
    assert 700.0 < header < 1500.0, (
        "the header should land between the separator and the wells, "
        "got %.1f" % header)


def test_the_guess_is_closer_than_the_constant_it_replaces():
    net = star(wells=20)
    answer = solve(net).pressures
    guess = linearised_pressures(net)
    unknown = list(net.unknown_node_ids())

    # What the solver would otherwise start from.
    fixed = [net.nodes[n].pressure_psia for n in net.pressure_node_ids()]
    constant = min(fixed) * 0.9

    from_guess = np.mean([abs(guess[n] - answer[n]) for n in unknown])
    from_constant = np.mean([abs(constant - answer[n]) for n in unknown])
    assert from_guess < from_constant


def test_a_network_with_no_branches_gives_no_guess_rather_than_raising():
    net = Network(fluid=GasProperties())
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    net.add_node(Node(1, FLOW, withdrawal=0.0))
    assert linearised_pressures(net) == {}


def test_more_passes_never_breaks_the_guess():
    net = star()
    for passes in (1, 2, 8):
        guess = linearised_pressures(net, passes=passes)
        assert all(np.isfinite(v) and v > 0.0 for v in guess.values())


# ----------------------------------------------------------- the benchmark --


def test_the_benchmark_reports_both_counts_and_checks_agreement():
    """The PRD asks for the speed-up to be measured rather than assumed,
    which means solving both ways and comparing the answers too."""
    result = benchmark(star(wells=30))
    assert result.cold_iterations > 0 and result.warm_iterations > 0
    assert result.comparable
    assert result.agrees
    assert result.max_pressure_difference < 1e-6


def test_the_warm_start_helps_on_a_gathering_system():
    """The topology the product is actually for. Measured at 33-50% across
    sizes; asserted loosely so a solver improvement that shrinks the
    headroom does not read as a regression here."""
    saved = [benchmark(star(wells=n, seed=s)).fraction_saved
             for n in (20, 40) for s in range(2)]
    assert np.median(saved) > 0.15, saved


def test_two_non_converged_solves_are_not_called_agreement():
    """Both runs hitting the iteration cap land in different places, and
    that disagreement is about the network rather than the starting
    point. Averaging such a case into a speed-up figure would hide an
    infeasible network behind a number."""
    # Far more demand than the line can carry.
    net = trunk(nodes=30, total=4.0e9)
    result = benchmark(net, max_iter=40)
    if result.cold_converged or result.warm_converged:
        pytest.skip("this case converged; it is not the one under test")
    assert not result.comparable


# ------------------------------------------------------------ warm_solve --


def test_warm_solve_matches_a_plain_solve():
    net = star(wells=25)
    assert warm_solve(net).pressures == pytest.approx(solve(net).pressures)


def test_warm_solve_falls_back_when_the_guess_does_not_converge(monkeypatch):
    """A guess that sends Newton somewhere it cannot recover from costs
    the answer, not just time, so that is the case worth guarding."""
    import phasespace.sim.warmstart as module

    net = star()
    monkeypatch.setattr(module, "linearised_pressures",
                        lambda n, p=4: {nid: 1e9 for nid in n.unknown_node_ids()})
    # A budget the cold solve clears comfortably (it needs 7) so the test
    # is about the guess and not about the iteration cap. A 1e9 psia start
    # never recovers here, even given 40 iterations, which is why this
    # fallback exists at all.
    result = module.warm_solve(net, max_iter=20)
    assert result.converged
    assert result.iterations <= 10
    assert any("fell back" in note for note in result.notes)
