"""Tests for gas-lift allocation.

The prototype these replace asserted that greedy allocation on concave
envelopes is exactly optimal. That is true when a well can take any injection
rate from zero, and false once wells have a minimum stable injection -- which
every well in the prototype's own demo had. So the optimality claim is tested
here only where it holds, and where it does not hold the gap is *bounded*
against dense search rather than assumed away.

The other thing worth testing is the constraint that the prototype violated:
no well may be left strictly between zero and its minimum stable injection.
Below that rate a well heads instead of lifting, so the curve's predicted oil
there describes a flow regime the well is not in -- and the number is a
setpoint someone would actually send.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.gaslift import (
    MIN_TESTS,
    GasLiftWell,
    allocate,
    compare_curves,
    concave_envelope,
    curve_from_nodal,
    show_the_maths,
)
from phasespace.sim.units import f_to_r
from phasespace.sim.wells import ProductivityIndexIPR, Tubing


def synthetic_well(name, natural, uplift, response, friction, q_min=1000.0,
                   current=9000.0, noise=0.0, seed=0):
    """A well whose true response is known, sampled as six well tests."""
    rng = np.random.default_rng(seed)
    rates = np.array([0, 3000, 6000, 9000, 13000, 18000], dtype=float)
    truth = (natural + uplift * (1.0 - np.exp(-response * rates))
             - friction * rates * 1e-2)
    tests = [(float(q), float(max(0.0, o + rng.normal(0.0, noise))))
             for q, o in zip(rates, truth)]
    return GasLiftWell(name=name, tests=tests, q_min=q_min, q_max=25000.0,
                       current=current).fit()


def field_of(count=4, q_min=1000.0):
    specs = [
        ("W-01", 0.0, 6.5, 2.2e-4, 3.0e-5),
        ("W-02", 1.2, 4.0, 1.6e-4, 2.0e-5),
        ("W-03", 0.0, 9.0, 1.0e-4, 1.5e-5),
        ("W-04", 0.5, 2.5, 3.5e-4, 5.0e-5),
        ("W-05", 0.0, 5.0, 0.8e-4, 1.0e-5),
        ("W-06", 2.0, 3.0, 2.8e-4, 4.0e-5),
    ][:count]
    return [synthetic_well(n, a, b, c, d, q_min=q_min) for n, a, b, c, d in specs]


# ------------------------------------------------------------- the curve ---


def test_fitting_needs_four_tests():
    """Four coefficients, four tests. Fewer is an underdetermined system that
    returns something confident and meaningless, not a loose fit."""
    well = GasLiftWell("X", tests=[(0, 1), (1000, 2), (2000, 3)])
    with pytest.raises(ValueError, match="at least %d" % MIN_TESTS):
        well.fit()


def test_a_clean_curve_tracks_its_own_tests():
    """How closely the Alarcon form can follow a well, with no measurement
    noise at all.

    Not exactly, and the reason is worth stating: the form is a quadratic plus
    a logarithm, and a real well's response is nearer an exponential approach
    to a plateau with a linear friction penalty. Four coefficients fitted
    across that family mismatch leave a residual of a percent or two even on
    perfect data, which is the floor on how well any test-fitted curve can do
    and is why a nodal cross-check is worth having.
    """
    well = synthetic_well("W", 1.0, 5.0, 2.0e-4, 2.0e-5, noise=0.0)
    span = max(o for _, o in well.tests) - min(o for _, o in well.tests)
    for rate, oil in well.tests:
        assert float(well.oil(rate)) == pytest.approx(oil, abs=0.03 * span)
    assert well.rms < 0.02 * span


def test_oil_is_never_negative():
    well = synthetic_well("W", 0.0, 1.0, 1e-4, 9e-4)
    assert float(np.min(well.oil(np.linspace(0, 25000, 50)))) >= 0.0


def test_a_loose_fit_is_reported():
    noisy = synthetic_well("W", 1.0, 5.0, 2.0e-4, 2.0e-5, noise=1.2, seed=3)
    assert any("loose" in note for note in noisy.warnings(9000.0))


def test_allocation_outside_the_curve_is_flagged():
    well = synthetic_well("W", 1.0, 5.0, 2.0e-4, 2.0e-5)
    assert any("outside the range" in note for note in well.warnings(24000.0))
    assert not any("outside the range" in note for note in well.warnings(9000.0))


def test_nonsense_limits_are_refused():
    with pytest.raises(ValueError, match="maximum injection"):
        GasLiftWell("X", q_max=0.0)
    with pytest.raises(ValueError, match="exceeds maximum"):
        GasLiftWell("X", q_min=9000.0, q_max=1000.0)


# ----------------------------------------------------------- the envelope --


def test_envelope_is_concave_and_touches_the_top():
    x = np.linspace(0, 10, 40)
    y = np.sin(x) + 0.3 * x          # deliberately non-concave
    hx, hy = concave_envelope(x, y)
    slopes = np.diff(hy) / np.diff(hx)
    assert np.all(np.diff(slopes) <= 1e-9), slopes
    assert hy[np.argmax(hy)] >= np.max(y) - 1e-9


def test_envelope_leaves_a_concave_curve_alone():
    x = np.linspace(0, 10, 25)
    y = -((x - 5.0) ** 2)
    hx, _ = concave_envelope(x, y)
    assert len(hx) == len(x)


# ------------------------------------------------------- the q_min rule ----


def test_no_well_is_left_below_its_minimum_stable_injection():
    """The bug this module was written to fix.

    The prototype allocated whatever gas was left to the best well regardless,
    so a well with a 1,500 minimum could come back with a 600 setpoint. At
    that rate the well heads rather than lifts, the curve's oil there is
    fiction, and the number is one an operator would actually send to a
    choke.
    """
    wells = field_of(6, q_min=1500.0)
    for budget in (600.0, 1200.0, 1800.0, 3000.0, 7000.0, 40000.0):
        result = allocate(wells, budget)
        for row in result.wells:
            assert row.injection == 0.0 or row.injection >= 1500.0 - 1e-6, (
                "%s got %g against a minimum of 1500 at a budget of %g"
                % (row.name, row.injection, budget))


def test_gas_that_cannot_be_used_is_reported_not_hidden():
    """A budget too small for any well's minimum leaves the gas unallocated,
    and says so. Silently absorbing it would show a field using gas it
    physically cannot take."""
    wells = field_of(3, q_min=5000.0)
    result = allocate(wells, 900.0)
    assert result.gas_used == 0.0
    assert result.gas_unused == pytest.approx(900.0)
    assert any("unallocated" in note for note in result.notes)


def test_the_budget_and_the_valve_limits_are_respected():
    wells = field_of(6)
    result = allocate(wells, 72000.0)
    assert result.gas_used <= 72000.0 + 1e-6
    assert sum(r.injection for r in result.wells) <= 72000.0 + 1e-6
    for row, well in zip(result.wells, wells):
        assert row.injection <= well.q_max + 1e-6


# ------------------------------------------------------- is it any good? ---


def test_unconstrained_greedy_is_exactly_optimal():
    """Where the optimality claim actually holds: no minimum injection, so
    every well can take any rate and greedy on concave envelopes is exact.
    Checked against dense exhaustive search on a small field."""
    wells = field_of(3, q_min=0.0)
    budget, step = 18000.0, 1500.0
    result = allocate(wells, budget, step=step)
    assert result.exactly_optimal

    grid = np.arange(0.0, budget + step / 2, step)
    best = -np.inf
    for a in grid:
        for b in grid:
            c = budget - a - b
            if c < -1e-9 or c > wells[2].q_max:
                continue
            total = sum(float(w.oil(q)) for w, q in zip(wells, (a, b, max(c, 0.0))))
            best = max(best, total)
    assert result.total_oil >= best - 1e-6, (result.total_oil, best)


def test_with_minimum_injection_the_gap_to_exhaustive_search_is_small():
    """Where it does not hold, the honest statement is a bound rather than a
    claim. The first increment is a discrete jump, so greedy is a heuristic --
    it should still land within a fraction of a percent of exhaustive search."""
    wells = field_of(3, q_min=2000.0)
    budget, step = 18000.0, 1000.0
    result = allocate(wells, budget, step=step)
    assert not result.exactly_optimal
    assert any("heuristic" in note for note in result.notes)

    feasible = [0.0] + list(np.arange(2000.0, 18000.0 + 1e-9, step))
    best = -np.inf
    for combo in itertools.product(feasible, repeat=3):
        if sum(combo) > budget + 1e-9:
            continue
        if any(q > w.q_max for q, w in zip(combo, wells)):
            continue
        best = max(best, sum(float(w.oil(q)) for w, q in zip(wells, combo)))

    shortfall = (best - result.total_oil) / best
    assert shortfall < 0.02, "greedy is %.2f%% below exhaustive" % (100 * shortfall)


def test_equal_slope_holds_for_wells_in_the_interior():
    """The optimality condition itself: every well receiving gas, and not
    pinned at a limit, has the same marginal oil per unit of gas."""
    wells = field_of(6)
    result = allocate(wells, 72000.0, step=50.0)
    for row, well in zip(result.wells, wells):
        q = row.injection
        if well.q_min + 500 < q < well.q_max - 500:
            slope = (float(well.oil(q + 100)) - float(well.oil(q - 100))) / 200.0
            assert abs(slope - result.shadow_price) < 0.25 * result.shadow_price + 2e-5


def test_more_gas_never_makes_less_oil():
    wells = field_of(6)
    totals = [allocate(wells, g).total_oil for g in (20000.0, 50000.0, 90000.0)]
    assert totals[0] <= totals[1] <= totals[2] + 1e-9


def test_a_gas_price_stops_uneconomic_injection():
    """Value, not volume. On a field with expensive compression the
    profit-maximising allocation is not the production-maximising one."""
    wells = field_of(6)
    free = allocate(wells, 200000.0)
    priced = allocate(wells, 200000.0, oil_price=1.0, gas_price=4e-4)
    assert priced.gas_used < free.gas_used
    assert priced.total_oil <= free.total_oil + 1e-9


def test_the_shadow_price_prices_the_next_unit_of_compression():
    """lambda should predict what a little more gas is worth, which is the
    number a compression business case is built on."""
    wells = field_of(6)
    base = allocate(wells, 60000.0, step=100.0)
    more = allocate(wells, 62000.0, step=100.0)
    predicted = base.shadow_price * 2000.0
    actual = more.total_oil - base.total_oil
    assert actual == pytest.approx(predicted, rel=0.35, abs=0.02), (predicted, actual)


# -------------------------------------------- curves from the nodal model --


def nodal_well(name, pi=1.5, **kwargs):
    oil = LiquidProperties(api_gravity=32.0, temperature_r=f_to_r(180.0),
                           bubble_point_psia=2200.0, water_cut=0.3)
    gas = GasProperties(gravity=0.75, temperature_r=f_to_r(180.0))
    return curve_from_nodal(
        name=name,
        ipr=ProductivityIndexIPR(2600.0, pi),
        tubing=Tubing(depth_ft=7000.0, inner_diameter_in=2.441,
                      wellhead_temperature_f=110.0, bottomhole_temperature_f=180.0),
        wellhead_pressure_psia=180.0, formation_gor_scf_stb=300.0,
        injection_rates=[0, 2.5e5, 5e5, 1e6, 1.5e6, 2e6],
        water_cut=0.3, oil=oil, gas=gas, **kwargs)


def test_a_well_with_no_tests_still_gets_a_curve():
    """The capability a test-fitted-only tool cannot offer: a new well, or one
    never tested at more than one injection rate, still has an IPR and a
    tubing string."""
    well = nodal_well("N-1", q_min=2.0e5, q_max=2.0e6)
    assert well.ready
    assert well.tests == []
    assert well.source == "nodal model"
    assert float(well.oil(1.0e6)) > 0.0


def test_the_nodal_curve_is_clamped_not_extrapolated():
    """Past the last computed point the model has said nothing. A line drawn
    onward is the tool inventing deliverability."""
    well = nodal_well("N-1", q_max=5.0e6)
    top = float(well.oil(2.0e6))
    assert float(well.oil(9.0e6)) == pytest.approx(top)


def test_modelled_and_fitted_wells_allocate_together():
    """Mixed sources in one field, which is the real case: some wells have a
    test history and some do not."""
    wells = [nodal_well("N-1", q_min=2.0e5, q_max=2.0e6),
             nodal_well("N-2", pi=2.2, q_min=2.0e5, q_max=2.0e6)]
    result = allocate(wells, 2.0e6, step=1.0e5)
    assert result.gas_used <= 2.0e6 + 1e-6
    assert result.total_oil > 0.0
    assert all(r.injection == 0.0 or r.injection >= 2.0e5 - 1e-6 for r in result.wells)


def test_comparing_a_fitted_curve_against_the_physics():
    """The cross-check the two sources make possible. Built by sampling the
    nodal curve and fitting the Alarcon form to those samples: the comparison
    should then agree closely, which is what makes it a usable detector when
    a real fitted curve does not."""
    modelled = nodal_well("N-1", q_max=2.0e6)
    samples = [(x, y) for x, y in modelled.curve_points]
    fitted = GasLiftWell("N-1", tests=samples, q_max=2.0e6).fit()

    report = compare_curves(fitted, modelled)
    assert report["agree"], report
    assert report["low"] < report["high"]


def test_curves_that_do_not_overlap_are_refused():
    modelled = nodal_well("N-1", q_max=2.0e6)
    elsewhere = GasLiftWell("other", tests=[(5e6, 1), (6e6, 2), (7e6, 3), (8e6, 4)]).fit()
    with pytest.raises(ValueError, match="do not overlap"):
        compare_curves(elsewhere, modelled)


# ------------------------------------------------------- show the maths ----


def test_the_working_is_printable_and_names_its_sources():
    wells = field_of(3)
    result = allocate(wells, 30000.0)
    lines = show_the_maths(wells, result)
    text = "\n".join(lines)
    assert "Shadow price" in text or "lambda" in text
    assert "concave envelope" in text
    for well in wells:
        assert well.name in text
        assert "[tests]" in text
    assert "heuristic" in text        # q_min is set, so the caveat must appear


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="no wells"):
        allocate([], 1000.0)
    with pytest.raises(ValueError, match="negative"):
        allocate(field_of(2), -1.0)
    with pytest.raises(ValueError, match="step"):
        allocate(field_of(2), 1000.0, step=0.0)
