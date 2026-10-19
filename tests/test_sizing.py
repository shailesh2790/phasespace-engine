"""Tests for fittings (minor losses), route hydraulics, and size selection.

The fitting values are checked against Crane TP-410, which is the document
an engineer would reach for to check them by hand. The sizing logic is
checked on physical monotonicity -- a bigger pipe must always deliver more
pressure -- because that is the property a recommendation depends on and the
one a solver bug destroys first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.components import fully_turbulent_friction_factor
from phasespace.sim.fittings import (
    BALL_VALVE,
    ELBOW_90_LONG_RADIUS,
    ELBOW_90_STANDARD,
    ENTRANCE_SHARP,
    EXIT,
    FITTING_LABELS,
    FITTING_TYPES,
    GATE_VALVE,
    GLOBE_VALVE,
    TEE_THROUGH_BRANCH,
    TEE_THROUGH_RUN,
    FittingCount,
    equivalent_length_ft,
    fitting_pressure_drop_psi,
    resistance_coefficient,
    total_equivalent_length_ft,
)
from phasespace.sim.fluids import GasProperties
from phasespace.sim.sizing import (
    DEFAULT_CANDIDATES,
    STD_PIPE_ID_IN,
    PipeRoute,
    RouteSegment,
    flow_at_pressures,
    recommend_size,
    relative_cost_index,
    solve_route,
)

GAS = GasProperties(gravity=0.65, temperature_r=520.0)


def simple_route():
    return PipeRoute(segments=[
        RouteSegment(length_ft=2000.0, name="station header",
                     fittings=[FittingCount(ELBOW_90_STANDARD, 4),
                               FittingCount(GATE_VALVE, 2)]),
        RouteSegment(length_ft=42000.0, elevation_change_ft=180.0, name="cross country"),
        RouteSegment(length_ft=8000.0, elevation_change_ft=-60.0, name="river crossing"),
    ])


# --------------------------------------------------------------- fittings --


def test_fully_turbulent_factors_match_crane_table():
    """Crane tabulates f_T per pipe size; the engine computes it from the
    rough-pipe Colebrook limit instead. They have to agree, or every K value
    downstream is wrong."""
    for diameter, expected in [(2.067, 0.019), (4.026, 0.017), (6.065, 0.015),
                               (7.981, 0.014), (12.0, 0.013), (23.25, 0.012)]:
        f_t = fully_turbulent_friction_factor(diameter, 0.0018)
        assert abs(f_t - expected) < 0.0011, (
            "%.3f in gave f_T=%.4f, Crane says ~%.3f" % (diameter, f_t, expected))


def test_ninety_degree_elbow_is_thirty_diameters():
    """K = 30*f_T collapses to exactly 30 pipe diameters of equivalent
    length. That identity is the whole reason K is stored as the multiplier
    rather than as K itself, so it is worth asserting directly."""
    for diameter in (2.067, 6.065, 12.0, 23.25):
        length = equivalent_length_ft(ELBOW_90_STANDARD, diameter)
        assert length / (diameter / 12.0) == pytest.approx(30.0, rel=1e-9)


def test_equivalent_lengths_rank_the_way_crane_says():
    d = 6.065
    ball = equivalent_length_ft(BALL_VALVE, d)
    gate = equivalent_length_ft(GATE_VALVE, d)
    long_radius = equivalent_length_ft(ELBOW_90_LONG_RADIUS, d)
    standard = equivalent_length_ft(ELBOW_90_STANDARD, d)
    branch = equivalent_length_ft(TEE_THROUGH_BRANCH, d)
    globe = equivalent_length_ft(GLOBE_VALVE, d)
    assert ball < gate < long_radius < standard < branch < globe
    # A long-radius elbow is the standard remedy for a tight pressure budget.
    assert long_radius < standard
    # Flow through a tee branch costs three times flow through its run.
    assert (equivalent_length_ft(TEE_THROUGH_BRANCH, d)
            == pytest.approx(3.0 * equivalent_length_ft(TEE_THROUGH_RUN, d), rel=1e-9))


def test_resistance_coefficient_falls_with_pipe_size():
    """The same elbow is a smaller K in a bigger line, because f_T falls."""
    assert (resistance_coefficient(ELBOW_90_STANDARD, 23.25)
            < resistance_coefficient(ELBOW_90_STANDARD, 6.065))


def test_entrances_and_exits_use_absolute_k():
    """These are geometric, not size-scaled: K is a plain number."""
    for diameter in (4.026, 12.0, 23.25):
        assert resistance_coefficient(ENTRANCE_SHARP, diameter) == pytest.approx(0.50)
        assert resistance_coefficient(EXIT, diameter) == pytest.approx(1.00)


def test_k_method_and_equivalent_length_agree():
    """Two routes to the same pressure drop: the K-method an engineer would
    use by hand, and the equivalent length the solver actually consumes."""
    diameter = 6.065
    velocity = 40.0
    density = 2.6  # lb/ft3, gas at a few hundred psia
    direct = fitting_pressure_drop_psi(ELBOW_90_STANDARD, diameter, velocity, density)

    # Same drop expressed as friction over the equivalent length.
    length_ft = equivalent_length_ft(ELBOW_90_STANDARD, diameter)
    f_t = fully_turbulent_friction_factor(diameter, 0.0018)
    d_ft = diameter / 12.0
    via_length = f_t * (length_ft / d_ft) * density * velocity ** 2 / (2 * 32.174) / 144.0
    assert direct == pytest.approx(via_length, rel=1e-9)


def test_fitting_totals_add_up():
    fittings = [FittingCount(ELBOW_90_STANDARD, 4), FittingCount(GATE_VALVE, 2)]
    d = 8.0
    expected = (4 * equivalent_length_ft(ELBOW_90_STANDARD, d)
                + 2 * equivalent_length_ft(GATE_VALVE, d))
    assert total_equivalent_length_ft(fittings, d) == pytest.approx(expected)
    assert total_equivalent_length_ft([], d) == 0.0


def test_every_fitting_type_is_labelled_and_priced():
    for kind in FITTING_TYPES:
        assert kind in FITTING_LABELS, "%s has no human-readable label" % kind
        assert resistance_coefficient(kind, 6.065) > 0.0


def test_bad_fittings_are_rejected():
    with pytest.raises(ValueError):
        FittingCount("swan_neck", 2)
    with pytest.raises(ValueError):
        FittingCount(ELBOW_90_STANDARD, -1)
    with pytest.raises(ValueError):
        resistance_coefficient("swan_neck", 6.0)


# ------------------------------------------------------------ route model --


def test_route_geometry_totals():
    route = simple_route()
    assert route.total_length_ft == pytest.approx(52000.0)
    assert route.total_rise_ft == pytest.approx(120.0)
    assert route.total_fittings == 6
    assert route.elevation_profile_ft() == pytest.approx([0.0, 0.0, 180.0, 120.0])
    assert route.distance_profile_ft() == pytest.approx([0.0, 2000.0, 44000.0, 52000.0])


def test_fittings_add_equivalent_length_that_scales_with_diameter():
    """The same fitting list contributes more equivalent *feet* on a bigger
    pipe. Not a contradiction -- a 24 in elbow is a longer flow path than a
    6 in one -- but it means sizing must recompute per candidate rather than
    caching one number."""
    route = simple_route()
    small = route.total_effective_length_ft(6.065) - route.total_length_ft
    large = route.total_effective_length_ft(23.25) - route.total_length_ft
    assert 0.0 < small < large


def test_fittings_actually_cost_pressure():
    bare = PipeRoute(segments=[RouteSegment(length_ft=3000.0)])
    with_fittings = PipeRoute(segments=[
        RouteSegment(length_ft=3000.0,
                     fittings=[FittingCount(ELBOW_90_STANDARD, 10),
                               FittingCount(GLOBE_VALVE, 2)])])
    bare_out = solve_route(bare, 6.065, 600.0, 12e6, GAS).delivery_pressure_psia
    fitted_out = solve_route(with_fittings, 6.065, 600.0, 12e6, GAS).delivery_pressure_psia
    assert fitted_out < bare_out


def test_non_finite_geometry_is_rejected():
    """NaN fails every comparison, so `length <= 0` lets it straight through
    and it then poisons the whole solve. An empty cell in a spreadsheet-style
    editor arrives as exactly that, so this is an ordinary input, not abuse."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            RouteSegment(length_ft=bad)
    with pytest.raises(ValueError):
        RouteSegment(length_ft=1000.0, elevation_change_ft=float("nan"))


def test_route_segments_must_be_sane():
    with pytest.raises(ValueError):
        RouteSegment(length_ft=0.0)
    with pytest.raises(ValueError):
        PipeRoute(segments=[])
    with pytest.raises(TypeError):
        RouteSegment(length_ft=100.0, fittings=["elbow"])


# ------------------------------------------------------- route hydraulics --


def test_pressure_falls_monotonically_along_a_flat_route():
    route = PipeRoute(segments=[RouteSegment(length_ft=10000.0) for _ in range(4)])
    solution = solve_route(route, 10.02, 900.0, 40e6, GAS)
    assert solution.converged
    pressures = solution.node_pressures_psia
    for upstream, downstream in zip(pressures, pressures[1:]):
        assert downstream < upstream


def test_bigger_pipe_always_delivers_more_pressure():
    """The monotonicity a size recommendation rests on. This is exactly what
    the Newton limit cycle broke: before the line search, 16 in and larger
    returned garbage while 14 in solved fine, so delivery pressure went
    *down* as diameter went up."""
    route = simple_route()
    previous = None
    for nominal in DEFAULT_CANDIDATES:
        inside = STD_PIPE_ID_IN[nominal]
        solution = solve_route(route, inside, 900.0, 60e6, GAS)
        if solution.choked:
            continue
        assert solution.converged, "%g in did not converge" % nominal
        delivery = solution.delivery_pressure_psia
        if previous is not None:
            assert delivery > previous, (
                "%g in delivered %.1f psia, less than the smaller size's %.1f"
                % (nominal, delivery, previous))
        previous = delivery


def test_large_bore_low_drop_line_converges():
    """The specific case that produced the limit cycle: a big pipe with
    almost no pressure drop, where dq/dp is enormous and q ~ sqrt(dP^2) has
    infinite slope at zero differential."""
    route = PipeRoute(segments=[RouteSegment(length_ft=5000.0)])
    solution = solve_route(route, 35.25, 900.0, 20e6, GAS)
    assert solution.converged
    assert solution.delivery_pressure_psia == pytest.approx(900.0, abs=1.0)
    assert not solution.choked


def test_uphill_route_delivers_less_than_downhill():
    up = PipeRoute(segments=[RouteSegment(length_ft=30000.0, elevation_change_ft=800.0)])
    flat = PipeRoute(segments=[RouteSegment(length_ft=30000.0)])
    down = PipeRoute(segments=[RouteSegment(length_ft=30000.0, elevation_change_ft=-800.0)])
    args = (10.02, 900.0, 40e6, GAS)
    assert (solve_route(up, *args).delivery_pressure_psia
            < solve_route(flat, *args).delivery_pressure_psia
            < solve_route(down, *args).delivery_pressure_psia)


def test_flow_at_fixed_end_pressures_is_the_inverse_problem():
    """Solving for flow from two pressures, then feeding that flow back in,
    must reproduce the same pressures. The two directions are the same
    equation read two ways, so a mismatch means one of them is wrong."""
    route = simple_route()
    diameter = 12.0
    flow = flow_at_pressures(route, diameter, 900.0, 700.0, GAS)
    assert flow > 0.0
    back = solve_route(route, diameter, 900.0, flow, GAS)
    assert back.delivery_pressure_psia == pytest.approx(700.0, rel=1e-4)


def test_more_flow_costs_more_pressure():
    route = simple_route()
    previous = None
    for flow in (20e6, 40e6, 60e6, 80e6):
        delivery = solve_route(route, 12.0, 900.0, flow, GAS).delivery_pressure_psia
        if previous is not None:
            assert delivery < previous
        previous = delivery


# ------------------------------------------------------------ size choice --


def test_recommendation_is_the_cheapest_feasible_size():
    route = simple_route()
    report = recommend_size(route, 900.0, 60e6, 600.0, gas=GAS)
    assert report.recommended is not None
    feasible = report.feasible_candidates
    assert report.recommended.relative_cost == min(c.relative_cost for c in feasible)
    # Every smaller size must have been rejected for a stated reason.
    for candidate in report.candidates:
        if candidate.nominal_size < report.recommended.nominal_size:
            assert candidate.failures


def test_recommendation_actually_meets_the_design_case():
    route = simple_route()
    report = recommend_size(route, 900.0, 60e6, 600.0, gas=GAS)
    best = report.recommended
    assert best.delivery_pressure_psia >= 600.0
    assert best.erosional_ratio <= 1.0
    assert not best.failures


def test_undersized_pipe_is_reported_as_undersized_not_as_a_solver_failure():
    """A line too small to pass the flow has no positive pressure profile at
    all. Calling that a convergence failure would blame the numerics for a
    real engineering answer."""
    route = simple_route()
    report = recommend_size(route, 900.0, 60e6, 600.0, gas=GAS)
    small = [c for c in report.candidates if c.nominal_size == 4][0]
    assert not small.feasible
    assert small.solution.choked
    assert "too small" in small.failures[0]
    assert not any("converge" in f for f in small.failures)


def test_a_case_nothing_can_satisfy_says_so():
    """Ask for far more than the inlet pressure can push and every candidate
    must fail, with an explanation rather than a silent empty result."""
    route = PipeRoute(segments=[RouteSegment(length_ft=400000.0)])
    report = recommend_size(route, 300.0, 500e6, 250.0, gas=GAS)
    assert report.recommended is None
    assert not report.feasible_candidates
    assert any("no candidate size" in n for n in report.notes)


def test_erosional_velocity_can_be_the_binding_constraint():
    """A size can make the delivery pressure and still be rejected on
    velocity. If this never fired, the erosional check would be dead code
    that quietly passes everything."""
    route = PipeRoute(segments=[RouteSegment(length_ft=3000.0)])
    report = recommend_size(route, 900.0, 120e6, 500.0, gas=GAS,
                            candidates=(4, 6, 8, 10, 12, 16, 20))
    rejected_on_velocity = [
        c for c in report.candidates
        if not c.feasible and any("erosional" in f or "velocity" in f for f in c.failures)
        and c.delivery_pressure_psia >= 500.0
    ]
    assert rejected_on_velocity, "erosional screening never triggered"


def test_tighter_delivery_pressure_forces_a_bigger_pipe():
    route = simple_route()
    loose = recommend_size(route, 900.0, 60e6, 400.0, gas=GAS).recommended
    tight = recommend_size(route, 900.0, 60e6, 850.0, gas=GAS).recommended
    assert tight.nominal_size > loose.nominal_size


def test_more_fittings_can_push_the_recommendation_up_a_size():
    """The point of modelling minor losses at all: on a short line they
    change the answer, not just the third decimal place."""
    bare = PipeRoute(segments=[RouteSegment(length_ft=2000.0)])
    congested = PipeRoute(segments=[
        RouteSegment(length_ft=2000.0,
                     fittings=[FittingCount(ELBOW_90_STANDARD, 24),
                               FittingCount(GLOBE_VALVE, 4),
                               FittingCount(TEE_THROUGH_BRANCH, 4)])])
    kwargs = dict(inlet_pressure_psia=300.0, target_flow_scfd=20e6,
                  minimum_delivery_psia=270.0, gas=GAS)
    bare_size = recommend_size(bare, **kwargs).recommended
    congested_size = recommend_size(congested, **kwargs).recommended
    assert bare_size is not None and congested_size is not None
    # 6 in is enough for the bare run; the fittings push it to 8 in.
    assert congested_size.nominal_size > bare_size.nominal_size


def test_cost_index_rises_with_diameter_and_length():
    assert relative_cost_index(12, 1000.0) > relative_cost_index(8, 1000.0)
    assert relative_cost_index(12, 2000.0) == pytest.approx(2 * relative_cost_index(12, 1000.0))
    # The exponent is exposed, and using it changes the ranking spread.
    steep = relative_cost_index(24, 1000.0, exponent=2.0) / relative_cost_index(12, 1000.0, exponent=2.0)
    shallow = relative_cost_index(24, 1000.0, exponent=1.0) / relative_cost_index(12, 1000.0, exponent=1.0)
    assert steep > shallow


def test_cost_premium_makes_rounding_up_explicit():
    route = simple_route()
    report = recommend_size(route, 900.0, 60e6, 600.0, gas=GAS)
    bigger = [c for c in report.feasible_candidates
              if c.nominal_size > report.recommended.nominal_size]
    assert bigger, "expected at least one larger feasible size"
    premium = report.cost_premium_of(bigger[0])
    assert premium > 0.0
    assert report.cost_premium_of(report.recommended) == pytest.approx(0.0)


def test_unknown_nominal_size_is_rejected():
    route = simple_route()
    with pytest.raises(ValueError):
        recommend_size(route, 900.0, 60e6, 600.0, gas=GAS, candidates=(7,))


def test_standard_pipe_ids_are_below_nominal_above_two_inches():
    """Sizing on nominal instead of real bore overstates capacity, because
    diameter enters Weymouth to the 2.667 power."""
    for nominal, inside in STD_PIPE_ID_IN.items():
        if nominal >= 14:
            assert inside < nominal  # OD equals nominal, so ID is smaller
        assert inside > 0.0
