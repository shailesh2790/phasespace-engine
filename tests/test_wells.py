"""Tests for well inflow, outflow, nodal analysis, and liquid loading.

Same principle as the network tests: check against closed-form answers,
hand-calculable limits, and physical monotonicity, never against the
module's own previous output.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.wells import (
    BEGGS_BRILL,
    HAGEDORN_BROWN,
    CompositeIPR,
    FetkovichIPR,
    GasBackpressureIPR,
    ProductionStream,
    ProductivityIndexIPR,
    Tubing,
    VogelIPR,
    beggs_brill_gradient,
    coleman_critical_velocity_ft_s,
    gas_lift_performance_curve,
    hagedorn_brown_gradient,
    liquid_loading_check,
    nodal_analysis,
    pressure_traverse,
    turner_critical_velocity_ft_s,
)

OIL = LiquidProperties(api_gravity=32.0, temperature_r=640.0,
                       bubble_point_psia=2200.0, gas_gravity=0.75)
GAS = GasProperties(gravity=0.75, temperature_r=640.0)
TUBING = Tubing(depth_ft=8000.0, inner_diameter_in=2.441, inclination_deg=90.0,
                wellhead_temperature_f=110.0, bottomhole_temperature_f=200.0)


# ------------------------------------------------------------------ IPR ----


def test_vogel_matches_the_published_equation():
    ipr = VogelIPR(reservoir_pressure_psia=3000.0, max_rate=2000.0)
    x = 1500.0 / 3000.0
    expected = 2000.0 * (1.0 - 0.2 * x - 0.8 * x * x)
    assert ipr.rate(1500.0) == pytest.approx(expected)
    assert ipr.rate(3000.0) == pytest.approx(0.0)
    assert ipr.absolute_open_flow() == pytest.approx(2000.0)


def test_productivity_index_ipr_is_linear():
    ipr = ProductivityIndexIPR(reservoir_pressure_psia=3000.0, productivity_index=1.5)
    assert ipr.rate(2000.0) == pytest.approx(1500.0)
    assert ipr.rate(3000.0) == pytest.approx(0.0)
    # Never negative above reservoir pressure -- that would be injection, and
    # an IPR silently returning a negative rate would flip the sign of every
    # mass balance downstream of it.
    assert ipr.rate(3500.0) == 0.0


def test_composite_ipr_is_continuous_and_smooth_at_the_bubble_point():
    """A kink here would put a false local optimum into any gas-lift or
    sizing optimisation running on top of this curve."""
    ipr = CompositeIPR(reservoir_pressure_psia=3000.0, bubble_point_psia=2200.0,
                       productivity_index=1.5)
    p_b = 2200.0
    assert ipr.rate(p_b) == pytest.approx(1200.0)  # J * (Pr - Pb)

    h = 1.0
    left = ipr.rate(p_b + h)
    right = ipr.rate(p_b - h)
    # Value continuous
    assert abs(left - right) < 0.01 * ipr.rate(p_b) / 100.0 + 5.0
    # Slope continuous: both sides must show the same dq/dp as J
    slope_above = (ipr.rate(p_b) - left) / h
    slope_below = (right - ipr.rate(p_b)) / h
    assert slope_above == pytest.approx(1.5, rel=1e-6)
    assert slope_below == pytest.approx(1.5, rel=1e-3)


def test_composite_ipr_reduces_to_a_straight_line_when_undersaturated():
    """With the bubble point at zero there is no saturated region, so the
    composite must collapse exactly onto the straight-line IPR."""
    composite = CompositeIPR(reservoir_pressure_psia=3000.0, bubble_point_psia=1e-9,
                             productivity_index=2.0)
    linear = ProductivityIndexIPR(reservoir_pressure_psia=3000.0, productivity_index=2.0)
    for p_wf in (2800.0, 2000.0, 1000.0, 100.0):
        assert composite.rate(p_wf) == pytest.approx(linear.rate(p_wf), rel=1e-6)


def test_ipr_curves_are_monotonically_decreasing_in_bottomhole_pressure():
    iprs = [
        ProductivityIndexIPR(3000.0, 1.5),
        VogelIPR(3000.0, 2000.0),
        CompositeIPR(3000.0, 2200.0, 1.5),
        FetkovichIPR(3000.0, c=1e-4, n=0.9),
        GasBackpressureIPR(3000.0, c=5e-4, n=0.85),
    ]
    for ipr in iprs:
        rates = [ipr.rate(p) for p in (3000.0, 2500.0, 2000.0, 1000.0, 0.0)]
        for earlier, later in zip(rates, rates[1:]):
            assert later >= earlier, "%s is not monotone" % type(ipr).__name__


def test_gas_backpressure_exponent_behaves():
    """n = 1 is laminar, n = 0.5 fully turbulent. At the same C, the more
    turbulent well delivers less at large drawdown."""
    laminar = GasBackpressureIPR(3000.0, c=1e-3, n=1.0)
    turbulent = GasBackpressureIPR(3000.0, c=1e-3, n=0.5)
    assert laminar.rate(500.0) > turbulent.rate(500.0)


# ------------------------------------------------------- pressure traverse --


def test_dead_oil_column_matches_the_hydrostatic_hand_calc():
    """With no gas, the tubing is a static liquid column plus a little
    friction. 0.433 psi/ft times specific gravity is the number an engineer
    would write on the back of an envelope, and the traverse must land just
    above it."""
    stream = ProductionStream(oil_rate_stb_d=1000.0, gas_rate_scf_d=0.0, oil=OIL, gas=GAS)
    result = pressure_traverse(200.0, stream, TUBING, correlation=BEGGS_BRILL, segments=40)
    hydrostatic = 200.0 + 0.4333 * OIL.oil_specific_gravity * TUBING.depth_ft
    assert result.bottomhole_pressure_psia > hydrostatic
    assert result.bottomhole_pressure_psia < hydrostatic * 1.05
    assert result.elevation_fraction > 0.95


def test_zero_rate_column_is_purely_hydrostatic():
    stream = ProductionStream(oil_rate_stb_d=0.0, gas_rate_scf_d=0.0, oil=OIL, gas=GAS)
    result = pressure_traverse(200.0, stream, TUBING, segments=20)
    assert result.friction_fraction == pytest.approx(0.0, abs=1e-9)
    assert result.elevation_fraction == pytest.approx(1.0)


def test_gas_lightens_the_column():
    """More produced gas means a lighter column and less bottomhole pressure
    for the same liquid rate -- the entire physical basis of gas lift."""
    previous = None
    for gor in (0.0, 200.0, 500.0, 1000.0):
        stream = ProductionStream(oil_rate_stb_d=1000.0, gas_rate_scf_d=1000.0 * gor,
                                  oil=OIL, gas=GAS)
        bhp = pressure_traverse(200.0, stream, TUBING, segments=30).bottomhole_pressure_psia
        if previous is not None:
            assert bhp < previous
        previous = bhp


def test_vlp_curve_is_j_shaped():
    """The defining feature of a VLP curve: bottomhole pressure falls with
    rate while the column is lightening, then rises again once friction
    dominates. The turning point is what makes nodal analysis need a
    bracketing root find rather than Newton."""
    rates = [200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    bhps = []
    for q in rates:
        stream = ProductionStream(oil_rate_stb_d=q, gas_rate_scf_d=q * 500.0,
                                  oil=OIL, gas=GAS)
        bhps.append(pressure_traverse(200.0, stream, TUBING, segments=30).bottomhole_pressure_psia)
    minimum = min(bhps)
    index = bhps.index(minimum)
    assert 0 < index < len(bhps) - 1, "no interior minimum: %s" % bhps
    assert bhps[0] > minimum and bhps[-1] > minimum


def test_friction_share_grows_with_rate():
    low = ProductionStream(oil_rate_stb_d=200.0, gas_rate_scf_d=100000.0, oil=OIL, gas=GAS)
    high = ProductionStream(oil_rate_stb_d=8000.0, gas_rate_scf_d=4000000.0, oil=OIL, gas=GAS)
    low_result = pressure_traverse(200.0, low, TUBING, segments=30)
    high_result = pressure_traverse(200.0, high, TUBING, segments=30)
    assert high_result.friction_fraction > low_result.friction_fraction
    assert low_result.elevation_fraction > 0.9


def test_traverse_is_converged_in_the_number_of_segments():
    """40 segments has to be enough. If the answer still moves materially
    between 40 and 200 segments the step size, not the physics, is setting
    the result."""
    stream = ProductionStream(oil_rate_stb_d=1500.0, gas_rate_scf_d=1500.0 * 800.0,
                              oil=OIL, gas=GAS)
    coarse = pressure_traverse(200.0, stream, TUBING, segments=40).bottomhole_pressure_psia
    fine = pressure_traverse(200.0, stream, TUBING, segments=200).bottomhole_pressure_psia
    assert abs(coarse - fine) / fine < 0.01


def test_horizontal_section_has_no_elevation_gradient():
    horizontal = Tubing(depth_ft=4000.0, inner_diameter_in=4.0, inclination_deg=0.0)
    stream = ProductionStream(oil_rate_stb_d=2000.0, gas_rate_scf_d=1e6, oil=OIL, gas=GAS)
    gradient = beggs_brill_gradient(800.0, 150.0, stream, horizontal)
    assert gradient["elevation_psi_ft"] == pytest.approx(0.0, abs=1e-12)
    assert gradient["friction_psi_ft"] > 0.0


def test_holdup_never_falls_below_no_slip_in_either_correlation():
    """Holdup below the no-slip value means the gas is moving slower than
    the liquid in upward flow, which does not happen. Hagedorn-Brown's chart
    fit will return exactly that without the published floor applied."""
    for gor in (100.0, 300.0, 800.0, 2000.0, 6000.0):
        stream = ProductionStream(oil_rate_stb_d=1000.0, gas_rate_scf_d=1000.0 * gor,
                                  oil=OIL, gas=GAS)
        for p in (150.0, 400.0, 1200.0, 2500.0):
            for fn in (beggs_brill_gradient, hagedorn_brown_gradient):
                out = fn(p, 150.0, stream, TUBING)
                v_sl = out["superficial_liquid_velocity_ft_s"]
                v_sg = out["superficial_gas_velocity_ft_s"]
                lambda_l = v_sl / (v_sl + v_sg) if (v_sl + v_sg) > 0 else 1.0
                assert out["holdup"] >= lambda_l - 1e-9, (
                    "%s gave holdup %.4f below no-slip %.4f (GOR %s, p %s)"
                    % (fn.__name__, out["holdup"], lambda_l, gor, p))
                assert 0.0 < out["holdup"] <= 1.0


def test_correlations_agree_closely_when_there_is_no_free_gas():
    """With a single-phase liquid column the two correlations reduce to the
    same hydrostatic answer, so any disagreement there is an implementation
    difference rather than a modelling one."""
    stream = ProductionStream(oil_rate_stb_d=800.0, gas_rate_scf_d=0.0, oil=OIL, gas=GAS)
    bb = pressure_traverse(200.0, stream, TUBING, BEGGS_BRILL, 30).bottomhole_pressure_psia
    hb = pressure_traverse(200.0, stream, TUBING, HAGEDORN_BROWN, 30).bottomhole_pressure_psia
    assert abs(bb - hb) / bb < 0.02


def test_unknown_correlation_is_rejected():
    stream = ProductionStream(oil_rate_stb_d=100.0, oil=OIL, gas=GAS)
    with pytest.raises(ValueError):
        pressure_traverse(200.0, stream, TUBING, correlation="orkiszewski")


def test_tubing_rejects_nonsense_geometry():
    with pytest.raises(ValueError):
        Tubing(depth_ft=8000.0, inner_diameter_in=0.0)
    with pytest.raises(ValueError):
        Tubing(depth_ft=-1.0, inner_diameter_in=2.441)


# -------------------------------------------------------- nodal analysis ----


def test_nodal_analysis_finds_a_consistent_operating_point():
    """The defining property of the answer: at the operating rate, the
    pressure the reservoir supplies and the pressure the tubing needs are
    the same number. Checked by recomputing both independently."""
    ipr = CompositeIPR(3000.0, 2200.0, 1.5)
    result = nodal_analysis(ipr=ipr, tubing=TUBING, wellhead_pressure_psia=200.0,
                            gas_oil_ratio_scf_stb=600.0, water_cut=0.2,
                            oil=OIL, gas=GAS)
    assert result.converged
    assert result.oil_rate_stb_d > 0.0

    q = result.oil_rate_stb_d
    q_water = q * 0.2 / 0.8
    stream = ProductionStream(oil_rate_stb_d=q, water_rate_stb_d=q_water,
                              gas_rate_scf_d=q * 600.0, oil=OIL, gas=GAS)
    p_vlp = pressure_traverse(200.0, stream, TUBING, segments=30).bottomhole_pressure_psia
    assert p_vlp == pytest.approx(result.bottomhole_pressure_psia, rel=0.02)
    assert ipr.rate(result.bottomhole_pressure_psia) == pytest.approx(q, rel=0.02)


def test_water_cut_is_reflected_in_the_liquid_rate():
    ipr = CompositeIPR(3000.0, 2200.0, 1.5)
    result = nodal_analysis(ipr=ipr, tubing=TUBING, wellhead_pressure_psia=200.0,
                            gas_oil_ratio_scf_stb=600.0, water_cut=0.25,
                            oil=OIL, gas=GAS)
    assert result.converged
    assert result.liquid_rate_stb_d == pytest.approx(result.oil_rate_stb_d / 0.75, rel=1e-6)


def test_raising_wellhead_pressure_chokes_the_well_back():
    ipr = CompositeIPR(3000.0, 2200.0, 1.5)
    previous_rate = None
    for p_wh in (100.0, 200.0, 400.0):
        result = nodal_analysis(ipr=ipr, tubing=TUBING, wellhead_pressure_psia=p_wh,
                                gas_oil_ratio_scf_stb=600.0, water_cut=0.2,
                                oil=OIL, gas=GAS)
        assert result.converged
        if previous_rate is not None:
            assert result.oil_rate_stb_d < previous_rate
        previous_rate = result.oil_rate_stb_d


def test_a_well_that_cannot_flow_says_so_rather_than_inventing_a_rate():
    """A dead well must come back not-converged with an explanation. Silently
    returning a small positive rate here is exactly the confidently-wrong
    behaviour PRD Section 6 rules out."""
    weak = CompositeIPR(reservoir_pressure_psia=900.0, bubble_point_psia=800.0,
                        productivity_index=0.05)
    result = nodal_analysis(ipr=weak, tubing=TUBING, wellhead_pressure_psia=600.0,
                            gas_oil_ratio_scf_stb=50.0, oil=OIL, gas=GAS)
    assert not result.converged
    assert result.oil_rate_stb_d == 0.0
    assert result.notes


def test_a_richer_reservoir_produces_more():
    strong = CompositeIPR(3500.0, 2200.0, 2.5)
    weak = CompositeIPR(2800.0, 2200.0, 1.0)
    kwargs = dict(tubing=TUBING, wellhead_pressure_psia=200.0,
                  gas_oil_ratio_scf_stb=600.0, oil=OIL, gas=GAS)
    strong_result = nodal_analysis(ipr=strong, **kwargs)
    weak_result = nodal_analysis(ipr=weak, **kwargs)
    assert strong_result.converged and weak_result.converged
    assert strong_result.oil_rate_stb_d > weak_result.oil_rate_stb_d


# --------------------------------------------------------- liquid loading ---


def test_turner_velocity_falls_as_pressure_rises():
    """Denser gas lifts droplets more easily, so the critical *velocity*
    drops with pressure -- while the critical standard-condition *rate*
    rises, because the same velocity now moves far more standard volume.
    Getting these two backwards is an easy and consequential mistake."""
    gas = GasProperties(gravity=0.6, temperature_r=560.0)
    v_low = turner_critical_velocity_ft_s(200.0, 560.0, gas)
    v_high = turner_critical_velocity_ft_s(1500.0, 560.0, gas)
    assert v_high < v_low

    tubing = Tubing(depth_ft=8000.0, inner_diameter_in=2.441)
    low = liquid_loading_check(1e6, 200.0, tubing, gas)
    high = liquid_loading_check(1e6, 1500.0, tubing, gas)
    assert high.critical_rate_scf_d > low.critical_rate_scf_d


def test_coleman_is_the_unadjusted_turner_form():
    gas = GasProperties(gravity=0.6, temperature_r=560.0)
    turner = turner_critical_velocity_ft_s(300.0, 560.0, gas)
    coleman = coleman_critical_velocity_ft_s(300.0, 560.0, gas)
    assert coleman == pytest.approx(turner / 1.2, rel=1e-9)


def test_loading_check_flags_a_low_rate_well():
    gas = GasProperties(gravity=0.6, temperature_r=560.0)
    tubing = Tubing(depth_ft=8000.0, inner_diameter_in=2.441)
    loaded = liquid_loading_check(0.3e6, 400.0, tubing, gas)
    flowing = liquid_loading_check(3.0e6, 400.0, tubing, gas)
    assert loaded.is_loading
    assert loaded.margin < 1.0
    assert not flowing.is_loading
    assert flowing.margin > 1.0


def test_critical_rate_for_a_typical_string_is_in_the_right_ballpark():
    """Turner's critical rate for 2-3/8 in tubing at a few hundred psi lands
    around 1 MMscfd in the field. An answer off by an order of magnitude
    means a unit conversion is wrong."""
    gas = GasProperties(gravity=0.6, temperature_r=560.0)
    tubing = Tubing(depth_ft=8000.0, inner_diameter_in=2.441)
    check = liquid_loading_check(1e6, 400.0, tubing, gas)
    assert 0.4e6 < check.critical_rate_scf_d < 2.5e6


def test_smaller_tubing_needs_less_gas_to_stay_unloaded():
    """The standard field remedy for a loading well is a velocity string --
    smaller tubing, same gas, higher velocity."""
    gas = GasProperties(gravity=0.6, temperature_r=560.0)
    big = Tubing(depth_ft=8000.0, inner_diameter_in=3.958)
    small = Tubing(depth_ft=8000.0, inner_diameter_in=1.995)
    assert (liquid_loading_check(1e6, 400.0, small, gas).critical_rate_scf_d
            < liquid_loading_check(1e6, 400.0, big, gas).critical_rate_scf_d)


# --------------------------------------------------------------- gas lift ---


def test_gas_lift_curve_has_an_interior_optimum():
    """Injecting more gas lightens the column but adds friction; past the
    optimum the well makes less oil for more gas. A curve that rises forever
    would mean the friction term is missing, and would have an optimiser
    recommending unlimited injection."""
    ipr = CompositeIPR(2600.0, 2000.0, 1.2)
    rates = [0.25e6, 0.5e6, 1.0e6, 1.5e6, 2.0e6, 3.0e6, 4.0e6]
    curve = gas_lift_performance_curve(ipr, TUBING, 150.0, 150.0, rates,
                                       water_cut=0.4, oil=OIL, gas=GAS)
    oil_rates = [p.oil_rate_stb_d for p in curve]
    assert all(q > 0.0 for q in oil_rates)
    peak = oil_rates.index(max(oil_rates))
    assert 0 < peak < len(oil_rates) - 1, "no interior optimum: %s" % oil_rates
    # And the fall-off past the peak is real, not numerical noise.
    assert oil_rates[-1] < max(oil_rates) * 0.98
