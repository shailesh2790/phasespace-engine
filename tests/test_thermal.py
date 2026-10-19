"""Tests for the flowing temperature model.

The load-bearing test here is the comparison against the closed-form solution
of the same balance. That is a real check rather than a self-consistency one,
because the analytic expression is derived on paper and the profile is
integrated numerically -- they share no code path, so agreement means the
integration is right rather than merely repeatable.

The Joule-Thomson tests matter for a different reason. JT cooling is the term a
simplified model drops, and dropping it errs in one direction only: it predicts
a *warmer* arrival temperature than reality, which is precisely the direction
that hides a hydrate risk. So its sign and magnitude are pinned here against
independent physics rather than against a stored number.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.fluids import GasProperties
from phasespace.sim.thermal import (
    DEFAULT_GAS_CP_BTU_LB_F,
    ThermalPath,
    ThermalStream,
    analytic_temperature_r,
    elevation_cooling_r_per_ft,
    hydrate_check,
    hydrate_risk_along,
    hydrate_temperature_r,
    joule_thomson_coefficient_f_per_psi,
    relaxation_length_ft,
    temperature_profile,
)
from phasespace.sim.units import f_to_r, r_to_f

# A 12 in buried gas line, 20 miles, in 50 degF soil.
PATH = ThermalPath(diameter_ft=1.0, overall_u=0.6, ambient_r=f_to_r(50.0))
STREAM = ThermalStream(mass_rate_lb_hr=120_000.0)
LENGTH = 20.0 * 5280.0


# ------------------------------------------------------------ relaxation ----


def test_relaxation_length_is_the_scale_of_the_problem():
    """L = m*Cp/(U*pi*D). Checked against the arithmetic, because every other
    result in this module is an exponential in x/L and so inherits its error."""
    expected = 120_000.0 * DEFAULT_GAS_CP_BTU_LB_F / (0.6 * math.pi * 1.0)
    assert relaxation_length_ft(PATH, STREAM) == pytest.approx(expected, rel=1e-12)


def test_perfect_insulation_gives_an_infinite_relaxation_length():
    """The correct answer, not an error: an insulated line never approaches
    ambient, and code downstream has to branch on that rather than divide by
    zero."""
    insulated = ThermalPath(diameter_ft=1.0, overall_u=0.0, ambient_r=f_to_r(50.0))
    assert math.isinf(relaxation_length_ft(insulated, STREAM))


def test_insulated_line_holds_its_temperature():
    profile = temperature_profile(f_to_r(120.0),
                                  ThermalPath(1.0, 0.0, f_to_r(50.0)),
                                  STREAM, LENGTH)
    assert profile.outlet_r == pytest.approx(f_to_r(120.0), abs=1e-9)
    assert profile.ambient_delta_r == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------- against the closed form ------


def test_matches_the_analytic_solution_with_no_gradient():
    """Numerical integration against the closed form of the same ODE. The
    analytic expression was derived on paper; the profile is stepped
    numerically. Nothing is shared but the physics."""
    inlet = f_to_r(120.0)
    profile = temperature_profile(inlet, PATH, STREAM, LENGTH, segments=200)
    for x, t in zip(profile.distances_ft, profile.temperatures_r):
        # A hundredth of a degree: the physically meaningful statement. The
        # midpoint rule is second order, so a blanket relative bound would be
        # asserting a convergence rate, which has its own test below.
        assert t == pytest.approx(analytic_temperature_r(inlet, PATH, STREAM, x),
                                  abs=0.01), "station %.0f ft" % x


def test_matches_the_analytic_solution_in_a_geothermal_gradient():
    """A producing well: the fluid climbs 10,000 ft through formation that
    cools by 1.2 degF per 100 ft on the way up, so ambient is a moving target.

    The interesting physics is that the stream does not approach ambient at
    all -- it approaches a fixed *offset* from it, lagging by exactly one
    relaxation length's worth of gradient. A model that assumes the fluid tends
    to local ambient gets a producing well's wellhead temperature wrong in the
    cold direction every time.
    """
    well = ThermalPath(diameter_ft=0.2, overall_u=0.5,
                       ambient_r=f_to_r(200.0),
                       ambient_gradient_r_per_ft=-0.012,
                       inclination_deg=90.0)
    stream = ThermalStream(mass_rate_lb_hr=30_000.0, cp_btu_lb_f=0.6)
    inlet = f_to_r(200.0)
    profile = temperature_profile(inlet, well, stream, 10_000.0, segments=400)
    for x, t in zip(profile.distances_ft, profile.temperatures_r):
        assert t == pytest.approx(
            analytic_temperature_r(inlet, well, stream, x), abs=0.01), (
            "station %.0f ft" % x)


def test_stream_lags_the_gradient_by_one_relaxation_length():
    """The offset in the closed form, stated as the physical claim it encodes."""
    well = ThermalPath(diameter_ft=0.2, overall_u=0.5, ambient_r=f_to_r(200.0),
                       ambient_gradient_r_per_ft=-0.012)
    stream = ThermalStream(mass_rate_lb_hr=30_000.0, cp_btu_lb_f=0.6)
    length = relaxation_length_ft(well, stream)

    # Far enough along that the exponential has died, the gap to local ambient
    # settles at -L * gradient.
    far = 40.0 * length
    t = analytic_temperature_r(f_to_r(200.0), well, stream, far)
    assert t - well.ambient_at(far) == pytest.approx(
        -length * well.ambient_gradient_r_per_ft, rel=1e-6)


def test_integration_is_stable_on_a_coarse_grid():
    """Explicit Euler oscillates once the step approaches 2L, and the
    oscillation looks like a plausible physical wave rather than a failure.
    The midpoint rule has to stay monotone even at one segment per relaxation
    length."""
    inlet = f_to_r(200.0)
    coarse = temperature_profile(inlet, PATH, STREAM, LENGTH, segments=3)
    temperatures = coarse.temperatures_r
    assert all(a >= b for a, b in zip(temperatures, temperatures[1:])), temperatures
    assert min(temperatures) >= PATH.ambient_r - 1e-9, "overshot ambient"


# ------------------------------------------------------- Joule-Thomson ------


def test_joule_thomson_cools_natural_gas_at_pipeline_conditions():
    """Sign and magnitude against independent knowledge, not a stored value.

    Natural gas at transmission pressures sits below its inversion temperature,
    so it cools on expansion; published coefficients are a few hundredths of a
    degree F per psi.
    """
    gas = GasProperties(gravity=0.65, temperature_r=f_to_r(80.0))
    mu = joule_thomson_coefficient_f_per_psi(gas, 1000.0, f_to_r(80.0))
    assert mu > 0.0, "natural gas should cool on expansion at 1000 psia"
    assert 0.005 < mu < 0.10, "%.4f degF/psi is outside the published range" % mu


def test_joule_thomson_vanishes_for_an_ideal_gas():
    """An ideal gas has dZ/dT = 0 and does not JT cool. This is the failure
    mode being guarded against: treat the gas as ideal and the model reports a
    warmer arrival than reality, in the direction that hides hydrate risk."""
    ideal = GasProperties(gravity=0.65, temperature_r=f_to_r(80.0), fixed_z=1.0)
    assert joule_thomson_coefficient_f_per_psi(
        ideal, 1000.0, f_to_r(80.0)) == pytest.approx(0.0, abs=1e-12)


def test_joule_thomson_alone_gives_exactly_mu_times_delta_p():
    """With no heat exchange, the temperature change is the coefficient times
    the pressure change -- by definition, so this pins the bookkeeping."""
    insulated = ThermalPath(diameter_ft=1.0, overall_u=0.0, ambient_r=f_to_r(50.0))
    stations = 21
    pressures = [1000.0 - 20.0 * i for i in range(stations)]
    profile = temperature_profile(f_to_r(120.0), insulated, STREAM, LENGTH,
                                  pressures_psia=pressures, segments=stations - 1,
                                  joule_thomson_f_per_psi=0.03)
    expected = 0.03 * (pressures[-1] - pressures[0])
    assert profile.joule_thomson_delta_r == pytest.approx(expected, rel=1e-12)
    assert profile.outlet_r - profile.inlet_r == pytest.approx(expected, rel=1e-9)


def test_joule_thomson_can_dominate_ambient_exchange():
    """On a well-insulated long line JT is the larger term. A model that keeps
    only ambient exchange concludes the line arrives at soil temperature; the
    truth is that it arrives colder, which is the whole hydrate question."""
    insulated = ThermalPath(diameter_ft=1.0, overall_u=0.05, ambient_r=f_to_r(50.0))
    stations = 101
    pressures = [1200.0 - 4.0 * i for i in range(stations)]
    profile = temperature_profile(f_to_r(60.0), insulated, STREAM, LENGTH,
                                  pressures_psia=pressures, segments=stations - 1,
                                  joule_thomson_f_per_psi=0.03)
    assert abs(profile.joule_thomson_delta_r) > abs(profile.ambient_delta_r)
    assert profile.outlet_r < insulated.ambient_r, (
        "the line should arrive below soil temperature, not at it")


# ---------------------------------------------------------- elevation -------


def test_lifting_a_fluid_costs_it_temperature():
    """10,000 ft of lift against a gas heat capacity is tens of degrees -- the
    same order as the other terms, so it cannot be dropped in a well."""
    well = ThermalPath(diameter_ft=0.2, overall_u=0.0, ambient_r=f_to_r(200.0),
                       inclination_deg=90.0)
    stream = ThermalStream(mass_rate_lb_hr=30_000.0, cp_btu_lb_f=0.6)
    rate = elevation_cooling_r_per_ft(well, stream)
    assert rate == pytest.approx(1.0 / (0.6 * 778.169), rel=1e-12)

    profile = temperature_profile(f_to_r(200.0), well, stream, 10_000.0)
    assert profile.elevation_delta_r == pytest.approx(-10_000.0 * rate, rel=1e-9)
    assert 15.0 < -profile.elevation_delta_r < 30.0


def test_a_horizontal_path_has_no_elevation_term():
    assert elevation_cooling_r_per_ft(PATH, STREAM) == pytest.approx(0.0, abs=1e-15)


# ------------------------------------------------------------- hydrates -----


def test_hammerschmidt_against_the_published_form():
    """T = 8.9 * P^0.285, checked against the arithmetic rather than a table."""
    t_r, extrapolated = hydrate_temperature_r(1000.0)
    assert r_to_f(t_r) == pytest.approx(8.9 * 1000.0 ** 0.285, rel=1e-12)
    assert not extrapolated


def test_hydrate_correlation_says_when_it_is_extrapolating():
    """A hydrate number that does not admit it is guessing is worse than none:
    it is acted on."""
    assert hydrate_temperature_r(150.0)[1] is True
    assert hydrate_temperature_r(6000.0)[1] is True
    assert hydrate_temperature_r(1500.0)[1] is False


def test_hydrate_temperature_rises_with_pressure():
    """Higher pressure forms hydrates at higher temperature, which is why a
    line can be safe at the outlet and not at the inlet."""
    temperatures = [hydrate_temperature_r(p)[0] for p in (400.0, 800.0, 1600.0)]
    assert temperatures[0] < temperatures[1] < temperatures[2]


def test_a_warm_line_can_still_plug_downstream():
    """The case the thermal model exists to catch. The gas leaves at 120 degF,
    comfortably clear of the hydrate curve, and is inside it well before the
    end -- so an isothermal model reports a safe line and a real one plugs."""
    stations = 81
    pressures = [1400.0 - 8.0 * i for i in range(stations)]
    profile = temperature_profile(f_to_r(120.0), PATH, STREAM, LENGTH,
                                  pressures_psia=pressures, segments=stations - 1,
                                  joule_thomson_f_per_psi=0.03)
    risk = hydrate_risk_along(profile)

    assert not risk[0].forms, "should be clear of the curve at the inlet"
    assert risk[-1].forms, "should be inside the curve by the outlet"

    first = next(i for i, check in enumerate(risk) if check.forms)
    assert 0 < first < stations - 1
    # And an isothermal model, holding the inlet temperature, sees no problem.
    assert hydrate_check(pressures[-1], f_to_r(120.0)).forms is False


def test_hydrate_margin_is_signed_the_intuitive_way():
    check = hydrate_check(1000.0, f_to_r(120.0))
    assert check.margin_f > 0.0 and not check.forms
    cold = hydrate_check(1000.0, f_to_r(40.0))
    assert cold.margin_f < 0.0 and cold.forms


# ------------------------------------------------------------ validation ----


def test_energy_balance_closes():
    """Whatever the three mechanisms did, they have to add up to the observed
    change. This catches a term being counted twice or dropped, which is the
    likeliest bookkeeping error in a model with several additive effects."""
    stations = 51
    pressures = [1000.0 - 6.0 * i for i in range(stations)]
    well = ThermalPath(diameter_ft=0.5, overall_u=0.4, ambient_r=f_to_r(70.0),
                       inclination_deg=30.0)
    profile = temperature_profile(f_to_r(150.0), well, STREAM, 8000.0,
                                  pressures_psia=pressures, segments=stations - 1,
                                  joule_thomson_f_per_psi=0.02)
    total = (profile.ambient_delta_r + profile.joule_thomson_delta_r
             + profile.elevation_delta_r)
    assert profile.outlet_r - profile.inlet_r == pytest.approx(total, rel=1e-9)


def test_refining_the_grid_converges():
    """Halving the step should roughly quarter the error for a midpoint rule.
    Asserting the trend rather than a tolerance catches an order-of-accuracy
    regression that a single loose tolerance would hide."""
    inlet = f_to_r(200.0)
    exact = analytic_temperature_r(inlet, PATH, STREAM, LENGTH)
    errors = []
    for segments in (10, 20, 40):
        profile = temperature_profile(inlet, PATH, STREAM, LENGTH, segments=segments)
        errors.append(abs(profile.outlet_r - exact))
    assert errors[1] < errors[0] / 3.0
    assert errors[2] < errors[1] / 3.0


def test_temperature_at_interpolates_between_stations():
    profile = temperature_profile(f_to_r(120.0), PATH, STREAM, LENGTH, segments=10)
    midpoint = profile.temperature_at(0.5 * (profile.distances_ft[0]
                                             + profile.distances_ft[1]))
    assert (min(profile.temperatures_r[0], profile.temperatures_r[1]) <= midpoint
            <= max(profile.temperatures_r[0], profile.temperatures_r[1]))
    assert profile.temperature_at(-100.0) == profile.temperatures_r[0]
    assert profile.temperature_at(LENGTH * 2) == profile.temperatures_r[-1]


# -------------------------------------------------------------- guards ------


def test_nonsense_inputs_are_rejected():
    for kwargs in ({"diameter_ft": 0.0}, {"overall_u": -1.0}, {"ambient_r": 0.0},
                   {"inclination_deg": 120.0}):
        args = {"diameter_ft": 1.0, "overall_u": 0.5, "ambient_r": f_to_r(50.0)}
        args.update(kwargs)
        with pytest.raises(ValueError):
            ThermalPath(**args)

    with pytest.raises(ValueError):
        ThermalStream(mass_rate_lb_hr=0.0)
    with pytest.raises(ValueError):
        ThermalStream(mass_rate_lb_hr=100.0, cp_btu_lb_f=0.0)
    with pytest.raises(ValueError):
        temperature_profile(f_to_r(100.0), PATH, STREAM, 0.0)
    with pytest.raises(ValueError):
        temperature_profile(f_to_r(100.0), PATH, STREAM, LENGTH, segments=0)


def test_pressure_list_length_is_checked():
    """A mismatched pressure list would silently shift the JT term along the
    line, which is the kind of error that produces a plausible wrong answer."""
    with pytest.raises(ValueError):
        temperature_profile(f_to_r(100.0), PATH, STREAM, LENGTH,
                            pressures_psia=[1000.0, 900.0], segments=10)
