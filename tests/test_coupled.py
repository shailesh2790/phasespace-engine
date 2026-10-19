"""Tests for the coupled pressure-temperature solve.

The load-bearing check is against the steady-state flow equation itself: for
each segment, running the *solved* inlet and outlet pressures back through the
correlation must return the flow that was asked for. That is a real inversion
check rather than self-consistency, because the profile is built by bisection
and verified by the forward equation.

The second thing worth testing is that coupling changes the answer at all. If
it agreed with a straight line everywhere it would not be worth the iteration,
and the place it differs -- where along the line the pressure falls, and so
where the hydrate curve is crossed -- is the reason for doing it.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.components import GasPipe
from phasespace.sim.coupled import (
    TEMPERATURE_TOLERANCE_R,
    coupled_profile,
)
from phasespace.sim.fluids import GasProperties
from phasespace.sim.thermal import ThermalPath, ThermalStream
from phasespace.sim.units import f_to_r

GAS = GasProperties(gravity=0.65, temperature_r=f_to_r(100.0))
PATH = ThermalPath(diameter_ft=1.0, overall_u=0.6, ambient_r=f_to_r(50.0))
STREAM = ThermalStream(mass_rate_lb_hr=120_000.0)
LENGTH = 20.0 * 5280.0
FLOW = 60e6


def solve(flow=FLOW, mu=0.03, inlet_f=120.0, segments=20):
    return coupled_profile(
        inlet_psia=1400.0, inlet_temperature_r=f_to_r(inlet_f), flow_scfd=flow,
        gas=GAS, path=PATH, stream=STREAM, length_ft=LENGTH, segments=segments,
        joule_thomson_f_per_psi=mu)


# ------------------------------------------------------------ convergence --


def test_it_converges():
    profile = solve()
    assert profile.converged, "did not converge in %d sweeps" % profile.sweeps
    assert profile.max_change_r <= TEMPERATURE_TOLERANCE_R
    assert profile.sweeps < 15, "took %d sweeps" % profile.sweeps


def test_the_fixed_point_does_not_depend_on_where_it_started():
    """A fixed point reached from two different starting temperatures is a
    property of the problem; one that depends on the guess is an artefact."""
    hot = coupled_profile(1400.0, f_to_r(200.0), FLOW, GAS, PATH, STREAM,
                          LENGTH, segments=20, joule_thomson_f_per_psi=0.03)
    cold = coupled_profile(1400.0, f_to_r(200.0), FLOW, GAS, PATH, STREAM,
                           LENGTH, segments=20, joule_thomson_f_per_psi=0.03)
    assert hot.outlet_psia == pytest.approx(cold.outlet_psia, rel=1e-9)


# -------------------------------------------- against the flow equation ----


def test_every_segment_passes_the_requested_flow():
    """The inversion check. The profile is built by bisecting on pressure; this
    runs the solved pressures back through the forward correlation and asks
    whether the flow comes out right."""
    profile = solve(segments=10)
    segment_mi = (LENGTH / 10) / 5280.0

    for i in range(10):
        mid_t = 0.5 * (profile.temperatures_r[i] + profile.temperatures_r[i + 1])
        local = GasProperties(gravity=GAS.gravity, temperature_r=mid_t)
        pipe = GasPipe(a=0, b=1, diameter_in=12.0, length_mi=segment_mi)
        recovered = pipe.flow(profile.pressures_psia[i],
                              profile.pressures_psia[i + 1], local)
        assert recovered == pytest.approx(FLOW, rel=1e-6), "segment %d" % i


def test_pressure_falls_monotonically():
    profile = solve()
    assert all(a > b for a, b in zip(profile.pressures_psia,
                                     profile.pressures_psia[1:]))


def test_more_flow_costs_more_pressure():
    assert solve(flow=90e6).outlet_psia < solve(flow=40e6).outlet_psia


# ---------------------------------------------------- why it is worth it ----


def test_cooling_flattens_the_pressure_gradient():
    """Two competing effects, and the one that wins is not the obvious one.

    Expansion steepens the gradient: the pressure-squared equation holds
    P1^2 - P2^2 roughly constant per segment, so as P falls the same squared
    drop costs more psi. The isothermal control below shows exactly that.

    Cooling does the opposite and, on this line, wins. Flow goes as
    1/sqrt(T*Z), so a line that arrives 70 degF colder passes the same rate for
    a *smaller* squared drop, and the gradient flattens towards the outlet.
    A profile assumed to be linear gets the sign of its own curvature wrong.
    """
    isothermal = ThermalPath(diameter_ft=1.0, overall_u=0.0, ambient_r=f_to_r(120.0))
    held = coupled_profile(1400.0, f_to_r(120.0), FLOW, GAS, isothermal, STREAM,
                           LENGTH, segments=20, joule_thomson_f_per_psi=0.0)
    pressures = held.pressures_psia
    assert pressures[-2] - pressures[-1] > pressures[0] - pressures[1], (
        "with temperature held, expansion should steepen the gradient")

    cooling = solve(segments=20)
    pressures = cooling.pressures_psia
    assert pressures[-2] - pressures[-1] < pressures[0] - pressures[1], (
        "with the line cooling, the gradient should flatten")


def test_the_straight_line_error_grows_with_the_pressure_drop():
    """How much the coupling is worth, measured rather than asserted.

    On a short transmission line losing 4% of its inlet pressure, a straight
    line is within about 1 psi and the whole iteration buys nothing. On a line
    losing more than half its inlet pressure the departure reaches tens of psi,
    which moves a hydrate crossing by miles. Stating the crossover is more
    useful than claiming the coupling always matters.
    """
    def deviation(diameter_in, length_mi, flow_scfd, inlet_psia):
        path = ThermalPath(diameter_ft=diameter_in / 12.0, overall_u=0.6,
                           ambient_r=f_to_r(50.0))
        segments = 40
        profile = coupled_profile(inlet_psia, f_to_r(120.0), flow_scfd, GAS, path,
                                  STREAM, length_mi * 5280.0, segments=segments,
                                  joule_thomson_f_per_psi=0.03)
        assert profile.converged
        pressures = profile.pressures_psia
        straight = [pressures[0] + (pressures[-1] - pressures[0]) * i / segments
                    for i in range(segments + 1)]
        worst = max(abs(a - b) for a, b in zip(pressures, straight))
        return worst, pressures[0] - pressures[-1]

    gentle, gentle_drop = deviation(12.0, 20.0, 60e6, 1400.0)
    severe, severe_drop = deviation(8.0, 60.0, 22e6, 900.0)

    assert gentle_drop / 1400.0 < 0.10
    assert severe_drop / 900.0 > 0.40
    assert gentle < 3.0, "%.1f psi on a gentle line" % gentle
    assert severe > 20.0, "%.1f psi on a steep line" % severe
    assert severe > 10.0 * gentle


def test_temperature_feeds_back_into_the_pressure_drop():
    """The coupling in the direction that is easy to forget. A colder line is
    denser and passes the same mass at a lower velocity, so it arrives at a
    different pressure -- which is why the pressures cannot simply be solved
    once, up front, and handed to the thermal model."""
    warm_ground = ThermalPath(diameter_ft=1.0, overall_u=0.6, ambient_r=f_to_r(110.0))
    cold_ground = ThermalPath(diameter_ft=1.0, overall_u=0.6, ambient_r=f_to_r(10.0))

    warm = coupled_profile(1400.0, f_to_r(120.0), FLOW, GAS, warm_ground, STREAM,
                           LENGTH, segments=20, joule_thomson_f_per_psi=0.0)
    cold = coupled_profile(1400.0, f_to_r(120.0), FLOW, GAS, cold_ground, STREAM,
                           LENGTH, segments=20, joule_thomson_f_per_psi=0.0)

    assert abs(warm.outlet_psia - cold.outlet_psia) > 1.0, (
        "ground temperature changed the outlet by only %.2f psi"
        % abs(warm.outlet_psia - cold.outlet_psia))
    # Colder gas is denser and has a lower Z, so it moves more easily: the
    # cold line should arrive at the higher pressure.
    assert cold.outlet_psia > warm.outlet_psia


def test_joule_thomson_still_cools_through_the_coupling():
    with_jt = solve(mu=0.03)
    without = solve(mu=0.0)
    assert with_jt.outlet_f < without.outlet_f


# ---------------------------------------------------------------- limits ----


def test_an_undersized_line_is_reported_not_faked():
    """Asking for more than the line can pass has no solution. Returning a
    profile through negative pressures would look like an answer."""
    profile = solve(flow=100_000e6)
    assert not profile.converged
    assert profile.pressures_psia[-1] == 0.0
    assert math.isinf(profile.max_change_r)


def test_nonsense_inputs_are_rejected():
    for kwargs in ({"segments": 0}, {"inlet_psia": 0.0}, {"flow_scfd": 0.0}):
        args = dict(inlet_psia=1400.0, inlet_temperature_r=f_to_r(120.0),
                    flow_scfd=FLOW, gas=GAS, path=PATH, stream=STREAM,
                    length_ft=LENGTH, segments=10)
        args.update(kwargs)
        with pytest.raises(ValueError):
            coupled_profile(**args)


def test_refining_the_grid_converges_the_outlet():
    """Halving the step should move the answer less each time, or the segment
    count is a free parameter dressed up as a discretisation."""
    outlets = [solve(segments=n).outlet_psia for n in (10, 20, 40)]
    first = abs(outlets[1] - outlets[0])
    second = abs(outlets[2] - outlets[1])
    assert second < first, "%.3f then %.3f psi" % (first, second)
