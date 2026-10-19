"""Tests for the time-domain solvers: gas line pack and liquid surge.

Both are checked against things known independently of this code -- an
analytic bound (Joukowsky), a conservation law (mass), a physical period
(4L/a), and the steady-state solver these must agree with in the limit.
Two independent solvers landing on the same steady answer is the strongest
evidence available here that either of them is right.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.components import AGA, GasPipe
from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.solver import FLOW, GAS, PRESSURE, Network, Node, solve
from phasespace.sim.transient import (
    GasLine,
    SurgeLine,
    pipe_period_s,
    simulate_gas_transient,
    simulate_valve_closure,
)

GAS_PROPS = GasProperties(gravity=0.65, temperature_r=520.0)
OIL = LiquidProperties(api_gravity=35.0, temperature_r=560.0)


def make_line(segments=40):
    return GasLine(length_mi=50.0, diameter_in=16.0, gas=GAS_PROPS, segments=segments)


# ------------------------------------------------------------- gas line ----


def test_isothermal_wave_speed_is_physical():
    """a = sqrt(ZRT/M) for natural gas is around 400 m/s. An answer an order
    of magnitude off means a unit conversion in the SI boundary is wrong,
    and every timescale in the model would be wrong with it."""
    line = make_line()
    a = math.sqrt(line.wave_speed_squared(800.0))
    assert 250.0 < a < 450.0


def test_constant_demand_holds_a_steady_state():
    """Nothing should drift when nothing changes. A scheme that slowly bleeds
    pressure or mass under constant boundary conditions will produce
    plausible-looking but meaningless answers on a long run."""
    line = make_line()
    result = simulate_gas_transient(line, 900.0, lambda t: 100e6,
                                    duration_s=6 * 3600, dt_s=60.0)
    first, last = result.outlet_pressure_psia[0], result.outlet_pressure_psia[-1]
    assert abs(last - first) / first < 0.002
    pack_first, pack_last = result.line_pack_scf[0], result.line_pack_scf[-1]
    assert abs(pack_last - pack_first) / pack_first < 0.002


def test_line_pack_conserves_mass():
    """The integral of (in - out) has to equal the change in stored gas. This
    is the property that makes a line-pack answer trustworthy at all."""
    line = make_line()

    def schedule(t):
        return 100e6 if t < 3600 else 150e6

    result = simulate_gas_transient(line, 900.0, schedule, duration_s=8 * 3600, dt_s=60.0)
    t = np.array(result.times_s)
    # `np.trapezoid`, not `np.trapz`: the old spelling was removed in NumPy
    # 2.0, and the web app pins 2.2.1, so this test failed everywhere the
    # deployed engine actually runs.
    net_in = np.trapezoid(
        np.array(result.inlet_flow_scfd) - np.array(result.outlet_flow_scfd),
        t) / 86400.0
    pack_change = result.line_pack_scf[-1] - result.line_pack_scf[0]
    assert abs(net_in - pack_change) / abs(pack_change) < 0.05


def test_demand_step_is_supplied_from_storage_before_inlet_flow_catches_up():
    """The whole reason a transient model exists: a steady-state answer jumps
    the inlet flow to the new demand instantly. A real line supplies the
    difference out of line pack for hours."""
    line = make_line()

    def schedule(t):
        return 100e6 if t < 3600 else 150e6

    result = simulate_gas_transient(line, 900.0, schedule, duration_s=10 * 3600, dt_s=60.0)
    times = np.array(result.times_s)
    inlet = np.array(result.inlet_flow_scfd)

    just_after = inlet[np.searchsorted(times, 3600 + 120)]
    assert just_after < 150e6, "inlet flow responded instantly -- storage is missing"

    # And it does catch up eventually.
    assert result.inlet_flow_scfd[-1] == pytest.approx(150e6, rel=0.02)
    # Pack falls, because the extra gas came out of the line.
    assert result.line_pack_scf[-1] < result.line_pack_scf[0]


def test_transient_steady_limit_matches_the_steady_state_solver():
    """Two completely independent solvers -- an implicit time-marching
    finite-difference scheme and a Newton-Raphson steady-state network --
    have to agree once the transient settles. They share no code path."""
    line = make_line()
    result = simulate_gas_transient(line, 900.0, lambda t: 150e6,
                                    duration_s=48 * 3600, dt_s=120.0)

    net = Network(phase=GAS, fluid=GAS_PROPS)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0))
    net.add_node(Node(1, FLOW, withdrawal=150e6))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=16.0, length_mi=50.0,
                           roughness_in=0.0007, equation=AGA, efficiency=1.0))
    steady = solve(net)
    assert steady.converged

    transient_outlet = result.outlet_pressure_psia[-1]
    assert transient_outlet == pytest.approx(steady.pressures[1], rel=0.02)


def test_bigger_demand_means_lower_delivery_pressure():
    line = make_line()
    previous = None
    for demand in (80e6, 120e6, 160e6):
        result = simulate_gas_transient(line, 900.0, lambda t, d=demand: d,
                                        duration_s=12 * 3600, dt_s=120.0)
        outlet = result.outlet_pressure_psia[-1]
        if previous is not None:
            assert outlet < previous
        previous = outlet


def test_time_to_breach_a_delivery_limit_is_reported():
    line = make_line()

    def schedule(t):
        return 100e6 if t < 1800 else 170e6

    result = simulate_gas_transient(line, 900.0, schedule, duration_s=8 * 3600, dt_s=60.0)
    assert result.feasible
    breach = result.time_to_outlet_below(600.0)
    assert breach > 1800.0, "delivery pressure fell before the demand step"
    assert result.minimum_outlet_pressure_psia < 600.0
    # A limit that is never breached must report -1, not 0 (which would read
    # as "breached at t=0").
    assert result.time_to_outlet_below(1.0) == -1.0


def test_an_undeliverable_demand_is_reported_not_silently_clamped():
    """Ask for more than the line can pass and it drains to the pressure
    floor. The honest answer is "this offtake is not deliverable", not a
    near-zero pressure presented as a prediction."""
    line = make_line()
    result = simulate_gas_transient(line, 900.0, lambda t: 100e6 if t < 1800 else 260e6,
                                    duration_s=8 * 3600, dt_s=60.0)
    assert not result.feasible
    assert result.floor_reached_at_s > 1800.0
    assert any("not deliverable" in note for note in result.notes)


def test_gas_line_rejects_nonsense_configuration():
    with pytest.raises(ValueError):
        GasLine(length_mi=50.0, diameter_in=16.0, segments=1)
    with pytest.raises(ValueError):
        GasLine(length_mi=50.0, diameter_in=0.0, segments=20)


# ----------------------------------------------------------- liquid surge --


def test_moc_timestep_satisfies_the_courant_condition_exactly():
    """dt = dx/a is not a suggestion in the method of characteristics: any
    other value needs interpolation, which numerically damps the wave the
    calculation exists to capture."""
    line = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=3300.0,
                     liquid=OIL, segments=20)
    dx = line.length_m / line.segments
    assert line.timestep_s() == pytest.approx(dx / line.wave_speed_m_s, rel=1e-12)


def test_instant_closure_reproduces_the_joukowsky_bound():
    """The analytic answer this whole solver has to reproduce: an
    instantaneous closure on a near-frictionless line raises pressure by
    exactly rho*a*dV, and the downsurge mirrors it about static head."""
    line = SurgeLine(length_mi=5.0, diameter_in=24.0, wave_speed_ft_s=3300.0,
                     roughness_in=1e-7, liquid=OIL, segments=40)
    q0 = 40000.0
    area_ft2 = math.pi * (24.0 / 12.0) ** 2 / 4.0
    v0 = q0 * 5.614583 / 86400.0 / area_ft2
    joukowsky = line.joukowsky_rise_psi(v0)

    result = simulate_valve_closure(line, 800.0, q0, closure_time_s=0.0, duration_s=60.0)
    initial_valve_pressure = result.valve_pressure_psia[0]

    rise = result.maximum_pressure_psia - initial_valve_pressure
    drop = 800.0 - result.minimum_pressure_psia
    assert rise == pytest.approx(joukowsky, rel=0.08)
    assert drop == pytest.approx(joukowsky, rel=0.08)


def test_slow_closure_produces_far_less_surge():
    """Closing slower than the pipe period 2L/a lets the relief wave return
    from the far end before the valve is shut. This is the entire physical
    basis of surge mitigation by closure scheduling, and a model that misses
    it would tell an operator that stroke time does not matter."""
    line = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=3300.0,
                     liquid=OIL, segments=20)
    period = pipe_period_s(line)
    rapid = simulate_valve_closure(line, 800.0, 60000.0, closure_time_s=0.0,
                                   duration_s=3 * period)
    slow = simulate_valve_closure(line, 800.0, 60000.0, closure_time_s=4 * period,
                                  duration_s=6 * period)
    assert slow.maximum_pressure_psia < rapid.maximum_pressure_psia
    rapid_rise = rapid.maximum_pressure_psia - 800.0
    slow_rise = slow.maximum_pressure_psia - 800.0
    assert slow_rise < 0.4 * rapid_rise


def test_surge_peak_grows_with_initial_velocity():
    line = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=3300.0,
                     liquid=OIL, segments=20)
    previous = None
    for q in (20000.0, 40000.0, 80000.0):
        result = simulate_valve_closure(line, 800.0, q, closure_time_s=0.0, duration_s=80.0)
        peak = result.maximum_pressure_psia
        if previous is not None:
            assert peak > previous
        previous = peak


def test_wave_period_matches_four_l_over_a():
    """The valve pressure trace has to oscillate at the pipe period. Getting
    this wrong means the wave speed or the grid spacing is inconsistent, and
    every timing conclusion drawn from the model would be wrong."""
    line = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=3300.0,
                     roughness_in=1e-7, liquid=OIL, segments=20)
    period = pipe_period_s(line)
    result = simulate_valve_closure(line, 800.0, 60000.0, closure_time_s=0.0,
                                    duration_s=3.5 * period)

    trace = np.array(result.valve_pressure_psia)
    times = np.array(result.times_s)
    # Measure the period from upward mean-crossings rather than by hunting
    # local maxima: the MOC trace is a staircase with many tiny plateaux, and
    # a naive local-maximum test counts those as peaks.
    centred = trace - trace.mean()
    crossings = [times[i + 1] for i in range(len(centred) - 1)
                 if centred[i] <= 0.0 < centred[i + 1]]
    assert len(crossings) >= 2, "wave did not oscillate"
    spacing = crossings[1] - crossings[0]
    assert spacing == pytest.approx(period, rel=0.15)


def test_faster_wave_speed_makes_a_worse_surge():
    """Joukowsky is linear in a, so a stiffer line (less entrained gas, more
    rigid pipe) surges harder for the same flow -- the reason a small amount
    of free gas is protective."""
    slow = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=2000.0,
                     liquid=OIL, segments=20)
    fast = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=4000.0,
                     liquid=OIL, segments=20)
    slow_peak = simulate_valve_closure(slow, 800.0, 60000.0, 0.0, 100.0).maximum_pressure_psia
    fast_peak = simulate_valve_closure(fast, 800.0, 60000.0, 0.0, 100.0).maximum_pressure_psia
    assert fast_peak > slow_peak


def test_pressure_envelope_is_per_node_not_per_instant():
    """The design envelope is the highest pressure each point sees at any
    time, which is never lower than the initial profile and is generally not
    reached everywhere at once."""
    line = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=3300.0,
                     liquid=OIL, segments=20)
    result = simulate_valve_closure(line, 800.0, 60000.0, closure_time_s=0.0, duration_s=100.0)
    envelope = result.peak_pressure_by_node_psia()
    assert len(envelope) == line.segments + 1
    assert max(envelope) == pytest.approx(result.maximum_pressure_psia)
    for node_peak, initial in zip(envelope, result.head_profiles[0]):
        assert node_peak >= initial - 1e-9


def test_a_late_closing_stroke_surges_harder_than_a_linear_one():
    """Surge is set by how much flow is still moving in the last part of the
    stroke, not by the nominal closure time. An exponent below 1 keeps the
    valve effectively open then shuts it abruptly, and must surge worse than
    a linear stroke of the same duration; an exponent above 1 does its
    closing early and must be gentler."""
    line = SurgeLine(length_mi=10.0, diameter_in=12.0, wave_speed_ft_s=3300.0,
                     liquid=OIL, segments=20)
    period = pipe_period_s(line)
    kwargs = dict(closure_time_s=2 * period, duration_s=6 * period)
    late = simulate_valve_closure(line, 800.0, 60000.0, closure_exponent=0.4, **kwargs)
    linear = simulate_valve_closure(line, 800.0, 60000.0, closure_exponent=1.0, **kwargs)
    early = simulate_valve_closure(line, 800.0, 60000.0, closure_exponent=2.5, **kwargs)
    assert late.maximum_pressure_psia > linear.maximum_pressure_psia
    assert early.maximum_pressure_psia < linear.maximum_pressure_psia


def test_invalid_closure_exponent_is_rejected():
    line = SurgeLine(length_mi=10.0, diameter_in=12.0, liquid=OIL, segments=10)
    with pytest.raises(ValueError):
        simulate_valve_closure(line, 800.0, 60000.0, closure_time_s=5.0,
                               duration_s=20.0, closure_exponent=0.0)
