"""Tests for the component-based simulation engine.

The solver is the audit anchor for everything else in PhaseSpace -- the
surrogate trains against it and the UI shows its answer next to every
prediction -- so it has to be trusted before anything downstream means
anything. These check it against closed-form answers and hand-calculable
cases, not against its own previous output.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.physics.weymouth import GasProperties as LegacyGas
from phasespace.physics.weymouth import weymouth_flow
from phasespace.sim.components import (
    AGA,
    FIXED_FLOW,
    OUTLET_PRESSURE,
    PANHANDLE_A,
    PANHANDLE_B,
    WEYMOUTH,
    CheckValve,
    Compressor,
    GasPipe,
    LiquidPipe,
    Pump,
    PumpCurve,
    Regulator,
    Valve,
    average_pressure,
    fully_turbulent_friction_factor,
    fully_turbulent_transmission_factor,
    swamee_jain_friction_factor,
)
from phasespace.sim.fluids import _DAK, GasProperties, LiquidProperties, z_factor_dak
from phasespace.sim.solver import FLOW, GAS, LIQUID, PRESSURE, Network, Node, solve

SPIKE_GAS = GasProperties(gravity=0.65, temperature_r=520.0, fixed_z=0.95)
REAL_GAS = GasProperties(gravity=0.65, temperature_r=520.0)


def max_imbalance(net, result):
    return max(abs(v) for v in result.node_imbalance(net).values())


# ---------------------------------------------------------------- fluids ---


def test_dak_z_factor_matches_standing_katz_reference_points():
    """Spot-checks against the Standing-Katz chart the DAK correlation fits.

    The tolerance is 0.02 in Z, about as finely as the chart can be read.
    This catches a transcribed coefficient; the Newton solve itself is
    checked separately and far more tightly below.
    """
    p_pc, t_pc = 670.0, 365.0
    # (Ppr, Tpr) -> Z read off the chart
    for p_pr, t_pr, expected in [
        (2.0, 1.05, 0.33),   # deep in the near-critical dip
        (1.0, 1.50, 0.90),
        (3.0, 1.50, 0.78),
        (2.0, 2.00, 0.95),
    ]:
        z = z_factor_dak(p_pr * p_pc, t_pr * t_pc, p_pc, t_pc)
        assert abs(z - expected) < 0.02, "Ppr=%s Tpr=%s gave Z=%.4f, expected ~%.3f" % (
            p_pr, t_pr, z, expected)


def test_dak_newton_solve_agrees_with_bracketed_bisection():
    """The DAK equation is implicit in Z, so the answer depends on an inner
    iterative solve. Newton is fast but can in principle land on a different
    root or stall; a bracketed method cannot. Agreeing to 1e-9 across the
    whole reduced range -- including the near-critical band at Tpr = 1.05
    where the curve is steepest -- is what says the fast path is safe.
    """
    from scipy.optimize import brentq

    p_pc, t_pc = 670.0, 365.0
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _DAK

    def z_by_bisection(p_pr, t_pr):
        c1 = a1 + a2 / t_pr + a3 / t_pr ** 3 + a4 / t_pr ** 4 + a5 / t_pr ** 5
        c2 = a6 + a7 / t_pr + a8 / t_pr ** 2
        c3 = a9 * (a7 / t_pr + a8 / t_pr ** 2)
        c4 = a10 / t_pr ** 3
        k = 0.27 * p_pr / t_pr

        def residual(rho):
            return rho * (1.0 + c1 * rho + c2 * rho ** 2 - c3 * rho ** 5
                          + c4 * (1.0 + a11 * rho ** 2) * rho ** 2
                          * math.exp(-a11 * rho * rho)) - k

        return k / brentq(residual, 1e-9, 5.0)

    for t_pr in (1.05, 1.2, 1.5, 2.0, 2.8):
        for p_pr in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
            fast = z_factor_dak(p_pr * p_pc, t_pr * t_pc, p_pc, t_pc)
            exact = z_by_bisection(p_pr, t_pr)
            assert fast == pytest.approx(exact, abs=1e-9), (
                "Ppr=%s Tpr=%s: Newton %.10f vs bisection %.10f" % (p_pr, t_pr, fast, exact))


def test_z_factor_dips_then_recovers_with_pressure():
    """Z has a minimum around 2000-2500 psia and climbs back past 1.0 at high
    pressure. A monotone Z would mean the correlation collapsed to an ideal
    gas somewhere, which is the failure mode a frozen constant has."""
    gas = GasProperties(gravity=0.65, temperature_r=560.0)
    z_low = gas.z_at(100.0)
    z_min = gas.z_at(2000.0)
    z_high = gas.z_at(8000.0)
    assert z_low > z_min
    assert z_high > 1.0
    assert 0.75 < z_min < 0.85


def test_gas_viscosity_and_density_increase_with_pressure():
    gas = GasProperties(gravity=0.65, temperature_r=560.0)
    assert gas.density_lbft3(1000.0) > gas.density_lbft3(100.0)
    assert gas.viscosity_cp(1000.0) > gas.viscosity_cp(100.0)
    # Natural gas viscosity is a few hundredths of a centipoise; an answer
    # off by an order of magnitude means the LGE unit conversion is wrong.
    assert 0.008 < gas.viscosity_cp(500.0) < 0.05


def test_sour_gas_correction_lowers_pseudo_critical_temperature():
    sweet = GasProperties(gravity=0.7)
    sour = GasProperties(gravity=0.7, y_co2=0.08, y_h2s=0.05)
    _, t_pc_sweet = sweet.pseudo_criticals()
    _, t_pc_sour = sour.pseudo_criticals()
    assert t_pc_sour < t_pc_sweet


def test_live_oil_is_thinner_than_dead_oil():
    oil = LiquidProperties(api_gravity=30.0, temperature_r=620.0, bubble_point_psia=2500.0)
    assert oil.viscosity_cp(2000.0) < oil.dead_oil_viscosity_cp()
    # More dissolved gas at higher pressure (up to the bubble point) means
    # thinner oil, not thicker.
    assert oil.viscosity_cp(2000.0) < oil.viscosity_cp(200.0)


def test_water_cut_raises_mixture_specific_gravity():
    dry = LiquidProperties(api_gravity=35.0, water_cut=0.0)
    wet = LiquidProperties(api_gravity=35.0, water_cut=0.5)
    assert wet.specific_gravity > dry.specific_gravity


# ------------------------------------------------------------ components ---


def test_gas_pipe_reproduces_closed_form_weymouth():
    """With Z pinned, one GasPipe must agree with the standalone Weymouth
    function to floating-point precision -- this is the bridge that lets the
    Phase 0 validation numbers still be compared against this engine."""
    pipe = GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0)
    q, _, _ = pipe.flow_and_grad(800.0, 750.0, SPIKE_GAS)
    expected = weymouth_flow(800.0, 750.0, 6.0, 5.0, LegacyGas(0.65, 0.95, 520.0))
    assert abs(q - expected) / abs(expected) < 1e-12


@pytest.mark.parametrize("equation", [WEYMOUTH, PANHANDLE_A, PANHANDLE_B, AGA])
def test_gas_pipe_gradients_match_finite_difference(equation):
    """The analytic Jacobian is the whole reason this solver converges near
    loop ties. If a correlation's derivative is wrong the solver still often
    converges -- just slowly and from a worse basin -- so it has to be
    checked directly rather than inferred from a solve succeeding."""
    pipe = GasPipe(a=0, b=1, diameter_in=10.0, length_mi=12.0, equation=equation,
                   elevation_b_ft=200.0)
    p_a, p_b = 900.0, 860.0
    q, dq_da, dq_db = pipe.flow_and_grad(p_a, p_b, SPIKE_GAS)
    h = 1e-4
    fd_a = (pipe.flow(p_a + h, p_b, SPIKE_GAS) - pipe.flow(p_a - h, p_b, SPIKE_GAS)) / (2 * h)
    fd_b = (pipe.flow(p_a, p_b + h, SPIKE_GAS) - pipe.flow(p_a, p_b - h, SPIKE_GAS)) / (2 * h)
    assert abs(dq_da - fd_a) / abs(fd_a) < 1e-5
    assert abs(dq_db - fd_b) / abs(fd_b) < 1e-5


def test_gas_pipe_flow_is_antisymmetric():
    pipe = GasPipe(a=0, b=1, diameter_in=8.0, length_mi=10.0)
    forward = pipe.flow(900.0, 800.0, REAL_GAS)
    reverse = pipe.flow(800.0, 900.0, REAL_GAS)
    assert forward > 0 and reverse < 0
    assert abs(forward + reverse) / forward < 1e-9


def test_near_zero_pressure_difference_stays_finite_and_signed():
    """The loop-tie case: two nearly-equal large pressures. Catastrophic
    cancellation here is what stalls Newton, so the flow must stay small,
    finite, and correctly signed rather than becoming noise."""
    pipe = GasPipe(a=0, b=1, diameter_in=8.0, length_mi=10.0)
    q_pos, _, _ = pipe.flow_and_grad(900.0, 900.0 - 1e-6, SPIKE_GAS)
    q_neg, _, _ = pipe.flow_and_grad(900.0 - 1e-6, 900.0, SPIKE_GAS)
    assert q_pos > 0.0
    assert q_neg < 0.0
    assert math.isfinite(q_pos) and math.isfinite(q_neg)
    assert abs(q_pos + q_neg) / abs(q_pos) < 1e-6


def test_elevation_costs_pressure_uphill_and_returns_it_downhill():
    uphill = GasPipe(a=0, b=1, diameter_in=12.0, length_mi=10.0, elevation_b_ft=1000.0)
    flat = GasPipe(a=0, b=1, diameter_in=12.0, length_mi=10.0)
    downhill = GasPipe(a=0, b=1, diameter_in=12.0, length_mi=10.0, elevation_b_ft=-1000.0)
    q_up = uphill.flow(1000.0, 950.0, REAL_GAS)
    q_flat = flat.flow(1000.0, 950.0, REAL_GAS)
    q_down = downhill.flow(1000.0, 950.0, REAL_GAS)
    assert q_up < q_flat < q_down


def test_panhandle_predicts_less_drop_than_weymouth_on_a_large_line():
    """Weymouth is the conservative correlation for large-diameter
    transmission; Panhandle's higher efficiency exponent predicts more
    throughput for the same pressure drop. If this inverts, a coefficient
    has been transcribed wrong."""
    common = dict(a=0, b=1, diameter_in=16.0, length_mi=20.0)
    q_weymouth = GasPipe(equation=WEYMOUTH, **common).flow(1000.0, 960.0, REAL_GAS)
    q_pan_a = GasPipe(equation=PANHANDLE_A, **common).flow(1000.0, 960.0, REAL_GAS)
    q_pan_b = GasPipe(equation=PANHANDLE_B, **common).flow(1000.0, 960.0, REAL_GAS)
    assert q_pan_a > q_weymouth
    assert q_pan_b > q_weymouth


def test_average_pressure_sits_above_the_arithmetic_mean():
    """Pressure falls non-linearly along a gas pipe -- most of its length is
    at the high-pressure end -- so the integrated average is above the
    arithmetic mean, not below it. Getting this backwards biases every
    property evaluation (Z, viscosity) on a line with a big drop."""
    p_avg = average_pressure(1000.0, 400.0)
    assert p_avg > 700.0
    assert p_avg == pytest.approx(742.857, rel=1e-4)
    assert average_pressure(500.0, 500.0) == pytest.approx(500.0)


def test_friction_factors_are_physical():
    # Laminar
    assert swamee_jain_friction_factor(1000.0, 6.0, 0.0018) == pytest.approx(0.064)
    # Turbulent smooth-ish pipe lands in the usual 0.01-0.05 band
    f = swamee_jain_friction_factor(1e5, 10.0, 0.0018)
    assert 0.01 < f < 0.05
    # Rougher pipe has more friction at the same Reynolds number
    assert swamee_jain_friction_factor(1e5, 10.0, 0.01) > f
    # Fully-turbulent limit for a big transmission line: real gas lines run
    # around f = 0.01, so an answer near 0.0026 means the transmission factor
    # has been used where a Darcy friction factor belongs.
    f_rough = fully_turbulent_friction_factor(16.0, 0.0007)
    assert 0.008 < f_rough < 0.015


def test_transmission_and_friction_factors_are_the_same_statement():
    """F = 2 / sqrt(f_darcy). Mixing the two is a silent factor-of-two error
    in pressure drop, so the identity is asserted rather than assumed."""
    for diameter, roughness in ((6.0, 0.0018), (16.0, 0.0007), (30.0, 0.0006)):
        f_darcy = fully_turbulent_friction_factor(diameter, roughness)
        f_trans = fully_turbulent_transmission_factor(diameter, roughness)
        assert f_trans == pytest.approx(2.0 / math.sqrt(f_darcy), rel=1e-12)


def test_friction_factor_is_continuous_across_the_transition_band():
    """A jump in f puts a kink in the residual that Newton oscillates
    across. Check the blend actually joins up at both edges."""
    below = swamee_jain_friction_factor(2099.0, 6.0, 0.0018)
    at_lower = swamee_jain_friction_factor(2101.0, 6.0, 0.0018)
    assert abs(below - at_lower) < 1e-3
    just_under = swamee_jain_friction_factor(3999.0, 6.0, 0.0018)
    just_over = swamee_jain_friction_factor(4001.0, 6.0, 0.0018)
    assert abs(just_under - just_over) < 1e-3


def test_liquid_pipe_matches_the_darcy_weisbach_hand_calc():
    """Pipeline handbooks give dP = 0.0605 * f * Q^2 * SG / d^5 psi per mile
    with Q in bbl/day. Reproduce that directly from the component."""
    oil = LiquidProperties(api_gravity=35.0, temperature_r=560.0)
    pipe = LiquidPipe(a=0, b=1, diameter_in=10.0, length_mi=30.0)
    q_bpd = 20000.0
    re = pipe.reynolds(q_bpd, oil, 700.0)
    f = swamee_jain_friction_factor(re, 10.0, pipe.roughness_in)
    expected_dp = 0.0605 * f * q_bpd ** 2 * oil.specific_gravity / 10.0 ** 5 * 30.0

    # Drive the component's lagged friction state to this flow, then ask what
    # pressure difference produces it.
    pipe._last_q = q_bpd
    solved = solve_single_liquid_pipe(pipe, oil, q_bpd)
    assert abs(solved - expected_dp) / expected_dp < 0.02


def solve_single_liquid_pipe(pipe, oil, q_bpd):
    """Helper: the pressure drop across `pipe` at `q_bpd`, via the solver."""
    net = Network(phase=LIQUID, fluid=oil)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0))
    net.add_node(Node(1, FLOW, withdrawal=q_bpd))
    net.add_branch(pipe)
    result = solve(net)
    assert result.converged
    return 900.0 - result.pressures[1]


def test_liquid_static_head_is_the_fluid_column():
    oil = LiquidProperties(api_gravity=35.0)
    pipe = LiquidPipe(a=0, b=1, diameter_in=8.0, length_mi=5.0, elevation_b_ft=300.0)
    # 0.433 psi/ft for fresh water, scaled by specific gravity.
    assert pipe.static_head_psi(oil) == pytest.approx(0.4333 * oil.specific_gravity * 300.0, rel=1e-3)


def test_check_valve_blocks_reverse_flow():
    inner = GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0)
    cv = CheckValve(inner=inner)
    forward = cv.flow(900.0, 800.0, REAL_GAS)
    reverse = cv.flow(800.0, 900.0, REAL_GAS)
    assert forward == pytest.approx(inner.flow(900.0, 800.0, REAL_GAS))
    assert abs(reverse) < abs(forward) * 1e-5


def test_component_construction_rejects_nonsense():
    with pytest.raises(ValueError):
        GasPipe(a=0, b=1, diameter_in=0.0, length_mi=5.0)
    with pytest.raises(ValueError):
        GasPipe(a=0, b=1, diameter_in=6.0, length_mi=-1.0)
    with pytest.raises(ValueError):
        GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0, equation="bernoulli")
    with pytest.raises(ValueError):
        Compressor(a=0, b=1, mode="vibes", setpoint=900.0)


# ---------------------------------------------------------------- solver ---


def test_single_pipe_matches_closed_form():
    net = Network(phase=GAS, fluid=SPIKE_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    net.add_node(Node(1, FLOW, withdrawal=12e6))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0))
    result = solve(net)
    assert result.converged
    recovered = weymouth_flow(800.0, result.pressures[1], 6.0, 5.0, LegacyGas(0.65, 0.95, 520.0))
    assert recovered == pytest.approx(12e6, rel=1e-8)


def test_symmetric_branch_splits_evenly():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0))
    net.add_node(Node(1, FLOW))
    net.add_node(Node(2, FLOW, withdrawal=5e6))
    net.add_node(Node(3, FLOW, withdrawal=5e6))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=10.0, length_mi=5.0))
    left = net.add_branch(GasPipe(a=1, b=2, diameter_in=6.0, length_mi=8.0))
    right = net.add_branch(GasPipe(a=1, b=3, diameter_in=6.0, length_mi=8.0))
    result = solve(net)
    assert result.converged
    assert result.flows[left] == pytest.approx(result.flows[right], rel=1e-6)
    assert result.pressures[2] == pytest.approx(result.pressures[3], rel=1e-9)


def test_loop_conserves_mass_at_every_node():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0))
    net.add_node(Node(1, FLOW))
    net.add_node(Node(2, FLOW))
    net.add_node(Node(3, FLOW, withdrawal=30e6))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=12.0, length_mi=8.0))
    net.add_branch(GasPipe(a=1, b=2, diameter_in=8.0, length_mi=10.0))
    net.add_branch(GasPipe(a=1, b=3, diameter_in=10.0, length_mi=12.0))
    net.add_branch(GasPipe(a=2, b=3, diameter_in=8.0, length_mi=6.0))
    result = solve(net)
    assert result.converged
    # Relative to throughput, not absolute: 30 MMscfd of throughput makes an
    # absolute tolerance meaningless on its own.
    assert max_imbalance(net, result) / 30e6 < 1e-7


def test_two_sources_feed_one_demand():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0))
    net.add_node(Node(1, PRESSURE, pressure_psia=850.0))
    net.add_node(Node(2, FLOW, withdrawal=20e6))
    strong = net.add_branch(GasPipe(a=0, b=2, diameter_in=10.0, length_mi=10.0))
    weak = net.add_branch(GasPipe(a=1, b=2, diameter_in=10.0, length_mi=10.0))
    result = solve(net)
    assert result.converged
    assert result.flows[strong] + result.flows[weak] == pytest.approx(20e6, rel=1e-7)
    # The higher-pressure source carries more of the load.
    assert result.flows[strong] > result.flows[weak]


def test_compressor_holds_its_discharge_pressure():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    net.add_node(Node(1, FLOW))
    net.add_node(Node(2, FLOW))
    net.add_node(Node(3, FLOW, withdrawal=60e6))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=12.0, length_mi=15.0))
    comp = net.add_control(Compressor(a=1, b=2, mode=OUTLET_PRESSURE, setpoint=1100.0,
                                      max_ratio=1.8, name="CS-1"))
    net.add_branch(GasPipe(a=2, b=3, diameter_in=12.0, length_mi=40.0))
    result = solve(net)
    assert result.converged
    assert result.pressures[2] == pytest.approx(1100.0, abs=1e-4)
    # All the demand passes through the machine -- there is no other path.
    assert result.flows[comp] == pytest.approx(60e6, rel=1e-6)
    assert comp.power_hp(result.flows[comp], result.pressures[1], 1100.0, REAL_GAS) > 0.0


def test_compressor_is_capped_at_its_maximum_ratio():
    """Asked for more compression than the machine has, the honest answer is
    a shortfall at the delivery point -- not a setpoint magically met."""
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    net.add_node(Node(1, FLOW))
    net.add_node(Node(2, FLOW))
    net.add_node(Node(3, FLOW, withdrawal=60e6))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=12.0, length_mi=15.0))
    comp = net.add_control(Compressor(a=1, b=2, mode=OUTLET_PRESSURE, setpoint=1400.0,
                                      max_ratio=1.2, name="CS-1"))
    net.add_branch(GasPipe(a=2, b=3, diameter_in=12.0, length_mi=40.0))
    result = solve(net)
    assert result.converged
    assert result.pressures[2] < 1400.0
    assert result.pressures[2] / result.pressures[1] == pytest.approx(1.2, rel=1e-4)
    assert any("max ratio" in note for note in result.notes)
    # The machine itself has been switched into its ratio-limited mode, so a
    # caller inspecting the model afterwards sees the same story as the note.
    assert comp.ratio_violation(result.pressures[1], result.pressures[2]) == 0.0


def test_compressor_in_fixed_flow_mode_moves_exactly_that_flow():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    net.add_node(Node(1, FLOW))
    net.add_node(Node(2, FLOW))
    net.add_node(Node(3, PRESSURE, pressure_psia=1000.0))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=12.0, length_mi=15.0))
    comp = net.add_control(Compressor(a=1, b=2, mode=FIXED_FLOW, setpoint=45e6,
                                      max_ratio=3.0, name="CS-1"))
    net.add_branch(GasPipe(a=2, b=3, diameter_in=12.0, length_mi=25.0))
    result = solve(net)
    assert result.converged
    assert result.flows[comp] == pytest.approx(45e6, rel=1e-6)


def test_regulator_holds_setpoint_then_gives_up_when_starved():
    """Both halves matter. Holding the setpoint is the easy case; reporting
    the setpoint anyway when upstream pressure has collapsed would be the
    model hiding exactly the low-pressure event the simulation is run to
    find."""
    def build(supply_psia):
        net = Network(phase=GAS, fluid=REAL_GAS)
        net.add_node(Node(0, PRESSURE, pressure_psia=supply_psia))
        net.add_node(Node(1, FLOW))
        net.add_node(Node(2, FLOW))
        net.add_node(Node(3, FLOW, withdrawal=8e6))
        net.add_branch(GasPipe(a=0, b=1, diameter_in=8.0, length_mi=5.0))
        reg = net.add_control(Regulator(a=1, b=2, setpoint=250.0, name="city gate"))
        net.add_branch(GasPipe(a=2, b=3, diameter_in=8.0, length_mi=6.0))
        return net, reg

    net, reg = build(900.0)
    result = solve(net)
    assert result.converged
    assert reg.controlling
    assert result.pressures[2] == pytest.approx(250.0, abs=1e-4)

    net, reg = build(220.0)
    result = solve(net)
    assert result.converged
    assert not reg.controlling
    assert result.pressures[2] < 250.0
    assert any("lost control" in note for note in result.notes)


def test_liquid_line_with_a_pump():
    oil = LiquidProperties(api_gravity=35.0, temperature_r=560.0)
    net = Network(phase=LIQUID, fluid=oil)
    net.add_node(Node(0, PRESSURE, pressure_psia=60.0))
    net.add_node(Node(1, FLOW))
    net.add_node(Node(2, FLOW, withdrawal=20000.0))
    pump = net.add_control(Pump(a=0, b=1, mode=OUTLET_PRESSURE, setpoint=1200.0))
    net.add_branch(LiquidPipe(a=1, b=2, diameter_in=10.0, length_mi=30.0, elevation_b_ft=300.0))
    result = solve(net)
    assert result.converged
    assert result.pressures[1] == pytest.approx(1200.0, abs=1e-4)
    assert result.flows[pump] == pytest.approx(20000.0, rel=1e-6)
    assert pump.hydraulic_hp(20000.0, 60.0, 1200.0) > 0.0


def test_pump_curve_delivers_less_head_at_higher_flow():
    oil = LiquidProperties(api_gravity=35.0)
    curve = PumpCurve(a=0, b=1, h0_ft=3000.0, h1_ft_per_bpd=0.0, h2_ft_per_bpd2=2.5e-6)
    assert curve.head_ft(10000.0) > curve.head_ft(25000.0)
    # A pump on its curve moves less as the network asks for more head.
    q_easy = curve.flow_and_grad(50.0, 700.0, oil)[0]
    q_hard = curve.flow_and_grad(50.0, 1000.0, oil)[0]
    assert q_easy > q_hard > 0.0


def test_valve_opening_throttles_flow():
    net_open = Network(phase=GAS, fluid=REAL_GAS)
    net_open.add_node(Node(0, PRESSURE, pressure_psia=900.0))
    net_open.add_node(Node(1, PRESSURE, pressure_psia=880.0))
    valve = net_open.add_branch(Valve(a=0, b=1, cv=500.0, opening=1.0))
    q_open = solve(net_open).flows[valve]

    net_shut = Network(phase=GAS, fluid=REAL_GAS)
    net_shut.add_node(Node(0, PRESSURE, pressure_psia=900.0))
    net_shut.add_node(Node(1, PRESSURE, pressure_psia=880.0))
    throttled = net_shut.add_branch(Valve(a=0, b=1, cv=500.0, opening=0.25))
    q_throttled = solve(net_shut).flows[throttled]

    assert q_open > q_throttled > 0.0
    assert q_throttled == pytest.approx(q_open * 0.25, rel=1e-6)


def test_solver_rejects_a_network_with_no_fixed_pressure():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, FLOW, withdrawal=1e6))
    net.add_node(Node(1, FLOW, withdrawal=-1e6))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0))
    with pytest.raises(ValueError) as excinfo:
        solve(net)
    assert "fixed-pressure" in str(excinfo.value)


def test_solver_rejects_disconnected_and_malformed_networks():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    net.add_node(Node(1, FLOW, withdrawal=1e6))
    net.add_node(Node(2, FLOW, withdrawal=1e6))  # orphan
    net.add_branch(GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0))
    with pytest.raises(ValueError) as excinfo:
        solve(net)
    assert "not connected" in str(excinfo.value)

    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    net.add_node(Node(1, FLOW, withdrawal=1e6))
    net.add_branch(GasPipe(a=0, b=9, diameter_in=6.0, length_mi=5.0))
    with pytest.raises(ValueError) as excinfo:
        solve(net)
    assert "unknown node id" in str(excinfo.value)


def test_duplicate_node_ids_are_rejected_at_construction():
    net = Network(phase=GAS, fluid=REAL_GAS)
    net.add_node(Node(0, PRESSURE, pressure_psia=800.0))
    with pytest.raises(ValueError):
        net.add_node(Node(0, FLOW, withdrawal=1e6))


def test_solution_is_independent_of_the_initial_guess():
    """Newton finding the same root from anywhere is the practical evidence
    that the answer is the network's, not the starting point's."""
    def build():
        net = Network(phase=GAS, fluid=REAL_GAS)
        net.add_node(Node(0, PRESSURE, pressure_psia=900.0))
        net.add_node(Node(1, FLOW))
        net.add_node(Node(2, FLOW, withdrawal=15e6))
        net.add_node(Node(3, FLOW, withdrawal=10e6))
        net.add_branch(GasPipe(a=0, b=1, diameter_in=12.0, length_mi=6.0))
        net.add_branch(GasPipe(a=1, b=2, diameter_in=8.0, length_mi=9.0))
        net.add_branch(GasPipe(a=1, b=3, diameter_in=8.0, length_mi=7.0))
        net.add_branch(GasPipe(a=2, b=3, diameter_in=6.0, length_mi=4.0))
        return net

    baseline = solve(build())
    for guess in (100.0, 500.0, 890.0):
        other = solve(build(), init_pressure=guess)
        assert other.converged
        for nid in baseline.pressures:
            assert other.pressures[nid] == pytest.approx(baseline.pressures[nid], rel=1e-6)


def test_erosional_velocity_screening_reports_a_sensible_limit():
    pipe = GasPipe(a=0, b=1, diameter_in=6.0, length_mi=5.0)
    v = pipe.velocity_ft_s(30e6, 800.0, REAL_GAS)
    v_max = pipe.erosional_velocity_ft_s(800.0, REAL_GAS)
    # API RP 14E limits land in the tens of ft/s for gas at these pressures.
    assert 10.0 < v_max < 60.0
    assert v > 0.0
