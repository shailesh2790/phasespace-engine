"""Network components: everything that can sit on an edge of the graph.

The Phase 0 solver hard-coded the Weymouth equation into its Jacobian
assembly, so gas pipes were the only thing it could ever represent. Here a
component only has to answer one question --

    given the pressures at my two ends, what flows through me, and how does
    that flow change with each of those pressures?

-- and the solver assembles whatever comes back. That one interface covers
gas pipes under four different flow correlations, liquid pipes under
Darcy-Weisbach, valves, and check valves.

Devices that *impose* a condition rather than obey a flow/pressure relation
(a compressor holding a discharge pressure, a regulator holding a setpoint)
cannot be written as q = f(p_a, p_b) at all -- their flow is whatever the
network demands. Those subclass `ControlBranch` instead: the solver gives
each one its own flow unknown and asks it for a device residual. See
`solver.py`.

Sign convention everywhere: positive q means flow from `a` to `b`.

Units: gas branches carry scf/day, liquid branches bbl/day, pressures psia,
diameters inches, lengths miles (pipes) with elevations in feet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fluids import GasProperties, LiquidProperties
from .units import FT_PER_MI, STD_PRESSURE_PSIA, STD_TEMPERATURE_R

# --------------------------------------------------------------------------
# Gas flow correlations
#
# Every pressure-squared correlation has the same shape:
#
#     Q = K * sign(X) * |X| ** n,   X = P_a^2 - e^s * P_b^2
#
# and differs only in the coefficient K and the exponent n. Writing them
# that way (rather than as four separate formulas) is what lets one
# derivative expression serve all of them -- and the analytic derivative is
# not optional here: near a loop tie's near-zero flow no finite-difference
# step is both fine enough to resolve the local slope and coarse enough to
# escape floating-point noise, which stalls Newton in a limit cycle.
# --------------------------------------------------------------------------

WEYMOUTH = "weymouth"
PANHANDLE_A = "panhandle_a"
PANHANDLE_B = "panhandle_b"
AGA = "aga"

_GAS_EQUATIONS = (WEYMOUTH, PANHANDLE_A, PANHANDLE_B, AGA)


def fully_turbulent_transmission_factor(diameter_in: float, roughness_in: float) -> float:
    """AGA fully-turbulent transmission factor, F = 4 log10(3.7 D / e).

    Gas pipeline literature states rough-pipe flow in terms of the
    transmission factor rather than the friction factor. The two are the
    same statement written twice: F = 2 / sqrt(f_darcy). Keeping both
    functions here, with that identity spelled out, is deliberate -- mixing
    the F-form constant (38.77) with a Darcy f is a factor-of-two error that
    is very easy to make and produces a pressure drop that still looks
    plausible.
    """
    if roughness_in <= 0.0:
        roughness_in = 1e-6
    return 4.0 * math.log10(3.7 * diameter_in / roughness_in)


def fully_turbulent_friction_factor(diameter_in: float, roughness_in: float) -> float:
    """Nikuradse fully-turbulent (rough-pipe) Darcy friction factor.

    The rough-pipe limit of Colebrook-White, 1/sqrt(f) = 2 log10(3.7 D / e).
    Transmission gas lines run at Reynolds numbers high enough that the
    friction factor is effectively independent of Re, so the AGA equation
    uses this limit rather than a full Colebrook solve.
    """
    f_transmission = fully_turbulent_transmission_factor(diameter_in, roughness_in)
    return 4.0 / (f_transmission ** 2)


def swamee_jain_friction_factor(reynolds: float, diameter_in: float, roughness_in: float) -> float:
    """Explicit Colebrook-White approximation (Swamee-Jain), Darcy factor.

    Explicit rather than an iterative Colebrook solve because this sits
    inside the Newton loop's Jacobian assembly: it is evaluated once per
    branch per iteration, and Swamee-Jain is within ~1% of Colebrook over
    the whole turbulent range -- far inside the uncertainty of the pipe
    roughness value the user typed in.
    """
    if reynolds < 1e-9:
        return 64.0 / 1e-9
    if reynolds < 2100.0:
        return 64.0 / reynolds  # laminar
    rel_rough = roughness_in / max(diameter_in, 1e-9)
    denom = math.log10(rel_rough / 3.7 + 5.74 / (reynolds ** 0.9))
    f_turb = 0.25 / (denom ** 2)
    if reynolds > 4000.0:
        return f_turb
    # Transitional band: blend laminar and turbulent rather than jumping
    # between them. A step discontinuity in f here puts a kink in the
    # residual that Newton can oscillate across without ever converging.
    f_lam = 64.0 / reynolds
    w = (reynolds - 2100.0) / 1900.0
    return (1.0 - w) * f_lam + w * f_turb


def average_pressure(p_a: float, p_b: float) -> float:
    """Flow-weighted average pressure in a gas pipe.

    Not the arithmetic mean: pressure does not fall linearly along a gas
    pipe (the pressure-*squared* difference is what is linear in length), so
    the integrated average (2/3)(P1 + P2 - P1*P2/(P1+P2)) is what the
    property evaluations (Z, viscosity) should use. It sits *above* the
    arithmetic mean -- 742.9 psia rather than 700 for a line running
    1000 -> 400 psia -- because most of the pipe's length is at the
    high-pressure end of that range.
    """
    p_a, p_b = abs(p_a), abs(p_b)
    total = p_a + p_b
    if total < 1e-9:
        return max(total / 2.0, 1e-9)
    return (2.0 / 3.0) * (p_a + p_b - (p_a * p_b) / total)


class Branch(object):
    """A component whose flow is a function of its two end pressures."""

    # Annotations only, never assignments: a class-level ``a = 0`` here would
    # be inherited as a *default* by every dataclass subclass, which makes
    # their required node ids optional and silently builds branches hanging
    # off node 0.
    a: int
    b: int
    name: str

    def flow_and_grad(self, p_a: float, p_b: float, fluid) -> tuple:
        """Return (q, dq/dp_a, dq/dp_b)."""
        raise NotImplementedError

    def flow(self, p_a: float, p_b: float, fluid) -> float:
        return self.flow_and_grad(p_a, p_b, fluid)[0]


# eq=False on every component: components are identities, not values. Two
# pipes with identical dimensions are still two different pipes, and the
# solver keys its flow results by the component object, which requires them
# to stay hashable.
@dataclass(eq=False)
class GasPipe(Branch):
    """A gas pipe segment under a pressure-squared flow correlation.

    Elevation enters through the standard static-head factor s, which turns
    the pressure-squared difference into P_a^2 - e^s * P_b^2 and stretches
    the length to an equivalent length Le. Ignoring it is defensible for a
    flat distribution grid and badly wrong for anything crossing terrain:
    at 1000 psia a 1000 ft rise costs roughly 40 psi of static head before
    any friction at all.
    """

    a: int
    b: int
    diameter_in: float
    length_mi: float
    roughness_in: float = 0.0007
    efficiency: float = 0.92
    elevation_a_ft: float = 0.0
    elevation_b_ft: float = 0.0
    equation: str = WEYMOUTH
    name: str = ""

    def __post_init__(self):
        if self.equation not in _GAS_EQUATIONS:
            raise ValueError(
                "unknown gas flow equation %r (expected one of %s)"
                % (self.equation, ", ".join(_GAS_EQUATIONS))
            )
        if self.diameter_in <= 0.0:
            raise ValueError("pipe diameter must be positive, got %r" % (self.diameter_in,))
        if self.length_mi <= 0.0:
            raise ValueError("pipe length must be positive, got %r" % (self.length_mi,))

    def _elevation_factor(self, gas: GasProperties, p_avg: float) -> float:
        """s in the elevation-adjusted pressure-squared equation."""
        dz = self.elevation_b_ft - self.elevation_a_ft
        if dz == 0.0:
            return 0.0
        z = gas.z_at(p_avg)
        return 0.0375 * gas.gravity * dz / (gas.temperature_r * z)

    def _coefficient(self, gas: GasProperties, p_avg: float, s: float) -> tuple:
        """Return (K, n) for Q = K * sign(X) * |X|**n."""
        z = gas.z_at(p_avg)
        g = gas.gravity
        t = gas.temperature_r
        base = STD_TEMPERATURE_R / STD_PRESSURE_PSIA

        # Equivalent length: with a non-zero elevation factor the friction
        # term is integrated over a stretched length, not the geometric one.
        if abs(s) < 1e-12:
            le = self.length_mi
        else:
            le = self.length_mi * (math.exp(s) - 1.0) / s

        d = self.diameter_in
        e = self.efficiency

        if self.equation == WEYMOUTH:
            k = 433.5 * e * base * d ** 2.667 / math.sqrt(g * t * le * z)
            return k, 0.5
        if self.equation == AGA:
            # The general flow equation in transmission-factor form:
            #   Q = 38.77 * F * E * (Tb/Pb) * sqrt(X / (G*Tf*Le*Z)) * D^2.5
            # which is the same thing as the 77.54/sqrt(f_darcy) form, since
            # F = 2/sqrt(f). Using the 38.77 constant with a Darcy f instead
            # of F would understate pressure drop by a factor of two.
            f_transmission = fully_turbulent_transmission_factor(d, self.roughness_in)
            k = 38.77 * f_transmission * e * base * d ** 2.5 / math.sqrt(g * t * le * z)
            return k, 0.5
        if self.equation == PANHANDLE_A:
            n = 0.5394
            k = (435.87 * e * (STD_TEMPERATURE_R / STD_PRESSURE_PSIA) ** 1.0788
                 * d ** 2.6182 / (g ** 0.8539 * t * le * z) ** n)
            return k, n
        # PANHANDLE_B
        n = 0.51
        k = (737.0 * e * (STD_TEMPERATURE_R / STD_PRESSURE_PSIA) ** 1.02
             * d ** 2.53 / (g ** 0.961 * t * le * z) ** n)
        return k, n

    def flow_and_grad(self, p_a: float, p_b: float, fluid: GasProperties) -> tuple:
        p_avg = average_pressure(p_a, p_b)
        s = self._elevation_factor(fluid, p_avg)
        k, n = self._coefficient(fluid, p_avg, s)
        es = math.exp(s)

        # X computed as a difference of squares directly would lose precision
        # by catastrophic cancellation on a loop tie, where the two end
        # pressures are large and nearly equal. Factoring it keeps the
        # significant digits that Newton needs exactly where flow is smallest.
        if es == 1.0:
            x = (p_a - p_b) * (p_a + p_b)
        else:
            x = p_a * p_a - es * p_b * p_b

        sign = 1.0 if x >= 0.0 else -1.0
        # The floor only guards literal division by zero at a perfectly
        # balanced tie, so it must sit far below any physically meaningful
        # |X| (a real near-zero loop-tie flow reaches |X| ~ 1e-3 psia^2).
        x_abs = max(abs(x), 1e-12)

        q = sign * k * x_abs ** n
        # dq/dp = K * n * |X|^(n-1) * dX/dp, and the sign cancels because
        # d|X|/dp = sign * dX/dp.
        common = k * n * x_abs ** (n - 1.0)
        dq_da = common * 2.0 * p_a
        dq_db = -common * 2.0 * es * p_b
        return q, dq_da, dq_db

    def velocity_ft_s(self, q_scfd: float, p_psia: float, gas: GasProperties) -> float:
        """Actual gas velocity at flowing conditions (ft/s).

        Erosional-velocity screening is the single most common reason a
        hydraulically-valid line size gets rejected in review, so the
        engine has to be able to report it, not just pressures.
        """
        bg = gas.formation_volume_factor(max(p_psia, 1e-6))
        q_actual_ft3_s = abs(q_scfd) * bg / 86400.0
        area_ft2 = math.pi * (self.diameter_in / 12.0) ** 2 / 4.0
        return q_actual_ft3_s / max(area_ft2, 1e-12)

    def erosional_velocity_ft_s(self, p_psia: float, gas: GasProperties,
                                c_factor: float = 100.0) -> float:
        """API RP 14E erosional velocity limit, Ve = C / sqrt(rho)."""
        rho = max(gas.density_lbft3(max(p_psia, 1e-6)), 1e-9)
        return c_factor / math.sqrt(rho)


@dataclass(eq=False)
class LiquidPipe(Branch):
    """A liquid pipe segment under Darcy-Weisbach with Colebrook friction.

    Liquid hydraulics is a pressure *difference* problem, not a
    pressure-squared one -- the fluid is effectively incompressible, so
    there is no gas-style density integration along the line, and static
    head is a plain elevation term rather than an exponential factor.

    The friction factor depends on Reynolds number, which depends on the
    flow this function is trying to compute. Rather than nesting an inner
    solve, f is evaluated at the flow from the previous Newton iteration
    (`_last_q`) and held fixed within the step. Newton then converges on
    the coupled system anyway, because the lagged f converges along with
    the pressures; the derivative below deliberately treats f as constant
    for the same reason a quasi-Newton step does not need an exact Jacobian.
    """

    a: int
    b: int
    diameter_in: float
    length_mi: float
    roughness_in: float = 0.0018  # commercial steel
    elevation_a_ft: float = 0.0
    elevation_b_ft: float = 0.0
    name: str = ""
    _last_q: float = field(default=1000.0, repr=False)

    def __post_init__(self):
        if self.diameter_in <= 0.0:
            raise ValueError("pipe diameter must be positive, got %r" % (self.diameter_in,))
        if self.length_mi <= 0.0:
            raise ValueError("pipe length must be positive, got %r" % (self.length_mi,))

    def reynolds(self, q_bpd: float, liquid: LiquidProperties, p_psia: float) -> float:
        """Re = 92.24 * Q(bpd) * SG / (d(in) * mu(cP))."""
        mu = max(liquid.viscosity_cp(p_psia), 1e-6)
        return 92.24 * abs(q_bpd) * liquid.specific_gravity / (self.diameter_in * mu)

    def _resistance(self, liquid: LiquidProperties, p_avg: float) -> float:
        """C in dP = C * Q^2, with dP in psi and Q in bbl/day.

        dP = 1.1472e-5 * f * L(ft) * SG * Q^2 / d(in)^5 -- the Darcy-Weisbach
        equation reduced to oilfield units (the 0.0605 psi/mile coefficient
        quoted in pipeline handbooks is this same constant times 5280).
        """
        re = self.reynolds(self._last_q, liquid, p_avg)
        f = swamee_jain_friction_factor(re, self.diameter_in, self.roughness_in)
        length_ft = self.length_mi * FT_PER_MI
        return 1.1472e-5 * f * length_ft * liquid.specific_gravity / self.diameter_in ** 5

    def static_head_psi(self, liquid: LiquidProperties) -> float:
        """Pressure the fluid column at `b` exerts back against `a` (psi)."""
        dz = self.elevation_b_ft - self.elevation_a_ft
        return liquid.density_lbft3() * dz / 144.0

    def flow_and_grad(self, p_a: float, p_b: float, fluid: LiquidProperties) -> tuple:
        p_avg = 0.5 * (p_a + p_b)
        c = max(self._resistance(fluid, p_avg), 1e-18)
        driving = p_a - p_b - self.static_head_psi(fluid)

        sign = 1.0 if driving >= 0.0 else -1.0
        dp_abs = max(abs(driving), 1e-12)
        q = sign * math.sqrt(dp_abs / c)
        self._last_q = q

        # d|dp|^0.5 / dp = sign / (2 sqrt(|dp|)), and the leading sign cancels.
        dq_da = 1.0 / (2.0 * math.sqrt(c * dp_abs))
        dq_db = -dq_da
        return q, dq_da, dq_db

    def velocity_ft_s(self, q_bpd: float) -> float:
        area_ft2 = math.pi * (self.diameter_in / 12.0) ** 2 / 4.0
        q_ft3_s = abs(q_bpd) * 5.614583 / 86400.0
        return q_ft3_s / max(area_ft2, 1e-12)


@dataclass(eq=False)
class Valve(Branch):
    """A valve or fitting characterised by a flow coefficient Cv.

    Also the way to model any lumped local loss (a meter run, a filter, a
    bank of fittings) without inventing an equivalent pipe length for it.
    """

    a: int
    b: int
    cv: float
    opening: float = 1.0  # 0..1, multiplies Cv (a linear valve trim)
    name: str = ""
    is_gas: bool = True

    def _effective_cv(self) -> float:
        return max(self.cv * max(min(self.opening, 1.0), 0.0), 1e-9)

    def flow_and_grad(self, p_a: float, p_b: float, fluid) -> tuple:
        cv = self._effective_cv()
        if self.is_gas:
            # Subcritical gas valve sizing, pressure-squared form; the
            # coefficient folds the standard-condition conversion so the
            # result is scf/day like every other gas branch.
            gas = fluid
            p_avg = average_pressure(p_a, p_b)
            z = gas.z_at(p_avg)
            k = 1360.0 * cv * 24.0 / math.sqrt(gas.gravity * gas.temperature_r * z)
            x = (p_a - p_b) * (p_a + p_b)
            sign = 1.0 if x >= 0.0 else -1.0
            x_abs = max(abs(x), 1e-12)
            q = sign * k * math.sqrt(x_abs)
            common = k * 0.5 / math.sqrt(x_abs)
            return q, common * 2.0 * p_a, -common * 2.0 * p_b

        sg = fluid.specific_gravity
        dp = p_a - p_b
        sign = 1.0 if dp >= 0.0 else -1.0
        dp_abs = max(abs(dp), 1e-12)
        # Cv is defined in US gpm; convert to bbl/day (1 gpm = 34.2857 bpd).
        q_gpm = cv * math.sqrt(dp_abs / sg)
        q = sign * q_gpm * 34.2857
        dq = 34.2857 * cv / (2.0 * math.sqrt(sg * dp_abs))
        return q, dq, -dq


@dataclass(eq=False)
class CheckValve(Branch):
    """Wraps another branch and blocks reverse flow.

    Implemented as a very large reverse resistance rather than a hard zero:
    a residual that snaps discontinuously to zero the instant flow reverses
    puts a step in the Jacobian, and Newton chatters across it forever
    instead of settling. A steep-but-finite reverse branch is differentiable,
    so the iteration slides to the closed state and stays there.
    """

    inner: Branch
    reverse_leak_fraction: float = 1e-6
    name: str = ""

    def __post_init__(self):
        self.a = self.inner.a
        self.b = self.inner.b

    def flow_and_grad(self, p_a: float, p_b: float, fluid) -> tuple:
        q, dq_da, dq_db = self.inner.flow_and_grad(p_a, p_b, fluid)
        if q >= 0.0:
            return q, dq_da, dq_db
        k = self.reverse_leak_fraction
        return q * k, dq_da * k, dq_db * k


# --------------------------------------------------------------------------
# Control devices -- these impose a condition instead of obeying a
# flow/pressure relation, so the solver carries a flow unknown for each one.
# --------------------------------------------------------------------------

OUTLET_PRESSURE = "outlet_pressure"
PRESSURE_RATIO = "pressure_ratio"
FIXED_FLOW = "fixed_flow"
INLET_PRESSURE = "inlet_pressure"

_CONTROL_MODES = (OUTLET_PRESSURE, PRESSURE_RATIO, FIXED_FLOW, INLET_PRESSURE)


class ControlBranch(object):
    """A device whose flow is a solver unknown, constrained by a residual.

    A compressor told to hold 900 psia at its discharge does not have a
    flow/pressure relation at all -- it passes whatever the downstream
    network draws, and the pressure is the thing it fixes. Trying to force
    that into q = f(p_a, p_b) is what makes naive network solvers unable to
    represent compression at all.
    """

    a = 0
    b = 0
    name = ""

    def residual_and_grad(self, p_a: float, p_b: float, q: float, fluid) -> tuple:
        """Return (residual, dR/dp_a, dR/dp_b, dR/dq); residual == 0 when
        the device is meeting its setpoint."""
        raise NotImplementedError


@dataclass(eq=False)
class Compressor(ControlBranch):
    """A compressor station.

    Modes: hold a discharge pressure, hold a compression ratio, hold a
    throughput, or hold a suction pressure. Every mode is bounded by
    `max_ratio` -- real machines cannot make arbitrary pressure, and a
    solver allowed to invent unbounded compression will happily converge to
    a network that cannot be built.
    """

    a: int
    b: int
    mode: str = OUTLET_PRESSURE
    setpoint: float = 0.0
    max_ratio: float = 1.6
    efficiency: float = 0.78
    name: str = ""

    def __post_init__(self):
        if self.mode not in _CONTROL_MODES:
            raise ValueError(
                "unknown compressor mode %r (expected one of %s)"
                % (self.mode, ", ".join(_CONTROL_MODES))
            )

    def residual_and_grad(self, p_a: float, p_b: float, q: float, fluid) -> tuple:
        if self.mode == OUTLET_PRESSURE:
            return p_b - self.setpoint, 0.0, 1.0, 0.0
        if self.mode == INLET_PRESSURE:
            return p_a - self.setpoint, 1.0, 0.0, 0.0
        if self.mode == PRESSURE_RATIO:
            return p_b - self.setpoint * p_a, -self.setpoint, 1.0, 0.0
        # FIXED_FLOW
        return q - self.setpoint, 0.0, 0.0, 1.0

    def power_hp(self, q_scfd: float, p_a: float, p_b: float, gas: GasProperties,
                 k_ratio: float = 1.28) -> float:
        """Adiabatic (isentropic) shaft power, horsepower.

        The classic single-stage form:
            HP = 0.0857 * (k/(k-1)) * (Q_MMscfd * T1 * Z_avg / eta)
                 * ((P2/P1)^((k-1)/k) - 1)
        Reported rather than solved for -- fuel gas and driver sizing are
        what an operator actually costs a compression scheme on.
        """
        if p_a <= 0.0 or p_b <= p_a:
            return 0.0
        q_mmscfd = abs(q_scfd) / 1e6
        z_avg = gas.z_at(average_pressure(p_a, p_b))
        exponent = (k_ratio - 1.0) / k_ratio
        return (0.0857 * (k_ratio / (k_ratio - 1.0)) * q_mmscfd * gas.temperature_r
                * z_avg / max(self.efficiency, 1e-6) * ((p_b / p_a) ** exponent - 1.0))

    def ratio_violation(self, p_a: float, p_b: float, tolerance: float = 1e-9) -> float:
        """How far past `max_ratio` this machine is being asked to work
        (0.0 when within its envelope).

        A machine held exactly at its limit lands there to floating-point
        precision, not exactly, so a bare `p_b / p_a - max_ratio` reports a
        violation of ~1e-16 and any caller that treats a positive value as an
        exceedance -- a UI warning, an alarm -- fires spuriously. The tolerance
        is relative and sits far below anything physically meaningful while far
        above float noise.
        """
        if p_a <= 0.0:
            return 0.0
        excess = p_b / p_a - self.max_ratio
        return excess if excess > self.max_ratio * tolerance else 0.0


@dataclass(eq=False)
class Regulator(ControlBranch):
    """A pressure-reducing regulator holding a downstream setpoint.

    A regulator can only reduce pressure. When upstream pressure falls to
    the setpoint the valve is wide open and the device stops controlling --
    it becomes a plain restriction. `solver.py` handles that by switching
    the device between its controlling and wide-open states between Newton
    iterations, rather than trying to express the switch inside a single
    smooth residual (which no smooth residual can do honestly).
    """

    a: int
    b: int
    setpoint: float = 0.0
    wide_open_cv: float = 5000.0
    name: str = ""
    is_gas: bool = True
    controlling: bool = field(default=True, repr=False)

    def residual_and_grad(self, p_a: float, p_b: float, q: float, fluid) -> tuple:
        return p_b - self.setpoint, 0.0, 1.0, 0.0

    def passthrough(self) -> Valve:
        """The wide-open branch this becomes when it can no longer control."""
        return Valve(a=self.a, b=self.b, cv=self.wide_open_cv,
                     name=self.name or "regulator", is_gas=self.is_gas)


@dataclass(eq=False)
class Pump(ControlBranch):
    """A centrifugal pump on a liquid line.

    Two useful idealisations: hold a discharge pressure (mode
    OUTLET_PRESSURE) or hold a throughput (FIXED_FLOW). A real head-flow
    curve is `PumpCurve` below, which is a passive branch instead -- the
    curve itself supplies the flow/pressure relation, so no extra unknown
    is needed.
    """

    a: int
    b: int
    mode: str = OUTLET_PRESSURE
    setpoint: float = 0.0
    efficiency: float = 0.72
    name: str = ""

    def __post_init__(self):
        if self.mode not in (OUTLET_PRESSURE, FIXED_FLOW, PRESSURE_RATIO, INLET_PRESSURE):
            raise ValueError("unknown pump mode %r" % (self.mode,))

    def residual_and_grad(self, p_a: float, p_b: float, q: float, fluid) -> tuple:
        if self.mode == OUTLET_PRESSURE:
            return p_b - self.setpoint, 0.0, 1.0, 0.0
        if self.mode == INLET_PRESSURE:
            return p_a - self.setpoint, 1.0, 0.0, 0.0
        if self.mode == PRESSURE_RATIO:
            return p_b - self.setpoint * p_a, -self.setpoint, 1.0, 0.0
        return q - self.setpoint, 0.0, 0.0, 1.0

    def hydraulic_hp(self, q_bpd: float, p_a: float, p_b: float) -> float:
        """Hydraulic horsepower = Q(bpd) * dP(psi) / 58,776, then divided by
        pump efficiency to get brake horsepower."""
        dp = max(p_b - p_a, 0.0)
        return abs(q_bpd) * dp / 58776.0 / max(self.efficiency, 1e-6)


@dataclass(eq=False)
class PumpCurve(Branch):
    """A pump described by its head-capacity curve, H = h0 - h1*Q - h2*Q^2.

    Unlike `Pump`, this is a passive branch: the curve *is* a relation
    between flow and pressure rise, so it needs no extra unknown. Head is in
    feet, flow in bbl/day, matching how a vendor curve is published.

    The relation is inverted numerically per Newton step (find Q such that
    the pump's rise equals the pressure difference across it), because the
    quadratic's useful branch is the one where head falls with flow.
    """

    a: int
    b: int
    h0_ft: float
    h1_ft_per_bpd: float = 0.0
    h2_ft_per_bpd2: float = 0.0
    name: str = ""

    def head_ft(self, q_bpd: float) -> float:
        return self.h0_ft - self.h1_ft_per_bpd * q_bpd - self.h2_ft_per_bpd2 * q_bpd * q_bpd

    def flow_and_grad(self, p_a: float, p_b: float, fluid: LiquidProperties) -> tuple:
        """Solve h0 - h1*Q - h2*Q^2 = required head, analytically."""
        rho = fluid.density_lbft3()
        required_head_ft = (p_b - p_a) * 144.0 / max(rho, 1e-9)

        # h2*Q^2 + h1*Q + (required - h0) = 0
        c = required_head_ft - self.h0_ft
        if abs(self.h2_ft_per_bpd2) < 1e-18:
            if abs(self.h1_ft_per_bpd) < 1e-18:
                # A flat curve cannot resolve a flow; treat as a rigid
                # constant-head device with a stiff numerical slope.
                return 0.0, 1e-6, -1e-6
            q = -c / self.h1_ft_per_bpd
            dq_dhead = -1.0 / self.h1_ft_per_bpd
        else:
            disc = self.h1_ft_per_bpd ** 2 - 4.0 * self.h2_ft_per_bpd2 * c
            if disc < 0.0:
                # Asked for more head than the pump can make anywhere on its
                # curve: return the shutoff point (zero flow) with a finite
                # slope so Newton can still walk back toward feasibility.
                return 0.0, 1e-6, -1e-6
            sqrt_disc = math.sqrt(disc)
            q = (-self.h1_ft_per_bpd + sqrt_disc) / (2.0 * self.h2_ft_per_bpd2)
            dq_dc = -1.0 / max(sqrt_disc, 1e-12)
            dq_dhead = dq_dc

        dhead_dpb = 144.0 / max(rho, 1e-9)
        dq_dpb = dq_dhead * dhead_dpb
        return q, -dq_dpb, dq_dpb
