"""Time-domain (transient) pipeline simulation.

PRD Section 2 puts transient flow outside v1, and that boundary was right
for the Phase 0 spike. It is here now because two questions cannot be
answered by any steady-state model at all, and both are the reason an
operator buys a simulator in the first place:

  * **Line pack.** A gas pipeline stores a large mass of gas. When demand
    steps up, the line supplies the difference out of storage for hours
    before inlet flow catches up. A steady-state model says the delivery
    pressure collapses immediately; the real line rides through. Sizing
    compression against the steady answer buys machines nobody needs.

  * **Surge.** Closing a valve on a liquid line converts the whole column's
    momentum into pressure. The Joukowsky rise can be many times the
    operating pressure, and it is the case that bursts pipe. No steady-state
    calculation contains the wave speed that governs it.

Two solvers, because the two problems are numerically opposite. Gas
transients are diffusive and slow (hours), so they want an implicit scheme
that can take long timesteps. Liquid surge is a nearly-undamped wave
problem (seconds), where the physics *is* the wave, so it wants the method
of characteristics, which propagates that wave exactly.

Internals are SI (Pa, m, kg/s, s); the dataclasses take and return field
units. Mixing unit systems inside a time-marching loop is how sign and
scale errors survive undetected, so the conversion happens once at each
boundary and never in between.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fluids import GasProperties, LiquidProperties
from .units import FT_PER_MI, M_PER_FT, STD_PRESSURE_PSIA, STD_TEMPERATURE_R

PA_PER_PSI = 6894.757293168
M3_PER_FT3 = M_PER_FT ** 3
G_SI = 9.80665  # m/s^2
# Absolute floor on node pressure during a transient. A gas line cannot go
# sub-atmospheric in normal operation, so reaching this means the demand
# being simulated is simply not deliverable -- reported, never silently
# clamped and returned as though it were an answer.
PRESSURE_FLOOR_PA = 101325.0
R_UNIVERSAL = 8314.462618  # J/(kmol*K)
AIR_MW_KG_KMOL = 28.9625


def _rankine_to_kelvin(t_r: float) -> float:
    return t_r * 5.0 / 9.0


# ==================================================== gas line transient ====


@dataclass
class GasLine:
    """A single gas pipeline for transient simulation."""

    length_mi: float
    diameter_in: float
    roughness_in: float = 0.0007
    elevation_change_ft: float = 0.0
    gas: GasProperties = field(default_factory=GasProperties)
    segments: int = 40

    def __post_init__(self):
        if self.segments < 2:
            raise ValueError("need at least 2 segments, got %r" % (self.segments,))
        if self.diameter_in <= 0.0 or self.length_mi <= 0.0:
            raise ValueError("pipe dimensions must be positive")

    @property
    def length_m(self) -> float:
        return self.length_mi * FT_PER_MI * M_PER_FT

    @property
    def diameter_m(self) -> float:
        return self.diameter_in / 12.0 * M_PER_FT

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m ** 2 / 4.0

    def wave_speed_squared(self, p_psia: float) -> float:
        """a^2 = ZRT/M, in m^2/s^2 -- the isothermal speed of sound, which
        sets how fast a pressure change travels down the line."""
        z = self.gas.z_at(max(p_psia, 1.0))
        t_k = _rankine_to_kelvin(self.gas.temperature_r)
        mw = self.gas.gravity * AIR_MW_KG_KMOL
        return z * R_UNIVERSAL * t_k / mw

    def standard_density_kg_m3(self) -> float:
        """Gas density at standard conditions, for scf/day <-> kg/s."""
        t_k = _rankine_to_kelvin(STD_TEMPERATURE_R)
        mw = self.gas.gravity * AIR_MW_KG_KMOL
        p_pa = STD_PRESSURE_PSIA * PA_PER_PSI
        return p_pa * mw / (R_UNIVERSAL * t_k)

    def scfd_to_kg_s(self, q_scfd: float) -> float:
        return q_scfd * M3_PER_FT3 / 86400.0 * self.standard_density_kg_m3()

    def kg_s_to_scfd(self, w: float) -> float:
        return w * 86400.0 / (M3_PER_FT3 * self.standard_density_kg_m3())


@dataclass
class TransientResult:
    times_s: list
    inlet_pressure_psia: list
    outlet_pressure_psia: list
    inlet_flow_scfd: list
    outlet_flow_scfd: list
    line_pack_scf: list
    pressure_profiles: list = field(default_factory=list)  # [(t, [psia per node]), ...]
    notes: list = field(default_factory=list)
    floor_reached_at_s: float = -1.0

    @property
    def feasible(self) -> bool:
        """False when the line was drawn down to the pressure floor.

        A run that hits the floor is not a low-pressure prediction -- it is
        the model saying the scheduled offtake cannot be delivered at all.
        Reporting the floor value as if it were a pressure would be exactly
        the confidently-wrong answer PRD Section 6 rules out.
        """
        return self.floor_reached_at_s < 0.0

    @property
    def minimum_outlet_pressure_psia(self) -> float:
        return min(self.outlet_pressure_psia)

    def time_to_outlet_below(self, threshold_psia: float) -> float:
        """Seconds until delivery pressure first drops below a limit, or -1
        if it never does. This -- not the final steady state -- is the number
        an operator plans against."""
        for t, p in zip(self.times_s, self.outlet_pressure_psia):
            if p < threshold_psia:
                return t
        return -1.0


def simulate_gas_transient(
    line: GasLine,
    inlet_pressure_psia: float,
    outlet_flow_schedule,
    duration_s: float,
    dt_s: float = 60.0,
    initial_outlet_flow_scfd: float = None,
    profile_every: int = 0,
) -> TransientResult:
    """Isothermal transient gas flow: fixed inlet pressure, scheduled offtake.

    `outlet_flow_schedule` is a callable t(seconds) -> scf/day withdrawn at
    the far end, so a demand step, a ramp, or a daily profile are all the
    same call.

    Scheme: staggered grid (pressure at nodes, mass flow at faces), backward
    Euler in time, with the quadratic friction term linearised around the
    previous step's flow. Implicit on purpose -- an explicit scheme here is
    limited by the acoustic CFL condition to timesteps of a fraction of a
    second, and simulating eight hours of line pack that way would take
    ~10^5 steps to answer a question the implicit scheme answers in 480.
    Backward Euler is only first-order accurate and slightly over-damps fast
    transients, which is an acceptable trade for line-pack studies and is
    *not* acceptable for surge -- that is why liquid surge below uses the
    method of characteristics instead.

    The equations, per unit length:
        continuity  (A/a^2) dp/dt + dw/dx = 0
        momentum    (1/A) dw/dt + dp/dx + f w|w| / (2 D A^2 rho) + rho g sin(theta) = 0
    """
    n = line.segments
    dx = line.length_m / n
    area = line.area_m2
    d_m = line.diameter_m
    sin_theta = _elevation_sin(line)

    p_in_pa = inlet_pressure_psia * PA_PER_PSI

    if initial_outlet_flow_scfd is None:
        initial_outlet_flow_scfd = outlet_flow_schedule(0.0)

    # Start from the steady state that matches the initial offtake, so the
    # run begins on the line's actual operating point rather than lurching
    # away from a flat guess in the first few steps.
    w0 = line.scfd_to_kg_s(initial_outlet_flow_scfd)
    p = _steady_profile(line, p_in_pa, w0, dx, sin_theta)
    w = np.full(n, w0, dtype=float)

    a2 = line.wave_speed_squared(inlet_pressure_psia)
    node_volume = area * dx

    times = []
    inlet_p, outlet_p, inlet_q, outlet_q, pack = [], [], [], [], []
    profiles = []
    notes = []
    floor_reached_at = -1.0

    n_steps = int(round(duration_s / dt_s))
    # Unknowns: p_0..p_n (n+1 nodes) then w_0..w_{n-1} (n faces)
    size = (n + 1) + n

    for step in range(n_steps + 1):
        t = step * dt_s
        w_out = line.scfd_to_kg_s(outlet_flow_schedule(t))

        times.append(t)
        inlet_p.append(p[0] / PA_PER_PSI)
        outlet_p.append(p[n] / PA_PER_PSI)
        inlet_q.append(line.kg_s_to_scfd(w[0]))
        outlet_q.append(line.kg_s_to_scfd(w_out))
        mass_kg = float(np.sum(_node_masses(p, a2, node_volume)))
        pack.append(mass_kg / line.standard_density_kg_m3() / M3_PER_FT3)
        if profile_every and step % profile_every == 0:
            profiles.append((t, [float(v / PA_PER_PSI) for v in p]))

        if step == n_steps:
            break

        matrix = np.zeros((size, size))
        rhs = np.zeros(size)

        # Node 0: fixed inlet pressure.
        matrix[0, 0] = 1.0
        rhs[0] = p_in_pa

        # Interior nodes: continuity over a full cell.
        for i in range(1, n):
            row = i
            matrix[row, i] = node_volume / (a2 * dt_s)
            rhs[row] = node_volume / (a2 * dt_s) * p[i]
            matrix[row, (n + 1) + (i - 1)] = -1.0
            matrix[row, (n + 1) + i] = 1.0

        # Node n: continuity over a half cell, closed by the scheduled offtake.
        row = n
        half = node_volume / 2.0
        matrix[row, n] = half / (a2 * dt_s)
        rhs[row] = half / (a2 * dt_s) * p[n]
        matrix[row, (n + 1) + (n - 1)] = -1.0
        rhs[row] -= w_out

        # Faces: momentum.
        for i in range(n):
            row = (n + 1) + i
            p_face = 0.5 * (p[i] + p[i + 1])
            rho = max(p_face / a2, 1e-6)
            f = _darcy_friction(line, abs(w[i]), rho)
            resistance = f * abs(w[i]) / (2.0 * d_m * area ** 2 * rho)

            matrix[row, (n + 1) + i] = 1.0 / (area * dt_s) + resistance
            matrix[row, i] = -1.0 / dx
            matrix[row, i + 1] = 1.0 / dx
            rhs[row] = w[i] / (area * dt_s) - rho * G_SI * sin_theta

        solution = np.linalg.solve(matrix, rhs)
        raw = solution[: n + 1]
        if floor_reached_at < 0.0 and float(np.min(raw)) <= PRESSURE_FLOOR_PA:
            floor_reached_at = t + dt_s
            notes.append(
                "line drawn down to the %.1f psia floor at t = %.2f h: the scheduled "
                "offtake is not deliverable at this inlet pressure, so pressures after "
                "this point are not a prediction"
                % (PRESSURE_FLOOR_PA / PA_PER_PSI, floor_reached_at / 3600.0))
        p = np.maximum(raw, PRESSURE_FLOOR_PA)
        w = solution[n + 1:]

    return TransientResult(
        times_s=times,
        inlet_pressure_psia=inlet_p,
        outlet_pressure_psia=outlet_p,
        inlet_flow_scfd=inlet_q,
        outlet_flow_scfd=outlet_q,
        line_pack_scf=pack,
        pressure_profiles=profiles,
        notes=notes,
        floor_reached_at_s=floor_reached_at,
    )


def _elevation_sin(line: GasLine) -> float:
    """sin(theta) for the line, treated as one uniform slope."""
    rise_m = line.elevation_change_ft * M_PER_FT
    length = line.length_m
    if length <= 0.0:
        return 0.0
    return max(min(rise_m / length, 1.0), -1.0)


def _node_masses(p: np.ndarray, a2: float, node_volume: float) -> np.ndarray:
    """Gas mass attributed to each node (end nodes own half a cell)."""
    rho = p / a2
    weights = np.full(p.shape[0], node_volume)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return rho * weights


def _darcy_friction(line: GasLine, w_kg_s: float, rho: float) -> float:
    """Darcy friction factor from the current mass flow."""
    if w_kg_s <= 1e-12:
        return 0.02
    velocity = w_kg_s / (rho * line.area_m2)
    mu_cp = line.gas.viscosity_cp(500.0)
    mu_pa_s = mu_cp * 1e-3
    reynolds = rho * abs(velocity) * line.diameter_m / max(mu_pa_s, 1e-9)
    rel_rough = (line.roughness_in / 12.0 * M_PER_FT) / line.diameter_m
    if reynolds < 2100.0:
        return 64.0 / max(reynolds, 1e-6)
    denom = math.log10(rel_rough / 3.7 + 5.74 / (reynolds ** 0.9))
    return 0.25 / (denom ** 2)


def _steady_profile(line: GasLine, p_in_pa: float, w_kg_s: float, dx: float,
                    sin_theta: float) -> np.ndarray:
    """March the steady momentum equation down the line for an initial condition."""
    n = line.segments
    p = np.zeros(n + 1)
    p[0] = p_in_pa
    a2 = line.wave_speed_squared(p_in_pa / PA_PER_PSI)
    area = line.area_m2
    for i in range(n):
        rho = max(p[i] / a2, 1e-6)
        f = _darcy_friction(line, abs(w_kg_s), rho)
        drop = dx * (f * w_kg_s * abs(w_kg_s) / (2.0 * line.diameter_m * area ** 2 * rho)
                     + rho * G_SI * sin_theta)
        p[i + 1] = max(p[i] - drop, 1e3)
    return p


# ================================================ liquid surge (waterhammer) =


@dataclass
class SurgeLine:
    """A liquid pipeline for surge (waterhammer) analysis."""

    length_mi: float
    diameter_in: float
    wave_speed_ft_s: float = 3300.0  # typical for oil in steel pipe
    roughness_in: float = 0.0018
    liquid: LiquidProperties = field(default_factory=LiquidProperties)
    segments: int = 20

    def __post_init__(self):
        if self.segments < 2:
            raise ValueError("need at least 2 segments, got %r" % (self.segments,))

    @property
    def length_m(self) -> float:
        return self.length_mi * FT_PER_MI * M_PER_FT

    @property
    def diameter_m(self) -> float:
        return self.diameter_in / 12.0 * M_PER_FT

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m ** 2 / 4.0

    @property
    def wave_speed_m_s(self) -> float:
        return self.wave_speed_ft_s * M_PER_FT

    @property
    def density_kg_m3(self) -> float:
        return self.liquid.specific_gravity * 999.0

    def timestep_s(self) -> float:
        """The MOC timestep is not a free parameter: dt = dx / a exactly, so
        that characteristics land on grid points. Any other value requires
        interpolation, which numerically damps the very wave being
        simulated."""
        return (self.length_m / self.segments) / self.wave_speed_m_s

    def joukowsky_rise_psi(self, velocity_change_ft_s: float) -> float:
        """The classic instantaneous-closure surge, dP = rho * a * dV.

        The upper bound any surge result must respect: a valve cannot make
        more pressure than stopping the column's whole momentum in one wave
        transit. A simulated surge above this means the numerics are ringing,
        not the pipe.
        """
        dv_m_s = abs(velocity_change_ft_s) * M_PER_FT
        return self.density_kg_m3 * self.wave_speed_m_s * dv_m_s / PA_PER_PSI


@dataclass
class SurgeResult:
    times_s: list
    head_profiles: list          # [[psia per node], ...] one row per stored step
    valve_pressure_psia: list
    upstream_pressure_psia: list
    node_positions_mi: list

    @property
    def maximum_pressure_psia(self) -> float:
        return max(max(row) for row in self.head_profiles)

    @property
    def minimum_pressure_psia(self) -> float:
        return min(min(row) for row in self.head_profiles)

    def peak_pressure_by_node_psia(self) -> list:
        """The envelope an engineer actually designs against: the highest
        pressure each point on the line sees at any time during the event,
        not the highest pressure anywhere at one instant."""
        columns = list(zip(*self.head_profiles))
        return [max(col) for col in columns]


def simulate_valve_closure(
    line: SurgeLine,
    upstream_pressure_psia: float,
    initial_flow_bpd: float,
    closure_time_s: float,
    duration_s: float,
    closure_exponent: float = 1.0,
) -> SurgeResult:
    """Method of characteristics: close a downstream valve and watch the wave.

    Upstream is a constant-pressure source (a tank or a large header);
    downstream is a valve whose open area follows `closure_time_s`. An
    instantaneous closure (closure_time_s = 0) reproduces the Joukowsky
    bound; anything slower than the pipe period 2L/a is "slow closure" and
    produces markedly less surge, which is the entire basis of surge
    mitigation by closure scheduling.

    `closure_exponent` m shapes the stroke: tau = (1 - t/tc)^m. m = 1 is a
    linear stroke. m < 1 holds the valve effectively open for most of the
    stroke and shuts it abruptly at the end, which is markedly worse for
    surge; m > 1 does most of the closing early and is gentler. Which
    behaviour a real trim has depends on its installed characteristic, so
    this is exposed as a number rather than guessed at behind a trim name.

    Solved on characteristics rather than by a finite-difference scheme
    because the C+ and C- characteristics carry the wave exactly when
    dt = dx/a. A general implicit scheme with the same grid would smear the
    wavefront over several cells and under-predict the peak -- the one
    number the whole calculation exists to produce.
    """
    n = line.segments
    dx = line.length_m / n
    dt = line.timestep_s()
    a = line.wave_speed_m_s
    area = line.area_m2
    rho = line.density_kg_m3
    g = G_SI

    # Work in head (m) and discharge (m^3/s), the natural MOC variables.
    b = a / (g * area)
    f = 0.02  # updated below from the initial Reynolds number
    q0 = initial_flow_bpd * 5.614583 * M3_PER_FT3 / 86400.0
    v0 = q0 / area
    mu_pa_s = max(line.liquid.viscosity_cp() * 1e-3, 1e-6)
    reynolds = rho * abs(v0) * line.diameter_m / mu_pa_s
    if reynolds > 2100.0:
        rel_rough = (line.roughness_in / 12.0 * M_PER_FT) / line.diameter_m
        f = 0.25 / (math.log10(rel_rough / 3.7 + 5.74 / (reynolds ** 0.9)) ** 2)
    elif reynolds > 1e-9:
        f = 64.0 / reynolds
    r = f * dx / (2.0 * g * line.diameter_m * area ** 2)

    head_upstream = upstream_pressure_psia * PA_PER_PSI / (rho * g)

    # Steady initial condition: head falls linearly with friction along the line.
    h = np.array([head_upstream - r * q0 * abs(q0) * i for i in range(n + 1)])
    q = np.full(n + 1, q0)

    h0_valve = max(h[n], 1e-6)
    times = []
    profiles = []
    valve_p = []
    upstream_p = []

    n_steps = max(int(round(duration_s / dt)), 1)
    for step in range(n_steps + 1):
        t = step * dt
        times.append(t)
        profiles.append([float(v * rho * g / PA_PER_PSI) for v in h])
        valve_p.append(float(h[n] * rho * g / PA_PER_PSI))
        upstream_p.append(float(h[0] * rho * g / PA_PER_PSI))
        if step == n_steps:
            break

        h_new = np.empty_like(h)
        q_new = np.empty_like(q)

        # Interior nodes from the two characteristics meeting there.
        cp = h[:-2] + b * q[:-2] - r * q[:-2] * np.abs(q[:-2])
        cm = h[2:] - b * q[2:] + r * q[2:] * np.abs(q[2:])
        h_new[1:-1] = 0.5 * (cp + cm)
        q_new[1:-1] = (cp - cm) / (2.0 * b)

        # Upstream boundary: constant-pressure source, only C- reaches it.
        cm0 = h[1] - b * q[1] + r * q[1] * abs(q[1])
        h_new[0] = head_upstream
        q_new[0] = (head_upstream - cm0) / b

        # Downstream boundary: the valve, only C+ reaches it.
        cpn = h[n - 1] + b * q[n - 1] - r * q[n - 1] * abs(q[n - 1])
        tau = _valve_opening(t + dt, closure_time_s, closure_exponent)
        if tau <= 1e-12:
            q_new[n] = 0.0
            h_new[n] = cpn
        else:
            # Orifice law Q = tau * Q0 * sqrt(H / H0) combined with the C+
            # characteristic H = Cp - B*Q gives a quadratic in Q.
            cv = (tau * q0) ** 2 / h0_valve
            # Q^2 + cv*B*Q - cv*Cp = 0
            disc = (cv * b) ** 2 + 4.0 * cv * max(cpn, 0.0)
            q_new[n] = 0.5 * (-cv * b + math.sqrt(max(disc, 0.0)))
            h_new[n] = cpn - b * q_new[n]

        h, q = h_new, q_new

    positions = [i * line.length_mi / n for i in range(n + 1)]
    return SurgeResult(
        times_s=times,
        head_profiles=profiles,
        valve_pressure_psia=valve_p,
        upstream_pressure_psia=upstream_p,
        node_positions_mi=positions,
    )


def _valve_opening(t: float, closure_time_s: float, exponent: float) -> float:
    """Dimensionless valve opening tau, 1 fully open to 0 shut.

    tau = (1 - t/tc)^m. The exponent is what decides how bad the surge is
    for a given nominal closure time: the damage is done by how much flow is
    still moving in the last fraction of the stroke, not by the stroke time
    alone.
    """
    if exponent <= 0.0:
        raise ValueError("closure exponent must be positive, got %r" % (exponent,))
    if closure_time_s <= 0.0:
        return 0.0
    if t >= closure_time_s:
        return 0.0
    return (1.0 - t / closure_time_s) ** exponent


def pipe_period_s(line: SurgeLine) -> float:
    """4L/a -- the time for a wave to travel to the far end and back twice.

    The dividing line between "rapid" and "slow" closure: a valve shut in
    less than 2L/a sees the full Joukowsky rise, because the relief wave
    from the far end has not returned yet.
    """
    return 4.0 * line.length_m / line.wave_speed_m_s
