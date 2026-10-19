"""Flowing temperature along a pipe or a well.

Everything else in this engine is isothermal: temperature is an input to the
flow equations, held at one value for a whole line. That is a reasonable
approximation for a short gas gathering line and a bad one for almost anything
else, because temperature is not a passive label. It sets Z, viscosity and
density, so it feeds straight back into the pressure drop; and it decides
whether a line makes hydrates, drops wax, or arrives warm enough to process.

Two mechanisms move the temperature of a flowing stream, and the second is the
one people forget:

  * Heat exchange with the surroundings -- the soil, the sea, the air, the
    formation. This pulls the stream towards ambient with a characteristic
    length, not a characteristic time, because the fluid is moving.

  * Joule-Thomson expansion. A real gas losing pressure cools even with
    perfectly insulated walls, because it does work against its own
    intermolecular attraction. On a long transmission line this is often the
    *larger* of the two effects, and a model that omits it will happily report
    a line arriving above the hydrate curve when it does not.

The governing balance, per unit length along the flow path:

    dT/dx = -(T - T_ambient(x)) / L + mu_JT * dP/dx - g*sin(theta) / (Cp * J)

where L = m_dot * Cp / (U * pi * D) is the relaxation length: the distance over
which the stream closes 63% of its gap to ambient. It is the single number that
says whether a line is "long" thermally, and it is worth reading off directly.

This module solves that balance rather than assuming its answer, and is checked
against the closed-form solution of the same equation -- including with a
geothermal gradient, where the fluid chases a moving target.

Conventions: x is distance along the flow path in feet (measured depth for a
well, so a producer has x running up the hole); temperatures in degrees
Rankine; pressure in psia; the pressure profile is supplied by whichever
hydraulics model produced it, because temperature does not get to invent its
own pressure drop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .fluids import AIR_MW, R_GAS, GasProperties, z_factor_dak
from .units import r_to_f

# Mechanical equivalent of heat, in the units this engine works in. One Btu is
# 5.40395 psia*ft3, which is what converts the Joule-Thomson group into
# degrees per psi.
PSIA_FT3_PER_BTU = 5.40395

# Gravitational term: converting potential energy per unit mass into degrees
# requires the mechanical equivalent of heat, 778.169 ft*lbf per Btu.
FT_LBF_PER_BTU = 778.169

# Typical lean natural gas at pipeline conditions, Btu/(lb*degF). This is an
# *input* with a documented default, not a correlation -- it varies with
# composition, pressure and temperature, and a real study should take it from
# the gas analysis rather than from here.
DEFAULT_GAS_CP_BTU_LB_F = 0.55

# Hammerschmidt's hydrate correlation was fitted over roughly this range.
# Outside it the number returned is an extrapolation and is reported as such,
# because a hydrate prediction that does not say when it is guessing is worse
# than no prediction.
HAMMERSCHMIDT_MIN_PSIA = 300.0
HAMMERSCHMIDT_MAX_PSIA = 4000.0


def joule_thomson_coefficient_f_per_psi(
    gas: GasProperties,
    p_psia: float,
    t_r: float,
    cp_btu_lb_f: float = DEFAULT_GAS_CP_BTU_LB_F,
) -> float:
    """Joule-Thomson coefficient, degrees F per psi, from the equation of state.

    For a real gas,

        mu_JT = (1/Cp) * [ T*(dv/dT)_P - v ]

    and with v = Z*R*T/(P*M) that whole bracket collapses to

        mu_JT = R * T^2 * (dZ/dT)_P / (P * M * Cp)

    so the entire temperature effect rides on how Z varies with temperature.
    That is worth stating plainly: an ideal gas has dZ/dT = 0 and therefore
    does not Joule-Thomson cool at all. Any model that treats the gas as ideal
    silently predicts a warmer arrival temperature than reality, in the
    direction that hides a hydrate risk.

    (dZ/dT)_P is taken numerically from the Dranchuk-Abou-Kassem correlation
    already used for compressibility, so this inherits that correlation's
    validity and needs no separate fit.

    Returns degrees per psi, positive meaning the gas *cools* as pressure
    falls -- which is the usual sign for natural gas at pipeline conditions.
    """
    if cp_btu_lb_f <= 0.0:
        raise ValueError("heat capacity must be positive, got %r" % (cp_btu_lb_f,))
    if p_psia <= 0.0:
        raise ValueError("pressure must be positive, got %r" % (p_psia,))
    if t_r <= 0.0:
        raise ValueError("temperature must be positive, got %r" % (t_r,))

    # A pinned Z is a constant, so its temperature derivative is zero and the
    # gas does not Joule-Thomson cool at all. Honouring the override matters:
    # reading the correlation directly here would have quietly given a fixed-Z
    # "ideal" gas the full real-gas cooling, which is the exact opposite of
    # what pinning Z means.
    if gas.fixed_z > 0.0:
        return 0.0

    p_pc, t_pc = gas.pseudo_criticals()

    # A central difference on temperature. The step is a small fraction of the
    # absolute temperature rather than a fixed number of degrees, so it stays
    # well-conditioned from cryogenic to hot-well conditions.
    step = max(t_r * 1e-4, 1e-3)
    z_hi = z_factor_dak(p_psia, t_r + step, p_pc, t_pc)
    z_lo = z_factor_dak(p_psia, t_r - step, p_pc, t_pc)
    dz_dt = (z_hi - z_lo) / (2.0 * step)

    molecular_weight = gas.gravity * AIR_MW
    group = R_GAS * t_r * t_r * dz_dt / (p_psia * molecular_weight * cp_btu_lb_f)
    return group / PSIA_FT3_PER_BTU


@dataclass
class ThermalPath:
    """The heat-transfer geometry of one pipe or one string of tubing.

    ``overall_u`` is the overall heat transfer coefficient referred to the
    outside diameter used here, in Btu/(hr*ft2*degF). Indicative values, all of
    which should be replaced by a real calculation for a real line:

        bare pipe, buried, wet soil        0.4 - 0.8
        bare pipe, submerged, flowing sea  2 - 6
        bare pipe, above ground, in air    1 - 3
        insulated (2 in, typical)          0.05 - 0.2
        tubing inside cased, cemented hole 0.2 - 1.0

    ``ambient_r`` is the surrounding temperature where the fluid enters, and
    ``ambient_gradient_r_per_ft`` is how that surrounding temperature changes
    along the flow path. For a producing well the fluid flows *up* a falling
    geothermal profile, so the gradient is negative in flow-path terms; for a
    buried line it is normally zero.
    """

    diameter_ft: float
    overall_u: float
    ambient_r: float
    ambient_gradient_r_per_ft: float = 0.0
    # Positive when the flow path climbs, i.e. +90 for a producing well.
    inclination_deg: float = 0.0

    def __post_init__(self):
        if self.diameter_ft <= 0.0:
            raise ValueError("diameter must be positive, got %r" % (self.diameter_ft,))
        if self.overall_u < 0.0:
            raise ValueError("overall U cannot be negative, got %r" % (self.overall_u,))
        if self.ambient_r <= 0.0:
            raise ValueError("ambient temperature must be positive (Rankine), got %r"
                             % (self.ambient_r,))
        if not -90.0 <= self.inclination_deg <= 90.0:
            raise ValueError("inclination must be within +/-90 degrees, got %r"
                             % (self.inclination_deg,))

    def ambient_at(self, x_ft: float) -> float:
        return self.ambient_r + self.ambient_gradient_r_per_ft * x_ft


@dataclass
class ThermalStream:
    """The flowing stream's thermal capacity."""

    mass_rate_lb_hr: float
    cp_btu_lb_f: float = DEFAULT_GAS_CP_BTU_LB_F

    def __post_init__(self):
        if self.mass_rate_lb_hr <= 0.0:
            raise ValueError("mass rate must be positive, got %r"
                             % (self.mass_rate_lb_hr,))
        if self.cp_btu_lb_f <= 0.0:
            raise ValueError("heat capacity must be positive, got %r"
                             % (self.cp_btu_lb_f,))


def relaxation_length_ft(path: ThermalPath, stream: ThermalStream) -> float:
    """Distance over which the stream closes 63% of its gap to ambient.

    The number worth quoting about any line. If it is far longer than the line,
    the line is effectively adiabatic and Joule-Thomson dominates; if it is far
    shorter, the fluid arrives at ambient no matter what it started at, and
    insulation upstream is wasted money.

    Infinite for a perfectly insulated path, which is the correct answer rather
    than an error.
    """
    denominator = path.overall_u * math.pi * path.diameter_ft
    if denominator <= 0.0:
        return math.inf
    return stream.mass_rate_lb_hr * stream.cp_btu_lb_f / denominator


def elevation_cooling_r_per_ft(path: ThermalPath, stream: ThermalStream) -> float:
    """Temperature lost per foot climbed, converting potential energy to heat.

    Small in a pipeline and not small in a deep well: lifting a fluid 10,000 ft
    costs it about 25 degF at a typical gas heat capacity, which is the same
    order as everything else in the balance.
    """
    return (math.sin(math.radians(path.inclination_deg))
            / (stream.cp_btu_lb_f * FT_LBF_PER_BTU))


@dataclass
class TemperatureProfile:
    """Flowing temperature along the path, with the terms that produced it."""

    distances_ft: list
    temperatures_r: list
    pressures_psia: list
    relaxation_length_ft: float
    # Degrees attributable to each mechanism over the whole path. These are
    # reported separately because the design response differs completely:
    # ambient loss is answered with insulation, JT cooling is answered with
    # heating or with pressure management.
    ambient_delta_r: float
    joule_thomson_delta_r: float
    elevation_delta_r: float

    @property
    def inlet_r(self) -> float:
        return self.temperatures_r[0]

    @property
    def outlet_r(self) -> float:
        return self.temperatures_r[-1]

    @property
    def outlet_f(self) -> float:
        return r_to_f(self.temperatures_r[-1])

    def temperature_at(self, x_ft: float) -> float:
        """Linear interpolation onto an arbitrary station."""
        xs, ts = self.distances_ft, self.temperatures_r
        if x_ft <= xs[0]:
            return ts[0]
        if x_ft >= xs[-1]:
            return ts[-1]
        for i in range(1, len(xs)):
            if x_ft <= xs[i]:
                span = xs[i] - xs[i - 1]
                if span <= 0.0:
                    return ts[i]
                weight = (x_ft - xs[i - 1]) / span
                return ts[i - 1] + weight * (ts[i] - ts[i - 1])
        return ts[-1]


def temperature_profile(
    inlet_temperature_r: float,
    path: ThermalPath,
    stream: ThermalStream,
    length_ft: float,
    pressures_psia: list = None,
    gas: GasProperties = None,
    segments: int = 50,
    joule_thomson_f_per_psi: float = None,
) -> TemperatureProfile:
    """Integrate the energy balance along the path.

    ``pressures_psia`` is the pressure at each station, supplied by whatever
    hydraulics model produced it. Temperature does not get to invent its own
    pressure drop, and coupling the two properly means iterating between this
    and the flow solver -- which is the caller's business, not this function's.

    If ``gas`` is given the Joule-Thomson coefficient is evaluated from the
    equation of state at each station, so it varies along the line as it
    physically does. ``joule_thomson_f_per_psi`` pins it to a constant instead,
    which is what makes the closed-form comparison in the tests possible.

    Integrated with the midpoint rule on each segment. The balance is a stiff
    linear relaxation when the line is much longer than L, and explicit Euler
    on a coarse grid will overshoot ambient and oscillate -- producing a
    temperature profile that visibly rings, which is a numerical artefact that
    has been mistaken for a physical effect more than once.
    """
    if segments < 1:
        raise ValueError("need at least one segment, got %r" % (segments,))
    if length_ft <= 0.0:
        raise ValueError("length must be positive, got %r" % (length_ft,))
    if inlet_temperature_r <= 0.0:
        raise ValueError("inlet temperature must be positive (Rankine), got %r"
                         % (inlet_temperature_r,))
    if gas is None and joule_thomson_f_per_psi is None:
        joule_thomson_f_per_psi = 0.0

    stations = segments + 1
    dx = length_ft / segments
    distances = [i * dx for i in range(stations)]

    if pressures_psia is None:
        pressures = [0.0] * stations
    elif len(pressures_psia) != stations:
        raise ValueError(
            "pressures must have one value per station: expected %d for %d "
            "segments, got %d" % (stations, segments, len(pressures_psia)))
    else:
        pressures = list(pressures_psia)

    relaxation = relaxation_length_ft(path, stream)
    elevation_rate = elevation_cooling_r_per_ft(path, stream)

    temperatures = [inlet_temperature_r]
    ambient_total = 0.0
    jt_total = 0.0
    elevation_total = 0.0

    for i in range(segments):
        t = temperatures[-1]
        x0, x1 = distances[i], distances[i + 1]

        dp = pressures[i + 1] - pressures[i] if pressures_psia is not None else 0.0
        p_mid = 0.5 * (pressures[i] + pressures[i + 1]) if pressures_psia else 0.0

        if joule_thomson_f_per_psi is not None:
            mu = joule_thomson_f_per_psi
        else:
            mu = joule_thomson_coefficient_f_per_psi(gas, p_mid, t, stream.cp_btu_lb_f)

        # Midpoint rule: evaluate the relaxation term at the half-step
        # temperature rather than at the segment inlet. Explicit Euler is
        # unstable here once dx approaches 2L, and the instability looks like
        # a physically plausible oscillation rather than like a failure.
        ambient_mid = path.ambient_at(0.5 * (x0 + x1))
        jt_and_elevation = mu * dp - elevation_rate * dx

        if math.isinf(relaxation):
            t_next = t + jt_and_elevation
            ambient_step = 0.0
        else:
            half = 0.5 * dx / relaxation
            t_half = t + 0.5 * jt_and_elevation - half * (t - ambient_mid)
            ambient_step = -(dx / relaxation) * (t_half - ambient_mid)
            t_next = t + ambient_step + jt_and_elevation

        temperatures.append(t_next)
        ambient_total += ambient_step
        jt_total += mu * dp
        elevation_total += -elevation_rate * dx

    return TemperatureProfile(
        distances_ft=distances,
        temperatures_r=temperatures,
        pressures_psia=pressures,
        relaxation_length_ft=relaxation,
        ambient_delta_r=ambient_total,
        joule_thomson_delta_r=jt_total,
        elevation_delta_r=elevation_total,
    )


def analytic_temperature_r(
    inlet_temperature_r: float,
    path: ThermalPath,
    stream: ThermalStream,
    x_ft: float,
    pressure_gradient_psi_per_ft: float = 0.0,
    joule_thomson_f_per_psi: float = 0.0,
) -> float:
    """Closed-form solution of the same balance, for constant properties.

    Solving

        dT/dx = -(T - Ta0 - a*x)/L + C,   C = mu*G - g*sin(theta)/(Cp*J)

    with the substitution theta = T - Ta(x) gives

        T(x) = Ta(x) + L*(C - a) + [T0 - Ta0 - L*(C - a)] * exp(-x/L)

    This is what the numerical integration is checked against, including the
    geothermal case where the fluid is chasing a moving ambient -- and the
    ``L*(C - a)`` offset is the physically interesting part of that: a stream
    in a linear gradient does not approach ambient, it approaches a fixed
    offset from it, lagging behind by exactly one relaxation length's worth of
    gradient.
    """
    relaxation = relaxation_length_ft(path, stream)
    drive = (joule_thomson_f_per_psi * pressure_gradient_psi_per_ft
             - elevation_cooling_r_per_ft(path, stream))
    ambient_here = path.ambient_at(x_ft)

    if math.isinf(relaxation):
        # Perfectly insulated: the ambient term drops out entirely and the
        # stream simply accumulates whatever JT and elevation do to it.
        return inlet_temperature_r + drive * x_ft

    offset = relaxation * (drive - path.ambient_gradient_r_per_ft)
    initial_gap = inlet_temperature_r - path.ambient_r - offset
    return ambient_here + offset + initial_gap * math.exp(-x_ft / relaxation)


@dataclass
class HydrateCheck:
    """Whether the stream is in the hydrate region, and by how much."""

    pressure_psia: float
    temperature_r: float
    hydrate_temperature_r: float
    extrapolated: bool

    @property
    def margin_f(self) -> float:
        """Degrees above the hydrate curve. Negative means hydrates form."""
        return self.temperature_r - self.hydrate_temperature_r

    @property
    def forms(self) -> bool:
        return self.margin_f < 0.0


def hydrate_temperature_r(p_psia: float) -> tuple:
    """Hydrate formation temperature for natural gas, Hammerschmidt.

        T [degF] = 8.9 * P^0.285,  P in psia

    A correlation, not an equilibrium calculation. It carries no composition
    beyond "natural gas", and a real inhibitor study needs a flash against a
    hydrate equation of state. It is here because it is the check that costs
    nothing and catches the case that matters: a line whose arrival
    temperature, once Joule-Thomson cooling is accounted for, has dropped into
    the region where it will plug.

    Returns (temperature_r, extrapolated), the flag being true outside the
    range the correlation was fitted over, because a hydrate number that does
    not say when it is guessing is worse than no number at all.
    """
    if p_psia <= 0.0:
        raise ValueError("pressure must be positive, got %r" % (p_psia,))
    extrapolated = not (HAMMERSCHMIDT_MIN_PSIA <= p_psia <= HAMMERSCHMIDT_MAX_PSIA)
    t_f = 8.9 * (p_psia ** 0.285)
    return t_f + 459.67, extrapolated


def hydrate_check(p_psia: float, t_r: float) -> HydrateCheck:
    hydrate_r, extrapolated = hydrate_temperature_r(p_psia)
    return HydrateCheck(pressure_psia=p_psia, temperature_r=t_r,
                        hydrate_temperature_r=hydrate_r, extrapolated=extrapolated)


def hydrate_risk_along(profile: TemperatureProfile) -> list:
    """Hydrate check at every station of a temperature profile.

    The reason the thermal model earns its place. A line can leave the
    compressor at 120 degF, well clear of any hydrate curve, and still plug
    twelve miles down where it has cooled to ambient and expanded.
    """
    return [hydrate_check(p, t)
            for p, t in zip(profile.pressures_psia, profile.temperatures_r)]
