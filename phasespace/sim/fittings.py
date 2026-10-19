"""Minor losses: bends, elbows, tees, valves, reducers, entrances and exits.

The engine could size a straight run but had no way to represent the thing
that actually fills a station or a river crossing -- the fittings. On a
50 mi transmission line they are a rounding error; on a 2000 ft station
header with fourteen elbows and a tee they can be a third of the total
pressure drop, and leaving them out makes an undersized line look feasible.

Method: Crane Technical Paper 410's resistance coefficients, which express
each fitting as K = n * f_T, where f_T is the *fully turbulent* friction
factor for that pipe size. Converting a K to an equivalent length of pipe,

    dP_fitting = K * rho * v^2 / 2
    dP_pipe    = f * (L/D) * rho * v^2 / 2      =>   L_eq = K * D / f

and substituting K = n * f_T with f = f_T collapses to

    L_eq = n * D

which is the familiar "a 90-degree elbow is worth 30 diameters" rule. That
collapse is why the K values below are stored as the dimensionless
multiplier n rather than as K itself: n is a property of the fitting's
shape alone, while K depends on the pipe size it is installed in.

Equivalent length is the right currency here because every flow correlation
in `components.py` -- Weymouth, Panhandle, AGA, Darcy-Weisbach -- takes a
length. Adding fitting length lets minor losses enter all four without
touching any of them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .components import fully_turbulent_friction_factor

# Crane TP-410 resistance coefficients, as the multiplier n in K = n * f_T.
# Equivalent length is then simply n pipe diameters.
ELBOW_90_STANDARD = "elbow_90_standard"
ELBOW_90_LONG_RADIUS = "elbow_90_long_radius"
ELBOW_45_STANDARD = "elbow_45_standard"
BEND_180_RETURN = "bend_180_return"
TEE_THROUGH_RUN = "tee_through_run"
TEE_THROUGH_BRANCH = "tee_through_branch"
GATE_VALVE = "gate_valve"
BALL_VALVE = "ball_valve"
PLUG_VALVE = "plug_valve"
GLOBE_VALVE = "globe_valve"
BUTTERFLY_VALVE = "butterfly_valve"
SWING_CHECK_VALVE = "swing_check_valve"
LIFT_CHECK_VALVE = "lift_check_valve"

_K_MULTIPLIERS = {
    ELBOW_90_STANDARD: 30.0,
    ELBOW_90_LONG_RADIUS: 20.0,
    ELBOW_45_STANDARD: 16.0,
    BEND_180_RETURN: 50.0,
    TEE_THROUGH_RUN: 20.0,
    TEE_THROUGH_BRANCH: 60.0,
    GATE_VALVE: 8.0,
    BALL_VALVE: 3.0,
    PLUG_VALVE: 18.0,
    GLOBE_VALVE: 340.0,
    BUTTERFLY_VALVE: 45.0,
    SWING_CHECK_VALVE: 100.0,
    LIFT_CHECK_VALVE: 600.0,
}

# Entrances and exits are geometric, not size-scaled: their K is a plain
# number rather than a multiple of f_T, so they need the full K * D / f
# conversion instead of collapsing to n diameters.
ENTRANCE_SHARP = "entrance_sharp"
ENTRANCE_ROUNDED = "entrance_rounded"
ENTRANCE_PROJECTING = "entrance_projecting"
EXIT = "exit"

_ABSOLUTE_K = {
    ENTRANCE_SHARP: 0.50,
    ENTRANCE_ROUNDED: 0.04,
    ENTRANCE_PROJECTING: 0.78,
    EXIT: 1.00,
}

FITTING_TYPES = tuple(sorted(list(_K_MULTIPLIERS) + list(_ABSOLUTE_K)))

# Human-readable labels, for UI dropdowns and printed reports.
FITTING_LABELS = {
    ELBOW_90_STANDARD: "90 deg elbow (standard)",
    ELBOW_90_LONG_RADIUS: "90 deg elbow (long radius)",
    ELBOW_45_STANDARD: "45 deg elbow",
    BEND_180_RETURN: "180 deg return bend",
    TEE_THROUGH_RUN: "Tee (flow through run)",
    TEE_THROUGH_BRANCH: "Tee (flow through branch)",
    GATE_VALVE: "Gate valve (full open)",
    BALL_VALVE: "Ball valve (full open)",
    PLUG_VALVE: "Plug valve (full open)",
    GLOBE_VALVE: "Globe valve (full open)",
    BUTTERFLY_VALVE: "Butterfly valve (full open)",
    SWING_CHECK_VALVE: "Swing check valve",
    LIFT_CHECK_VALVE: "Lift check valve",
    ENTRANCE_SHARP: "Pipe entrance (sharp edged)",
    ENTRANCE_ROUNDED: "Pipe entrance (rounded)",
    ENTRANCE_PROJECTING: "Pipe entrance (projecting)",
    EXIT: "Pipe exit",
}


def resistance_coefficient(kind: str, diameter_in: float, roughness_in: float = 0.0018) -> float:
    """Crane K for one fitting of this type in this pipe size.

    K is size-dependent for everything expressed as n * f_T, because f_T
    falls as diameter rises: the same elbow is worth K = 0.45 in a 6 in line
    and K = 0.36 in a 24 in one.
    """
    if kind in _ABSOLUTE_K:
        return _ABSOLUTE_K[kind]
    if kind not in _K_MULTIPLIERS:
        raise ValueError("unknown fitting %r (expected one of %s)"
                         % (kind, ", ".join(FITTING_TYPES)))
    f_t = fully_turbulent_friction_factor(diameter_in, roughness_in)
    return _K_MULTIPLIERS[kind] * f_t


def equivalent_length_ft(kind: str, diameter_in: float, roughness_in: float = 0.0018) -> float:
    """Length of straight pipe of the same diameter that loses as much pressure."""
    k = resistance_coefficient(kind, diameter_in, roughness_in)
    f_t = fully_turbulent_friction_factor(diameter_in, roughness_in)
    return k * (diameter_in / 12.0) / f_t


@dataclass
class FittingCount:
    """A number of identical fittings on one pipe segment."""

    kind: str
    count: int = 1

    def __post_init__(self):
        if self.kind not in FITTING_TYPES:
            raise ValueError("unknown fitting %r (expected one of %s)"
                             % (self.kind, ", ".join(FITTING_TYPES)))
        if self.count < 0:
            raise ValueError("fitting count cannot be negative, got %r" % (self.count,))

    @property
    def label(self) -> str:
        return FITTING_LABELS.get(self.kind, self.kind)

    def equivalent_length_ft(self, diameter_in: float, roughness_in: float = 0.0018) -> float:
        return self.count * equivalent_length_ft(self.kind, diameter_in, roughness_in)


def total_equivalent_length_ft(fittings, diameter_in: float,
                               roughness_in: float = 0.0018) -> float:
    """Summed equivalent length for a list of FittingCount.

    Note this scales with diameter: the *same* fitting list on a bigger pipe
    contributes more equivalent feet, even though it costs less pressure.
    That is not a contradiction -- a 24 in elbow really is a longer flow path
    than a 6 in one, while the velocity through it is far lower. Sizing code
    must therefore recompute this per candidate diameter rather than
    computing it once, which is why `sizing.py` takes fittings and diameter
    together rather than a precomputed length.
    """
    if not fittings:
        return 0.0
    return sum(f.equivalent_length_ft(diameter_in, roughness_in) for f in fittings)


def fitting_pressure_drop_psi(kind: str, diameter_in: float, velocity_ft_s: float,
                              density_lb_ft3: float, roughness_in: float = 0.0018) -> float:
    """Direct K-method pressure drop, dP = K * rho * v^2 / (2 * gc), in psi.

    Provided alongside the equivalent-length route as a cross-check: the two
    must agree, and a test asserts they do. Equivalent length is what the
    network solver consumes; this is what an engineer would compute by hand
    to verify it.
    """
    k = resistance_coefficient(kind, diameter_in, roughness_in)
    gc = 32.174
    return k * density_lb_ft3 * velocity_ft_s ** 2 / (2.0 * gc) / 144.0
