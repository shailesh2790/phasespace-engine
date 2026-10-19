"""Will this gas-lifted well head? Asheim's two criteria.

`nodal_analysis` returns the intersection of the inflow and tubing curves.
It brackets on the high side, so it does not hand back the unstable lower
root of the J-shaped tubing curve -- but a stable root is not a stable
well. A continuously gas-lifted well can have a perfectly good steady-state
solution and still be unproducible, because the casing and the tubing
exchange gas in a cycle the steady state knows nothing about. The well
heads: the casing charges, dumps through the orifice, the tubing unloads,
the casing recharges, and the cycle repeats on a twenty-minute period.

This matters most on exactly the wells where the answer is least obvious.
Heading is the usual reason a well is moved to intermittent lift, so a
comparison between continuous and intermittent lift that cannot test for it
will systematically recommend continuous -- confidently, and wrongly, on
the low-pressure low-productivity wells where the choice is live.

**Two criteria, and the well is stable if either one holds.**

*F1, the inflow response.* Both the reservoir and the lift gas respond to a
fall in tubing pressure. If the extra gas drawn through the orifice
lightens the column faster than the reservoir can refill it, the response
runs away. High productivity index and a small orifice port stabilise it.

*F2, the pressure-depletion response.* An unstable inflow response can
still be arrested if the surge empties the annulus faster than it empties
the tubing, because the falling casing pressure throttles the gas back.
A shallow injection point and a higher tubing-head pressure stabilise it.

    F1 = rho_gsc * B_g * Q_gsc^2 * J / ((E * A_i)^2 * Q_lsc)

    F2 = (V_t / V_c) * (p_t / (g * D * (rho_fi - rho_gi)))
         * (Q_fi + Q_gi) / (Q_fi * (1 - F1))

Both are dimensionless, which is worth checking rather than trusting: F1
comes out as kg/(s^2 * Pa * m) and Pa is kg/(m * s^2), so the units cancel.
They are evaluated here in SI internally and take field units at the door.

**Provenance, and the part that is not verified.** The criteria are
Asheim, "Criteria for Gas-Lift Stability", JPT 40(11), Nov 1988, SPE
16468-PA. The algebraic form above is transcribed from Durham's
SPE-EuALF 2021 presentation on applying them, which cites that paper; the
original was not available.

That transcription has **not** been reproduced against a worked example.
The presentation gives a field case (productivity index 18 stb/d/psi, 2.5
MMscf/d of lift gas, 14,000 stb/d liquid, 15/64 in orifice near 6000 ft)
and reports F1 = 0.76. It does not give the injection-point pressure,
temperature, Z factor or gas gravity, all of which F1 depends on through
B_g. Reconstructing those gives F1 between 0.81 and 1.33 as the assumed
injection pressure falls from 3500 to 2000 psia -- the right magnitude,
bracketing their figure, and not a confirmation. Treat an F near 1 as
"check this well", not as a decision, and calibrate against a well known to
head before using the absolute value.

**What the criteria assume**, from the same source, because every one of
these is violable in a real well: transient inflow is neglected; the
orifice is isothermal, which needs a small pressure drop across it; the
gas rate through the surface choke is constant while casing pressure
varies; the tubing pressure at the orifice moves with the bottomhole
flowing pressure, which holds only when the orifice is near the
perforations; and the tubing response is gravity-dominated, with friction
and acceleration ignored. It is not a transient solution and says nothing
about multi-pointing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Standard-condition air density, lb/ft3, for gas density from gravity.
AIR_DENSITY_LB_FT3 = 0.076362

LB_FT3_TO_KG_M3 = 16.018463
IN_TO_M = 0.0254
FT_TO_M = 0.3048
PSI_TO_PA = 6894.757
BBL_TO_M3 = 0.158987
FT3_TO_M3 = 0.0283168
SECONDS_PER_DAY = 86400.0
GRAVITY_M_S2 = 9.80665

# PROSPER fixes orifice efficiency at this value, and the presentation
# notes F1 is sensitive to it -- a higher efficiency lowers F1 and so
# reports the well as less stable.
DEFAULT_ORIFICE_EFFICIENCY = 0.9

# Within this of 1.0 the criterion is not deciding anything. Given the
# unverified absolute calibration above, a band is more honest than a
# threshold.
MARGINAL_BAND = 0.25


@dataclass
class StabilityInputs:
    """Everything the two criteria need, in field units."""

    # --- F1
    lift_gas_scf_d: float
    liquid_stb_d: float
    productivity_stb_d_psi: float
    orifice_diameter_in: float
    injection_pressure_psia: float
    injection_temperature_r: float = 620.0
    gas_gravity: float = 0.7
    z_factor: float = 0.9
    orifice_efficiency: float = DEFAULT_ORIFICE_EFFICIENCY

    # --- F2
    depth_to_valve_ft: float = 0.0
    tubing_head_psia: float = 0.0
    tubing_volume_above_valve_bbl: float = 0.0
    annulus_volume_to_valve_bbl: float = 0.0
    liquid_density_lb_ft3: float = 55.0
    liquid_fvf: float = 1.1

    def __post_init__(self):
        if self.lift_gas_scf_d <= 0.0:
            raise ValueError("lift gas rate must be positive, got %r"
                             % (self.lift_gas_scf_d,))
        if self.liquid_stb_d <= 0.0:
            raise ValueError("liquid rate must be positive, got %r"
                             % (self.liquid_stb_d,))
        if self.orifice_diameter_in <= 0.0:
            raise ValueError("orifice diameter must be positive, got %r"
                             % (self.orifice_diameter_in,))
        if self.injection_pressure_psia <= 0.0:
            raise ValueError("injection pressure must be positive, got %r"
                             % (self.injection_pressure_psia,))
        if self.productivity_stb_d_psi <= 0.0:
            raise ValueError("productivity index must be positive, got %r"
                             % (self.productivity_stb_d_psi,))

    @property
    def gas_density_sc_kg_m3(self) -> float:
        return AIR_DENSITY_LB_FT3 * self.gas_gravity * LB_FT3_TO_KG_M3

    @property
    def gas_fvf(self) -> float:
        """B_g: in-situ gas volume per standard volume, at the valve."""
        return ((14.696 / self.injection_pressure_psia)
                * (self.injection_temperature_r / 519.67) * self.z_factor)


@dataclass
class Stability:
    """The verdict, the two numbers behind it, and what it rests on."""

    f1: float
    f2: float
    stable: bool
    marginal: bool
    mechanism: str
    reasons: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)

    def describe(self) -> str:
        if self.stable and not self.marginal:
            return ("stable on %s (F1 = %.2f, F2 = %.2f); either above 1 is "
                    "enough" % (self.mechanism, self.f1, self.f2))
        if self.marginal:
            return ("marginal (F1 = %.2f, F2 = %.2f). Both are near 1, and "
                    "the absolute calibration of these criteria is not "
                    "verified in this engine, so this is a well to watch "
                    "rather than a verdict" % (self.f1, self.f2))
        return ("unstable: F1 = %.2f and F2 = %.2f, both below 1. The "
                "inflow response runs away and the annulus does not empty "
                "fast enough to arrest it, which is heading"
                % (self.f1, self.f2))


ASSUMPTIONS = (
    "transient inflow response is neglected",
    "the orifice is assumed isothermal, which needs a small pressure drop "
    "across it",
    "gas through the surface choke is held constant while casing pressure "
    "varies",
    "tubing pressure at the orifice is assumed to move with the bottomhole "
    "flowing pressure, which holds only if the orifice is near the "
    "perforations",
    "the tubing response is gravity-dominated; friction and acceleration "
    "are ignored",
    "not a transient solution, and silent on multi-pointing",
)


def inflow_response(inputs: StabilityInputs) -> float:
    """F1. Above 1 the reservoir can refill the tubing faster than the
    extra lift gas empties it."""
    rho_gsc = inputs.gas_density_sc_kg_m3
    b_g = inputs.gas_fvf
    q_gsc = inputs.lift_gas_scf_d * FT3_TO_M3 / SECONDS_PER_DAY
    q_lsc = inputs.liquid_stb_d * BBL_TO_M3 / SECONDS_PER_DAY
    productivity = (inputs.productivity_stb_d_psi * BBL_TO_M3
                    / SECONDS_PER_DAY / PSI_TO_PA)
    area = math.pi / 4.0 * (inputs.orifice_diameter_in * IN_TO_M) ** 2
    throat = inputs.orifice_efficiency * area
    return rho_gsc * b_g * q_gsc ** 2 * productivity / (throat ** 2 * q_lsc)


def depletion_response(inputs: StabilityInputs, f1: float) -> float:
    """F2. Above 1 the annulus empties fast enough to throttle the surge.

    Only defined for F1 < 1: the (1 - F1) term is the inflow response it is
    arresting, and when F1 >= 1 there is nothing to arrest -- the well is
    already stable on the first criterion. Returns infinity there rather
    than a negative number that would read as "unstable".
    """
    if f1 >= 1.0:
        return math.inf
    if (inputs.depth_to_valve_ft <= 0.0
            or inputs.annulus_volume_to_valve_bbl <= 0.0
            or inputs.tubing_volume_above_valve_bbl <= 0.0
            or inputs.tubing_head_psia <= 0.0):
        return float("nan")

    b_g = inputs.gas_fvf
    rho_gi = inputs.gas_density_sc_kg_m3 / b_g
    rho_fi = inputs.liquid_density_lb_ft3 * LB_FT3_TO_KG_M3
    if rho_fi <= rho_gi:
        return float("nan")

    # In-situ volumetric rates at the injection point.
    q_fi = (inputs.liquid_stb_d * inputs.liquid_fvf * BBL_TO_M3
            / SECONDS_PER_DAY)
    q_gi = inputs.lift_gas_scf_d * b_g * FT3_TO_M3 / SECONDS_PER_DAY

    volume_ratio = (inputs.tubing_volume_above_valve_bbl
                    / inputs.annulus_volume_to_valve_bbl)
    head = (GRAVITY_M_S2 * inputs.depth_to_valve_ft * FT_TO_M
            * (rho_fi - rho_gi))
    return (volume_ratio * inputs.tubing_head_psia * PSI_TO_PA / head
            * (q_fi + q_gi) / (q_fi * (1.0 - f1)))


def asheim_stability(inputs: StabilityInputs) -> Stability:
    """Both criteria, and the verdict. Stable if either exceeds 1."""
    f1 = inflow_response(inputs)
    f2 = depletion_response(inputs, f1)

    f2_known = f2 == f2                       # NaN check without numpy
    stable = f1 > 1.0 or (f2_known and f2 > 1.0)

    nearest = min(abs(f1 - 1.0),
                  abs(f2 - 1.0) if f2_known and math.isfinite(f2) else math.inf)
    marginal = nearest <= MARGINAL_BAND

    if f1 > 1.0:
        mechanism = "the inflow response (F1)"
    elif f2_known and f2 > 1.0:
        mechanism = "the pressure-depletion response (F2)"
    else:
        mechanism = "neither criterion"

    reasons = []
    if not stable:
        reasons.append(
            "F1 = %.2f: the lift gas lightens the column faster than the "
            "reservoir refills it. A larger productivity index or a smaller "
            "orifice port would raise it." % f1)
        if f2_known:
            reasons.append(
                "F2 = %.2f: the annulus does not empty fast enough to "
                "throttle the surge. A shallower injection point or a "
                "higher tubing-head pressure would raise it." % f2)
        else:
            reasons.append(
                "F2 could not be evaluated: it needs the depth to the "
                "valve, the tubing-head pressure, and the tubing and "
                "annulus volumes. Without it only F1 is deciding, and F1 "
                "alone is the pessimistic half of the test.")
    if marginal:
        reasons.append(
            "this is within %.2f of the threshold, and the absolute "
            "calibration of these criteria has not been verified against a "
            "worked example in this engine -- see the module docstring. "
            "Treat it as a well to instrument, not a decision."
            % MARGINAL_BAND)

    return Stability(f1=f1, f2=f2, stable=stable, marginal=marginal,
                     mechanism=mechanism, reasons=reasons,
                     assumptions=list(ASSUMPTIONS))
