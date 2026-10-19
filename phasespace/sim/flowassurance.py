"""Flow assurance: where a line gets into trouble, and what to do about it.

A temperature profile is a diagnosis. This module is the prescription, and the
distinction matters because the three available answers cost wildly different
amounts of money:

    inhibit     cheap per day, forever, and someone has to keep trucking it
    insulate    expensive once, then free
    heat        expensive once and expensive forever

So the useful output is not "there is a hydrate risk". It is "you need 23 wt%
methanol in the free water, which is 118 gal/day at this water rate" next to
"or U = 0.18, which is about two inches of foam" -- and, when it applies, "no
amount of insulation fixes this, because the cooling is Joule-Thomson and
insulation does not touch it."

That last case is the one worth building for. A line can be driven into the
hydrate region purely by expansion, with perfectly insulated walls, and an
engineer who has only been told "insulate more" will spend the money and watch
it plug anyway.

Everything here reports the envelope it was fitted over. A hydrate number
without its validity range is acted on exactly as confidently as one with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .thermal import (
    ThermalPath,
    ThermalStream,
    hydrate_temperature_r,
    temperature_profile,
)
from .units import r_to_f

# Hammerschmidt's constant in field units: degrees F, molecular weight in
# lb/lbmol, concentration in weight percent of the *free water phase*. That
# last qualifier is the one most often got wrong -- it is not weight percent of
# the total stream, and confusing the two under-doses by whatever the water cut
# happens to be.
HAMMERSCHMIDT_K_F = 2335.0

# Above roughly this concentration Hammerschmidt over-predicts depression and
# Nielsen-Bucklin is the published successor for methanol.
HAMMERSCHMIDT_MAX_WT_PCT = 20.0
NIELSEN_BUCKLIN_MIN_WT_PCT = 20.0
NIELSEN_BUCKLIN_MAX_WT_PCT = 90.0

WATER_MW = 18.015

# Density at 60 degF, lb/gal, for turning a mass rate into something a truck
# can be ordered in.
INHIBITOR_DENSITY_LB_GAL = {
    "methanol": 6.60,
    "meg": 9.29,
    "deg": 9.34,
    "ethanol": 6.58,
}

INHIBITOR_MW = {
    "methanol": 32.04,
    "meg": 62.07,
    "deg": 106.12,
    "ethanol": 46.07,
}


@dataclass
class InhibitorDose:
    """How much inhibitor, and how confident the number is."""

    inhibitor: str
    required_depression_f: float
    weight_percent_in_water: float
    mass_rate_lb_hr: float
    volume_rate_gal_day: float
    correlation: str
    extrapolated: bool
    notes: list = field(default_factory=list)


def hydrate_depression_f(weight_percent: float, inhibitor: str = "methanol") -> float:
    """Hydrate temperature depression from inhibitor concentration.

        dT = K * W / (M * (100 - W))        Hammerschmidt

    W is weight percent *in the free water phase*, not in the total stream.

    Hammerschmidt is a colligative approximation and drifts high above about
    20 wt%. For methanol beyond that, Nielsen-Bucklin is used instead:

        dT = -129.6 * ln(x_water)

    with x_water the mole fraction of water in the aqueous phase. The two agree
    to about a degree where they meet, which is a genuine cross-check: they
    were derived from different data by different routes, so their agreement at
    the handover is evidence rather than bookkeeping.
    """
    key = inhibitor.lower()
    if key not in INHIBITOR_MW:
        raise ValueError("unknown inhibitor %r (known: %s)"
                         % (inhibitor, ", ".join(sorted(INHIBITOR_MW))))
    if not 0.0 <= weight_percent < 100.0:
        raise ValueError("weight percent must be in [0, 100), got %r" % (weight_percent,))
    if weight_percent == 0.0:
        return 0.0

    if key == "methanol" and weight_percent > HAMMERSCHMIDT_MAX_WT_PCT:
        return _nielsen_bucklin_depression_f(weight_percent)

    molecular_weight = INHIBITOR_MW[key]
    return (HAMMERSCHMIDT_K_F * weight_percent
            / (molecular_weight * (100.0 - weight_percent)))


def _nielsen_bucklin_depression_f(weight_percent: float) -> float:
    """Methanol depression at high concentration, from water mole fraction."""
    moles_methanol = weight_percent / INHIBITOR_MW["methanol"]
    moles_water = (100.0 - weight_percent) / WATER_MW
    x_water = moles_water / (moles_water + moles_methanol)
    return -129.6 * math.log(x_water)


def weight_percent_for_depression(depression_f: float,
                                  inhibitor: str = "methanol") -> tuple:
    """Concentration needed to depress the hydrate curve by a given amount.

    Hammerschmidt inverts in closed form:

        W = 100 * M * dT / (K + M * dT)

    Nielsen-Bucklin does not, so methanol above the handover is solved by
    bisection on the forward correlation -- which also keeps the two branches
    consistent by construction, since the inverse calls the same function the
    forward direction uses.

    Returns (weight_percent, correlation_name, extrapolated).
    """
    key = inhibitor.lower()
    if key not in INHIBITOR_MW:
        raise ValueError("unknown inhibitor %r (known: %s)"
                         % (inhibitor, ", ".join(sorted(INHIBITOR_MW))))
    if depression_f < 0.0:
        raise ValueError("depression cannot be negative, got %r" % (depression_f,))
    if depression_f == 0.0:
        return 0.0, "none", False

    molecular_weight = INHIBITOR_MW[key]
    closed_form = (100.0 * molecular_weight * depression_f
                   / (HAMMERSCHMIDT_K_F + molecular_weight * depression_f))

    if key != "methanol" or closed_form <= HAMMERSCHMIDT_MAX_WT_PCT:
        # Glycols have no published high-concentration successor here, so above
        # the Hammerschmidt envelope the answer is flagged rather than swapped
        # for a correlation that does not exist.
        return closed_form, "hammerschmidt", closed_form > HAMMERSCHMIDT_MAX_WT_PCT

    low, high = HAMMERSCHMIDT_MAX_WT_PCT, NIELSEN_BUCKLIN_MAX_WT_PCT
    if _nielsen_bucklin_depression_f(high) < depression_f:
        return high, "nielsen-bucklin", True
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _nielsen_bucklin_depression_f(mid) < depression_f:
            low = mid
        else:
            high = mid
    weight_percent = 0.5 * (low + high)
    return weight_percent, "nielsen-bucklin", False


def inhibitor_requirement(
    pressure_psia: float,
    temperature_r: float,
    water_rate_lb_hr: float,
    inhibitor: str = "methanol",
    safety_margin_f: float = 5.0,
) -> InhibitorDose:
    """Inhibitor needed to keep a stream out of the hydrate region.

    ``water_rate_lb_hr`` is the *free water*, which is what the inhibitor has
    to dose. Using total liquid instead over-doses by the condensate fraction;
    using total stream over-doses by roughly the water cut, and both errors are
    expensive in opposite directions.

    The returned rate accounts for none of the inhibitor lost to the vapour and
    hydrocarbon-liquid phases. For methanol that loss is real and can approach
    the aqueous-phase requirement at high gas rates, so the figure here is a
    floor -- it is noted on every dose rather than buried in documentation.
    """
    if water_rate_lb_hr < 0.0:
        raise ValueError("water rate cannot be negative, got %r" % (water_rate_lb_hr,))

    hydrate_r, hydrate_extrapolated = hydrate_temperature_r(pressure_psia)
    shortfall = (hydrate_r + safety_margin_f) - temperature_r

    notes = []
    if hydrate_extrapolated:
        notes.append(
            "Pressure is outside the range Hammerschmidt's hydrate curve was "
            "fitted over, so the target temperature is an extrapolation.")

    if shortfall <= 0.0:
        return InhibitorDose(
            inhibitor=inhibitor.lower(), required_depression_f=0.0,
            weight_percent_in_water=0.0, mass_rate_lb_hr=0.0,
            volume_rate_gal_day=0.0, correlation="none",
            extrapolated=hydrate_extrapolated,
            notes=notes + ["Stream is already clear of the hydrate curve with "
                           "%.1f degF to spare." % (temperature_r - hydrate_r)])

    weight_percent, correlation, extrapolated = weight_percent_for_depression(
        shortfall, inhibitor)

    # W = mass inhibitor / (mass inhibitor + mass water), so the inhibitor rate
    # that reaches concentration W in a given water rate is W/(100-W) times it.
    ratio = weight_percent / (100.0 - weight_percent)
    mass_rate = ratio * water_rate_lb_hr
    density = INHIBITOR_DENSITY_LB_GAL[inhibitor.lower()]
    volume_rate = mass_rate * 24.0 / density

    notes.append(
        "Aqueous-phase requirement only: inhibitor lost to the vapour and "
        "hydrocarbon liquid is not included, so treat this as a floor.")
    if extrapolated:
        notes.append(
            "%.0f wt%% is outside the correlation's fitted range -- verify "
            "against a hydrate equation of state before dosing."
            % weight_percent)

    return InhibitorDose(
        inhibitor=inhibitor.lower(),
        required_depression_f=shortfall,
        weight_percent_in_water=weight_percent,
        mass_rate_lb_hr=mass_rate,
        volume_rate_gal_day=volume_rate,
        correlation=correlation,
        extrapolated=extrapolated or hydrate_extrapolated,
        notes=notes,
    )


@dataclass
class InsulationRequirement:
    """The U-value that keeps a line clear, when one exists."""

    achievable: bool
    required_u: float
    margin_at_required_u_f: float
    limiting_station_ft: float
    notes: list = field(default_factory=list)


def required_insulation_u(
    inlet_temperature_r: float,
    path: ThermalPath,
    stream: ThermalStream,
    length_ft: float,
    pressures_psia: list,
    segments: int,
    joule_thomson_f_per_psi: float,
    safety_margin_f: float = 5.0,
    max_u: float = 20.0,
) -> InsulationRequirement:
    """Largest U that still keeps the whole line above the hydrate curve.

    Lower U means less heat lost means a warmer line, so the worst margin along
    the line rises monotonically as U falls -- which is what makes bisection
    valid here rather than merely convenient.

    The important branch is the one that returns ``achievable=False``. If the
    line is still inside the hydrate region at U = 0, the cooling is
    Joule-Thomson, insulation cannot touch it, and the honest answer is to stop
    quoting insulation and price inhibitor or heating instead. Software that
    returns "insulate to U = 0.001" for that case has told an engineer to spend
    money on something that cannot work.
    """
    if max_u <= 0.0:
        raise ValueError("max_u must be positive, got %r" % (max_u,))

    def worst_margin(u_value: float) -> tuple:
        candidate = ThermalPath(
            diameter_ft=path.diameter_ft, overall_u=u_value,
            ambient_r=path.ambient_r,
            ambient_gradient_r_per_ft=path.ambient_gradient_r_per_ft,
            inclination_deg=path.inclination_deg)
        profile = temperature_profile(
            inlet_temperature_r, candidate, stream, length_ft,
            pressures_psia=pressures_psia, segments=segments,
            joule_thomson_f_per_psi=joule_thomson_f_per_psi)
        worst, where = math.inf, 0.0
        for x, p, t in zip(profile.distances_ft, profile.pressures_psia,
                           profile.temperatures_r):
            margin = t - hydrate_temperature_r(p)[0] - safety_margin_f
            if margin < worst:
                worst, where = margin, x
        return worst, where

    perfect_margin, perfect_where = worst_margin(0.0)
    if perfect_margin < 0.0:
        return InsulationRequirement(
            achievable=False, required_u=0.0,
            margin_at_required_u_f=perfect_margin,
            limiting_station_ft=perfect_where,
            notes=["Even perfectly insulated the line is %.1f degF inside the "
                   "hydrate region at %.0f ft. The cooling is Joule-Thomson "
                   "expansion, which insulation does not affect -- inhibition, "
                   "heating or a different pressure profile are the only "
                   "remedies." % (-perfect_margin, perfect_where)])

    if worst_margin(max_u)[0] >= 0.0:
        return InsulationRequirement(
            achievable=True, required_u=max_u,
            margin_at_required_u_f=worst_margin(max_u)[0],
            limiting_station_ft=worst_margin(max_u)[1],
            notes=["The line stays clear even bare (U = %.1f), so no insulation "
                   "is required for hydrate control." % max_u])

    low, high = 0.0, max_u          # low is safe, high is not
    for _ in range(60):
        mid = 0.5 * (low + high)
        if worst_margin(mid)[0] >= 0.0:
            low = mid
        else:
            high = mid

    margin, where = worst_margin(low)
    return InsulationRequirement(
        achievable=True, required_u=low, margin_at_required_u_f=margin,
        limiting_station_ft=where,
        notes=["Worst margin occurs at %.0f ft (%.1f miles)."
               % (where, where / 5280.0)])


@dataclass
class WaxRisk:
    """Where the stream drops below its wax appearance temperature."""

    forms: bool
    first_station_ft: float
    coldest_f: float
    appearance_f: float


def wax_risk_along(profile, wax_appearance_r: float) -> WaxRisk:
    """Wax deposition window along a profile.

    The wax appearance temperature is a property of the specific crude and
    comes from a lab measurement, so it is an input here rather than a
    correlation. There is no general correlation worth trusting: WAT depends on
    the heavy-end distribution, which is exactly what a gravity or a GOR does
    not tell you.
    """
    first = None
    coldest = math.inf
    for x, t in zip(profile.distances_ft, profile.temperatures_r):
        coldest = min(coldest, t)
        if t < wax_appearance_r and first is None:
            first = x
    return WaxRisk(
        forms=first is not None,
        first_station_ft=first if first is not None else math.inf,
        coldest_f=r_to_f(coldest),
        appearance_f=r_to_f(wax_appearance_r),
    )


@dataclass
class FlowAssuranceReport:
    """Everything that threatens a line, in one pass."""

    hydrate_forms: bool
    hydrate_first_station_ft: float
    worst_hydrate_margin_f: float
    worst_margin_station_ft: float
    wax: WaxRisk
    outlet_f: float
    notes: list = field(default_factory=list)


def flow_assurance_report(profile, wax_appearance_r: float = None,
                          safety_margin_f: float = 0.0) -> FlowAssuranceReport:
    """Screen a temperature profile for everything that can shut a line in."""
    first = None
    worst, worst_where = math.inf, 0.0
    for x, p, t in zip(profile.distances_ft, profile.pressures_psia,
                       profile.temperatures_r):
        if p <= 0.0:
            continue
        margin = t - hydrate_temperature_r(p)[0] - safety_margin_f
        if margin < worst:
            worst, worst_where = margin, x
        if margin < 0.0 and first is None:
            first = x

    notes = []
    if first is not None:
        notes.append(
            "Hydrate region entered at %.0f ft (%.1f miles); worst margin "
            "%.1f degF at %.0f ft." % (first, first / 5280.0, worst, worst_where))
    if abs(profile.joule_thomson_delta_r) > abs(profile.ambient_delta_r):
        notes.append(
            "Joule-Thomson cooling (%.1f degF) exceeds ambient heat loss "
            "(%.1f degF), so insulation alone cannot fix this line."
            % (profile.joule_thomson_delta_r, profile.ambient_delta_r))

    wax = (wax_risk_along(profile, wax_appearance_r)
           if wax_appearance_r is not None
           else WaxRisk(False, math.inf, r_to_f(min(profile.temperatures_r)), 0.0))
    if wax.forms:
        notes.append("Below the wax appearance temperature from %.0f ft."
                     % wax.first_station_ft)

    return FlowAssuranceReport(
        hydrate_forms=first is not None,
        hydrate_first_station_ft=first if first is not None else math.inf,
        worst_hydrate_margin_f=worst if worst is not math.inf else 0.0,
        worst_margin_station_ft=worst_where,
        wax=wax,
        outlet_f=profile.outlet_f,
        notes=notes,
    )
