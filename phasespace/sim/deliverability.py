"""What a multi-rate test tells you about the reservoir, and what it cannot.

A flow-after-flow test is three or four choke settings on one day, each
with the bottomhole pressure the well settled at, against the static
pressure from a build-up. It is the oldest well test there is and it
answers the two questions an operator actually asks: how much can this
well make wide open, and is it being choked by the rock or by turbulence
at the perforations.

**Two analyses, because they disagree in a useful way.**

*Rawlins and Schellhardt (1935), the back-pressure form.* `q = C(m_r -
m_wf)^n`, with `n` between 0.5 and 1. Entirely empirical, fitted in log
space, and the exponent is itself a diagnostic: `n` near 1 is laminar
flow, `n` near 0.5 is fully turbulent.

*Houpeurt, the laminar-inertial-turbulent form.* `m_r - m_wf = a q + b
q^2`, which comes out of Forchheimer rather than curve-fitting. The two
terms separate: `a` is Darcy flow through the rock, `b` is the inertial
loss near the wellbore. That separation is the thing worth having,
because the two have different remedies -- `a` wants a stimulation, `b`
wants bigger perforations or tubing.

Both give an absolute open flow potential, and **they should agree**.
Where they do not, the test is telling you something: too few rates, a
well that had not stabilised, or a static pressure paired from the wrong
campaign.

**Pseudopressure, not pressure squared.** The `p^2` form every textbook
starts with assumes `mu z` is constant, which holds below roughly 2000
psia and fails above it. Half the wells in the archive this was built
for flow at 3600-3950 psia, where `p^2` is simply the wrong variable.
`pseudopressure` integrates `2p/(mu z)` with the engine's own Z factor
and viscosity, so the analysis is correct across the range instead of
across the easy part of it.

**What this cannot do, stated because the temptation is strong.**
Permeability and skin are *not separable* from a steady multi-rate test.
They enter `a` together:

    a = (1422 mu z T / k h) * [ln(0.472 re / rw) + s]

One equation, two unknowns, and no amount of fitting changes that. A
transient -- a pressure build-up sampled over time -- is what separates
them, and the archive these tests came from records only the start and
end pressures of its build-ups. `DeliverabilityFit.separable` says so
rather than returning a skin that is really an assumption about kh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Rawlins and Schellhardt's exponent is bounded by the physics: 1 is
# laminar, 0.5 fully turbulent. A fit outside that is a bad test, not a
# discovery.
MIN_EXPONENT = 0.5
MAX_EXPONENT = 1.0

# Below this relative spread the rates are one operating point repeated,
# and the curvature that separates the laminar term from the turbulent
# one is not in the data.
MIN_RATE_SPREAD = 0.20

# Two points fit two parameters exactly and say nothing about whether the
# model is right. Three is the minimum that leaves a residual.
MIN_RATES = 3

# Atmospheric, where absolute open flow is defined.
ATMOSPHERIC_PSIA = 14.696

# A drawdown this small, as a fraction of the static pressure, is close
# enough to gauge resolution that the exponent is being fitted to
# measurement noise. Seen on real tests: a well drawing down 15 to 26 psi
# from 1784 psia returned an exponent of 1.06, which is outside the
# physics but is three readings a couple of psi apart, not a bad test.
MIN_DRAWDOWN_FRACTION = 0.03


def pseudopressure(pressure_psia: float, gas, steps: int = 64) -> float:
    """m(p) = 2 integral p / (mu z) dp, from atmospheric.

    The right variable for gas deliverability. Integrated numerically
    with the engine's own Z factor and viscosity rather than a chart, so
    the analysis uses the same physics as everything else in the engine
    and moves with it.
    """
    if pressure_psia <= ATMOSPHERIC_PSIA:
        return 0.0
    grid = np.linspace(ATMOSPHERIC_PSIA, pressure_psia, steps)
    values = np.empty(steps)
    for index, p in enumerate(grid):
        z = max(gas.z_at(float(p)), 1e-9)
        mu = max(_viscosity(gas, float(p)), 1e-9)
        values[index] = 2.0 * p / (mu * z)
    return float(np.trapezoid(values, grid)) if hasattr(np, "trapezoid") \
        else float(np.trapz(values, grid))


def _viscosity(gas, pressure_psia: float) -> float:
    """Gas viscosity in cp, from the engine where it has one."""
    for name in ("viscosity_cp", "viscosity"):
        method = getattr(gas, name, None)
        if callable(method):
            try:
                return float(method(pressure_psia))
            except TypeError:
                return float(method())
    # Lee-Gonzalez-Eakin, the standard correlation, as a fallback so this
    # works against a fluid object that does not expose viscosity.
    temperature_r = getattr(gas, "temperature_r", 560.0)
    molecular_weight = getattr(gas, "gravity", 0.65) * 28.9625
    z = max(gas.z_at(pressure_psia), 1e-9)
    density = 1.4935e-3 * pressure_psia * molecular_weight / (z * temperature_r)
    k = ((9.4 + 0.02 * molecular_weight) * temperature_r ** 1.5
         / (209.0 + 19.0 * molecular_weight + temperature_r))
    x = 3.5 + 986.0 / temperature_r + 0.01 * molecular_weight
    y = 2.4 - 0.2 * x
    return 1e-4 * k * math.exp(x * density ** y)


@dataclass
class RatePoint:
    """One stabilised flow point."""

    rate_scf_d: float
    bottomhole_psia: float
    label: str = ""


@dataclass
class DeliverabilityFit:
    """Both analyses of one test, and what the test can support."""

    static_psia: float
    points: list
    # Rawlins and Schellhardt
    coefficient: float = 0.0           # C
    exponent: float = 0.0              # n
    aof_backpressure: float = 0.0
    # Houpeurt
    laminar: float = 0.0               # a
    turbulent: float = 0.0             # b
    aof_lit: float = 0.0
    turbulence_fraction: float = 0.0   # of the drawdown, at the top rate
    usable: bool = True
    notes: list = field(default_factory=list)

    @property
    def separable(self) -> bool:
        """Can permeability and skin be told apart? From this data, no.

        They enter the laminar coefficient as one group. Saying so is the
        difference between a result and a number with an assumption
        hidden inside it.
        """
        return False

    @property
    def agreement(self) -> float:
        """How far apart the two analyses put the open flow."""
        if self.aof_lit <= 0.0:
            return float("inf")
        return abs(self.aof_backpressure - self.aof_lit) / self.aof_lit

    @property
    def flow_regime(self) -> str:
        if self.exponent >= 0.92:
            return "laminar"
        if self.exponent <= 0.6:
            return "turbulence-dominated"
        return "mixed laminar and turbulent"

    def describe(self) -> str:
        if not self.usable:
            return "; ".join(self.notes)
        return ("AOF %.1f MMscf/d (back-pressure) against %.1f (LIT), "
                "n = %.2f -- %s, with %.0f%% of the drawdown at the top "
                "rate lost to turbulence"
                % (self.aof_backpressure / 1e6, self.aof_lit / 1e6,
                   self.exponent, self.flow_regime,
                   self.turbulence_fraction * 100.0))


def fit(points: list, static_psia: float, gas,
        use_pseudopressure: bool = True) -> DeliverabilityFit:
    """Fit both deliverability forms to a multi-rate test."""
    result = DeliverabilityFit(static_psia=static_psia, points=list(points))

    usable = [p for p in points
              if p.rate_scf_d > 0.0 and 0.0 < p.bottomhole_psia < static_psia]
    rejected = len(points) - len(usable)
    if rejected:
        result.notes.append(
            "%d of %d points have a flowing pressure at or above the static "
            "pressure of %.0f psia. A well cannot flow against itself, so "
            "either the build-up is from a different campaign or the test "
            "was not stabilised."
            % (rejected, len(points), static_psia))

    if len(usable) < MIN_RATES:
        result.usable = False
        result.notes.append(
            "%d usable rate(s); %d are needed. Two points fit two "
            "parameters exactly and leave no residual to judge the fit by."
            % (len(usable), MIN_RATES))
        return result

    rates = np.array([p.rate_scf_d for p in usable])
    spread = (rates.max() - rates.min()) / rates.max()
    if spread < MIN_RATE_SPREAD:
        result.usable = False
        result.notes.append(
            "the rates span only %.0f%% of the highest; below %.0f%% the "
            "curvature that separates laminar from turbulent flow is not "
            "in the data, and the split between them would be arbitrary."
            % (spread * 100.0, MIN_RATE_SPREAD * 100.0))
        return result

    if use_pseudopressure:
        m_static = pseudopressure(static_psia, gas)
        drop = np.array([m_static - pseudopressure(p.bottomhole_psia, gas)
                         for p in usable])
        m_aof = m_static
    else:
        m_static = static_psia ** 2
        drop = np.array([m_static - p.bottomhole_psia ** 2 for p in usable])
        m_aof = m_static

    if np.any(drop <= 0.0):
        result.usable = False
        result.notes.append("a flowing point sits at or above the static "
                            "pressure after conversion; the test cannot be "
                            "analysed")
        return result

    # Rawlins and Schellhardt, fitted in log space because that is where
    # the relation is linear.
    slope, intercept = np.polyfit(np.log10(drop), np.log10(rates), 1)
    result.exponent = float(slope)
    result.coefficient = float(10.0 ** intercept)
    # How hard the well was actually drawn down, which decides whether
    # the exponent means anything.
    pressures = np.array([p.bottomhole_psia for p in usable])
    drawdown_fraction = float((static_psia - pressures.min()) / static_psia)

    if not MIN_EXPONENT - 0.05 <= result.exponent <= MAX_EXPONENT + 0.05:
        if drawdown_fraction < MIN_DRAWDOWN_FRACTION:
            result.notes.append(
                "the back-pressure exponent came out at %.2f, outside the "
                "%.1f to %.1f the physics allows -- but the well was only "
                "drawn down %.0f psi from %.0f psia, which is %.1f%% and "
                "close to what a gauge resolves. The exponent is fitted to "
                "noise here, not to a bad test. Widen the chokes before "
                "reading anything into it."
                % (result.exponent, MIN_EXPONENT, MAX_EXPONENT,
                   static_psia - pressures.min(), static_psia,
                   drawdown_fraction * 100.0))
        else:
            result.notes.append(
                "the back-pressure exponent came out at %.2f, outside the "
                "%.1f to %.1f the physics allows, on a drawdown of %.0f psi "
                "that is large enough to measure. The points are not on one "
                "deliverability curve -- most often a rate that had not "
                "stabilised."
                % (result.exponent, MIN_EXPONENT, MAX_EXPONENT,
                   static_psia - pressures.min()))
    result.aof_backpressure = float(result.coefficient * m_aof ** result.exponent)

    # Houpeurt: drop/q is linear in q, so a and b come from one fit.
    turbulent, laminar = np.polyfit(rates, drop / rates, 1)
    result.laminar = float(laminar)
    result.turbulent = float(turbulent)
    if result.turbulent < 0.0:
        result.notes.append(
            "the inertial coefficient came out negative, which has no "
            "physical meaning: the drawdown per unit rate falls as rate "
            "rises, the opposite of turbulence.%s Treat the laminar "
            "coefficient as the whole answer and the split as unmeasured."
            % (" On a drawdown of only %.1f%% that is measurement noise "
               "rather than a finding about the well."
               % (drawdown_fraction * 100.0)
               if drawdown_fraction < MIN_DRAWDOWN_FRACTION else ""))
        result.turbulent = 0.0

    # AOF from the quadratic, taking the positive root.
    a, b = result.laminar, result.turbulent
    if b > 0.0:
        result.aof_lit = float((-a + math.sqrt(a * a + 4.0 * b * m_aof))
                               / (2.0 * b))
    elif a > 0.0:
        result.aof_lit = float(m_aof / a)

    top = float(rates.max())
    laminar_part = a * top
    turbulent_part = b * top * top
    total = laminar_part + turbulent_part
    result.turbulence_fraction = (turbulent_part / total) if total > 0 else 0.0

    if result.agreement > 0.25:
        result.notes.append(
            "the two analyses disagree about open flow by %.0f%%. With "
            "three points that is within what the fits can differ by, but "
            "it is worth another rate before quoting either."
            % (result.agreement * 100.0))

    if result.turbulence_fraction > 0.3:
        result.notes.append(
            "%.0f%% of the drawdown at the top rate is inertial loss near "
            "the wellbore rather than flow through the rock. That is a "
            "completion problem -- perforations or tubing -- and a "
            "stimulation would not fix it."
            % (result.turbulence_fraction * 100.0))

    result.notes.append(
        "permeability and skin are not separable from a steady multi-rate "
        "test: they enter the laminar coefficient as one group. Separating "
        "them needs a pressure transient sampled over time, which a "
        "build-up's start and end pressures do not provide.")
    return result
