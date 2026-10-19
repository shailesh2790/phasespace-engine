"""Where each correlation's data came from, and whether you are inside it.

Every correlation in this engine is a curve fitted to somebody's experiment.
Beggs and Brill ran air and water through 1 in and 1.5 in acrylic pipe at 35 to
95 psia. Hagedorn and Brown instrumented one 1,500 ft well with tubing up to
1.5 in. Those are the conditions the numbers are *known* to reproduce. Outside
them the correlation still returns a number, confidently, with no indication
that it is now extrapolating several-fold beyond anything it was ever checked
against.

That is the failure mode this module exists to remove. It does not refuse to
compute -- an engineer running a 5 in deviated well with 40 cp oil needs an
answer, and every commercial package will give them one. It says which
assumptions are being stretched, by how much, and which direction the error
usually goes, so the answer can be weighted rather than believed.

Three deliberate choices:

**Ranges are the experimental basis, not a licence boundary.** A correlation
does not stop working one foot outside the pipe it was fitted in. What changes
is the evidence, so being outside is reported as reduced support rather than as
an error.

**Every range carries its source and its consequence.** "Outside 1.0-1.5 in"
is not actionable. "Fitted on 1.0-1.5 in pipe; on larger tubing the holdup
correlation tends to over-predict, so bottomhole pressure comes out high" tells
an engineer which way to lean.

**Severity is graded by how far outside, not merely whether.** A 1.6 in tubing
against a 1.5 in basis is a footnote; a 7 in casing is a different flow regime.
Treating those the same trains people to ignore the warning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# How far outside a fitted range counts as a note rather than a warning. A
# 10% overshoot on a diameter is interpolation in all but name.
MARGINAL_FRACTION = 0.10


@dataclass
class Envelope:
    """One condition a correlation was fitted over."""

    quantity: str
    units: str
    low: float
    high: float
    basis: str
    consequence: str

    def check(self, value: float) -> "EnvelopeCheck":
        if value < self.low:
            span = self.low - value
            reference = abs(self.low) if self.low else 1.0
        elif value > self.high:
            span = value - self.high
            reference = abs(self.high) if self.high else 1.0
        else:
            span = 0.0
            reference = 1.0
        return EnvelopeCheck(envelope=self, value=value,
                             excess_fraction=span / reference)


@dataclass
class EnvelopeCheck:
    envelope: Envelope
    value: float
    excess_fraction: float

    @property
    def inside(self) -> bool:
        return self.excess_fraction <= 0.0

    @property
    def severity(self) -> str:
        if self.inside:
            return "inside"
        return "marginal" if self.excess_fraction <= MARGINAL_FRACTION else "outside"

    def describe(self) -> str:
        if self.inside:
            return "%s %.4g %s is within the fitted %.4g-%.4g %s." % (
                self.envelope.quantity, self.value, self.envelope.units,
                self.envelope.low, self.envelope.high, self.envelope.units)
        direction = "below" if self.value < self.envelope.low else "above"
        return (
            "%s %.4g %s is %s the fitted range %.4g-%.4g %s (%s). %s"
            % (self.envelope.quantity, self.value, self.envelope.units, direction,
               self.envelope.low, self.envelope.high, self.envelope.units,
               self.envelope.basis, self.envelope.consequence))


@dataclass
class Applicability:
    """The verdict for one correlation against one set of conditions."""

    correlation: str
    reference: str
    checks: list = field(default_factory=list)

    @property
    def violations(self) -> list:
        return [c for c in self.checks if not c.inside]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def severity(self) -> str:
        """The worst thing wrong, which is what a badge should show."""
        if self.ok:
            return "inside"
        return ("outside" if any(c.severity == "outside" for c in self.violations)
                else "marginal")

    @property
    def worst_excess(self) -> float:
        return max((c.excess_fraction for c in self.checks), default=0.0)

    def messages(self) -> list:
        return [c.describe() for c in self.violations]

    def as_dict(self) -> dict:
        return {
            "correlation": self.correlation,
            "reference": self.reference,
            "severity": self.severity,
            "ok": self.ok,
            "violations": [
                {
                    "quantity": c.envelope.quantity,
                    "value": c.value,
                    "low": c.envelope.low,
                    "high": c.envelope.high,
                    "units": c.envelope.units,
                    "severity": c.severity,
                    "message": c.describe(),
                }
                for c in self.violations
            ],
        }


# --------------------------------------------------------------- catalogue --

# Beggs & Brill (1973), "A Study of Two-Phase Flow in Inclined Pipes", JPT.
# The experimental facility was 1 in and 1.5 in acrylic pipe, air and water,
# inclinable through the full range. The angle coverage is the correlation's
# real strength and the reason it is the default here; the diameter and
# viscosity coverage is its weakness.
BEGGS_BRILL = (
    Envelope("Pipe diameter", "in", 1.0, 1.5,
             "Beggs & Brill 1973 used 1 in and 1.5 in acrylic pipe",
             "On larger tubing the holdup correlation tends to over-predict, "
             "which raises the computed hydrostatic head and so overstates "
             "bottomhole pressure."),
    Envelope("Inclination", "deg from horizontal", -90.0, 90.0,
             "the test loop inclined through the full range",
             "Outside this is not a physical angle."),
    Envelope("Pressure", "psia", 35.0, 95.0,
             "the air-water loop ran at 35-95 psia",
             "At field pressures the gas density is orders of magnitude "
             "higher than anything tested, which is the single largest "
             "extrapolation in normal use of this correlation."),
    Envelope("Liquid viscosity", "cp", 0.5, 2.0,
             "air and water only, so roughly 1 cp",
             "Viscous oil holds up more liquid than water does; the "
             "correlation has no data for it and under-predicts holdup."),
)

# Hagedorn & Brown (1965), from a single instrumented 1,500 ft well. Vertical
# only, which is why applying it to a deviated well is the mistake it is: the
# correlation has no angle term at all, so a 60-degree section is computed as
# though it were vertical.
HAGEDORN_BROWN = (
    Envelope("Pipe diameter", "in", 1.0, 1.5,
             "Hagedorn & Brown 1965 used 1, 1.25 and 1.5 in tubing",
             "Larger tubing is an extrapolation; the holdup correlation was "
             "never checked there."),
    Envelope("Inclination", "deg from horizontal", 88.0, 90.0,
             "a vertical 1,500 ft experimental well",
             "The correlation carries no inclination term whatsoever, so a "
             "deviated section is computed as if vertical -- which overstates "
             "the hydrostatic column by roughly the cosine of the deviation."),
    Envelope("Depth", "ft", 0.0, 1500.0,
             "the experimental well was 1,500 ft deep",
             "Deeper wells are routine and usually fine, but nothing in the "
             "original data constrains the pressure and temperature range "
             "they introduce."),
    Envelope("Liquid viscosity", "cp", 0.86, 110.0,
             "water and oils from 10 to 110 cp were tested",
             "Outside this the viscosity number group is extrapolated."),
)

# Dranchuk & Abou-Kassem (1975), fitted to the Standing-Katz chart over the
# pseudo-reduced ranges below. This one matters more than it looks: Z feeds
# density, velocity, and the Joule-Thomson coefficient.
DRANCHUK_ABOU_KASSEM = (
    Envelope("Pseudo-reduced pressure", "-", 0.2, 30.0,
             "fitted to the Standing-Katz chart over 0.2 <= Ppr < 30",
             "Outside this the polynomial is unconstrained and can return a "
             "physically impossible Z."),
    Envelope("Pseudo-reduced temperature", "-", 1.0, 3.0,
             "fitted over 1.0 < Tpr <= 3.0",
             "Below Tpr = 1 the gas may be two-phase, which a single-phase "
             "compressibility factor cannot represent at all."),
)

# Turner et al. (1969), droplet model for gas wells unloading liquid.
TURNER = (
    Envelope("Liquid production", "bbl/MMscf", 0.0, 130.0,
             "Turner's field data came from wells making under ~130 bbl/MMscf",
             "Above this the droplet model's assumption of a dispersed liquid "
             "phase breaks down and the critical rate is under-predicted."),
)


def _check(envelopes, values: dict, correlation: str, reference: str) -> Applicability:
    checks = [envelope.check(values[envelope.quantity])
              for envelope in envelopes if envelope.quantity in values]
    return Applicability(correlation=correlation, reference=reference, checks=checks)


def beggs_brill_applicability(diameter_in: float, inclination_deg: float,
                              pressure_psia: float,
                              liquid_viscosity_cp: float = None) -> Applicability:
    values = {
        "Pipe diameter": diameter_in,
        "Inclination": inclination_deg,
        "Pressure": pressure_psia,
    }
    if liquid_viscosity_cp is not None:
        values["Liquid viscosity"] = liquid_viscosity_cp
    return _check(BEGGS_BRILL, values, "Beggs-Brill (with Payne correction)",
                  "Beggs & Brill, JPT 1973")


def hagedorn_brown_applicability(diameter_in: float, inclination_deg: float,
                                 depth_ft: float,
                                 liquid_viscosity_cp: float = None) -> Applicability:
    values = {
        "Pipe diameter": diameter_in,
        "Inclination": abs(inclination_deg),
        "Depth": depth_ft,
    }
    if liquid_viscosity_cp is not None:
        values["Liquid viscosity"] = liquid_viscosity_cp
    return _check(HAGEDORN_BROWN, values,
                  "Hagedorn-Brown (with Griffith correction)",
                  "Hagedorn & Brown, JPT 1965")


def z_factor_applicability(p_psia: float, t_r: float, p_pc: float,
                           t_pc: float) -> Applicability:
    if p_pc <= 0.0 or t_pc <= 0.0:
        raise ValueError("pseudo-criticals must be positive, got %r, %r"
                         % (p_pc, t_pc))
    return _check(DRANCHUK_ABOU_KASSEM,
                  {"Pseudo-reduced pressure": p_psia / p_pc,
                   "Pseudo-reduced temperature": t_r / t_pc},
                  "Dranchuk-Abou-Kassem", "Dranchuk & Abou-Kassem, JCPT 1975")


def turner_applicability(liquid_bbl_per_mmscf: float) -> Applicability:
    return _check(TURNER, {"Liquid production": liquid_bbl_per_mmscf},
                  "Turner droplet model", "Turner et al., JPT 1969")


def worst(*applicabilities) -> Applicability:
    """The applicability with the most serious problem.

    A result computed through several correlations is only as well supported as
    its weakest link, and reporting them separately invites reading the
    reassuring one.
    """
    candidates = [a for a in applicabilities if a is not None]
    if not candidates:
        raise ValueError("nothing to compare")
    order = {"inside": 0, "marginal": 1, "outside": 2}
    return max(candidates,
               key=lambda a: (order[a.severity],
                              a.worst_excess if math.isfinite(a.worst_excess) else 0.0))
