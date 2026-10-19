"""Flow through a bean: the smallest calculation an operator does every day.

A wellhead choke sets the rate. Changing a bean is the cheapest intervention
there is, and "what will this well make on a 10 mm bean" is asked far more
often than any nodal analysis. The published correlations are all one line:

    P1 = A * qL * GLR^B / S^C

with P1 the upstream pressure in psig, qL the liquid rate in STB/day, GLR the
gas-liquid ratio in scf/STB, and S the bean size in 64ths of an inch. Gilbert
fitted A, B and C to Ten Section field data in 1954; Ros, Baxendell and Achong
refitted the same form to other fields and got noticeably different constants,
which is itself the most useful thing to know about them.

Two properties of this form decide whether an answer means anything.

**It assumes critical flow.** Above a sonic velocity in the bean the downstream
pressure stops affecting the rate, which is why the correlation has no P2 term
at all. Below it, the downstream pressure matters and the correlation is simply
the wrong model -- it will still return a number, and that number will be
independent of a pressure it should depend on.

**It was fitted on oil wells.** The gas-liquid ratios in Gilbert's data reach
about 1,500 scf/STB. Applied to a gas well making a little condensate, at tens
of thousands of scf/STB, it is extrapolating one to two orders of magnitude
beyond anything it has seen. It does not degrade gracefully: on a real well
from the validation set at 36,000 scf/STB every one of these four correlations
is 50-68% low.

So every result carries the envelope check, and the check is worth more than
the number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .envelopes import Envelope, _check

# A, B, C for P1 = A * qL * GLR^B / S^C.
#
# Four fits of one equation to four different fields. Their spread on the same
# well is the honest uncertainty of the method, which is why all four are
# returned rather than a house favourite.
CHOKE_CORRELATIONS = {
    "gilbert": (10.00, 0.546, 1.89),
    "ros": (17.40, 0.500, 2.00),
    "baxendell": (9.56, 0.546, 1.93),
    "achong": (3.82, 0.650, 1.88),
}

CORRELATION_SOURCES = {
    "gilbert": "Gilbert, API Drilling and Production Practice 1954 (Ten Section field)",
    "ros": "Ros, Applied Scientific Research 1960",
    "baxendell": "Baxendell 1957",
    "achong": "Achong 1961 (Lake Maracaibo)",
}

# Below this downstream/upstream ratio the bean is choked and the downstream
# pressure no longer sets the rate. The conventional figure for a multiphase
# bean; the single-phase gas value is 0.55 and liquids do not choke at all in
# this sense, so it is a guide rather than a threshold in the physics.
CRITICAL_PRESSURE_RATIO = 0.55

MM_PER_IN = 25.4
SIXTY_FOURTHS_PER_IN = 64.0

CHOKE_ENVELOPE = (
    Envelope("Gas-liquid ratio", "scf/STB", 0.0, 1500.0,
             "Gilbert's Ten Section data reached roughly 1,500 scf/STB",
             "Above this the correlation is extrapolating past any oil well it "
             "was fitted on. On a gas well making condensate it under-predicts "
             "the upstream pressure by half or more, and does so confidently."),
    Envelope("Liquid rate", "STB/day", 30.0, 1200.0,
             "the fitted data covered roughly 30 to 1,200 STB/day",
             "Outside this the linear dependence on rate is unsupported."),
    Envelope("Bean size", "64ths in", 4.0, 36.0,
             "beans from 4/64 to 36/64 in were tested",
             "The exponent on bean size is the most sensitive constant in the "
             "correlation, so extrapolating it is the worst of the three."),
)


@dataclass
class ChokeResult:
    """What a bean calculation produced, and whether to believe it."""

    correlation: str
    upstream_psig: float
    liquid_rate_stb_d: float
    gas_liquid_ratio: float
    bean_64ths: float
    critical: bool
    pressure_ratio: float
    applicability: object = None
    notes: list = field(default_factory=list)

    @property
    def source(self) -> str:
        return CORRELATION_SOURCES.get(self.correlation, self.correlation)


def bean_64ths(diameter_mm: float = None, diameter_in: float = None) -> float:
    """Bean size in 64ths of an inch, from millimetres or inches.

    The correlations are written in 64ths because that is how beans are
    stocked and logged in North America. Most of the rest of the world records
    them in millimetres, and an 8 mm bean entered as 8/64 in is a quarter of
    its real area -- an error that shows up as a wildly wrong pressure rather
    than as anything obviously unit-shaped.
    """
    if (diameter_mm is None) == (diameter_in is None):
        raise ValueError("give exactly one of diameter_mm or diameter_in")
    inches = diameter_in if diameter_in is not None else diameter_mm / MM_PER_IN
    if inches <= 0.0:
        raise ValueError("bean diameter must be positive, got %r" % (inches,))
    return inches * SIXTY_FOURTHS_PER_IN


def choke_applicability(liquid_rate_stb_d: float, gas_liquid_ratio: float,
                        bean_size_64ths: float):
    return _check(CHOKE_ENVELOPE,
                  {"Gas-liquid ratio": gas_liquid_ratio,
                   "Liquid rate": liquid_rate_stb_d,
                   "Bean size": bean_size_64ths},
                  "Multiphase choke (critical flow)",
                  "Gilbert 1954 and later refits")


def upstream_pressure_psig(
    liquid_rate_stb_d: float,
    gas_liquid_ratio: float,
    bean_size_64ths: float,
    correlation: str = "gilbert",
    downstream_psig: float = None,
) -> ChokeResult:
    """Pressure upstream of the bean for a given rate through it.

    ``downstream_psig`` is not used in the calculation -- the correlation has
    no term for it -- and is accepted solely so the critical-flow assumption
    can be *checked*. A model whose governing assumption is never tested is
    a model that fails silently.
    """
    name = correlation.lower()
    if name not in CHOKE_CORRELATIONS:
        raise ValueError("unknown choke correlation %r (known: %s)"
                         % (correlation, ", ".join(sorted(CHOKE_CORRELATIONS))))
    if liquid_rate_stb_d <= 0.0:
        raise ValueError("liquid rate must be positive, got %r" % (liquid_rate_stb_d,))
    if gas_liquid_ratio < 0.0:
        raise ValueError("gas-liquid ratio cannot be negative, got %r"
                         % (gas_liquid_ratio,))
    if bean_size_64ths <= 0.0:
        raise ValueError("bean size must be positive, got %r" % (bean_size_64ths,))

    a, b, c = CHOKE_CORRELATIONS[name]
    pressure = a * liquid_rate_stb_d * (gas_liquid_ratio ** b) / (bean_size_64ths ** c)

    notes = []
    critical = True
    ratio = 0.0
    if downstream_psig is not None:
        upstream_abs = pressure + 14.6959
        downstream_abs = downstream_psig + 14.6959
        ratio = downstream_abs / upstream_abs if upstream_abs > 0 else 1.0
        critical = ratio <= CRITICAL_PRESSURE_RATIO
        if not critical:
            notes.append(
                "Downstream/upstream pressure ratio is %.2f, above the %.2f at "
                "which a bean chokes. Below critical flow the downstream "
                "pressure sets the rate, and this correlation has no term for "
                "it -- the answer is independent of a pressure it should "
                "depend on." % (ratio, CRITICAL_PRESSURE_RATIO))

    applicability = choke_applicability(liquid_rate_stb_d, gas_liquid_ratio,
                                        bean_size_64ths)
    notes.extend(applicability.messages())

    return ChokeResult(correlation=name, upstream_psig=pressure,
                       liquid_rate_stb_d=liquid_rate_stb_d,
                       gas_liquid_ratio=gas_liquid_ratio,
                       bean_64ths=bean_size_64ths, critical=critical,
                       pressure_ratio=ratio, applicability=applicability,
                       notes=notes)


def liquid_rate_stb_d(
    upstream_psig: float,
    gas_liquid_ratio: float,
    bean_size_64ths: float,
    correlation: str = "gilbert",
) -> float:
    """The question actually asked: what will it make on this bean?

    The correlation is linear in liquid rate, so the inverse is exact rather
    than iterative -- worth noting because it means the inverse inherits the
    forward direction's error exactly, with nothing added and nothing hidden.
    """
    name = correlation.lower()
    if name not in CHOKE_CORRELATIONS:
        raise ValueError("unknown choke correlation %r (known: %s)"
                         % (correlation, ", ".join(sorted(CHOKE_CORRELATIONS))))
    if upstream_psig <= 0.0:
        raise ValueError("upstream pressure must be positive, got %r" % (upstream_psig,))
    if bean_size_64ths <= 0.0:
        raise ValueError("bean size must be positive, got %r" % (bean_size_64ths,))

    a, b, c = CHOKE_CORRELATIONS[name]
    denominator = a * (max(gas_liquid_ratio, 1e-9) ** b)
    return upstream_psig * (bean_size_64ths ** c) / denominator


def compare_correlations(liquid_rate_stb_d: float, gas_liquid_ratio: float,
                         bean_size_64ths: float,
                         downstream_psig: float = None) -> dict:
    """All four fits on one well.

    Shown together rather than averaged, in the same way the two multiphase
    VLP correlations are: the spread between four independent fits of the same
    equation is the honest uncertainty of the method, and averaging it away
    reports a precision nobody has.
    """
    return {name: upstream_pressure_psig(liquid_rate_stb_d, gas_liquid_ratio,
                                         bean_size_64ths, name, downstream_psig)
            for name in CHOKE_CORRELATIONS}


def correlation_spread(results: dict) -> float:
    """Fractional spread across the correlations, as an uncertainty estimate."""
    pressures = [r.upstream_psig for r in results.values()]
    if not pressures:
        return 0.0
    middle = sum(pressures) / len(pressures)
    if middle <= 0.0 or not math.isfinite(middle):
        return 0.0
    return (max(pressures) - min(pressures)) / middle
