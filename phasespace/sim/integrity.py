"""Where the string is being eaten, and how fast.

The market survey puts integrity in the first wave because it runs daily
on data the engine already reads: pressures and temperatures down the
traverse, rates, water cut, and the CO2 fraction the fluid model already
carries for Wichert-Aziz. No new instrumentation, no new survey.

Three mechanisms, and they are not interchangeable:

  erosion     solids and velocity cutting the wall. API RP 14E, already
              in `components.erosional_velocity_ft_s`; pulled into the
              same report here so one call answers the whole question.
  corrosion   carbon dioxide dissolving into produced water and
              attacking steel. de Waard and Milliams.
  scale       minerals dropping out of the water as pressure falls.
              Not implemented, and the reason is stated below rather
              than left as an absence.

**The single most important thing this module does is say where, not
whether.** A corrosion rate for "the well" is close to useless: the rate
depends on temperature and CO2 partial pressure, both of which vary by a
factor of several down a string, so the attack concentrates somewhere
specific. Given a traverse, this reports the depth where it is worst,
which is where a coupon goes and where an inspection should look.

**de Waard and Milliams is a worst case and is used as one.** The 1991
correlation

    log10(V) = 5.8 - 1710/T + 0.67 log10(pCO2)

with V in mm/year, T in kelvin and the CO2 partial pressure in bar,
describes bare steel, fully water-wetted, with no inhibitor and no
protective film. Real rates on an inhibited well are routinely five to
ten times lower. NORSOK M-506 is the other standard in common use and
disagrees, generally predicting less. A number from this is an upper
bound for ranking wells and siting inspections, not a remaining-life
calculation, and `CorrosionPoint.basis` says so on every result.

**Two things suppress it that the bare correlation does not know.**
Above about 60 degrees C the corrosion product itself forms a siderite
film that slows further attack, which is the scale factor applied here.
And steel only corrodes where water touches it: at low water cut and
high enough velocity the water is carried as droplets in oil and the
wall stays oil-wetted. That is a rule of thumb rather than a
correlation, so it is reported as a flag on the result instead of
silently multiplying the rate.

**Sour service is refused, not extrapolated.** Hydrogen sulphide attacks
steel by a different mechanism and brings sulphide stress cracking,
which is a metallurgical failure rather than a thinning rate. Feeding
H2S into a CO2 correlation would produce a number with nothing behind
it, so above a trace the module declines and says what is needed
instead.

**Scale is not implemented, deliberately.** Calcite and barite
saturation need a water analysis -- calcium, bicarbonate, barium,
sulphate and ionic strength -- and no archive this engine reads carries
one. The input would have to be invented, and a saturation index
computed from invented chemistry is worse than no answer. `scale_inputs`
names exactly what to ask the operator for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Above this mole fraction the gas is sour and the CO2 correlation does
# not describe what is happening to the steel. 100 ppm is the usual
# threshold for sour-service metallurgy to come into play.
SOUR_H2S_FRACTION = 1.0e-4

# Siderite starts to protect around here, which is where the de Waard
# scale factor begins to bite.
FILM_TEMPERATURE_C = 60.0

# Below this water cut, with enough velocity to keep the water
# entrained, the wall is oil-wetted and the CO2 rate does not apply.
# A rule of thumb from operating practice, not a correlation, and it is
# reported rather than applied.
OIL_WETTING_WATER_CUT = 0.30
OIL_WETTING_VELOCITY_FT_S = 3.3          # about 1 m/s

PSI_PER_BAR = 14.503773773
MM_PER_INCH = 25.4


@dataclass
class CorrosionPoint:
    """The predicted attack at one place in the string."""

    depth_ft: float
    temperature_f: float
    pressure_psia: float
    co2_partial_bar: float
    rate_mm_yr: float
    film_factor: float
    oil_wetted: bool
    basis: str = ("de Waard & Milliams 1991, bare steel, fully water-wetted, "
                  "uninhibited -- an upper bound for ranking, not a "
                  "remaining-life figure")

    @property
    def rate_mpy(self) -> float:
        """Mils per year, which is what a US inspection report uses."""
        return self.rate_mm_yr / MM_PER_INCH * 1000.0

    def years_to(self, wall_thickness_in: float) -> float:
        if self.rate_mm_yr <= 0.0:
            return float("inf")
        return wall_thickness_in * MM_PER_INCH / self.rate_mm_yr


@dataclass
class IntegrityReport:
    """Every mechanism, in one answer, with the places that matter named."""

    points: list = field(default_factory=list)
    erosional_ratio: float = 0.0        # velocity / API RP 14E limit
    erosional_depth_ft: float = 0.0
    sour: bool = False
    notes: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.points)

    @property
    def worst(self) -> CorrosionPoint:
        return max(self.points, key=lambda p: p.rate_mm_yr) if self.points else None

    @property
    def oil_wetted_throughout(self) -> bool:
        return bool(self.points) and all(p.oil_wetted for p in self.points)

    def describe(self) -> str:
        if not self.usable:
            return "; ".join(self.notes) or "nothing to report"
        worst = self.worst
        where = ("oil-wetted the whole way, so the CO2 rate below is what "
                 "would happen if the water reached the wall"
                 if self.oil_wetted_throughout else
                 "worst at %.0f ft, %.0f degF" % (worst.depth_ft,
                                                  worst.temperature_f))
        return ("CO2 corrosion up to %.2f mm/yr (%.0f mpy), %s; erosion at "
                "%.0f%% of the API RP 14E limit"
                % (worst.rate_mm_yr, worst.rate_mpy, where,
                   self.erosional_ratio * 100.0))


def co2_partial_pressure_bar(total_psia: float, co2_fraction: float) -> float:
    """Dalton: the CO2 share of the total pressure.

    Fugacity would be more correct at high pressure, where CO2 departs
    from ideal by tens of percent. The correlation this feeds was fitted
    against partial pressure, so using fugacity here would be more right
    in isolation and less right in context.
    """
    return max(total_psia, 0.0) * max(co2_fraction, 0.0) / PSI_PER_BAR


def film_factor(temperature_c: float, co2_partial_bar: float) -> float:
    """How much the corrosion product protects the steel beneath it.

    Below about 60 degrees C there is no protective siderite and the
    factor is 1. Above it the film slows the attack, which is why the
    hottest point in a well is often not the worst-corroding one -- a
    result that surprises people and is the reason to report depth.
    """
    if temperature_c <= FILM_TEMPERATURE_C:
        return 1.0
    kelvin = temperature_c + 273.15
    exponent = 2400.0 / kelvin - 0.6 * math.log10(max(co2_partial_bar, 1e-9)) - 6.7
    return min(1.0, 10.0 ** exponent)


def corrosion_rate_mm_yr(temperature_f: float, pressure_psia: float,
                         co2_fraction: float) -> tuple:
    """de Waard & Milliams, with the protective-film correction.

    Returns (rate, film factor, CO2 partial pressure in bar).
    """
    partial = co2_partial_pressure_bar(pressure_psia, co2_fraction)
    if partial <= 0.0:
        return 0.0, 1.0, 0.0
    kelvin = (temperature_f - 32.0) / 1.8 + 273.15
    base = 10.0 ** (5.8 - 1710.0 / kelvin + 0.67 * math.log10(partial))
    factor = film_factor(kelvin - 273.15, partial)
    return base * factor, factor, partial


def assess(depths_ft, temperatures_f, pressures_psia, co2_fraction: float,
           water_cut: float, velocity_ft_s: float = 0.0,
           h2s_fraction: float = 0.0,
           erosional_ratio: float = 0.0,
           erosional_depth_ft: float = 0.0) -> IntegrityReport:
    """Corrosion along a traverse, with erosion folded in.

    `depths_ft`, `temperatures_f` and `pressures_psia` are the traverse
    the engine already computes, so this costs one pass over an answer
    that exists rather than a new solve.
    """
    report = IntegrityReport(erosional_ratio=erosional_ratio,
                             erosional_depth_ft=erosional_depth_ft)

    if h2s_fraction > SOUR_H2S_FRACTION:
        report.sour = True
        report.notes.append(
            "this gas carries %.0f ppm H2S, which is sour. Hydrogen "
            "sulphide attacks steel by a different mechanism and brings "
            "sulphide stress cracking, which is a metallurgical failure "
            "rather than a thinning rate, so a CO2 correlation does not "
            "describe it. This needs a NACE MR0175 materials review and a "
            "sour-service model, not a number from here."
            % (h2s_fraction * 1e6))
        return report

    if co2_fraction <= 0.0:
        report.notes.append(
            "no CO2 in the gas composition, so there is no carbonic acid "
            "corrosion to predict. If the fluid model simply has not been "
            "given a composition, that is the thing to fix -- an absent "
            "CO2 fraction and a true zero look identical here.")
        return report

    lengths = (len(depths_ft), len(temperatures_f), len(pressures_psia))
    if len(set(lengths)) != 1 or lengths[0] == 0:
        report.notes.append(
            "the traverse arrays do not line up: %d depths, %d "
            "temperatures, %d pressures" % lengths)
        return report

    oil_wetted = (water_cut < OIL_WETTING_WATER_CUT
                  and velocity_ft_s >= OIL_WETTING_VELOCITY_FT_S)

    for depth, temperature, pressure in zip(depths_ft, temperatures_f,
                                            pressures_psia):
        rate, factor, partial = corrosion_rate_mm_yr(
            temperature, pressure, co2_fraction)
        report.points.append(CorrosionPoint(
            depth_ft=float(depth), temperature_f=float(temperature),
            pressure_psia=float(pressure), co2_partial_bar=partial,
            rate_mm_yr=rate, film_factor=factor, oil_wetted=oil_wetted))

    worst = report.worst
    hottest = max(report.points, key=lambda p: p.temperature_f)
    if worst.depth_ft != hottest.depth_ft:
        report.notes.append(
            "the worst corrosion is at %.0f ft and %.0f degF, not at the "
            "hottest point (%.0f ft, %.0f degF). Above about %.0f degC the "
            "corrosion product forms a protective film, so the bottom of a "
            "hot well is often not where it thins fastest -- which is "
            "where an inspection usually looks."
            % (worst.depth_ft, worst.temperature_f, hottest.depth_ft,
               hottest.temperature_f, FILM_TEMPERATURE_C))

    if oil_wetted:
        report.notes.append(
            "at %.0f%% water cut and %.1f ft/s the water should stay "
            "entrained in the oil and the wall oil-wetted, which suppresses "
            "CO2 attack entirely. That is operating experience rather than "
            "a correlation, and it fails at any point the well slows down "
            "or the water cut climbs -- so the rate above is what you get "
            "when it does."
            % (water_cut * 100.0, velocity_ft_s))

    if erosional_ratio > 1.0:
        report.notes.append(
            "velocity is %.0f%% of the API RP 14E erosional limit at %.0f "
            "ft. That limit is a C-factor rule of thumb from 1991 with no "
            "solids term in it, and it is widely treated as physics; treat "
            "an exceedance as a reason to check for sand, not as a "
            "prediction of metal loss."
            % (erosional_ratio * 100.0, erosional_depth_ft))

    report.notes.append(worst.basis)
    return report


def scale_inputs() -> list:
    """What an operator must supply before scale can be predicted at all.

    Named rather than defaulted. A saturation index computed from assumed
    water chemistry is a number about the assumption, and the archives
    this engine reads carry no water analysis at all.
    """
    return [
        "calcium, mg/l",
        "bicarbonate, mg/l",
        "barium, mg/l",
        "sulphate, mg/l",
        "total dissolved solids or ionic strength",
        "the sample date and the well it came from",
    ]
