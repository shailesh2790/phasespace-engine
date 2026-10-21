"""Static bottomhole pressures, referenced to a common depth.

A static bottomhole pressure means nothing on its own. It is the pressure
*at the depth the gauge stopped*, and two surveys of the same well stopped
at different depths are not measuring the same thing. In one real archive
15 of 26 wells had their static surveys run to depths differing by more
than 200 ft, and one by 3,852 ft -- enough that the depth difference is
worth more than the pressure change anyone was trying to observe. Compared
raw, that well's pressure appeared to *rise* over nine years.

So before a sequence of static pressures is a depletion history, every one
of them has to be moved to the same datum.

**The correction is from the survey's own gradient, not an assumed
density.** A static gradient survey records pressure at several depths, and
the slope between two points *is* the density of whatever stands between
them. Using the survey's own slope removes the largest source of error in
this calculation, which is guessing what fluid the well was full of.

**Which slope, though.** A shut-in well is not one fluid. Gas segregates
above liquid, and the two gradients differ by more than a factor of ten:
measured across one archive, the lower quartile of deepest-interval
gradients was 0.033 psi/ft and the median 0.353. Correcting with the gas
figure where the interval is liquid is wrong by an order of magnitude, in
the direction that invents reservoir pressure. So the gradient used is the
one from the interval the datum actually falls in, and where the datum is
below every measured point, the deepest interval's.

**Extrapolation is bounded and its cost is reported.** Below the deepest
gauge reading nobody knows what the well is full of -- that is exactly
where a gas-liquid contact tends to sit. Within a liquid column the
gradient is uncertain by perhaps 0.05 psi/ft, so every foot of
extrapolation costs about 0.05 psi of confidence, and `uncertainty_psi`
says how much has accumulated. Past `MAX_EXTRAPOLATION_FT` the correction
is refused rather than reported with a large error bar, because a figure
with a 200 psi uncertainty gets quoted without it.

**What this does not do.** It does not decide the datum. A field has a
reference depth agreed by the people who mapped it, and inventing one here
would produce pressures that cannot be compared with anybody else's.
`suggest_datum` offers the median of a well's own survey depths, which
minimises extrapolation and is defensible for watching one well over time,
and is explicitly *not* suitable for comparing wells or for a material
balance across a reservoir -- both of which need the field's own datum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Below this a static column is gas; above it, liquid. The gap between is
# left as "unclear" rather than forced, because a gassy oil or a foaming
# column genuinely sits there and the honest answer is that the fluid is
# not identifiable from the gradient alone.
GAS_GRADIENT_MAX = 0.15          # psi/ft
LIQUID_GRADIENT_MIN = 0.25       # psi/ft

# A gradient outside this range is not a fluid column. Steeper than water
# means the gauge was still moving, or the readings are mislabelled; a
# negative gradient means pressure fell with depth, which a static column
# cannot do.
MAX_PLAUSIBLE_GRADIENT = 0.60    # psi/ft, above fresh water with margin
MIN_PLAUSIBLE_GRADIENT = 0.0

# How far a gradient may be carried beyond the measured interval. Each foot
# costs roughly GRADIENT_UNCERTAINTY of confidence, so this caps the error
# at about 50 psi -- small against the hundreds of psi a depletion history
# is made of, and large enough to be worth reporting.
MAX_EXTRAPOLATION_FT = 1000.0
GRADIENT_UNCERTAINTY = 0.05      # psi/ft within a liquid column

# Two points closer together than this give a slope dominated by gauge
# resolution rather than by fluid density.
MIN_INTERVAL_FT = 50.0

GAS = "gas"
LIQUID = "liquid"
UNCLEAR = "unclear"


@dataclass
class Segment:
    """One interval of a static survey, and what stands in it."""

    top_ft: float
    bottom_ft: float
    top_psia: float
    bottom_psia: float

    @property
    def thickness_ft(self) -> float:
        return self.bottom_ft - self.top_ft

    @property
    def gradient_psi_ft(self) -> float:
        return (self.bottom_psia - self.top_psia) / self.thickness_ft

    @property
    def fluid(self) -> str:
        gradient = self.gradient_psi_ft
        if gradient <= GAS_GRADIENT_MAX:
            return GAS
        if gradient >= LIQUID_GRADIENT_MIN:
            return LIQUID
        return UNCLEAR

    @property
    def plausible(self) -> bool:
        return (MIN_PLAUSIBLE_GRADIENT <= self.gradient_psi_ft
                <= MAX_PLAUSIBLE_GRADIENT)

    def contains(self, depth_ft: float) -> bool:
        return self.top_ft <= depth_ft <= self.bottom_ft


@dataclass
class DatumResult:
    """A pressure moved to a datum, and what the move cost."""

    datum_ft: float
    pressure_psia: float = 0.0
    measured_psia: float = 0.0
    measured_at_ft: float = 0.0
    gradient_psi_ft: float = 0.0
    fluid: str = ""
    extrapolated_ft: float = 0.0
    usable: bool = False
    notes: list = field(default_factory=list)

    @property
    def uncertainty_psi(self) -> float:
        """What the extrapolation is worth in confidence.

        Zero when the datum sits inside a measured interval, because then
        the gradient is the one the gauge actually recorded there.
        """
        return abs(self.extrapolated_ft) * GRADIENT_UNCERTAINTY

    @property
    def correction_psi(self) -> float:
        return self.pressure_psia - self.measured_psia

    def describe(self) -> str:
        if not self.usable:
            return "; ".join(self.notes) or "not correctable"
        where = ("extrapolated %.0f ft below the deepest reading"
                 % self.extrapolated_ft if self.extrapolated_ft
                 else "inside a measured interval")
        return ("%.1f psia at %.0f ft datum, from %.1f psia measured at "
                "%.0f ft (%+.1f psi over %s at %.4f psi/ft, %s), "
                "uncertainty +/-%.0f psi"
                % (self.pressure_psia, self.datum_ft, self.measured_psia,
                   self.measured_at_ft, self.correction_psi, where,
                   self.gradient_psi_ft, self.fluid, self.uncertainty_psi))


def segments(points) -> list:
    """Intervals between consecutive readings, deepest last.

    `points` is any sequence of objects with `tvd_ft` and `pressure_psia`,
    which is what the survey readers already produce.
    """
    ordered = sorted(points, key=lambda p: float(p.tvd_ft))
    out = []
    for upper, lower in zip(ordered, ordered[1:]):
        thickness = float(lower.tvd_ft) - float(upper.tvd_ft)
        if thickness < MIN_INTERVAL_FT:
            # A slope over 20 ft is gauge resolution, not fluid density.
            continue
        out.append(Segment(top_ft=float(upper.tvd_ft),
                           bottom_ft=float(lower.tvd_ft),
                           top_psia=float(upper.pressure_psia),
                           bottom_psia=float(lower.pressure_psia)))
    return out


def fluid_contact_ft(points) -> float:
    """Depth where the gradient changes from gas to liquid, or 0.

    Useful on its own: a shut-in well's gas-liquid contact is where the
    liquid level has risen to, and watching it move over time says more
    about loading than any single pressure does.
    """
    found = segments(points)
    for upper, lower in zip(found, found[1:]):
        if upper.fluid == GAS and lower.fluid == LIQUID:
            return lower.top_ft
    return 0.0


def pressure_at_datum(points, datum_ft: float) -> DatumResult:
    """Move a static survey's pressure to `datum_ft`.

    The gradient used is the one measured in the interval containing the
    datum. Where the datum is deeper than every reading, the deepest
    interval's gradient is carried down, and how far is reported.
    """
    result = DatumResult(datum_ft=float(datum_ft))
    found = segments(points)

    if not found:
        result.notes.append(
            "fewer than two readings more than %.0f ft apart, so there is "
            "no gradient to correct with. A single-point static pressure "
            "cannot be moved to a datum without assuming a fluid density, "
            "and that assumption is the largest error in this calculation."
            % MIN_INTERVAL_FT)
        return result

    usable = [s for s in found if s.plausible]
    if not usable:
        result.notes.append(
            "every interval has an implausible gradient (%s psi/ft). A "
            "static column cannot lose pressure with depth, and nothing is "
            "denser than water here, so these readings are not a static "
            "survey -- the gauge was probably still moving."
            % ", ".join("%.3f" % s.gradient_psi_ft for s in found[:4]))
        return result

    ordered = sorted(points, key=lambda p: float(p.tvd_ft))
    deepest = ordered[-1]

    inside = [s for s in usable if s.contains(datum_ft)]
    if inside:
        segment = inside[-1]
        result.gradient_psi_ft = segment.gradient_psi_ft
        result.fluid = segment.fluid
        result.measured_psia = segment.bottom_psia
        result.measured_at_ft = segment.bottom_ft
        result.extrapolated_ft = 0.0
        result.pressure_psia = (segment.bottom_psia
                                + segment.gradient_psi_ft
                                * (datum_ft - segment.bottom_ft))
        result.usable = True
        result.notes.append(
            "the datum falls inside a measured interval, so the gradient "
            "used is the one the gauge recorded there rather than an "
            "extrapolation")
        if segment.fluid == UNCLEAR:
            result.notes.append(
                "the gradient of %.3f psi/ft is between gas and liquid, so "
                "the fluid is not identifiable from it -- a gassy oil or a "
                "foaming column both sit here"
                % segment.gradient_psi_ft)
        return result

    distance = float(datum_ft) - float(deepest.tvd_ft)
    if distance < 0.0:
        # Above every reading. The shallowest interval's gradient applies,
        # and this is usually a mistake rather than a question: a datum
        # above the gauge means the datum belongs to a different well or a
        # different field.
        segment = usable[0]
        result.notes.append(
            "the datum at %.0f ft is above the shallowest reading at %.0f "
            "ft. That is possible but unusual, and is more often a datum "
            "from another field than a survey that stopped deep."
            % (datum_ft, segment.top_ft))
        result.gradient_psi_ft = segment.gradient_psi_ft
        result.fluid = segment.fluid
        result.measured_psia = segment.top_psia
        result.measured_at_ft = segment.top_ft
        result.extrapolated_ft = distance
        result.pressure_psia = (segment.top_psia
                                + segment.gradient_psi_ft
                                * (datum_ft - segment.top_ft))
        result.usable = abs(distance) <= MAX_EXTRAPOLATION_FT
        if not result.usable:
            result.notes.append(
                "and %.0f ft is beyond the %.0f ft this will extrapolate"
                % (abs(distance), MAX_EXTRAPOLATION_FT))
        return result

    segment = usable[-1]
    if distance > MAX_EXTRAPOLATION_FT:
        result.notes.append(
            "the datum is %.0f ft below the deepest reading, and this will "
            "carry a gradient at most %.0f ft. Below the deepest gauge "
            "nobody knows what the well is full of, which is exactly where "
            "a gas-liquid contact tends to sit, and %.0f ft of it would be "
            "worth +/-%.0f psi -- a figure that gets quoted without its "
            "error bar."
            % (distance, MAX_EXTRAPOLATION_FT, distance,
               distance * GRADIENT_UNCERTAINTY))
        return result

    result.gradient_psi_ft = segment.gradient_psi_ft
    result.fluid = segment.fluid
    result.measured_psia = segment.bottom_psia
    result.measured_at_ft = segment.bottom_ft
    result.extrapolated_ft = distance
    result.pressure_psia = (segment.bottom_psia
                            + segment.gradient_psi_ft * distance)
    result.usable = True
    result.notes.append(
        "carried %.0f ft below the deepest reading on the deepest measured "
        "gradient (%.4f psi/ft, %s), worth +/-%.0f psi"
        % (distance, segment.gradient_psi_ft, segment.fluid,
           distance * GRADIENT_UNCERTAINTY))
    if segment.fluid == GAS:
        result.notes.append(
            "that deepest gradient is a gas gradient, so this assumes gas "
            "continues below the gauge. If liquid stands there the real "
            "pressure is higher, by up to %.0f psi at a water gradient."
            % (distance * (0.433 - segment.gradient_psi_ft)))
    return result


def suggest_datum(surveys) -> float:
    """A datum for watching one well over time: the median of its own
    survey depths.

    This minimises total extrapolation and is defensible for a single
    well's trend. It is **not** suitable for comparing wells, or for a
    material balance across a reservoir, because every well would get a
    different datum and the pressures would not be on the same basis. Both
    of those need the field's agreed reference depth, which is a decision
    made by whoever mapped the field and not one this function can take.

    `surveys` is any sequence of point-sequences.
    """
    depths = []
    for points in surveys:
        ordered = sorted(points, key=lambda p: float(p.tvd_ft))
        if ordered:
            depths.append(float(ordered[-1].tvd_ft))
    if not depths:
        return 0.0
    depths.sort()
    middle = len(depths) // 2
    if len(depths) % 2:
        return depths[middle]
    return 0.5 * (depths[middle - 1] + depths[middle])
