"""Rod string geometry and material properties.

A sucker-rod string is tapered: heavy rods at the top where the load is
greatest, lighter ones below. That taper is not a detail -- it changes the
acoustic impedance at every section boundary, and a wave travelling down the
string partially reflects at each one. A diagnosis that models the string as
uniform will misplace those reflections and mis-shape the downhole card.

Sizes follow the API rod numbering convention, where the number is the rod
diameter in eighths of an inch: a 76 string is 7/8 in at the top and 6/8 in
below it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Steel rod properties. E and density are the values used throughout the
# rod-pump literature; changing them changes the acoustic velocity and so the
# timing of every reflection.
STEEL_ELASTIC_MODULUS_PSI = 30.5e6
STEEL_DENSITY_LB_FT3 = 490.0
GRAVITY_FT_S2 = 32.174

# API rod number -> diameter in inches. The number is eighths of an inch.
API_ROD_DIAMETER_IN = {
    5: 5 / 8,
    6: 6 / 8,
    7: 7 / 8,
    8: 8 / 8,
    9: 9 / 8,
    10: 10 / 8,
}


@dataclass(frozen=True)
class RodSection:
    """One uniform run of rod, ordered from surface downward."""

    api_number: int
    length_ft: float

    def __post_init__(self):
        if self.api_number not in API_ROD_DIAMETER_IN:
            raise ValueError(
                "unknown API rod number %r (known: %s)"
                % (self.api_number, ", ".join(str(k) for k in sorted(API_ROD_DIAMETER_IN))))
        if self.length_ft <= 0.0:
            raise ValueError("rod section length must be positive, got %r" % (self.length_ft,))

    @property
    def diameter_in(self) -> float:
        return API_ROD_DIAMETER_IN[self.api_number]

    @property
    def area_in2(self) -> float:
        return math.pi * self.diameter_in ** 2 / 4.0

    @property
    def weight_lb_per_ft(self) -> float:
        """Rod weight in air."""
        return STEEL_DENSITY_LB_FT3 * self.area_in2 / 144.0

    @property
    def weight_lb(self) -> float:
        return self.weight_lb_per_ft * self.length_ft

    @property
    def acoustic_velocity_ft_s(self) -> float:
        """Speed of a stress wave in steel, sqrt(E g / rho).

        About 16,300 ft/s, and independent of rod diameter -- the taper
        changes impedance, not wave speed. A round trip down a 6,000 ft
        string therefore takes about 0.74 s, which is a substantial fraction
        of a pump cycle at 8 strokes per minute. That is precisely why the
        surface card and the downhole card look different at all.
        """
        density_lb_in3 = STEEL_DENSITY_LB_FT3 / 1728.0
        # E is in psi = lbf/in^2; converting to a velocity needs g to turn
        # pounds mass into pounds force.
        return math.sqrt(
            STEEL_ELASTIC_MODULUS_PSI * GRAVITY_FT_S2 * 12.0 / density_lb_in3) / 12.0


@dataclass
class RodString:
    """A tapered string, ordered surface-first."""

    sections: list

    def __post_init__(self):
        if not self.sections:
            raise ValueError("a rod string needs at least one section")
        for section in self.sections:
            if not isinstance(section, RodSection):
                raise TypeError("sections must be RodSection, got %r" % (section,))

    @property
    def total_length_ft(self) -> float:
        return sum(section.length_ft for section in self.sections)

    @property
    def total_weight_lb(self) -> float:
        """Rod weight in air, which is what a static surface load starts from."""
        return sum(section.weight_lb for section in self.sections)

    def buoyant_weight_lb(self, fluid_specific_gravity: float = 0.85) -> float:
        """Rod weight in fluid.

        The rods are submerged, so they lose the weight of the fluid they
        displace. Ignoring buoyancy overstates the static load by several
        percent, which shifts the whole downhole card and corrupts any load
        threshold read off it.
        """
        steel_sg = STEEL_DENSITY_LB_FT3 / 62.4
        return self.total_weight_lb * (1.0 - fluid_specific_gravity / steel_sg)

    def stretch_in_per_lb(self) -> float:
        """Elastic stretch of the whole string per pound of added load.

        Sum of L/(A E) over sections. This sets the width of the load change
        between the upstroke and downstroke lines on the card, and it is the
        single number that most directly controls how much the surface card
        is smeared relative to the downhole one.
        """
        return sum(section.length_ft * 12.0
                   / (section.area_in2 * STEEL_ELASTIC_MODULUS_PSI)
                   for section in self.sections)

    def describe(self) -> str:
        taper = "-".join(str(section.api_number) for section in self.sections)
        return "%s taper, %.0f ft, %.0f lb in air" % (
            taper, self.total_length_ft, self.total_weight_lb)


def uniform_string(api_number: int, length_ft: float) -> RodString:
    """A single-size string, for tests and for the simplest wells."""
    return RodString([RodSection(api_number=api_number, length_ft=length_ft)])


def taper_86(length_ft: float, top_fraction: float = 0.5) -> RodString:
    """A common two-section taper: 8 (1 in) over 6 (3/4 in)."""
    top = length_ft * top_fraction
    return RodString([
        RodSection(api_number=8, length_ft=top),
        RodSection(api_number=6, length_ft=length_ft - top),
    ])
