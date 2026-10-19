"""Propagate a dynamometer card down the rod string.

This is the deterministic core of rod-pump diagnostics, and it is the whole
reason surface and downhole cards look different. A stress wave travels down
steel at about 17,000 ft/s, so a round trip on a 6,000 ft string takes ~0.7 s
-- a large fraction of a pump cycle at 8 SPM. The surface card is therefore
the downhole card plus the string's own dynamics: rod weight, elastic stretch,
inertia, damping, and reflections off every taper change.

Method
------
Gibbs (1963). The rod string obeys a damped wave equation,

    d2u/dt2 + c du/dt = a^2 d2u/dx2

with u the axial displacement, x measured downward from the polished rod,
a = sqrt(E g / rho) the acoustic velocity, and c a viscous damping
coefficient. A dynamometer records one full cycle, so the surface data is
periodic and can be written as a Fourier series. Each harmonic then
propagates independently, which turns a PDE into one ordinary transfer per
frequency:

    U(x) = U(0) cosh(kx) + F(0) / (E A k) * sinh(kx)
    F(x) = F(0) cosh(kx) + E A k * U(0) * sinh(kx)

with k = sqrt(-w^2 + i c w) / a complex, so a single expression carries both
the phase lag and the attenuation. Damping is what makes k complex; with
c = 0 the string is lossless and the card propagates without decay.

The zero-frequency term is handled separately and is not a special case for
convenience -- it is where the rod weight lives. A harmonic carries no net
force, so if the static term were propagated with the same transfer the
downhole card would come back without the string's weight subtracted, which
is the largest single difference between the two cards.

Tapered strings are propagated section by section. Each taper change is an
impedance discontinuity that partially reflects the wave, and treating the
string as uniform puts those reflections in the wrong place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rodstring import STEEL_ELASTIC_MODULUS_PSI, RodString


@dataclass
class Card:
    """One dynamometer card: load against position over a complete cycle.

    Position is inches of polished-rod travel, zero at the bottom of the
    stroke. Load is pounds force. Samples are evenly spaced over exactly one
    cycle, which is what makes the Fourier treatment valid.
    """

    position_in: np.ndarray
    load_lb: np.ndarray
    period_s: float

    def __post_init__(self):
        self.position_in = np.asarray(self.position_in, dtype=float)
        self.load_lb = np.asarray(self.load_lb, dtype=float)
        if self.position_in.shape != self.load_lb.shape:
            raise ValueError("position and load must have the same length, got %s and %s"
                             % (self.position_in.shape, self.load_lb.shape))
        if self.position_in.size < 8:
            raise ValueError("a card needs at least 8 samples, got %d" % self.position_in.size)
        if self.period_s <= 0.0:
            raise ValueError("stroke period must be positive, got %r" % (self.period_s,))

    @property
    def n_samples(self) -> int:
        return int(self.position_in.size)

    @property
    def strokes_per_minute(self) -> float:
        return 60.0 / self.period_s

    @property
    def stroke_length_in(self) -> float:
        return float(self.position_in.max() - self.position_in.min())

    @property
    def peak_load_lb(self) -> float:
        return float(self.load_lb.max())

    @property
    def minimum_load_lb(self) -> float:
        return float(self.load_lb.min())

    def area_lb_in(self) -> float:
        """Net work per cycle, the shoelace area of the closed loop.

        Positive by convention for a card traversed in the physical direction
        (load high on the upstroke). Downhole, this area is the work the pump
        does on the fluid, so it is proportional to what the well produced
        that stroke -- which is why a downhole card's area is a production
        measurement, not merely a shape.
        """
        x = self.position_in
        y = self.load_lb
        return float(0.5 * np.sum((np.roll(x, -1) - x) * (np.roll(y, -1) + y)))


def _fourier_transfer(
    position_in: np.ndarray,
    load_lb: np.ndarray,
    period_s: float,
    length_ft: float,
    area_in2: float,
    damping_per_s: float,
    acoustic_velocity_ft_s: float,
    weight_lb: float,
    downward: bool = True,
) -> tuple:
    """Propagate one uniform section, surface-to-bottom or the reverse.

    Everything below is per-harmonic and independent, which is the whole
    value of the Fourier treatment: an inherently coupled PDE becomes a
    diagonal operator.
    """
    n = position_in.size
    # Work in feet and pounds so E*A has force units consistent with load.
    x_ft = length_ft if downward else -length_ft

    position_fft = np.fft.rfft(position_in)
    load_fft = np.fft.rfft(load_lb)
    frequencies = 2.0 * np.pi * np.arange(position_fft.size) / period_s

    stiffness = STEEL_ELASTIC_MODULUS_PSI * area_in2  # lb per unit strain

    out_position = np.zeros_like(position_fft)
    out_load = np.zeros_like(load_fft)

    for i, omega in enumerate(frequencies):
        if i == 0:
            # Static term. Tension falls by the weight of rod above, and the
            # string stretches under its own load. This is where rod weight
            # enters; no harmonic can carry it.
            mean_load = load_fft[0].real / n
            if downward:
                out_load[0] = load_fft[0] - weight_lb * n
                # Average tension over the section, seen from the top: the load
                # at this end less half the weight hanging below within it.
                average_tension = mean_load - 0.5 * weight_lb
            else:
                out_load[0] = load_fft[0] + weight_lb * n
                # Seen from the bottom the same average tension is the local
                # load *plus* half the section weight. Using the same
                # expression in both directions makes the transfer asymmetric,
                # and the round trip then fails to recover the position it
                # started from -- by 43 in on a 100 in stroke, which is how
                # this was caught.
                average_tension = mean_load + 0.5 * weight_lb
            stretch_in = average_tension * length_ft * 12.0 / stiffness
            out_position[0] = position_fft[0] + (-stretch_in * n if downward
                                                 else stretch_in * n)
            continue

        # k^2 = (-w^2 + i c w) / a^2, taken in inches to match position units.
        a_in_s = acoustic_velocity_ft_s * 12.0
        k = np.sqrt(complex(-omega * omega, damping_per_s * omega)) / a_in_s
        kx = k * (x_ft * 12.0)

        cosh_kx = np.cosh(kx)
        sinh_kx = np.sinh(kx)

        # Guard the k -> 0 limit, where sinh(kx)/k -> x and the transfer
        # degenerates to the static one. Without this a very low harmonic
        # divides by ~0 and the card fills with NaN.
        if abs(k) < 1e-14:
            sinh_over_k = x_ft * 12.0
            k_sinh = 0.0
        else:
            sinh_over_k = sinh_kx / k
            k_sinh = k * sinh_kx

        # Sign convention matters here and is easy to get wrong. Position is
        # positive *upward* (as the polished rod rises) while x runs *downward*
        # into the well, so tension is F = -EA du/dx, not +EA du/dx. Taking the
        # positive sign flips the coupling between position and load, and the
        # symptom is subtle: the round trip still closes exactly, because an
        # inverse of a wrong transfer is still its inverse. What gives it away
        # is energy -- with the wrong sign, more damping made *more* work reach
        # the pump.
        out_position[i] = position_fft[i] * cosh_kx - load_fft[i] * sinh_over_k / stiffness
        out_load[i] = load_fft[i] * cosh_kx - stiffness * k_sinh * position_fft[i]

    return np.fft.irfft(out_position, n), np.fft.irfft(out_load, n)


def surface_to_downhole(
    card: Card,
    string: RodString,
    damping_per_s: float = 0.4,
    fluid_specific_gravity: float = 0.85,
) -> Card:
    """The pump card implied by a measured surface card.

    `damping_per_s` is the viscous damping coefficient. It is the one tuned
    parameter in the method and it matters: too little and the downhole card
    rings with reflections that are not physically there, too much and real
    features are smoothed away. Values between about 0.1 and 1.0 per second
    cover most wells, and it is conventionally adjusted until the downhole
    card's upstroke and downstroke lines close cleanly.
    """
    if damping_per_s < 0.0:
        raise ValueError("damping cannot be negative, got %r" % (damping_per_s,))

    position = card.position_in.copy()
    load = card.load_lb.copy()

    steel_sg = 490.0 / 62.4
    buoyancy = 1.0 - fluid_specific_gravity / steel_sg

    for section in string.sections:
        position, load = _fourier_transfer(
            position, load, card.period_s,
            length_ft=section.length_ft,
            area_in2=section.area_in2,
            damping_per_s=damping_per_s,
            acoustic_velocity_ft_s=section.acoustic_velocity_ft_s,
            weight_lb=section.weight_lb * buoyancy,
            downward=True,
        )

    return Card(position_in=position, load_lb=load, period_s=card.period_s)


def downhole_to_surface(
    card: Card,
    string: RodString,
    damping_per_s: float = 0.4,
    fluid_specific_gravity: float = 0.85,
) -> Card:
    """The surface card a given pump card would produce.

    The inverse direction, and the one that makes a labelled training set
    possible: a fault can be described exactly as a downhole card shape, then
    propagated *up* to the surface card a dynamometer would actually record.
    That is physics generating labelled data, the same strategy the gas
    surrogate uses.
    """
    if damping_per_s < 0.0:
        raise ValueError("damping cannot be negative, got %r" % (damping_per_s,))

    position = card.position_in.copy()
    load = card.load_lb.copy()

    steel_sg = 490.0 / 62.4
    buoyancy = 1.0 - fluid_specific_gravity / steel_sg

    # Reverse order: coming up the string, the bottom section is traversed first.
    for section in reversed(string.sections):
        position, load = _fourier_transfer(
            position, load, card.period_s,
            length_ft=section.length_ft,
            area_in2=section.area_in2,
            damping_per_s=damping_per_s,
            acoustic_velocity_ft_s=section.acoustic_velocity_ft_s,
            weight_lb=section.weight_lb * buoyancy,
            downward=False,
        )

    return Card(position_in=position, load_lb=load, period_s=card.period_s)
