"""Oil PVT: solution gas, oil volume factor, viscosity, compressibility.

Three things in this engine need to know how much a barrel of oil shrinks and
how much gas comes out of it on the way to surface: the multiphase
correlations, which need in-situ volumes rather than stock-tank ones; material
balance, which is an accounting identity in reservoir barrels and is meaningless
without them; and any forecast that crosses the bubble point.

Until now oil properties were supplied as constants -- a bubble point handed in
as a number, no solution gas at all. That is defensible for hydraulics over a
narrow pressure range and indefensible for depletion, where the whole question
is what happens as pressure falls through the bubble point.

The correlations are named, because which one was used changes the answer and
an engineer is entitled to know:

  Standing (1947)        Rs, pb, Bo -- California crudes, the industry default
  Vazquez-Beggs (1980)   Rs, Bo, co -- 6,000 measurements, worldwide, split
                         at 30 API
  Beggs-Robinson (1975)  dead- and live-oil viscosity
  Standing/Katz          gas volume factor from the Z already in `fluids`

These are correlations fitted to sample sets, not laws. On a crude outside the
set that produced them they are routinely 10-20% out on Rs, and Bo errors feed
straight into reserves. Where a laboratory PVT report exists it should be used
instead, and `calibrate_to_lab` exists for the case where one point is known:
it returns the single multiplier that makes the correlation match the
measurement, which is the honest minimum of what "tuned to field data" should
mean.

Units: pressure psia, temperature degF, Rs scf/STB, Bo and Bw bbl/STB,
Bg bbl/scf, viscosity cp, compressibility 1/psi.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

STANDING = "standing"
VAZQUEZ_BEGGS = "vazquez_beggs"

# Reservoir barrels per cubic foot. Bg is quoted in bbl/scf here so material
# balance can add it to Bo without a conversion in the middle of an accounting
# identity, which is where sign and unit errors hide.
BBL_PER_FT3 = 1.0 / 5.614583

CITATIONS = {
    STANDING: "Standing (1947), Calif. crudes",
    VAZQUEZ_BEGGS: "Vazquez & Beggs (1980), 6,000 measurements worldwide",
}


def oil_specific_gravity(api: float) -> float:
    if api <= -131.5:
        raise ValueError("API gravity below -131.5 is not a fluid, got %r" % (api,))
    return 141.5 / (131.5 + api)


def _check(pressure_psia, temperature_f, api, gas_gravity):
    if pressure_psia <= 0.0:
        raise ValueError("pressure must be positive, got %r" % (pressure_psia,))
    if temperature_f <= -459.67:
        raise ValueError("temperature below absolute zero, got %r" % (temperature_f,))
    if not 5.0 <= api <= 60.0:
        raise ValueError(
            "API gravity %r is outside 5-60, where every one of these "
            "correlations was fitted. Supply laboratory PVT instead." % (api,))
    if gas_gravity <= 0.0:
        raise ValueError("gas gravity must be positive, got %r" % (gas_gravity,))


# ====================================================== solution gas-oil ====


def solution_gas_standing(pressure_psia: float, temperature_f: float,
                          api: float, gas_gravity: float) -> float:
    """Standing's Rs, scf/STB.

        Rs = gg [ (p/18.2 + 1.4) 10^(0.0125 API - 0.00091 T) ]^1.2048

    Valid only at or below the bubble point: above it no more gas can dissolve,
    and the caller is responsible for capping Rs at Rsb. Returning the
    unbounded correlation value above pb is the classic way to make an
    undersaturated oil appear to keep absorbing gas.
    """
    _check(pressure_psia, temperature_f, api, gas_gravity)
    exponent = 0.0125 * api - 0.00091 * temperature_f
    return gas_gravity * ((pressure_psia / 18.2 + 1.4) * 10.0 ** exponent) ** 1.2048


def bubble_point_standing(rs_scf_stb: float, temperature_f: float,
                          api: float, gas_gravity: float) -> float:
    """Standing's bubble point, psia -- the inverse of `solution_gas_standing`."""
    if rs_scf_stb < 0.0:
        raise ValueError("solution gas cannot be negative, got %r" % (rs_scf_stb,))
    _check(1.0, temperature_f, api, gas_gravity)
    if rs_scf_stb == 0.0:
        return 14.7
    exponent = 0.00091 * temperature_f - 0.0125 * api
    inner = (rs_scf_stb / gas_gravity) ** 0.83 * 10.0 ** exponent - 1.4
    return max(18.2 * inner, 14.7)


def _corrected_gas_gravity(gas_gravity, api, separator_temperature_f,
                           separator_psia):
    """Vazquez-Beggs' gas gravity corrected to a 100 psig separator.

    The correlation was fitted on gravities referenced to one separator
    condition. Skipping this correction is a few percent on Rs, and it is the
    kind of few percent that is invisible until a material balance refuses to
    close.
    """
    if separator_psia <= 0.0:
        raise ValueError("separator pressure must be positive, got %r"
                         % (separator_psia,))
    return gas_gravity * (1.0 + 5.912e-5 * api * separator_temperature_f
                          * math.log10(separator_psia / 114.7))


def solution_gas_vazquez_beggs(
    pressure_psia: float,
    temperature_f: float,
    api: float,
    gas_gravity: float,
    separator_temperature_f: float = 100.0,
    separator_psia: float = 114.7,
) -> float:
    """Vazquez-Beggs Rs, scf/STB. Split at 30 API, as the authors fitted it."""
    _check(pressure_psia, temperature_f, api, gas_gravity)
    gg = _corrected_gas_gravity(gas_gravity, api, separator_temperature_f,
                                separator_psia)
    if api <= 30.0:
        c1, c2, c3 = 0.0362, 1.0937, 25.7240
    else:
        c1, c2, c3 = 0.0178, 1.1870, 23.9310
    return c1 * gg * pressure_psia ** c2 * math.exp(
        c3 * api / (temperature_f + 459.67))


# ======================================================= volume factors =====


def oil_fvf_standing(rs_scf_stb: float, temperature_f: float, api: float,
                     gas_gravity: float) -> float:
    """Standing's Bo at or below the bubble point, bbl/STB.

        Bo = 0.9759 + 0.000120 [ Rs (gg/go)^0.5 + 1.25 T ]^1.2048
    """
    if rs_scf_stb < 0.0:
        raise ValueError("solution gas cannot be negative, got %r" % (rs_scf_stb,))
    _check(1.0, temperature_f, api, gas_gravity)
    go = oil_specific_gravity(api)
    group = rs_scf_stb * math.sqrt(gas_gravity / go) + 1.25 * temperature_f
    return 0.9759 + 0.000120 * group ** 1.2048


def oil_fvf_vazquez_beggs(
    rs_scf_stb: float,
    temperature_f: float,
    api: float,
    gas_gravity: float,
    separator_temperature_f: float = 100.0,
    separator_psia: float = 114.7,
) -> float:
    """Vazquez-Beggs Bo at or below the bubble point, bbl/STB."""
    if rs_scf_stb < 0.0:
        raise ValueError("solution gas cannot be negative, got %r" % (rs_scf_stb,))
    _check(1.0, temperature_f, api, gas_gravity)
    gg = _corrected_gas_gravity(gas_gravity, api, separator_temperature_f,
                                separator_psia)
    if api <= 30.0:
        c1, c2, c3 = 4.677e-4, 1.751e-5, -1.811e-8
    else:
        c1, c2, c3 = 4.670e-4, 1.100e-5, 1.337e-9
    return (1.0 + c1 * rs_scf_stb
            + (temperature_f - 60.0) * (api / gg) * (c2 + c3 * rs_scf_stb))


def oil_compressibility(pressure_psia: float, rs_scf_stb: float,
                        temperature_f: float, api: float, gas_gravity: float,
                        separator_temperature_f: float = 100.0,
                        separator_psia: float = 114.7) -> float:
    """Vazquez-Beggs undersaturated oil compressibility, 1/psi.

        co = (-1433 + 5 Rs + 17.2 T - 1180 ggs + 12.61 API) / (1e5 p)

    Only meaningful above the bubble point. Below it the oil is saturated and
    its apparent compressibility is dominated by gas coming out of solution,
    which Bo already accounts for; using this there double-counts it.
    """
    _check(pressure_psia, temperature_f, api, gas_gravity)
    gg = _corrected_gas_gravity(gas_gravity, api, separator_temperature_f,
                                separator_psia)
    value = (-1433.0 + 5.0 * rs_scf_stb + 17.2 * temperature_f
             - 1180.0 * gg + 12.61 * api) / (1.0e5 * pressure_psia)
    # The correlation can go negative on a heavy, gas-poor oil at high
    # pressure, which is physically impossible -- oil does not expand under
    # compression. Floored at a small positive value rather than passed on.
    return max(value, 1.0e-7)


def gas_fvf(pressure_psia: float, temperature_f: float, z_factor: float) -> float:
    """Bg in bbl/scf, from the real-gas law.

        Bg = 0.00503 z T / p      (T in R, p in psia)

    Not a correlation: this is the gas law with the Z factor the `fluids`
    module already computes from Dranchuk-Abou-Kassem. The 0.00503 is
    5.615 ft3/bbl into the 0.02827 ft3/scf form.
    """
    if pressure_psia <= 0.0:
        raise ValueError("pressure must be positive, got %r" % (pressure_psia,))
    if z_factor <= 0.0:
        raise ValueError("Z factor must be positive, got %r" % (z_factor,))
    return 0.02827 * z_factor * (temperature_f + 459.67) / pressure_psia * BBL_PER_FT3


def water_fvf(pressure_psia: float, temperature_f: float) -> float:
    """Bw in bbl/STB, McCain's form.

    Water expands slightly with temperature and compresses slightly with
    pressure, and the two nearly cancel, which is why Bw is usually taken as
    1.0. It is included because material balance in a waterdrive field
    multiplies it by a large Wp, and 2% of a large number is not small.
    """
    if pressure_psia <= 0.0:
        raise ValueError("pressure must be positive, got %r" % (pressure_psia,))
    d_vwt = (-1.0001e-2 + 1.33391e-4 * temperature_f
             + 5.50654e-7 * temperature_f ** 2)
    d_vwp = ((-1.95301e-9 * pressure_psia * temperature_f
              - 1.72834e-13 * pressure_psia ** 2 * temperature_f
              - 3.58922e-7 * pressure_psia
              - 2.25341e-10 * pressure_psia ** 2))
    return (1.0 + d_vwt) * (1.0 + d_vwp)


# ============================================================= viscosity ====


def dead_oil_viscosity(temperature_f: float, api: float) -> float:
    """Beggs-Robinson dead-oil viscosity, cp.

        z = 3.0324 - 0.02023 API,  y = 10^z,  x = y T^-1.163,  mu = 10^x - 1

    The weakest link in any oil viscosity estimate. Beggs and Robinson
    themselves reported an average error around 115% on dead oil against
    their own data set; it is used because a measurement is usually absent,
    not because it is good.
    """
    if temperature_f <= 0.0:
        raise ValueError("Beggs-Robinson needs temperature in degF above zero, "
                         "got %r" % (temperature_f,))
    _check(1.0, temperature_f, api, 1.0)
    z = 3.0324 - 0.02023 * api
    x = 10.0 ** z * temperature_f ** -1.163
    return 10.0 ** x - 1.0


def saturated_oil_viscosity(rs_scf_stb: float, temperature_f: float,
                            api: float) -> float:
    """Beggs-Robinson live-oil viscosity at or below the bubble point, cp."""
    if rs_scf_stb < 0.0:
        raise ValueError("solution gas cannot be negative, got %r" % (rs_scf_stb,))
    mu_dead = dead_oil_viscosity(temperature_f, api)
    a = 10.715 * (rs_scf_stb + 100.0) ** -0.515
    b = 5.44 * (rs_scf_stb + 150.0) ** -0.338
    return a * mu_dead ** b


def undersaturated_oil_viscosity(pressure_psia: float, bubble_point_psia: float,
                                 viscosity_at_pb: float) -> float:
    """Vazquez-Beggs pressure correction above the bubble point, cp.

    Undersaturated oil gets *more* viscous with pressure, having no more gas
    to dissolve -- the opposite of its behaviour below the bubble point. A
    model that misses the turn predicts the viscosity minimum in the wrong
    place, and the viscosity minimum is where the well produces best.
    """
    if pressure_psia < bubble_point_psia:
        raise ValueError(
            "this correction is for pressures above the bubble point; %r is "
            "below %r, where `saturated_oil_viscosity` applies"
            % (pressure_psia, bubble_point_psia))
    if viscosity_at_pb <= 0.0:
        raise ValueError("viscosity at the bubble point must be positive")
    m = 2.6 * pressure_psia ** 1.187 * math.exp(
        -11.513 - 8.98e-5 * pressure_psia)
    return viscosity_at_pb * (pressure_psia / max(bubble_point_psia, 1.0)) ** m


# ================================================================== bundle ==


@dataclass
class OilPVT:
    """One oil's properties, evaluated consistently across pressure.

    The point of bundling is the bubble point. Rs, Bo and viscosity all change
    behaviour there, and each of the free functions above is valid on one side
    only; calling them independently is how a saturated correlation ends up
    applied to undersaturated oil. This evaluates the right branch and makes
    the bubble point a derived quantity rather than a guess, unless the caller
    supplies a measured one.

    `rs_multiplier` and `bo_multiplier` are the calibration handles: one
    laboratory point is usually all that exists, and a single multiplier is
    the honest amount of correction it supports.
    """

    api: float
    gas_gravity: float
    temperature_f: float
    correlation: str = STANDING
    bubble_point_psia: float = 0.0        # 0 = derive from rsb
    rsb_scf_stb: float = 0.0              # solution gas at the bubble point
    separator_temperature_f: float = 100.0
    separator_psia: float = 114.7
    rs_multiplier: float = 1.0
    bo_multiplier: float = 1.0

    def __post_init__(self):
        if self.correlation not in CITATIONS:
            raise ValueError("unknown correlation %r; choose from %s"
                             % (self.correlation, sorted(CITATIONS)))
        _check(1.0, self.temperature_f, self.api, self.gas_gravity)
        if self.rsb_scf_stb <= 0.0 and self.bubble_point_psia <= 0.0:
            raise ValueError(
                "an oil needs either its solution gas at the bubble point or "
                "the bubble point itself; neither was given")
        if self.bubble_point_psia <= 0.0:
            self.bubble_point_psia = bubble_point_standing(
                self.rsb_scf_stb, self.temperature_f, self.api, self.gas_gravity)
        if self.rsb_scf_stb <= 0.0:
            self.rsb_scf_stb = self._raw_rs(self.bubble_point_psia)

    @property
    def citation(self) -> str:
        return CITATIONS[self.correlation]

    def _raw_rs(self, pressure_psia: float) -> float:
        if self.correlation == STANDING:
            return solution_gas_standing(pressure_psia, self.temperature_f,
                                         self.api, self.gas_gravity)
        return solution_gas_vazquez_beggs(
            pressure_psia, self.temperature_f, self.api, self.gas_gravity,
            self.separator_temperature_f, self.separator_psia)

    def rs(self, pressure_psia: float) -> float:
        """Solution gas, scf/STB, capped at the bubble-point value."""
        if pressure_psia >= self.bubble_point_psia:
            return self.rsb_scf_stb * self.rs_multiplier
        return min(self._raw_rs(pressure_psia),
                   self.rsb_scf_stb) * self.rs_multiplier

    def _raw_bo(self, rs: float) -> float:
        if self.correlation == STANDING:
            return oil_fvf_standing(rs, self.temperature_f, self.api,
                                    self.gas_gravity)
        return oil_fvf_vazquez_beggs(
            rs, self.temperature_f, self.api, self.gas_gravity,
            self.separator_temperature_f, self.separator_psia)

    def bo(self, pressure_psia: float) -> float:
        """Oil volume factor, bbl/STB, on the correct side of the bubble point.

        Below pb, Bo falls with pressure because gas is leaving solution.
        Above it, Bo falls with *rising* pressure because the oil is simply
        being compressed. Bo therefore peaks at the bubble point, and a model
        that applies the saturated branch throughout misses the peak -- which
        is exactly the pressure a depleting reservoir passes through.
        """
        if pressure_psia <= self.bubble_point_psia:
            return self._raw_bo(self.rs(pressure_psia) / self.rs_multiplier) \
                * self.bo_multiplier
        bob = self._raw_bo(self.rsb_scf_stb) * self.bo_multiplier
        co = oil_compressibility(
            pressure_psia, self.rsb_scf_stb, self.temperature_f, self.api,
            self.gas_gravity, self.separator_temperature_f, self.separator_psia)
        return bob * math.exp(-co * (pressure_psia - self.bubble_point_psia))

    def viscosity(self, pressure_psia: float) -> float:
        """Oil viscosity, cp, on the correct side of the bubble point."""
        rs_at = self.rs(min(pressure_psia, self.bubble_point_psia))
        saturated = saturated_oil_viscosity(rs_at, self.temperature_f, self.api)
        if pressure_psia <= self.bubble_point_psia:
            return saturated
        return undersaturated_oil_viscosity(
            pressure_psia, self.bubble_point_psia, saturated)

    def calibrate_to_lab(self, pressure_psia: float, rs_measured: float = 0.0,
                         bo_measured: float = 0.0) -> dict:
        """Scale the correlation to one measured PVT point.

        Returns what changed and by how much, because the size of the
        correction is the finding. A 3% multiplier means the correlation suits
        this crude; a 30% one means it does not, and the rest of the curve
        should not be trusted just because one point now matches.
        """
        applied = {}
        if rs_measured > 0.0:
            raw = self._raw_rs(min(pressure_psia, self.bubble_point_psia))
            raw = min(raw, self.rsb_scf_stb)
            if raw <= 0.0:
                raise ValueError("the correlation gives no solution gas at %r psia"
                                 % (pressure_psia,))
            self.rs_multiplier = rs_measured / raw
            applied["rs_multiplier"] = self.rs_multiplier
        if bo_measured > 0.0:
            self.bo_multiplier = 1.0
            raw = self.bo(pressure_psia)
            if raw <= 0.0:
                raise ValueError("the correlation gives no volume factor at %r psia"
                                 % (pressure_psia,))
            self.bo_multiplier = bo_measured / raw
            applied["bo_multiplier"] = self.bo_multiplier

        # Iterated over a snapshot, because the loop adds a key.
        for name, value in list(applied.items()):
            if abs(value - 1.0) > 0.20:
                applied.setdefault("warnings", []).append(
                    "%s is %.2f -- the correlation is more than 20%% out on "
                    "this crude. Matching one point does not make the rest of "
                    "the curve right; use a full laboratory PVT table."
                    % (name, value))
        return applied
