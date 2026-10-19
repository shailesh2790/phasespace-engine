"""Well performance: inflow, outflow, and the nodal-analysis operating point.

This is wedge A from PRD Section 3 -- gas lift and liquid loading -- built on
the same deterministic-first principle as the network engine: every answer
here is a published correlation that an engineer can check by hand, and the
surrogate is trained against it rather than replacing it.

The three pieces of a nodal analysis:

  IPR   what the reservoir will give at a given bottomhole flowing pressure
  VLP   what bottomhole pressure the tubing needs to lift that rate to surface
  node  the rate where those two agree -- the well's actual operating point

IPR is cheap and analytic. VLP is not: it needs a pressure traverse that
marches down the tubing in segments, re-evaluating fluid properties and a
multiphase holdup correlation at every step, because gas comes out of
solution as pressure falls and the mixture density changes all the way down.

Correlations implemented:
  Vogel (1968)                  saturated-oil IPR
  Fetkovich (1973)              backpressure-form oil IPR
  Rawlins-Schellhardt (1935)    gas-well backpressure IPR
  Beggs-Brill (1973) + Payne    multiphase pressure traverse, any inclination
  Hagedorn-Brown (1965)         multiphase traverse, vertical, with Griffith
  Turner (1969) / Coleman (1991) critical gas rate for liquid unloading
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fluids import GasProperties, LiquidProperties
from .units import RANKINE_OFFSET

GRAVITY_FT_S2 = 32.174
BBL_TO_FT3 = 5.614583
SECONDS_PER_DAY = 86400.0

# Beggs-Brill flow regimes
SEGREGATED = "segregated"
INTERMITTENT = "intermittent"
DISTRIBUTED = "distributed"
TRANSITION = "transition"


# ============================================================== inflow ======


class InflowPerformance(object):
    """Reservoir deliverability: rate as a function of flowing bottomhole
    pressure. Rate units are STB/day for oil wells, scf/day for gas wells."""

    def rate(self, p_wf: float) -> float:
        raise NotImplementedError

    def absolute_open_flow(self) -> float:
        """Rate at zero bottomhole pressure -- the theoretical maximum."""
        return self.rate(0.0)


@dataclass
class ProductivityIndexIPR(InflowPerformance):
    """Straight-line IPR, q = J * (Pr - Pwf).

    Valid only above the bubble point, where the oil is single-phase and
    undersaturated. Below it, free gas in the reservoir reduces oil mobility
    and the real IPR curves over -- use `VogelIPR` or `CompositeIPR` there.
    """

    reservoir_pressure_psia: float
    productivity_index: float  # STB/day per psi

    def rate(self, p_wf: float) -> float:
        return max(self.productivity_index * (self.reservoir_pressure_psia - p_wf), 0.0)


@dataclass
class VogelIPR(InflowPerformance):
    """Vogel's saturated-oil IPR.

        q / q_max = 1 - 0.2 (Pwf/Pr) - 0.8 (Pwf/Pr)^2
    """

    reservoir_pressure_psia: float
    max_rate: float  # q at Pwf = 0, STB/day

    def rate(self, p_wf: float) -> float:
        if self.reservoir_pressure_psia <= 0.0:
            return 0.0
        x = max(min(p_wf / self.reservoir_pressure_psia, 1.0), 0.0)
        return max(self.max_rate * (1.0 - 0.2 * x - 0.8 * x * x), 0.0)


@dataclass
class CompositeIPR(InflowPerformance):
    """Straight-line above the bubble point, Vogel below it.

    This is the case that actually occurs in most producing oil wells, and
    the one a pure Vogel or pure PI model gets wrong at exactly the drawdown
    an engineer is trying to evaluate. The two segments are joined so that
    both rate and slope are continuous at the bubble point -- a kink there
    would put a false local optimum into any gas-lift optimisation run on
    top of this.
    """

    reservoir_pressure_psia: float
    bubble_point_psia: float
    productivity_index: float  # STB/day/psi, from the undersaturated region

    def _rate_at_bubble_point(self) -> float:
        return self.productivity_index * max(
            self.reservoir_pressure_psia - self.bubble_point_psia, 0.0)

    def rate(self, p_wf: float) -> float:
        p_b = self.bubble_point_psia
        q_b = self._rate_at_bubble_point()
        if p_wf >= p_b:
            return max(self.productivity_index * (self.reservoir_pressure_psia - p_wf), 0.0)
        # Vogel's curve, scaled so its slope at Pwf = Pb matches J. The
        # standard result is q_max_vogel = J * Pb / 1.8 beyond q_b.
        x = max(min(p_wf / p_b, 1.0), 0.0)
        return q_b + (self.productivity_index * p_b / 1.8) * (1.0 - 0.2 * x - 0.8 * x * x)


@dataclass
class FetkovichIPR(InflowPerformance):
    """q = C * (Pr^2 - Pwf^2)^n, the backpressure form for oil wells."""

    reservoir_pressure_psia: float
    c: float
    n: float = 1.0

    def rate(self, p_wf: float) -> float:
        dp2 = self.reservoir_pressure_psia ** 2 - max(p_wf, 0.0) ** 2
        if dp2 <= 0.0:
            return 0.0
        return self.c * dp2 ** self.n


@dataclass
class GasBackpressureIPR(InflowPerformance):
    """Rawlins-Schellhardt gas-well deliverability, q = C (Pr^2 - Pwf^2)^n.

    n runs from 1.0 (fully laminar) to 0.5 (fully turbulent); most real gas
    wells test out between 0.7 and 0.9. Rate is scf/day.
    """

    reservoir_pressure_psia: float
    c: float
    n: float = 0.85

    def rate(self, p_wf: float) -> float:
        dp2 = self.reservoir_pressure_psia ** 2 - max(p_wf, 0.0) ** 2
        if dp2 <= 0.0:
            return 0.0
        return self.c * dp2 ** self.n


# ============================================================= outflow ======


@dataclass
class Tubing:
    """The flow path from the perforations to the wellhead.

    `inclination_deg` is measured from horizontal: 90 is a vertical well,
    0 a horizontal lateral. Beggs-Brill was developed on an inclined test
    loop precisely so it could cover the whole range, which is why it is the
    default correlation here rather than a vertical-only one.
    """

    depth_ft: float
    inner_diameter_in: float
    roughness_in: float = 0.0006
    inclination_deg: float = 90.0
    wellhead_temperature_f: float = 100.0
    bottomhole_temperature_f: float = 180.0

    def __post_init__(self):
        if self.inner_diameter_in <= 0.0:
            raise ValueError("tubing ID must be positive, got %r" % (self.inner_diameter_in,))
        if self.depth_ft <= 0.0:
            raise ValueError("tubing depth must be positive, got %r" % (self.depth_ft,))

    @property
    def area_ft2(self) -> float:
        return math.pi * (self.inner_diameter_in / 12.0) ** 2 / 4.0

    @property
    def diameter_ft(self) -> float:
        return self.inner_diameter_in / 12.0

    def temperature_f_at(self, depth_ft: float) -> float:
        """Linear geothermal profile between wellhead and total depth."""
        fraction = max(min(depth_ft / self.depth_ft, 1.0), 0.0)
        return (self.wellhead_temperature_f
                + fraction * (self.bottomhole_temperature_f - self.wellhead_temperature_f))


@dataclass
class ProductionStream:
    """What the well is producing, at stock-tank conditions."""

    oil_rate_stb_d: float = 0.0
    water_rate_stb_d: float = 0.0
    gas_rate_scf_d: float = 0.0
    oil: LiquidProperties = field(default_factory=LiquidProperties)
    gas: GasProperties = field(default_factory=GasProperties)

    @property
    def liquid_rate_stb_d(self) -> float:
        return self.oil_rate_stb_d + self.water_rate_stb_d

    @property
    def water_cut(self) -> float:
        total = self.liquid_rate_stb_d
        return self.water_rate_stb_d / total if total > 0.0 else 0.0

    @property
    def gas_liquid_ratio(self) -> float:
        total = self.liquid_rate_stb_d
        return self.gas_rate_scf_d / total if total > 0.0 else 0.0


def _no_slip_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Darcy friction factor for the no-slip mixture (Swamee-Jain form)."""
    if reynolds < 2100.0:
        return 64.0 / max(reynolds, 1e-9)
    denom = math.log10(relative_roughness / 3.7 + 5.74 / (reynolds ** 0.9))
    return 0.25 / (denom ** 2)


def beggs_brill_gradient(
    p_psia: float,
    t_f: float,
    stream: ProductionStream,
    tubing: Tubing,
) -> dict:
    """Beggs-Brill pressure gradient (psi/ft) at one point in the tubing.

    Returns the gradient split into its elevation and friction parts, plus
    the holdup and flow regime -- the split matters, because in a loaded-up
    well almost all of the gradient is elevation (a static liquid column)
    while in a high-rate well friction dominates, and an engineer reading a
    single lumped number cannot tell those apart.

    The Payne et al. (1979) correction factors are applied to holdup: the
    original Beggs-Brill correlation was regressed on air-water data in a
    small-diameter loop and is known to over-predict holdup, which shows up
    as a systematically pessimistic bottomhole pressure.
    """
    theta = math.radians(tubing.inclination_deg)
    area = tubing.area_ft2
    d_ft = tubing.diameter_ft

    oil = stream.oil
    gas = stream.gas
    t_r = t_f + RANKINE_OFFSET
    oil_at_t = LiquidProperties(
        api_gravity=oil.api_gravity,
        water_cut=stream.water_cut,
        water_specific_gravity=oil.water_specific_gravity,
        temperature_r=t_r,
        gas_gravity=oil.gas_gravity,
        bubble_point_psia=oil.bubble_point_psia,
    )
    gas_at_t = GasProperties(
        gravity=gas.gravity, temperature_r=t_r,
        y_co2=gas.y_co2, y_h2s=gas.y_h2s, y_n2=gas.y_n2, fixed_z=gas.fixed_z,
    )

    # In-situ rates. Gas still in solution is part of the liquid, not free
    # gas -- treating produced GOR as free gas at depth is the single most
    # common way to get a wildly wrong bottomhole pressure.
    rs = oil_at_t.solution_gor_scf_stb(p_psia)
    free_gas_scf_d = max(stream.gas_rate_scf_d - rs * stream.oil_rate_stb_d, 0.0)
    bg = gas_at_t.formation_volume_factor(max(p_psia, 1.0))

    q_liquid_ft3_s = stream.liquid_rate_stb_d * BBL_TO_FT3 / SECONDS_PER_DAY
    q_gas_ft3_s = free_gas_scf_d * bg / SECONDS_PER_DAY

    v_sl = q_liquid_ft3_s / area
    v_sg = q_gas_ft3_s / area
    v_m = v_sl + v_sg

    if v_m < 1e-12:
        rho_l = oil_at_t.density_lbft3(p_psia)
        return {
            "gradient_psi_ft": rho_l / 144.0,
            "elevation_psi_ft": rho_l / 144.0,
            "friction_psi_ft": 0.0,
            "holdup": 1.0,
            "regime": SEGREGATED,
            "mixture_velocity_ft_s": 0.0,
        }

    lambda_l = v_sl / v_m
    rho_l = oil_at_t.density_lbft3(p_psia)
    rho_g = gas_at_t.density_lbft3(max(p_psia, 1.0))
    mu_l = oil_at_t.viscosity_cp(p_psia)
    mu_g = gas_at_t.viscosity_cp(max(p_psia, 1.0))
    surface_tension = 30.0  # dyne/cm, a representative oil/gas value

    n_fr = v_m * v_m / (GRAVITY_FT_S2 * d_ft)
    n_lv = 1.938 * v_sl * (rho_l / max(surface_tension, 1e-6)) ** 0.25

    # Flow-regime boundaries
    l1 = 316.0 * lambda_l ** 0.302
    l2 = 0.0009252 * lambda_l ** -2.4684
    l3 = 0.10 * lambda_l ** -1.4516
    l4 = 0.5 * lambda_l ** -6.738

    if (lambda_l < 0.01 and n_fr < l1) or (lambda_l >= 0.01 and n_fr < l2):
        regime = SEGREGATED
    elif lambda_l >= 0.01 and l2 <= n_fr <= l3:
        regime = TRANSITION
    elif (0.01 <= lambda_l < 0.4 and l3 < n_fr <= l1) or (lambda_l >= 0.4 and l3 < n_fr <= l4):
        regime = INTERMITTENT
    else:
        regime = DISTRIBUTED

    def holdup_for(reg):
        coefficients = {
            SEGREGATED: (0.98, 0.4846, 0.0868),
            INTERMITTENT: (0.845, 0.5351, 0.0173),
            DISTRIBUTED: (1.065, 0.5824, 0.0609),
        }[reg]
        a, b, c = coefficients
        h_l0 = a * lambda_l ** b / max(n_fr, 1e-12) ** c
        h_l0 = max(h_l0, lambda_l)  # holdup can never be below no-slip holdup

        if reg == DISTRIBUTED and theta >= 0.0:
            correction = 1.0
        else:
            if theta >= 0.0:
                d_, e_, f_, g_ = {
                    SEGREGATED: (0.011, -3.768, 3.539, -1.614),
                    INTERMITTENT: (2.96, 0.305, -0.4473, 0.0978),
                    DISTRIBUTED: (1.0, 0.0, 0.0, 0.0),
                }[reg]
            else:
                d_, e_, f_, g_ = 4.70, -0.3692, 0.1244, -0.5056
            inner = (d_ * lambda_l ** e_ * max(n_lv, 1e-12) ** f_
                     * max(n_fr, 1e-12) ** g_)
            c_coef = max((1.0 - lambda_l) * math.log(max(inner, 1e-12)), 0.0)
            psi_angle = 1.8 * theta
            correction = (1.0 + c_coef
                          * (math.sin(psi_angle) - 0.333 * math.sin(psi_angle) ** 3))
        h_l = h_l0 * correction
        # Payne correction: the original correlation over-predicts holdup.
        h_l *= 0.924 if theta >= 0.0 else 0.685
        return max(min(h_l, 1.0), lambda_l)

    if regime == TRANSITION:
        # Interpolate between segregated and intermittent rather than
        # snapping between them: a discontinuity in holdup here puts a step
        # in the pressure traverse that shows up as a false kink in the VLP
        # curve, which then traps the nodal-analysis root find.
        weight = (l3 - n_fr) / max(l3 - l2, 1e-12)
        weight = max(min(weight, 1.0), 0.0)
        h_l = weight * holdup_for(SEGREGATED) + (1.0 - weight) * holdup_for(INTERMITTENT)
    else:
        h_l = holdup_for(regime)

    rho_slip = rho_l * h_l + rho_g * (1.0 - h_l)
    rho_ns = rho_l * lambda_l + rho_g * (1.0 - lambda_l)
    mu_ns = mu_l * lambda_l + mu_g * (1.0 - lambda_l)

    reynolds = 1488.0 * rho_ns * v_m * d_ft / max(mu_ns, 1e-9)
    f_ns = _no_slip_friction_factor(reynolds, tubing.roughness_in / tubing.inner_diameter_in)

    y = lambda_l / max(h_l * h_l, 1e-12)
    if 1.0 < y < 1.2:
        s = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(max(y, 1e-12))
        denom = (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y ** 2 + 0.01853 * ln_y ** 4)
        s = ln_y / denom if abs(denom) > 1e-12 else 0.0
    f_tp = f_ns * math.exp(max(min(s, 50.0), -50.0))

    elevation_psi_ft = rho_slip * math.sin(theta) / 144.0
    friction_psi_ft = f_tp * rho_ns * v_m * v_m / (2.0 * GRAVITY_FT_S2 * d_ft) / 144.0

    # Acceleration (kinetic-energy) term. Negligible except near the surface
    # of a high-rate gas well, where it is not.
    e_k = rho_slip * v_m * v_sg / (GRAVITY_FT_S2 * max(p_psia, 1.0) * 144.0)
    e_k = min(e_k, 0.9)

    total = (elevation_psi_ft + friction_psi_ft) / (1.0 - e_k)
    return {
        "gradient_psi_ft": total,
        "elevation_psi_ft": elevation_psi_ft,
        "friction_psi_ft": friction_psi_ft,
        "holdup": h_l,
        "regime": regime,
        "mixture_velocity_ft_s": v_m,
        "superficial_gas_velocity_ft_s": v_sg,
        "superficial_liquid_velocity_ft_s": v_sl,
        "slip_density_lb_ft3": rho_slip,
    }


def hagedorn_brown_gradient(
    p_psia: float,
    t_f: float,
    stream: ProductionStream,
    tubing: Tubing,
) -> dict:
    """Hagedorn-Brown pressure gradient (psi/ft), with the Griffith bubble
    correction.

    Kept alongside Beggs-Brill because the two disagree in exactly the
    conditions that matter for gas lift: Hagedorn-Brown was built on
    vertical wells with high liquid loading and tends to be the better match
    there, while Beggs-Brill handles deviation that Hagedorn-Brown was never
    intended to cover. Running both and looking at the spread is a cheap,
    honest uncertainty estimate on the VLP curve -- which is the same
    argument as the confidence tier in PRD Section 6, applied to a
    correlation instead of a neural network.

    The published correlation reads holdup off three charts; those are
    reproduced here by the standard curve fits.

    Known limit, worth stating rather than hiding: Hagedorn-Brown was
    regressed on a 1500 ft test well at moderate gas-liquid ratios, and it
    under-predicts pressure drop badly in high-GLR mist flow. Against
    Beggs-Brill on a 2.441 in string it agrees within ~7% at 200 scf/STB and
    diverges to ~45% by 2000 scf/STB, with Beggs-Brill the more trustworthy
    of the two up there. Use the spread as a signal that the VLP is
    uncertain, not as two equally good answers to average.
    """
    theta = math.radians(tubing.inclination_deg)
    area = tubing.area_ft2
    d_ft = tubing.diameter_ft
    t_r = t_f + RANKINE_OFFSET

    oil = stream.oil
    oil_at_t = LiquidProperties(
        api_gravity=oil.api_gravity, water_cut=stream.water_cut,
        water_specific_gravity=oil.water_specific_gravity, temperature_r=t_r,
        gas_gravity=oil.gas_gravity, bubble_point_psia=oil.bubble_point_psia)
    gas = stream.gas
    gas_at_t = GasProperties(gravity=gas.gravity, temperature_r=t_r, y_co2=gas.y_co2,
                             y_h2s=gas.y_h2s, y_n2=gas.y_n2, fixed_z=gas.fixed_z)

    rs = oil_at_t.solution_gor_scf_stb(p_psia)
    free_gas = max(stream.gas_rate_scf_d - rs * stream.oil_rate_stb_d, 0.0)
    bg = gas_at_t.formation_volume_factor(max(p_psia, 1.0))

    v_sl = stream.liquid_rate_stb_d * BBL_TO_FT3 / SECONDS_PER_DAY / area
    v_sg = free_gas * bg / SECONDS_PER_DAY / area
    v_m = v_sl + v_sg
    if v_m < 1e-12:
        rho_l = oil_at_t.density_lbft3(p_psia)
        return {"gradient_psi_ft": rho_l / 144.0, "elevation_psi_ft": rho_l / 144.0,
                "friction_psi_ft": 0.0, "holdup": 1.0, "regime": "static",
                "mixture_velocity_ft_s": 0.0}

    rho_l = oil_at_t.density_lbft3(p_psia)
    rho_g = gas_at_t.density_lbft3(max(p_psia, 1.0))
    mu_l = oil_at_t.viscosity_cp(p_psia)
    sigma = 30.0

    lambda_g = v_sg / v_m

    # Griffith bubble-flow criterion. Below it, Hagedorn-Brown's holdup
    # correlation is outside its data and Griffith's bubble-flow model is
    # used instead -- without this switch the correlation returns holdups
    # below the no-slip value at low gas rates, which is unphysical.
    l_b = max(1.071 - 0.2218 * v_m * v_m / d_ft, 0.13)
    if lambda_g < l_b:
        v_s = 0.8  # slip velocity, ft/s
        h_l = 1.0 - 0.5 * (1.0 + v_m / v_s
                           - math.sqrt(max((1.0 + v_m / v_s) ** 2 - 4.0 * v_sg / v_s, 0.0)))
        h_l = max(min(h_l, 1.0), 1e-6)
        regime = "bubble (Griffith)"
        mu_for_re = mu_l
        rho_for_friction = rho_l
        velocity_for_friction = v_sl / h_l
    else:
        # Dimensionless groups
        n_lv = 1.938 * v_sl * (rho_l / sigma) ** 0.25
        n_gv = 1.938 * v_sg * (rho_l / sigma) ** 0.25
        n_d = 120.872 * d_ft * (rho_l / sigma) ** 0.5
        n_l = 0.15726 * mu_l * (1.0 / (rho_l * sigma ** 3)) ** 0.25

        # Chart 1: CNL(NL)
        cnl = 0.0019 + 0.0322 * n_l - 0.6642 * n_l ** 2 + 4.9951 * n_l ** 3
        cnl = max(min(cnl, 0.095), 0.0019)

        # Chart 2: HL/psi as a function of the holdup group
        h_group = (n_lv / n_gv ** 0.575) * (p_psia / 14.7) ** 0.1 * (cnl / max(n_d, 1e-9))
        h_over_psi = (0.0047 + 1123.32 * h_group + 729489.64 * h_group ** 2) / \
                     (1.0 + 1097.1566 * h_group + 722153.97 * h_group ** 2)

        # Chart 3: secondary correction psi
        psi_group = n_gv * n_l ** 0.38 / max(n_d, 1e-9) ** 2.14
        if psi_group <= 0.01:
            psi = 1.0
        elif psi_group >= 0.09:
            psi = 1.0
        else:
            psi = (0.91163 - 4.82176 * psi_group + 1232.25 * psi_group ** 2
                   - 22253.6 * psi_group ** 3 + 116174.3 * psi_group ** 4)
        psi = max(psi, 1.0)  # the correction only ever increases holdup

        h_l = max(min(h_over_psi * psi, 1.0), 1e-6)

        # The published procedure's own check, and it is not optional: the
        # chart fit can return a holdup *below* the no-slip holdup, which
        # would mean the gas is travelling slower than the liquid in upward
        # flow. Without this floor the correlation reported a holdup of 0.42
        # where no-slip alone was 0.83, which understated bottomhole pressure
        # by roughly 1500 psi on an 8000 ft well -- a difference big enough
        # to turn a well that flows into one the model says is dead.
        lambda_l = 1.0 - lambda_g
        h_l = max(h_l, lambda_l)
        regime = "hagedorn-brown"
        mu_g = gas_at_t.viscosity_cp(max(p_psia, 1.0))
        mu_for_re = mu_l ** h_l * mu_g ** (1.0 - h_l)
        rho_for_friction = (rho_l * (1.0 - lambda_g) + rho_g * lambda_g)
        velocity_for_friction = v_m

    rho_slip = rho_l * h_l + rho_g * (1.0 - h_l)
    reynolds = 1488.0 * rho_for_friction * velocity_for_friction * d_ft / max(mu_for_re, 1e-9)
    f = _no_slip_friction_factor(reynolds, tubing.roughness_in / tubing.inner_diameter_in)

    elevation_psi_ft = rho_slip * math.sin(theta) / 144.0
    # Hagedorn-Brown's friction term uses the no-slip mass flux over the
    # slip density, which is not the same grouping Beggs-Brill uses.
    mass_flux = (rho_l * v_sl + rho_g * v_sg)
    friction_psi_ft = (f * mass_flux ** 2
                       / (2.0 * GRAVITY_FT_S2 * d_ft * max(rho_slip, 1e-9)) / 144.0)

    return {
        "gradient_psi_ft": elevation_psi_ft + friction_psi_ft,
        "elevation_psi_ft": elevation_psi_ft,
        "friction_psi_ft": friction_psi_ft,
        "holdup": h_l,
        "regime": regime,
        "mixture_velocity_ft_s": v_m,
        "superficial_gas_velocity_ft_s": v_sg,
        "superficial_liquid_velocity_ft_s": v_sl,
        "slip_density_lb_ft3": rho_slip,
    }


BEGGS_BRILL = "beggs_brill"
HAGEDORN_BROWN = "hagedorn_brown"

_GRADIENT_MODELS = {
    BEGGS_BRILL: beggs_brill_gradient,
    HAGEDORN_BROWN: hagedorn_brown_gradient,
}


@dataclass
class TraverseResult:
    depths_ft: list
    pressures_psia: list
    holdups: list
    regimes: list
    elevation_fraction: float
    friction_fraction: float

    @property
    def bottomhole_pressure_psia(self) -> float:
        return self.pressures_psia[-1]

    @property
    def wellhead_pressure_psia(self) -> float:
        return self.pressures_psia[0]


def pressure_traverse(
    wellhead_pressure_psia: float,
    stream: ProductionStream,
    tubing: Tubing,
    correlation: str = BEGGS_BRILL,
    segments: int = 40,
    gravity_multiplier: float = 1.0,
    friction_multiplier: float = 1.0,
) -> TraverseResult:
    """March from the wellhead down to total depth, integrating the gradient.

    Marching in segments rather than applying one average gradient over the
    whole string is not refinement for its own sake: solution gas comes out
    as pressure falls, so mixture density near the surface can be a third of
    what it is at the perforations. A single-step calculation gets the
    bottomhole pressure wrong by hundreds of psi in a gassy well.

    Each segment uses the midpoint pressure (a predictor-corrector step)
    rather than its inlet pressure, which is second-order accurate and lets
    40 segments do what a first-order scheme would need several hundred for.

    **The two multipliers are the calibration handles**, and they are separate
    on purpose. A multiphase correlation fitted on somebody else's test loop
    is routinely 10-20% out on a particular well, but not uniformly: the
    gravity term dominates at low rate and the friction term at high rate, so
    one lumped correction cannot fix both ends of a well's operating range and
    will trade one off against the other. Keeping them apart is also what
    makes them separately identifiable from well tests -- but only if those
    tests span a range of rates, which `wellcalibration` checks before
    fitting rather than after.

    Both default to 1.0, so an uncalibrated well is the raw correlation and
    nothing here changes silently.
    """
    if correlation not in _GRADIENT_MODELS:
        raise ValueError("unknown correlation %r (expected one of %s)"
                         % (correlation, ", ".join(sorted(_GRADIENT_MODELS))))
    if gravity_multiplier <= 0.0 or friction_multiplier <= 0.0:
        raise ValueError(
            "pressure-drop multipliers must be positive, got gravity %r and "
            "friction %r" % (gravity_multiplier, friction_multiplier))
    gradient_fn = _GRADIENT_MODELS[correlation]

    def scaled(result: dict) -> float:
        """Recompose the gradient with the two components scaled.

        The ratio of the reported total to the sum of its two parts carries
        whatever else the correlation did -- Beggs-Brill divides by (1 - E_k)
        for acceleration -- so applying it back preserves that term instead of
        quietly dropping it when a multiplier is in use.
        """
        elevation = result["elevation_psi_ft"]
        friction = result["friction_psi_ft"]
        parts = elevation + friction
        if parts <= 0.0:
            return result["gradient_psi_ft"]
        extra = result["gradient_psi_ft"] / parts
        return (elevation * gravity_multiplier
                + friction * friction_multiplier) * extra

    dz = tubing.depth_ft / segments
    depth = 0.0
    p = wellhead_pressure_psia

    depths = [0.0]
    pressures = [p]
    holdups = []
    regimes = []
    elevation_total = 0.0
    friction_total = 0.0

    for _ in range(segments):
        t_f = tubing.temperature_f_at(depth + 0.5 * dz)
        first = gradient_fn(p, t_f, stream, tubing)
        p_mid = p + scaled(first) * dz * 0.5
        p_mid = max(p_mid, 1.0)
        second = gradient_fn(p_mid, t_f, stream, tubing)

        p = p + scaled(second) * dz
        p = max(p, 1.0)
        depth += dz

        depths.append(depth)
        pressures.append(p)
        holdups.append(second["holdup"])
        regimes.append(second["regime"])
        elevation_total += second["elevation_psi_ft"] * gravity_multiplier * dz
        friction_total += second["friction_psi_ft"] * friction_multiplier * dz

    total = elevation_total + friction_total
    return TraverseResult(
        depths_ft=depths,
        pressures_psia=pressures,
        holdups=holdups,
        regimes=regimes,
        elevation_fraction=elevation_total / total if total > 0.0 else 0.0,
        friction_fraction=friction_total / total if total > 0.0 else 0.0,
    )


# ====================================================== nodal analysis ======


@dataclass
class NodalResult:
    oil_rate_stb_d: float
    liquid_rate_stb_d: float
    bottomhole_pressure_psia: float
    converged: bool
    ipr_curve: list = field(default_factory=list)   # [(rate, p_wf), ...]
    vlp_curve: list = field(default_factory=list)   # [(rate, p_wf), ...]
    notes: list = field(default_factory=list)


def nodal_analysis(
    ipr: InflowPerformance,
    tubing: Tubing,
    wellhead_pressure_psia: float,
    gas_oil_ratio_scf_stb: float,
    water_cut: float = 0.0,
    oil: LiquidProperties = None,
    gas: GasProperties = None,
    correlation: str = BEGGS_BRILL,
    max_rate_stb_d: float = 0.0,
    points: int = 24,
    segments: int = 30,
    gravity_multiplier: float = 1.0,
    friction_multiplier: float = 1.0,
) -> NodalResult:
    """Find the operating point where inflow and outflow agree.

    Solved by building both curves on a rate grid and bracketing the sign
    change of (VLP - IPR), then bisecting. Newton is not used here on
    purpose: the VLP curve of a gassy well is J-shaped -- it turns back up
    at low rate as the well starts loading with liquid -- so the difference
    function can have two roots, and a derivative-based method will happily
    converge on the unstable left-hand one. That root is a well that dies.
    Bracketing from the high-rate side finds the stable operating point,
    which is the one the well actually produces at.
    """
    oil = oil or LiquidProperties()
    gas = gas or GasProperties()
    notes = []

    aof = ipr.absolute_open_flow()
    q_max = max_rate_stb_d if max_rate_stb_d > 0.0 else aof
    if q_max <= 0.0:
        return NodalResult(0.0, 0.0, 0.0, False, notes=["IPR gives no inflow at all"])

    def build_stream(q_oil):
        q_water = q_oil * water_cut / (1.0 - water_cut) if water_cut < 1.0 else 0.0
        return ProductionStream(
            oil_rate_stb_d=q_oil,
            water_rate_stb_d=q_water,
            gas_rate_scf_d=q_oil * gas_oil_ratio_scf_stb,
            oil=oil,
            gas=gas,
        )

    def pwf_at(q_oil: float) -> float:
        """Bottomhole pressure the tubing needs to deliver this rate.

        Wrapped so the calibration multipliers reach every traverse in the
        solve. They were threaded through three separate call sites first,
        and the bisection's bracket check uses a fourth -- a VLP evaluated
        with different multipliers from the curve it is bisecting would
        converge on a point that is on neither.
        """
        return pressure_traverse(
            wellhead_pressure_psia, build_stream(q_oil), tubing,
            correlation=correlation, segments=segments,
            gravity_multiplier=gravity_multiplier,
            friction_multiplier=friction_multiplier
        ).bottomhole_pressure_psia

    rates = [q_max * (i + 1) / points for i in range(points)]
    ipr_curve = []
    vlp_curve = []
    difference = []

    for q in rates:
        p_ipr = _pwf_for_rate(ipr, q, aof)
        p_vlp = pwf_at(q)
        ipr_curve.append((q, p_ipr))
        vlp_curve.append((q, p_vlp))
        difference.append(p_vlp - p_ipr)

    # Walk down from the highest rate to find the first sign change: that is
    # the stable intersection.
    bracket = None
    for i in range(len(rates) - 1, 0, -1):
        if difference[i] * difference[i - 1] <= 0.0:
            bracket = (rates[i - 1], rates[i])
            break

    if bracket is None:
        if difference[0] > 0.0:
            notes.append(
                "the tubing needs more bottomhole pressure than the reservoir can "
                "supply at every rate tried -- this well will not flow naturally")
        else:
            notes.append(
                "no intersection below the rate limit tried; the operating point is "
                "above %.0f STB/day" % q_max)
        return NodalResult(0.0, 0.0, 0.0, False, ipr_curve, vlp_curve, notes)

    low, high = bracket
    for _ in range(60):
        mid = 0.5 * (low + high)
        p_ipr = _pwf_for_rate(ipr, mid, aof)
        p_vlp = pwf_at(mid)
        if (p_vlp - p_ipr) * (
                pwf_at(low) - _pwf_for_rate(ipr, low, aof)) <= 0.0:
            high = mid
        else:
            low = mid
        if abs(high - low) < 1e-6 * q_max:
            break

    q_op = 0.5 * (low + high)
    p_op = _pwf_for_rate(ipr, q_op, aof)
    q_liquid = q_op / (1.0 - water_cut) if water_cut < 1.0 else q_op
    return NodalResult(q_op, q_liquid, p_op, True, ipr_curve, vlp_curve, notes)


def _pwf_for_rate(ipr: InflowPerformance, q: float, aof: float) -> float:
    """Invert an IPR: the bottomhole pressure at which it delivers rate q.

    Inverted numerically rather than analytically so that every IPR form --
    including ones added later -- works without needing its own inverse.
    """
    lo, hi = 0.0, ipr.reservoir_pressure_psia
    if q >= aof:
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ipr.rate(mid) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===================================================== liquid loading =======


def turner_critical_velocity_ft_s(
    p_psia: float,
    t_r: float,
    gas: GasProperties,
    liquid_density_lb_ft3: float = 67.0,
    surface_tension_dyne_cm: float = 60.0,
    adjustment: float = 1.2,
) -> float:
    """Turner's critical gas velocity for lifting the largest stable droplet.

    Turner's own 1969 data needed a +20% adjustment to match field
    observations, which is the `adjustment` default here. Coleman's later
    work found that for wellhead pressures below ~500 psi the unadjusted
    form fits better -- `coleman_critical_velocity_ft_s` is that variant.
    Which one applies is a real judgement call, so both are exposed rather
    than one being silently chosen.
    """
    rho_g = max(gas.density_lbft3(max(p_psia, 1.0)), 1e-9)
    numerator = surface_tension_dyne_cm * (liquid_density_lb_ft3 - rho_g)
    return adjustment * 1.593 * numerator ** 0.25 / rho_g ** 0.5


def coleman_critical_velocity_ft_s(
    p_psia: float,
    t_r: float,
    gas: GasProperties,
    liquid_density_lb_ft3: float = 67.0,
    surface_tension_dyne_cm: float = 60.0,
) -> float:
    """Coleman's unadjusted form, for wellhead pressures below ~500 psi."""
    return turner_critical_velocity_ft_s(
        p_psia, t_r, gas, liquid_density_lb_ft3, surface_tension_dyne_cm, adjustment=1.0)


def critical_gas_rate_scf_d(
    velocity_ft_s: float,
    p_psia: float,
    t_r: float,
    tubing: Tubing,
    gas: GasProperties,
) -> float:
    """Convert a critical velocity to the standard-condition gas rate that
    produces it -- the number an operator actually compares against."""
    bg = gas.formation_volume_factor(max(p_psia, 1.0))
    q_actual_ft3_s = velocity_ft_s * tubing.area_ft2
    return q_actual_ft3_s * SECONDS_PER_DAY / max(bg, 1e-12)


@dataclass
class LoadingCheck:
    critical_velocity_ft_s: float
    critical_rate_scf_d: float
    actual_velocity_ft_s: float
    actual_rate_scf_d: float

    @property
    def is_loading(self) -> bool:
        return self.actual_rate_scf_d < self.critical_rate_scf_d

    @property
    def margin(self) -> float:
        """Actual rate as a multiple of critical. Below 1.0 the well is
        loading up; operators generally want to see 1.2 or better."""
        if self.critical_rate_scf_d <= 0.0:
            return float("inf")
        return self.actual_rate_scf_d / self.critical_rate_scf_d


def liquid_loading_check(
    gas_rate_scf_d: float,
    wellhead_pressure_psia: float,
    tubing: Tubing,
    gas: GasProperties,
    use_coleman: bool = False,
    liquid_density_lb_ft3: float = 67.0,
) -> LoadingCheck:
    """Is this gas well carrying its liquids to surface, or loading up?

    Evaluated at the wellhead, which is where velocity is highest but
    pressure -- and so gas density, and so lifting capacity -- is lowest.
    That combination makes the wellhead the usual first place a well starts
    to load, which is why the check belongs there rather than at depth.
    """
    t_r = tubing.wellhead_temperature_f + RANKINE_OFFSET
    fn = coleman_critical_velocity_ft_s if use_coleman else turner_critical_velocity_ft_s
    v_crit = fn(wellhead_pressure_psia, t_r, gas, liquid_density_lb_ft3=liquid_density_lb_ft3)
    q_crit = critical_gas_rate_scf_d(v_crit, wellhead_pressure_psia, t_r, tubing, gas)

    bg = gas.formation_volume_factor(max(wellhead_pressure_psia, 1.0))
    v_actual = gas_rate_scf_d * bg / SECONDS_PER_DAY / tubing.area_ft2

    return LoadingCheck(
        critical_velocity_ft_s=v_crit,
        critical_rate_scf_d=q_crit,
        actual_velocity_ft_s=v_actual,
        actual_rate_scf_d=gas_rate_scf_d,
    )


# ============================================================ gas lift ======


@dataclass
class GasLiftPoint:
    injection_rate_scf_d: float
    oil_rate_stb_d: float
    bottomhole_pressure_psia: float


def gas_lift_performance_curve(
    ipr: InflowPerformance,
    tubing: Tubing,
    wellhead_pressure_psia: float,
    formation_gor_scf_stb: float,
    injection_rates_scf_d: list,
    water_cut: float = 0.0,
    oil: LiquidProperties = None,
    gas: GasProperties = None,
    correlation: str = BEGGS_BRILL,
) -> list:
    """Oil rate against lift-gas injection rate -- the gas-lift performance
    curve.

    The curve has a maximum. Injecting more gas lightens the column (good)
    but also raises friction (bad), and past the optimum the second effect
    wins and the well makes *less* oil for more gas. Finding that turning
    point is the entire economic question in gas lift, and it is why this
    returns the whole curve rather than a single optimum: the flat region
    around the peak is usually wide, and an operator choosing an injection
    rate cares where the curve starts to fall off, not just where it peaks.
    """
    points = []
    for q_inj in injection_rates_scf_d:
        result = nodal_analysis(
            ipr=ipr,
            tubing=tubing,
            wellhead_pressure_psia=wellhead_pressure_psia,
            gas_oil_ratio_scf_stb=formation_gor_scf_stb,
            water_cut=water_cut,
            oil=oil,
            gas=gas,
            correlation=correlation,
            points=18,
            segments=20,
            max_rate_stb_d=0.0,
        ) if q_inj <= 0.0 else _nodal_with_injection(
            ipr, tubing, wellhead_pressure_psia, formation_gor_scf_stb, q_inj,
            water_cut, oil, gas, correlation)
        points.append(GasLiftPoint(
            injection_rate_scf_d=q_inj,
            oil_rate_stb_d=result.oil_rate_stb_d,
            bottomhole_pressure_psia=result.bottomhole_pressure_psia,
        ))
    return points


def _nodal_with_injection(ipr, tubing, p_wh, formation_gor, q_inj, water_cut,
                          oil, gas, correlation):
    """Nodal analysis with lift gas added to the produced stream.

    Injected gas raises the effective GOR, and the effective GOR depends on
    the oil rate the analysis is solving for -- so the total GOR cannot be
    fixed in advance. The outer iteration below settles that: guess a rate,
    derive the GOR it implies, re-solve, repeat. It converges in a handful
    of passes because the dependence is weak and monotone.
    """
    q_guess = max(ipr.absolute_open_flow() * 0.5, 1.0)
    result = None
    for _ in range(12):
        total_gor = formation_gor + q_inj / max(q_guess, 1e-6)
        result = nodal_analysis(
            ipr=ipr, tubing=tubing, wellhead_pressure_psia=p_wh,
            gas_oil_ratio_scf_stb=total_gor, water_cut=water_cut,
            oil=oil, gas=gas, correlation=correlation, points=18, segments=20)
        if not result.converged:
            return result
        if abs(result.oil_rate_stb_d - q_guess) < 1e-3 * max(q_guess, 1.0):
            break
        # Damped update: taking the new rate outright makes the effective GOR
        # oscillate between passes instead of settling.
        q_guess = 0.5 * (q_guess + result.oil_rate_stb_d)
    return result
