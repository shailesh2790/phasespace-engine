"""Where the pressure goes, and what is pushing the oil out.

Material balance is the only reservoir calculation in this engine that is not a
correlation. It is an accounting identity: the volume of fluid taken out of a
tank must equal the volume the remaining fluid expanded into, plus whatever
came in from outside. Nothing is fitted; the physics is exact, and the
uncertainty is entirely in the PVT and the pressure measurements.

That makes it the right foundation for a forecast. A decline curve extrapolates
a trend and cannot say why the trend exists or what would change it. Material
balance ties rate to pressure, pressure to cumulative production, and therefore
makes a forecast answer the question an operator actually asks: not "what will
this well do if nothing changes" but "what happens if we draw it down harder".

Two directions, and both are used:

  **Forward** -- given the tank, what is the pressure at this cumulative
  production. This is what a forecast steps through, and what feeds the IPR in
  `wells` so the rate falls as the reservoir depletes.

  **Backward** -- given measured pressures and cumulative production, what size
  is the tank and what is driving it. This is the Havlena-Odeh straight line,
  and its answer is usually worth more than the volume: the *drive indices*
  say whether the field is being pushed by expanding oil, by a gas cap, or by
  an aquifer. Those are three different development plans.

**What this will not do quietly.** A tank model assumes one pressure everywhere,
which is false in any real reservoir and tolerable when the field is well
connected and the pressure data is from build-ups rather than flowing wells. It
assumes the PVT describes the fluid. And the gas p/z line is straight *only*
for a volumetric reservoir -- a curve above it means water is coming in, which
is the single most consequential thing this calculation can tell you and the
one most often fitted away by moving GIIP up. So the fit reports its own
residual curvature and names what it implies rather than absorbing it.

Units: pressure psia, gas scf, oil STB, water STB, volume factors as in
`pvt_oil`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fluids import GasProperties
from .pvt_oil import OilPVT, gas_fvf, water_fvf

# Below this the tank model has nothing to say: a reservoir at atmospheric
# pressure is not depleting, it is finished.
MIN_PRESSURE_PSIA = 14.7

# How much systematic curvature in the p/z line counts as water influx rather
# than as scatter. Expressed as the ratio of the quadratic term's contribution
# over the data range to the linear term's -- 3% is comfortably above the noise
# of a handful of build-up surveys and well below anything a real aquifer does.
CURVATURE_THRESHOLD = 0.03


# ============================================================ gas: p/z ======


@dataclass
class GasTank:
    """A volumetric gas reservoir.

        p/z = (pi/zi) (1 - Gp/G)

    The real-gas law applied to a fixed pore volume, nothing more. Its
    straightness in p/z is what makes gas material balance the most reliable
    reserves calculation in the industry -- and what makes a departure from
    straight so informative.
    """

    giip_scf: float
    initial_psia: float
    gas: GasProperties
    connate_water_fraction: float = 0.0
    rock_compressibility: float = 0.0     # 1/psi, pore volume

    def __post_init__(self):
        if self.giip_scf <= 0.0:
            raise ValueError("gas in place must be positive, got %r" % (self.giip_scf,))
        if self.initial_psia <= MIN_PRESSURE_PSIA:
            raise ValueError("initial pressure must exceed %r psia, got %r"
                             % (MIN_PRESSURE_PSIA, self.initial_psia))
        if not 0.0 <= self.connate_water_fraction < 1.0:
            raise ValueError("connate water fraction must be in [0, 1), got %r"
                             % (self.connate_water_fraction,))

    @property
    def initial_z(self) -> float:
        return self.gas.z_at(self.initial_psia)

    @property
    def initial_pz(self) -> float:
        return self.initial_psia / self.initial_z

    def _effective_pz(self, pressure_psia: float) -> float:
        """p/z adjusted for pore and connate-water compression.

        Usually negligible in a normally pressured gas reservoir and not
        negligible in an overpressured one, where ignoring it makes the p/z
        line concave and the GIIP look larger than it is. Included because
        getting this wrong is a well-documented way to over-book gas reserves.
        """
        if self.rock_compressibility <= 0.0:
            return pressure_psia / self.gas.z_at(pressure_psia)
        drop = max(self.initial_psia - pressure_psia, 0.0)
        shrink = 1.0 - self.rock_compressibility * drop / max(
            1.0 - self.connate_water_fraction, 1e-6)
        return pressure_psia / self.gas.z_at(pressure_psia) / max(shrink, 1e-6)

    def cumulative_at(self, pressure_psia: float) -> float:
        """Gas produced by the time the tank has fallen to this pressure."""
        if pressure_psia <= 0.0:
            raise ValueError("pressure must be positive, got %r" % (pressure_psia,))
        if pressure_psia > self.initial_psia:
            return 0.0
        return self.giip_scf * (1.0 - self._effective_pz(pressure_psia)
                                / self.initial_pz)

    def pressure_at(self, cumulative_scf: float) -> float:
        """Tank pressure at this cumulative production.

        Inverted by bisection rather than algebraically, because Z depends on
        pressure: p/z is linear in Gp but p is not.
        """
        if cumulative_scf < 0.0:
            raise ValueError("cumulative production cannot be negative, got %r"
                             % (cumulative_scf,))
        if cumulative_scf == 0.0:
            # Short-circuited so the undepleted case is exactly the initial
            # pressure rather than a bisection's last interval, which came
            # back a few scf away from zero cumulative on the round trip.
            return self.initial_psia
        if cumulative_scf >= self.giip_scf:
            return MIN_PRESSURE_PSIA
        target = self.initial_pz * (1.0 - cumulative_scf / self.giip_scf)
        low, high = MIN_PRESSURE_PSIA, self.initial_psia
        for _ in range(200):
            middle = 0.5 * (low + high)
            if self._effective_pz(middle) < target:
                low = middle
            else:
                high = middle
            if high - low < 1e-6:
                break
        return 0.5 * (low + high)

    def recovery_factor(self, abandonment_psia: float) -> float:
        """Fraction of the gas recovered by the time the tank reaches this
        pressure. For a volumetric gas reservoir this is nearly 1 - pa/pi,
        which is why gas fields recover 70-90% and oil fields recover 30%."""
        return self.cumulative_at(abandonment_psia) / self.giip_scf


@dataclass
class GasBalanceFit:
    """GIIP from measured pressures, and whether the tank is really volumetric."""

    giip_scf: float
    initial_pz: float
    slope: float
    points_used: int
    curvature_fraction: float
    rms_pz_residual: float
    drive: str
    notes: list = field(default_factory=list)

    @property
    def water_influx_suspected(self) -> bool:
        return self.drive != "volumetric depletion"


def fit_gas_balance(pressures, cumulatives, gas: GasProperties) -> GasBalanceFit:
    """Fit the p/z line, and say what its shape means.

    GIIP comes from the straight line's intercept over its slope. The part
    that matters more is the residual: fitting a quadratic as well and
    comparing tells you whether the data is straight.

    A p/z plot that bends *above* the straight line means pressure is being
    held up by something -- water encroaching, or a second compartment feeding
    in. The usual response is to re-fit the line through the late points and
    report a larger GIIP, which produces a reserves number that is too high
    and a development plan that does not expect water. So this reports the
    curvature and names it instead.
    """
    p = np.asarray(pressures, dtype=float)
    gp = np.asarray(cumulatives, dtype=float)
    if p.shape != gp.shape:
        raise ValueError("pressures and cumulatives must be the same length, "
                         "got %d and %d" % (p.size, gp.size))
    keep = (p > 0.0) & np.isfinite(p) & np.isfinite(gp) & (gp >= 0.0)
    p, gp = p[keep], gp[keep]
    if p.size < 3:
        raise ValueError(
            "need at least 3 pressure surveys to fit a p/z line, got %d. Two "
            "points define a line through any noise at all." % p.size)

    order = np.argsort(gp)
    p, gp = p[order], gp[order]
    pz = np.array([value / gas.z_at(float(value)) for value in p])

    linear = np.polyfit(gp, pz, 1)
    slope, intercept = float(linear[0]), float(linear[1])
    if slope >= 0.0:
        raise ValueError(
            "p/z rises with cumulative production in this data, which no tank "
            "does. Check that the pressures and cumulatives are paired and in "
            "the right units.")
    giip = intercept / -slope
    residual = pz - np.polyval(linear, gp)
    rms = float(np.sqrt(np.mean(residual ** 2)))

    notes = []
    curvature = 0.0
    drive = "volumetric depletion"
    if p.size >= 4:
        quadratic = np.polyfit(gp, pz, 2)
        span = float(gp[-1] - gp[0])
        if span > 0.0 and abs(intercept) > 0.0:
            # How much the quadratic term moves p/z across the data range,
            # relative to the whole p/z decline over that range.
            bend = abs(quadratic[0]) * span ** 2
            fall = abs(slope) * span
            curvature = bend / fall if fall > 0.0 else 0.0
        if curvature > CURVATURE_THRESHOLD:
            # Sign of the quadratic term says which way it bends: positive
            # curvature holds p/z up as production rises.
            if quadratic[0] > 0.0:
                drive = "water influx or a second compartment"
                notes.append(
                    "The p/z line bends upward by %.0f%% of its own fall. "
                    "Pressure is being supported by something outside the "
                    "tank -- an aquifer, or gas feeding in from a connected "
                    "compartment. Re-fitting a straight line through the late "
                    "points would raise GIIP and hide it."
                    % (curvature * 100.0))
            else:
                drive = "pressure-dependent pore volume"
                notes.append(
                    "The p/z line bends downward by %.0f%% of its own fall, "
                    "which is the signature of compaction or of an "
                    "overpressured reservoir whose pore volume is shrinking. "
                    "GIIP from a straight line is then too high."
                    % (curvature * 100.0))

    if gp[-1] < 0.1 * giip:
        notes.append(
            "Only %.0f%% of the fitted gas in place has been produced. A p/z "
            "line extrapolated from the first tenth of depletion is an "
            "estimate of the slope, not of the reserves."
            % (gp[-1] / giip * 100.0))

    return GasBalanceFit(giip_scf=giip, initial_pz=intercept, slope=slope,
                         points_used=int(p.size), curvature_fraction=curvature,
                         rms_pz_residual=rms, drive=drive, notes=notes)


# ====================================================== oil: Havlena-Odeh ===


@dataclass
class OilTank:
    """An oil reservoir as a tank, with an optional gas cap and aquifer.

        F = N (Eo + m Eg + Ef,w) + We

    where F is the reservoir volume withdrawn, Eo the expansion of oil and its
    dissolved gas, Eg the expansion of the gas cap, Ef,w the contribution of
    rock and connate water, and We cumulative water influx.
    """

    ooip_stb: float
    initial_psia: float
    pvt: OilPVT
    gas_cap_ratio: float = 0.0            # m: gas-cap reservoir volume / oil volume
    connate_water_fraction: float = 0.20
    rock_compressibility: float = 3.0e-6   # 1/psi
    water_compressibility: float = 3.0e-6  # 1/psi
    water_influx_constant: float = 0.0     # STB/psi, a simple steady-state aquifer

    def __post_init__(self):
        if self.ooip_stb <= 0.0:
            raise ValueError("oil in place must be positive, got %r" % (self.ooip_stb,))
        if self.initial_psia <= MIN_PRESSURE_PSIA:
            raise ValueError("initial pressure must exceed %r psia, got %r"
                             % (MIN_PRESSURE_PSIA, self.initial_psia))
        if self.gas_cap_ratio < 0.0:
            raise ValueError("gas cap ratio cannot be negative, got %r"
                             % (self.gas_cap_ratio,))
        if not 0.0 <= self.connate_water_fraction < 1.0:
            raise ValueError("connate water fraction must be in [0, 1), got %r"
                             % (self.connate_water_fraction,))
        if self.initial_psia > self.pvt.bubble_point_psia and self.gas_cap_ratio > 0.0:
            raise ValueError(
                "an undersaturated reservoir (%.0f psia, bubble point %.0f) "
                "cannot have a gas cap: there is no free gas at initial "
                "conditions" % (self.initial_psia, self.pvt.bubble_point_psia))

    # ------------------------------------------------------ expansion terms --

    @property
    def boi(self) -> float:
        return self.pvt.bo(self.initial_psia)

    @property
    def rsi(self) -> float:
        return self.pvt.rs(self.initial_psia)

    def bg(self, pressure_psia: float) -> float:
        z = self.pvt_gas_z(pressure_psia)
        return gas_fvf(pressure_psia, self.pvt.temperature_f, z)

    def pvt_gas_z(self, pressure_psia: float) -> float:
        """Z for the associated gas, from the same DAK correlation the rest of
        the engine uses rather than a separate approximation."""
        gas = GasProperties(gravity=self.pvt.gas_gravity,
                            temperature_r=self.pvt.temperature_f + 459.67)
        return gas.z_at(pressure_psia)

    def oil_expansion(self, pressure_psia: float) -> float:
        """Eo: oil plus its originally dissolved gas, reservoir bbl/STB."""
        bo = self.pvt.bo(pressure_psia)
        rs = self.pvt.rs(pressure_psia)
        return (bo - self.boi) + (self.rsi - rs) * self.bg(pressure_psia)

    def gas_cap_expansion(self, pressure_psia: float) -> float:
        """Eg: expansion of the gas cap, reservoir bbl per STB of oil."""
        bgi = self.bg(self.initial_psia)
        if bgi <= 0.0:
            return 0.0
        return self.boi * (self.bg(pressure_psia) / bgi - 1.0)

    def rock_water_expansion(self, pressure_psia: float) -> float:
        """Ef,w: pore compaction plus connate-water expansion.

        Small in a normally pressured reservoir and the dominant term in a
        highly undersaturated one, where there is no free gas to expand and
        the only thing pushing oil out is the rock closing on it.
        """
        swc = self.connate_water_fraction
        drop = max(self.initial_psia - pressure_psia, 0.0)
        group = (self.water_compressibility * swc + self.rock_compressibility) \
            / max(1.0 - swc, 1e-6)
        return self.boi * (1.0 + self.gas_cap_ratio) * group * drop

    def water_influx(self, pressure_psia: float) -> float:
        """We: a steady-state aquifer, linear in drawdown.

        The simplest aquifer model there is -- Schilthuis. Real aquifers lag
        pressure and need van Everdingen-Hurst or Fetkovich; this is here so a
        waterdrive field can be modelled at all, and it is deliberately a
        single number an engineer can fit and argue with rather than a model
        hiding behind one.
        """
        if self.water_influx_constant <= 0.0:
            return 0.0
        return self.water_influx_constant * max(
            self.initial_psia - pressure_psia, 0.0)

    # ------------------------------------------------------------- withdraw --

    def withdrawal(self, pressure_psia: float, np_stb: float,
                   produced_gor: float, wp_stb: float = 0.0) -> float:
        """F: reservoir volume taken out, bbl.

            F = Np [Bo + (Rp - Rs) Bg] + Wp Bw

        `produced_gor` is cumulative gas produced over cumulative oil, which
        is not the instantaneous GOR and is the term most often supplied
        wrongly. Using the instantaneous value understates withdrawal on a
        field whose GOR has been rising, which is every field past its bubble
        point.
        """
        if np_stb < 0.0 or wp_stb < 0.0:
            raise ValueError("cumulative production cannot be negative")
        bo = self.pvt.bo(pressure_psia)
        rs = self.pvt.rs(pressure_psia)
        free_gas = max(produced_gor - rs, 0.0) * self.bg(pressure_psia)
        bw = water_fvf(pressure_psia, self.pvt.temperature_f)
        return np_stb * (bo + free_gas) + wp_stb * bw

    def pressure_at(self, np_stb: float, produced_gor: float,
                    wp_stb: float = 0.0) -> float:
        """Tank pressure at this cumulative production. Bisection on F = ...

        Every term depends on pressure, so there is no closed form. The
        residual is monotone in pressure over the range where the tank has any
        meaning, which is what makes bisection safe here.
        """
        if np_stb < 0.0:
            raise ValueError("cumulative oil cannot be negative, got %r" % (np_stb,))
        if np_stb == 0.0 and wp_stb == 0.0:
            return self.initial_psia

        def residual(p):
            supply = self.ooip_stb * (
                self.oil_expansion(p)
                + self.gas_cap_ratio * self.gas_cap_expansion(p)
                + self.rock_water_expansion(p)) + self.water_influx(p)
            return supply - self.withdrawal(p, np_stb, produced_gor, wp_stb)

        # The residual is positive at low pressure (the tank has expanded more
        # than was taken out) and negative at high pressure, so the root is
        # bracketed by construction. The bracket is still found by scanning
        # rather than assumed: below the bubble point both the supply and the
        # withdrawal terms grow as pressure falls, and which grows faster
        # depends on the produced GOR. Asserting monotonicity here and
        # bisecting on the assumption is how this returned the wrong end of
        # the interval -- confidently, and for every input.
        grid = np.linspace(self.initial_psia, MIN_PRESSURE_PSIA, 97)
        values = [residual(float(p)) for p in grid]
        if values[-1] < 0.0:
            # Even emptied to atmospheric the tank cannot have supplied this
            # much. Reported as depleted rather than extrapolated past the
            # point where the model means anything.
            return MIN_PRESSURE_PSIA
        bracket = None
        for i in range(len(grid) - 1):
            if values[i] <= 0.0 <= values[i + 1]:
                bracket = (float(grid[i + 1]), float(grid[i]))
                break
        if bracket is None:
            return self.initial_psia
        low, high = bracket                      # low pressure, high pressure
        for _ in range(200):
            middle = 0.5 * (low + high)
            if residual(middle) > 0.0:
                low = middle                     # still below the root
            else:
                high = middle
            if high - low < 1e-6:
                break
        return 0.5 * (low + high)

    # ----------------------------------------------------------- diagnosis ---

    def drive_indices(self, pressure_psia: float, np_stb: float,
                      produced_gor: float, wp_stb: float = 0.0) -> dict:
        """Which mechanism is actually producing the oil (Pirson).

        The four indices sum to one and say where the energy is coming from.
        This is the most actionable number material balance produces, because
        each answer implies a different plan: a depletion-drive field needs
        injection, a gas-cap field needs its cap protected from blowdown, and
        a waterdrive field needs its water handled rather than its pressure
        worried about.
        """
        f = self.withdrawal(pressure_psia, np_stb, produced_gor, wp_stb)
        if f <= 0.0:
            return {"depletion": 0.0, "gas_cap": 0.0, "water": 0.0,
                    "compaction": 0.0, "closure": 0.0,
                    "dominant": "nothing produced yet"}
        bw = water_fvf(pressure_psia, self.pvt.temperature_f)
        indices = {
            "depletion": self.ooip_stb * self.oil_expansion(pressure_psia) / f,
            "gas_cap": self.ooip_stb * self.gas_cap_ratio
            * self.gas_cap_expansion(pressure_psia) / f,
            "water": (self.water_influx(pressure_psia) - wp_stb * bw) / f,
            "compaction": self.ooip_stb
            * self.rock_water_expansion(pressure_psia) / f,
        }
        dominant = max(indices, key=lambda key: indices[key])
        indices["dominant"] = dominant
        indices["closure"] = sum(
            value for key, value in indices.items()
            if key not in ("dominant", "closure"))
        return indices


@dataclass
class OilBalanceFit:
    """OOIP and gas-cap ratio from a pressure and production history."""

    ooip_stb: float
    gas_cap_ratio: float
    points_used: int
    rms_relative_residual: float
    notes: list = field(default_factory=list)


def fit_oil_balance(
    pressures,
    oil_cumulatives,
    produced_gors,
    pvt: OilPVT,
    initial_psia: float,
    water_cumulatives=None,
    fit_gas_cap: bool = False,
    connate_water_fraction: float = 0.20,
    rock_compressibility: float = 3.0e-6,
) -> OilBalanceFit:
    """Havlena-Odeh: the straight line that gives OOIP.

    With no gas cap and no aquifer, F = N Et is a line through the origin, so
    N is a single least-squares ratio. With a gas cap, F/Et against Eg/Et is a
    line whose intercept is N and whose slope is N m -- the classic two-unknown
    form, solved here by ordinary least squares on that transformed plot.

    An aquifer is *not* fitted. Three unknowns on a handful of noisy pressure
    surveys will always find a combination that fits, and the combination it
    finds is usually a large aquifer offsetting a small N, which is both
    wrong and reassuring. Water influx is diagnosed instead -- from the
    residual's trend with time -- and left for the user to supply as a number
    they can defend.
    """
    p = np.asarray(pressures, dtype=float)
    np_oil = np.asarray(oil_cumulatives, dtype=float)
    rp = np.asarray(produced_gors, dtype=float)
    wp = (np.zeros_like(np_oil) if water_cumulatives is None
          else np.asarray(water_cumulatives, dtype=float))
    if not (p.shape == np_oil.shape == rp.shape == wp.shape):
        raise ValueError("pressures, cumulatives and GORs must be the same length")

    keep = (p > 0.0) & (np_oil > 0.0) & np.isfinite(p) & np.isfinite(np_oil)
    p, np_oil, rp, wp = p[keep], np_oil[keep], rp[keep], wp[keep]
    minimum = 3 if fit_gas_cap else 2
    if p.size < minimum:
        raise ValueError("need at least %d surveys with production, got %d"
                         % (minimum, p.size))

    # A unit tank evaluates the expansion terms, which do not depend on N.
    probe = OilTank(ooip_stb=1.0, initial_psia=initial_psia, pvt=pvt,
                    gas_cap_ratio=0.0,
                    connate_water_fraction=connate_water_fraction,
                    rock_compressibility=rock_compressibility)
    f = np.array([probe.withdrawal(float(pressure), float(oil), float(gor),
                                   float(water))
                  for pressure, oil, gor, water in zip(p, np_oil, rp, wp)])
    eo = np.array([probe.oil_expansion(float(pressure)) for pressure in p])
    # Evaluated at m = 0, so the (1 + m) factor can be reapplied once m is
    # known rather than baked in at the wrong value.
    efw = np.array([probe.rock_water_expansion(float(pressure)) for pressure in p])
    eg = np.array([probe.gas_cap_expansion(float(pressure)) for pressure in p])
    et = eo + efw

    notes = []
    if np.any(et <= 0.0):
        raise ValueError(
            "the fluid does not expand at one or more of these pressures, so "
            "material balance cannot be formed. Usually a pressure above the "
            "stated initial pressure.")

    if fit_gas_cap:
        # F/Et = N + N m (Eg/Et), solved by ordinary least squares.
        #
        # Iterated, because Ef,w carries a factor of (1 + m) and m is one of
        # the unknowns: evaluating the rock term at m = 0 and fitting once
        # understates it by the gas-cap fraction, and the regression absorbs
        # the difference into N. On a tank with m = 0.5 that was a 4.5% error
        # in the oil in place -- small enough to look like a fit and large
        # enough to matter. Three passes converge to machine precision here;
        # the dependence is weak because Ef,w is a small part of Et.
        ooip, m = 0.0, 0.0
        for _pass in range(6):
            et_pass = eo + efw * (1.0 + m)
            design = np.column_stack([np.ones_like(et_pass), eg / et_pass])
            coefficients, *_ = np.linalg.lstsq(design, f / et_pass, rcond=None)
            ooip = float(coefficients[0])
            if ooip <= 0.0:
                break
            updated = float(coefficients[1] / ooip)
            if abs(updated - m) < 1e-12:
                m = updated
                break
            m = updated
        et = eo + efw * (1.0 + max(m, 0.0))
        if ooip <= 0.0:
            raise ValueError(
                "the gas-cap fit gives a negative oil in place, which means "
                "this history is not described by a tank with a gas cap. Fit "
                "without one, or check the produced GOR.")
        if m < 0.0:
            notes.append(
                "The fit wants a negative gas cap, which does not exist. That "
                "usually means pressure support from an aquifer is being "
                "absorbed by the gas-cap term; m is reported as zero.")
            m = 0.0
    else:
        ooip = float(np.sum(et * f) / np.sum(et * et))
        m = 0.0

    predicted = ooip * (et + m * eg)
    relative = (predicted - f) / np.maximum(np.abs(f), 1e-9)
    rms = float(np.sqrt(np.mean(relative ** 2)))

    # A trend in the residual against cumulative production is the signature
    # of something entering the tank, because an aquifer's contribution grows
    # with time while scatter does not.
    if p.size >= 4:
        trend = np.polyfit(np_oil, relative, 1)[0]
        swing = abs(trend) * (np_oil[-1] - np_oil[0])
        if swing > 0.05:
            direction = ("less" if trend > 0.0 else "more")
            notes.append(
                "The balance misses by a growing amount -- %.0f%% drift across "
                "the history, with the tank supplying %s than is produced over "
                "time. That is what water influx looks like. Fit an aquifer "
                "constant you can defend rather than raising oil in place."
                % (swing * 100.0, direction))

    if rms > 0.10:
        notes.append(
            "The straight line misses by %.0f%% on average. Either the "
            "pressures are not tank pressures (flowing rather than built-up), "
            "or the field is not one tank." % (rms * 100.0))

    return OilBalanceFit(ooip_stb=ooip, gas_cap_ratio=m, points_used=int(p.size),
                         rms_relative_residual=rms, notes=notes)
