"""Fluid property correlations: real gas, oil, water.

The Phase 0 spike carried a frozen ``z_factor=0.95`` constant. That is fine
for a low-pressure distribution network and quietly wrong everywhere else --
Z falls to ~0.85 around 1500 psia and climbs back above 1.0 past ~5000 psia,
so a fixed 0.95 misprices flow capacity by 10%+ at exactly the gathering and
transmission pressures this engine now has to cover.

Correlations used (all standard, all published):
  Z-factor          Dranchuk-Abou-Kassem (1975), fit to Standing-Katz
  pseudo-criticals  Sutton (1985), with Wichert-Aziz (1972) sour correction
  gas viscosity     Lee-Gonzalez-Eakin (1966)
  oil viscosity     Beggs-Robinson (1975) dead oil + Chew-Connally live oil
  solution GOR      Standing (1947)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import RANKINE_OFFSET, STD_PRESSURE_PSIA, STD_TEMPERATURE_R

R_GAS = 10.7316  # psia * ft3 / (lbmol * degR)
AIR_MW = 28.9625  # lbm/lbmol
WATER_DENSITY_LBFT3 = 62.4

# Dranchuk-Abou-Kassem coefficients A1..A11
_DAK = (
    0.3265, -1.0700, -0.5339, 0.01569, -0.05165,
    0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210,
)


def sutton_pseudo_criticals(gravity: float) -> tuple:
    """Pseudo-critical pressure (psia) and temperature (degR) from gas gravity.

    Sutton's fit rather than Standing's older one: Standing regressed on dry
    gases only and drifts for the richer gravities (>0.75) typical of
    associated gas from an oil well, which this engine now has to handle.
    """
    p_pc = 756.8 - 131.07 * gravity - 3.6 * gravity ** 2
    t_pc = 169.2 + 349.5 * gravity - 74.0 * gravity ** 2
    return p_pc, t_pc


def wichert_aziz_correction(p_pc: float, t_pc: float, y_co2: float, y_h2s: float) -> tuple:
    """Adjust pseudo-criticals for acid-gas content (CO2/H2S mole fractions).

    Gas from several Indian onshore fields is mildly sour; skipping this
    correction overestimates Z, and so flow capacity, in exactly those
    fields. A sweet mixture returns the inputs unchanged.
    """
    if y_co2 <= 0.0 and y_h2s <= 0.0:
        return p_pc, t_pc
    a = y_co2 + y_h2s
    epsilon = 120.0 * (a ** 0.9 - a ** 1.6) + 15.0 * (y_h2s ** 0.5 - y_h2s ** 4.0)
    t_pc_corr = t_pc - epsilon
    p_pc_corr = p_pc * t_pc_corr / (t_pc + y_h2s * (1.0 - y_h2s) * epsilon)
    return p_pc_corr, t_pc_corr


def z_factor_dak(p_psia: float, t_r: float, p_pc: float, t_pc: float,
                 tol: float = 1e-10, max_iter: int = 60) -> float:
    """Dranchuk-Abou-Kassem compressibility factor.

    Solved by Newton iteration on reduced density rho_r, because the DAK
    form is a polynomial in rho_r with a clean analytic derivative. Z itself
    only appears through rho_r = 0.27*Ppr/(Z*Tpr); successive substitution on
    Z converges far more slowly near the critical region.
    """
    t_pr = t_r / t_pc
    p_pr = p_psia / p_pc
    if p_pr <= 0.0:
        return 1.0
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _DAK

    c1 = a1 + a2 / t_pr + a3 / t_pr ** 3 + a4 / t_pr ** 4 + a5 / t_pr ** 5
    c2 = a6 + a7 / t_pr + a8 / t_pr ** 2
    c3 = a9 * (a7 / t_pr + a8 / t_pr ** 2)
    c4 = a10 / t_pr ** 3

    k = 0.27 * p_pr / t_pr  # rho_r = k / Z, so Z = k / rho_r
    rho = k  # initial guess corresponds to Z = 1

    for _ in range(max_iter):
        e = math.exp(-a11 * rho * rho)
        bracket = (1.0 + c1 * rho + c2 * rho ** 2 - c3 * rho ** 5
                   + c4 * (1.0 + a11 * rho ** 2) * rho ** 2 * e)
        f = rho * bracket - k
        # d(bracket)/d(rho); the exponential term differentiates to
        # c4*e*(2*rho + 4*a11*rho^3 - 2*a11*rho^3*(1 + a11*rho^2))
        dbracket = (
            c1 + 2.0 * c2 * rho - 5.0 * c3 * rho ** 4
            + c4 * e * (2.0 * rho + 4.0 * a11 * rho ** 3
                        - 2.0 * a11 * rho ** 3 * (1.0 + a11 * rho ** 2))
        )
        df = bracket + rho * dbracket
        if abs(df) < 1e-14:
            break
        rho_new = rho - f / df
        if rho_new <= 0.0:
            rho_new = rho * 0.5
        converged = abs(rho_new - rho) < tol
        rho = rho_new
        if converged:
            break

    if rho <= 0.0:
        return 1.0
    return k / rho


# The Z-factor interpolation table. 512 linear points over the full range the
# engine sees, which measured worst-case 8.1e-06 in Z across four gases --
# linear spacing beat logarithmic at every point count tried, and 256 points
# gave 3.3e-05, which is still far below correlation accuracy but buys little
# for the 1.2 ms saved.
Z_TABLE_LOW = 14.7
Z_TABLE_HIGH = 12_000.0
Z_TABLE_POINTS = 512

# Keyed by composition and temperature, not by instance: the engine builds
# many equal GasProperties -- one per well, per branch, per forecast step --
# and they should share one table.
_Z_TABLES = {}


def clear_z_tables() -> None:
    """Drop the cached tables. For tests and for long-lived processes that
    sweep temperature, where the cache would otherwise grow without bound."""
    _Z_TABLES.clear()


@dataclass(frozen=True)
class GasProperties:
    """Real-gas properties for one composition at one flowing temperature.

    Drop-in superset of ``physics.weymouth.GasProperties``: it keeps the
    ``gravity`` / ``temperature_r`` fields and exposes a ``z_factor``
    attribute, so existing call sites still read the same, but Z is now
    evaluated from an equation of state at the actual pressure rather than
    frozen at a constant.
    """

    gravity: float = 0.65
    temperature_r: float = STD_TEMPERATURE_R
    y_co2: float = 0.0
    y_h2s: float = 0.0
    y_n2: float = 0.0
    fixed_z: float = 0.0  # > 0 pins Z to this value (reproduces the Phase 0 spike exactly)

    @property
    def molecular_weight(self) -> float:
        return self.gravity * AIR_MW

    def pseudo_criticals(self) -> tuple:
        p_pc, t_pc = sutton_pseudo_criticals(self.gravity)
        return wichert_aziz_correction(p_pc, t_pc, self.y_co2, self.y_h2s)

    def z_at(self, p_psia: float) -> float:
        """Compressibility factor at this pressure.

        Interpolated from a table built once per gas, not solved per call.

        This is the hottest function in the engine by a wide margin. Profiling
        a 400-node network solve: `z_factor_dak` was called **76,760 times**
        and accounted for **53% of total runtime** -- the linear algebra did
        not appear in the top fourteen functions. A single nodal analysis
        calls it 10,080 times. Every one of those was an iterative Newton
        solve for a quantity that depends only on pressure once the gas and
        temperature are fixed, which they are for the whole of any one solve.

        Accuracy is not the trade here. Measured against direct DAK across
        four gases over 14.7-12,000 psia, the table is worst-case 8.1e-06 in
        Z -- roughly 250 times finer than the +/-0.002 anyone can read off the
        Standing-Katz chart that DAK is a fit to. The table costs about 2.5 ms
        to build and is amortised over tens of thousands of lookups.

        Outside the tabulated range the correlation is called directly, so an
        unusual pressure is slower but never wrong.
        """
        if self.fixed_z > 0.0:
            return self.fixed_z
        if Z_TABLE_LOW <= p_psia <= Z_TABLE_HIGH:
            pressures, values = self._z_table()
            return float(np.interp(p_psia, pressures, values))
        p_pc, t_pc = self.pseudo_criticals()
        return z_factor_dak(max(p_psia, 1e-6), self.temperature_r, p_pc, t_pc)

    def _z_table(self):
        """The interpolation table for this gas, built once.

        Cached on the composition and temperature rather than on the instance
        because `GasProperties` is frozen and because the engine creates many
        equal instances -- one per well, per branch, per forecast step -- and
        they should share one table rather than each paying to build it.
        """
        key = (self.gravity, self.temperature_r, self.y_co2, self.y_h2s,
               self.y_n2)
        table = _Z_TABLES.get(key)
        if table is None:
            p_pc, t_pc = self.pseudo_criticals()
            pressures = np.linspace(Z_TABLE_LOW, Z_TABLE_HIGH, Z_TABLE_POINTS)
            values = np.array(
                [z_factor_dak(float(p), self.temperature_r, p_pc, t_pc)
                 for p in pressures])
            table = (pressures, values)
            _Z_TABLES[key] = table
        return table

    @property
    def z_factor(self) -> float:
        """Backwards-compatible scalar Z, evaluated at standard pressure.

        Present so code written against the Phase 0 ``GasProperties`` still
        runs. Anything pressure-aware should call ``z_at`` instead: this
        property cannot know the flowing pressure, so it is only right at
        near-atmospheric conditions.
        """
        if self.fixed_z > 0.0:
            return self.fixed_z
        return self.z_at(STD_PRESSURE_PSIA)

    def density_lbft3(self, p_psia: float) -> float:
        z = self.z_at(p_psia)
        return p_psia * self.molecular_weight / (z * R_GAS * self.temperature_r)

    def viscosity_cp(self, p_psia: float) -> float:
        """Lee-Gonzalez-Eakin gas viscosity (centipoise)."""
        m = self.molecular_weight
        t = self.temperature_r
        rho = self.density_lbft3(p_psia)
        k = ((9.4 + 0.02 * m) * t ** 1.5) / (209.0 + 19.0 * m + t)
        x = 3.5 + 986.0 / t + 0.01 * m
        y = 2.4 - 0.2 * x
        return 1e-4 * k * math.exp(x * (rho / WATER_DENSITY_LBFT3) ** y)

    def formation_volume_factor(self, p_psia: float) -> float:
        """Bg, reservoir ft3 per standard ft3."""
        z = self.z_at(p_psia)
        return (STD_PRESSURE_PSIA / STD_TEMPERATURE_R) * (z * self.temperature_r / p_psia)


@dataclass(frozen=True)
class LiquidProperties:
    """Oil/water properties for single-phase liquid hydraulics, and the
    liquid phase of the multiphase correlations."""

    api_gravity: float = 35.0
    water_cut: float = 0.0            # volume fraction of the liquid that is water
    water_specific_gravity: float = 1.02
    temperature_r: float = 560.0
    gas_gravity: float = 0.65
    bubble_point_psia: float = 2500.0

    @property
    def temperature_f(self) -> float:
        return self.temperature_r - RANKINE_OFFSET

    @property
    def oil_specific_gravity(self) -> float:
        return 141.5 / (131.5 + self.api_gravity)

    @property
    def specific_gravity(self) -> float:
        """Volume-weighted SG of the oil/water mixture."""
        return ((1.0 - self.water_cut) * self.oil_specific_gravity
                + self.water_cut * self.water_specific_gravity)

    def density_lbft3(self, p_psia: float = STD_PRESSURE_PSIA) -> float:
        # Liquids are near-incompressible over the pressure range this engine
        # covers. The pressure argument is accepted so callers can treat the
        # gas and liquid interfaces uniformly, and is deliberately unused.
        return self.specific_gravity * WATER_DENSITY_LBFT3

    def dead_oil_viscosity_cp(self) -> float:
        """Beggs-Robinson dead-oil viscosity."""
        t_f = max(self.temperature_f, 1.0)
        x = 10.0 ** (3.0324 - 0.02023 * self.api_gravity) * t_f ** -1.163
        return 10.0 ** x - 1.0

    def solution_gor_scf_stb(self, p_psia: float) -> float:
        """Standing's solution gas-oil ratio, capped at the bubble point."""
        p = min(max(p_psia, 0.0), self.bubble_point_psia)
        y = 0.00091 * self.temperature_f - 0.0125 * self.api_gravity
        return self.gas_gravity * ((p / 18.2 + 1.4) * 10.0 ** y) ** 1.2048

    def viscosity_cp(self, p_psia: float = STD_PRESSURE_PSIA) -> float:
        """Live-oil viscosity via Chew-Connally, volume-blended with water.

        Below the bubble point, dissolved gas thins the oil substantially.
        Using dead-oil viscosity in a producing well overstates friction loss
        by a factor of two or more, which is the difference between a well
        the model says flows and one it says is dead.
        """
        mu_od = self.dead_oil_viscosity_cp()
        rs = self.solution_gor_scf_stb(p_psia)
        a = 10.715 * (rs + 100.0) ** -0.515
        b = 5.44 * (rs + 150.0) ** -0.338
        mu_oil = a * mu_od ** b
        mu_water = 0.5  # water at typical producing temperatures, cP
        return (1.0 - self.water_cut) * mu_oil + self.water_cut * mu_water
