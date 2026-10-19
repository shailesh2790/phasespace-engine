"""Intermittent gas lift: the cycle, and when to use it instead of continuous.

Continuous gas lift lightens a flowing column. It needs a well that can
sustain flow once lightened, and on a low-pressure, low-productivity oil
well there is no injection rate that does: inject enough to lift the column
and the drawdown collapses the inflow; inject less and the column does not
move. The well heads, slugs, or dies. `gaslift.py` will happily compute an
allocation for such a well, and the allocation will be wrong in a way the
allocation cannot see, because its whole model is a steady state the well
never reaches.

Intermittent lift accepts that and stops pretending the well is steady. The
valve stays shut while liquid feeds in from the formation; then gas is
admitted fast, drives the accumulated slug to surface as a piston, and the
tubing blows down and starts again. The well produces in pulses. Nothing
about it is a steady state, and nothing in this engine before now modelled
it -- `wells.INTERMITTENT` is a Beggs-Brill *flow regime*, which is an
unrelated use of the word.

**What decides whether it works is fallback, and fallback is where the
honesty has to be.** The slug is not a solid piston. Gas penetrates it, and
liquid is stripped onto the tubing wall and left behind as a film. The
fraction that never reaches surface is the fallback, and it is the
difference between a well that makes 90 bbl/d and the same well making 40.
Brown ("The Technology of Artificial Lift Methods", Vol. 2a) puts it at
roughly 5-7% per 1000 ft of lift at a well-designed slug velocity of about
1000 ft/min, rising at both lower velocities (liquid drains back through the
gas) and higher ones (more film stripped).

`fallback_fraction` reproduces that: the 5%/1000 ft anchor and the 1000
ft/min optimum are Brown's, **the smooth U-shape between them is not a
published correlation**. It is an interpolation chosen to have the right
minimum in the right place, because the published form is a chart. It
should be calibrated against a well's own test before any number from it is
quoted as a rate, and `IntermittentWell.fallback_per_1000ft` exists for
exactly that. Scatter on published fallback data is wide -- tens of percent
relative -- and that uncertainty dominates everything else in this module.

**The optimum is a real optimum, not a monotone curve**, and it is worth
being exact about which mechanism makes it one, because the plausible
explanation is not the operative one here.

Cycle too slowly and the liquid accumulating above the valve raises the
bottomhole pressure until it reaches reservoir pressure and the formation
simply stops feeding. On the low-pressure well in the tests that happens
outright at two cycles a day: the slug stops growing at 16.4 bbl no matter
how long it is left, so halving the frequency halves the production. Cycle
too fast and the lift and blowdown times, which are fixed, eat the period
that the slug needed to build in.

What does *not* drive it, in this model, is slug size: fallback here is a
fraction per 1000 ft of lift, so it is the same 25% whether the slug is
2800 ft long or 500. That is Brown's formulation and it is the standard
design method, but it means the short-slug regime -- where gas channels
through a slug too thin to hold together and recovery collapses -- is
**not represented**. A cycle whose slug is under about 100 ft of tubing is
outside what this model can be trusted for, and `Cycle.notes` says so
rather than returning a number that looks like the others.

**It plugs into the existing allocator.** Sweeping cycle frequency produces
exactly what `gaslift.allocate` consumes -- a curve of oil against gas -- so
an intermittent well competes for the same compressor as the continuous
ones, on the same marginal-oil-per-unit-gas basis, instead of being managed
on a separate spreadsheet.

What this is not: a transient multiphase simulator. It is a quasi-steady
cycle model of the kind used for design, which means it is honest about
averages over a cycle and says nothing trustworthy about what happens
within one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fluids import GasProperties, LiquidProperties
from .units import ATM_PSIA, STD_PRESSURE_PSIA, STD_TEMPERATURE_R
from .wells import InflowPerformance

# Tubing capacity: C [bbl/ft] = d^2 / 1029.4 for d in inches. From
# (pi/4) d^2 / 144 ft^2 per foot, divided by 5.615 ft^3/bbl.
TUBING_CAPACITY_CONSTANT = 1029.4

# Brown's design optimum for slug velocity. Fallback is least here.
OPTIMUM_SLUG_VELOCITY_FT_MIN = 1000.0

# Fallback at the optimum velocity, per 1000 ft of lift (Brown, Vol. 2a:
# "5 to 7 percent per 1000 ft"). The low end is taken as the design value
# because 5% corresponds to a well-conditioned slug, and the module reports
# the band rather than hiding it.
FALLBACK_AT_OPTIMUM = 0.05
FALLBACK_BAND = (0.05, 0.07)

# Curvature of the U in velocity. Not published: chosen so the penalty at
# half and double the optimum velocity is about one percentage point per
# 1000 ft, which matches the slope of the published chart either side of
# the minimum without claiming to be it.
FALLBACK_VELOCITY_CURVATURE = 0.02

# A slug cannot lose more than all of itself. Reaching this means the
# geometry is outside what intermittent lift can do, not that the well
# makes nothing.
MAX_FALLBACK = 0.95

# Below this much liquid in the tubing the slug is too thin to hold
# together: gas channels through it rather than pushing it, and recovery
# collapses in a way a fraction-per-1000-ft fallback does not represent.
# The model does not cover this regime and says so instead of guessing.
MIN_COMPETENT_SLUG_FT = 100.0

# Time for the tubing to blow down and the valve to reseat, in minutes.
# Field practice is one to three minutes; the midpoint is the default and
# it is an input because it is set by the equipment, not by the reservoir.
DEFAULT_BLOWDOWN_MIN = 2.0

# Below this the cycle model is being asked about a well that is nearly
# continuous, and above it about one cycling faster than the hardware can.
MIN_CYCLES_PER_DAY = 1.0
MAX_CYCLES_PER_DAY = 200.0

# Screening thresholds for continuous versus intermittent. These are
# rules of thumb and are reported as such: the real comparison is to run
# both models, which `compare_with_continuous` does.
LOW_PRODUCTIVITY_PI = 0.5          # stb/d/psi
LOW_RATE_STB_D = 150.0


def tubing_capacity_bbl_ft(diameter_in: float) -> float:
    if diameter_in <= 0.0:
        raise ValueError("tubing diameter must be positive, got %r" % (diameter_in,))
    return diameter_in ** 2 / TUBING_CAPACITY_CONSTANT


def fallback_fraction(velocity_ft_min: float, depth_ft: float,
                      per_1000ft: float = FALLBACK_AT_OPTIMUM) -> float:
    """Fraction of the slug that never reaches surface.

    The anchor (5%/1000 ft at 1000 ft/min) is Brown's. The shape in
    velocity is an interpolation with the minimum in the documented place,
    not a published correlation -- see the module docstring. Calibrate
    `per_1000ft` against a well test before quoting a rate from it.
    """
    if depth_ft <= 0.0:
        raise ValueError("lift depth must be positive, got %r" % (depth_ft,))
    if velocity_ft_min <= 0.0:
        return MAX_FALLBACK

    ratio = velocity_ft_min / OPTIMUM_SLUG_VELOCITY_FT_MIN
    # ratio + 1/ratio - 2 is zero at the optimum and positive either side,
    # which is the whole of the shape being claimed.
    penalty = FALLBACK_VELOCITY_CURVATURE * (ratio + 1.0 / ratio - 2.0)
    per_1000 = per_1000ft + penalty
    return min(per_1000 * depth_ft / 1000.0, MAX_FALLBACK)


@dataclass
class IntermittentWell:
    """A well on intermittent lift, and the equipment that constrains it."""

    ipr: InflowPerformance
    depth_ft: float                       # to the operating valve
    tubing_id_in: float
    wellhead_psia: float = 100.0
    liquid: LiquidProperties = field(default_factory=LiquidProperties)
    gas: GasProperties = field(default_factory=GasProperties)
    blowdown_min: float = DEFAULT_BLOWDOWN_MIN
    slug_velocity_ft_min: float = OPTIMUM_SLUG_VELOCITY_FT_MIN
    # Per 1000 ft at the optimum velocity. The one parameter worth fitting
    # to a well test, and the one that dominates the answer.
    fallback_per_1000ft: float = FALLBACK_AT_OPTIMUM
    # Pressure the formation sees in addition to the column above the
    # valve: the static head between the valve and the perforations.
    below_valve_psi: float = 0.0

    def __post_init__(self):
        if self.depth_ft <= 0.0:
            raise ValueError("depth must be positive, got %r" % (self.depth_ft,))
        if self.wellhead_psia <= 0.0:
            raise ValueError("wellhead pressure must be positive, got %r"
                             % (self.wellhead_psia,))
        if self.slug_velocity_ft_min <= 0.0:
            raise ValueError("slug velocity must be positive, got %r"
                             % (self.slug_velocity_ft_min,))

    @property
    def capacity_bbl_ft(self) -> float:
        return tubing_capacity_bbl_ft(self.tubing_id_in)

    @property
    def liquid_gradient_psi_ft(self) -> float:
        """Static gradient of the produced liquid, oil and water together."""
        return self.liquid.density_lbft3(self.wellhead_psia) / 144.0

    @property
    def lift_time_min(self) -> float:
        return self.depth_ft / self.slug_velocity_ft_min

    def bottomhole_psia(self, column_ft: float) -> float:
        """Pressure the formation sees with this much liquid above the valve."""
        return (self.wellhead_psia + self.below_valve_psi
                + self.liquid_gradient_psi_ft * column_ft)


@dataclass
class Cycle:
    """One cycle, and what it cost."""

    cycles_per_day: float
    build_minutes: float
    lift_minutes: float
    blowdown_minutes: float
    slug_bbl: float                 # accumulated above the valve
    recovered_bbl: float            # what reached surface
    fallback: float
    final_column_ft: float
    final_bottomhole_psia: float
    gas_per_cycle_scf: float
    feasible: bool = True
    notes: list = field(default_factory=list)

    @property
    def oil_stb_d(self) -> float:
        """Liquid to surface per day. Named for the allocator's convention."""
        return self.recovered_bbl * self.cycles_per_day

    @property
    def gas_scf_d(self) -> float:
        return self.gas_per_cycle_scf * self.cycles_per_day

    @property
    def gas_liquid_ratio(self) -> float:
        return self.gas_scf_d / self.oil_stb_d if self.oil_stb_d > 0.0 else math.inf


def _accumulate(well: IntermittentWell, minutes: float, steps: int = 240) -> tuple:
    """Integrate inflow against a rising column. Returns (bbl, column_ft).

    Explicit stepping rather than a closed form because the IPR is an
    arbitrary object -- Vogel, Fetkovich, composite -- and only its `rate`
    is available. The step count is fixed rather than adaptive so two runs
    of the same case give the same number, which the provenance bundle
    requires.
    """
    if minutes <= 0.0:
        return 0.0, 0.0

    capacity = well.capacity_bbl_ft
    dt = minutes / steps
    volume = 0.0
    for _ in range(steps):
        column = volume / capacity
        p_wf = well.bottomhole_psia(column)
        rate = max(float(well.ipr.rate(p_wf)), 0.0)        # stb/d
        volume += rate * dt / 1440.0
        if volume / capacity >= well.depth_ft:
            # The tubing is full to the valve; nothing more can accumulate
            # above it, and waiting longer only kills the inflow.
            return capacity * well.depth_ft, well.depth_ft
    return volume, volume / capacity


def _gas_per_cycle_scf(well: IntermittentWell, slug_bbl: float) -> float:
    """Gas needed to drive one slug to surface.

    The gas fills the tubing from the valve to the surface behind the
    slug. Its average pressure over the lift is taken as the wellhead
    pressure plus half the slug's own weight: the slug's hydrostatic load
    on the gas falls from its full value to nothing as it is produced.
    This is a design estimate. Real usage is measured, is higher, and the
    difference is the gas that breaks through the slug.
    """
    slug_ft = slug_bbl / well.capacity_bbl_ft
    average_psia = (well.wellhead_psia
                    + 0.5 * well.liquid_gradient_psi_ft * slug_ft)
    volume_ft3 = well.capacity_bbl_ft * well.depth_ft * 5.615
    z = well.gas.z_at(average_psia)
    return (volume_ft3 * (average_psia / STD_PRESSURE_PSIA)
            * (STD_TEMPERATURE_R / well.gas.temperature_r) / max(z, 1e-6))


def cycle(well: IntermittentWell, cycles_per_day: float) -> Cycle:
    """One cycle at this frequency, with the time budget enforced."""
    if cycles_per_day <= 0.0:
        raise ValueError("cycles per day must be positive, got %r"
                         % (cycles_per_day,))

    period = 1440.0 / cycles_per_day
    lift = well.lift_time_min
    overhead = lift + well.blowdown_min
    build = period - overhead

    if build <= 0.0:
        return Cycle(
            cycles_per_day=cycles_per_day, build_minutes=0.0,
            lift_minutes=lift, blowdown_minutes=well.blowdown_min,
            slug_bbl=0.0, recovered_bbl=0.0, fallback=1.0,
            final_column_ft=0.0,
            final_bottomhole_psia=well.bottomhole_psia(0.0),
            gas_per_cycle_scf=0.0, feasible=False,
            notes=["%.0f cycles a day leaves %.1f min for the slug to build, "
                   "and the lift and blowdown alone need %.1f min. The well "
                   "cannot cycle this fast."
                   % (cycles_per_day, build, overhead)])

    slug, column = _accumulate(well, build)
    fall = fallback_fraction(well.slug_velocity_ft_min, well.depth_ft,
                             well.fallback_per_1000ft)
    recovered = slug * (1.0 - fall)

    notes = []
    if column >= well.depth_ft - 1e-9:
        notes.append(
            "the tubing filled to the valve before the build period ended; "
            "cycling faster would produce more")
    if 0.0 < column < MIN_COMPETENT_SLUG_FT:
        notes.append(
            "the slug is only %.0f ft of tubing. Below about %.0f ft gas "
            "channels through rather than lifting it, and recovery falls "
            "away faster than a per-1000-ft fallback can represent, so "
            "this rate is optimistic by an amount the model cannot state."
            % (column, MIN_COMPETENT_SLUG_FT))
    if fall >= MAX_FALLBACK - 1e-9:
        notes.append(
            "fallback is at its ceiling: at %.0f ft this depth is beyond "
            "what intermittent lift can carry at this slug velocity"
            % well.depth_ft)

    return Cycle(
        cycles_per_day=cycles_per_day, build_minutes=build,
        lift_minutes=lift, blowdown_minutes=well.blowdown_min,
        slug_bbl=slug, recovered_bbl=recovered, fallback=fall,
        final_column_ft=column,
        final_bottomhole_psia=well.bottomhole_psia(column),
        gas_per_cycle_scf=_gas_per_cycle_scf(well, slug),
        notes=notes)


@dataclass
class CycleOptimum:
    """The best cycle frequency, and the curve it was chosen from."""

    best: Cycle
    curve: list = field(default_factory=list)
    oil_price: float = 1.0
    gas_price: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def cycles_per_day(self) -> float:
        return self.best.cycles_per_day

    def performance_curve(self) -> tuple:
        """(gas scf/d, oil stb/d), for `gaslift.allocate`.

        An intermittent well handed to the allocator on the same terms as a
        continuous one: the compressor does not care which kind of well
        the gas went to, only what the next unit of it buys.
        """
        usable = [c for c in self.curve if c.feasible and c.oil_stb_d > 0.0]
        return (np.array([c.gas_scf_d for c in usable]),
                np.array([c.oil_stb_d for c in usable]))


def optimise_cycles(well: IntermittentWell, oil_price: float = 1.0,
                    gas_price: float = 0.0, points: int = 60) -> CycleOptimum:
    """Find the cycle frequency worth running.

    There is a genuine interior optimum, from two mechanisms that bite at
    opposite ends. Cycling too slowly lets the accumulated column raise the
    bottomhole pressure towards reservoir pressure, at which point the
    formation stops feeding and the extra waiting buys nothing. Cycling too
    fast spends the period on lift and blowdown, which are fixed, instead
    of on building a slug.

    With `gas_price` in the same units as `oil_price` the objective becomes
    value rather than volume, and the two optima are not the same well: gas
    per barrel rises steeply on the fast side, so a priced optimum cycles
    more slowly than a volume one.
    """
    if oil_price <= 0.0:
        raise ValueError("oil price must be positive, got %r" % (oil_price,))
    if gas_price < 0.0:
        raise ValueError("gas price cannot be negative, got %r" % (gas_price,))

    # The fastest the hardware allows, with at least a little build time.
    overhead = well.lift_time_min + well.blowdown_min
    fastest = min(MAX_CYCLES_PER_DAY, 1440.0 / (overhead * 1.05))
    if fastest <= MIN_CYCLES_PER_DAY:
        raise ValueError(
            "lift and blowdown need %.1f min, so this well cannot complete "
            "even one cycle a day. Check the depth (%.0f ft) and slug "
            "velocity (%.0f ft/min)."
            % (overhead, well.depth_ft, well.slug_velocity_ft_min))

    frequencies = np.geomspace(MIN_CYCLES_PER_DAY, fastest, points)
    curve = [cycle(well, float(n)) for n in frequencies]
    usable = [c for c in curve if c.feasible]
    if not usable:
        raise ValueError("no feasible cycle frequency for this well")

    def value(entry: Cycle) -> float:
        return oil_price * entry.oil_stb_d - gas_price * entry.gas_scf_d

    best = max(usable, key=value)
    notes = []
    if best.cycles_per_day >= fastest * 0.99:
        notes.append(
            "the optimum is at the fastest cycle the equipment allows, so "
            "the well is limited by lift and blowdown time rather than by "
            "the reservoir. A faster valve or a shallower lift point would "
            "produce more.")
    if best.cycles_per_day <= MIN_CYCLES_PER_DAY * 1.01:
        notes.append(
            "the optimum is at the slowest frequency searched; the well "
            "takes longer than a day to build a worthwhile slug.")
    notes.append(
        "fallback is %.0f%% of each slug and is the largest single "
        "uncertainty here. Published scatter on it is wide, so fit "
        "`fallback_per_1000ft` to a well test before quoting this rate."
        % (best.fallback * 100.0))
    return CycleOptimum(best=best, curve=curve, oil_price=oil_price,
                        gas_price=gas_price, notes=notes)


def calibrate_fallback(well: IntermittentWell, cycles_per_day: float,
                       measured_stb_d: float) -> float:
    """Back out the fallback this well actually shows from one test.

    One measured rate at a known cycle frequency pins the one parameter the
    correlation cannot supply. This is the whole of the calibration because
    everything else in the cycle -- the inflow, the geometry, the timing --
    is either measured or known, and fallback is the only term carrying
    real uncertainty.

    Returns fallback per 1000 ft, which is what `IntermittentWell` takes.
    """
    if measured_stb_d < 0.0:
        raise ValueError("measured rate cannot be negative, got %r"
                         % (measured_stb_d,))
    result = cycle(well, cycles_per_day)
    if not result.feasible or result.slug_bbl <= 0.0:
        raise ValueError(
            "the well does not cycle at %g a day, so no fallback can be "
            "inferred from a rate measured there" % cycles_per_day)

    accumulated = result.slug_bbl * cycles_per_day
    if measured_stb_d > accumulated:
        raise ValueError(
            "the well is measured at %.1f stb/d but the formation can only "
            "feed %.1f stb/d into the tubing at this cycle frequency. Either "
            "the inflow model is wrong or the measurement is not this well."
            % (measured_stb_d, accumulated))

    observed = 1.0 - measured_stb_d / accumulated
    # Undo the velocity shape to report the anchor the correlation takes.
    ratio = well.slug_velocity_ft_min / OPTIMUM_SLUG_VELOCITY_FT_MIN
    penalty = FALLBACK_VELOCITY_CURVATURE * (ratio + 1.0 / ratio - 2.0)
    return observed * 1000.0 / well.depth_ft - penalty


@dataclass
class LiftChoice:
    """Which kind of lift this well wants, and on what evidence."""

    recommendation: str                  # "continuous" | "intermittent"
    intermittent_oil_stb_d: float
    continuous_oil_stb_d: float
    intermittent_gas_scf_d: float
    continuous_gas_scf_d: float
    reasons: list = field(default_factory=list)
    screened_by: str = ""
    # The Asheim verdict, when the inputs for it were supplied. None means
    # it was not evaluated, which is different from "stable".
    stability: object = None

    @property
    def margin(self) -> float:
        """How much more oil the recommended method makes, as a fraction."""
        best = max(self.intermittent_oil_stb_d, self.continuous_oil_stb_d)
        worst = min(self.intermittent_oil_stb_d, self.continuous_oil_stb_d)
        return (best - worst) / best if best > 0.0 else 0.0


def compare_with_continuous(well: IntermittentWell, tubing, oil: LiquidProperties,
                            gas: GasProperties, gas_available_scf_d: float,
                            oil_price: float = 1.0, gas_price: float = 0.0,
                            formation_gor: float = 0.0,
                            reservoir_psia: float = 0.0,
                            points: int = 24,
                            stability_inputs=None) -> LiftChoice:
    """Run both models on the same well and let the numbers choose.

    The screening rules say whether the question is worth asking. This
    answers it, and it is the answer worth trusting, because the rules of
    thumb are about typical wells and the reason a well is on intermittent
    lift is usually that it is not typical.

    Continuous lift is evaluated over the same gas the intermittent cycle
    would use, so the comparison is at equal compression rather than at
    each method's own best case -- which is the comparison an asset with a
    fixed compressor actually faces.

    **This comparison is biased towards continuous lift, and the bias is
    not small.** `nodal_analysis` returns the intersection of the inflow
    and tubing curves. It brackets on the high side, so it does not return
    the unstable lower root of the J-shaped tubing curve -- but a stable
    root is not the same as a stable well. Gas-lifted wells head: the
    casing and tubing exchange gas in a cycle the steady-state intersection
    knows nothing about, and a well can have a perfectly good nodal
    solution and still be unproducible in practice. Heading is precisely
    the failure that sends a well to intermittent lift in the first place,
    so the wells where the answer matters most are the wells where the
    continuous figure is least trustworthy.

    The test for it is Asheim's (1988) two stability criteria. Pass
    `stability_inputs` and they are evaluated: a well that fails both
    criteria is heading, and the recommendation flips to intermittent even
    where the continuous model predicts more oil, because the continuous
    figure is then a rate the well will not hold. Without them the caveat
    is reported in words and the continuous figure is labelled an upper
    bound, which is the most that can honestly be said.
    """
    from .wells import gas_lift_performance_curve

    best_cycle = optimise_cycles(well, oil_price=oil_price, gas_price=gas_price)
    intermittent_oil = best_cycle.best.oil_stb_d
    intermittent_gas = best_cycle.best.gas_scf_d

    reasons = []
    continuous_oil, continuous_gas = 0.0, 0.0
    try:
        ceiling = max(gas_available_scf_d, intermittent_gas)
        points = gas_lift_performance_curve(
            ipr=well.ipr, tubing=tubing,
            wellhead_pressure_psia=well.wellhead_psia,
            formation_gor_scf_stb=formation_gor,
            injection_rates_scf_d=list(np.linspace(0.0, ceiling, points)),
            water_cut=oil.water_cut, oil=oil, gas=gas)
        usable = [p for p in points if p.oil_rate_stb_d > 0.0]
        if usable:
            pick = max(usable, key=lambda p: oil_price * p.oil_rate_stb_d
                       - gas_price * p.injection_rate_scf_d)
            continuous_oil = float(pick.oil_rate_stb_d)
            continuous_gas = float(pick.injection_rate_scf_d)
        else:
            reasons.append(
                "continuous lift has no converged operating point on this "
                "well at any injection rate up to %.0f scf/d -- the usual "
                "signature of a well that cannot sustain a flowing column"
                % ceiling)
    except Exception as error:                        # noqa: BLE001
        # A continuous model that will not solve is itself the finding on a
        # low-pressure well, so it is reported rather than raised.
        reasons.append("the continuous model did not solve: %s" % error)

    verdict = None
    if stability_inputs is not None:
        from .stability import asheim_stability

        verdict = asheim_stability(stability_inputs)

    if verdict is not None and not verdict.stable and continuous_oil > 0.0:
        # The continuous model found a steady state the well cannot hold.
        recommendation = "intermittent"
        reasons.append(
            "continuous lift computes %.1f stb/d, but the well fails both "
            "Asheim criteria (%s), so that rate is a steady state it will "
            "not hold. Intermittent lift makes %.1f stb/d on a cycle that "
            "does not depend on one."
            % (continuous_oil, verdict.describe(), intermittent_oil))
        reasons.extend(verdict.reasons)
    elif continuous_oil > intermittent_oil:
        recommendation = "continuous"
        reasons.append(
            "continuous lift makes %.1f stb/d against %.1f on the best "
            "intermittent cycle, at comparable gas"
            % (continuous_oil, intermittent_oil))
        if verdict is not None:
            reasons.append("stability: " + verdict.describe())
        elif reservoir_psia > 0.0 and screen(well.ipr, reservoir_psia)[0]:
            reasons.append(
                "treat that with caution. This well meets the screening "
                "rules for intermittent lift, and the continuous figure "
                "comes from a steady-state intersection that does not test "
                "whether the well would head. Asheim's (1988) stability "
                "criteria are the test and this engine does not implement "
                "them, so the continuous number is an upper bound on what "
                "this well would actually deliver.")
    else:
        recommendation = "intermittent"
        reasons.append(
            "the best intermittent cycle (%.1f a day) makes %.1f stb/d "
            "against %.1f for continuous lift"
            % (best_cycle.cycles_per_day, intermittent_oil, continuous_oil))
    reasons.extend(best_cycle.notes)

    if verdict is not None and recommendation == "intermittent":
        reasons.append("stability: " + verdict.describe())

    return LiftChoice(
        recommendation=recommendation,
        stability=verdict,
        intermittent_oil_stb_d=intermittent_oil,
        continuous_oil_stb_d=continuous_oil,
        intermittent_gas_scf_d=intermittent_gas,
        continuous_gas_scf_d=continuous_gas,
        reasons=reasons, screened_by="both models run")


def screen(ipr: InflowPerformance, reservoir_psia: float) -> tuple:
    """A first look at whether intermittent lift is even the question.

    Rules of thumb, and labelled as such. The real answer is to run both
    models and compare, which is what `compare_with_continuous` does; this
    is here because it costs nothing and it tells an engineer whether the
    comparison is worth setting up.
    """
    reasons = []
    # Productivity index near the top of the drawdown range, which is where
    # a low-pressure well actually operates.
    probe = max(reservoir_psia * 0.9, 1.0)
    rate = max(float(ipr.rate(probe)), 0.0)
    pi = rate / (reservoir_psia - probe) if reservoir_psia > probe else 0.0
    open_flow = max(float(ipr.rate(ATM_PSIA)), 0.0)

    if pi < LOW_PRODUCTIVITY_PI:
        reasons.append(
            "productivity index is about %.2f stb/d/psi, below the %.1f at "
            "which continuous lift usually stops being the better choice"
            % (pi, LOW_PRODUCTIVITY_PI))
    if open_flow < LOW_RATE_STB_D:
        reasons.append(
            "absolute open flow is about %.0f stb/d, below the %.0f stb/d "
            "where continuous lift is normally preferred"
            % (open_flow, LOW_RATE_STB_D))
    return bool(reasons), reasons
