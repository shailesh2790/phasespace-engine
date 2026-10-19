"""The field, forward in time.

Everything upstream of this file answers a question about one instant. Material
balance says what the reservoir pressure is at a given cumulative production.
The integrated solver says what the wells and the gathering system deliver at a
given reservoir pressure. Neither alone is a forecast, and the loop between
them is:

    reservoir pressure -> inflow -> well and network solve -> rate
         ^                                                     |
         +------------------ cumulative production <------------+

That loop is what a production forecast is, and it is the thing a decline curve
cannot do. A decline curve extrapolates the trend a field happens to be on. It
cannot say what the trend would become if the separator pressure were dropped,
a compressor added, a well worked over, or three new wells drilled -- because
the trend is the only thing it knows. This answers those questions, because
every link in the loop is a physical model with the intervention in it.

Two honest limits, both reported rather than assumed away.

**The time step is an approximation.** Rate falls during a step, so using the
rate at the start of it over-produces, and that error compounds into the
reserves. The step is therefore taken by predictor-corrector, and the gap
between the predictor and the corrector is the local truncation error -- which
is measured and reported instead of being quoted as a tolerance. If the step
error is a significant share of the forecast, the answer needs smaller steps
and says so.

**The productivity index is not constant.** As a reservoir depletes, oil
viscosity rises and the oil volume factor falls, and both change how much oil
a given drawdown delivers. Holding J fixed is the usual simplification and it
over-forecasts a depleting oil field. Where PVT is available, J is scaled by
the mobility group that actually governs it; where it is not, J is held and the
assumption is named.

Units as elsewhere: pressure psia, oil STB/day, gas scf/day, time days.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .integrated import CoupledWell, WellState, solve_integrated
from .materialbalance import GasTank, OilTank
from .solver import Network
from .wells import (
    CompositeIPR,
    FetkovichIPR,
    GasBackpressureIPR,
    ProductivityIndexIPR,
    VogelIPR,
)

DAYS_PER_YEAR = 365.25

# Thresholds on the step-error *indicator*, which is the gap between the
# explicit and midpoint estimates of a step's production.
#
# The indicator is not the error of the answer. It is the error of the
# explicit estimate, which is first order in the step, while the midpoint
# value actually used is third order -- so the indicator overstates the real
# error by about two orders of magnitude. Both numbers below come from
# measuring that on a two-well undersaturated field over four years, against
# a run at an indicator target of 0.04:
#
#   indicator target   steps   cumulative production   error
#          0.30          11          4.5955 MMSTB      -1.04%
#          0.15          12          4.6374 MMSTB      -0.14%
#          0.08          16          4.6413 MMSTB      -0.05%
#          0.04          21          4.6437 MMSTB          --
#
# So 0.15 buys 0.1% accuracy in the reserves at half the cost of 0.04, which
# is well inside the uncertainty of the PVT underneath it, and the warning
# threshold sits where the measured error reaches about 1%.
STEP_ERROR_WARNING = 0.30

DEFAULT_MAX_STEP_ERROR = 0.15

# How far the step may be shortened, and the absolute floor. A reservoir that
# needs steps shorter than a week is being asked a question material balance
# cannot answer -- tank models assume the pressure has equilibrated.
MAX_STEP_HALVINGS = 5
MIN_STEP_DAYS = 7.0


@dataclass
class Compartment:
    """One tank and the wells that drain it.

    A field is rarely one tank and is almost never one tank per well. Grouping
    wells onto a shared compartment is what makes interference show up in a
    forecast: two wells on one tank deplete it twice as fast, and each sees the
    other's production as its own pressure decline.
    """

    name: str
    tank: object                           # OilTank or GasTank
    wells: list = field(default_factory=list)   # well names draining it
    produced_gor_scf_stb: float = 0.0      # cumulative gas / cumulative oil
    water_cumulative_stb: float = 0.0

    def __post_init__(self):
        if not isinstance(self.tank, (OilTank, GasTank)):
            raise ValueError(
                "a compartment's tank must be an OilTank or a GasTank, got %r"
                % (type(self.tank).__name__,))
        if not self.wells:
            raise ValueError("compartment %r has no wells draining it" % (self.name,))

    @property
    def is_oil(self) -> bool:
        return isinstance(self.tank, OilTank)

    @property
    def initial_psia(self) -> float:
        return self.tank.initial_psia

    def pressure_at(self, cumulative: float) -> float:
        """Tank pressure after this much production.

        For an oil tank the cumulative is STB of oil and the produced GOR and
        water carried on the compartment complete the withdrawal; for a gas
        tank it is scf.
        """
        if self.is_oil:
            gor = self.produced_gor_scf_stb or self.tank.rsi
            return self.tank.pressure_at(cumulative, gor,
                                         self.water_cumulative_stb)
        return self.tank.pressure_at(cumulative)

    def volume_in_place(self) -> float:
        return self.tank.ooip_stb if self.is_oil else self.tank.giip_scf


@dataclass
class ForecastStep:
    """The field at one instant, and how it got there."""

    day: float
    compartment_psia: dict = field(default_factory=dict)
    cumulative: dict = field(default_factory=dict)
    wells: list = field(default_factory=list)
    field_oil_stb_d: float = 0.0
    field_liquid_stb_d: float = 0.0
    converged: bool = True
    # The gap between the explicit and midpoint estimates of this step's
    # production. A bound on the explicit estimate's error, not on the
    # midpoint value reported -- see the thresholds at the top of the module.
    step_error_fraction: float = 0.0
    event: bool = False
    notes: list = field(default_factory=list)

    @property
    def year(self) -> float:
        return self.day / DAYS_PER_YEAR

    @property
    def flowing_count(self) -> int:
        return sum(1 for well in self.wells if well.flowing)


@dataclass
class ForecastResult:
    steps: list = field(default_factory=list)
    compartments: list = field(default_factory=list)
    step_days: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def final(self) -> ForecastStep:
        return self.steps[-1] if self.steps else None

    @property
    def cumulative_oil_stb(self) -> float:
        last = self.final
        if last is None:
            return 0.0
        return sum(volume for name, volume in last.cumulative.items()
                   if self._is_oil(name))

    @property
    def cumulative_gas_scf(self) -> float:
        last = self.final
        if last is None:
            return 0.0
        return sum(volume for name, volume in last.cumulative.items()
                   if not self._is_oil(name))

    def _is_oil(self, name) -> bool:
        for compartment in self.compartments:
            if compartment.name == name:
                return compartment.is_oil
        return True

    @property
    def recovery_factor(self) -> float:
        """Produced over in place, across every compartment of the same kind.

        Mixed oil and gas compartments are not summed -- STB and scf do not
        add -- so this reports the oil recovery when there is oil and the gas
        recovery otherwise.
        """
        oil = [c for c in self.compartments if c.is_oil]
        chosen = oil if oil else self.compartments
        in_place = sum(c.volume_in_place() for c in chosen)
        if in_place <= 0.0:
            return 0.0
        produced = (self.cumulative_oil_stb if oil else self.cumulative_gas_scf)
        return produced / in_place

    @property
    def worst_step_error(self) -> float:
        """The largest integration error, ignoring steps that contained an event.

        A step in which a well died reports a near-total disagreement between
        its predictor and its corrector, and that is not integration error --
        no step size integrates across a discontinuity. Counting it here made
        every forecast that ran a well to its death report 100% error and
        declare itself unreliable, which buried the real signal in the one
        case where the forecast was most worth reading.
        """
        return max((step.step_error_fraction for step in self.steps
                    if not step.event), default=0.0)

    @property
    def events(self) -> list:
        """Steps in which a well stopped flowing."""
        return [step for step in self.steps if step.event]

    @property
    def shut_ins(self) -> list:
        """(year, well name) for every well that stopped flowing, in order.

        Derived from the sequence of solved steps rather than from the
        corrector's view inside a step. A well usually crosses from flowing to
        not flowing between one step's solve and the next, not between a
        step's start and its midpoint, so detecting it only at the midpoint
        missed most of them -- including both deaths in the field this was
        first measured on.
        """
        found = []
        previous = None
        for step in self.steps:
            flowing = {well.name for well in step.wells if well.flowing}
            if previous is not None:
                for name in sorted(previous - flowing):
                    found.append((step.year, name))
            previous = flowing
        return found

    @property
    def natural_flow_ends_year(self) -> float:
        """When the last well stops flowing, or infinity if none does.

        One of the more useful numbers a coupled forecast produces, and one no
        decline curve can produce at all: the date is set by the tubing and
        the back pressure, not by the trend, so it moves when either changes.
        """
        for step in self.steps:
            if step.flowing_count == 0:
                return step.year
        return math.inf

    @property
    def reliable(self) -> bool:
        return (all(step.converged for step in self.steps)
                and self.worst_step_error <= STEP_ERROR_WARNING)


# ============================================================== the inflow ==


def baseline_inflow(wells: list) -> dict:
    """Snapshot the inflow parameters a forecast will rescale.

    Taken once, because every rescaling is relative to the *initial*
    conditions and not to the previous step. Applying the mobility ratio to
    whatever J currently holds compounds it: on an eight-step forecast of an
    undersaturated oil, where viscosity genuinely falls on the way down to the
    bubble point, the productivity index grew by 23% per step and the field
    rate rose as the reservoir depleted. The physics was right and the
    bookkeeping was wrong, which is the harder of the two to notice.
    """
    captured = {}
    for well in wells:
        ipr = well.ipr
        captured[well.name] = {
            "productivity_index": getattr(ipr, "productivity_index", 0.0),
            "max_rate": getattr(ipr, "max_rate", 0.0),
            "reservoir_pressure_psia": getattr(ipr, "reservoir_pressure_psia", 0.0),
            "min_rate_stb_d": well.min_rate_stb_d,
        }
    return captured


def _scale_inflow(well: CoupledWell, pressure_psia: float,
                  mobility_ratio: float, baseline: dict) -> None:
    """Point a well's IPR at the reservoir's current pressure.

    `mobility_ratio` is (mu_o Bo) at initial conditions over its value now:
    the group a productivity index is inversely proportional to. Below one it
    reduces J, which is what happens as an oil loses its dissolved gas and
    thickens. Passed in as 1.0 where no PVT is available, which holds J
    constant and over-forecasts -- named in the result rather than hidden.

    Always written from `baseline`, never from the IPR's current value.
    """
    ipr = well.ipr
    start = baseline[well.name]
    if isinstance(ipr, (ProductivityIndexIPR, CompositeIPR)):
        ipr.reservoir_pressure_psia = pressure_psia
        ratio = mobility_ratio if mobility_ratio > 0.0 else 1.0
        ipr.productivity_index = start["productivity_index"] * ratio
    elif isinstance(ipr, VogelIPR):
        # Vogel's own scaling: q_max = J p_r / 1.8, so with J fixed the
        # maximum rate is proportional to reservoir pressure. Holding q_max
        # constant while dropping p_r would make the well *more* productive
        # per psi of drawdown as the field depleted.
        initial_pressure = start["reservoir_pressure_psia"]
        if initial_pressure > 0.0:
            ipr.max_rate = start["max_rate"] * (pressure_psia / initial_pressure)
        ipr.reservoir_pressure_psia = pressure_psia
    elif isinstance(ipr, (FetkovichIPR, GasBackpressureIPR)):
        # C carries the permeability-thickness group and is held; the
        # pressure-squared difference does the work. For a gas well this is
        # the standard treatment and a good one, because gas mobility rises
        # as pressure falls and largely offsets the loss in C.
        ipr.reservoir_pressure_psia = pressure_psia
    else:
        raise ValueError(
            "well %s has an IPR this forecast does not know how to deplete "
            "(%s). Add it to _scale_inflow rather than letting it forecast at "
            "constant reservoir pressure." % (well.name, type(ipr).__name__))


def _mobility_ratio(compartment: Compartment, pressure_psia: float) -> float:
    """(mu_o Bo)_initial / (mu_o Bo)_now, or 1.0 where there is no PVT."""
    if not compartment.is_oil:
        return 1.0
    pvt = getattr(compartment.tank, "pvt", None)
    if pvt is None:
        return 1.0
    initial = compartment.initial_psia
    now = max(pressure_psia, 1.0)
    group_initial = pvt.viscosity(initial) * pvt.bo(initial)
    group_now = pvt.viscosity(now) * pvt.bo(now)
    if group_now <= 0.0:
        return 1.0
    return group_initial / group_now


# ================================================================ the loop ==


def forecast(
    network: Network,
    wells: list,
    compartments: list,
    years: float = 10.0,
    step_days: float = 182.625,
    economic_limit_stb_d: float = 0.0,
    corrector: bool = True,
    max_step_error: float = DEFAULT_MAX_STEP_ERROR,
    max_steps: int = 120,
    max_passes: int = 40,
) -> ForecastResult:
    """Step the field forward, re-solving wells and network at each pressure.

    Each step costs one integrated solve, or two with the corrector on. That
    is the real cost of a coupled forecast and the reason commercial packages
    make the same trade: there is no way to know what the network does at next
    year's reservoir pressure without solving it there.
    """
    if years <= 0.0:
        raise ValueError("forecast horizon must be positive, got %r" % (years,))
    if step_days <= 0.0:
        raise ValueError("step must be positive, got %r" % (step_days,))
    if not compartments:
        raise ValueError("a forecast needs at least one compartment to deplete")
    if max_step_error <= 0.0:
        raise ValueError("max_step_error must be positive, got %r"
                         % (max_step_error,))
    if not corrector and max_step_error < 1.0:
        # Without the corrector there is no error estimate, so there is
        # nothing for the adaptive step to adapt to. Said plainly rather than
        # silently taking fixed steps while accepting a tolerance.
        raise ValueError(
            "the step error can only be measured by the corrector, so "
            "corrector=False and a max_step_error are contradictory. Pass "
            "corrector=True, or accept fixed steps with max_step_error=1.0.")

    by_name = {well.name: well for well in wells}
    assigned = {}
    for compartment in compartments:
        for name in compartment.wells:
            if name not in by_name:
                raise ValueError(
                    "compartment %r drains well %r, which is not in the well "
                    "list" % (compartment.name, name))
            if name in assigned:
                raise ValueError(
                    "well %r is drained by both %r and %r. A well produces "
                    "from one compartment here; splitting its rate between "
                    "two tanks needs an allocation this does not invent."
                    % (name, assigned[name], compartment.name))
            assigned[name] = compartment.name
    orphans = sorted(set(by_name) - set(assigned))
    if orphans:
        raise ValueError(
            "these wells belong to no compartment, so nothing depletes as "
            "they produce: %s" % ", ".join(orphans))

    # A dry-gas well cannot go through this path, and the failure would be
    # silent: GasBackpressureIPR returns scf/day, while every stage downstream
    # of it -- the nodal solve, the multiphase traverse, the network injection
    # -- carries STB/day. The rate would be wrong by about five orders of
    # magnitude and would still solve. Refused rather than scaled, because the
    # fix is a gas-rate coupled path and not a conversion factor here.
    gas_iprs = [well.name for well in wells
                if isinstance(well.ipr, GasBackpressureIPR)]
    if gas_iprs:
        raise ValueError(
            "a coupled forecast cannot yet carry dry-gas wells (%s): the "
            "well and network solve works in liquid rates with gas as a "
            "ratio, so a gas deliverability curve in scf/day would be read as "
            "STB/day. Gas compartments here hold *associated* gas, produced "
            "with the oil. For a dry-gas field, use GasTank with the decline "
            "module instead." % ", ".join(gas_iprs))

    baseline = baseline_inflow(wells)
    result = ForecastResult(compartments=list(compartments), step_days=step_days)
    cumulative = {compartment.name: 0.0 for compartment in compartments}

    if any(_mobility_ratio(c, c.initial_psia * 0.5) == 1.0 and c.is_oil
           for c in compartments):
        result.notes.append(
            "One or more oil compartments have no PVT attached, so the "
            "productivity index is held constant as they deplete. Real oil "
            "thickens as it loses dissolved gas, so this forecast leans high.")

    horizon = years * DAYS_PER_YEAR
    day = 0.0
    current = min(step_days, horizon)
    floor_days = max(step_days / 2 ** MAX_STEP_HALVINGS, MIN_STEP_DAYS)
    halvings = 0

    try:
        halvings = _step_through(
            network, wells, compartments, result, cumulative, by_name,
            baseline, horizon, current, step_days, floor_days,
            economic_limit_stb_d, corrector, max_step_error, max_steps,
            max_passes, years)
    finally:
        # The wells are the caller's, and a forecast depletes them. Handed
        # back as they arrived so a second call is not a different experiment.
        restore_inflow(wells, baseline)

    if halvings:
        result.notes.append(
            "The step was shortened %d time(s) to hold the step-error "
            "indicator under %.0f%%, which is why the points are not evenly "
            "spaced." % (halvings, max_step_error * 100.0))
    if result.worst_step_error > STEP_ERROR_WARNING:
        result.notes.append(
            "The step-error indicator reaches %.0f%%, which puts the "
            "cumulative production about a percent out. That is integration "
            "error, not physics -- lower max_step_error if the reserves "
            "matter." % (result.worst_step_error * 100.0))
    shut_ins = result.shut_ins
    if shut_ins:
        # Mark the steps where the flowing set shrank, so the step-error
        # summary does not read a discontinuity as integration error.
        shut_years = {year for year, _name in shut_ins}
        for step in result.steps:
            if step.year in shut_years:
                step.event = True
        result.notes.append(
            "%s stopped flowing during the forecast (%s). Those dates are set "
            "by the tubing and the back pressure the field works against, so "
            "artificial lift or a lower separator pressure moves them -- "
            "which is the question a decline curve cannot answer."
            % ("%d wells" % len(shut_ins) if len(shut_ins) > 1 else "1 well",
               ", ".join("%s at year %.1f" % (name, year)
                         for year, name in shut_ins)))
    if not all(step.converged for step in result.steps):
        result.notes.append(
            "One or more steps did not converge, so the cumulative production "
            "after them is built on an unconverged rate.")
    return result


def _step_through(network, wells, compartments, result, cumulative, by_name,
                  baseline, horizon, current, step_days, floor_days,
                  economic_limit_stb_d, corrector, max_step_error, max_steps,
                  max_passes, years):
    """Walk the field forward. Returns how many times the step was shortened."""
    day = 0.0
    halvings = 0

    while True:
        pressures = {c.name: c.pressure_at(cumulative[c.name])
                     for c in compartments}
        states, converged = _solve_at(network, wells, compartments, pressures,
                                      economic_limit_stb_d, max_passes,
                                      baseline)

        step = ForecastStep(
            day=day,
            compartment_psia=dict(pressures),
            cumulative=dict(cumulative),
            wells=states,
            field_oil_stb_d=sum(s.oil_rate_stb_d for s in states if s.flowing),
            field_liquid_stb_d=sum(s.liquid_rate_stb_d for s in states if s.flowing),
            converged=converged)
        result.steps.append(step)

        if day >= horizon - 1e-9:
            break
        if len(result.steps) >= max_steps:
            step.notes.append(
                "Stopped at the %d-step limit, %.1f years into a %.1f-year "
                "horizon. Each step is a full well-and-network solve, so the "
                "limit is a cost ceiling rather than a numerical one -- raise "
                "it, or lengthen the step, to go further."
                % (max_steps, day / DAYS_PER_YEAR, years))
            break
        if step.flowing_count == 0:
            step.notes.append(
                "No well can flow at this reservoir pressure against this "
                "network. The field is finished before the horizon.")
            break

        # Take the step, shortening it until the integration error is small.
        #
        # The decline is steepest at the start -- an undersaturated oil field
        # can lose a third of its pressure in the first six months -- so a
        # step size that is right for year five is badly wrong for year one.
        # A uniform step either wastes most of the forecast's compute on the
        # flat tail or gets the early years wrong, and the early years are
        # where the production is. Halving on the measured error puts the
        # effort where the curvature is, which is what a reservoir simulator
        # does for the same reason.
        attempt = min(current, horizon - day)
        while True:
            predictor, corrected, error, step_converged, died = _advance(
                network, wells, compartments, cumulative, pressures, states,
                attempt, by_name, economic_limit_stb_d, max_passes, baseline,
                corrector)
            if (not corrector or error <= max_step_error
                    or attempt <= floor_days * (1.0 + 1e-9)):
                break
            attempt = max(attempt * _shrink(error, max_step_error), floor_days)
            halvings += 1

        step.step_error_fraction = error
        step.converged = step.converged and step_converged
        at_floor = attempt <= floor_days * (1.0 + 1e-9)
        produced = corrected

        # The midpoint rule is not valid across a discontinuity, and a well
        # dying inside the step is one.
        #
        # When that happens the corrector solves a field in which the well has
        # already stopped and returns zero production, while the predictor --
        # solving at the start of the step, where it was still flowing --
        # returns a real rate. Taking the corrector then means the cumulative
        # does not advance, so the next step starts from the same pressure and
        # the same thing happens again: the forecast stalls at that pressure
        # and walks to the step limit in floor-sized steps without time
        # passing. That is what this did before the check below.
        #
        # The predictor is used instead. It over-produces, but only over a
        # floor-length step, and the well's death is then registered by the
        # next solve rather than preventing one.
        # Recording the event and handling the stall are separate.
        #
        # A well dying inside a step is always worth reporting -- it is one of
        # the forecast's most useful outputs. Whether the step also has to
        # fall back to the explicit estimate is a different question, and it
        # only arises when the corrector produces nothing at all: then the
        # cumulative does not advance, the next step starts from the same
        # pressure, and the forecast walks to its step limit without time
        # passing. Tying the report to the fallback meant a well death that
        # happened to need no fallback went unreported.
        if died:
            step.event = True
            step.notes.append(
                "%s stopped flowing during this step."
                % ", ".join(died))

        if _total(corrected) <= 0.0 < _total(predictor):
            produced = predictor
            step.notes.append(
                "Every well that was flowing had stopped by the midpoint of "
                "this step, so it is taken on the rate at its start. "
                "Averaging across the event would advance time by nothing at "
                "all; over %.0f days this over-produces by less than the "
                "event itself costs." % attempt)
        elif error > max_step_error and at_floor and not died:
            step.notes.append(
                "Still %.1f%% integration error at the %.0f-day floor. "
                "Something changes discontinuously inside this step and no "
                "step size integrates across that." % (error * 100.0, attempt))

        if _total(produced) <= 0.0 and step.flowing_count > 0:
            # Wells are flowing and nothing is being produced, so the next
            # step would be identical to this one. Stopped rather than looped.
            step.notes.append(
                "Wells are flowing but the step produced nothing, so the "
                "forecast cannot advance. This is a bug rather than a field "
                "condition; the forecast stops here instead of repeating.")
            break

        for name in cumulative:
            cumulative[name] += produced[name]
        day += attempt

        # The step that would just meet the target is the current one scaled
        # by the error ratio, the indicator being first order in the step.
        # Using a controller rather than an ad-hoc doubling matters at a
        # discontinuity: the first version kept the step at the floor after a
        # well died inside one, because the error stays near 1.0 there however
        # short the step gets, and an eight-year forecast then took four
        # hundred seven-day steps.
        current = min(attempt * _grow(error, max_step_error), step_days)

    return halvings


def _total(produced: dict) -> float:
    return sum(abs(volume) for volume in produced.values())


def _shrink(error: float, target: float) -> float:
    """How much to shorten a rejected step.

    The indicator is first order in the step, not second: the gap between a
    left-endpoint sum and a midpoint sum is the *left-endpoint* error, which
    goes as h^2 over a step and so as h relative to the step's production.
    Controlling it with a square-root exponent, as if it were the midpoint
    rule's own error, under-corrects and leaves the step pinned.
    """
    if error <= 0.0:
        return 1.0
    return max(min(0.8 * (target / error), 0.7), 0.25)


def _grow(error: float, target: float) -> float:
    """How much to lengthen the next step after an accepted one.

    Capped at four, because a step that grows without limit overshoots the
    next piece of curvature and is immediately rejected again. Floored at one
    so an accepted step never shrinks the next one -- that is the retry
    loop's job, and doing it here as well is how the step ratchets down to the
    floor and stays there.
    """
    if error <= 0.0:
        return 4.0
    return max(min(0.9 * (target / error), 4.0), 1.0)


def _advance(network, wells, compartments, cumulative, pressures, states,
             attempt_days, by_name, economic_limit, max_passes, baseline,
             corrector):
    """Production over one trial step, and the error that step size costs.

    The predictor uses the rate at the start of the step, which over-produces
    because the rate falls during it. The corrector re-solves at the midpoint
    cumulative and uses that rate across the whole step -- the midpoint rule.
    Their difference is a direct measure of the local truncation error, which
    is why it is reported rather than a tolerance being claimed.
    """
    produced = _produced_over_step(
        compartments, {s.name: s for s in states}, attempt_days, by_name)
    if not corrector:
        return produced, produced, 0.0, True, []

    midpoint = {name: cumulative[name] + 0.5 * produced[name]
                for name in cumulative}
    mid_pressures = {c.name: c.pressure_at(midpoint[c.name])
                     for c in compartments}
    mid_states, converged = _solve_at(
        network, wells, compartments, mid_pressures, economic_limit,
        max_passes, baseline)
    corrected = _produced_over_step(
        compartments, {s.name: s for s in mid_states}, attempt_days, by_name)

    # Which wells stopped flowing between the start of the step and its
    # midpoint. Compared as a set rather than inferred from how much the
    # production fell: a two-well field losing one well halves its rate, and a
    # threshold on the ratio put that exactly on the boundary -- the step where
    # the first well died was reported as 48% integration error instead of as
    # the event it was.
    flowing_before = {s.name for s in states if s.flowing}
    flowing_after = {s.name for s in mid_states if s.flowing}
    died = sorted(flowing_before - flowing_after)

    error = 0.0
    for name in cumulative:
        scale = max(abs(produced[name]), abs(corrected[name]), 1.0)
        error = max(error, abs(corrected[name] - produced[name]) / scale)

    # The IPRs were last written at the midpoint pressure. Restored, because
    # the caller's next solve starts from the step's own pressure and a stale
    # midpoint IPR would silently shift it.
    _write_inflow(wells, compartments, pressures, baseline)
    return produced, corrected, error, converged, died


def restore_inflow(wells: list, baseline: dict) -> None:
    """Put every IPR back to the state `baseline_inflow` captured.

    A forecast depletes the wells it is given, which leaves their IPRs at the
    last pressure it reached. Without this, a second forecast on the same
    wells snapshots those depleted values as its starting point and returns a
    different answer -- 4.9% lower cumulative on the two-well field this was
    found on, from calling the same function twice with the same arguments.
    """
    for well in wells:
        start = baseline.get(well.name)
        if start is None:
            continue
        ipr = well.ipr
        if hasattr(ipr, "productivity_index"):
            ipr.productivity_index = start["productivity_index"]
        if hasattr(ipr, "max_rate"):
            ipr.max_rate = start["max_rate"]
        if hasattr(ipr, "reservoir_pressure_psia"):
            ipr.reservoir_pressure_psia = start["reservoir_pressure_psia"]
        well.min_rate_stb_d = start["min_rate_stb_d"]


def _write_inflow(wells, compartments, pressures, baseline):
    by_name = {well.name: well for well in wells}
    for compartment in compartments:
        pressure = pressures[compartment.name]
        ratio = _mobility_ratio(compartment, pressure)
        for name in compartment.wells:
            _scale_inflow(by_name[name], pressure, ratio, baseline)


def _solve_at(network, wells, compartments, pressures, economic_limit,
              max_passes, baseline):
    """Set every IPR to its compartment's pressure and solve the field."""
    by_name = {well.name: well for well in wells}
    for compartment in compartments:
        pressure = pressures[compartment.name]
        ratio = _mobility_ratio(compartment, pressure)
        for name in compartment.wells:
            well = by_name[name]
            _scale_inflow(well, pressure, ratio, baseline)
            if economic_limit > 0.0:
                well.min_rate_stb_d = max(well.min_rate_stb_d, economic_limit)

    solved = solve_integrated(network, wells, max_passes=max_passes)
    return solved.wells, solved.converged


def _produced_over_step(compartments, states, step_days, wells_by_name):
    """Volume each compartment gives up over one step.

    Oil compartments are charged in STB of oil. Gas compartments are charged
    in scf of *associated* gas -- the oil rate multiplied by that well's
    gas-oil ratio -- because that is the only way gas leaves a reservoir in
    this engine's coupled path, which carries liquid rates throughout and
    represents gas as a ratio to them.
    """
    produced = {}
    for compartment in compartments:
        total = 0.0
        for name in compartment.wells:
            state = states.get(name)
            if state is None or not state.flowing:
                continue
            if compartment.is_oil:
                total += state.oil_rate_stb_d * step_days
            else:
                gor = wells_by_name[name].gas_oil_ratio_scf_stb
                total += state.oil_rate_stb_d * gor * step_days
        produced[compartment.name] = total
    return produced


# ============================================================ convenience ===


def forecast_single_tank(
    network: Network,
    wells: list,
    tank,
    years: float = 10.0,
    step_days: float = 182.625,
    produced_gor_scf_stb: float = 0.0,
    economic_limit_stb_d: float = 0.0,
    max_step_error: float = DEFAULT_MAX_STEP_ERROR,
    max_steps: int = 120,
) -> ForecastResult:
    """The common case: every well on one tank.

    Worth a named entry point because the common case is also the one where
    interference matters most -- every well in the field competing for the
    same pressure -- and because assembling a single-compartment list by hand
    invites the wells list and the compartment's well names to disagree.
    """
    compartment = Compartment(
        name="field", tank=tank, wells=[well.name for well in wells],
        produced_gor_scf_stb=produced_gor_scf_stb)
    return forecast(network, wells, [compartment], years=years,
                    step_days=step_days,
                    economic_limit_stb_d=economic_limit_stb_d,
                    max_step_error=max_step_error, max_steps=max_steps)
