"""Fit a well model to that well's own tests, instead of hand-tuning it.

Every production engineer who uses a nodal package does this by hand. They
build the well, run it, find the predicted rate is 15% off the last well test,
and then adjust something -- usually the productivity index, sometimes a
correlation multiplier, often whichever one they adjusted last time -- until
the number matches. It takes an afternoon per well, it is not reproducible,
and the model drifts out of date the week after it is built.

That loop is mechanical and belongs in the engine. What makes it worth doing
properly rather than just wrapping an optimiser around a solver is that the
mechanical version is *easy to do badly*, in ways that produce a model which
matches every test and predicts nothing.

**Three parameters, chosen because they are the three that are genuinely
uncertain.**

  productivity index    the reservoir's deliverability, which nobody measures
                        directly
  gravity multiplier    on the hydrostatic term of the tubing correlation
  friction multiplier   on the friction term

The two multipliers are separate rather than one lumped correction because
gravity dominates at low rate and friction at high rate. One factor cannot
fix both ends of a well's range; it trades one off against the other and
looks fine on the tests it was fitted to.

**The traps, each of which this refuses rather than walks into.**

*Fitting more than the data supports.* Three tests and three parameters is an
exact fit through three points: the residual is zero and the model has
learned nothing. There must be more independent measurements than unknowns,
and comfortably more.

*Fitting parameters that cannot be told apart.* This is the subtle one. If
every test is at roughly the same rate, a higher productivity index and a
lighter fluid column produce the *same* predicted rate -- the optimiser
slides along a ridge and reports wherever it stopped. The fit looks excellent
and the parameters are arbitrary, so the model fails the moment it is asked
about a rate it was not fitted at, which is the only reason anyone built it.
`identifiability` says so before any fitting happens, and the fit reports the
parameter correlations afterwards.

A measured bottomhole pressure breaks that degeneracy outright: it pins the
tubing side directly, so the inflow and outflow stop competing to explain one
number. Measured on a synthetic well with known parameters, five tests
spanning a 100% rate range, 4% noise on each rate, over five noise seeds:

                    without a survey      with one        true
    productivity    2.78 +/- 0.26         2.62 +/- 0.03   2.60
    gravity         1.129 +/- 0.009       1.121 +/- 0.006 1.120
    friction        1.008 +/- 0.158       0.899 +/- 0.057 0.920
    worst correlation    0.989                 0.833

One flowing gradient survey cuts the scatter in the productivity index by
nine times and moves the friction multiplier from wrong to right. It is worth
more here than several more well tests.

*Letting a multiplier absorb a modelling error.* The multipliers are bounded,
and a fit that pushes one to its bound is reported as a mismatch rather than
as a calibration. A correlation 40% out on a well is not a correlation that
needs scaling; it is the wrong correlation, or the inputs are wrong.

numpy only, deliberately. A calibration that needs scipy cannot run inside a
customer's network, and "runs on your own hardware" is the promise this is
supposed to support.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fluids import GasProperties, LiquidProperties
from .wells import (
    InflowPerformance,
    Tubing,
    _pwf_for_rate,
    nodal_analysis,
    pressure_traverse,
)

# How far a correlation may be scaled before the honest answer is that it is
# the wrong correlation. The PRD's +/-20% band: a multiphase correlation
# fitted on someone else's test loop is routinely 10-20% out on a given well,
# and past that the error is structural rather than a scale factor.
MIN_MULTIPLIER = 0.80
MAX_MULTIPLIER = 1.20

# The productivity index is a property of the reservoir and the completion,
# not a fudge factor, but the value a model starts with is usually a guess
# from an offset well. A wide band is right here; the bound exists to stop
# the optimiser running away, not to express a belief.
MIN_PI_SCALE = 0.20
MAX_PI_SCALE = 5.00

# Correlation above which two parameters are not separately determined by the
# data. 0.9 already means nine parts in ten of one can be absorbed by the
# other.
CONFOUNDED_CORRELATION = 0.90

# Relative spread of test rates below which the tests are all effectively at
# one operating point, and the gravity and friction terms cannot be told
# apart whatever the optimiser reports.
MIN_RATE_SPREAD = 0.25

PRODUCTIVITY = "productivity_index"
GRAVITY = "gravity_multiplier"
FRICTION = "friction_multiplier"
BOUNDS = {
    PRODUCTIVITY: (MIN_PI_SCALE, MAX_PI_SCALE),
    GRAVITY: (MIN_MULTIPLIER, MAX_MULTIPLIER),
    FRICTION: (MIN_MULTIPLIER, MAX_MULTIPLIER),
}


@dataclass
class WellTest:
    """One measured operating point.

    `bottomhole_psia` is optional and is worth far more than another test
    without it: it pins the tubing side directly and so separates the inflow
    from the outflow, which rate and wellhead pressure alone cannot do.
    """

    rate_stb_d: float          # total liquid, as a test separator reports it
    wellhead_psia: float
    water_cut: float = 0.0
    gas_oil_ratio_scf_stb: float = 0.0
    bottomhole_psia: float = 0.0
    label: str = ""

    def __post_init__(self):
        if self.rate_stb_d <= 0.0:
            raise ValueError("a test with no rate is not an operating point "
                             "(%r)" % (self.label or self.rate_stb_d,))
        if self.wellhead_psia <= 0.0:
            raise ValueError("wellhead pressure must be positive, got %r"
                             % (self.wellhead_psia,))
        if not 0.0 <= self.water_cut < 1.0:
            raise ValueError("water cut must be in [0, 1), got %r"
                             % (self.water_cut,))


@dataclass
class IdentifiabilityReport:
    """What this set of tests can and cannot determine, before fitting."""

    tests: int
    residuals: int
    parameters: list
    rate_spread: float
    has_bottomhole: bool
    supported: bool
    reasons: list = field(default_factory=list)

    @property
    def degrees_of_freedom(self) -> int:
        return self.residuals - len(self.parameters)


def identifiability(tests: list, parameters: list = None) -> IdentifiabilityReport:
    """Can these tests support fitting these parameters? Asked before fitting.

    An optimiser handed an under-determined problem still returns numbers,
    and they look exactly as authoritative as a well-posed fit. The two
    conditions below are both necessary, and neither is about how well the
    fit matches.
    """
    parameters = list(parameters or [PRODUCTIVITY, GRAVITY, FRICTION])
    unknown = [name for name in parameters if name not in BOUNDS]
    if unknown:
        raise ValueError("unknown parameter(s) %s; choose from %s"
                         % (unknown, sorted(BOUNDS)))
    if not tests:
        raise ValueError("no well tests supplied")

    rates = np.array([test.rate_stb_d for test in tests], dtype=float)
    with_bottomhole = [t for t in tests if t.bottomhole_psia > 0.0]
    residuals = len(tests) + len(with_bottomhole)
    spread = float((rates.max() - rates.min()) / rates.mean()) if rates.mean() else 0.0

    reasons = []
    supported = True

    # More measurements than unknowns, and not by one. An exact fit through
    # as many points as it has parameters has a zero residual and no content.
    if residuals <= len(parameters):
        supported = False
        reasons.append(
            "%d measurement(s) against %d parameter(s): the fit would pass "
            "exactly through the data and have learned nothing. Add tests, "
            "add a flowing bottomhole pressure, or fit fewer parameters."
            % (residuals, len(parameters)))
    elif residuals < len(parameters) + 2:
        reasons.append(
            "only %d measurement(s) for %d parameter(s). The fit will be "
            "dominated by whichever test is least representative."
            % (residuals, len(parameters)))

    # Excitation. At one operating point the gravity and friction terms are
    # indistinguishable, because what separates them is how each responds to
    # changing rate.
    if GRAVITY in parameters and FRICTION in parameters:
        if spread < MIN_RATE_SPREAD and not with_bottomhole:
            supported = False
            reasons.append(
                "the tests span only %.0f%% in rate. Gravity dominates at low "
                "rate and friction at high rate, so at one operating point "
                "they cannot be told apart -- the optimiser will report "
                "whichever combination it reached first. Fit one multiplier, "
                "or test the well at a second rate."
                % (spread * 100.0))
        elif spread < MIN_RATE_SPREAD:
            reasons.append(
                "the tests span only %.0f%% in rate, so the two tubing "
                "multipliers are weakly separated. The measured bottomhole "
                "pressure is doing most of the work."
                % (spread * 100.0))

    if not with_bottomhole and len(parameters) > 1:
        reasons.append(
            "no measured bottomhole pressure. Rate and wellhead pressure "
            "alone leave the inflow and the tubing competing to explain the "
            "same number; one flowing gradient survey separates them better "
            "than several more well tests would.")

    return IdentifiabilityReport(
        tests=len(tests), residuals=residuals, parameters=parameters,
        rate_spread=spread, has_bottomhole=bool(with_bottomhole),
        supported=supported, reasons=reasons)


@dataclass
class CalibrationResult:
    """The fitted well, and how much of it the data actually determined."""

    parameters: dict
    rms_before: float
    rms_after: float
    per_test: list = field(default_factory=list)
    correlations: dict = field(default_factory=dict)
    at_bound: list = field(default_factory=list)
    iterations: int = 0
    identifiability: IdentifiabilityReport = None
    notes: list = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.rms_after < self.rms_before

    @property
    def improvement(self) -> float:
        if self.rms_before <= 0.0:
            return 0.0
        return 1.0 - self.rms_after / self.rms_before

    @property
    def trustworthy(self) -> bool:
        """Fitted well, determined by the data, and inside the physics."""
        return (self.improved and not self.at_bound
                and (self.identifiability is None
                     or self.identifiability.supported)
                and not self.confounded)

    @property
    def confounded(self) -> list:
        return [pair for pair, value in self.correlations.items()
                if abs(value) > CONFOUNDED_CORRELATION]


def calibrate_well(
    ipr: InflowPerformance,
    tubing: Tubing,
    tests: list,
    oil: LiquidProperties = None,
    gas: GasProperties = None,
    correlation: str = None,
    parameters: list = None,
    default_gor_scf_stb: float = 600.0,
    max_iterations: int = 40,
) -> CalibrationResult:
    """Fit the well to its own tests.

    The forward model is the ordinary nodal solve, so what is calibrated is
    the same model the rest of the engine runs -- not a reduced stand-in that
    agrees with it only near the fitted point.

    Levenberg-Marquardt with a finite-difference Jacobian. Autodiff would be
    the textbook answer and is not needed at three parameters: a gradient
    costs four forward solves, each a few hundred milliseconds, and the
    bisection inside the nodal solve is not differentiable anyway. Where
    autodiff *would* earn its keep is history-matching a gridded model with
    thousands of parameters, which is a different problem.
    """
    parameters = list(parameters or [PRODUCTIVITY, GRAVITY, FRICTION])
    oil = oil or LiquidProperties()
    gas = gas or GasProperties()
    if not tests:
        raise ValueError("no well tests supplied")

    report = identifiability(tests, parameters)

    base_pi = _inflow_scale(ipr)
    if PRODUCTIVITY in parameters and base_pi is None:
        raise ValueError(
            "this IPR (%s) has no single deliverability parameter to scale. "
            "Fit the tubing multipliers only, or use a PI-based IPR."
            % type(ipr).__name__)

    start = np.array([1.0 for _ in parameters], dtype=float)
    lower = np.array([BOUNDS[name][0] for name in parameters], dtype=float)
    upper = np.array([BOUNDS[name][1] for name in parameters], dtype=float)

    def residuals(values: np.ndarray) -> np.ndarray:
        settings = dict(zip(parameters, values))
        return _residuals(ipr, tubing, tests, oil, gas, correlation, settings,
                          base_pi, default_gor_scf_stb)

    fitted, iterations = _levenberg_marquardt(
        residuals, start, lower, upper, max_iterations)

    settings = dict(zip(parameters, fitted))
    jacobian = _jacobian(residuals, fitted, lower, upper)
    correlations = _correlations(jacobian, parameters)

    # Scored in rate, with a full nodal solve per test, at the start and at
    # the finish. The optimiser works in the gap between the two pressure
    # curves because that is cheap and well-behaved; nobody reports a well in
    # those units. "Predicted 840 against a measured 900" is the sentence an
    # engineer checks, so it is worth 2n extra solves once at the end.
    before = _rate_errors(ipr, tubing, tests, oil, gas, correlation,
                          dict(zip(parameters, start)), base_pi,
                          default_gor_scf_stb)
    after = _rate_errors(ipr, tubing, tests, oil, gas, correlation, settings,
                         base_pi, default_gor_scf_stb)

    errors_before = np.array([row["error"] for row in before], dtype=float)
    errors_after = np.array([row["error"] for row in after], dtype=float)

    # A calibration that leaves the model worse than it found it is not a
    # calibration. The objective is the rate error only to first order --
    # the gap-to-rate conversion is a local slope -- so a fit can still come
    # out behind, and reporting the fitted numbers anyway would hand back a
    # worse model with a confident label on it.
    if _rms(errors_after) >= _rms(errors_before):
        settings = dict(zip(parameters, start))
        after, errors_after, fitted = before, errors_before, start

    at_bound = []
    for name, value in zip(parameters, fitted):
        low, high = BOUNDS[name]
        if value <= low * (1.0 + 1e-6) or value >= high * (1.0 - 1e-6):
            at_bound.append(name)

    per_test = []
    for test, start_row, final_row in zip(tests, before, after):
        per_test.append({
            "label": test.label or ("%.0f STB/d" % test.rate_stb_d),
            "rate_stb_d": test.rate_stb_d,
            "wellhead_psia": test.wellhead_psia,
            "predicted_before": start_row["predicted"],
            "predicted_after": final_row["predicted"],
            "error_before": start_row["error"],
            "error_after": final_row["error"],
            "flows_after": final_row["flows"],
        })

    result = CalibrationResult(
        parameters=_report_parameters(settings, base_pi),
        rms_before=_rms(errors_before),
        rms_after=_rms(errors_after),
        per_test=per_test, correlations=correlations, at_bound=at_bound,
        iterations=iterations, identifiability=report)

    _annotate(result, parameters)
    return result


def apply_calibration(result: CalibrationResult, ipr: InflowPerformance) -> dict:
    """Push the fitted inflow onto the IPR, and hand back the tubing factors.

    Split because they live in different places: the productivity index
    belongs to the IPR object, while the two multipliers are arguments to
    every traverse. Returning the latter rather than stashing them somewhere
    keeps it visible that an uncalibrated call is still possible -- the
    multipliers have to be passed on, which is the honest state of affairs
    until a well object owns them.
    """
    scale = result.parameters.get("productivity_scale")
    if scale is not None:
        _scale_inflow(ipr, scale)
    return {
        "gravity_multiplier": result.parameters.get(GRAVITY, 1.0),
        "friction_multiplier": result.parameters.get(FRICTION, 1.0),
    }


# ================================================================ internals ==


def _inflow_scale(ipr: InflowPerformance):
    """The single number that scales this IPR's deliverability, if it has one."""
    for attribute in ("productivity_index", "max_rate", "c"):
        if hasattr(ipr, attribute):
            return (attribute, float(getattr(ipr, attribute)))
    return None


def _scale_inflow(ipr: InflowPerformance, scale: float) -> None:
    found = _inflow_scale(ipr)
    if found is None:
        return
    attribute, _value = found
    setattr(ipr, attribute, getattr(ipr, attribute) * scale)


def _with_scaled_inflow(ipr, base_pi, scale):
    """A copy of the IPR with its deliverability scaled.

    Copied rather than mutated and restored: an exception part way through a
    fit would otherwise leave the caller's IPR holding a trial value from
    whichever iteration happened to fail.
    """
    import copy

    if base_pi is None or scale == 1.0:
        return ipr
    attribute, original = base_pi
    clone = copy.copy(ipr)
    setattr(clone, attribute, original * scale)
    return clone


def _residuals(ipr, tubing, tests, oil, gas, correlation, settings, base_pi,
               default_gor):
    """How far apart the inflow and outflow curves sit at each measured rate.

    **Not** a nodal solve per test, which is what this did first and why a
    five-test fit took longer than ten minutes. A nodal analysis bisects to
    *find* the rate where the two curves meet -- and the rate is the thing
    that was measured. Searching for a number already in hand, a thousand
    times over inside an optimiser loop, is the whole cost of the original
    version and none of the information.

    Evaluated directly instead: at the measured rate, the tubing needs one
    bottomhole pressure and the reservoir supplies another, and the gap
    between them is the error. At the true parameters it is zero, which is
    what "the well tests at this rate" means. One traverse per test, no
    bisection, no convergence failures, and it is also the quantity an
    engineer reads off a nodal plot by eye.

    A measured bottomhole pressure splits that one residual into two, and
    that is exactly why a gradient survey is worth more than another test:

        tubing side     traverse against the measured pressure -- constrains
                        the two multipliers and nothing else
        inflow side     IPR against the measured pressure -- constrains the
                        productivity index and nothing else

    Without it the two sides only ever appear as a difference, and any pair
    that keeps the difference fixed fits equally well.

    Relative to the inflow pressure so that tests at very different drawdowns
    weigh comparably, and so rate and pressure residuals share a scale
    without an invented weighting.
    """
    scale = settings.get(PRODUCTIVITY, 1.0)
    gravity = settings.get(GRAVITY, 1.0)
    friction = settings.get(FRICTION, 1.0)
    scaled_ipr = _with_scaled_inflow(ipr, base_pi, scale)
    kwargs = {} if correlation is None else {"correlation": correlation}

    errors = []
    for test in tests:
        gor = test.gas_oil_ratio_scf_stb or default_gor
        stream = _stream(test, gor, oil, gas, default_gor)
        tubing_pwf = pressure_traverse(
            test.wellhead_psia, stream, tubing,
            gravity_multiplier=gravity, friction_multiplier=friction,
            **kwargs).bottomhole_pressure_psia
        # The IPR is in oil rate; a well test is reported in liquid. Feeding
        # one to the other is a factor of (1 - water cut), which at 25% water
        # is 33% -- and it does not announce itself, because the optimiser
        # simply scales the productivity index up to absorb it and reports a
        # confident, wrong deliverability.
        inflow_pwf = _pwf_for_rate(
            scaled_ipr, _oil_rate(test), scaled_ipr.absolute_open_flow())

        if test.bottomhole_psia > 0.0:
            measured = test.bottomhole_psia
            errors.append((tubing_pwf - measured) / measured)
            errors.append((inflow_pwf - measured) / measured)
        else:
            # Converted from a pressure gap into the rate error it implies,
            # by the local slope of the gap against rate.
            #
            # The two are not interchangeable, and assuming they were made
            # the fit actively worse on noisy data: a *small* pressure gap
            # at a shallow intersection is a *large* rate error, because
            # near tangency a few psi moves the operating point hundreds of
            # barrels. Minimising pressure while scoring rate let the
            # optimiser trade a big rate error for a small pressure one and
            # call it progress -- on one noise seed the rate error went from
            # 28% to 51% and the fit reported itself converged.
            #
            # One extra traverse per test buys the slope, and the objective
            # then agrees with what is reported. Where the curves really are
            # tangent the implied rate error is genuinely enormous, which is
            # the right answer: that operating point is not determined.
            gap = tubing_pwf - inflow_pwf
            nudge = 1.05
            bumped = _curve_gap(ipr=scaled_ipr, tubing=tubing, test=test,
                                rate=test.rate_stb_d * nudge, gor=gor,
                                oil=oil, gas=gas, gravity=gravity,
                                friction=friction, kwargs=kwargs)
            slope = (bumped - gap) / (test.rate_stb_d * (nudge - 1.0))
            if abs(slope) < 1e-9:
                # Parallel curves: no rate closes the gap. Reported as a
                # large but finite error so the optimiser can still move.
                errors.append(math.copysign(2.0, gap))
            else:
                errors.append(float(-gap / slope) / test.rate_stb_d)

    return np.array(errors, dtype=float)


def _curve_gap(ipr, tubing, test, rate, gor, oil, gas, gravity, friction,
               kwargs) -> float:
    """VLP minus IPR bottomhole pressure at one rate."""
    from .wells import ProductionStream

    oil_rate = rate * (1.0 - test.water_cut)
    stream = ProductionStream(
        oil_rate_stb_d=oil_rate, water_rate_stb_d=rate - oil_rate,
        gas_rate_scf_d=oil_rate * gor, oil=oil, gas=gas)
    tubing_pwf = pressure_traverse(
        test.wellhead_psia, stream, tubing, gravity_multiplier=gravity,
        friction_multiplier=friction, **kwargs).bottomhole_pressure_psia
    inflow_pwf = _pwf_for_rate(ipr, oil_rate, ipr.absolute_open_flow())
    return tubing_pwf - inflow_pwf


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _rate_errors(ipr, tubing, tests, oil, gas, correlation, settings,
                 base_pi, default_gor):
    """Predicted against measured rate, one proper nodal solve per test."""
    scaled_ipr = _with_scaled_inflow(ipr, base_pi, settings.get(PRODUCTIVITY, 1.0))
    kwargs = {} if correlation is None else {"correlation": correlation}
    rows = []
    for test in tests:
        solved = nodal_analysis(
            scaled_ipr, tubing, test.wellhead_psia,
            test.gas_oil_ratio_scf_stb or default_gor, test.water_cut,
            oil, gas, gravity_multiplier=settings.get(GRAVITY, 1.0),
            friction_multiplier=settings.get(FRICTION, 1.0), **kwargs)
        flows = bool(solved.converged and solved.liquid_rate_stb_d > 0.0)
        predicted = solved.liquid_rate_stb_d if flows else 0.0
        rows.append({
            "predicted": float(predicted),
            "flows": flows,
            # A well the model says will not flow at all is not a percentage
            # error; -100% is the honest reading and keeps it in the rms.
            "error": float((predicted - test.rate_stb_d) / test.rate_stb_d),
        })
    return rows


def _oil_rate(test) -> float:
    """Oil rate from a test reported as total liquid."""
    return test.rate_stb_d * (1.0 - test.water_cut)


def _stream(test, gor, oil, gas, default_gor):
    from .wells import ProductionStream

    oil_rate = _oil_rate(test)
    water = test.rate_stb_d - oil_rate
    return ProductionStream(
        oil_rate_stb_d=oil_rate, water_rate_stb_d=water,
        gas_rate_scf_d=oil_rate * (gor or default_gor), oil=oil, gas=gas)


def _jacobian(residuals, values, lower, upper, relative_step=1e-3):
    """Forward differences, stepped inward at a bound.

    A forward step at the upper bound would evaluate outside the feasible
    set, where the multipliers mean nothing and the solve may not converge.
    """
    base = residuals(values)
    columns = []
    for index in range(values.size):
        step = relative_step * max(abs(values[index]), 1.0)
        trial = values.copy()
        if trial[index] + step > upper[index]:
            step = -step
        trial[index] += step
        columns.append((residuals(trial) - base) / step)
    return np.column_stack(columns) if columns else np.zeros((base.size, 0))


def _correlations(jacobian, parameters) -> dict:
    """Parameter correlations from the approximate covariance.

    (J^T J)^-1 up to the residual variance, which is all that is needed: the
    correlation matrix is scale-free, so the unknown variance divides out. A
    pair near +/-1 means the data cannot tell those two apart, whatever the
    fitted values say.
    """
    if jacobian.shape[1] < 2:
        return {}
    normal = jacobian.T @ jacobian
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        # Exactly singular: the parameters are perfectly confounded, which is
        # the strongest possible version of the finding rather than an error.
        return {"%s/%s" % (parameters[i], parameters[j]): 1.0
                for i in range(len(parameters))
                for j in range(i + 1, len(parameters))}

    deviation = np.sqrt(np.clip(np.diag(covariance), 1e-30, None))
    out = {}
    for i in range(len(parameters)):
        for j in range(i + 1, len(parameters)):
            value = covariance[i, j] / (deviation[i] * deviation[j])
            out["%s/%s" % (parameters[i], parameters[j])] = float(
                np.clip(value, -1.0, 1.0))
    return out


def _levenberg_marquardt(residuals, start, lower, upper, max_iterations,
                         tolerance=1e-8):
    """Bounded Levenberg-Marquardt, numpy only.

    Projected rather than transformed: a trial step is clipped back into the
    box. Crude next to an interior-point method and entirely adequate at
    three smooth parameters, where the bounds exist to stop the optimiser
    running somewhere unphysical rather than to be sat on.
    """
    values = np.clip(start.astype(float), lower, upper)
    current = residuals(values)
    cost = float(current @ current)
    damping = 1e-3
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        jacobian = _jacobian(residuals, values, lower, upper)
        gradient = jacobian.T @ current
        normal = jacobian.T @ jacobian

        stepped = False
        for _attempt in range(12):
            try:
                step = np.linalg.solve(
                    normal + damping * np.diag(np.diag(normal) + 1e-12),
                    -gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue

            trial = np.clip(values + step, lower, upper)
            trial_residuals = residuals(trial)
            trial_cost = float(trial_residuals @ trial_residuals)
            if trial_cost < cost:
                improvement = cost - trial_cost
                values, current, cost = trial, trial_residuals, trial_cost
                damping = max(damping / 3.0, 1e-9)
                stepped = True
                if improvement < tolerance * max(cost, 1e-12):
                    return values, iterations
                break
            damping *= 10.0

        if not stepped:
            break

    return values, iterations


def _report_parameters(settings, base_pi) -> dict:
    out = {}
    if PRODUCTIVITY in settings:
        out["productivity_scale"] = float(settings[PRODUCTIVITY])
        if base_pi is not None:
            attribute, original = base_pi
            out[attribute] = float(original * settings[PRODUCTIVITY])
    for name in (GRAVITY, FRICTION):
        if name in settings:
            out[name] = float(settings[name])
    return out


def _annotate(result: CalibrationResult, parameters: list) -> None:
    report = result.identifiability
    if report is not None and not report.supported:
        result.notes.append(
            "These tests cannot support fitting %d parameters. The numbers "
            "below are what the optimiser reached, not what the data "
            "determined." % len(parameters))
    for reason in (report.reasons if report else []):
        result.notes.append(reason)

    if not result.improved:
        result.notes.append(
            "Calibration did not improve on the starting model (%.1f%% error "
            "across the tests), so the original parameters are returned "
            "unchanged. Either the model was already as close as these three "
            "parameters can bring it, or the error is somewhere they do not "
            "reach -- a wrong water cut or gas-oil ratio does not come out in "
            "the wash here." % (result.rms_before * 100.0))
    else:
        result.notes.append(
            "Error across the tests fell from %.1f%% to %.1f%%."
            % (result.rms_before * 100.0, result.rms_after * 100.0))

    for name in result.at_bound:
        low, high = BOUNDS[name]
        value = result.parameters.get(name, result.parameters.get(
            "productivity_scale", 0.0))
        result.notes.append(
            "%s hit its %s bound of %.2f. A correlation that needs scaling "
            "past this is the wrong correlation for this well, or an input is "
            "wrong -- physics model mismatch, check the data rather than "
            "trusting the fit."
            % (name, "lower" if value <= low * (1.0 + 1e-6) else "upper",
               low if value <= low * (1.0 + 1e-6) else high))

    for pair in result.confounded:
        result.notes.append(
            "%s are %.2f correlated: the tests cannot tell them apart, so "
            "their individual values mean little even though their combined "
            "effect is fitted. The model will be wrong at rates it was not "
            "fitted at." % (pair.replace("/", " and "),
                            result.correlations[pair]))
