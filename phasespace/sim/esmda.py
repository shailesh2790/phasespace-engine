"""ES-MDA: calibration that returns a distribution instead of an answer.

Emerick & Reynolds, "Ensemble smoother with multiple data assimilation",
Computers & Geosciences 55 (2013) 3-15.

`wellcalibration.calibrate_well` fits by bounded Levenberg-Marquardt and
returns one parameter vector plus a correlation matrix. That is the right
tool for "what are this well's parameters", and it is the wrong tool for
the question underneath it, which is "how wrong could they be". The
correlation matrix says which parameters trade off; it does not say how
far they can trade before the fit stops matching, and on a well with four
tests that distance is most of the answer.

`identifiability` already refuses the worst cases -- too few tests, no rate
spread -- but a refusal is binary and the truth is not. A well can be
*weakly* determined: fittable, not confounded enough to refuse, and still
carrying a productivity index uncertain by a factor of two. Nothing in the
engine could say so until now.

ES-MDA answers it by carrying an ensemble. Start from a prior sample of
parameter vectors, run the forward model on each, and pull the whole
ensemble towards the measurements in several tempered steps. What comes
back is a posterior sample: the spread *is* the uncertainty, it costs no
extra assumption, and it propagates -- feed the ensemble through a forecast
and the forecast has a band for the same reason.

**Why multiple assimilations rather than one.** A single ensemble-smoother
update is a linear-Gaussian step, and the well model is neither. Emerick
& Reynolds' result is that splitting the update into `N_a` steps, each
with the observation covariance inflated by `alpha_i`, where the
inflations satisfy `sum(1/alpha_i) = 1`, tracks a nonlinear posterior far
better at `N_a` times the cost. Equal inflations, `alpha_i = N_a`, is the
standard choice and the one used here.

**Three implementation details that decide whether this works.**

*Bounds are handled by transformation, not clipping.* The productivity
index is positive and the correlation multipliers live in [0.8, 1.2].
ES-MDA's update is an unconstrained Gaussian shift, so the obvious fix is
to clip afterwards -- and clipping piles probability mass exactly on the
bound, producing a posterior with spikes at the limits that reads as
confidence. Parameters are mapped to an unbounded space (log for positive,
logit for an interval), assimilated there, and mapped back, so the bound
is respected by construction and never reached.

*The observation covariance is an input and dominates the answer.*
Understate the measurement error and the posterior is overconfident --
narrow, wrong, and indistinguishable from a well-determined fit. There is
no way to infer it from the data being fitted, so it is required rather
than defaulted.

*Ensemble collapse is detected, not hidden.* With few parameters and
several assimilations the spread can shrink to nothing, at which point the
posterior is an artefact of the ensemble size rather than a statement
about the well. `EnsembleResult.collapsed` reports it.

numpy only, like the rest of the engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Emerick & Reynolds use 4 to 10 assimilations in their examples and note
# the result is not sensitive in that range. Four is the cheapest that
# behaves nonlinearly, and each one costs a full ensemble of forward runs.
DEFAULT_ASSIMILATIONS = 4

# Below this the sample covariance of the predicted data is too poor to
# invert meaningfully, whatever the parameter count.
MIN_ENSEMBLE = 20

# Fraction of its prior standard deviation a parameter must retain for the
# posterior spread to mean anything. Below it the ensemble has collapsed
# and the narrowness is numerical, not informational.
COLLAPSE_FRACTION = 0.02

# Singular values below this fraction of the largest are dropped when
# inverting the data covariance. The matrix is a sample covariance from a
# modest ensemble and is routinely near-singular; a plain inverse amplifies
# the smallest, worst-estimated directions the most.
SVD_TRUNCATION = 1e-8


def _to_unbounded(value, lower, upper):
    """Map a bounded parameter onto the whole line.

    log for a one-sided bound, logit for an interval. The update is a
    Gaussian shift and Gaussians are unbounded, so this is where the
    constraint is enforced -- afterwards is too late.
    """
    value = np.asarray(value, dtype=float)
    if upper is None or not np.isfinite(upper):
        return np.log(value - lower)
    span = upper - lower
    fraction = np.clip((value - lower) / span, 1e-12, 1.0 - 1e-12)
    return np.log(fraction / (1.0 - fraction))


def _from_unbounded(value, lower, upper):
    value = np.asarray(value, dtype=float)
    if upper is None or not np.isfinite(upper):
        return lower + np.exp(value)
    span = upper - lower
    return lower + span / (1.0 + np.exp(-value))


@dataclass
class Parameter:
    """One uncertain quantity, with the prior belief about it."""

    name: str
    prior_mean: float
    prior_sd: float            # in the parameter's own units
    lower: float = 0.0
    upper: float = float("inf")

    def __post_init__(self):
        if self.prior_sd <= 0.0:
            raise ValueError("%s needs a positive prior spread; a parameter "
                             "known exactly does not need calibrating"
                             % self.name)
        if not self.lower < self.prior_mean < self.upper:
            raise ValueError(
                "%s has a prior mean of %g outside its bounds (%g, %g)"
                % (self.name, self.prior_mean, self.lower, self.upper))

    def sample(self, size: int, rng) -> np.ndarray:
        """Prior draw, in unbounded space.

        The spread is translated through the transform at the mean rather
        than sampled and clipped, so the prior is a genuine distribution on
        the bounded interval instead of a truncated one.
        """
        centre = _to_unbounded(self.prior_mean, self.lower, self.upper)
        edge = _to_unbounded(
            min(self.prior_mean + self.prior_sd,
                self.prior_mean + 0.98 * (self.upper - self.prior_mean))
            if np.isfinite(self.upper) else self.prior_mean + self.prior_sd,
            self.lower, self.upper)
        return rng.normal(centre, abs(edge - centre), size)


@dataclass
class EnsembleResult:
    """The posterior sample, and whether it is worth believing."""

    names: list
    prior: np.ndarray                 # (members, parameters), natural units
    posterior: np.ndarray             # (members, parameters), natural units
    predicted: np.ndarray             # (members, observations), final
    observations: np.ndarray
    observation_sd: np.ndarray
    assimilations: int
    notes: list = field(default_factory=list)

    @property
    def mean(self) -> dict:
        return dict(zip(self.names, self.posterior.mean(axis=0)))

    @property
    def sd(self) -> dict:
        return dict(zip(self.names, self.posterior.std(axis=0, ddof=1)))

    def interval(self, confidence: float = 0.90) -> dict:
        """Percentile interval per parameter, straight off the sample.

        Percentiles rather than mean +/- k sd, because the posterior in
        natural units is not symmetric -- the transform that keeps a
        parameter inside its bounds also skews it near them.
        """
        tail = (1.0 - confidence) / 2.0
        low = np.percentile(self.posterior, tail * 100.0, axis=0)
        high = np.percentile(self.posterior, (1.0 - tail) * 100.0, axis=0)
        return {name: (float(a), float(b))
                for name, a, b in zip(self.names, low, high)}

    @property
    def reduction(self) -> dict:
        """How much each parameter learned, as 1 - posterior/prior spread.

        The number that says whether the data had anything to say about
        this parameter. Near zero means the tests did not constrain it and
        the posterior is just the prior back again -- which is honest, and
        is information an engineer needs before quoting the value.
        """
        prior_sd = self.prior.std(axis=0, ddof=1)
        post_sd = self.posterior.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            learned = np.where(prior_sd > 0.0, 1.0 - post_sd / prior_sd, 0.0)
        return {name: float(value) for name, value in zip(self.names, learned)}

    @property
    def collapsed(self) -> list:
        """Parameters whose spread has vanished rather than narrowed."""
        prior_sd = self.prior.std(axis=0, ddof=1)
        post_sd = self.posterior.std(axis=0, ddof=1)
        return [name for name, before, after
                in zip(self.names, prior_sd, post_sd)
                if before > 0.0 and after / before < COLLAPSE_FRACTION]

    @property
    def data_misfit(self) -> float:
        """RMS of the normalised residual of the posterior mean prediction.

        About 1 means the model matches the data to within the measurement
        error, which is the target. Much below 1 means the observation
        error was overstated, or the model is fitting noise.
        """
        residual = (self.predicted.mean(axis=0) - self.observations)
        return float(np.sqrt(np.mean((residual / self.observation_sd) ** 2)))

    def describe(self) -> str:
        parts = []
        intervals = self.interval()
        for name in self.names:
            low, high = intervals[name]
            parts.append("%s %.4g to %.4g (learned %.0f%%)"
                         % (name, low, high, self.reduction[name] * 100.0))
        return "; ".join(parts)


def esmda(parameters: list, forward, observations, observation_sd,
          members: int = 100, assimilations: int = DEFAULT_ASSIMILATIONS,
          seed: int = 0, inflations=None) -> EnsembleResult:
    """Ensemble smoother with multiple data assimilation.

    `forward` takes a dict of parameter values and returns the predicted
    observations. It is called `members * assimilations` times, so its cost
    sets the cost of the whole calibration.

    `observation_sd` is the measurement error on each observation, in the
    observation's own units. It is required: there is no way to infer it
    from the data being fitted, and understating it produces a confident
    posterior rather than a visible failure.
    """
    observations = np.asarray(observations, dtype=float).ravel()
    observation_sd = np.asarray(observation_sd, dtype=float).ravel()
    if observation_sd.size == 1:
        observation_sd = np.full(observations.shape,
                                 float(observation_sd.reshape(())))
    if observation_sd.shape != observations.shape:
        raise ValueError(
            "observation_sd must be one value or one per observation; got "
            "%d for %d observations"
            % (observation_sd.size, observations.size))
    if np.any(observation_sd <= 0.0):
        raise ValueError("every observation needs a positive measurement "
                         "error; a value known exactly would force the fit")
    if members < MIN_ENSEMBLE:
        raise ValueError(
            "an ensemble of %d cannot support a usable sample covariance; "
            "%d is the minimum here" % (members, MIN_ENSEMBLE))
    if assimilations < 1:
        raise ValueError("at least one assimilation, got %r" % (assimilations,))

    if inflations is None:
        inflations = [float(assimilations)] * assimilations
    inflations = [float(a) for a in inflations]
    # The condition that makes the tempered updates add up to one full
    # assimilation. Getting it wrong does not fail loudly: it quietly
    # over- or under-weights the data and the posterior spread is wrong.
    total = sum(1.0 / a for a in inflations)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "inflation coefficients must satisfy sum(1/alpha) = 1 "
            "(Emerick & Reynolds 2013); got %.6f" % total)

    rng = np.random.default_rng(seed)
    names = [p.name for p in parameters]
    state = np.column_stack([p.sample(members, rng) for p in parameters])
    prior_natural = _natural(state, parameters)

    notes = []
    predicted = None
    for step, alpha in enumerate(inflations):
        natural = _natural(state, parameters)
        predicted = np.array([
            np.asarray(forward(dict(zip(names, row))), dtype=float).ravel()
            for row in natural])
        if predicted.shape[1] != observations.size:
            raise ValueError(
                "forward returned %d values but %d observations were given"
                % (predicted.shape[1], observations.size))

        usable = np.isfinite(predicted).all(axis=1)
        if usable.sum() < MIN_ENSEMBLE:
            raise ValueError(
                "only %d of %d ensemble members produced a finite forward "
                "run at assimilation %d. The forward model is failing over "
                "most of the prior, which usually means the prior is too "
                "wide rather than that the data is bad."
                % (int(usable.sum()), members, step + 1))
        if not usable.all():
            notes.append(
                "%d member(s) failed to run at assimilation %d and were "
                "held at their previous values"
                % (int((~usable).sum()), step + 1))

        state = _update(state, predicted, observations, observation_sd,
                        alpha, usable, rng)

    natural = _natural(state, parameters)
    predicted = np.array([
        np.asarray(forward(dict(zip(names, row))), dtype=float).ravel()
        for row in natural])

    result = EnsembleResult(
        names=names, prior=prior_natural, posterior=natural,
        predicted=predicted, observations=observations,
        observation_sd=observation_sd, assimilations=len(inflations),
        notes=notes)

    for name in result.collapsed:
        result.notes.append(
            "%s has collapsed to %.1f%% of its prior spread. That is not a "
            "well-determined parameter, it is an ensemble too small to hold "
            "a distribution -- raise the member count before quoting it."
            % (name, COLLAPSE_FRACTION * 100.0))
    for name, learned in result.reduction.items():
        if learned < 0.05:
            result.notes.append(
                "%s came back as its prior: these observations say almost "
                "nothing about it, so the value is an assumption rather "
                "than a measurement." % name)
    misfit = result.data_misfit
    if misfit > 3.0:
        result.notes.append(
            "the posterior still misses the data by %.1f times the stated "
            "measurement error. Either the model is wrong for this well or "
            "the measurement error is understated." % misfit)
    elif misfit < 0.3:
        result.notes.append(
            "the posterior matches the data to %.2f of the stated "
            "measurement error, which is closer than the measurements "
            "themselves. Suspect the error estimate." % misfit)
    return result


def _natural(state, parameters) -> np.ndarray:
    return np.column_stack([
        _from_unbounded(state[:, index], p.lower, p.upper)
        for index, p in enumerate(parameters)])


def _update(state, predicted, observations, observation_sd, alpha, usable, rng):
    """One tempered ensemble-smoother step, in unbounded parameter space."""
    live_state = state[usable]
    live_predicted = predicted[usable]
    members = live_state.shape[0]

    # Perturbed observations, inflated by alpha. Perturbing them is what
    # keeps the posterior spread right: assimilating the same unperturbed
    # vector into every member collapses the ensemble towards it.
    noise = rng.normal(0.0, 1.0, (members, observations.size))
    perturbed = observations + math.sqrt(alpha) * observation_sd * noise

    state_anomaly = live_state - live_state.mean(axis=0)
    data_anomaly = live_predicted - live_predicted.mean(axis=0)
    scale = 1.0 / (members - 1.0)
    cross = scale * state_anomaly.T @ data_anomaly          # C_MD
    auto = scale * data_anomaly.T @ data_anomaly            # C_DD
    inflated = auto + alpha * np.diag(observation_sd ** 2)

    gain = cross @ _pseudo_inverse(inflated)
    updated = live_state + (perturbed - live_predicted) @ gain.T

    out = state.copy()
    out[usable] = updated
    return out


def _pseudo_inverse(matrix) -> np.ndarray:
    """Truncated SVD inverse.

    The data covariance is a sample covariance from a modest ensemble and
    is routinely near-singular. A plain inverse puts the most weight on its
    smallest singular values, which are exactly the directions the ensemble
    estimated worst, and the update then chases sampling noise.
    """
    u, singular, vt = np.linalg.svd(matrix, hermitian=True)
    cutoff = SVD_TRUNCATION * singular.max() if singular.size else 0.0
    keep = singular > cutoff
    inverted = np.zeros_like(singular)
    inverted[keep] = 1.0 / singular[keep]
    return (vt.T * inverted) @ u.T


# ---------------------------------------------------- wired to a well model --


def calibrate_well_ensemble(ipr, tubing, tests, oil=None, gas=None,
                            correlation=None, parameters=None,
                            default_gor_scf_stb: float = 600.0,
                            rate_error_fraction: float = 0.05,
                            members: int = 60,
                            assimilations: int = DEFAULT_ASSIMILATIONS,
                            seed: int = 0) -> EnsembleResult:
    """ES-MDA on the same three parameters `calibrate_well` fits.

    The forward model is the ordinary nodal solve, deliberately the same one
    the rest of the engine runs, so the posterior describes the model that
    will actually be used rather than a reduced stand-in.

    That honesty costs: every member is a full nodal solve per test, so the
    work is `members * assimilations * tests` bisections. Sixty members over
    four assimilations on five tests is 1200 solves -- a couple of minutes,
    not the couple of seconds `calibrate_well` takes. This is the overnight
    answer to "how well do we know this well", not the interactive one, and
    the two are complementary rather than competing: Levenberg-Marquardt for
    the value, ES-MDA for the range around it.

    `rate_error_fraction` is the measurement error on a well test, relative.
    5% is a reasonable test-separator figure and it is the single most
    consequential input here -- state it too tight and the posterior comes
    back narrow and wrong.
    """
    from .wellcalibration import (
        BOUNDS,
        FRICTION,
        GRAVITY,
        PRODUCTIVITY,
        _inflow_scale,
        _rate_errors,
    )
    from .fluids import GasProperties, LiquidProperties

    oil = oil or LiquidProperties()
    gas = gas or GasProperties()
    names = list(parameters or [PRODUCTIVITY, GRAVITY, FRICTION])
    if not tests:
        raise ValueError("no tests supplied")
    if rate_error_fraction <= 0.0:
        raise ValueError("a well test has a measurement error; "
                         "rate_error_fraction must be positive")

    base_pi = _inflow_scale(ipr)

    # Priors centred on "the model as built", which is what an engineer has
    # before calibrating. The spread on the multipliers spans most of their
    # allowed band, because a correlation being 10% out on a given fluid is
    # ordinary; the spread on the productivity index is wide because it is
    # usually inherited from an offset well.
    priors = {
        PRODUCTIVITY: (1.0, 0.6),
        GRAVITY: (1.0, 0.07),
        FRICTION: (1.0, 0.07),
    }
    spec = [Parameter(name, priors[name][0], priors[name][1],
                      lower=BOUNDS[name][0], upper=BOUNDS[name][1])
            for name in names]

    measured = np.array([t.rate_stb_d for t in tests], dtype=float)

    def forward(values):
        settings = {name: values.get(name, 1.0)
                    for name in (PRODUCTIVITY, GRAVITY, FRICTION)}
        rows = _rate_errors(ipr, tubing, tests, oil, gas, correlation,
                            settings, base_pi, default_gor_scf_stb)
        # A member whose well will not flow is not a zero-rate prediction,
        # it is a failed run: feeding 0 in would drag the ensemble mean
        # towards parameters that kill the well.
        return [row["predicted"] if row["flows"] else np.nan for row in rows]

    return esmda(spec, forward, measured, rate_error_fraction * measured,
                 members=members, assimilations=assimilations, seed=seed)
