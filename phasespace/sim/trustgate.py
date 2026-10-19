"""The check every learned answer passes before anyone sees it.

PRD v2, Workstream D. The architecture only works if "the engine is the
referee" is a mechanism rather than a slogan: a surrogate is allowed to
answer because something cheap and automatic has established that this
particular query is one it can answer, and the engine answers instead when
that fails.

Six checks, in the order they get cheaper to fail:

  range        is this query inside what the model was trained on
  sanity       is the answer physically possible at all
  conservation does it close mass balance
  spread       do the ensemble members agree
  coverage     are the model's stated intervals still covering
  referee      does the engine agree on a sampled fraction of queries

A failure is never silently absorbed. `Verdict.answer_with_engine` says
the surrogate must not be used, `Verdict.reasons` says why in the terms an
engineer checks, and the case is marked for the active-learning queue.

**Three places the obvious implementation is wrong, and this is most of
why the file is longer than the list above.**

*Out-of-distribution needs two tests, not one.* A joint distance alone
misses a query that is ordinary in every feature but in a combination
never seen; a per-feature range alone misses a query inside every margin
whose combination is absurd. Both are computed and either can fail. The
joint test is Mahalanobis against the training covariance, scored against
the training set's *own* distribution of distances rather than a
chi-squared table, because the inputs are not Gaussian and the table would
be answering a different question.

*Ensemble spread is not a calibrated uncertainty.* Deep ensembles are
known to be overconfident, and five members give a spread estimated from
four degrees of freedom. It is used here as a relative flag -- these
members disagree more than they usually do -- and never as the interval
quoted to a user. The interval comes from conformal prediction, which
carries a guarantee the spread does not.

*A coverage band has sampling error, and the PRD's +/-2% ignores it.*
Realised coverage over a rolling window of n points is a binomial
proportion: at n = 50 and p = 0.9 its standard deviation is 4.2%, so a
fixed 2% band fires constantly on a model that is behaving perfectly.
`coverage_drift` computes the binomial interval and flags only when
realised coverage is outside what sampling error explains, which means the
window size decides the sensitivity rather than being ignored by it.

numpy only, like the rest of the engine, and cheap: the PRD budgets 20 ms
for the whole gate and the expensive part of each check is a matrix
multiply against a stored factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Quantile of the training set's own Mahalanobis distances that a query
# must sit inside. The PRD says 99th percentile; it is a parameter because
# a model with a large, well-covered training set can afford to be
# stricter and a thin one cannot.
DEFAULT_RANGE_QUANTILE = 0.99

# How far outside a feature's training range counts as extrapolation,
# as a fraction of that feature's training span. A little margin is
# reasonable -- the edge of a training set is not a cliff -- and a lot is
# not.
DEFAULT_FEATURE_MARGIN = 0.05

# Mass imbalance a learned answer may carry. The PRD's figure.
DEFAULT_IMBALANCE = 0.001

# Fraction of answered queries the engine re-solves. Also the PRD's.
DEFAULT_SPOT_CHECK = 0.02

# Singular values below this fraction of the largest are dropped when
# inverting the training covariance, which is routinely near-singular
# because real features are correlated.
SVD_TRUNCATION = 1e-10


@dataclass
class Check:
    """One gate check and what it found."""

    name: str
    passed: bool
    detail: str
    value: float = float("nan")
    threshold: float = float("nan")


@dataclass
class Verdict:
    """Whether the learned answer may be shown, and the evidence."""

    checks: list = field(default_factory=list)
    model: str = ""
    version: str = ""

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def answer_with_engine(self) -> bool:
        return not self.passed

    @property
    def failed(self) -> list:
        return [check for check in self.checks if not check.passed]

    @property
    def reasons(self) -> list:
        return [check.detail for check in self.failed]

    def describe(self) -> str:
        if self.passed:
            return ("%s passed all %d gate checks"
                    % (self.model or "the model", len(self.checks)))
        return ("%s did not pass the gate: %s. The engine answers instead."
                % (self.model or "the model", "; ".join(self.reasons)))


# ------------------------------------------------------------- the range --


@dataclass
class TrainingEnvelope:
    """What a model was trained on, in the form the gate needs at query time.

    Fitted once when a model is registered and stored with it, so the
    check at query time is a subtraction and a matrix multiply rather
    than a pass over the training set.
    """

    names: list
    low: np.ndarray
    high: np.ndarray
    mean: np.ndarray
    precision: np.ndarray            # inverse covariance, truncated
    distance_limit: float            # the training set's own quantile
    samples: int

    @classmethod
    def fit(cls, matrix, names: list = None,
            quantile: float = DEFAULT_RANGE_QUANTILE) -> "TrainingEnvelope":
        data = np.atleast_2d(np.asarray(matrix, dtype=float))
        if data.shape[0] < data.shape[1] + 2:
            raise ValueError(
                "an envelope over %d features needs more than %d training "
                "rows; the covariance is not estimable from %d"
                % (data.shape[1], data.shape[1] + 1, data.shape[0]))

        mean = data.mean(axis=0)
        centred = data - mean
        covariance = centred.T @ centred / (data.shape[0] - 1)
        precision = _pseudo_inverse(covariance)
        distances = np.einsum("ij,jk,ik->i", centred, precision, centred)
        return cls(
            names=list(names or ["x%d" % i for i in range(data.shape[1])]),
            low=data.min(axis=0), high=data.max(axis=0),
            mean=mean, precision=precision,
            distance_limit=float(np.quantile(distances, quantile)),
            samples=int(data.shape[0]))

    def distance(self, row) -> float:
        """Squared Mahalanobis distance from the training centre."""
        delta = np.asarray(row, dtype=float) - self.mean
        return float(delta @ self.precision @ delta)

    def outside(self, row, margin: float = DEFAULT_FEATURE_MARGIN) -> list:
        """Features this query sits outside the training range on."""
        values = np.asarray(row, dtype=float)
        span = np.maximum(self.high - self.low, 1e-12)
        below = values < self.low - margin * span
        above = values > self.high + margin * span
        return [self.names[i] for i in range(values.size) if below[i] or above[i]]


def _pseudo_inverse(matrix) -> np.ndarray:
    u, singular, vt = np.linalg.svd(matrix, hermitian=True)
    cutoff = SVD_TRUNCATION * singular.max() if singular.size else 0.0
    inverted = np.where(singular > cutoff, 1.0 / np.maximum(singular, 1e-300), 0.0)
    return (vt.T * inverted) @ u.T


def check_range(envelope: TrainingEnvelope, row,
                margin: float = DEFAULT_FEATURE_MARGIN) -> list:
    """Two tests, because either alone misses a real case.

    A query can be inside every feature's range and still be a
    combination the model never saw -- high rate at high water cut, when
    the training set only paired high rate with low. And it can be close
    to the training centre on the joint measure while one feature is far
    outside anything seen, because Mahalanobis averages over directions.
    """
    outside = envelope.outside(row, margin)
    distance = envelope.distance(row)
    return [
        Check(name="range.features",
              passed=not outside,
              value=float(len(outside)), threshold=0.0,
              detail=("query is outside the training range on %s"
                      % ", ".join(outside)) if outside
              else "every feature is inside the training range"),
        Check(name="range.joint",
              passed=distance <= envelope.distance_limit,
              value=distance, threshold=envelope.distance_limit,
              detail=("this combination of inputs is further from the "
                      "training set than %.0f%% of the training set itself "
                      "(%.1f against %.1f)"
                      % (DEFAULT_RANGE_QUANTILE * 100.0, distance,
                         envelope.distance_limit))
              if distance > envelope.distance_limit
              else "the combination of inputs is one the model has seen"),
    ]


# ------------------------------------------------------------ the physics --


def check_sanity(values: dict, bounds: dict) -> list:
    """Is the answer possible at all, before asking whether it is right."""
    out = []
    for name, (low, high) in sorted(bounds.items()):
        if name not in values:
            continue
        value = float(values[name])
        ok = (low is None or value >= low) and (high is None or value <= high)
        out.append(Check(
            name="sanity.%s" % name, passed=ok, value=value,
            threshold=float(low if low is not None else high),
            detail=("%s came back as %.4g, outside the %s to %s it can "
                    "physically be" % (name, value, low, high)) if not ok
            else "%s is physically possible" % name))
    return out


def check_conservation(inflow: float, outflow: float,
                       tolerance: float = DEFAULT_IMBALANCE) -> Check:
    """Mass balance across whatever the model answered about.

    Relative to the throughput, not absolute: 10 scf/d of imbalance is
    nothing on a 50 MMscf/d trunk and is the whole flow on a stripper.
    """
    scale = max(abs(inflow), abs(outflow), 1e-12)
    imbalance = abs(inflow - outflow) / scale
    return Check(
        name="conservation", passed=imbalance <= tolerance,
        value=imbalance, threshold=tolerance,
        detail=("mass balance is out by %.3f%%, above the %.3f%% a learned "
                "answer may carry" % (imbalance * 100.0, tolerance * 100.0))
        if imbalance > tolerance
        else "mass closes to %.4f%%" % (imbalance * 100.0))


# ----------------------------------------------------------- the ensemble --


def check_spread(predictions, tolerance: float, label: str = "value") -> Check:
    """Do the ensemble members agree with each other.

    **Not an uncertainty.** Deep ensembles are known to be overconfident,
    and five members estimate a spread from four degrees of freedom. This
    is a relative flag -- the members disagree more than usual, so treat
    the answer with suspicion -- and the interval quoted to a user comes
    from conformal prediction, which carries a guarantee this does not.
    """
    values = np.asarray(predictions, dtype=float).ravel()
    if values.size < 2:
        return Check(name="spread", passed=False, detail=(
            "an ensemble of %d cannot show disagreement; the spread check "
            "is not meaningful below two members" % values.size))
    spread = float(values.std(ddof=1))
    return Check(
        name="spread", passed=spread <= tolerance,
        value=spread, threshold=tolerance,
        detail=("the %d ensemble members disagree by %.4g on %s, above the "
                "%.4g tolerance" % (values.size, spread, label, tolerance))
        if spread > tolerance
        else "the %d members agree to %.4g" % (values.size, spread))


# ----------------------------------------------------------- the coverage --


def _binomial_halfwidth(proportion: float, n: int, z: float = 1.96) -> float:
    """Sampling error on a realised coverage, which the PRD's band ignores.

    Wilson rather than the normal approximation: at the coverages that
    matter here (0.9 and up) the normal interval runs past 1 and
    understates the lower tail, which is the tail this check is about.
    """
    if n <= 0:
        return 1.0
    centre = (proportion + z * z / (2 * n)) / (1 + z * z / n)
    spread = (z / (1 + z * z / n)) * math.sqrt(
        proportion * (1 - proportion) / n + z * z / (4 * n * n))
    return max(abs(centre + spread - proportion),
               abs(proportion - (centre - spread)))


@dataclass
class CoverageMonitor:
    """A rolling check that the model's stated intervals still hold.

    Conformal gives a guarantee under exchangeability between calibration
    and deployment. Drift is exactly the violation of that assumption, and
    it shows up as realised coverage falling away from nominal. This is
    the detector.
    """

    nominal: float = 0.90
    window: int = 200
    hits: list = field(default_factory=list)

    def record(self, truth: float, low: float, high: float) -> None:
        self.hits.append(bool(low <= truth <= high))
        if len(self.hits) > self.window:
            del self.hits[0:len(self.hits) - self.window]

    @property
    def realised(self) -> float:
        return float(np.mean(self.hits)) if self.hits else float("nan")

    @property
    def n(self) -> int:
        return len(self.hits)

    def check(self, minimum: int = 40) -> Check:
        if self.n < minimum:
            return Check(
                name="coverage", passed=True, value=self.realised,
                detail=("only %d results checked; coverage cannot be judged "
                        "below %d and is not being held against the model"
                        % (self.n, minimum)))
        realised = self.realised
        allowance = _binomial_halfwidth(self.nominal, self.n)
        short = self.nominal - realised
        return Check(
            name="coverage", passed=short <= allowance,
            value=realised, threshold=self.nominal - allowance,
            detail=("stated %.0f%% intervals are covering %.1f%% over the "
                    "last %d results. Sampling error at this window is "
                    "%.1f%%, so the shortfall is real and the model has "
                    "drifted"
                    % (self.nominal * 100.0, realised * 100.0, self.n,
                       allowance * 100.0))
            if short > allowance
            else ("stated %.0f%% intervals are covering %.1f%% over %d "
                  "results, inside the %.1f%% sampling error"
                  % (self.nominal * 100.0, realised * 100.0, self.n,
                     allowance * 100.0)))


# ------------------------------------------------------------- the referee --


@dataclass
class RefereeLog:
    """The engine's opinion of a sampled fraction of answered queries.

    Sampling is deterministic given the seed, so a run can be replayed
    from the audit log and the same queries get re-solved. A sampler the
    model could predict would be worse than none, but reproducibility
    matters more here than unpredictability: nothing in the loop is
    adversarial, and an auditor has to be able to repeat the check.
    """

    fraction: float = DEFAULT_SPOT_CHECK
    tolerance: float = 0.02
    seed: int = 0
    errors: list = field(default_factory=list)
    seen: int = 0

    def should_check(self) -> bool:
        self.seen += 1
        rng = np.random.default_rng((self.seed, self.seen))
        return bool(rng.random() < self.fraction)

    def record(self, learned: float, engine: float) -> float:
        scale = max(abs(engine), 1e-12)
        error = abs(learned - engine) / scale
        self.errors.append(error)
        return error

    def check(self, minimum: int = 10) -> Check:
        if len(self.errors) < minimum:
            return Check(
                name="referee", passed=True,
                value=float(np.median(self.errors)) if self.errors else float("nan"),
                detail=("%d spot checks so far; %d are needed before the "
                        "engine's agreement means anything"
                        % (len(self.errors), minimum)))
        median = float(np.median(self.errors))
        return Check(
            name="referee", passed=median <= self.tolerance,
            value=median, threshold=self.tolerance,
            detail=("the engine disagrees with the model by a median %.2f%% "
                    "over %d spot checks, above the %.2f%% tolerance. The "
                    "model is demoted to advisory until retrained"
                    % (median * 100.0, len(self.errors),
                       self.tolerance * 100.0))
            if median > self.tolerance
            else ("the engine agrees to a median %.2f%% over %d spot checks"
                  % (median * 100.0, len(self.errors))))


# ---------------------------------------------------------------- the gate --


@dataclass
class TrustGate:
    """Everything a model needs to be allowed to answer."""

    model: str
    version: str
    envelope: TrainingEnvelope
    bounds: dict = field(default_factory=dict)
    spread_tolerance: float = float("inf")
    coverage: CoverageMonitor = field(default_factory=CoverageMonitor)
    referee: RefereeLog = field(default_factory=RefereeLog)
    feature_margin: float = DEFAULT_FEATURE_MARGIN
    imbalance: float = DEFAULT_IMBALANCE

    def evaluate(self, features, answer: dict,
                 ensemble=None, balance: tuple = None) -> Verdict:
        """Run the gate on one query. Order is cheapest-to-fail first."""
        checks = list(check_range(self.envelope, features, self.feature_margin))
        checks.extend(check_sanity(answer, self.bounds))
        if balance is not None:
            checks.append(check_conservation(balance[0], balance[1],
                                             self.imbalance))
        if ensemble is not None and math.isfinite(self.spread_tolerance):
            checks.append(check_spread(ensemble, self.spread_tolerance))
        checks.append(self.coverage.check())
        checks.append(self.referee.check())
        return Verdict(checks=checks, model=self.model, version=self.version)
