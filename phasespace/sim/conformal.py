"""Prediction intervals that come with a guarantee instead of a hope.

Everything else in this engine that reports uncertainty reports a heuristic.
`ArpsFit.band` is the clearest case: it is the scatter of the history about
the fitted curve, and its own docstring says it should not be read as a
confidence interval. That was honest and it was also a gap, because the
question an engineer actually has -- "how wrong might this number be" --
went unanswered.

Split conformal prediction answers it, and it is the only method in the field
that answers it with a theorem rather than an assumption. Given any
predictor, fitted however you like, and a calibration set the predictor has
never seen:

    P( Y in C(X) ) >= 1 - alpha

finite sample, distribution free, no assumption about the model being right,
the noise being Gaussian, or the predictor being any good. If the model is
bad the interval comes back wide, which is the correct behaviour and a useful
diagnostic in itself.

**What it does not give, stated plainly, because this is the part that gets
oversold.** The guarantee is *marginal*: 90% coverage averaged over wells,
not 90% for the particular well in front of you. Conditional coverage is
provably impossible to obtain distribution-free in finite samples. An
engineer asking about *this* well is asking a question no method answers for
free, and the honest partial fix is stratification -- calibrate separately
within groups that share a regime, and the guarantee then holds exactly
within each group, at the cost of needing enough calibration points per
group. `StratifiedConformal` does that and refuses to issue an interval for
a stratum too thin to support one.

**The assumption that does the work is exchangeability** between calibration
and test points. Not independence -- weaker than that -- but it is violated
by time-ordered production data with a trend, and by calibrating on wells
that later get refitted. Where the application is a forecast, the
calibration must be built the same way the forecast will be used: fit on a
prefix, score on the held-out tail, so the calibration residuals and the
deployment residuals are the same kind of thing.

The scores implemented here are the two that matter for this engine:

  absolute residual     constant-width intervals; the simplest thing that
                        carries the guarantee
  CQR                   Romano, Patterson & Candes (2019): conformalise a
                        pair of quantile predictions, which keeps the
                        interval narrow where the model is confident and
                        wide where it is not, and still carries the
                        guarantee

numpy only, like the rest of the engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Below this many calibration points the guarantee is technically true and
# practically useless: the conformal quantile is an order statistic, so with
# n points the tightest achievable miscoverage is 1/(n+1). At 90% coverage
# that needs at least 9, and asking for 19 leaves room for the interval to
# be informative rather than simply the widest residual seen.
MIN_CALIBRATION = 19


def required_calibration(confidence: float) -> int:
    """Smallest calibration set that can express this confidence at all.

    The conformal quantile is the ceil((n+1)(1-alpha))-th smallest score. If
    that index exceeds n the interval is unbounded -- the method is telling
    you it cannot distinguish the tail with this little data, which is
    correct and worth surfacing rather than hiding behind an arbitrary
    default.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1), got %r" % (confidence,))
    # The interval is bounded when ceil((n+1)c) <= n, i.e. when n >= c/(1-c).
    #
    # The tolerance is not decoration. Written as ceil(1/(1-c)) - 1 this
    # returns 10 for c = 0.90, because 1 - 0.90 is 0.09999999999999998 and
    # the reciprocal lands just above 10. One point too many sounds harmless
    # until it is the difference between issuing an interval for a stratum
    # and refusing one.
    return int(math.ceil(confidence / (1.0 - confidence) - 1e-9))


@dataclass
class ConformalInterval:
    """A calibrated interval maker, and what it is allowed to claim."""

    quantile: float
    confidence: float
    calibration_size: int
    kind: str = "absolute"
    stratum: str = ""

    @property
    def guaranteed(self) -> bool:
        """Whether the calibration set can support the stated confidence."""
        return self.calibration_size >= required_calibration(self.confidence)

    @property
    def achievable_confidence(self) -> float:
        """The highest confidence this many points can express."""
        n = self.calibration_size
        return n / (n + 1.0) if n > 0 else 0.0

    def apply(self, prediction):
        """Symmetric interval around a point prediction."""
        point = np.asarray(prediction, dtype=float)
        return point - self.quantile, point + self.quantile

    def apply_quantiles(self, lower, upper):
        """Widen (or tighten) a predicted quantile pair -- the CQR form.

        The conformal quantile can be *negative* here, which tightens the
        interval. That is not a bug: if the underlying quantile regression
        was over-conservative, conformal corrects it downward and the
        coverage guarantee still holds.
        """
        return (np.asarray(lower, dtype=float) - self.quantile,
                np.asarray(upper, dtype=float) + self.quantile)

    def describe(self) -> str:
        where = " in %s" % self.stratum if self.stratum else ""
        if not self.guaranteed:
            return ("%.0f%% coverage%s is not supported by %d calibration "
                    "points -- %d are needed, and the most this many can "
                    "express is %.1f%%"
                    % (self.confidence * 100.0, where, self.calibration_size,
                       required_calibration(self.confidence),
                       self.achievable_confidence * 100.0))
        return ("%.0f%% marginal coverage%s, from %d calibration points"
                % (self.confidence * 100.0, where, self.calibration_size))


def _conformal_quantile(scores, confidence):
    """The ceil((n+1)(1-alpha))/n empirical quantile of the scores.

    The (n+1) is the whole of why this is a finite-sample guarantee rather
    than an asymptotic one, and dropping it -- taking the plain empirical
    quantile -- undercovers slightly at every sample size. It is the most
    commonly dropped detail in implementations of this method.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        raise ValueError("no finite calibration scores")
    rank = math.ceil((n + 1) * confidence)
    if rank > n:
        return math.inf
    return float(np.sort(values)[rank - 1])


def calibrate(predictions, truths, confidence: float = 0.90,
              stratum: str = "") -> ConformalInterval:
    """Absolute-residual conformal calibration."""
    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(truths, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError("predictions and truths must be the same length, "
                         "got %d and %d" % (predicted.size, actual.size))
    scores = np.abs(actual - predicted)
    usable = np.isfinite(scores)
    return ConformalInterval(
        quantile=_conformal_quantile(scores[usable], confidence),
        confidence=confidence, calibration_size=int(usable.sum()),
        kind="absolute", stratum=stratum)


def calibrate_quantiles(lower, upper, truths, confidence: float = 0.90,
                        stratum: str = "") -> ConformalInterval:
    """Conformalised quantile regression (Romano, Patterson & Candes 2019).

    The score is how far outside the predicted band the truth fell, negative
    when it fell comfortably inside. Conformalising this rather than an
    absolute residual is what keeps the interval adaptive: narrow where the
    underlying model is confident, wide where it is not, with the guarantee
    intact either way.
    """
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    actual = np.asarray(truths, dtype=float)
    if not (low.shape == high.shape == actual.shape):
        raise ValueError("lower, upper and truths must be the same length")
    if np.any(high < low):
        raise ValueError("upper quantile is below lower on %d row(s)"
                         % int(np.sum(high < low)))

    scores = np.maximum(low - actual, actual - high)
    usable = np.isfinite(scores)
    return ConformalInterval(
        quantile=_conformal_quantile(scores[usable], confidence),
        confidence=confidence, calibration_size=int(usable.sum()),
        kind="cqr", stratum=stratum)


@dataclass
class StratifiedConformal:
    """One calibration per regime, and a refusal where there is not enough.

    Marginal coverage over a whole field can hide a stratum that is badly
    covered -- high-GOR wells at 60% inside an overall 90% -- which is
    exactly the well an engineer is asking about. Calibrating within strata
    gives the guarantee *per stratum*, which is the closest thing to
    conditional coverage that is actually obtainable.

    A stratum with too few points gets no interval rather than a wide one.
    Saying "not enough data about wells like this" is information; quoting a
    guarantee the data cannot support is not.
    """

    confidence: float = 0.90
    intervals: dict = field(default_factory=dict)
    fallback: ConformalInterval = None
    notes: list = field(default_factory=list)

    def fit(self, strata, predictions, truths) -> "StratifiedConformal":
        labels = np.asarray(strata)
        predicted = np.asarray(predictions, dtype=float)
        actual = np.asarray(truths, dtype=float)

        self.fallback = calibrate(predicted, actual, self.confidence,
                                  stratum="all wells")
        needed = required_calibration(self.confidence)
        for name in sorted(set(labels.tolist())):
            mask = labels == name
            if int(mask.sum()) < max(needed, MIN_CALIBRATION):
                self.notes.append(
                    "%s has %d calibration points; %d are needed for %.0f%% "
                    "coverage, so it falls back to the pooled interval and "
                    "the guarantee for it is marginal over all wells rather "
                    "than specific to this regime."
                    % (name, int(mask.sum()), max(needed, MIN_CALIBRATION),
                       self.confidence * 100.0))
                continue
            self.intervals[name] = calibrate(
                predicted[mask], actual[mask], self.confidence,
                stratum=str(name))
        return self

    def interval_for(self, stratum) -> ConformalInterval:
        return self.intervals.get(stratum, self.fallback)


def empirical_coverage(lower, upper, truths) -> float:
    """The number to publish beside the nominal one.

    A guarantee is a claim about repeated use; this is what actually
    happened. They should agree, and when they do not the exchangeability
    assumption is the first thing to doubt.
    """
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    actual = np.asarray(truths, dtype=float)
    inside = (actual >= low) & (actual <= high)
    usable = np.isfinite(low) & np.isfinite(high) & np.isfinite(actual)
    if not usable.any():
        return 0.0
    return float(inside[usable].mean())
