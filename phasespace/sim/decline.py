"""What a well will produce, and how little that is really known.

Everything else in this engine answers a question about now: given these
boundary conditions, what is each pressure and each rate. An operator's
decisions are about later -- whether a well pays for an intervention, what the
field delivers next year, what reserves to book -- and that question needs a
rate against time.

Arps (1945) is how the industry answers it, and it has been doing so for eighty
years because the three-parameter family happens to fit production history
remarkably well. The parameters are empirical, not physical: nothing derives
`b` from rock. So this module treats decline as what it is, a curve fit, and
spends most of its effort on the part that is usually left out.

**The honest part is the uncertainty.** A hyperbolic fit to two years of
history has a well-determined `qi`, a reasonably determined `Di`, and a badly
determined `b` -- and `b` is the one that sets the reserves. Fits that are
indistinguishable over the measured history (within the scatter of the data
itself) routinely differ by a factor of two in EUR. Reporting a single EUR from
a least-squares fit is therefore not a forecast, it is a selection from a range
the data cannot narrow, and the range is what an engineer needs.

So `fit_arps` returns the best fit *and* the span of `b` that fits the history
about as well, with the EUR each implies. If that range is wide, the data is
telling you it does not know yet.

**Two guards against a known failure.** Arps with b >= 1 has infinite
cumulative production: the harmonic integral diverges. Unmodified, a shale well
fitted with b = 1.3 books unbounded reserves, which is why the modified-Arps
switch to a terminal exponential decline exists and why `d_min_per_year`
defaults to a real value here rather than to zero. And `b` is capped during
fitting, because a short history of a well still in transient flow will happily
fit b = 4 and forecast a well that never stops.

Duong (2011) is included for tight and shale wells, where linear flow in a
fracture network gives a rate decline Arps fits badly over the first years.

Units: rates per day in whatever volume unit the caller uses (STB/day,
scf/day); time in years; declines nominal, per year.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DAYS_PER_YEAR = 365.25

# Modified Arps: the decline rate at which a hyperbolic curve is switched to an
# exponential tail. Without a switch, b >= 1 integrates to infinite reserves.
# 5%/year is a common convention for the terminal decline of a mature well; it
# is a convention, not a measurement, so it is a visible default rather than a
# hidden constant.
DEFAULT_TERMINAL_DECLINE = 0.05

# The widest b worth fitting. Arps' own survey found b between 0 and 1 for
# almost all conventional wells; unconventional wells fit higher, and values
# above about 2 are a sign of a well still in transient flow rather than of a
# real reserve. Fitting past this produces forecasts that cannot be defended.
MAX_FITTED_B = 2.0

# How much worse than the best fit still counts as "fits the history about as
# well". 5% on the rms of log-rate residuals -- comfortably inside the
# month-to-month scatter of real production data, which is dominated by
# downtime rather than by reservoir behaviour.
FIT_TOLERANCE_FRACTION = 0.05

MIN_POINTS = 6

# Log-rate rms above which a Duong fit is reporting the wrong model rather
# than a noisy right one. Monthly shale production scatters around 10%; a
# structural mismatch runs 20% and up.
DUONG_MISFIT_WARNING = 0.15


@dataclass
class ArpsDecline:
    """Rate against time, with a terminal exponential tail.

        q(t) = qi / (1 + b Di t)^(1/b)        b > 0
        q(t) = qi exp(-Di t)                  b = 0

    `d_min_per_year` switches to exponential decline once the instantaneous
    decline rate has fallen that far. For b >= 1 this is not a refinement, it
    is the difference between a finite and an infinite answer.
    """

    qi: float
    di_per_year: float
    b: float = 0.0
    d_min_per_year: float = DEFAULT_TERMINAL_DECLINE

    def __post_init__(self):
        # Cast on the way in so the declared types are the real ones. These
        # fields come out of a least-squares solve and so arrive as numpy
        # scalars, and a fitted decline is a public result that gets
        # serialised, compared and printed.
        #
        # Not a bug fix: np.float64 is a subclass of float and json handles
        # it. np.float32 and np.int64 are not, so this is the kind of hygiene
        # that keeps a future dtype change from becoming a serialisation
        # failure in an endpoint rather than a test.
        self.qi = float(self.qi)
        self.di_per_year = float(self.di_per_year)
        self.b = float(self.b)
        self.d_min_per_year = float(self.d_min_per_year)
        if self.qi < 0.0:
            raise ValueError("initial rate cannot be negative, got %r" % (self.qi,))
        if self.di_per_year < 0.0:
            raise ValueError("initial decline cannot be negative, got %r"
                             % (self.di_per_year,))
        if self.b < 0.0:
            raise ValueError("b cannot be negative, got %r" % (self.b,))
        if self.d_min_per_year < 0.0:
            raise ValueError("terminal decline cannot be negative, got %r"
                             % (self.d_min_per_year,))
        if self.b >= 1.0 and self.d_min_per_year <= 0.0:
            raise ValueError(
                "b = %.2f with no terminal decline integrates to infinite "
                "reserves. Set d_min_per_year, or fit b below 1." % (self.b,))

    # ---------------------------------------------------------------- shape --

    @property
    def switch_year(self) -> float:
        """When the hyperbolic decline reaches the terminal rate.

        D(t) = Di / (1 + b Di t), so the switch is where that equals d_min.
        Exponential decline never slows, so it never switches.
        """
        if self.b <= 0.0 or self.d_min_per_year <= 0.0:
            return math.inf
        if self.d_min_per_year >= self.di_per_year:
            return 0.0
        return (self.di_per_year / self.d_min_per_year - 1.0) / (
            self.b * self.di_per_year)

    def decline_at(self, t_years: float) -> float:
        """Instantaneous nominal decline, per year."""
        if t_years >= self.switch_year:
            return self.d_min_per_year
        if self.b <= 0.0:
            return self.di_per_year
        return self.di_per_year / (1.0 + self.b * self.di_per_year * t_years)

    def rate(self, t_years: float) -> float:
        if t_years < 0.0:
            raise ValueError("time cannot be negative, got %r" % (t_years,))
        switch = self.switch_year
        if t_years <= switch:
            return self._hyperbolic_rate(t_years)
        q_switch = self._hyperbolic_rate(switch)
        return q_switch * math.exp(-self.d_min_per_year * (t_years - switch))

    def _hyperbolic_rate(self, t_years: float) -> float:
        if self.b <= 0.0:
            return self.qi * math.exp(-self.di_per_year * t_years)
        base = 1.0 + self.b * self.di_per_year * t_years
        return self.qi * base ** (-1.0 / self.b)

    def rates(self, t_years) -> np.ndarray:
        return np.array([self.rate(float(t)) for t in np.atleast_1d(t_years)])

    # ----------------------------------------------------------- cumulative --

    def cumulative(self, t_years: float) -> float:
        """Volume produced by `t_years`, in rate-units x day."""
        if t_years < 0.0:
            raise ValueError("time cannot be negative, got %r" % (t_years,))
        switch = self.switch_year
        if t_years <= switch:
            return self._hyperbolic_cumulative(t_years)
        total = self._hyperbolic_cumulative(switch)
        q_switch = self._hyperbolic_rate(switch)
        q_now = self.rate(t_years)
        return total + (q_switch - q_now) / self.d_min_per_year * DAYS_PER_YEAR

    def _hyperbolic_cumulative(self, t_years: float) -> float:
        if self.di_per_year <= 0.0:
            return self.qi * t_years * DAYS_PER_YEAR
        q = self._hyperbolic_rate(t_years)
        if self.b <= 0.0:
            return (self.qi - q) / self.di_per_year * DAYS_PER_YEAR
        if abs(self.b - 1.0) < 1e-9:
            # Harmonic. The hyperbolic expression is 0/0 at b = 1 and the
            # limit is logarithmic, so it is taken separately rather than
            # evaluated near the singularity.
            if q <= 0.0:
                return math.inf
            return self.qi / self.di_per_year * math.log(self.qi / q) * DAYS_PER_YEAR
        power = 1.0 - self.b
        return (self.qi / (power * self.di_per_year)
                * (1.0 - (q / self.qi) ** power) * DAYS_PER_YEAR)

    def eur(self, q_abandon: float = 0.0, max_years: float = 50.0) -> float:
        """Cumulative production to an economic limit.

        Both bounds matter and both are the operator's, not the curve's: a
        well is abandoned at the rate that stops paying its own lifting cost,
        and a forecast beyond a few decades is arithmetic rather than
        knowledge. Returning an EUR without either would be quoting the
        integral to infinity, which for b >= 1 does not converge.
        """
        if q_abandon < 0.0:
            raise ValueError("abandonment rate cannot be negative, got %r"
                             % (q_abandon,))
        if max_years <= 0.0:
            raise ValueError("horizon must be positive, got %r" % (max_years,))
        if q_abandon <= 0.0 or self.qi <= 0.0:
            return self.cumulative(max_years)
        if q_abandon >= self.qi:
            return 0.0
        return self.cumulative(min(self.time_to_rate(q_abandon), max_years))

    def time_to_rate(self, q: float) -> float:
        """When the rate falls to `q`. Infinite if it never does."""
        if q <= 0.0:
            return math.inf
        if q >= self.qi:
            return 0.0
        switch = self.switch_year
        q_switch = self._hyperbolic_rate(switch) if switch < math.inf else 0.0
        if switch == math.inf or q > q_switch:
            if self.di_per_year <= 0.0:
                return math.inf
            if self.b <= 0.0:
                return math.log(self.qi / q) / self.di_per_year
            return ((self.qi / q) ** self.b - 1.0) / (self.b * self.di_per_year)
        if self.d_min_per_year <= 0.0:
            return math.inf
        return switch + math.log(q_switch / q) / self.d_min_per_year

    def describe(self) -> str:
        kind = ("exponential" if self.b <= 0.0
                else "harmonic" if abs(self.b - 1.0) < 1e-6
                else "hyperbolic")
        text = ("%s decline, b = %.2f, %.1f%%/yr initial"
                % (kind, self.b, self.di_per_year * 100.0))
        if self.switch_year < math.inf:
            text += (", terminal %.1f%%/yr from year %.1f"
                     % (self.d_min_per_year * 100.0, self.switch_year))
        return text


# ================================================================== fitting ==


@dataclass
class ArpsFit:
    """A fitted decline, and the range the history cannot distinguish."""

    decline: ArpsDecline
    rms_log_residual: float
    points_used: int
    b_low: float
    b_high: float
    eur: float
    eur_low: float
    eur_high: float
    # The fits at the edges of the acceptable b range. Retained because they
    # are how far a forecast can honestly be said to reach: anything that
    # extrapolates this fit is also extrapolating these, and the spread
    # between them is the extrapolation's own uncertainty -- which the
    # residual scatter says nothing about.
    decline_low: ArpsDecline = None
    decline_high: ArpsDecline = None
    notes: list = field(default_factory=list)

    def prediction_spread(self, t_years) -> float:
        """Half-range of the acceptable fits' predictions, in log rate.

        The quantity a residual-scatter comparison is missing. Two fits that
        are indistinguishable over the measured history diverge as soon as
        they are extrapolated, and that divergence grows with the distance
        extrapolated while the scatter does not.
        """
        edges = [d for d in (self.decline_low, self.decline_high) if d is not None]
        if not edges:
            return 0.0
        middle = np.maximum(self.decline.rates(t_years), 1e-12)
        spread = 0.0
        for edge in edges:
            other = np.maximum(edge.rates(t_years), 1e-12)
            spread = max(spread, float(np.mean(np.abs(np.log(other / middle)))))
        return spread

    @property
    def eur_spread_fraction(self) -> float:
        """How wide the reserve range is, relative to the best estimate.

        This is the number to read before the EUR itself. A fit whose spread
        is 0.8 has not measured the reserves; it has measured that they are
        somewhere within a factor of two.
        """
        if self.eur <= 0.0:
            return 0.0
        return (self.eur_high - self.eur_low) / self.eur

    def band(self, t_years, confidence: float = 0.80):
        """A rate band from the scatter of the history about the fit.

        Deliberately *not* called P10/P90. It is the spread of the data around
        the fitted curve, which is mostly downtime, and it says nothing about
        whether the curve itself is right. The forecast's real uncertainty is
        in `b`, and that is reported as an EUR range instead -- a band drawn
        from residual scatter narrows with more history even as the reserves
        stay just as unknown, and reading it as a confidence interval on the
        forecast is the standard way to be precisely wrong about reserves.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1), got %r" % (confidence,))
        z = _normal_quantile(0.5 + confidence / 2.0)
        spread = math.exp(z * self.rms_log_residual)
        middle = self.decline.rates(t_years)
        return middle / spread, middle, middle * spread


def conformal_band(histories, confidence: float = 0.90,
                   fit_fraction: float = 0.6, horizon_months: int = 12,
                   d_min_per_year: float = DEFAULT_TERMINAL_DECLINE):
    """Calibrate a guaranteed forecast interval across a field of wells.

    The band method above is the scatter of a history about its own curve. It
    is useful and it is not a guarantee -- its own docstring says so. This
    is the guarantee, and getting it requires something  never did:
    other wells.

    The calibration is built the way the forecast is used. Each well is fitted
    on the first fit_fraction of its history and scored against the part
    held back, so a calibration residual is the same kind of object as a
    deployment residual: a real extrapolation error over a real horizon, not
    an in-sample one. Calibrating on in-sample residuals would produce a
    tight interval and no coverage, which is the usual way this is got wrong.

    Residuals are taken in log rate, because decline error is multiplicative
    -- being 20% out on a 2,000 bbl/day well and on a 20 bbl/day well are the
    same mistake -- and because an absolute-rate interval calibrated across a
    field would be set entirely by its biggest wells.

    **One caveat that must travel with the number.** The theorem assumes
    exchangeable units, and the units here are well-months, which cluster:
    twelve months from one well share that well's fit error and are not
    exchangeable with twelve months from another. The effective sample size
    is therefore nearer the well count than the residual count, and the
    finite-sample guarantee is weakened to an approximation.

    Measured on sixty synthetic wells calibrated and sixty held out, the
    realised coverage came to 0.918 against a nominal 0.90 -- above the
    theoretical upper bound of 0.90 + 1/(n+1), which is the clustering
    showing itself. It errs wide, which is the safe direction, but it is an
    empirical result rather than a theorem and should be reported as one.
    Calibrating one residual per well restores strict exchangeability at the
    cost of a much smaller calibration set. The conformal module's
    empirical_coverage exists so the realised number is always published
    next to the nominal one.

    Returns (interval, diagnostics). The interval is in log space; exponentiate
    its bounds around a forecast rate to get a rate interval.
    """
    from .conformal import calibrate, required_calibration

    predicted, actual, skipped = [], [], []
    for name, days, rates in histories:
        day = np.asarray(days, dtype=float)
        rate = np.asarray(rates, dtype=float)
        producing = rate > 0.0
        if producing.sum() < MIN_POINTS * 2:
            skipped.append(name)
            continue

        order = np.argsort(day)
        day, rate = day[order], rate[order]
        cut = int(len(day) * fit_fraction)
        if cut < MIN_POINTS or len(day) - cut < 1:
            skipped.append(name)
            continue

        try:
            fit = fit_arps(day[:cut], rate[:cut], d_min_per_year=d_min_per_year)
        except ValueError:
            skipped.append(name)
            continue

        origin = day[:cut][rate[:cut] > 0.0][0]
        horizon_days = horizon_months * 30.44
        ahead = (day[cut:] - day[cut - 1] <= horizon_days) & (rate[cut:] > 0.0)
        if not ahead.any():
            skipped.append(name)
            continue

        years = (day[cut:][ahead] - origin) / DAYS_PER_YEAR
        forecast = fit.decline.rates(years)
        usable = forecast > 0.0
        if not usable.any():
            skipped.append(name)
            continue
        predicted.extend(np.log(forecast[usable]))
        actual.extend(np.log(rate[cut:][ahead][usable]))

    diagnostics = {
        "wells_calibrated": len(histories) - len(skipped),
        "wells_skipped": skipped,
        "residuals": len(predicted),
        "required": required_calibration(confidence),
        "horizon_months": horizon_months,
        "fit_fraction": fit_fraction,
    }
    if not predicted:
        raise ValueError(
            "no well had enough history to both fit and score -- each needs "
            "at least %d producing points before the cut and one after"
            % MIN_POINTS)
    return calibrate(predicted, actual, confidence,
                     stratum="%d-month forecast" % horizon_months), diagnostics


def fit_arps(
    days,
    rates,
    b_bounds: tuple = (0.0, MAX_FITTED_B),
    d_min_per_year: float = DEFAULT_TERMINAL_DECLINE,
    q_abandon: float = 0.0,
    max_years: float = 50.0,
) -> ArpsFit:
    """Fit Arps to production history.

    For a fixed `b` the problem is linear, which is why this does not need an
    optimiser: raising the Arps relation to the power -b gives

        q^-b = qi^-b (1 + b Di t)

    so q^-b is a straight line in t, and least squares in that variable
    recovers qi and Di exactly. Only `b` is searched, over a coarse-to-fine
    scan. The fit minimises residuals in *log* rate rather than rate, because
    an absolute-error fit is dominated by the first few months and ignores
    the tail -- which is the part being extrapolated.

    Zero and negative rates are dropped. In monthly production data they are
    downtime, not decline, and a log-space fit cannot represent them anyway.
    """
    t = np.asarray(days, dtype=float) / DAYS_PER_YEAR
    q = np.asarray(rates, dtype=float)
    if t.shape != q.shape:
        raise ValueError("days and rates must be the same length, got %d and %d"
                         % (t.size, q.size))

    notes = []
    keep = (q > 0.0) & np.isfinite(q) & np.isfinite(t)
    dropped = int((~keep).sum())
    if dropped:
        notes.append(
            "%d of %d points dropped as zero or missing rate -- downtime, not "
            "decline." % (dropped, q.size))
    t, q = t[keep], q[keep]
    if t.size < MIN_POINTS:
        raise ValueError(
            "need at least %d producing points to fit a decline, got %d. A "
            "curve through fewer than that forecasts the noise."
            % (MIN_POINTS, t.size))

    order = np.argsort(t)
    t, q = t[order] - t[order][0], q[order]

    low, high = float(b_bounds[0]), float(b_bounds[1])
    if not 0.0 <= low <= high:
        raise ValueError("b bounds must satisfy 0 <= low <= high, got %r" % (b_bounds,))
    if high > MAX_FITTED_B:
        notes.append(
            "b capped at %.1f. Higher values fit a well still in transient "
            "flow and forecast reserves that do not exist."% MAX_FITTED_B)
        high = MAX_FITTED_B

    best = None
    grid = np.linspace(low, high, 41)
    for _refinement in range(3):
        for b in grid:
            candidate = _fit_at_b(t, q, float(b), d_min_per_year)
            if candidate is None:
                continue
            if best is None or candidate[0] < best[0]:
                best = candidate
        # Before refining, not after. A history that is flat or rising has
        # no acceptable fit at any b -- the exponential branch wants a
        # negative decline and every b > 0 wants a negative slope -- so the
        # whole grid returns None and there is nothing to refine around.
        # Reading `best[1].b` here raised TypeError and skipped the
        # ValueError below, which is the error the caller was given instead
        # of the one that says what is wrong.
        if best is None:
            break
        step = (grid[1] - grid[0]) if grid.size > 1 else 0.0
        if step <= 0.0:
            break
        grid = np.linspace(max(best[1].b - step, low),
                           min(best[1].b + step, high), 21)

    if best is None:
        raise ValueError(
            "no decline curve fits this history: it does not decline. Over "
            "the %d points given, rate is flat or rising, and every Arps "
            "curve through it needs a negative decline rate." % t.size)

    rms, decline = best
    eur = decline.eur(q_abandon, max_years)

    # The range of b that fits about as well. This is the answer to "how well
    # do we know the reserves", and it is usually much wider than the quality
    # of the fit suggests.
    threshold = rms * (1.0 + FIT_TOLERANCE_FRACTION)
    b_low, b_high = decline.b, decline.b
    eur_low, eur_high = eur, eur
    edge_low, edge_high = decline, decline
    for b in np.linspace(low, high, 161):
        candidate = _fit_at_b(t, q, float(b), d_min_per_year)
        if candidate is None or candidate[0] > threshold:
            continue
        if candidate[1].b < b_low:
            b_low, edge_low = candidate[1].b, candidate[1]
        if candidate[1].b > b_high:
            b_high, edge_high = candidate[1].b, candidate[1]
        volume = candidate[1].eur(q_abandon, max_years)
        eur_low, eur_high = min(eur_low, volume), max(eur_high, volume)

    fit = ArpsFit(decline=decline, rms_log_residual=float(rms),
                  points_used=int(t.size), b_low=float(b_low),
                  b_high=float(b_high), eur=float(eur), eur_low=float(eur_low),
                  eur_high=float(eur_high), decline_low=edge_low,
                  decline_high=edge_high, notes=notes)

    if fit.eur_spread_fraction > 0.5:
        fit.notes.append(
            "History fits b anywhere from %.2f to %.2f, which is a %.0f%% "
            "spread in reserves. That is what this data supports; a single EUR "
            "from it would be a choice, not a measurement."
            % (b_low, b_high, fit.eur_spread_fraction * 100.0))
    if t[-1] < 1.0:
        fit.notes.append(
            "Only %.0f months of history. Early decline is usually transient "
            "flow, which fits a high b and over-forecasts the tail."
            % (t[-1] * 12.0))
    if decline.b >= high - 1e-6 and high >= MAX_FITTED_B - 1e-6:
        # The least-squares optimum is against the bound, so the reported b is
        # the cap rather than the fit. Saying "b = 2.00" without this reads as
        # a measurement of a well that has not yet declined enough to measure.
        fit.notes.append(
            "The best fit is at the b = %.1f ceiling, meaning the history "
            "wants a flatter decline than can be defended. Treat the reserves "
            "as an upper bound until there is more history." % high)
    return fit


def _fit_at_b(t, q, b, d_min_per_year):
    """Least squares in log rate for one b. Returns (rms, decline) or None."""
    log_q = np.log(q)
    if b <= 1e-9:
        # Exponential: log q is linear in t, so this is exact rather than
        # searched.
        design = np.column_stack([np.ones_like(t), t])
        coefficients, *_ = np.linalg.lstsq(design, log_q, rcond=None)
        qi = math.exp(float(coefficients[0]))
        di = -float(coefficients[1])
        if not (qi > 0.0 and di >= 0.0 and math.isfinite(qi)):
            return None
        decline = ArpsDecline(qi=qi, di_per_year=di, b=0.0, d_min_per_year=0.0)
    else:
        # q^-b is linear in t. Fitted in that variable, then the parameters it
        # implies are scored in log space so every b is judged on the same
        # error measure -- scoring each in its own transformed space would
        # make the comparison meaningless.
        y = q ** (-b)
        design = np.column_stack([np.ones_like(t), t])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept, slope = float(coefficients[0]), float(coefficients[1])
        if intercept <= 0.0 or slope <= 0.0:
            return None
        qi = intercept ** (-1.0 / b)
        di = slope / (b * intercept)
        if not (math.isfinite(qi) and math.isfinite(di) and qi > 0.0 and di > 0.0):
            return None
        terminal = d_min_per_year
        if b >= 1.0 and terminal <= 0.0:
            terminal = DEFAULT_TERMINAL_DECLINE
        decline = ArpsDecline(qi=qi, di_per_year=di, b=b, d_min_per_year=terminal)

    predicted = decline.rates(t)
    if np.any(predicted <= 0.0) or not np.all(np.isfinite(predicted)):
        return None
    residual = log_q - np.log(predicted)
    return float(np.sqrt(np.mean(residual ** 2))), decline


# ==================================================================== Duong ==


@dataclass
class DuongDecline:
    """Duong (2011), for fracture-dominated tight and shale wells.

        q(t) = q1 t^-m exp[ a/(1-m) (t^(1-m) - 1) ]

    The reasoning behind it is the observation that q/Gp plots as a straight
    line against time on log-log axes for wells in long-term linear flow --
    which is what a hydraulically fractured well in a tight matrix is doing
    for most of its life. Arps assumes boundary-dominated flow and fits that
    period badly, usually by pushing b above 1.

    `t` is in days here, not years, and starts at 1: the form is singular at
    t = 0 and its parameters are conventionally fitted to daily time.
    """

    q1: float
    a: float
    m: float

    def __post_init__(self):
        self.q1 = float(self.q1)
        self.a = float(self.a)
        self.m = float(self.m)
        if self.q1 <= 0.0:
            raise ValueError("q1 must be positive, got %r" % (self.q1,))
        if self.a <= 0.0:
            raise ValueError("a must be positive, got %r" % (self.a,))
        if self.m <= 1.0:
            raise ValueError(
                "m must exceed 1; at m <= 1 the Duong form does not decline "
                "to zero. Got %r" % (self.m,))

    def _shape(self, t_days: float) -> float:
        power = 1.0 - self.m
        return math.exp(self.a / power * (t_days ** power - 1.0))

    def rate(self, t_days: float) -> float:
        if t_days < 1.0:
            # Fitted from day 1 by convention; the form diverges at zero.
            t_days = 1.0
        return self.q1 * t_days ** (-self.m) * self._shape(t_days)

    def cumulative(self, t_days: float) -> float:
        """Gp = q t^m / a, which is the relation the method is built on."""
        if t_days < 1.0:
            t_days = 1.0
        return self.q1 / self.a * self._shape(t_days)

    def rates(self, t_days) -> np.ndarray:
        return np.array([self.rate(float(t)) for t in np.atleast_1d(t_days)])


@dataclass
class DuongFit:
    """A fitted Duong curve and how well it actually describes the history.

    The misfit is reported rather than assumed small, because Duong's premise
    -- long-term linear flow -- is either true of a well or badly false, and
    nothing in the fitting arithmetic notices the difference. Fitted to a
    boundary-dominated well it returns parameters that look like parameters
    and miss the rates by 20-40%.
    """

    decline: DuongDecline
    rms_log_residual: float
    points_used: int
    notes: list = field(default_factory=list)

    @property
    def suspect(self) -> bool:
        return self.rms_log_residual > DUONG_MISFIT_WARNING


def fit_duong(days, rates, cumulative=None) -> DuongFit:
    """Fit Duong's parameters.

    `a` and `m` come from the straight line of log(q/Gp) against log(t), which
    is the whole basis of the method, so cumulative production is required
    rather than inferred. Where it is not supplied it is integrated from the
    rate history by trapezoid, which is what a monthly production record
    supports.
    """
    t = np.asarray(days, dtype=float)
    q = np.asarray(rates, dtype=float)
    if t.shape != q.shape:
        raise ValueError("days and rates must be the same length")

    keep = (q > 0.0) & (t > 0.0) & np.isfinite(q)
    t, q = t[keep], q[keep]
    if t.size < MIN_POINTS:
        raise ValueError("need at least %d producing points, got %d"
                         % (MIN_POINTS, t.size))
    order = np.argsort(t)
    t, q = t[order], q[order]

    if cumulative is None:
        gp = np.concatenate([[q[0] * t[0]], np.cumsum(
            0.5 * (q[1:] + q[:-1]) * np.diff(t)) + q[0] * t[0]])
    else:
        gp = np.asarray(cumulative, dtype=float)[keep][order]
    if np.any(gp <= 0.0):
        raise ValueError("cumulative production must be positive throughout")

    # log(q/Gp) = log a - m log t
    design = np.column_stack([np.ones_like(t), np.log(t)])
    coefficients, *_ = np.linalg.lstsq(design, np.log(q / gp), rcond=None)
    a = math.exp(float(coefficients[0]))
    m = -float(coefficients[1])
    if m <= 1.0:
        raise ValueError(
            "this history fits m = %.3f. Below 1 the Duong form does not "
            "decline, which means the well is not in the linear flow the "
            "method assumes -- fit Arps instead." % m)

    # q1 is then a scale, recovered by least squares on the fitted shape.
    shape = np.array([DuongDecline(q1=1.0, a=a, m=m).rate(float(x)) for x in t])
    if not np.all(np.isfinite(shape)) or np.sum(shape * shape) <= 0.0:
        raise ValueError(
            "the fitted a = %.2f and m = %.3f make the Duong shape function "
            "overflow, which means this history is not in linear flow. Fit "
            "Arps instead." % (a, m))
    q1 = float(np.sum(shape * q) / np.sum(shape * shape))
    if not (q1 > 0.0 and math.isfinite(q1)):
        # A well with no initial rate is not a model. This is what a
        # boundary-dominated history produces here: the shape function runs
        # away, and the scale that best matches it collapses to zero.
        raise ValueError(
            "the fit gives an initial rate of %r, which is not a well. The "
            "history does not follow Duong's q/Gp straight line -- fit Arps "
            "instead." % (q1,))

    decline = DuongDecline(q1=q1, a=a, m=m)
    predicted = decline.rates(t)
    residual = np.log(q) - np.log(np.maximum(predicted, 1e-300))
    rms = float(np.sqrt(np.mean(residual ** 2)))

    notes = []
    if rms > DUONG_MISFIT_WARNING:
        notes.append(
            "Duong misses these rates by %.0f%% rms. Its premise is long-term "
            "linear flow; a well in boundary-dominated flow fits Arps and not "
            "this." % (rms * 100.0))
    return DuongFit(decline=decline, rms_log_residual=rms,
                    points_used=int(t.size), notes=notes)


# =================================================================== shared ==


def _normal_quantile(p: float) -> float:
    """Inverse standard normal, Acklam's rational approximation.

    Needed because the engine is numpy-and-stdlib only -- scipy cannot be
    vendored into an air-gapped or serverless deployment. Accurate to about
    1e-9 in probability, far past what a production band needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1), got %r" % (p,))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    split = 0.02425
    if p < split:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - split:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                  + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
