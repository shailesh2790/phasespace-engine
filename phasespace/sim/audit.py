"""Which wells' reported numbers their own history cannot support.

The rod-pump audit in `rodpump.batch` answers this from dynamometer cards,
which means it needs the operator's data before it can say anything useful.
This answers a weaker version of the same question from monthly production
alone -- which is public for most of the world's onshore wells, and so can be
run on a prospect's field before anyone has signed anything.

What a production history can and cannot support is the whole point of the
module, so it is worth being exact about it.

**It can detect inconsistency with a well's own trend.** Fit the decline to
all but the most recent months, predict those months, and compare the
prediction against what was reported. Crucially, the comparison is against the
*well's own scatter* about its own fit, not against a fixed percentage: a well
whose monthly rate bounces 15% from downtime has to miss by much more than one
that bounces 3% before anything has been learned. An absolute threshold flags
the noisy wells and misses the quiet ones, which is exactly backwards.

**It cannot tell you why.** A well above its trend has usually had a workover,
a stimulation, a choke change or a neighbouring injector turned up. A well
below it may be down for repair. Allocation error is one explanation among
several and often not the likeliest, so a finding here is "this number is not
consistent with this well's own history, by this much" and never "this well is
being misreported". Every finding carries the innocent explanations with it,
because a report that cries fraud at a workover gets the whole tool dismissed.

**Ranked by barrels, then priced.** A 60% deviation on a 4 bbl/day stripper is
arithmetic; 8% on a 400 bbl/day well is worth someone's week. The oil price
used is the caller's, stated in the output, because it is an assumption about
the future and not a measurement.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field

import numpy as np

from .decline import MIN_POINTS, ArpsFit, fit_arps

# How many of the most recent points are held out of the fit and then tested
# against it. Six monthly points is half a year: long enough that a single
# bad month cannot carry the finding, short enough that a real change in the
# well is still recent news.
DEFAULT_HOLDOUT = 6

# How many standard deviations of a well's own scatter counts as a finding.
# Two is about one month in twenty by chance for a single point, and the test
# below averages the holdout window, so the effective rate is far lower.
SIGMA_THRESHOLD = 2.0

# Below this the fit's residual is too small to divide by -- a history this
# clean is usually synthetic or already smoothed, and treating a 0.1% scatter
# as the yardstick would flag every well in the field.
MIN_SCATTER = 0.02

# A reserve range wider than this means the booked number is a choice rather
# than a measurement.
RESERVE_SPREAD_THRESHOLD = 0.50

# How much steeper than the field's median decline is worth remarking on.
STEEP_DECLINE_MULTIPLE = 1.6

ABOVE_TREND = "above trend"
BELOW_TREND = "below trend"
RESERVES_UNSUPPORTED = "reserves unsupported"
STEEP_DECLINE = "steep decline"
NO_HISTORY = "not enough history"


@dataclass
class WellHistory:
    """One well's monthly production, and what it is currently credited with."""

    name: str
    days: list
    rates: list
    # What the well is reported at now -- from allocation, a well test, or a
    # booked figure. Left at zero when the history's own last point is the
    # reported number, which is the usual case for public data.
    reported_rate: float = 0.0

    def __post_init__(self):
        if len(self.days) != len(self.rates):
            raise ValueError(
                "well %r has %d dates and %d rates; they must pair up"
                % (self.name, len(self.days), len(self.rates)))


@dataclass
class Finding:
    """One thing worth someone's attention, and how sure we are of it."""

    well: str
    kind: str
    barrels_per_day: float
    detail: str
    sigma: float = 0.0
    expected_rate: float = 0.0
    reported_rate: float = 0.0
    innocent_explanations: list = field(default_factory=list)

    def dollars_per_year(self, oil_price: float) -> float:
        return self.barrels_per_day * oil_price * 365.25


@dataclass
class AuditReport:
    findings: list = field(default_factory=list)
    fits: dict = field(default_factory=dict)
    wells_audited: int = 0
    wells_skipped: list = field(default_factory=list)
    oil_price: float = 0.0
    holdout: int = DEFAULT_HOLDOUT
    notes: list = field(default_factory=list)

    @property
    def ranked(self) -> list:
        """Findings worst first, by barrels rather than by percentage."""
        return sorted(self.findings, key=lambda f: -f.barrels_per_day)

    @property
    def barrels_at_stake(self) -> float:
        """Gross, not net.

        A field with one well 150 bbl/day under-credited and two 75 bbl/day
        over-credited nets to zero and is wrong by 300 barrels a day. The net
        figure is the one that makes an allocation problem invisible.
        """
        return sum(finding.barrels_per_day for finding in self.findings
                   if finding.kind in (ABOVE_TREND, BELOW_TREND))

    def dollars_at_stake(self) -> float:
        return self.barrels_at_stake * self.oil_price * 365.25


def audit_histories(
    wells: list,
    oil_price: float = 70.0,
    holdout: int = DEFAULT_HOLDOUT,
    sigma_threshold: float = SIGMA_THRESHOLD,
    q_abandon: float = 0.0,
) -> AuditReport:
    """Audit a field's production histories.

    One decline fit per well on the retained history, then the held-out months
    tested against it. Milliseconds per well, so a four-hundred-well field is
    a single request -- which is what makes this usable as a first contact
    rather than as a study.
    """
    if oil_price < 0.0:
        raise ValueError("oil price cannot be negative, got %r" % (oil_price,))
    if holdout < 1:
        raise ValueError("hold out at least one point, got %r" % (holdout,))
    if sigma_threshold <= 0.0:
        raise ValueError("sigma threshold must be positive, got %r"
                         % (sigma_threshold,))

    report = AuditReport(oil_price=oil_price, holdout=holdout)
    declines = {}

    for well in wells:
        days = np.asarray(well.days, dtype=float)
        rates = np.asarray(well.rates, dtype=float)
        order = np.argsort(days)
        days, rates = days[order], rates[order]

        producing = int(np.sum(rates > 0.0))
        if producing < MIN_POINTS + holdout:
            report.wells_skipped.append(well.name)
            report.findings.append(Finding(
                well=well.name, kind=NO_HISTORY, barrels_per_day=0.0,
                detail=("%d producing months, and %d are needed to fit a "
                        "decline and still hold %d back to test it. Nothing "
                        "can be said about this well yet."
                        % (producing, MIN_POINTS + holdout, holdout))))
            continue

        retained_days, retained_rates = days[:-holdout], rates[:-holdout]
        held_days, held_rates = days[-holdout:], rates[-holdout:]
        try:
            fit = fit_arps(retained_days, retained_rates, q_abandon=q_abandon)
        except ValueError as error:
            report.wells_skipped.append(well.name)
            report.findings.append(Finding(
                well=well.name, kind=NO_HISTORY, barrels_per_day=0.0,
                detail="no decline could be fitted: %s" % error))
            continue

        report.wells_audited += 1
        report.fits[well.name] = fit
        declines[well.name] = fit.decline.di_per_year

        _check_trend(report, well, fit, retained_days, retained_rates,
                     held_days, held_rates, sigma_threshold)
        _check_reserves(report, well, fit)

    _check_relative_decline(report, declines)

    if report.wells_skipped:
        report.notes.append(
            "%d of %d wells had too little history to audit: %s."
            % (len(report.wells_skipped),
               len(report.wells_skipped) + report.wells_audited,
               ", ".join(report.wells_skipped[:8])
               + (" and others" if len(report.wells_skipped) > 8 else "")))
    if report.barrels_at_stake > 0.0:
        report.notes.append(
            "%.0f barrels a day sit in rates that disagree with their own "
            "well's history, gross rather than net. At $%.0f that is $%.1fM a "
            "year -- a price this tool takes from you rather than knowing."
            % (report.barrels_at_stake, oil_price,
               report.dollars_at_stake() / 1e6))
    return report


def _check_trend(report, well, fit: ArpsFit, retained_days, retained_rates,
                 held_days, held_rates, sigma_threshold):
    """Do the held-out months agree with the trend fitted before them?"""
    producing = held_rates > 0.0
    if not np.any(producing):
        # Every held-out month is zero. That is a shut-in well, not a
        # discrepancy, and reporting it as one wastes the reader's attention
        # on something they already know.
        return

    # The fit's time origin is the first *producing* day of the retained
    # window, not the first day of it.
    #
    # `fit_arps` drops zero-rate months as downtime and only then rebases time
    # to zero, so a well whose record starts with months of pre-production or
    # downtime has a fit whose t = 0 is that many months later. Passing the
    # window's first day instead evaluated the decline up to a year and a half
    # further along its curve than intended, which predicts a lower rate and
    # therefore manufactures an "above trend" finding out of nothing.
    #
    # It did not show up as a false positive, which is the uncomfortable part:
    # extrapolating that much further also widens the acceptable fits' spread,
    # so the inflated denominator swallowed the error. A wrong number hidden
    # by a wrong uncertainty is still wrong, and the next change to either one
    # would have exposed it.
    producing_days = retained_days[retained_rates > 0.0]
    origin = producing_days[0] if producing_days.size else retained_days[0]
    years = (held_days[producing] - origin) / 365.25
    predicted = fit.decline.rates(years)
    observed = held_rates[producing]
    if np.any(predicted <= 0.0):
        return

    # Compared in log space, and against the well's own scatter about its own
    # fit. A well that bounces 15% from downtime has to miss by much more than
    # one that bounces 3% before anything has been learned.
    #
    # Divided by the scatter itself, not by the standard error of the mean.
    # The first version used sigma/sqrt(n), on the reasoning that averaging
    # six months tightens the estimate -- which would be true if the deviation
    # were independent noise. It is not: a workover, a choke change or an
    # allocation error shifts every month in the window the same way, so the
    # sqrt(n) is unearned. It inflated every score by about 2.5 and put two
    # wells on the report whose rates were entirely consistent with their own
    # history, which is the exact failure this test exists to avoid. It also
    # ignores the uncertainty in extrapolating the fit at all, which pushes
    # the honest denominator up rather than down.
    # Two sources of uncertainty, added in quadrature.
    #
    # The scatter is how much this well's monthly rate bounces about its own
    # fit -- downtime, metering, allocation noise. The spread is how much the
    # *fits that are indistinguishable over the measured history* disagree
    # once extrapolated into the tested months, which is a different thing
    # entirely and grows with the distance extrapolated while the scatter does
    # not.
    #
    # Leaving the spread out flagged the cleanest well in the test field. Its
    # rate was exactly on trend; what was off was the extrapolation, because
    # a b fitted to two years of even 3% noise is uncertain enough to drift
    # several percent over the following six months. Calling that the well's
    # problem would be reporting the method's own error as a finding.
    scatter = max(fit.rms_log_residual, MIN_SCATTER)
    spread = fit.prediction_spread(years)
    uncertainty = math.sqrt(scatter ** 2 + spread ** 2)

    deviation = float(np.mean(np.log(observed / predicted)))
    sigma = abs(deviation) / uncertainty
    if sigma < sigma_threshold:
        return

    expected = float(np.mean(predicted))
    reported = well.reported_rate if well.reported_rate > 0.0 else float(
        np.mean(observed))
    barrels = abs(reported - expected)

    if deviation > 0.0:
        report.findings.append(Finding(
            well=well.name, kind=ABOVE_TREND, barrels_per_day=barrels,
            sigma=sigma, expected_rate=expected, reported_rate=reported,
            detail=("Producing %.0f bbl/day against %.0f on its own fitted "
                    "decline -- %.0f%% high, and %.1f times the combined "
                    "uncertainty of this well's own scatter and the "
                    "extrapolation."
                    % (reported, expected, (reported / expected - 1.0) * 100.0,
                       sigma)),
            innocent_explanations=[
                "a workover, stimulation or recompletion since the fitted period",
                "a choke or lift change",
                "a neighbouring injector turned up",
                "allocation crediting this well with another's production",
            ]))
    else:
        report.findings.append(Finding(
            well=well.name, kind=BELOW_TREND, barrels_per_day=barrels,
            sigma=sigma, expected_rate=expected, reported_rate=reported,
            detail=("Producing %.0f bbl/day against %.0f on its own fitted "
                    "decline -- %.0f%% low, and %.1f times the combined "
                    "uncertainty of this well's own scatter and the "
                    "extrapolation."
                    % (reported, expected, (1.0 - reported / expected) * 100.0,
                       sigma)),
            innocent_explanations=[
                "downtime or a pump failure inside the tested months",
                "a deliberate rate restriction",
                "rising water cut, which this test does not see",
                "liquid loading or a failing lift system",
            ]))


def _check_reserves(report, well, fit: ArpsFit):
    """Is the reserve number this history implies a measurement or a choice?"""
    if fit.eur_spread_fraction <= RESERVE_SPREAD_THRESHOLD:
        return
    report.findings.append(Finding(
        well=well.name, kind=RESERVES_UNSUPPORTED, barrels_per_day=0.0,
        detail=("Every b from %.2f to %.2f fits this history about as well, "
                "and they imply %.0f to %.0f thousand barrels. A single "
                "booked figure from this data is a selection, not a "
                "measurement."
                % (fit.b_low, fit.b_high, fit.eur_low / 1000.0,
                   fit.eur_high / 1000.0))))


def _check_relative_decline(report, declines: dict):
    """Which wells are falling away faster than the rest of the field.

    Compared against the field's median rather than an absolute rate, because
    what counts as a steep decline is a property of the reservoir and not of
    the number. A field declining at 40% a year has no 15%-a-year wells to
    compare against.
    """
    if len(declines) < 4:
        # With three wells a median is one well's opinion.
        return
    values = np.array(list(declines.values()), dtype=float)
    median = float(np.median(values))
    if median <= 0.0:
        return
    for name, decline in declines.items():
        if decline > median * STEEP_DECLINE_MULTIPLE:
            report.findings.append(Finding(
                well=name, kind=STEEP_DECLINE, barrels_per_day=0.0,
                detail=("Declining at %.0f%% a year against a field median of "
                        "%.0f%%. Steeper than its neighbours is a well "
                        "problem rather than a reservoir one, which is the "
                        "kind that can be fixed."
                        % (decline * 100.0, median * 100.0))))
