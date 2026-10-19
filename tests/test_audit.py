"""Tests for the production-history audit.

This is the module most likely to be read by someone deciding whether to trust
the product, so the tests are built around what it must and must not claim.

**Must detect**: a well whose recent rate is inconsistent with the trend its
own earlier history established, in either direction.

**Must not detect**: a noisy well that is on trend. This is the harder half.
An absolute percentage threshold flags every well with downtime and misses the
quiet ones, so the comparison is against each well's own scatter -- and the
test below includes a well bouncing 15% month to month that must come back
clean.

**Must rank by barrels**: a 60% deviation on a 6 bbl/day stripper has to sit
below an 8% deviation on a 400 bbl/day well, or the report leads with
arithmetic and buries the finding.

The fixture is synthetic because the ground truth has to be known: these wells
are generated from Arps curves and then deliberately shifted, so "detected the
shift" is checkable rather than a matter of opinion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.audit import (
    ABOVE_TREND,
    BELOW_TREND,
    NO_HISTORY,
    RESERVES_UNSUPPORTED,
    STEEP_DECLINE,
    WellHistory,
    audit_histories,
)
from phasespace.sim.decline import DAYS_PER_YEAR, ArpsDecline

MONTHS = 30


def history(name, qi=300.0, di=0.45, b=0.8, noise=0.04, shift=1.0,
            shift_from=24, months=MONTHS, seed=5, reported=0.0):
    """One well's monthly rates, optionally shifted over the recent months.

    `shift` multiplies the last few months, which is how a workover, a choke
    change or an allocation error looks in a production record.
    """
    rng = np.random.default_rng(seed)
    days = np.round(np.arange(months) * 30.44)
    decline = ArpsDecline(qi=qi, di_per_year=di, b=b, d_min_per_year=0.05)
    rates = decline.rates(days / DAYS_PER_YEAR) * np.exp(
        rng.normal(0.0, noise, days.size))
    rates = rates.copy()
    rates[shift_from:] *= shift
    return WellHistory(name=name, days=list(days), rates=list(rates),
                       reported_rate=reported)


def field():
    """Six wells with known truth.

    QUIET-1  clean and on trend          -- must not be flagged
    NOISY-2  15% scatter, on trend       -- must not be flagged
    HIGH-3   35% above trend, 400 bbl/d  -- must be flagged, top by barrels
    LOW-4    30% below trend, 350 bbl/d  -- must be flagged
    STRIP-5  60% above trend, 6 bbl/d    -- flagged but ranked last
    STEEP-6  twice the field's decline   -- steep, not off-trend
    """
    return [
        history("QUIET-1", qi=300.0, noise=0.03, seed=1),
        history("NOISY-2", qi=280.0, noise=0.15, seed=2),
        history("HIGH-3", qi=400.0, noise=0.04, shift=1.35, seed=3),
        history("LOW-4", qi=350.0, noise=0.04, shift=0.70, seed=4),
        history("STRIP-5", qi=6.0, noise=0.05, shift=1.60, seed=5),
        history("STEEP-6", qi=250.0, di=0.95, noise=0.04, seed=6),
    ]


def kinds_for(report, well):
    return {finding.kind for finding in report.findings if finding.well == well}


# ------------------------------------------------------------ what it finds --


def test_a_well_above_its_own_trend_is_found():
    report = audit_histories(field())
    assert ABOVE_TREND in kinds_for(report, "HIGH-3")
    finding = next(f for f in report.findings
                   if f.well == "HIGH-3" and f.kind == ABOVE_TREND)
    assert finding.reported_rate > finding.expected_rate
    assert finding.sigma > 3.0
    assert finding.barrels_per_day > 40.0


def test_a_well_below_its_own_trend_is_found():
    report = audit_histories(field())
    assert BELOW_TREND in kinds_for(report, "LOW-4")
    finding = next(f for f in report.findings
                   if f.well == "LOW-4" and f.kind == BELOW_TREND)
    assert finding.reported_rate < finding.expected_rate
    assert finding.barrels_per_day > 30.0


def test_the_direction_is_kept_rather_than_collapsed():
    """A well producing more than it is credited with is usually an allocation
    problem, which means some other well in the same battery is being
    over-credited. Taking the absolute value loses that."""
    report = audit_histories(field())
    assert ABOVE_TREND in kinds_for(report, "HIGH-3")
    assert BELOW_TREND in kinds_for(report, "LOW-4")


def test_every_rate_finding_carries_its_innocent_explanations():
    """A report that reads a workover as fraud gets the whole tool dismissed.
    The finding is 'inconsistent with its own history', and what else could
    cause that travels with it."""
    report = audit_histories(field())
    for finding in report.findings:
        if finding.kind in (ABOVE_TREND, BELOW_TREND):
            assert finding.innocent_explanations, finding.well
            assert any("workover" in text or "downtime" in text
                       for text in finding.innocent_explanations), finding.well


# --------------------------------------------------------- what it must not --


def test_a_noisy_well_on_trend_is_not_flagged():
    """The harder half, and the reason the comparison is against each well's
    own scatter rather than a fixed percentage.

    NOISY-2 bounces 15% month to month and is exactly on its trend. An
    absolute threshold flags it; this must not. An earlier version did, by
    dividing the deviation by the standard error of the mean rather than by
    the scatter -- which would be right if the deviation were independent
    noise, and is not, because a real change shifts every month in the window
    the same way.
    """
    report = audit_histories(field())
    assert not (kinds_for(report, "NOISY-2") & {ABOVE_TREND, BELOW_TREND}), \
        kinds_for(report, "NOISY-2")


def test_a_clean_well_on_trend_is_not_flagged():
    report = audit_histories(field())
    assert not (kinds_for(report, "QUIET-1") & {ABOVE_TREND, BELOW_TREND})


def test_a_steeply_declining_well_is_not_called_off_trend():
    """STEEP-6 declines twice as fast as the field and is perfectly consistent
    with itself. Steepness is a separate finding, not a discrepancy."""
    report = audit_histories(field())
    assert not (kinds_for(report, "STEEP-6") & {ABOVE_TREND, BELOW_TREND})
    assert STEEP_DECLINE in kinds_for(report, "STEEP-6")


def test_a_well_with_leading_downtime_is_measured_from_the_right_origin():
    """Regression, and the kind that hides.

    `fit_arps` drops zero-rate months as downtime and only then rebases time
    to zero, so a well whose record begins with months of pre-production has a
    fit whose t = 0 is that many months later. The audit was passing the
    window's first day instead, evaluating the decline up to a year and a half
    further along its curve than intended -- which predicts a lower rate and
    manufactures an "above trend" finding.

    It produced no false positive, which is why it needed looking for rather
    than waiting for: extrapolating that much further also widens the
    acceptable fits' spread, so the inflated denominator swallowed the error.
    Both halves of this test matter -- the on-trend well must stay clean at
    every gap, and the genuinely shifted one must still be caught.
    """
    months = 42
    for gap in (0, 6, 12, 18):
        clean = history("ON-TREND", months=months, noise=0.03, seed=1)
        rates = list(clean.rates)
        rates[:gap] = [0.0] * gap
        clean.rates = rates
        report = audit_histories([clean])
        assert not (kinds_for(report, "ON-TREND") & {ABOVE_TREND, BELOW_TREND}),             "a leading gap of %d months manufactured a finding" % gap

    for gap in (0, 12):
        shifted = history("OFF-TREND", months=months, noise=0.03, seed=1,
                          shift=1.35, shift_from=36)
        rates = list(shifted.rates)
        rates[:gap] = [0.0] * gap
        shifted.rates = rates
        report = audit_histories([shifted])
        assert ABOVE_TREND in kinds_for(report, "OFF-TREND"),             "a real shift was missed with a leading gap of %d months" % gap


def test_a_shut_in_well_is_not_a_discrepancy():
    """Zero rates in the tested months mean the well is down, which the
    operator already knows. Reporting it as a finding spends the reader's
    attention on nothing."""
    well = history("DOWN-1", qi=300.0)
    well.rates = list(well.rates[:24]) + [0.0] * (MONTHS - 24)
    report = audit_histories([well])
    assert not (kinds_for(report, "DOWN-1") & {ABOVE_TREND, BELOW_TREND})


# ---------------------------------------------------------------- the rank ---


def test_the_ranking_is_by_barrels_not_by_percentage():
    """The decision this module turns on. STRIP-5 is 60% out and HIGH-3 is
    35% out, but STRIP-5 is a 6 bbl/day well. Sorting by percentage puts the
    stripper at the top of the report and buries the finding that pays for the
    study."""
    report = audit_histories(field())
    rate_findings = [f for f in report.ranked
                     if f.kind in (ABOVE_TREND, BELOW_TREND)]
    assert rate_findings[0].well == "HIGH-3"
    assert rate_findings[-1].well == "STRIP-5"

    stripper = next(f for f in rate_findings if f.well == "STRIP-5")
    top = rate_findings[0]
    # The stripper's percentage error is the larger of the two.
    assert (stripper.reported_rate / stripper.expected_rate
            > top.reported_rate / top.expected_rate)
    # And its barrels are the smaller, which is what decides the order.
    assert stripper.barrels_per_day < top.barrels_per_day


def test_barrels_at_stake_are_gross_rather_than_net():
    """A field with one well under-credited by 150 and two over-credited by 75
    nets to zero and is wrong by 300 barrels a day. The net figure is the one
    that makes an allocation problem invisible."""
    report = audit_histories(field())
    rate_findings = [f for f in report.findings
                     if f.kind in (ABOVE_TREND, BELOW_TREND)]
    signed = sum(
        (f.reported_rate - f.expected_rate) for f in rate_findings)
    assert report.barrels_at_stake > abs(signed)


def test_the_oil_price_is_the_callers_and_is_stated():
    """A dollar figure is an assumption about the future, not a measurement,
    so the price it came from travels with it."""
    report = audit_histories(field(), oil_price=85.0)
    assert report.oil_price == 85.0
    assert report.dollars_at_stake() == pytest.approx(
        report.barrels_at_stake * 85.0 * 365.25)
    assert any("$85" in note for note in report.notes), report.notes

    cheaper = audit_histories(field(), oil_price=40.0)
    assert cheaper.dollars_at_stake() < report.dollars_at_stake()
    assert cheaper.barrels_at_stake == pytest.approx(report.barrels_at_stake)


# --------------------------------------------------------------- reserves ----


def test_a_history_that_cannot_pin_b_says_the_reserves_are_a_choice():
    report = audit_histories(field())
    assert RESERVES_UNSUPPORTED in kinds_for(report, "NOISY-2")
    finding = next(f for f in report.findings
                   if f.well == "NOISY-2" and f.kind == RESERVES_UNSUPPORTED)
    # Carries no barrels, so it never outranks a rate finding.
    assert finding.barrels_per_day == 0.0


def test_only_rate_findings_carry_barrels():
    """Reserve spread and steep decline are real findings with no barrels per
    day attached. Giving them a number to make them rank would be inventing
    one."""
    report = audit_histories(field())
    for finding in report.findings:
        if finding.kind not in (ABOVE_TREND, BELOW_TREND):
            assert finding.barrels_per_day == 0.0, finding.kind


# ------------------------------------------------------------ the inputs -----


def test_a_well_with_too_little_history_says_so_rather_than_guessing():
    short = history("NEW-1", months=8)
    report = audit_histories([short])
    assert NO_HISTORY in kinds_for(report, "NEW-1")
    assert report.wells_audited == 0
    assert "NEW-1" in report.wells_skipped
    assert any("Nothing can be said" in f.detail for f in report.findings)


def test_mismatched_columns_are_refused():
    with pytest.raises(ValueError, match="must pair up"):
        WellHistory(name="BAD", days=[0, 30, 61], rates=[100.0, 90.0])


def test_nonsense_settings_are_refused():
    with pytest.raises(ValueError, match="oil price"):
        audit_histories(field(), oil_price=-10.0)
    with pytest.raises(ValueError, match="hold out at least"):
        audit_histories(field(), holdout=0)
    with pytest.raises(ValueError, match="sigma threshold"):
        audit_histories(field(), sigma_threshold=0.0)


def test_a_steep_decline_needs_a_field_to_compare_against():
    """With three wells a median is one well's opinion, so no comparison is
    made rather than one being made badly."""
    report = audit_histories(field()[:3])
    assert not any(f.kind == STEEP_DECLINE for f in report.findings)


def test_the_audit_does_not_modify_its_input():
    wells = field()
    before = [(well.name, list(well.days), list(well.rates)) for well in wells]
    audit_histories(wells)
    after = [(well.name, list(well.days), list(well.rates)) for well in wells]
    assert before == after


def test_a_whole_field_audits_in_one_pass():
    """One decline fit per well and milliseconds each, which is what makes
    this usable as a first contact rather than as a study."""
    many = [history("W-%03d" % index, qi=100.0 + index, seed=index)
            for index in range(60)]
    report = audit_histories(many)
    assert report.wells_audited == 60
    assert len(report.fits) == 60
