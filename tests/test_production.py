"""Reading a monthly production sheet, and what it refuses to infer from one.

The fixtures here are written by hand rather than taken from any operator's
file. A sheet is a small enough format to construct exactly, and the cases
that matter -- a restart, a workover inside a producing run, a blank cell
against a zero -- are easier to state deliberately than to find.

Most of these tests are about structure rather than numbers, because the
numbers in a production sheet are the easy part. What decides whether a
forecast is right is which months belong to the same history.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.production import (
    STEP_RATIO,
    days_in_period,
    declinable,
    field_decline,
    parse_period,
    read_wide_csv,
    series_for_fit,
)

HEADER = "Well,Sand,Oct-22,Nov-22,Dec-22,Jan-23,Feb-23,Mar-23,Apr-23,May-23,status"


def sheet(tmp_path, *rows, header=HEADER):
    path = tmp_path / "sheet.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return read_wide_csv(path)


# ------------------------------------------------------------- the header --


@pytest.mark.parametrize("label,expected", [
    ("Oct-22", (2022, 10)),
    ("oct-22", (2022, 10)),
    ("Oct 22", (2022, 10)),
    ("October 2022", (2022, 10)),
    ("Oct/2022", (2022, 10)),
    ("Well No.", None),
    ("status", None),
    ("", None),
    ("Octopus-22", None),
])
def test_period_columns_are_recognised_by_parsing_not_by_position(label, expected):
    assert parse_period(label) == expected


def test_a_sheet_with_no_month_columns_says_so_instead_of_reading_nothing():
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "bad.csv"
        path.write_text("Well,Sand,status\nA-1,X,flowing\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no month columns"):
            read_wide_csv(path)


def test_months_out_of_order_are_reported_rather_than_silently_sorted():
    """Sorting them would be the helpful thing and the wrong one: a sheet
    with its columns out of order has been edited by hand, and which edit
    went wrong is the engineer's question, not the reader's."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "s.csv"
        path.write_text("Well,Sand,Dec-22,Oct-22,Nov-22\nA-1,X,10,20,30\n",
                        encoding="utf-8")
        result = read_wide_csv(path)
    assert any(n.code == "unsorted_periods" for n in result.notes)


def test_february_is_shorter_than_january():
    assert days_in_period((2023, 1)) == 31
    assert days_in_period((2023, 2)) == 28
    assert days_in_period((2024, 2)) == 29


# -------------------------------------------------------------- the cells --


def test_a_blank_cell_is_not_a_zero(tmp_path):
    """Blank means nobody recorded anything. Zero means somebody recorded
    that the well made nothing. Collapsing them turns an unmeasured month
    into a month of downtime, and a decline fit believes it."""
    result = sheet(tmp_path, "A-1,X,100,,100,0,100,100,100,100,flowing")
    volumes = result.well("A-1").volumes
    assert np.isnan(volumes[1])
    assert volumes[3] == 0.0


def test_thousands_separators_are_read(tmp_path):
    result = sheet(tmp_path, '"A-1",X,"1,250",900,900,900,900,900,900,900,flowing')
    assert result.well("A-1").volumes[0] == 1250.0


def test_a_daily_rate_uses_the_real_length_of_each_month(tmp_path):
    """Dividing by 30 is wrong by up to 7%, and the error is seasonal --
    exactly the shape a decline fit absorbs into the decline exponent."""
    result = sheet(tmp_path, "A-1,X,3100,3000,3100,3100,2800,3100,3000,3100,flowing")
    rates = result.well("A-1").daily_rates()
    assert rates[0] == pytest.approx(100.0)      # October, 31 days
    assert rates[1] == pytest.approx(100.0)      # November, 30 days
    assert rates[4] == pytest.approx(100.0)      # February 2023, 28 days


def test_elapsed_time_is_measured_to_the_middle_of_each_month(tmp_path):
    """A monthly volume is an integral over the month, so the rate it
    implies belongs to the middle of it. Anchoring at the start shifts
    every point half a month early and inflates the fitted initial rate."""
    result = sheet(tmp_path, "A-1,X,100,100,100,100,100,100,100,100,flowing")
    elapsed = result.well("A-1").elapsed_days()
    assert elapsed[0] == pytest.approx(15.5)     # half of October
    assert elapsed[1] == pytest.approx(31 + 15)  # October, then half of November


# ----------------------------------------------------------- the structure --


def test_a_shut_in_month_splits_the_history(tmp_path):
    result = sheet(tmp_path, "A-1,X,100,90,80,0,0,60,55,50,flowing")
    runs = result.well("A-1").segments()
    assert [(s.start, s.stop) for s in runs] == [(0, 3), (5, 8)]
    assert runs[0].reason == "shut in"


def test_a_run_reaching_the_end_of_the_sheet_did_not_shut_in(tmp_path):
    """The sheet stopped, not the well. Reporting otherwise would mark
    every currently-producing well in the field as having died in the last
    month covered."""
    result = sheet(tmp_path, "A-1,X,100,90,80,70,60,55,50,45,flowing")
    runs = result.well("A-1").segments()
    assert len(runs) == 1
    assert "still producing" in runs[0].reason


def test_a_step_change_inside_a_producing_run_splits_it(tmp_path):
    """A well going 50, 500 in consecutive months was worked over. Nothing
    went to zero, so splitting on zeros alone misses it, and no single
    decline curve describes both sides."""
    result = sheet(tmp_path, "A-1,X,100,90,80,70,700,650,600,550,flowing")
    runs = result.well("A-1").segments()
    assert [(s.start, s.stop) for s in runs] == [(0, 4), (4, 8)]
    assert "stepped" in runs[0].reason


def test_ordinary_decline_is_not_mistaken_for_a_step(tmp_path):
    """The false positive that would split every well in the field. The
    steepest month-on-month fall any Arps curve produces is far gentler
    than STEP_RATIO."""
    rates = [1000.0 * 0.80 ** i for i in range(8)]
    result = sheet(tmp_path, "A-1,X," + ",".join("%.0f" % r for r in rates) + ",flowing")
    assert len(result.well("A-1").segments()) == 1
    assert 0.80 > 1.0 / STEP_RATIO


def test_the_last_run_is_used_not_the_longest(tmp_path):
    """A well's current behaviour is what it is doing now. The longest run
    may be from before a workover that changed the well."""
    result = sheet(tmp_path, "A-1,X,100,95,90,85,0,60,55,50,flowing")
    assert (result.well("A-1").last_segment().start,
            result.well("A-1").last_segment().stop) == (5, 8)


def test_a_well_shut_in_at_the_end_is_flagged(tmp_path):
    result = sheet(tmp_path, "A-1,X,100,95,90,85,80,75,70,0,flowing")
    well = result.well("A-1")
    assert well.shut_in_now()
    _segment, reasons = declinable(well)
    assert any("shut in at the end" in r for r in reasons)


# ------------------------------------------------------------ the refusals --


def test_a_rising_well_is_refused_rather_than_fitted(tmp_path):
    """Arps on a rising history returns a negative decline and an unbounded
    reserve, and reports neither as an error."""
    result = sheet(tmp_path, "A-1,X,100,110,120,130,140,150,160,170,flowing")
    segment, reasons = declinable(result.well("A-1"))
    assert segment is None
    assert any("rising" in r for r in reasons)


def test_a_run_too_short_to_fit_is_refused_with_its_length(tmp_path):
    result = sheet(tmp_path, "A-1,X,100,90,80,0,0,0,60,55,flowing")
    segment, reasons = declinable(result.well("A-1"))
    assert segment is None
    assert any("2 month(s)" in r for r in reasons)


def test_a_well_that_never_produced_is_a_fact_not_an_error(tmp_path):
    result = sheet(tmp_path, "A-1,X,0,0,0,0,0,0,0,0,flowing")
    segment, reasons = declinable(result.well("A-1"))
    assert segment is None
    assert reasons == ["the well has no producing month in this sheet"]


def test_a_clean_declining_well_is_accepted(tmp_path):
    rates = [3100.0 * 0.95 ** i for i in range(8)]
    result = sheet(tmp_path, "A-1,X," + ",".join("%.0f" % r for r in rates) + ",flowing")
    segment, reasons = declinable(result.well("A-1"))
    assert segment is not None
    assert reasons == []


def test_the_fitted_series_is_rebased_to_its_own_run(tmp_path):
    """The fit's clock starts at the run, not at the sheet. A run beginning
    in month five must not be handed four months of absent history."""
    result = sheet(tmp_path, "A-1,X,900,850,0,0,800,760,720,680,flowing")
    well = result.well("A-1")
    days, rates = series_for_fit(well, well.last_segment())
    assert days[0] == 0.0
    assert rates.size == 4


# ---------------------------------------------------------------- the units --


def test_the_sheet_is_not_assumed_to_be_in_any_unit(tmp_path):
    result = sheet(tmp_path, "A-1,X,100,90,80,70,60,55,50,45,flowing")
    codes = {n.code for n in result.notes}
    assert "undeclared_units" in codes
    assert "volume_or_rate" in codes


# --------------------------------------------------------------- the field --


def two_well_sheet(tmp_path):
    """One well fittable, one rising and therefore not."""
    falling = [3100.0 * 0.93 ** i for i in range(8)]
    rising = [1000.0 * 1.05 ** i for i in range(8)]
    return sheet(
        tmp_path,
        "A-1,X," + ",".join("%.0f" % r for r in falling) + ",flowing",
        "A-2,Y," + ",".join("%.0f" % r for r in rising) + ",flowing")


def test_the_field_total_says_how_much_of_the_field_it_covers(tmp_path):
    """The usual way a field forecast is wrong: the wells that could not be
    fitted are dropped, the rest are summed, and nothing on the chart says
    the total is short by the dropped wells' production."""
    outlook = field_decline(two_well_sheet(tmp_path))
    assert len(outlook.fitted) == 1
    assert len(outlook.blocked) == 1
    assert 0.0 < outlook.covered_fraction < 1.0
    assert outlook.covered_rate < outlook.current_rate
    assert "not in any field total" in outlook.summary()


def test_blocked_wells_keep_their_reason_and_their_rate(tmp_path):
    outlook = field_decline(two_well_sheet(tmp_path))
    blocked = outlook.blocked[0]
    assert blocked.name == "A-2"
    assert blocked.last_rate > 0.0
    assert any("rising" in r for r in blocked.reasons)


def test_wells_are_summed_on_one_clock_not_each_on_its_own(tmp_path):
    """The regression that matters. Each well's fit has its own origin --
    the start of that well's last producing run -- and wells are worked
    over at different times. Summing `rate(0)` adds up each well's rate on
    its own start date, which is not a field rate on any date. On the sheet
    this was built against it overstated the field by 39%.

    Here A-2 restarts halfway through, so its run begins four months after
    A-1's. If the two are summed on their own clocks, the total at the end
    of the sheet comes out well above what was actually measured.
    """
    long_header = ("Well,Sand," + ",".join(
        "%s-%02d" % (m, y) for y, months in
        ((22, ["Oct", "Nov", "Dec"]),
         (23, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]))
        for m in months) + ",status")
    falling = [3100.0 * 0.93 ** i for i in range(15)]
    restarted = [0.0] * 5 + [3100.0 * 0.93 ** i for i in range(10)]
    result = sheet(
        tmp_path,
        "A-1,X," + ",".join("%.0f" % r for r in falling) + ",flowing",
        "A-2,Y," + ",".join("%.0f" % r for r in restarted) + ",flowing",
        header=long_header)
    outlook = field_decline(result)
    assert len(outlook.fitted) == 2

    ages = sorted(w.age_years for w in outlook.fitted)
    assert ages[0] < ages[1], "the two runs must not share an age"

    at_end = outlook.rate_at(0.0)
    assert at_end == pytest.approx(outlook.covered_rate, rel=0.05), (
        "the field total at the end of the sheet must match what the sheet "
        "measured there, not the sum of each well's own starting rate")


def test_a_field_forecast_declines(tmp_path):
    outlook = field_decline(two_well_sheet(tmp_path))
    assert outlook.rate_at(2.0) < outlook.rate_at(0.0)


# ---------------------------------------------------------------- the audit --


def test_a_flat_history_is_refused_with_a_sentence_not_a_typeerror():
    """A regression on `decline.fit_arps`, found by running the audit over a
    real sheet. A flat or rising history has no acceptable fit at any b --
    the exponential branch wants a negative decline, every b > 0 wants a
    negative slope -- so the whole scan returns nothing. The guard for that
    existed but sat *after* `centre = best[1].b`, so the caller got
    TypeError: 'NoneType' object is not subscriptable instead of the
    sentence saying the well does not decline."""
    from phasespace.sim.decline import fit_arps

    days = [0.0, 30.5, 61.0, 92.0, 121.5, 151.0, 181.5, 212.0]
    flat = [1700.0, 1702.0, 1698.0, 1701.0, 1699.0, 1703.0, 1700.0, 1702.0]
    with pytest.raises(ValueError, match="does not decline"):
        fit_arps(days, flat)

    rising = [1000.0 + 40.0 * i for i in range(8)]
    with pytest.raises(ValueError, match="does not decline"):
        fit_arps(days, rising)


def test_the_audit_sees_only_the_last_run_of_each_well(tmp_path):
    """An audit fitted across a workover flags the workover. The rate
    genuinely jumped and the fit genuinely cannot explain it, so the report
    is correct and useless -- it has rediscovered an intervention the
    operator performed on purpose, and buried the wells that are actually
    down among the ones that are not."""
    long_header = ("Well,Sand," + ",".join(
        "%s-%02d" % (m, y) for y, months in
        ((22, ["Oct", "Nov", "Dec"]),
         (23, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]))
        for m in months) + ",status")
    # Six months at one level, then a workover, then a clean decline.
    before = [500.0] * 6
    after = [4000.0 * 0.96 ** i for i in range(9)]
    result = sheet(
        tmp_path,
        "A-1,X," + ",".join("%.0f" % r for r in before + after) + ",flowing",
        header=long_header)

    from phasespace.sim.production import audit_sheet

    report = audit_sheet(result, holdout=3)
    assert report.wells_audited == 1
    # The post-workover run is a clean decline, so nothing should be flagged
    # as off-trend. Fitting across the step would flag it loudly.
    assert not [f for f in report.findings
                if f.kind in ("above trend", "below trend")]


def test_wells_the_sheet_cannot_support_are_named_in_the_audit(tmp_path):
    """A well left out of an audit is a fact the engineer needs, not a
    silent omission that makes the field look cleaner than it is."""
    from phasespace.sim.production import audit_sheet

    rising = [1000.0 * 1.05 ** i for i in range(8)]
    falling = [3100.0 * 0.93 ** i for i in range(8)]
    result = sheet(
        tmp_path,
        "A-1,X," + ",".join("%.0f" % r for r in falling) + ",flowing",
        "A-2,Y," + ",".join("%.0f" % r for r in rising) + ",flowing")
    report = audit_sheet(result, holdout=3)
    assert any("A-2" in entry and "rising" in entry
               for entry in report.wells_skipped)
