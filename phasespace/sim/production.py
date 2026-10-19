"""Read a field's monthly production sheet, and say what may be fitted to it.

PRD FR-1: connect a field's well-test sheets without custom code. The sheet
this is built against is the one every asset actually has -- a wide grid
with a well per row, a month per column, and a few attribute columns at
either end -- and it is the form that every tool in the industry imports
badly, because the hard parts are not the parsing.

**What makes it hard is that the numbers mean different things in different
cells, and the sheet does not say so.**

*A zero is not a small number.* It is a month the well did not produce, and
it is the single most consequential cell in the file. Averaged in, it drags
a well's rate down and invents a decline that never happened. Dropped
silently, it hides downtime and overstates deliverability. Interpolated
across, it invents production. The only correct handling is to treat it as
what it is -- a shut-in month -- and that means it belongs to the *structure*
of the history rather than to its values.

*A restart is not a continuation.* A well that stopped for three months and
came back is two histories, not one, because the restart resets the initial
rate. Fitting one Arps curve across the gap fits neither segment and
extrapolates from the average of two different wells. `segments` splits
them, and `last_segment` is what a forecast should actually use.

*A step up inside a producing run is a workover, not decline.* A well going
5020, 24480 in consecutive months was worked over, and no decline curve
describes both sides of that. Splitting on zeros alone will not catch it,
because nothing went to zero.

*The units are not in the file.* Four or five digits a month could be
barrels, cubic metres, or standard cubic metres of gas, and these differ by
factors this module cannot resolve from the numbers. Nor can it tell a
monthly *volume* from an average daily *rate*, and the difference is a
factor of thirty. Both are reported and neither is assumed, because a
decline fit on a mislabelled column is wrong by 30x while looking perfect:
the shape is identical and only the axis is wrong.

*Months are not equal.* Converting a monthly volume to a daily rate by
dividing by 30 is wrong by up to 7% and the error is seasonal, which is
exactly the shape a decline fit will absorb into the decline exponent.
`daily_rates` divides by the real length of each month.

Nothing here converts units or fills a gap. It reads the sheet, describes
what is in it, and refuses the fits the data cannot support.
"""
from __future__ import annotations

import calendar
import csv
import re
from dataclasses import dataclass, field

import numpy as np

# A producing run shorter than this cannot support a decline fit, and
# `decline.MIN_POINTS` is the authority on how short that is. Imported
# lazily in `declinable` so this module stays usable without it.
MIN_SEGMENT = 6

# A month-on-month jump larger than this inside a producing run is a
# workover, a re-completion or a choke change, not decline. Chosen well
# above what any Arps curve can produce between consecutive months: at
# b = 2 and a 90%/year decline the steepest one-month *fall* is about 8%,
# so a doubling in either direction is not the reservoir.
STEP_RATIO = 2.0

# Column headers that are periods rather than attributes. Covers the two
# spellings a monthly sheet is exported with.
MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_abbr)
          if name}
MONTHS.update({name.lower(): index
               for index, name in enumerate(calendar.month_name) if name})

PERIOD = re.compile(r"^\s*([A-Za-z]{3,9})[\s\-/]*(\d{2}|\d{4})\s*$")


def parse_period(label: str):
    """`Oct-22` -> (2022, 10). Returns None for an attribute column.

    Two-digit years are read as 2000-2099. A production sheet covering the
    1960s exists but is not what anyone is importing into a live model, and
    guessing a century from a two-digit year is how a 2022 column silently
    becomes 1922 and sorts to the front.
    """
    match = PERIOD.match(label or "")
    if not match:
        return None
    name, year = match.group(1).lower(), int(match.group(2))
    if name not in MONTHS:
        return None
    return (2000 + year if year < 100 else year), MONTHS[name]


def days_in_period(period) -> int:
    return calendar.monthrange(period[0], period[1])[1]


@dataclass
class Segment:
    """One continuous producing run, in index space over the sheet's months."""

    start: int
    stop: int               # exclusive
    reason: str = ""        # why it ended, where that is known

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class WellHistory:
    """One row of the sheet, with the structure its zeros imply."""

    name: str
    periods: list
    volumes: np.ndarray
    attributes: dict = field(default_factory=dict)

    @property
    def group(self) -> str:
        return str(self.attributes.get("group", ""))

    def days(self) -> np.ndarray:
        return np.array([days_in_period(p) for p in self.periods], dtype=float)

    def daily_rates(self) -> np.ndarray:
        """Volume per day, using the real length of each month.

        Only meaningful if the column really is a monthly volume. Whether
        it is is `ProductionSheet.measure`, which this module determines it
        cannot determine.
        """
        return np.asarray(self.volumes, dtype=float) / self.days()

    def elapsed_days(self) -> np.ndarray:
        """Days from the first month's start to each month's midpoint.

        The midpoint, not the start: a monthly volume is the integral over
        the month, so the rate it implies is the rate in the middle of it.
        Anchoring at the start shifts every point half a month early, which
        a decline fit absorbs as a slightly larger initial rate -- small,
        systematic, and free to avoid.
        """
        lengths = self.days()
        starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        return starts + lengths / 2.0

    def producing(self) -> np.ndarray:
        values = np.asarray(self.volumes, dtype=float)
        return np.isfinite(values) & (values > 0.0)

    def segments(self) -> list:
        """Maximal producing runs, split at shut-in months and at step changes.

        Both splits are necessary and neither is sufficient. Splitting only
        on zeros misses a well worked over without stopping; splitting only
        on steps misses a well that stopped and came back at a similar rate.
        """
        live = self.producing()
        values = np.asarray(self.volumes, dtype=float)
        found, start = [], None
        for index in range(len(values) + 1):
            inside = index < len(values) and live[index]
            if inside and start is None:
                start = index
            elif not inside and start is not None:
                # A run that reaches the end of the sheet did not stop; the
                # sheet did. Calling that "shut in" would report every
                # currently-producing well in the field as having died in
                # the last month covered.
                found.append(Segment(
                    start, index,
                    "shut in" if index < len(values)
                    else "still producing at the end of the sheet"))
                start = None
            elif inside and index > start:
                ratio = values[index] / values[index - 1]
                if ratio >= STEP_RATIO or ratio <= 1.0 / STEP_RATIO:
                    found.append(Segment(
                        start, index,
                        "rate stepped by %.1fx" % (ratio if ratio >= 1.0
                                                   else 1.0 / ratio)))
                    start = index
        return found

    def last_segment(self) -> Segment:
        """The run a forecast should start from, or None if there is none.

        The most recent one, not the longest. A well's current behaviour is
        what it is doing now, and the longest run may be from before a
        workover that changed the well.
        """
        found = self.segments()
        return found[-1] if found else None

    def shut_in_now(self) -> bool:
        live = self.producing()
        return bool(live.size) and not bool(live[-1])

    def rising(self, segment: Segment = None) -> bool:
        """Is this run going up? Arps cannot describe it, and will try."""
        segment = segment or self.last_segment()
        if segment is None or segment.length < 3:
            return False
        values = np.asarray(self.volumes, dtype=float)[segment.start:segment.stop]
        slope = np.polyfit(np.arange(values.size, dtype=float), values, 1)[0]
        return bool(slope > 0.0)


@dataclass
class SheetNote:
    """Something about the sheet the reader will not decide for you."""

    code: str
    detail: str


@dataclass
class ProductionSheet:
    """A wide monthly sheet, read but not interpreted."""

    wells: list = field(default_factory=list)
    periods: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    measure: str = "unknown"
    id_column: str = ""

    def well(self, name: str) -> WellHistory:
        for history in self.wells:
            if history.name == name:
                return history
        raise KeyError(name)

    def groups(self) -> dict:
        out = {}
        for history in self.wells:
            out.setdefault(history.group, []).append(history.name)
        return out

    def summary(self) -> str:
        first, last = self.periods[0], self.periods[-1]
        return ("%d wells, %d months %04d-%02d to %04d-%02d, values in an "
                "undeclared unit"
                % (len(self.wells), len(self.periods),
                   first[0], first[1], last[0], last[1]))


def read_wide_csv(path, id_column: str = "", group_column: str = "") -> ProductionSheet:
    """Read a well-per-row, month-per-column sheet.

    The period columns are found by parsing the header rather than by
    position, because the attribute columns sit on both sides of them in
    real exports and their count varies by operator.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError("%s has no data rows" % path)

    header = [cell.strip() for cell in rows[0]]
    parsed = [parse_period(cell) for cell in header]
    period_at = [index for index, value in enumerate(parsed) if value]
    if not period_at:
        raise ValueError(
            "no month columns found in the header of %s. Expected columns "
            "like 'Oct-22' or 'October 2022'; found %s"
            % (path, ", ".join(header[:8])))

    # The periods must be consecutive columns and in order. A sheet where
    # they are not has been edited by hand -- a month inserted in the wrong
    # place, or a duplicated column -- and reading it in header order would
    # silently produce a history that runs backwards over that join.
    notes = []
    if period_at != list(range(period_at[0], period_at[-1] + 1)):
        notes.append(SheetNote(
            "scattered_periods",
            "the month columns are not contiguous; attribute columns appear "
            "between them, so check the header before trusting the order"))
    periods = [parsed[index] for index in period_at]
    if periods != sorted(periods):
        notes.append(SheetNote(
            "unsorted_periods",
            "the month columns are not in chronological order and have been "
            "left in the sheet's own order; a decline fit on this would run "
            "over a join in time"))

    attribute_at = [index for index in range(len(header))
                    if index not in set(period_at) and header[index]]
    identifier = id_column or (header[attribute_at[0]] if attribute_at else "")
    group = group_column or (header[attribute_at[1]]
                             if len(attribute_at) > 1 else "")

    wells, blank = [], 0
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        cells = row + [""] * (len(header) - len(row))
        name = cells[header.index(identifier)].strip() if identifier else ""
        if not name:
            blank += 1
            continue
        attributes = {header[index]: cells[index].strip()
                      for index in attribute_at}
        if group:
            attributes["group"] = cells[header.index(group)].strip()
        wells.append(WellHistory(
            name=name, periods=periods,
            volumes=np.array([_number(cells[index]) for index in period_at],
                             dtype=float),
            attributes=attributes))

    if blank:
        notes.append(SheetNote(
            "unnamed_rows",
            "%d row(s) had no well identifier and were skipped" % blank))
    notes.extend(_measure_notes(wells))
    return ProductionSheet(wells=wells, periods=periods, notes=notes,
                           id_column=identifier)


def _number(cell: str) -> float:
    """A cell's value, with an empty cell distinguished from a zero.

    Empty means nobody recorded anything; zero means somebody recorded that
    the well made nothing. Collapsing the two is the error this file exists
    to avoid, so the blank becomes NaN and the zero stays a zero.
    """
    text = (cell or "").strip().replace(",", "")
    if not text or text in {"-", "--", "n/a", "N/A", "NA"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _measure_notes(wells: list) -> list:
    """Say what the numbers might be, and that the sheet does not say."""
    if not wells:
        return []
    values = np.concatenate([w.volumes[np.isfinite(w.volumes)] for w in wells]) \
        if any(np.isfinite(w.volumes).any() for w in wells) else np.zeros(0)
    live = values[values > 0.0]
    if live.size == 0:
        return [SheetNote("empty", "no positive production anywhere in the sheet")]

    median = float(np.median(live))
    return [SheetNote(
        "undeclared_units",
        "values have a median of %.4g per month and the sheet does not "
        "declare a unit. Nothing here converts them, and a decline fit is "
        "unaffected by the choice -- but a rate, a reserve or a revenue "
        "figure is not, so the unit has to be supplied before any of those "
        "are quoted." % median),
        SheetNote(
        "volume_or_rate",
        "whether a cell is the month's total volume or its average daily "
        "rate is not stated either, and the two differ by about 30x. "
        "`daily_rates` assumes a monthly volume; if the sheet is already "
        "rates, use the values as they stand.")]


def declinable(history: WellHistory) -> tuple:
    """May a decline curve be fitted to this well, and from which run.

    Returns (segment or None, reasons). The reasons are the point: a well
    that cannot be forecast is not an error, it is a fact about the well,
    and the engineer needs to know which fact.
    """
    from .decline import MIN_POINTS

    reasons = []
    segment = history.last_segment()
    if segment is None:
        return None, ["the well has no producing month in this sheet"]

    if segment.length < max(MIN_POINTS, MIN_SEGMENT):
        reasons.append(
            "the most recent producing run is %d month(s); %d are needed. "
            "%s" % (segment.length, max(MIN_POINTS, MIN_SEGMENT),
                    "The earlier run(s) are not usable either: a restart "
                    "resets the initial rate, so fitting across the gap "
                    "describes neither side."
                    if len(history.segments()) > 1
                    else "There is not enough history yet."))

    if history.shut_in_now():
        reasons.append(
            "the well is shut in at the end of the sheet, so any forecast "
            "is of what it would do if restarted, not of what it will make")

    if history.rising(segment):
        reasons.append(
            "production is rising over this run, which no Arps curve "
            "describes -- fitting it returns a negative decline and an "
            "unbounded reserve")

    usable = segment if segment.length >= max(MIN_POINTS, MIN_SEGMENT) \
        and not history.rising(segment) else None
    return usable, reasons


@dataclass
class WellOutlook:
    """One well's fit, or the reason there isn't one."""

    name: str
    group: str
    fit: object = None                  # decline.ArpsFit, or None
    reasons: list = field(default_factory=list)
    last_rate: float = 0.0              # per day, in the sheet's own unit
    # Years from this fit's own origin -- the start of the well's last
    # producing run -- to the end of the sheet. Wells are worked over at
    # different times, so their runs start at different dates, and their
    # fits do not share a clock. Summing `rate(0)` across the field would
    # add up each well's rate on its own start date, which is not a field
    # rate on any date. Measured on the sheet this was built against, that
    # error was +39% at t = 0.
    age_years: float = 0.0

    @property
    def fitted(self) -> bool:
        return self.fit is not None

    def rate_at(self, years: float) -> float:
        """Rate `years` after the end of the sheet."""
        if not self.fitted:
            return 0.0
        return float(self.fit.decline.rate(self.age_years + years))


@dataclass
class FieldOutlook:
    """Every well's outlook, and how much of the field that actually covers.

    The number that matters here is `covered_fraction`, and it exists
    because of the way a field forecast is usually wrong. Wells that cannot
    be fitted get dropped, the remaining curves are summed, and the total is
    presented as the field. It is lower than the field by exactly the
    production of the dropped wells, and nothing on the chart says so. On a
    sheet where a fifth of the wells are too freshly worked over to fit,
    that is a material understatement being reported as a forecast.
    """

    wells: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def fitted(self) -> list:
        return [w for w in self.wells if w.fitted]

    @property
    def blocked(self) -> list:
        return [w for w in self.wells if not w.fitted]

    @property
    def current_rate(self) -> float:
        return sum(w.last_rate for w in self.wells)

    @property
    def covered_rate(self) -> float:
        return sum(w.last_rate for w in self.fitted)

    @property
    def covered_fraction(self) -> float:
        total = self.current_rate
        return self.covered_rate / total if total > 0.0 else 0.0

    def rate_at(self, years: float) -> float:
        """Field rate `years` after the end of the sheet, fitted wells only.

        Always read with `covered_fraction`: this is a total over the wells
        that could be fitted, not over the field.
        """
        return sum(w.rate_at(years) for w in self.fitted)

    def summary(self) -> str:
        return ("%d of %d wells fitted, covering %.0f%% of the latest "
                "month's production; the remaining %d well(s) are reported "
                "separately and are not in any field total"
                % (len(self.fitted), len(self.wells),
                   self.covered_fraction * 100.0, len(self.blocked)))


def field_decline(sheet: ProductionSheet, **fit_kwargs) -> FieldOutlook:
    """Fit every well that can be fitted, and account for every well that cannot."""
    from .decline import fit_arps

    outlook = FieldOutlook(notes=[note.detail for note in sheet.notes])
    for history in sheet.wells:
        rates = history.daily_rates()
        live = history.producing()
        last = float(rates[-1]) if live.size and live[-1] else 0.0
        segment, reasons = declinable(history)
        entry = WellOutlook(name=history.name, group=history.group,
                            reasons=list(reasons), last_rate=last)
        if segment is not None:
            days, values = series_for_fit(history, segment)
            entry.fit = fit_arps(days, values, **fit_kwargs)
            elapsed = history.elapsed_days()
            entry.age_years = float(
                elapsed[segment.stop - 1] - elapsed[segment.start]) / 365.25
        outlook.wells.append(entry)
    return outlook


def audit_sheet(sheet: ProductionSheet, oil_price: float = 70.0,
                holdout: int = 3, **audit_kwargs):
    """Rank the wells that are off their own trend. PRD FR-5's daily question.

    `audit.audit_histories` already answers "which wells' reported numbers
    their own history cannot support". What it needs from a sheet is the
    *right* history, and that is the whole value of running it from here
    rather than over the raw columns.

    An audit fitted across a restart or a workover flags the workover. The
    rate genuinely jumped, the fit genuinely cannot explain it, and the
    report is correct and useless -- it has rediscovered the intervention
    the operator performed on purpose, and buried the two wells that are
    actually down among forty that are not. Feeding only the last producing
    run means a well is compared against the well it is now.

    The cost is honest and worth stating: a well worked over recently has a
    short run, so it gets held out of the audit rather than audited badly.
    `wells_skipped` carries those, and on a short sheet they are a large
    minority.

    The default holdout is 3 rather than the module's 6, because segments on
    a monthly sheet are short; holding out 6 of a 9-month run leaves 3 to
    fit, which cannot support a decline.
    """
    from . import audit as audit_module

    histories, skipped = [], []
    for history in sheet.wells:
        segment, reasons = declinable(history)
        if segment is None:
            skipped.append((history.name, reasons[0] if reasons else "unusable"))
            continue
        days, rates = series_for_fit(history, segment)
        histories.append(audit_module.WellHistory(
            name=history.name, days=list(days), rates=list(rates)))

    report = audit_module.audit_histories(
        histories, oil_price=oil_price, holdout=holdout, **audit_kwargs)
    report.wells_skipped = list(report.wells_skipped) + [
        "%s: %s" % (name, reason) for name, reason in skipped]
    return report


def series_for_fit(history: WellHistory, segment: Segment) -> tuple:
    """(days, rates) for `decline.fit_arps`, rebased to the run's own start."""
    elapsed = history.elapsed_days()[segment.start:segment.stop]
    rates = history.daily_rates()[segment.start:segment.stop]
    return elapsed - elapsed[0], rates
