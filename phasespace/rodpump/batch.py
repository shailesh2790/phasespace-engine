"""Diagnose a field's worth of dynamometer cards in one pass.

One card at a time is a demonstration. A field has four hundred rod-pumped
wells, and the question an asset manager actually has is not "what is wrong
with P-114" but "which of my wells are lying to me about their production".

That question has a specific answer here, and it is the reason this module
exists rather than being a loop over the single-card endpoint. Pump
displacement computed from the recovered downhole card is an independent
estimate of what a well is making. Set it against the rate the well is
*reported* at -- from allocation, from a well test, from whatever the operator
has -- and the disagreements are a ranked list of wells whose numbers do not
add up.

Two design decisions in that ranking matter more than the diagnosis itself:

**Rank by barrels, not by percentage.** A 60% overstatement on a 4 bbl/day
stripper is arithmetic. A 20% overstatement on a 300 bbl/day well is sixty
barrels a day, and it is the second one that pays for the study. Sorting by
percentage puts the stripper at the top of the report and buries the finding.

**Report the direction.** A well producing *more* than it is credited with is
not an error to be corrected downward -- it is usually an allocation problem,
and it means some other well in the same battery is being over-credited.
Signing the discrepancy keeps that visible instead of collapsing it into an
absolute error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import measure
from .diagnosis import diagnose
from .rodstring import RodSection, RodString
from .synthetic import FAULT_LABELS, PumpSpec
from .wave import Card, surface_to_downhole

# Above this the well's reported rate and its pump card disagree enough to be
# worth someone's afternoon. Conventional rather than derived: it is roughly
# where allocation error stops being explicable by meter tolerance.
DISCREPANCY_THRESHOLD = 0.20

TAPERS = {
    "86": (8, 6),
    "876": (8, 7, 6),
    "7": (7,),
    "8": (8,),
    "9": (9,),
    "987": (9, 8, 7),
}


@dataclass
class WellInput:
    """One well: its card, its pump, and what it is currently credited with."""

    name: str
    position_in: np.ndarray
    load_lb: np.ndarray
    depth_ft: float = 4500.0
    taper: str = "86"
    plunger_in: float = 1.5
    net_lift_ft: float = 4000.0
    fluid_sg: float = 0.9
    stroke_in: float = 100.0
    spm: float = 8.0
    damping: float = 0.4
    reported_bbl_d: float = None

    def __post_init__(self):
        self.position_in = np.asarray(self.position_in, dtype=float)
        self.load_lb = np.asarray(self.load_lb, dtype=float)
        if self.position_in.size != self.load_lb.size:
            raise ValueError(
                "%s: %d positions against %d loads"
                % (self.name, self.position_in.size, self.load_lb.size))
        if self.position_in.size < 8:
            raise ValueError("%s: a card needs at least 8 samples, got %d"
                             % (self.name, self.position_in.size))
        if not np.all(np.isfinite(self.position_in)) or not np.all(np.isfinite(self.load_lb)):
            raise ValueError("%s: card contains blank or non-finite values" % self.name)
        if float(self.position_in.max() - self.position_in.min()) <= 0.0:
            raise ValueError("%s: the position column does not vary" % self.name)
        if self.taper not in TAPERS:
            raise ValueError("%s: unknown taper %r (known: %s)"
                             % (self.name, self.taper, ", ".join(sorted(TAPERS))))

    def string(self) -> RodString:
        numbers = TAPERS[self.taper]
        share = self.depth_ft / len(numbers)
        return RodString([RodSection(api_number=n, length_ft=share) for n in numbers])

    def spec(self) -> PumpSpec:
        return PumpSpec(plunger_diameter_in=self.plunger_in,
                        net_lift_ft=self.net_lift_ft,
                        fluid_specific_gravity=self.fluid_sg,
                        stroke_in=self.stroke_in,
                        strokes_per_minute=self.spm)


@dataclass
class WellResult:
    name: str
    ok: bool
    fault: str = ""
    label: str = ""
    confidence: float = 0.0
    evidence: str = ""
    fillage: float = 0.0
    displacement_bbl_d: float = 0.0
    full_displacement_bbl_d: float = 0.0
    load_ratio: float = 0.0
    reported_bbl_d: float = None
    learned_fault: str = None
    learned_label: str = None
    learned_agrees: bool = None
    error: str = ""

    @property
    def discrepancy_fraction(self) -> float:
        """Signed: positive means the well is credited with more than the pump
        can be moving."""
        if self.reported_bbl_d is None or self.reported_bbl_d <= 0.0:
            return 0.0
        return (self.reported_bbl_d - self.displacement_bbl_d) / self.reported_bbl_d

    @property
    def barrels_at_stake(self) -> float:
        if self.reported_bbl_d is None:
            return 0.0
        return abs(self.reported_bbl_d - self.displacement_bbl_d)

    @property
    def flagged(self) -> bool:
        return (self.reported_bbl_d is not None
                and abs(self.discrepancy_fraction) > DISCREPANCY_THRESHOLD)


@dataclass
class BatchResult:
    wells: list = field(default_factory=list)
    failures: list = field(default_factory=list)

    @property
    def analysed(self) -> list:
        return [w for w in self.wells if w.ok]

    @property
    def flagged(self) -> list:
        """Wells whose reported rate and pump card disagree, worst first.

        Ordered by barrels rather than by percentage, because that is the order
        in which they are worth investigating.
        """
        return sorted((w for w in self.analysed if w.flagged),
                      key=lambda w: -w.barrels_at_stake)

    @property
    def total_reported_bbl_d(self) -> float:
        return sum(w.reported_bbl_d or 0.0 for w in self.analysed)

    @property
    def total_computed_bbl_d(self) -> float:
        return sum(w.displacement_bbl_d for w in self.analysed
                   if w.reported_bbl_d is not None)

    @property
    def gross_misallocated_bbl_d(self) -> float:
        """Barrels attributed to the wrong well, ignoring sign.

        The number that matters, and the one a net total hides. A field can
        have one well under-credited by 150 bbl/day and two over-credited by 75
        each, and net to zero: the allocation is wrong by 300 barrels a day and
        a net figure reports perfect agreement. Since re-allocation is a
        zero-sum exercise within a battery, near-cancellation is the *expected*
        signature of an allocation problem rather than evidence against one.
        """
        return sum(w.barrels_at_stake for w in self.analysed if w.flagged)

    @property
    def fault_counts(self) -> dict:
        counts = {}
        for well in self.analysed:
            counts[well.fault] = counts.get(well.fault, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def headline(self) -> str:
        """The sentence an asset manager reads first."""
        if not self.analysed:
            return "No cards could be analysed."
        if self.total_reported_bbl_d <= 0.0:
            underfilled = sum(1 for w in self.analysed if w.fillage < 0.92)
            return ("%d cards analysed. %d wells are not filling the pump. No "
                    "reported rates were supplied, so no production "
                    "reconciliation was possible."
                    % (len(self.analysed), underfilled))

        gap = self.total_reported_bbl_d - self.total_computed_bbl_d
        gross = self.gross_misallocated_bbl_d
        direction = "above" if gap > 0 else "below"

        if not self.flagged:
            return ("%d cards analysed. Reported production totals %.0f "
                    "bbl/day and the pump cards support %.0f. No well "
                    "disagrees by more than %.0f%%."
                    % (len(self.analysed), self.total_reported_bbl_d,
                       self.total_computed_bbl_d, 100 * DISCREPANCY_THRESHOLD))

        headline = (
            "%d cards analysed. %.0f bbl/day is attributed to the wrong wells "
            "across %d of them."
            % (len(self.analysed), gross, len(self.flagged)))

        # Say it explicitly when the net nearly cancels, because that is the
        # moment a reader is most likely to conclude there is no problem.
        if gross > 0.0 and abs(gap) < 0.5 * gross:
            return headline + (
                " The field total is only %.0f bbl/day %s, so this nets out "
                "almost entirely -- which is what an allocation error looks "
                "like, not evidence against one." % (abs(gap), direction))
        return headline + (
            " The field total is %.0f bbl/day %s what the pumps can be moving."
            % (abs(gap), direction))


def analyse_batch(wells: list, classifier=None) -> BatchResult:
    """Run every card through the full chain, and keep going past failures.

    A batch that aborts on the first malformed card is useless on real data:
    one well with a truncated trace should not cost the other three hundred
    diagnoses. Failures are collected per well and reported alongside the
    results rather than raised.
    """
    result = BatchResult()

    for well in wells:
        try:
            # Checked here and not only in WellInput, because this is the entry
            # point for bulk data from someone else's historian and the chain
            # below does not defend itself. A card of NaN loads propagates
            # through the FFT, makes every percentile comparison false, and
            # comes out the far end as a confident diagnosis with a plausible
            # fillage -- which is worse than a crash by a wide margin.
            if not np.all(np.isfinite(well.position_in)):
                raise ValueError("card positions contain non-finite values")
            if not np.all(np.isfinite(well.load_lb)):
                raise ValueError("card loads contain non-finite values")

            surface = Card(well.position_in, well.load_lb,
                           period_s=60.0 / well.spm)
            string, spec = well.string(), well.spec()
            downhole = surface_to_downhole(surface, string,
                                           damping_per_s=well.damping)
            diagnosis = diagnose(downhole, spec)
            measurements = measure(downhole, spec)

            row = WellResult(
                name=well.name, ok=True,
                fault=diagnosis.primary.fault,
                label=diagnosis.primary.label,
                confidence=float(diagnosis.primary.confidence),
                evidence=diagnosis.primary.evidence,
                fillage=float(measurements.fillage),
                displacement_bbl_d=float(measurements.displacement_bbl_per_day),
                full_displacement_bbl_d=float(spec.displacement_bbl_per_day(1.0)),
                load_ratio=float(diagnosis.load_ratio),
                reported_bbl_d=well.reported_bbl_d,
            )

            # And again on the way out: a finite card can still produce a
            # non-finite measurement through a degenerate stroke, and a NaN
            # displacement silently poisons the field total it is summed into.
            for label, value in (("fillage", row.fillage),
                                 ("displacement", row.displacement_bbl_d),
                                 ("load ratio", row.load_ratio)):
                if not np.isfinite(value):
                    raise ValueError("computed %s is not a finite number; the "
                                     "card is degenerate" % label)

            if classifier is not None:
                prediction = classifier.predict(downhole, spec)
                row.learned_fault = prediction.fault
                row.learned_label = prediction.label
                row.learned_agrees = prediction.fault == diagnosis.primary.fault

            result.wells.append(row)
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            result.wells.append(WellResult(name=well.name, ok=False, error=str(exc)))
            result.failures.append((well.name, str(exc)))

    return result


def parse_long_csv(rows: list, defaults: dict = None) -> list:
    """Build well inputs from one long-format table.

    Expected columns: ``well``, ``position_in``, ``load_lb``, plus any of the
    pump metadata columns, which are read once per well from its first row.
    Everything a spreadsheet export can produce, in one file -- asking an
    operator to supply a card archive plus a separate correctly-keyed metadata
    file is how a free trial becomes a two-week integration.

    Missing metadata falls back to ``defaults``, and every substitution is
    recorded so the report can say which wells were diagnosed against assumed
    geometry rather than real geometry.
    """
    defaults = dict(defaults or {})
    if not rows:
        raise ValueError("no rows supplied")

    numeric_fields = ("depth_ft", "plunger_in", "net_lift_ft", "fluid_sg",
                      "stroke_in", "spm", "damping", "reported_bbl_d")

    grouped = {}
    order = []
    for index, row in enumerate(rows):
        name = str(row.get("well", "")).strip()
        if not name:
            raise ValueError("row %d has no well name" % (index + 1))
        if name not in grouped:
            grouped[name] = {"position": [], "load": [], "meta": row}
            order.append(name)
        try:
            grouped[name]["position"].append(float(row["position_in"]))
            grouped[name]["load"].append(float(row["load_lb"]))
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "row %d of well %s: position_in and load_lb must both be "
                "numbers" % (index + 1, name))

    wells, assumed = [], {}
    for name in order:
        entry = grouped[name]
        meta = entry["meta"]
        kwargs = {}
        for key in numeric_fields:
            raw = meta.get(key)
            if raw is None or str(raw).strip() == "":
                if key in defaults:
                    kwargs[key] = defaults[key]
                    assumed.setdefault(name, []).append(key)
                continue
            kwargs[key] = float(raw)

        taper = str(meta.get("taper", "") or defaults.get("taper", "86")).strip()
        wells.append(WellInput(name=name,
                               position_in=entry["position"],
                               load_lb=entry["load"],
                               taper=taper, **kwargs))

    for well in wells:
        if well.name in assumed:
            well.__dict__["_assumed"] = assumed[well.name]
    return wells


def assumed_fields(well: WellInput) -> list:
    return list(well.__dict__.get("_assumed", []))


def summarise(result: BatchResult, limit: int = 25) -> str:
    """A plain-text report, for a CLI run or an email attachment."""
    lines = [result.headline(), ""]

    if result.flagged:
        lines.append("Wells where the card and the reported rate disagree, "
                     "worst first by barrels:")
        lines.append("%-14s %10s %10s %9s  %s"
                     % ("well", "reported", "from card", "gap", "diagnosis"))
        for well in result.flagged[:limit]:
            lines.append("%-14s %10.0f %10.0f %8.0f%%  %s"
                         % (well.name, well.reported_bbl_d or 0.0,
                            well.displacement_bbl_d,
                            100 * well.discrepancy_fraction,
                            FAULT_LABELS.get(well.fault, well.fault)))
        lines.append("")

    counts = result.fault_counts
    if counts:
        lines.append("Pump conditions across the field:")
        for fault, count in counts.items():
            lines.append("  %-36s %d" % (FAULT_LABELS.get(fault, fault), count))

    if result.failures:
        lines.append("")
        lines.append("%d card(s) could not be analysed:" % len(result.failures))
        for name, error in result.failures[:limit]:
            lines.append("  %-14s %s" % (name, error))

    return "\n".join(lines)
