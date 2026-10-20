"""Recommendations, who approved them, and what they actually delivered.

Three things, and the third is the one almost nobody ships.

  propose    an optimiser says what to change and what it expects to gain
  approve    a named engineer accepts, edits or rejects it, before anything
             touches the field
  measure    afterwards, the realised gain is compared with the forecast

The first two are bookkeeping. The third is the product's whole claim to
be worth paying for, and it is also the easiest number in this industry to
overstate without lying, because production changes for a dozen reasons at
once and the convenient ones are easy to attribute to yourself.

**How a realised gain goes wrong, and what is done about each.**

*Decline.* A well making 100 bbl/d today makes less next month whatever
anyone does, so the question is never "did it go up?" but "did it beat
where it was heading?". Comparing a raw after-average against a raw
before-average measures the wrong thing, and the bias runs in both
directions: on a declining well it *understates* the gain, because the
well would have fallen anyway and the action is not credited for the
fall it prevented; on a rising well it overstates. Measured on a well
declining 30%/yr, a month-on-month rise of 8.0 bbl/d is a gain of 10.4
against where the well was actually heading.

So the baseline is carried forward along the well's own fitted decline
to the midpoint of the after-window, and the gain is measured against
that moved target rather than a frozen one.

*Confounding.* If two actions touched the same well in the same window,
neither can be separately credited, and this refuses to try. If a workover
happened, the number is about the workover. There is no statistical repair
for this from one well's history, so the honest output is a refusal naming
the other action rather than a split nobody can defend.

*Downtime.* A well down for six days of a thirty-day window has an average
that is about uptime, not about the change. Days on production are
required rather than inferred, and a window whose uptime moved materially
between before and after is reported as not comparable.

*Causation.* Even with all of that, this measures association. The well
did better after the change; it is not proven that it did better *because*
of the change, and only a control -- a comparable well left alone -- would
support that word. `Measurement.basis` says so on every result, and the
scorecard repeats it, because a number that travels without its caveat
eventually gets quoted as proof.

**The scorecard is deliberately unflattering.** It reports realised over
forecast across every measured action including the ones that lost
production, counts the refusals separately so they cannot be quietly
dropped, and never reports a ratio from fewer than a handful of actions.
A vendor's deck shows the wins; the point of this is to be the thing a
customer can audit at renewal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# A pilot's value is usually quoted per month, and a window shorter than
# this is mostly noise on a well whose rate wanders day to day.
MIN_WINDOW_DAYS = 7

# Below this many measured actions a ratio is a story about one or two
# wells, so `Scorecard.realised_fraction` declines to produce one.
MIN_ACTIONS_FOR_A_RATIO = 5

# If uptime moves by more than this between the before and after windows,
# the averages are comparing different things.
MAX_UPTIME_DRIFT = 0.15

PROPOSED = "proposed"
APPROVED = "approved"
REJECTED = "rejected"
EXECUTED = "executed"
MEASURED = "measured"

STATES = (PROPOSED, APPROVED, REJECTED, EXECUTED, MEASURED)


@dataclass
class Window:
    """What a well did over a period, as recorded rather than inferred."""

    days: float
    total_volume: float
    days_on_production: float = 0.0

    def __post_init__(self):
        if self.days <= 0.0:
            raise ValueError("a window must cover a positive number of days")
        if self.days_on_production <= 0.0:
            # Defaulting to "fully up" would silently invent uptime, and
            # uptime is exactly what distinguishes a real gain from a well
            # that simply stopped being shut in.
            self.days_on_production = self.days
        if self.days_on_production > self.days:
            raise ValueError(
                "days on production (%g) exceeds the window (%g)"
                % (self.days_on_production, self.days))

    @property
    def uptime(self) -> float:
        return self.days_on_production / self.days

    @property
    def rate(self) -> float:
        """Volume per producing day, not per calendar day.

        Per calendar day would fold uptime into the rate, so a well that
        was simply up more would read as a well that was producing better.
        """
        return self.total_volume / self.days_on_production


@dataclass
class Measurement:
    """What an action delivered, and how much of that is believable."""

    forecast_gain: float
    realised_gain: float
    baseline_rate: float
    expected_rate_without_action: float
    actual_rate: float
    usable: bool = True
    notes: list = field(default_factory=list)
    basis: str = ("measured against the baseline carried forward along its "
                  "own decline; association, not causation -- only a "
                  "control well would support the word 'because'")

    @property
    def realised_fraction(self) -> float:
        """Realised over forecast. Negative when the action lost ground."""
        if self.forecast_gain == 0.0:
            return 0.0
        return self.realised_gain / self.forecast_gain

    def describe(self) -> str:
        if not self.usable:
            return "; ".join(self.notes) or "not measurable"
        return ("forecast %+.1f, realised %+.1f (%.0f%% of forecast). "
                "Baseline %.1f, expected %.1f without the action, actual "
                "%.1f. %s"
                % (self.forecast_gain, self.realised_gain,
                   self.realised_fraction * 100.0, self.baseline_rate,
                   self.expected_rate_without_action, self.actual_rate,
                   self.basis))


@dataclass
class Action:
    """One recommended change, through its whole life."""

    action_id: str
    well: str
    change: str                      # "raise lift gas to 1,750 mcf/d"
    forecast_gain: float             # in the units the optimiser worked in
    unit: str = "stb/d"
    proposed_by: str = ""            # which calculation proposed it
    provenance_digest: str = ""      # the bundle that justifies it
    proposed_at: str = ""
    state: str = PROPOSED
    decided_by: str = ""             # a named engineer, not a role
    decided_at: str = ""
    decision_note: str = ""
    executed_at: str = ""
    measurement: Measurement = None

    def __post_init__(self):
        if not self.action_id:
            raise ValueError("an action needs an id")
        if not self.well:
            raise ValueError("an action needs a well")

    # ------------------------------------------------------- decisions --

    def approve(self, by: str, at: str = "", note: str = "") -> "Action":
        """Accept it. `by` is a person, and it is required.

        A system that records "approved" without recording who approved it
        has an audit trail that cannot answer the only question anyone asks
        of one afterwards.
        """
        return self._decide(APPROVED, by, at, note)

    def reject(self, by: str, at: str = "", note: str = "") -> "Action":
        return self._decide(REJECTED, by, at, note)

    def edit(self, by: str, change: str, forecast_gain: float,
             at: str = "", note: str = "") -> "Action":
        """Accept a modified version.

        Recorded as an approval of something different, with both the new
        change and the note, because an engineer who halves a recommended
        injection has made a judgement the scorecard must not later
        attribute to the optimiser's original number.
        """
        self.change = change
        self.forecast_gain = float(forecast_gain)
        return self._decide(APPROVED, by, at,
                            note or "edited before approval")

    def _decide(self, state, by, at, note) -> "Action":
        if not by:
            raise ValueError(
                "a decision needs a named person: an approval nobody signed "
                "cannot answer the question an audit asks of it")
        if self.state not in (PROPOSED,):
            raise ValueError("%s is already %s" % (self.action_id, self.state))
        self.state = state
        self.decided_by = by
        self.decided_at = at
        self.decision_note = note
        return self

    def execute(self, at: str = "") -> "Action":
        if self.state != APPROVED:
            raise ValueError(
                "%s is %s; only an approved action may be executed"
                % (self.action_id, self.state))
        self.state = EXECUTED
        self.executed_at = at
        return self

    # ----------------------------------------------------- measurement --

    def measure(self, before: Window, after: Window, decline=None,
                other_actions_in_window: list = None) -> Measurement:
        """Compare what happened with what was forecast.

        `decline` is the well's own fitted decline, used to carry the
        baseline forward rather than freeze it. Without one the comparison
        assumes a flat well and says so, which overstates the gain on a
        declining well and understates it on a rising one.
        """
        notes = []
        others = [a for a in (other_actions_in_window or [])
                  if a is not self and a.well == self.well]

        if self.state not in (EXECUTED, MEASURED):
            return self._unmeasurable(
                "%s is %s; only an executed action can have delivered "
                "anything" % (self.action_id, self.state))

        if others:
            return self._unmeasurable(
                "%s also changed %s in this window, so neither can be "
                "separately credited. There is no way to split the "
                "difference from one well's history that anyone could "
                "defend." % (", ".join(a.action_id for a in others), self.well))

        if before.days < MIN_WINDOW_DAYS or after.days < MIN_WINDOW_DAYS:
            return self._unmeasurable(
                "windows of %.0f and %.0f days are too short to separate a "
                "change from the noise a well makes day to day; %d days is "
                "the minimum here"
                % (before.days, after.days, MIN_WINDOW_DAYS))

        drift = abs(after.uptime - before.uptime)
        if drift > MAX_UPTIME_DRIFT:
            return self._unmeasurable(
                "uptime moved from %.0f%% to %.0f%%, so the two averages "
                "are not comparing the same thing. A well that was simply "
                "up more is not a well that was producing better."
                % (before.uptime * 100.0, after.uptime * 100.0))

        baseline = before.rate
        if decline is not None:
            # Carry the baseline to the midpoint of the after-window, which
            # is where its average sits in time.
            midpoint_years = (before.days / 2.0 + after.days / 2.0) / 365.25
            expected = baseline * _decline_ratio(decline, midpoint_years)
            notes.append(
                "the baseline was carried forward %.0f days along the "
                "well's own decline, from %.1f to %.1f, so the action is "
                "not credited with production nobody could have kept"
                % (before.days / 2.0 + after.days / 2.0, baseline, expected))
        else:
            expected = baseline
            notes.append(
                "no decline curve was supplied, so the baseline is treated "
                "as flat. On a declining well that understates the gain, "
                "because the well would have fallen anyway; on a rising "
                "one it overstates. Supply the fitted decline to remove "
                "the bias."
                "fitted decline to remove the bias.")

        realised = after.rate - expected
        measurement = Measurement(
            forecast_gain=self.forecast_gain,
            realised_gain=realised,
            baseline_rate=baseline,
            expected_rate_without_action=expected,
            actual_rate=after.rate,
            notes=notes)

        if self.forecast_gain > 0.0 and realised < 0.0:
            measurement.notes.append(
                "the well produced less than its own decline predicted. "
                "That is a real outcome and is counted as one.")

        self.measurement = measurement
        self.state = MEASURED
        return measurement

    def _unmeasurable(self, reason: str) -> Measurement:
        measurement = Measurement(
            forecast_gain=self.forecast_gain, realised_gain=0.0,
            baseline_rate=0.0, expected_rate_without_action=0.0,
            actual_rate=0.0, usable=False, notes=[reason])
        self.measurement = measurement
        return measurement

    def to_dict(self) -> dict:
        """Ledger-shaped, so an approval lands beside the calculation that
        justified it.

        `calculation` is what `Ledger.summary` groups by, and an approval
        is as much a thing to audit as a pressure: the question asked of a
        record afterwards is usually "who agreed to this, and on what
        basis", which is answerable only if both are in the same file.
        """
        record = {
            "calculation": "action",
            "action_id": self.action_id, "well": self.well,
            "change": self.change, "forecast_gain": self.forecast_gain,
            "unit": self.unit, "proposed_by": self.proposed_by,
            "provenance_digest": self.provenance_digest,
            "proposed_at": self.proposed_at, "state": self.state,
            "decided_by": self.decided_by, "decided_at": self.decided_at,
            "decision_note": self.decision_note,
            "executed_at": self.executed_at,
        }
        if self.measurement is not None:
            record["measurement"] = {
                "forecast_gain": self.measurement.forecast_gain,
                "realised_gain": self.measurement.realised_gain,
                "realised_fraction": self.measurement.realised_fraction,
                "usable": self.measurement.usable,
                "notes": list(self.measurement.notes),
                "basis": self.measurement.basis,
            }
        return record


def _decline_ratio(decline, years: float) -> float:
    """How much of today's rate survives `years` of this decline."""
    now = decline.rate(0.0)
    if now <= 0.0:
        return 1.0
    return decline.rate(years) / now


@dataclass
class Scorecard:
    """What the recommendations have been worth, across a field.

    Deliberately unflattering: losses count, refusals are visible rather
    than dropped, and a ratio from two actions is not reported at all.
    """

    actions: list = field(default_factory=list)

    def _measured(self) -> list:
        return [a for a in self.actions
                if a.measurement is not None and a.measurement.usable]

    def _refused(self) -> list:
        return [a for a in self.actions
                if a.measurement is not None and not a.measurement.usable]

    @property
    def proposed(self) -> int:
        return len(self.actions)

    @property
    def approved(self) -> int:
        return sum(1 for a in self.actions
                   if a.state in (APPROVED, EXECUTED, MEASURED))

    @property
    def rejected(self) -> int:
        return sum(1 for a in self.actions if a.state == REJECTED)

    @property
    def approval_rate(self) -> float:
        decided = self.approved + self.rejected
        return self.approved / decided if decided else 0.0

    @property
    def total_forecast(self) -> float:
        return sum(a.measurement.forecast_gain for a in self._measured())

    @property
    def total_realised(self) -> float:
        return sum(a.measurement.realised_gain for a in self._measured())

    @property
    def losses(self) -> int:
        return sum(1 for a in self._measured()
                   if a.measurement.realised_gain < 0.0)

    def realised_fraction(self):
        """Realised over forecast across the field, or None if too few.

        Returning None rather than a number is the point: a ratio from two
        actions is a story about two wells, and it will be quoted as a
        track record.
        """
        measured = self._measured()
        if len(measured) < MIN_ACTIONS_FOR_A_RATIO:
            return None
        if self.total_forecast == 0.0:
            return None
        return self.total_realised / self.total_forecast

    def describe(self) -> str:
        measured = self._measured()
        refused = self._refused()
        lines = [
            "%d proposed, %d approved, %d rejected (%.0f%% approved)"
            % (self.proposed, self.approved, self.rejected,
               self.approval_rate * 100.0),
            "%d measured, %d not measurable" % (len(measured), len(refused)),
        ]
        fraction = self.realised_fraction()
        if fraction is None:
            lines.append(
                "no field ratio: %d measured actions, and fewer than %d is "
                "a story about individual wells rather than a track record"
                % (len(measured), MIN_ACTIONS_FOR_A_RATIO))
        else:
            lines.append(
                "forecast %+.1f, realised %+.1f -- %.0f%% of forecast, "
                "across %d actions of which %d lost production"
                % (self.total_forecast, self.total_realised,
                   fraction * 100.0, len(measured), self.losses))
        if refused:
            lines.append(
                "not measurable: "
                + "; ".join("%s (%s)" % (a.action_id,
                                         a.measurement.notes[0][:60])
                            for a in refused[:3]))
        lines.append(
            "All of this is association rather than causation: these wells "
            "did better after the changes, which is not the same as better "
            "because of them. Only a comparable well left alone would "
            "support the stronger word.")
        return "\n".join(lines)
