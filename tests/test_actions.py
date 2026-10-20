"""Approving a recommendation, and measuring what it actually delivered.

The lifecycle tests are bookkeeping. The ones worth reading are the
measurement refusals, because this is the module where a vendor's numbers
get flattering without anybody lying: production moves for a dozen reasons
at once, and the convenient ones are easy to claim.

Each refusal below is a specific way a realised gain gets overstated --
unsubtracted decline, a second change in the same window, uptime that
moved, a window too short to mean anything -- and the test is that the
module declines rather than producing a number it cannot defend.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.actions import (
    APPROVED,
    EXECUTED,
    MEASURED,
    MIN_ACTIONS_FOR_A_RATIO,
    PROPOSED,
    REJECTED,
    Action,
    Scorecard,
    Window,
)
from phasespace.sim.decline import ArpsDecline


def action(action_id="A-1", well="W-1", gain=10.0):
    return Action(action_id=action_id, well=well,
                  change="raise lift gas to 1,750 mcf/d",
                  forecast_gain=gain, unit="stb/d",
                  proposed_by="gas_lift_allocate",
                  provenance_digest="deadbeef1234")


def executed(action_id="A-1", well="W-1", gain=10.0, by="S. Tripathi"):
    return action(action_id, well, gain).approve(by=by).execute()


# ------------------------------------------------------------ lifecycle --


def test_an_action_starts_as_a_proposal():
    assert action().state == PROPOSED


def test_an_approval_needs_a_named_person():
    """An approval nobody signed cannot answer the only question an audit
    asks of one afterwards."""
    with pytest.raises(ValueError, match="named person"):
        action().approve(by="")


def test_approval_records_who_and_when():
    a = action().approve(by="S. Tripathi", at="2026-10-20T09:00:00Z",
                         note="agreed at morning meeting")
    assert a.state == APPROVED
    assert a.decided_by == "S. Tripathi"
    assert a.decision_note == "agreed at morning meeting"


def test_a_rejection_is_recorded_rather_than_discarded():
    """How often engineers reject the optimiser is a measure of the
    optimiser, so a rejection has to survive in the record."""
    a = action().reject(by="S. Tripathi", note="compressor down for repair")
    assert a.state == REJECTED
    assert "compressor" in a.decision_note


def test_an_edit_is_an_approval_of_something_different():
    """An engineer who halves a recommended injection has made a judgement,
    and the scorecard must not later credit the optimiser's original
    number for it."""
    a = action(gain=10.0).edit(by="S. Tripathi",
                               change="raise lift gas to 1,200 mcf/d only",
                               forecast_gain=6.0)
    assert a.state == APPROVED
    assert a.forecast_gain == 6.0
    assert "1,200" in a.change
    assert a.decision_note


def test_only_an_approved_action_can_be_executed():
    with pytest.raises(ValueError, match="only an approved action"):
        action().execute()


def test_an_action_cannot_be_decided_twice():
    a = action().approve(by="S. Tripathi")
    with pytest.raises(ValueError, match="already"):
        a.reject(by="Someone Else")


# -------------------------------------------------------- the windows --


def test_a_window_reports_rate_per_producing_day():
    """Per calendar day would fold uptime into the rate, so a well that was
    simply up more would read as a well producing better."""
    window = Window(days=30.0, total_volume=2400.0, days_on_production=24.0)
    assert window.rate == 100.0
    assert window.uptime == pytest.approx(0.8)


def test_a_window_will_not_invent_uptime():
    """Defaulting to fully up is the honest default only because it is
    stated; what it must never do is exceed the window."""
    with pytest.raises(ValueError, match="exceeds the window"):
        Window(days=30.0, total_volume=100.0, days_on_production=31.0)


# ------------------------------------------------------- measurement --


def test_a_clean_measurement_subtracts_the_wells_own_decline():
    """The headline behaviour. A well making 100 today makes less next
    month whatever anyone does, and crediting the optimiser with the
    decline it did not stop is the commonest way this number is
    overstated."""
    a = executed(gain=10.0)
    before = Window(days=30.0, total_volume=3000.0)      # 100/day
    after = Window(days=30.0, total_volume=3240.0)       # 108/day
    decline = ArpsDecline(qi=100.0, di_per_year=0.30, b=0.5)

    m = a.measure(before, after, decline=decline)

    assert m.usable
    assert m.baseline_rate == pytest.approx(100.0)
    # Carried forward 30 days, so the no-action expectation is below 100.
    assert m.expected_rate_without_action < 100.0
    # And the realised gain is therefore *larger* than the naive 8.
    assert m.realised_gain > 8.0
    assert a.state == MEASURED
    assert any("carried forward" in note for note in m.notes)


def test_without_a_decline_curve_the_bias_is_stated_rather_than_hidden():
    a = executed()
    m = a.measure(Window(days=30.0, total_volume=3000.0),
                  Window(days=30.0, total_volume=3240.0))
    assert m.usable
    assert m.realised_gain == pytest.approx(8.0)
    assert any("treated as flat" in note for note in m.notes)
    # The bias runs both ways and the note has to say which: on this
    # declining well the uncorrected 8.0 is an *under*-statement, since
    # the well would have fallen to 97.6 without the action.
    assert any("understates the gain" in note for note in m.notes)


def test_a_loss_is_counted_as_a_loss():
    """A well that produced less than its own decline predicted is a real
    outcome, and the point of measuring is that it appears."""
    a = executed(gain=10.0)
    m = a.measure(Window(days=30.0, total_volume=3000.0),
                  Window(days=30.0, total_volume=2700.0))
    assert m.usable
    assert m.realised_gain < 0.0
    assert m.realised_fraction < 0.0
    assert any("produced less" in note for note in m.notes)


def test_two_actions_on_one_well_cannot_be_separately_credited():
    """There is no split from one well's history that anyone could defend,
    so it refuses rather than halving the difference."""
    first = executed("A-1", "W-1")
    second = executed("A-2", "W-1")
    m = first.measure(Window(days=30.0, total_volume=3000.0),
                      Window(days=30.0, total_volume=3300.0),
                      other_actions_in_window=[second])
    assert not m.usable
    assert "A-2" in m.notes[0]
    assert "separately credited" in m.notes[0]


def test_an_action_on_a_different_well_does_not_block_it():
    first = executed("A-1", "W-1")
    elsewhere = executed("A-2", "W-2")
    m = first.measure(Window(days=30.0, total_volume=3000.0),
                      Window(days=30.0, total_volume=3300.0),
                      other_actions_in_window=[elsewhere])
    assert m.usable


def test_uptime_that_moved_makes_the_windows_incomparable():
    """A well that was simply up more is not a well producing better."""
    a = executed()
    m = a.measure(Window(days=30.0, total_volume=1500.0,
                         days_on_production=15.0),       # 50% uptime
                  Window(days=30.0, total_volume=3000.0,
                         days_on_production=30.0))       # 100% uptime
    assert not m.usable
    assert "uptime moved" in m.notes[0]


def test_a_window_too_short_is_refused():
    a = executed()
    m = a.measure(Window(days=3.0, total_volume=300.0),
                  Window(days=3.0, total_volume=330.0))
    assert not m.usable
    assert "too short" in m.notes[0]


def test_an_unexecuted_action_has_delivered_nothing_by_definition():
    a = action().approve(by="S. Tripathi")
    m = a.measure(Window(days=30.0, total_volume=3000.0),
                  Window(days=30.0, total_volume=3300.0))
    assert not m.usable
    assert "only an executed action" in m.notes[0]


def test_every_measurement_carries_the_causation_caveat():
    """A number that travels without its caveat gets quoted as proof."""
    a = executed()
    m = a.measure(Window(days=30.0, total_volume=3000.0),
                  Window(days=30.0, total_volume=3300.0))
    assert "association, not causation" in m.basis
    assert "control" in m.basis


# -------------------------------------------------------- the scorecard --


def measured_action(action_id, gain, after_volume, well=None):
    a = executed(action_id, well or action_id, gain)
    a.measure(Window(days=30.0, total_volume=3000.0),
              Window(days=30.0, total_volume=after_volume))
    return a


def test_a_scorecard_will_not_report_a_ratio_from_too_few_actions():
    """A ratio from two actions is a story about two wells, and it will be
    quoted as a track record."""
    card = Scorecard([measured_action("A-1", 10.0, 3240.0),
                      measured_action("A-2", 10.0, 3240.0)])
    assert card.realised_fraction() is None
    assert "fewer than %d" % MIN_ACTIONS_FOR_A_RATIO in card.describe()


def test_a_scorecard_reports_the_field_ratio_once_there_is_enough():
    card = Scorecard([measured_action("A-%d" % i, 10.0, 3240.0)
                      for i in range(MIN_ACTIONS_FOR_A_RATIO)])
    fraction = card.realised_fraction()
    assert fraction == pytest.approx(0.8)      # 8 realised against 10 forecast
    assert "80% of forecast" in card.describe()


def test_losses_are_in_the_total_rather_than_excluded():
    """The arithmetic that makes this unflattering: a loss pulls the field
    ratio down instead of being dropped as an outlier."""
    good = [measured_action("A-%d" % i, 10.0, 3240.0) for i in range(4)]
    bad = measured_action("A-bad", 10.0, 2700.0)
    card = Scorecard(good + [bad])

    assert card.losses == 1
    assert card.total_realised == pytest.approx(4 * 8.0 - 10.0)
    assert card.realised_fraction() < 0.5
    assert "1 lost production" in card.describe()


def test_refusals_are_visible_rather_than_dropped():
    """An unmeasurable action that vanished from the count would flatter
    the ratio by selection."""
    first = executed("A-1", "W-1")
    second = executed("A-2", "W-1")
    first.measure(Window(days=30.0, total_volume=3000.0),
                  Window(days=30.0, total_volume=3300.0),
                  other_actions_in_window=[second])
    card = Scorecard([first, second])
    assert "1 not measurable" in card.describe()


def test_the_approval_rate_is_reported():
    """How often engineers reject the optimiser is a measure of the
    optimiser."""
    card = Scorecard([
        action("A-1").approve(by="S. Tripathi"),
        action("A-2").approve(by="S. Tripathi"),
        action("A-3").reject(by="S. Tripathi", note="compressor down"),
    ])
    assert card.approved == 2
    assert card.rejected == 1
    assert card.approval_rate == pytest.approx(2 / 3)


def test_the_scorecard_repeats_the_causation_caveat():
    card = Scorecard([measured_action("A-%d" % i, 10.0, 3240.0)
                      for i in range(MIN_ACTIONS_FOR_A_RATIO)])
    assert "association rather than causation" in card.describe()


def test_an_action_serialises_with_its_decision_and_measurement():
    """So it can go in the ledger beside the calculation that proposed
    it."""
    a = measured_action("A-1", 10.0, 3240.0)
    record = a.to_dict()
    assert record["state"] == MEASURED
    assert record["decided_by"] == "S. Tripathi"
    assert record["provenance_digest"] == "deadbeef1234"
    assert record["measurement"]["realised_fraction"] == pytest.approx(0.8)
    assert "causation" in record["measurement"]["basis"]


# ------------------------------------------------ recorded in the ledger --


def test_an_action_lands_in_the_ledger_beside_its_calculation(tmp_path):
    """The question asked of an audit record afterwards is usually "who
    agreed to this, and on what basis". That is answerable only if the
    approval and the calculation that justified it are in the same file."""
    from phasespace.sim.ledger import Ledger

    led = Ledger(tmp_path / "evidence.jsonl")
    a = measured_action("A-1", 10.0, 3240.0)
    led.append(a.to_dict())

    assert led.verify().ok
    record = next(led.records())
    assert record.calculation == "action"
    assert record.payload["decided_by"] == "S. Tripathi"
    assert record.payload["provenance_digest"] == "deadbeef1234"
    assert record.payload["measurement"]["realised_fraction"] == \
        pytest.approx(0.8)


def test_a_rejection_is_recorded_too(tmp_path):
    """An optimiser's rejection rate is a measure of the optimiser, and it
    only exists if the rejections were kept."""
    from phasespace.sim.ledger import Ledger

    led = Ledger(tmp_path / "evidence.jsonl")
    led.append(action("A-1").reject(by="S. Tripathi",
                                    note="compressor down").to_dict())
    record = next(led.records())
    assert record.payload["state"] == REJECTED
    assert "compressor" in record.payload["decision_note"]
