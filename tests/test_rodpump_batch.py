"""Tests for field-wide card analysis.

The batch is not a loop over the single-card path, and the tests that matter
are the ones about that difference: it must survive bad rows, it must rank
findings the way an engineer would act on them, and the production
reconciliation -- the reason anyone runs it -- must be signed and scaled
correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.rodpump.batch import (
    DISCREPANCY_THRESHOLD,
    WellInput,
    analyse_batch,
    assumed_fields,
    parse_long_csv,
    summarise,
)
from phasespace.rodpump.rodstring import taper_86
from phasespace.rodpump.synthetic import PumpSpec, surface_card_for


def card_for(fault, severity=0.7, spm=8.0):
    spec = PumpSpec(plunger_diameter_in=1.5, net_lift_ft=4000.0,
                    fluid_specific_gravity=0.9, stroke_in=100.0,
                    strokes_per_minute=spm)
    surface, _ = surface_card_for(fault, taper_86(4500.0), spec,
                                  severity=severity, damping_per_s=0.4)
    return surface


def well(name, fault, severity=0.7, reported=None):
    card = card_for(fault, severity)
    return WellInput(name=name, position_in=card.position_in,
                     load_lb=card.load_lb, reported_bbl_d=reported)


# ------------------------------------------------------------- the batch ---


def test_every_well_is_diagnosed():
    result = analyse_batch([well("P-1", "full_pump", 0.0),
                            well("P-2", "fluid_pound", 0.8),
                            well("P-3", "gas_interference", 0.8)])
    assert len(result.analysed) == 3
    assert not result.failures
    assert {w.fault for w in result.analysed} == {
        "full_pump", "fluid_pound", "gas_interference"}


def test_one_bad_card_does_not_cost_the_others():
    """A field export has a truncated trace in it somewhere. Aborting the run
    is how a free trial becomes nothing at all."""
    good = well("P-1", "fluid_pound", 0.8)
    broken = WellInput.__new__(WellInput)
    broken.__dict__.update(good.__dict__)
    broken.name = "P-BAD"
    broken.load_lb = np.full_like(good.load_lb, np.nan)

    result = analyse_batch([good, broken, well("P-3", "worn_pump", 0.8)])
    assert len(result.analysed) == 2
    assert len(result.failures) == 1
    assert result.failures[0][0] == "P-BAD"
    assert not result.wells[1].ok


def test_malformed_wells_are_rejected_at_construction():
    card = card_for("full_pump")
    with pytest.raises(ValueError, match="at least 8 samples"):
        WellInput(name="X", position_in=card.position_in[:4],
                  load_lb=card.load_lb[:4])
    with pytest.raises(ValueError, match="does not vary"):
        WellInput(name="X", position_in=np.zeros(20), load_lb=card.load_lb[:20])
    with pytest.raises(ValueError, match="unknown taper"):
        WellInput(name="X", position_in=card.position_in,
                  load_lb=card.load_lb, taper="99")


# ----------------------------------------------------- the reconciliation --


def test_an_overstated_well_is_flagged():
    """The finding the whole batch exists to produce. A pounding pump cannot be
    making its reported rate, and the card says so independently of whatever
    the allocation says."""
    result = analyse_batch([well("P-1", "fluid_pound", 0.9, reported=210.0)])
    row = result.analysed[0]
    assert row.flagged
    assert row.discrepancy_fraction > DISCREPANCY_THRESHOLD
    assert row.displacement_bbl_d < row.reported_bbl_d


def test_a_well_reported_correctly_is_not_flagged():
    """The false-positive case. A report that flags healthy wells gets
    discarded wholesale."""
    result = analyse_batch([well("P-1", "full_pump", 0.0, reported=210.0)])
    row = result.analysed[0]
    assert abs(row.discrepancy_fraction) < DISCREPANCY_THRESHOLD, (
        "%.0f reported vs %.0f from card" % (row.reported_bbl_d,
                                             row.displacement_bbl_d))
    assert not row.flagged


def test_discrepancy_is_signed():
    """A well producing more than it is credited with is an allocation problem,
    not an error to correct downward -- and it means some other well in the
    battery is being over-credited. Collapsing it to an absolute error hides
    that."""
    over = analyse_batch([well("P-1", "fluid_pound", 0.9, reported=210.0)]).analysed[0]
    under = analyse_batch([well("P-2", "full_pump", 0.0, reported=80.0)]).analysed[0]
    assert over.discrepancy_fraction > 0
    assert under.discrepancy_fraction < 0


def test_flagged_wells_are_ranked_by_barrels_not_percentage():
    """A 60% error on a 4 bbl/day stripper is arithmetic; a 25% error on a
    300 bbl/day well is the finding that pays for the study. Ranking by
    percentage puts the stripper first and buries it."""
    stripper = card_for("fluid_pound", 0.95, spm=2.2)
    big = card_for("fluid_pound", 0.5, spm=14.0)

    wells = [
        WellInput(name="STRIPPER", position_in=stripper.position_in,
                  load_lb=stripper.load_lb, spm=2.2, reported_bbl_d=40.0),
        WellInput(name="BIG", position_in=big.position_in,
                  load_lb=big.load_lb, spm=14.0, reported_bbl_d=340.0),
    ]
    result = analyse_batch(wells)
    assert [w.name for w in result.flagged][0] == "BIG", (
        [(w.name, round(w.barrels_at_stake), round(100 * w.discrepancy_fraction))
         for w in result.flagged])


def test_wells_without_a_reported_rate_are_never_flagged():
    """No reported rate means nothing to reconcile against. Treating a missing
    value as zero would flag every well on the lease."""
    result = analyse_batch([well("P-1", "fluid_pound", 0.9)])
    row = result.analysed[0]
    assert row.reported_bbl_d is None
    assert not row.flagged
    assert row.discrepancy_fraction == 0.0
    assert row.barrels_at_stake == 0.0


def test_headline_states_the_field_gap():
    wells = [well("P-%d" % i, "fluid_pound", 0.85, reported=200.0) for i in range(5)]
    headline = analyse_batch(wells).headline()
    assert "bbl/day" in headline
    assert "5 cards analysed" in headline
    assert "wrong wells" in headline


def test_a_clean_field_is_not_given_a_scary_headline():
    wells = [well("P-%d" % i, "full_pump", 0.0, reported=210.0) for i in range(4)]
    headline = analyse_batch(wells).headline()
    assert "No well disagrees" in headline


def test_gross_misallocation_is_reported_when_the_net_cancels():
    """The reporting trap. One well under-credited and two over-credited can
    net to nothing while the allocation is wrong by hundreds of barrels a day.
    Re-allocation is zero-sum within a battery, so near-cancellation is the
    expected signature of the problem rather than evidence against it."""
    wells = [well("UNDER", "full_pump", 0.0, reported=60.0),
             well("OVER-1", "fluid_pound", 0.85, reported=200.0),
             well("OVER-2", "gas_interference", 0.7, reported=190.0)]
    result = analyse_batch(wells)

    net = abs(result.total_reported_bbl_d - result.total_computed_bbl_d)
    assert result.gross_misallocated_bbl_d > 3.0 * net, (
        "gross %.0f vs net %.0f" % (result.gross_misallocated_bbl_d, net))
    assert "nets out almost entirely" in result.headline()


def test_headline_without_reported_rates_says_so():
    """Half the value of the batch needs reported rates. When they are absent
    the report must say what it could not do, rather than quietly omitting the
    reconciliation."""
    headline = analyse_batch([well("P-1", "fluid_pound", 0.8)]).headline()
    assert "no production reconciliation" in headline.lower()


def test_totals_only_count_wells_that_can_be_compared():
    """A well with no reported rate must not contribute to the computed total
    either, or the two sides of the comparison cover different wells and the
    field gap is meaningless."""
    result = analyse_batch([well("A", "fluid_pound", 0.8, reported=200.0),
                            well("B", "fluid_pound", 0.8)])
    assert result.total_reported_bbl_d == pytest.approx(200.0)
    assert result.total_computed_bbl_d == pytest.approx(
        result.analysed[0].displacement_bbl_d)


# ------------------------------------------------------------- the input ---


def test_long_csv_groups_rows_into_wells():
    a, b = card_for("fluid_pound", 0.8), card_for("worn_pump", 0.8)
    rows = []
    for name, card in (("P-1", a), ("P-2", b)):
        for position, load in zip(card.position_in, card.load_lb):
            rows.append({"well": name, "position_in": position, "load_lb": load,
                         "depth_ft": 5200, "plunger_in": 1.75,
                         "reported_bbl_d": 180})
    wells = parse_long_csv(rows)
    assert [w.name for w in wells] == ["P-1", "P-2"]
    assert wells[0].depth_ft == 5200.0
    assert wells[0].plunger_in == 1.75
    assert wells[0].reported_bbl_d == 180.0
    assert wells[0].position_in.size == a.position_in.size


def test_missing_metadata_falls_back_and_is_recorded():
    """Diagnosing against assumed geometry is often the only option, and is
    fine -- provided the report says which wells it happened to, because a
    fillage computed against a guessed plunger size is a guess."""
    card = card_for("fluid_pound", 0.8)
    rows = [{"well": "P-1", "position_in": p, "load_lb": l}
            for p, l in zip(card.position_in, card.load_lb)]
    wells = parse_long_csv(rows, defaults={"depth_ft": 3800.0, "plunger_in": 1.25})
    assert wells[0].depth_ft == 3800.0
    assert set(assumed_fields(wells[0])) == {"depth_ft", "plunger_in"}


def test_blank_metadata_cells_are_treated_as_missing():
    """A spreadsheet export is full of empty strings, not nulls."""
    card = card_for("full_pump", 0.0)
    rows = [{"well": "P-1", "position_in": p, "load_lb": l, "depth_ft": "",
             "reported_bbl_d": ""}
            for p, l in zip(card.position_in, card.load_lb)]
    wells = parse_long_csv(rows, defaults={"depth_ft": 4000.0})
    assert wells[0].depth_ft == 4000.0
    assert wells[0].reported_bbl_d is None


def test_bad_rows_name_the_well_and_the_row():
    """An error on row 4,113 of a 90,000-row export has to say which well and
    which row, or it cannot be acted on."""
    with pytest.raises(ValueError, match="row 2 of well P-1"):
        parse_long_csv([{"well": "P-1", "position_in": 1.0, "load_lb": 2.0},
                        {"well": "P-1", "position_in": 2.0, "load_lb": "n/a"}])
    with pytest.raises(ValueError, match="row 1 has no well name"):
        parse_long_csv([{"well": "", "position_in": 1.0, "load_lb": 2.0}])
    with pytest.raises(ValueError, match="no rows"):
        parse_long_csv([])


# ---------------------------------------------------------------- report ---


def test_summary_leads_with_the_gap_and_lists_the_worst():
    wells = [well("P-%d" % i, "fluid_pound", 0.85, reported=200.0) for i in range(3)]
    wells.append(well("P-OK", "full_pump", 0.0, reported=210.0))
    text = summarise(analyse_batch(wells))
    assert "cards analysed" in text
    assert "worst first by barrels" in text
    assert "Pump conditions across the field" in text


def test_summary_reports_failures():
    good = well("P-1", "fluid_pound", 0.8, reported=200.0)
    broken = WellInput.__new__(WellInput)
    broken.__dict__.update(good.__dict__)
    broken.name = "P-BAD"
    broken.load_lb = np.full_like(good.load_lb, np.inf)
    text = summarise(analyse_batch([good, broken]))
    assert "could not be analysed" in text
    assert "P-BAD" in text


def test_fault_counts_summarise_the_field():
    wells = ([well("A%d" % i, "fluid_pound", 0.85) for i in range(4)]
             + [well("B%d" % i, "full_pump", 0.0) for i in range(2)])
    counts = analyse_batch(wells).fault_counts
    assert counts["fluid_pound"] == 4
    assert list(counts)[0] == "fluid_pound", "counts should be ordered by frequency"
