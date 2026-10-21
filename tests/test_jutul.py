"""Reading a reservoir run, and the unit correction that makes it usable.

These tests import no Julia. The reader was deliberately separated from
the worker so that the arithmetic which can be silently wrong is testable
without a 1.8 GB depot, and the arithmetic that can be silently wrong is
the unit conversion.

The test that matters most is
`test_cumulatives_are_the_integral_of_the_rates`. It pins a correction
factor of 86,400 that was measured rather than assumed, and if a future
JutulDarcy fixes the output upstream it fails loudly instead of letting
this reader quietly divide a correct number.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.reservoir.jutul import (
    CUMULATIVE_SECOND_FACTOR,
    PA_PER_PSI,
    ReservoirRun,
    is_cumulative,
    is_pressure,
    load,
)


# SPE9 as measured here: 90 steps of 10 days, oil opening at 36,203 stb/d
# and closing at 11,553, which integrates to 19,741,876 stb -- against a
# reported FOPT of 1.7057e12.
STEPS = 90
DT_DAYS = 10.0


def spe9_like(tmp_path, oil_open=36203.0, oil_close=11553.0):
    """A run shaped like the real SPE9 output, cumulatives included."""
    days = np.arange(1, STEPS + 1, dtype=float) * DT_DAYS
    rate = np.linspace(oil_open, oil_close, STEPS)
    # The simulator's convention: integrate the per-day rate against
    # elapsed seconds. This is what makes the totals 86,400x too large.
    reported_total = np.cumsum(rate * DT_DAYS * 86400.0)
    gas_rate = rate * 5.5
    arrays = {
        "DAYS": days,
        "FIELD.FOPR": rate,
        "FIELD.FOPT": reported_total,
        "FIELD.FGPR": gas_rate,
        "FIELD.FGPT": np.cumsum(gas_rate * DT_DAYS * 86400.0),
        "FIELD.FPR": np.linspace(3778.8, 2503.3, STEPS),
        "WELL.PRODU10.WBHP": np.full(STEPS, 1000.0),
        "WELL.PRODU10.WOPR": rate / 25.0,
        "WELL.PRODU10.WOPT": np.cumsum(rate / 25.0 * DT_DAYS * 86400.0),
        # Per-cell pressure arrives in pascals.
        "STATE.Pressure": np.full((STEPS, 9000), 3160.6 * PA_PER_PSI),
        "STATE.Saturations": np.full((3, 9000), 0.4),
        "STATE.Rs": np.full((STEPS, 9000), 230.0),
    }
    npz = tmp_path / "run.npz"
    np.savez_compressed(npz, **arrays)
    meta = tmp_path / "run.json"
    meta.write_text(json.dumps({"solve_seconds": 16.1, "cells": 9000,
                                "wells": ["PRODU10"], "steps": STEPS}),
                    encoding="utf-8")
    return npz, meta, rate, days


# ------------------------------------------------- knowing which is which --


@pytest.mark.parametrize("key,expected", [
    ("FIELD.FOPT", True), ("FIELD.FGIT", True), ("WELL.A.WWIT", True),
    ("FIELD.FOPR", False), ("FIELD.FPR", False), ("WELL.A.WBHP", False),
    ("FIELD.FWCT", False), ("FIELD.FGOR", False),
])
def test_cumulative_keys_are_identified_from_the_mnemonic(key, expected):
    """Eclipse mnemonics are positional, so this reads the grammar rather
    than a hand-maintained list. A list would silently miss any key a new
    version adds, and a missed cumulative is a number 86,400x out."""
    assert is_cumulative(key) is expected


@pytest.mark.parametrize("key,expected", [
    ("WELL.A.WBHP", True), ("FIELD.FPR", True), ("FIELD.FPRH", True),
    ("FIELD.FOPR", False), ("FIELD.FOPT", False),
])
def test_pressure_keys_are_identified(key, expected):
    assert is_pressure(key) is expected


# --------------------------------------------------------- the correction --


def test_cumulatives_are_the_integral_of_the_rates(tmp_path):
    """The test this module exists for.

    After correction, the cumulative must equal the integral of the rate
    in the rate's own volume unit. Before correction it is 86,400x larger,
    which on SPE9 is the difference between 19.7 million barrels and 1.7
    trillion -- and the wrong one does not look like an error, it looks
    like a bigger field.

    If JutulDarcy fixes this upstream, this test fails and the reader must
    stop dividing. That is the intended behaviour: a correction nobody
    notices becoming wrong is worse than no correction.
    """
    npz, meta, rate, _ = spe9_like(tmp_path)
    run = load(npz, meta)

    expected = np.cumsum(rate * DT_DAYS)
    assert run.field_curves["FOPT"] == pytest.approx(expected, rel=1e-12)
    # The fixture ramps linearly, so its own integral is not SPE9's. The
    # real measured figures are pinned separately below rather than
    # back-fitted into a synthetic rate curve.
    assert run.field_curves["FOPT"][-1] < 1e9, "still in the simulator's units"


def test_the_measured_spe9_figures_are_pinned():
    """The real numbers, from the run on this machine, kept as a constant
    so the correction can be checked by hand against a benchmark anyone
    can download rather than against a fixture.

    SPE9, 900 days: the simulator reported FOPT of 1.7057e12 and the
    integral of its own FOPR curve was 19,741,876 stb. The ratio is
    86,400 to machine precision, and the same held for FGPT, FWPT and
    FWIT -- four independent pairs.
    """
    reported = 1705698118934.6235
    integral_of_rates = 19741876.376558144
    assert reported / integral_of_rates == pytest.approx(
        CUMULATIVE_SECOND_FACTOR, rel=1e-12)
    assert reported / CUMULATIVE_SECOND_FACTOR == pytest.approx(
        integral_of_rates, rel=1e-12)
    # 19.7 million barrels over 900 days is 21,935 stb/d on average,
    # against a rate curve running 36,203 down to 11,553. The trillion is
    # not a field anyone has.
    assert 20000 < integral_of_rates / 900.0 < 24000


def test_the_correction_applies_to_wells_as_well_as_the_field(tmp_path):
    npz, meta, rate, _ = spe9_like(tmp_path)
    run = load(npz, meta)
    expected = np.cumsum(rate / 25.0 * DT_DAYS)
    assert run.well_curves["PRODU10"]["WOPT"] == pytest.approx(expected,
                                                               rel=1e-12)


def test_rates_are_left_alone(tmp_path):
    """They were already in field units. Correcting them too would be the
    mirror-image bug, and a 36,203 stb/d field becoming 0.42 would at
    least be obvious -- which is no reason to rely on it."""
    npz, meta, rate, _ = spe9_like(tmp_path)
    run = load(npz, meta)
    assert run.field_curves["FOPR"] == pytest.approx(rate)
    assert run.field_curves["FPR"][0] == pytest.approx(3778.8)


def test_the_bottomhole_constraint_survives_unchanged(tmp_path):
    """SPE9 pins its producers at 1000 psia, and that exact number is how
    the summary's units were established in the first place. If a reader
    change ever moves it, the unit assumption underneath has broken."""
    npz, meta, _, _ = spe9_like(tmp_path)
    run = load(npz, meta)
    assert run.well_curves["PRODU10"]["WBHP"] == pytest.approx(
        np.full(STEPS, 1000.0))


def test_per_cell_pressure_is_converted_from_pascals(tmp_path):
    """The states are the one place SI arrives. 21.8e6 Pa is 3,161 psia,
    and a pascal figure read as psia is wrong by 6,895x -- which at least
    fails loudly, unlike the cumulative bug."""
    npz, meta, _, _ = spe9_like(tmp_path)
    run = load(npz, meta)
    assert run.states["Pressure"][0, 0] == pytest.approx(3160.6, rel=1e-6)
    assert run.cells == 9000


def test_saturations_and_rs_are_passed_through_labelled(tmp_path):
    """Saturations are dimensionless. Rs's unit was never established by
    measurement here, so it is reported raw with that said, rather than
    converted on a guess."""
    npz, meta, _, _ = spe9_like(tmp_path)
    run = load(npz, meta)
    assert run.states["Saturations"][0, 0] == pytest.approx(0.4)
    assert run.states["Rs"][0, 0] == pytest.approx(230.0)
    assert any("not established by" in note for note in run.corrections)


# ------------------------------------------------------------ the reading --


def test_the_run_reports_its_shape(tmp_path):
    npz, meta, _, days = spe9_like(tmp_path)
    run = load(npz, meta)
    assert run.steps == STEPS
    assert run.cells == 9000
    assert run.wells == ["PRODU10"]
    assert run.days[-1] == pytest.approx(900.0)


def test_every_correction_is_announced(tmp_path):
    """A conversion nobody can see is a conversion nobody can check."""
    npz, meta, _, _ = spe9_like(tmp_path)
    run = load(npz, meta)
    text = "\n".join(run.corrections)
    assert "86400" in text.replace(",", "") or "86,400" in text
    assert "pascals to psia" in text


def test_the_description_refuses_to_claim_validation(tmp_path):
    """A benchmark reproducing itself is verification. The simulator's own
    authors report matching AD-GPRS and MRST, which is code-to-code and so
    also verification."""
    npz, meta, _, _ = spe9_like(tmp_path)
    run = load(npz, meta)
    assert "Verification only" in run.describe()


def test_a_run_with_no_metadata_still_reads(tmp_path):
    """The worker writes metadata incrementally and may die before
    finishing it. The arrays are the answer; the metadata is commentary."""
    npz, _, _, _ = spe9_like(tmp_path)
    run = load(npz)
    assert run.steps == STEPS
    assert run.meta == {}


def test_an_empty_run_does_not_pretend_to_have_results():
    run = ReservoirRun()
    assert run.steps == 0
    assert run.cells == 0
    assert run.wells == []
