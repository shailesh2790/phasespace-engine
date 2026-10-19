"""The Z-factor interpolation table: accuracy, range, and cache behaviour.

`z_at` is the hottest function in the engine by a wide margin. Profiling a
400-node network solve found `z_factor_dak` called **76,760 times**, taking
**53% of total runtime** -- the linear algebra did not appear in the top
fourteen functions. A single nodal analysis calls it 10,080 times. Every one
of those was an iterative Newton solve for a quantity that depends only on
pressure once the gas and temperature are fixed, which they are for the whole
of any one solve.

Tabulating it is therefore worth roughly 2.5-2.8x on a network solve and 1.5x
on a coupled forecast. That is a large enough change to the engine's numbers
to need its own guard rails, which is what this file is.

The accuracy bound is the point. The table must stay far enough below the
uncertainty of the correlation it approximates that no engineering answer can
depend on the difference -- and "far enough" has to be a measured number in a
test, not a claim in a docstring, because the easiest way to lose this is a
well-meaning change to the grid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.fluids import (
    Z_TABLE_HIGH,
    Z_TABLE_LOW,
    Z_TABLE_POINTS,
    GasProperties,
    clear_z_tables,
    z_factor_dak,
)

# What the table must beat. The Standing-Katz chart that Dranchuk-Abou-Kassem
# is a fit to can be read to about 0.002 in Z, and DAK's own fit error against
# the chart is of that order. An interpolation error two orders below that
# cannot affect any engineering answer.
CHART_READING_ACCURACY = 0.002
REQUIRED_MARGIN = 100.0

GASES = [
    GasProperties(gravity=0.65, temperature_r=540.0),
    GasProperties(gravity=0.70, temperature_r=660.0),
    GasProperties(gravity=0.85, temperature_r=620.0),
    GasProperties(gravity=0.60, temperature_r=700.0),
    GasProperties(gravity=0.75, temperature_r=580.0, y_co2=0.05, y_h2s=0.02),
]


def direct(gas: GasProperties, pressure: float) -> float:
    p_pc, t_pc = gas.pseudo_criticals()
    return z_factor_dak(pressure, gas.temperature_r, p_pc, t_pc)


# ------------------------------------------------------------- accuracy ----


def test_the_table_is_far_below_the_accuracy_of_the_correlation_it_fits():
    """The measured bound, across five gases including a sour one, over the
    whole range the engine works in."""
    probe = np.linspace(Z_TABLE_LOW, Z_TABLE_HIGH, 2000)
    worst = 0.0
    for gas in GASES:
        for pressure in probe:
            error = abs(gas.z_at(float(pressure)) - direct(gas, float(pressure)))
            worst = max(worst, error)

    assert worst < CHART_READING_ACCURACY / REQUIRED_MARGIN, (
        "table error %.2e is within %.0fx of chart-reading accuracy"
        % (worst, CHART_READING_ACCURACY / worst))
    # And the specific number measured when this was built, so a change to
    # the grid that quietly costs an order of magnitude is visible.
    assert worst < 2.0e-05, worst


def test_the_table_is_exact_at_its_own_knots():
    """Interpolation error is zero where the table was evaluated. If this
    fails, the grid and the lookup disagree about where the points are."""
    gas = GASES[0]
    knots = np.linspace(Z_TABLE_LOW, Z_TABLE_HIGH, Z_TABLE_POINTS)
    for pressure in (knots[0], knots[1], knots[len(knots) // 2], knots[-1]):
        assert gas.z_at(float(pressure)) == pytest.approx(
            direct(gas, float(pressure)), abs=1e-12)


def test_z_still_behaves_like_z():
    """Independent of the table: Z dips below one at moderate pressure and
    climbs back above it at high pressure. A table that broke the shape would
    still pass an error bound if the error bound were loose."""
    gas = GasProperties(gravity=0.65, temperature_r=620.0)
    assert gas.z_at(50.0) > 0.98
    assert gas.z_at(1500.0) < 0.90
    assert gas.z_at(10000.0) > 1.0


# ---------------------------------------------------------------- range ----


def test_outside_the_table_the_correlation_is_called_directly():
    """An unusual pressure is slower, never wrong."""
    gas = GASES[0]
    for pressure in (Z_TABLE_HIGH + 1000.0, 20000.0):
        assert gas.z_at(pressure) == pytest.approx(direct(gas, pressure), abs=1e-14)
    below = Z_TABLE_LOW / 2.0
    assert gas.z_at(below) == pytest.approx(direct(gas, below), abs=1e-14)


def test_the_range_covers_what_the_engine_actually_sees():
    """Atmospheric at a separator through to a deep high-pressure reservoir.
    If the range shrinks below this, common cases fall off the fast path and
    the speedup quietly disappears."""
    assert Z_TABLE_LOW <= 14.7
    assert Z_TABLE_HIGH >= 10_000.0


def test_a_pinned_z_still_short_circuits():
    """`fixed_z` reproduces the Phase 0 spike exactly and must not acquire an
    interpolation error."""
    gas = GasProperties(fixed_z=0.9)
    for pressure in (20.0, 900.0, 5000.0, 50_000.0):
        assert gas.z_at(pressure) == 0.9


# ---------------------------------------------------------------- cache ----


def test_equal_gases_share_one_table():
    """The engine builds many equal GasProperties -- one per well, per branch,
    per forecast step. Each building its own table would spend more on
    construction than the lookups save."""
    clear_z_tables()
    first = GasProperties(gravity=0.68, temperature_r=600.0)
    second = GasProperties(gravity=0.68, temperature_r=600.0)
    first.z_at(500.0)
    second.z_at(500.0)
    # Same object, not merely equal values.
    assert first._z_table()[1] is second._z_table()[1]


def test_different_gases_get_different_tables():
    """The obvious way to make this fast and wrong."""
    clear_z_tables()
    sweet = GasProperties(gravity=0.65, temperature_r=600.0)
    sour = GasProperties(gravity=0.65, temperature_r=600.0, y_co2=0.10,
                         y_h2s=0.05)
    hot = GasProperties(gravity=0.65, temperature_r=700.0)
    heavy = GasProperties(gravity=0.85, temperature_r=600.0)

    values = [g.z_at(2000.0) for g in (sweet, sour, hot, heavy)]
    assert len(set(values)) == 4, values
    for gas in (sweet, sour, hot, heavy):
        assert gas.z_at(2000.0) == pytest.approx(direct(gas, 2000.0), abs=2e-05)


def test_clearing_the_cache_changes_nothing_but_timing():
    gas = GASES[1]
    before = [gas.z_at(p) for p in (100.0, 1500.0, 6000.0)]
    clear_z_tables()
    after = [gas.z_at(p) for p in (100.0, 1500.0, 6000.0)]
    assert before == after


# ------------------------------------------------------- it is worth it ----


def test_the_table_is_actually_faster():
    """The entire justification. Guarded because a future change -- rebuilding
    per call, or falling off the fast path -- would leave every other test in
    this file passing while the engine got slower."""
    import time

    gas = GasProperties(gravity=0.66, temperature_r=550.0)
    p_pc, t_pc = gas.pseudo_criticals()
    probe = np.linspace(100.0, 8000.0, 3000)

    gas.z_at(500.0)                      # build the table once, outside timing
    start = time.perf_counter()
    for pressure in probe:
        gas.z_at(float(pressure))
    tabled = time.perf_counter() - start

    start = time.perf_counter()
    for pressure in probe:
        z_factor_dak(float(pressure), gas.temperature_r, p_pc, t_pc)
    straight = time.perf_counter() - start

    assert straight > 2.0 * tabled, (
        "table is only %.1fx faster than calling the correlation"
        % (straight / tabled))
