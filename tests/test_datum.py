"""Moving a static pressure to a datum, and refusing to when it cannot.

The arithmetic is a multiplication. What is worth testing is the choice of
gradient, because that is where this goes wrong by an order of magnitude
rather than by a few percent: a shut-in well has gas standing above liquid,
the two gradients differ by more than ten times, and picking the gas one
where the interval is liquid invents reservoir pressure.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.datum import (
    GAS,
    LIQUID,
    MAX_EXTRAPOLATION_FT,
    UNCLEAR,
    fluid_contact_ft,
    pressure_at_datum,
    segments,
    suggest_datum,
)


@dataclass
class Point:
    """Minimal stand-in for a survey reading."""

    tvd_ft: float
    pressure_psia: float


def liquid_column(top_ft=5000.0, top_psia=2000.0, gradient=0.40,
                  points=5, spacing=500.0):
    return [Point(top_ft + i * spacing, top_psia + i * spacing * gradient)
            for i in range(points)]


def gas_over_liquid(contact_ft=6000.0):
    """A shut-in well: gas from 4,000 to the contact, liquid below."""
    gas = [Point(4000.0, 1000.0),
           Point(5000.0, 1030.0),
           Point(contact_ft, 1060.0)]          # 0.030 psi/ft
    liquid = [Point(contact_ft + 1000.0, 1060.0 + 400.0),
              Point(contact_ft + 2000.0, 1060.0 + 800.0)]   # 0.400 psi/ft
    return gas + liquid


# -------------------------------------------------------------- segments --


def test_a_gradient_identifies_the_fluid():
    found = segments(liquid_column(gradient=0.40))
    assert all(s.fluid == LIQUID for s in found)
    found = segments(liquid_column(gradient=0.03))
    assert all(s.fluid == GAS for s in found)


def test_a_gradient_between_gas_and_liquid_is_left_unclear():
    """A gassy oil or a foaming column genuinely sits there, and the honest
    answer is that the gradient does not identify the fluid."""
    found = segments(liquid_column(gradient=0.20))
    assert all(s.fluid == UNCLEAR for s in found)


def test_readings_too_close_together_are_not_an_interval():
    """A slope over 20 ft is gauge resolution, not fluid density."""
    close = [Point(5000.0, 2000.0), Point(5020.0, 2008.0)]
    assert segments(close) == []


def test_segments_come_back_deepest_last_whatever_order_they_arrive_in():
    shuffled = list(reversed(liquid_column()))
    found = segments(shuffled)
    assert found[0].top_ft < found[-1].bottom_ft


# ------------------------------------------------------- the gas contact --


def test_the_gas_liquid_contact_is_found_where_the_gradient_changes():
    """Worth having on its own: in a shut-in well this is where the liquid
    level has risen to, and watching it move says more about loading than
    any single pressure does."""
    assert fluid_contact_ft(gas_over_liquid(contact_ft=6000.0)) == 6000.0


def test_a_single_fluid_column_has_no_contact():
    assert fluid_contact_ft(liquid_column()) == 0.0


# ----------------------------------------------- correcting to the datum --


def test_a_datum_inside_a_measured_interval_uses_the_measured_gradient():
    """The best case, and the one with no extrapolation cost at all."""
    points = liquid_column(top_ft=5000.0, top_psia=2000.0, gradient=0.40)
    result = pressure_at_datum(points, 6250.0)

    assert result.usable
    assert result.extrapolated_ft == 0.0
    assert result.uncertainty_psi == 0.0
    assert result.gradient_psi_ft == pytest.approx(0.40)
    # 5,000 ft at 2,000 psia, 0.40 psi/ft, so 6,250 ft is 2,500 psia.
    assert result.pressure_psia == pytest.approx(2500.0, abs=1.0)


def test_correcting_below_the_deepest_reading_reports_what_it_cost():
    points = liquid_column(top_ft=5000.0, top_psia=2000.0, gradient=0.40,
                           points=3, spacing=500.0)   # deepest 6,000 ft
    result = pressure_at_datum(points, 6500.0)

    assert result.usable
    assert result.extrapolated_ft == pytest.approx(500.0)
    assert result.uncertainty_psi == pytest.approx(25.0)
    assert result.pressure_psia == pytest.approx(2600.0, abs=1.0)
    assert any("carried 500 ft" in note for note in result.notes)


def test_the_datum_picks_the_gradient_of_its_own_interval_not_the_deepest():
    """The whole point. This well is gas above 6,000 ft and liquid below.
    A datum at 5,500 ft must be corrected on the gas gradient; using the
    liquid one would be out by a factor of thirteen."""
    points = gas_over_liquid(contact_ft=6000.0)
    shallow = pressure_at_datum(points, 5500.0)

    assert shallow.usable
    assert shallow.fluid == GAS
    assert shallow.gradient_psi_ft == pytest.approx(0.030, abs=0.005)
    # 1,030 psia at 5,000 ft on a gas gradient, not 1,030 + 200.
    assert shallow.pressure_psia == pytest.approx(1045.0, abs=2.0)


def test_a_deep_datum_in_the_same_well_uses_the_liquid_gradient():
    points = gas_over_liquid(contact_ft=6000.0)
    deep = pressure_at_datum(points, 7500.0)
    assert deep.usable
    assert deep.fluid == LIQUID
    assert deep.gradient_psi_ft == pytest.approx(0.400, abs=0.01)


def test_extrapolating_too_far_is_refused_rather_than_error_barred():
    """A figure with a 200 psi uncertainty gets quoted without it."""
    points = liquid_column(points=3)             # deepest 6,000 ft
    result = pressure_at_datum(points, 6000.0 + MAX_EXTRAPOLATION_FT + 500.0)

    assert not result.usable
    assert "will carry a gradient at most" in result.notes[0]
    assert "gas-liquid contact" in result.notes[0]


def test_carrying_a_gas_gradient_downwards_says_what_it_assumes():
    """Below the deepest gauge nobody knows what stands there, and if it is
    liquid rather than gas the real pressure is much higher."""
    gas_only = [Point(4000.0, 1000.0), Point(5000.0, 1030.0),
                Point(6000.0, 1060.0)]
    result = pressure_at_datum(gas_only, 6800.0)

    assert result.usable
    assert result.fluid == GAS
    assert any("assumes gas continues" in note for note in result.notes)
    assert any("real pressure is higher" in note for note in result.notes)


def test_one_reading_cannot_be_moved_at_all():
    """Without a gradient the only way to correct is to assume a fluid
    density, and that assumption is the largest error in the
    calculation."""
    result = pressure_at_datum([Point(6000.0, 2400.0)], 6500.0)
    assert not result.usable
    assert "no gradient to correct with" in result.notes[0]


def test_a_column_that_loses_pressure_with_depth_is_refused():
    """A static column cannot do that, so these are not static readings --
    the gauge was probably still moving."""
    rising = [Point(5000.0, 2000.0), Point(6000.0, 1800.0),
              Point(7000.0, 1600.0)]
    result = pressure_at_datum(rising, 6500.0)
    assert not result.usable
    assert "implausible gradient" in result.notes[0]
    assert "still moving" in result.notes[0]


def test_a_datum_above_every_reading_is_flagged_as_unusual():
    points = liquid_column(top_ft=5000.0)
    result = pressure_at_datum(points, 4800.0)
    assert any("above the shallowest reading" in n for n in result.notes)
    assert any("another field" in n for n in result.notes)


def test_the_correction_is_reported_so_it_can_be_checked():
    points = liquid_column(top_ft=5000.0, top_psia=2000.0, gradient=0.40)
    result = pressure_at_datum(points, 6250.0)
    assert result.correction_psi == pytest.approx(
        result.pressure_psia - result.measured_psia)
    assert "psi/ft" in result.describe()
    assert "uncertainty" in result.describe()


# ---------------------------------------------------- choosing the datum --


def test_the_suggested_datum_is_the_median_of_a_wells_own_depths():
    surveys = [liquid_column(points=3),            # deepest 6,000
               liquid_column(points=5),            # deepest 7,000
               liquid_column(points=4)]            # deepest 6,500
    assert suggest_datum(surveys) == 6500.0


def test_the_suggested_datum_handles_an_even_count():
    surveys = [liquid_column(points=3), liquid_column(points=5)]
    assert suggest_datum(surveys) == pytest.approx(6500.0)


def test_no_surveys_gives_no_datum_rather_than_zero_pretending_to_be_one():
    assert suggest_datum([]) == 0.0
