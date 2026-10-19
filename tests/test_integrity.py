"""Corrosion and erosion along a string, and the four things refused.

The correlation is checked against values anyone can look up, because a
corrosion rate that is plausible but wrong by a factor of three is worse
than no number: it goes into an inspection plan.

The rest of the file is about restraint. Sour gas, a missing
composition, a mismatched traverse and scale chemistry that does not
exist are all cases where the honest output is a sentence rather than a
figure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.integrity import (
    FILM_TEMPERATURE_C,
    OIL_WETTING_WATER_CUT,
    PSI_PER_BAR,
    assess,
    co2_partial_pressure_bar,
    corrosion_rate_mm_yr,
    film_factor,
    scale_inputs,
)


def at(celsius, bar):
    """Rate at a temperature and CO2 partial pressure, in the units the
    correlation was published in."""
    return corrosion_rate_mm_yr(celsius * 1.8 + 32.0, bar * PSI_PER_BAR, 1.0)[0]


# ------------------------------------------------- against the published --


@pytest.mark.parametrize("celsius,expected", [(25, 1.16), (40, 2.18), (60, 4.65)])
def test_the_rate_matches_de_waard_at_one_bar(celsius, expected):
    """Below 60 degC there is no protective film, so this is the bare
    1991 correlation and the numbers are the ones in the paper's own
    nomogram."""
    assert at(celsius, 1.0) == pytest.approx(expected, rel=0.02)


def test_the_rate_follows_the_two_thirds_power_of_partial_pressure():
    """log10(V) carries 0.67 log10(pCO2), so ten times the CO2 is about
    4.7 times the rate, not ten."""
    assert at(60, 10.0) / at(60, 1.0) == pytest.approx(10.0 ** 0.67, rel=0.01)


def test_corrosion_rises_steeply_with_temperature_below_the_film():
    assert at(60, 1.0) > 3.5 * at(25, 1.0)


# ----------------------------------------------------------- the film --


def test_there_is_no_film_below_sixty_degrees():
    assert film_factor(FILM_TEMPERATURE_C - 1.0, 1.0) == 1.0
    assert film_factor(20.0, 1.0) == 1.0


def test_the_film_slows_attack_above_sixty():
    assert film_factor(100.0, 1.0) < 1.0
    assert film_factor(140.0, 1.0) < film_factor(100.0, 1.0)


def test_the_hottest_point_is_not_the_worst_one():
    """The result worth having, and the reason this reports a depth. On a
    real well the peak sat at 3,194 ft and 146 degF while the bottom was
    9,747 ft and 211 degF -- so an inspection at total depth looks in the
    wrong place."""
    depths = [0.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0]
    temperatures = [100.0, 130.0, 160.0, 185.0, 205.0, 215.0]
    pressures = [3000.0, 3200.0, 3400.0, 3600.0, 3800.0, 4000.0]
    report = assess(depths, temperatures, pressures, co2_fraction=0.02,
                    water_cut=0.6)
    assert report.usable
    assert report.worst.depth_ft < depths[-1]
    assert any("not at the hottest point" in note for note in report.notes)


# --------------------------------------------------------- the refusals --


def test_sour_gas_is_refused_rather_than_run_through_a_co2_correlation():
    """H2S attacks by a different mechanism and brings cracking, which is
    not a thinning rate at all."""
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.5, h2s_fraction=0.001)
    assert report.sour
    assert not report.usable
    assert "MR0175" in report.describe()


def test_a_trace_of_h2s_does_not_trip_the_refusal():
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.5, h2s_fraction=1e-6)
    assert not report.sour
    assert report.usable


def test_no_co2_says_the_composition_may_simply_be_missing():
    """An absent composition and a true zero look identical in the data,
    and the difference decides whether there is nothing to worry about or
    nothing to go on."""
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.0, water_cut=0.5)
    assert not report.usable
    assert any("look identical here" in note for note in report.notes)


def test_a_mismatched_traverse_is_refused_with_the_counts():
    report = assess([0.0, 100.0, 200.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.5)
    assert not report.usable
    assert "3 depths, 2 temperatures" in report.describe()


def test_scale_names_what_it_would_need_instead_of_guessing():
    """No archive this engine reads carries a water analysis, and a
    saturation index from assumed chemistry is a number about the
    assumption."""
    needed = scale_inputs()
    assert any("calcium" in item for item in needed)
    assert any("sulphate" in item for item in needed)
    assert any("barium" in item for item in needed)


# ------------------------------------------------------- the conditions --


def test_a_dry_fast_well_is_reported_as_oil_wetted():
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02,
                    water_cut=OIL_WETTING_WATER_CUT - 0.1,
                    velocity_ft_s=12.0)
    assert report.oil_wetted_throughout
    assert any("oil-wetted" in note for note in report.notes)
    # The rate is still computed and reported: it is what happens the
    # moment the well slows down or the water cut climbs.
    assert report.worst.rate_mm_yr > 0.0


def test_a_wet_well_is_not_called_oil_wetted():
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.8, velocity_ft_s=12.0)
    assert not report.oil_wetted_throughout


def test_a_slow_dry_well_is_not_called_oil_wetted_either():
    """Low water cut is not enough on its own -- the water has to be
    moving fast enough to stay entrained."""
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.1, velocity_ft_s=0.5)
    assert not report.oil_wetted_throughout


def test_an_erosional_exceedance_is_reported_with_its_caveat():
    """API RP 14E's C-factor has no solids term and is widely treated as
    physics. An exceedance is a reason to check for sand."""
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.5,
                    erosional_ratio=1.4, erosional_depth_ft=5000.0)
    assert any("check for sand" in note for note in report.notes)


# ---------------------------------------------------------- the reporting --


def test_every_result_carries_the_basis_it_was_computed_on():
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.5)
    assert "upper bound" in report.worst.basis
    assert any("upper bound" in note for note in report.notes)


def test_years_to_wall_loss_is_available_and_uses_the_same_rate():
    report = assess([0.0, 5000.0], [100.0, 180.0], [2000.0, 3000.0],
                    co2_fraction=0.02, water_cut=0.5)
    worst = report.worst
    years = worst.years_to(0.25)
    assert years == pytest.approx(0.25 * 25.4 / worst.rate_mm_yr)


def test_partial_pressure_is_dalton():
    assert co2_partial_pressure_bar(1450.38, 0.02) == pytest.approx(2.0, rel=1e-3)
