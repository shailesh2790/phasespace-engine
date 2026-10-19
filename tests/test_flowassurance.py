"""Tests for flow assurance: inhibitor dosing, insulation, wax.

The strongest test here is the agreement between Hammerschmidt and
Nielsen-Bucklin where they hand over at 20 wt% methanol. They were derived by
different routes from different data, so agreeing to about a degree at the
boundary is real evidence rather than bookkeeping -- and it is the only check
available that does not reduce to re-running the same formula.

The other test worth reading is the one asserting that insulation cannot fix a
Joule-Thomson-driven line. That is the case where software that always returns
a number sends someone to spend money on insulation that physically cannot
work.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.flowassurance import (
    HAMMERSCHMIDT_K_F,
    INHIBITOR_MW,
    flow_assurance_report,
    hydrate_depression_f,
    inhibitor_requirement,
    required_insulation_u,
    wax_risk_along,
    weight_percent_for_depression,
)
from phasespace.sim.thermal import (
    ThermalPath,
    ThermalStream,
    hydrate_temperature_r,
    temperature_profile,
)
from phasespace.sim.units import f_to_r

PATH = ThermalPath(diameter_ft=1.0, overall_u=0.6, ambient_r=f_to_r(50.0))
STREAM = ThermalStream(mass_rate_lb_hr=120_000.0)
LENGTH = 20.0 * 5280.0
SEGMENTS = 80
PRESSURES = [1400.0 - 8.0 * i for i in range(SEGMENTS + 1)]


def cooled_profile(inlet_f=120.0, u_value=0.6, mu=0.03):
    path = ThermalPath(diameter_ft=1.0, overall_u=u_value, ambient_r=f_to_r(50.0))
    return temperature_profile(f_to_r(inlet_f), path, STREAM, LENGTH,
                               pressures_psia=PRESSURES, segments=SEGMENTS,
                               joule_thomson_f_per_psi=mu)


# ------------------------------------------------------- depression ---------


def test_hammerschmidt_against_the_published_form():
    """dT = K*W/(M*(100-W)), checked against the arithmetic."""
    for inhibitor in ("meg", "deg", "ethanol"):
        expected = (HAMMERSCHMIDT_K_F * 15.0
                    / (INHIBITOR_MW[inhibitor] * 85.0))
        assert hydrate_depression_f(15.0, inhibitor) == pytest.approx(expected, rel=1e-12)


def test_methanol_and_meg_match_published_magnitudes():
    """20 wt% methanol depresses about 18 degF and 20 wt% MEG about 9 -- the
    numbers a flow assurance engineer carries in their head. Methanol is the
    stronger inhibitor per unit mass because it is the lighter molecule, which
    is the whole content of the colligative argument."""
    methanol = hydrate_depression_f(20.0, "methanol")
    meg = hydrate_depression_f(20.0, "meg")
    assert 16.0 < methanol < 21.0, methanol
    assert 8.0 < meg < 11.0, meg
    assert methanol > meg


def test_the_two_correlations_agree_where_they_hand_over():
    """Hammerschmidt and Nielsen-Bucklin were derived by different routes from
    different data. Agreement at the 20 wt% handover is independent evidence
    that both are implemented correctly; a unit slip in either would show up
    here as a step change."""
    hammerschmidt = HAMMERSCHMIDT_K_F * 20.0 / (INHIBITOR_MW["methanol"] * 80.0)
    just_above = hydrate_depression_f(20.0001, "methanol")
    assert just_above == pytest.approx(hammerschmidt, abs=1.5), (
        "%.2f vs %.2f degF across the handover" % (just_above, hammerschmidt))


def test_depression_rises_with_concentration():
    previous = 0.0
    for weight_percent in (5.0, 10.0, 20.0, 35.0, 50.0):
        current = hydrate_depression_f(weight_percent, "methanol")
        assert current > previous, weight_percent
        previous = current


def test_zero_inhibitor_depresses_nothing():
    assert hydrate_depression_f(0.0, "methanol") == 0.0


# ---------------------------------------------------------- inversion ------


def test_inversion_round_trips_through_both_branches():
    """Concentration -> depression -> concentration, across the handover, so
    both the closed form and the bisected branch are exercised."""
    for target in (5.0, 15.0, 25.0, 40.0, 60.0):
        weight_percent, _, _ = weight_percent_for_depression(target, "methanol")
        assert hydrate_depression_f(weight_percent, "methanol") == pytest.approx(
            target, rel=1e-6), "%.1f degF" % target


def test_inversion_round_trips_for_glycol():
    for target in (5.0, 10.0, 18.0):
        weight_percent, correlation, _ = weight_percent_for_depression(target, "meg")
        assert correlation == "hammerschmidt"
        assert hydrate_depression_f(weight_percent, "meg") == pytest.approx(
            target, rel=1e-9)


def test_glycol_beyond_its_envelope_is_flagged_not_swapped():
    """There is no published high-concentration successor for glycol here, so
    the answer is marked as extrapolated rather than silently computed with a
    correlation that does not apply."""
    weight_percent, correlation, extrapolated = weight_percent_for_depression(30.0, "meg")
    assert correlation == "hammerschmidt"
    assert weight_percent > 20.0
    assert extrapolated is True


def test_unknown_inhibitor_is_rejected():
    for call in (lambda: hydrate_depression_f(10.0, "brandy"),
                 lambda: weight_percent_for_depression(10.0, "brandy")):
        with pytest.raises(ValueError):
            call()


# ------------------------------------------------------------- dosing ------


def test_no_dose_when_the_stream_is_already_warm():
    """The false-positive case. A tool that prescribes methanol for a line that
    does not need it gets switched off, and rightly."""
    dose = inhibitor_requirement(1000.0, f_to_r(150.0), water_rate_lb_hr=500.0)
    assert dose.weight_percent_in_water == 0.0
    assert dose.volume_rate_gal_day == 0.0
    assert any("already clear" in note for note in dose.notes)


def test_a_cold_stream_gets_a_dose_that_actually_clears_the_curve():
    """The dose has to achieve the margin it was sized for -- checked by
    putting the prescribed concentration back through the forward
    correlation."""
    pressure, temperature = 1000.0, f_to_r(50.0)
    dose = inhibitor_requirement(pressure, temperature, water_rate_lb_hr=500.0,
                                 safety_margin_f=5.0)
    assert dose.weight_percent_in_water > 0.0
    achieved = hydrate_depression_f(dose.weight_percent_in_water, "methanol")
    assert achieved == pytest.approx(dose.required_depression_f, rel=1e-4)

    hydrate_r = hydrate_temperature_r(pressure)[0]
    assert temperature + achieved >= hydrate_r + 5.0 - 1e-6


def test_dose_scales_with_the_water_rate():
    """Concentration is set by temperature; the *rate* is set by how much water
    there is. Doubling the water doubles the truck."""
    cold = f_to_r(50.0)
    small = inhibitor_requirement(1000.0, cold, water_rate_lb_hr=500.0)
    large = inhibitor_requirement(1000.0, cold, water_rate_lb_hr=1000.0)
    assert large.weight_percent_in_water == pytest.approx(
        small.weight_percent_in_water, rel=1e-12)
    assert large.volume_rate_gal_day == pytest.approx(
        2.0 * small.volume_rate_gal_day, rel=1e-12)


def test_dose_says_it_is_a_floor():
    """Methanol lost to the vapour and condensate phases is not counted, and at
    high gas rates that loss approaches the aqueous requirement. A number that
    does not say so gets under-dosed in the field."""
    dose = inhibitor_requirement(1000.0, f_to_r(50.0), water_rate_lb_hr=500.0)
    assert any("floor" in note for note in dose.notes)


def test_methanol_needs_less_volume_than_glycol_for_the_same_duty():
    cold = f_to_r(45.0)
    methanol = inhibitor_requirement(1000.0, cold, 500.0, inhibitor="methanol")
    glycol = inhibitor_requirement(1000.0, cold, 500.0, inhibitor="meg")
    assert methanol.weight_percent_in_water < glycol.weight_percent_in_water
    assert methanol.mass_rate_lb_hr < glycol.mass_rate_lb_hr


def test_negative_water_rate_is_rejected():
    with pytest.raises(ValueError):
        inhibitor_requirement(1000.0, f_to_r(50.0), water_rate_lb_hr=-1.0)


# --------------------------------------------------------- insulation ------


def test_insulation_cannot_fix_joule_thomson_cooling():
    """The case this module exists for.

    With a large JT coefficient the line is driven into the hydrate region by
    expansion alone. At U = 0 the walls are perfect and it still plugs, so the
    honest answer is that insulation is the wrong purchase -- not "insulate to
    some very small U", which is what a routine that always returns a number
    would say.
    """
    requirement = required_insulation_u(
        f_to_r(70.0), PATH, STREAM, LENGTH, PRESSURES, SEGMENTS,
        joule_thomson_f_per_psi=0.06, safety_margin_f=5.0)
    assert requirement.achievable is False
    assert requirement.margin_at_required_u_f < 0.0
    assert any("Joule-Thomson" in note for note in requirement.notes)


def test_insulating_enough_clears_the_line():
    """A line that is marginal bare can be saved by insulation, and the U
    returned has to actually achieve the margin."""
    requirement = required_insulation_u(
        f_to_r(150.0), PATH, STREAM, LENGTH, PRESSURES, SEGMENTS,
        joule_thomson_f_per_psi=0.015, safety_margin_f=5.0)
    assert requirement.achievable is True
    assert requirement.margin_at_required_u_f >= -1e-6

    profile = cooled_profile(inlet_f=150.0, u_value=requirement.required_u, mu=0.015)
    report = flow_assurance_report(profile, safety_margin_f=5.0)
    assert not report.hydrate_forms


def test_a_line_that_never_needs_insulation_says_so():
    """The answer "do nothing" has to be reachable, or the tool always sells
    insulation.

    It needs warm surroundings, and that is not a contrivance to make the test
    pass -- it is the physics. In 50 degF soil every line eventually arrives at
    50 degF, which is below the hydrate curve at any transmission pressure, so
    a cold-soil line always needs *something*. Only where ambient itself sits
    above the curve is bare pipe safe at any length.
    """
    warm = ThermalPath(diameter_ft=1.0, overall_u=0.6, ambient_r=f_to_r(110.0))
    requirement = required_insulation_u(
        f_to_r(250.0), warm, STREAM, LENGTH, PRESSURES, SEGMENTS,
        joule_thomson_f_per_psi=0.0, safety_margin_f=5.0)
    assert requirement.achievable is True
    assert any("no insulation is required" in note for note in requirement.notes)


def test_cold_soil_always_needs_something():
    """The corollary, stated as its own claim because it is the operationally
    important one: in cold ground, "it is hot at the compressor" is never an
    argument that a long line is safe."""
    requirement = required_insulation_u(
        f_to_r(250.0), PATH, STREAM, LENGTH, PRESSURES, SEGMENTS,
        joule_thomson_f_per_psi=0.0, safety_margin_f=5.0)
    assert requirement.achievable is True
    assert requirement.required_u < 20.0, (
        "bare pipe in 50 degF soil should not be declared safe")


def test_more_insulation_is_monotonically_safer():
    """The property that makes bisection valid rather than merely convenient."""
    margins = []
    for u_value in (0.05, 0.2, 0.6, 2.0):
        report = flow_assurance_report(
            cooled_profile(inlet_f=150.0, u_value=u_value, mu=0.015))
        margins.append(report.worst_hydrate_margin_f)
    assert all(a >= b for a, b in zip(margins, margins[1:])), margins


# --------------------------------------------------------------- wax -------


def test_wax_window_is_found_where_the_line_goes_cold():
    profile = cooled_profile()
    wax = wax_risk_along(profile, f_to_r(90.0))
    assert wax.forms
    assert 0.0 < wax.first_station_ft < LENGTH
    assert wax.coldest_f < 90.0


def test_no_wax_on_a_line_that_stays_warm():
    profile = cooled_profile(inlet_f=200.0, u_value=0.05, mu=0.0)
    wax = wax_risk_along(profile, f_to_r(60.0))
    assert not wax.forms
    assert math.isinf(wax.first_station_ft)


# ------------------------------------------------------------ report -------


def test_report_finds_the_line_that_an_isothermal_model_calls_safe():
    """End to end: the 20 mile line that leaves the compressor at 120 degF,
    comfortably clear of the hydrate curve, and is inside it well before the
    end."""
    report = flow_assurance_report(cooled_profile(), wax_appearance_r=f_to_r(80.0))
    assert report.hydrate_forms
    assert 0.0 < report.hydrate_first_station_ft < LENGTH
    assert report.worst_hydrate_margin_f < 0.0
    assert any("Hydrate region entered" in note for note in report.notes)


def test_report_calls_out_when_insulation_will_not_help():
    report = flow_assurance_report(cooled_profile(inlet_f=70.0, u_value=0.05, mu=0.06))
    assert any("insulation alone cannot fix" in note for note in report.notes)


def test_clean_line_reports_nothing():
    report = flow_assurance_report(cooled_profile(inlet_f=250.0, u_value=0.02, mu=0.0))
    assert not report.hydrate_forms
    assert not report.wax.forms
    assert report.notes == []
