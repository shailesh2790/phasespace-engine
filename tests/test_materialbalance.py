"""Tests for material balance and oil PVT.

Material balance is an accounting identity, not a correlation, so it can be
checked much harder than anything else in this engine. Three kinds of check.

Against the identity itself: the drive indices must sum to one, the forward
and backward directions must invert each other, and a synthetic history
generated from a known tank must give that tank's volume back exactly.

Against independent physics: a volumetric gas reservoir recovers nearly
1 - pa/pi of its gas, Bo peaks at the bubble point and falls on both sides of
it, oil viscosity has its minimum there, and a reservoir with no free gas is
driven by rock compaction.

Against the failure that matters: a p/z line bent by water influx fits a
straight line perfectly well and reports gas that is not there. The test is
that the bend is detected and named, not fitted away.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.fluids import GasProperties
from phasespace.sim.materialbalance import (
    MIN_PRESSURE_PSIA,
    GasTank,
    OilTank,
    fit_gas_balance,
    fit_oil_balance,
)
from phasespace.sim.pvt_oil import (
    STANDING,
    VAZQUEZ_BEGGS,
    OilPVT,
    bubble_point_standing,
    dead_oil_viscosity,
    gas_fvf,
    oil_fvf_standing,
    solution_gas_standing,
    water_fvf,
)

GAS = GasProperties(gravity=0.65, temperature_r=620.0)


def oil():
    return OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0,
                  rsb_scf_stb=600.0)


# ================================================================ oil PVT ===


def test_standing_rs_and_bubble_point_are_near_inverses():
    """pb is Standing's Rs solved for pressure, and the two round-trip to
    about one part in ten thousand rather than exactly.

    That is a property of the published correlation, not of this code: the
    forward form uses the exponent 1.2048 and the inverse uses 0.83, and
    1/1.2048 is 0.83008. Standing rounded. Asserting an exact inverse here
    would be asserting something the industry's own equations do not satisfy,
    so the tolerance is set where the rounding puts it -- tight enough to
    catch a transcribed coefficient, which is what this test is for.
    """
    for rsb in (120.0, 400.0, 900.0):
        pb = bubble_point_standing(rsb, 200.0, 35.0, 0.70)
        round_tripped = solution_gas_standing(pb, 200.0, 35.0, 0.70)
        assert round_tripped == pytest.approx(rsb, rel=1e-3), rsb
        assert abs(round_tripped - rsb) / rsb > 1e-9, (
            "an exact round trip would mean the exponents are reciprocals, "
            "which they are not")


def test_solution_gas_rises_with_pressure_and_falls_with_heat():
    assert (solution_gas_standing(1000.0, 200.0, 35.0, 0.7)
            < solution_gas_standing(2000.0, 200.0, 35.0, 0.7))
    # Hotter oil holds less gas, which is why a hot reservoir has a higher
    # bubble point for the same solution gas.
    assert (solution_gas_standing(2000.0, 250.0, 35.0, 0.7)
            < solution_gas_standing(2000.0, 150.0, 35.0, 0.7))


def test_bo_peaks_at_the_bubble_point():
    """Below pb, Bo falls with pressure because gas leaves solution. Above it,
    Bo falls with *rising* pressure because the oil is being compressed. The
    peak is at pb, and a model that applies one branch throughout misses it --
    at exactly the pressure a depleting reservoir passes through."""
    pvt = oil()
    pb = pvt.bubble_point_psia
    at_pb = pvt.bo(pb)
    assert pvt.bo(pb - 500.0) < at_pb
    assert pvt.bo(pb + 500.0) < at_pb
    assert pvt.bo(pb + 2000.0) < pvt.bo(pb + 500.0)


def test_viscosity_is_lowest_at_the_bubble_point():
    """The same turn, in the property that decides how well the well flows."""
    pvt = oil()
    pb = pvt.bubble_point_psia
    assert pvt.viscosity(pb) < pvt.viscosity(pb - 1000.0)
    assert pvt.viscosity(pb) < pvt.viscosity(pb + 1000.0)


def test_solution_gas_is_capped_above_the_bubble_point():
    """No more gas can dissolve once the oil is undersaturated. The unbounded
    correlation keeps rising, which is the classic way to make an
    undersaturated oil appear to absorb gas it cannot."""
    pvt = oil()
    pb = pvt.bubble_point_psia
    assert pvt.rs(pb + 2000.0) == pytest.approx(pvt.rsb_scf_stb)
    assert solution_gas_standing(pb + 2000.0, 200.0, 35.0, 0.70) > pvt.rsb_scf_stb


def test_the_two_correlations_disagree_by_enough_to_matter():
    """Standing and Vazquez-Beggs are both defensible and differ by 10-15% on
    Rs. That spread is the honest uncertainty in an oil model with no
    laboratory PVT, and it is why the correlation is named in the output."""
    standing = OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0,
                      rsb_scf_stb=600.0, correlation=STANDING)
    vb = OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0,
                rsb_scf_stb=600.0, correlation=VAZQUEZ_BEGGS)
    ratio = vb.rs(1500.0) / standing.rs(1500.0)
    assert 0.80 < ratio < 1.20, ratio
    assert abs(ratio - 1.0) > 0.03, ratio
    assert "Standing" in standing.citation and "Vazquez" in vb.citation


def test_calibrating_to_a_lab_point_matches_it_and_reports_the_size():
    pvt = oil()
    target = pvt.rs(1500.0) * 1.08
    applied = pvt.calibrate_to_lab(1500.0, rs_measured=target)
    assert pvt.rs(1500.0) == pytest.approx(target, rel=1e-9)
    assert applied["rs_multiplier"] == pytest.approx(1.08, rel=1e-6)
    assert "warnings" not in applied

    big = oil()
    applied = big.calibrate_to_lab(1500.0, rs_measured=big.rs(1500.0) * 1.45)
    assert any("20%" in text for text in applied["warnings"]), applied


def test_an_oil_needs_either_a_bubble_point_or_its_solution_gas():
    with pytest.raises(ValueError, match="bubble point"):
        OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0)


def test_pvt_outside_the_fitted_api_range_is_refused():
    """Every one of these correlations was fitted between 5 and 60 API. A
    bitumen or a condensate needs laboratory data, not extrapolation."""
    with pytest.raises(ValueError, match="outside 5-60"):
        OilPVT(api=70.0, gas_gravity=0.70, temperature_f=200.0, rsb_scf_stb=600.0)
    with pytest.raises(ValueError, match="outside 5-60"):
        solution_gas_standing(2000.0, 200.0, 3.0, 0.70)


def test_dead_oil_viscosity_rises_as_oil_gets_heavier_and_colder():
    assert dead_oil_viscosity(150.0, 20.0) > dead_oil_viscosity(150.0, 40.0)
    assert dead_oil_viscosity(100.0, 30.0) > dead_oil_viscosity(220.0, 30.0)


def test_gas_volume_factor_is_the_gas_law():
    """Bg = 0.00503 z T / p, in bbl/scf. A value in the wrong decade here is
    a units error that propagates into every reservoir volume."""
    bg = gas_fvf(2000.0, 200.0, 0.85)
    expected = 0.02827 * 0.85 * 659.67 / 2000.0 / 5.614583
    assert bg == pytest.approx(expected, rel=1e-9)
    assert 5e-4 < bg < 5e-3, bg
    # Halving the pressure doubles the volume, Z held fixed.
    assert gas_fvf(1000.0, 200.0, 0.85) == pytest.approx(2.0 * bg, rel=1e-9)


def test_water_volume_factor_stays_near_one():
    for p in (500.0, 3000.0, 6000.0):
        for t in (100.0, 250.0):
            assert 0.95 < water_fvf(p, t) < 1.10, (p, t)


# ======================================================== gas: the p/z line =


def test_the_gas_tank_round_trips_pressure_and_cumulative():
    tank = GasTank(giip_scf=100e9, initial_psia=4000.0, gas=GAS)
    for gp in (0.0, 10e9, 50e9, 85e9):
        pressure = tank.pressure_at(gp)
        # Absolute, against the gas in place. A relative tolerance cannot be
        # satisfied at gp = 0, where the only honest statement is that the
        # round trip lands within a rounding of nothing.
        assert tank.cumulative_at(pressure) == pytest.approx(gp, abs=1e9 * 1e-4), gp


def test_gas_recovery_is_nearly_one_minus_the_pressure_ratio():
    """The reason gas fields recover 80% and oil fields recover 30%: for a
    volumetric gas reservoir the recovery factor is 1 - (pa/za)/(pi/zi), and Z
    varies little enough that the pressure ratio nearly gives it."""
    tank = GasTank(giip_scf=100e9, initial_psia=4000.0, gas=GAS)
    recovery = tank.recovery_factor(500.0)
    assert 0.80 < recovery < 0.95, recovery
    crude = 1.0 - 500.0 / 4000.0
    assert abs(recovery - crude) < 0.07, (recovery, crude)


def test_a_volumetric_history_gives_its_own_gas_back():
    tank = GasTank(giip_scf=100e9, initial_psia=4000.0, gas=GAS)
    cumulatives = np.linspace(0.0, 60e9, 7)
    pressures = np.array([tank.pressure_at(float(g)) for g in cumulatives])
    fit = fit_gas_balance(pressures, cumulatives, GAS)
    assert fit.giip_scf == pytest.approx(100e9, rel=1e-4)
    assert fit.drive == "volumetric depletion"
    assert not fit.water_influx_suspected


def _pressure_for_pz(target, gas):
    low, high = MIN_PRESSURE_PSIA, 10_000.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle / gas.z_at(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def test_water_influx_is_named_rather_than_booked_as_extra_gas():
    """The most consequential thing gas material balance can tell you, and the
    one most often fitted away.

    A p/z plot bent upward by an aquifer still fits a straight line, and that
    line's intercept over slope is a GIIP about 16% larger than the gas that
    is actually there. Raising reserves is the comfortable reading; the
    uncomfortable one is that water is on its way to the wells.
    """
    giip = 100e9
    pz_initial = 4000.0 / GAS.z_at(4000.0)
    cumulatives = np.linspace(0.0, 60e9, 8)
    fraction = cumulatives / giip
    supported = pz_initial * (1.0 - fraction + 0.25 * fraction ** 2)
    pressures = np.array([_pressure_for_pz(float(v), GAS) for v in supported])

    fit = fit_gas_balance(pressures, cumulatives, GAS)
    assert fit.drive == "water influx or a second compartment"
    assert fit.water_influx_suspected
    assert any("outside the tank" in note for note in fit.notes), fit.notes
    # And the straight-line volume really is too big, which is the point.
    assert fit.giip_scf > giip * 1.10


def test_compaction_bends_the_line_the_other_way_and_is_named_separately():
    """Opposite sign, opposite consequence: here the straight line
    *overstates* nothing and understates the gas, and the cause is the rock
    rather than an aquifer. The two must not be reported as one finding."""
    giip = 100e9
    pz_initial = 4000.0 / GAS.z_at(4000.0)
    cumulatives = np.linspace(0.0, 60e9, 8)
    fraction = cumulatives / giip
    compacting = pz_initial * (1.0 - fraction - 0.25 * fraction ** 2)
    pressures = np.array([_pressure_for_pz(float(v), GAS) for v in compacting])

    fit = fit_gas_balance(pressures, cumulatives, GAS)
    assert fit.drive == "pressure-dependent pore volume"
    assert fit.giip_scf < giip


def test_early_depletion_is_flagged_as_a_slope_not_a_reserve():
    tank = GasTank(giip_scf=100e9, initial_psia=4000.0, gas=GAS)
    cumulatives = np.linspace(0.0, 4e9, 5)
    pressures = np.array([tank.pressure_at(float(g)) for g in cumulatives])
    fit = fit_gas_balance(pressures, cumulatives, GAS)
    assert any("not of the reserves" in note for note in fit.notes), fit.notes


def test_two_surveys_are_refused():
    with pytest.raises(ValueError, match="at least 3"):
        fit_gas_balance([4000.0, 3000.0], [0.0, 20e9], GAS)


def test_a_rising_p_over_z_is_refused_as_a_data_error():
    with pytest.raises(ValueError, match="p/z rises"):
        fit_gas_balance([2000.0, 3000.0, 4000.0], [0.0, 20e9, 40e9], GAS)


# =================================================== oil: Havlena-Odeh ======


def test_the_oil_tank_reaches_the_bubble_point_where_expansion_says_it_should():
    """An undersaturated oil's only energy is its own compressibility and the
    rock's, so pressure falls steeply for very little production -- a couple
    of percent of the oil in place to cross a 1,300 psi undersaturated
    range. That steepness is the whole reason undersaturated fields need
    pressure maintenance from day one."""
    pvt = oil()
    tank = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt)
    at_bubble = None
    for fraction in np.linspace(0.001, 0.05, 50):
        pressure = tank.pressure_at(fraction * 50e6, pvt.rsb_scf_stb)
        if pressure <= pvt.bubble_point_psia:
            at_bubble = fraction
            break
    assert at_bubble is not None
    assert 0.005 < at_bubble < 0.04, at_bubble


def test_pressure_falls_monotonically_as_oil_is_produced():
    pvt = oil()
    tank = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt)
    pressures = [tank.pressure_at(f * 50e6, 600.0 + f * 8000.0)
                 for f in (0.0, 0.01, 0.02, 0.05, 0.10, 0.15)]
    assert pressures == sorted(pressures, reverse=True), pressures


def test_the_drive_indices_sum_to_one():
    """They are fractions of one withdrawal, so this is an identity. If it
    fails, a term has been left out of either the supply or the withdrawal --
    which is the only way material balance can be wrong without being
    obviously wrong."""
    pvt = oil()
    tank = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt)
    for fraction in (0.01, 0.03, 0.08, 0.15):
        np_oil = fraction * 50e6
        gor = 600.0 + fraction * 9000.0
        pressure = tank.pressure_at(np_oil, gor)
        indices = tank.drive_indices(pressure, np_oil, gor)
        assert indices["closure"] == pytest.approx(1.0, abs=1e-6), fraction


def test_an_undersaturated_reservoir_is_driven_partly_by_the_rock():
    """With no free gas anywhere, the only things pushing oil out are the
    oil's own expansion and the pore volume closing on it. Compaction is then
    a large share rather than a rounding term, and a model that drops Ef,w
    silently over-estimates the oil in place to compensate."""
    pvt = oil()
    tank = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt)
    np_oil = 0.004 * 50e6
    pressure = tank.pressure_at(np_oil, pvt.rsb_scf_stb)
    assert pressure > pvt.bubble_point_psia
    indices = tank.drive_indices(pressure, np_oil, pvt.rsb_scf_stb)
    assert indices["gas_cap"] == 0.0
    assert indices["compaction"] > 0.15, indices


def test_a_gas_cap_adds_energy_and_slows_the_pressure_fall():
    """Which is the reason a gas cap is protected from blowdown: the same
    production costs less pressure."""
    pvt = OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0,
                 rsb_scf_stb=600.0)
    saturated = pvt.bubble_point_psia
    without = OilTank(ooip_stb=50e6, initial_psia=saturated, pvt=pvt)
    with_cap = OilTank(ooip_stb=50e6, initial_psia=saturated, pvt=pvt,
                       gas_cap_ratio=0.4)
    np_oil, gor = 2.5e6, 1200.0
    assert with_cap.pressure_at(np_oil, gor) > without.pressure_at(np_oil, gor)
    indices = with_cap.drive_indices(with_cap.pressure_at(np_oil, gor),
                                     np_oil, gor)
    assert indices["gas_cap"] > 0.0


def test_an_undersaturated_reservoir_cannot_have_a_gas_cap():
    """There is no free gas at initial conditions, so the input is
    contradictory rather than merely unusual."""
    pvt = oil()
    with pytest.raises(ValueError, match="cannot have a gas cap"):
        OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt, gas_cap_ratio=0.3)


def test_an_aquifer_supports_pressure():
    pvt = oil()
    dry = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt)
    wet = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt,
                  water_influx_constant=800.0)
    np_oil, gor = 3e6, 900.0
    assert wet.pressure_at(np_oil, gor) > dry.pressure_at(np_oil, gor)


def test_a_synthetic_history_gives_its_own_oil_back():
    """The inverse of `pressure_at`, through an entirely different route --
    least squares on the Havlena-Odeh line rather than bisection on the
    balance. Agreement to six figures means both directions implement the
    same identity."""
    pvt = oil()
    tank = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt)
    fractions = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])
    cumulatives = fractions * 50e6
    gors = np.where(fractions <= 0.015, 600.0, 600.0 + (fractions - 0.015) * 12000.0)
    pressures = np.array([tank.pressure_at(float(n), float(g))
                          for n, g in zip(cumulatives, gors)])

    fit = fit_oil_balance(pressures, cumulatives, gors, pvt, 4000.0)
    assert fit.ooip_stb == pytest.approx(50e6, rel=1e-5)
    assert fit.rms_relative_residual < 1e-6
    assert fit.notes == []


def test_a_gas_cap_history_recovers_both_n_and_m():
    pvt = OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0,
                 rsb_scf_stb=600.0)
    tank = OilTank(ooip_stb=40e6, initial_psia=pvt.bubble_point_psia, pvt=pvt,
                   gas_cap_ratio=0.5)
    cumulatives = np.array([0.4, 0.8, 1.6, 2.4, 3.6, 5.0]) * 1e6
    gors = np.array([700.0, 820.0, 1000.0, 1200.0, 1500.0, 1900.0])
    pressures = np.array([tank.pressure_at(float(n), float(g))
                          for n, g in zip(cumulatives, gors)])

    fit = fit_oil_balance(pressures, cumulatives, gors, pvt,
                          pvt.bubble_point_psia, fit_gas_cap=True)
    assert fit.ooip_stb == pytest.approx(40e6, rel=1e-3)
    assert fit.gas_cap_ratio == pytest.approx(0.5, rel=1e-2)


def test_water_influx_shows_as_a_growing_miss_not_as_more_oil():
    """An aquifer is deliberately not fitted. Three unknowns on a handful of
    noisy surveys will always find a combination that fits, and the one it
    finds is a large aquifer offsetting a small N -- wrong, and reassuring.
    So the balance reports that it is drifting and leaves the aquifer to be
    supplied as a number someone can defend."""
    pvt = oil()
    wet = OilTank(ooip_stb=50e6, initial_psia=4000.0, pvt=pvt,
                  water_influx_constant=1500.0)
    fractions = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])
    cumulatives = fractions * 50e6
    gors = np.where(fractions <= 0.015, 600.0, 600.0 + (fractions - 0.015) * 12000.0)
    pressures = np.array([wet.pressure_at(float(n), float(g))
                          for n, g in zip(cumulatives, gors)])

    fit = fit_oil_balance(pressures, cumulatives, gors, pvt, 4000.0)
    assert any("water influx looks like" in note for note in fit.notes), fit.notes
