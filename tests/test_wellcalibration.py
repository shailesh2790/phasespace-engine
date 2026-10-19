"""Tests for fitting a well model to its own tests.

Built around a synthetic well with known parameters, because "the fit
recovered the right answer" is only checkable when there is a right answer.
The model is always handed the *wrong* starting values and has to find its
way back; a test that starts the optimiser at the truth checks that it can
sit still.

Three things are being tested, and the second matters more than the first.

**It finds the parameters.** From a 30% starting error, on clean data, to
within a percent.

**It says when it cannot.** An under-determined fit still returns numbers
that look exactly like a well-posed one. The tests below check that a narrow
rate range is refused before fitting, that confounded parameters are reported
as confounded, and that a multiplier pinned to its bound is called a model
mismatch rather than a calibration.

**It never leaves the model worse.** The objective is the rate error only to
first order, so a fit can come out behind; when it does, the original
parameters are returned.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.wellcalibration import (
    CONFOUNDED_CORRELATION,
    FRICTION,
    GRAVITY,
    MAX_MULTIPLIER,
    MIN_MULTIPLIER,
    PRODUCTIVITY,
    WellTest,
    calibrate_well,
    identifiability,
)
from phasespace.sim.wells import CompositeIPR, Tubing, nodal_analysis

OIL = LiquidProperties(api_gravity=35.0, water_cut=0.25, temperature_r=660.0,
                       gas_gravity=0.7, bubble_point_psia=2700.0)
GAS = GasProperties(gravity=0.7, temperature_r=660.0)
TUBING = Tubing(depth_ft=8000.0, inner_diameter_in=2.992)

TRUE_PI = 2.6
TRUE_GRAVITY = 1.12
TRUE_FRICTION = 0.92
ASSUMED_PI = 2.0          # a guess from an offset well, 23% low

WIDE = [150.0, 300.0, 500.0, 750.0, 1000.0]
NARROW = [280.0, 300.0, 320.0, 340.0]


def ipr(productivity_index=ASSUMED_PI):
    return CompositeIPR(reservoir_pressure_psia=3500.0,
                        bubble_point_psia=2700.0,
                        productivity_index=productivity_index)


def make_tests(wellheads, noise=0.0, seed=0, bottomhole=False):
    """Well tests generated from the true well."""
    rng = np.random.default_rng(seed)
    tests = []
    for wellhead in wellheads:
        solved = nodal_analysis(ipr(TRUE_PI), TUBING, wellhead, 600.0, 0.25,
                                OIL, GAS, gravity_multiplier=TRUE_GRAVITY,
                                friction_multiplier=TRUE_FRICTION)
        if not solved.converged:
            continue
        rate = solved.liquid_rate_stb_d
        if noise:
            rate *= float(np.exp(rng.normal(0.0, noise)))
        tests.append(WellTest(
            rate_stb_d=rate, wellhead_psia=wellhead, water_cut=0.25,
            gas_oil_ratio_scf_stb=600.0,
            bottomhole_psia=(solved.bottomhole_pressure_psia if bottomhole
                             else 0.0),
            label="%.0f psi" % wellhead))
    return tests


# ---------------------------------------------------------- it finds them --


def test_a_clean_fit_recovers_the_true_parameters():
    """From a 23% error in the productivity index and no knowledge of either
    multiplier, on noiseless tests across a 100% rate range."""
    result = calibrate_well(ipr(), TUBING, make_tests(WIDE), OIL, GAS)

    assert result.parameters["productivity_index"] == pytest.approx(
        TRUE_PI, rel=0.03)
    assert result.parameters[GRAVITY] == pytest.approx(TRUE_GRAVITY, rel=0.02)
    assert result.parameters[FRICTION] == pytest.approx(TRUE_FRICTION, rel=0.05)
    assert result.rms_after < 0.01
    assert result.improved


def test_the_starting_model_really_is_wrong():
    """Otherwise the test above proves nothing: a fit that changes nothing
    would pass it if the starting point were already right."""
    result = calibrate_well(ipr(), TUBING, make_tests(WIDE), OIL, GAS)
    assert result.rms_before > 0.15, (
        "the starting model is already within %.1f%% -- there is nothing to "
        "recover" % (result.rms_before * 100.0))
    assert result.improvement > 0.9


def test_the_liquid_rate_is_not_fed_to_an_oil_rate_inflow():
    """Regression. A well test is reported as total liquid; the IPR returns
    oil. Passing one for the other is a factor of (1 - water cut), and it
    does not announce itself -- the optimiser scales the productivity index
    up to absorb it and reports a confident, wrong deliverability. At 25%
    water that was a 33% error in the fitted PI."""
    wet = calibrate_well(ipr(), TUBING, make_tests(WIDE), OIL, GAS)
    assert wet.parameters["productivity_index"] == pytest.approx(
        TRUE_PI, rel=0.05)
    # The bug inflated PI by roughly 1/(1 - water cut).
    assert wet.parameters["productivity_index"] < TRUE_PI / (1.0 - 0.25) * 0.9


def test_calibration_improves_the_fit_under_realistic_noise():
    """Four percent on each measured rate, which is a good well test."""
    for seed in range(4):
        result = calibrate_well(
            ipr(), TUBING, make_tests(WIDE, noise=0.04, seed=seed), OIL, GAS)
        assert result.improved, "seed %d made the model worse" % seed
        assert result.rms_after < 0.10, (seed, result.rms_after)


# -------------------------------------------------- it says when it cannot --


def test_a_narrow_rate_range_is_refused_before_fitting():
    """Gravity dominates at low rate and friction at high rate, so what
    separates them is rate. At one operating point they cannot be told
    apart, whatever the optimiser reports."""
    report = identifiability(make_tests(NARROW))
    assert not report.supported
    assert report.rate_spread < 0.25
    assert any("cannot be told apart" in reason for reason in report.reasons)


def test_an_exact_fit_through_as_many_points_as_parameters_is_refused():
    """Three tests and three parameters is interpolation, not calibration:
    the residual is zero and the model has learned nothing."""
    report = identifiability(make_tests(WIDE)[:3])
    assert report.residuals == 3
    assert report.degrees_of_freedom == 0
    assert not report.supported
    assert any("learned nothing" in reason for reason in report.reasons)


def test_fewer_parameters_can_be_supported_by_the_same_tests():
    """The refusal is about the ratio, not the data being bad. The same
    three tests support fitting one parameter."""
    tests = make_tests(WIDE)[:3]
    assert not identifiability(tests).supported
    assert identifiability(tests, [PRODUCTIVITY]).supported


def test_confounded_parameters_are_reported_as_confounded():
    """Without a bottomhole measurement the inflow and the tubing only ever
    appear as a difference, so pairs that preserve it fit equally well. The
    fit still matches the tests; the individual values do not mean much."""
    result = calibrate_well(ipr(), TUBING, make_tests(WIDE), OIL, GAS)
    assert result.confounded, result.correlations
    assert not result.trustworthy
    assert max(abs(v) for v in result.correlations.values()) > CONFOUNDED_CORRELATION


def test_a_bottomhole_measurement_separates_the_parameters():
    """The module's central claim, as a measurement rather than an assertion.

    A flowing gradient survey pins the tubing side directly, so the two
    sides stop competing to explain one number. Checked two ways: the
    parameter correlation falls, and -- the part that matters -- the fitted
    values stop moving when the data is perturbed.
    """
    without, with_survey = [], []
    for seed in range(4):
        plain = calibrate_well(
            ipr(), TUBING, make_tests(WIDE, noise=0.04, seed=seed), OIL, GAS)
        surveyed = calibrate_well(
            ipr(), TUBING,
            make_tests(WIDE, noise=0.04, seed=seed, bottomhole=True),
            OIL, GAS)
        without.append(plain.parameters["productivity_index"])
        with_survey.append(surveyed.parameters["productivity_index"])

    assert np.std(with_survey) < 0.5 * np.std(without), (
        "survey scatter %.3f against %.3f without"
        % (np.std(with_survey), np.std(without)))
    assert abs(np.mean(with_survey) - TRUE_PI) < abs(np.mean(without) - TRUE_PI)

    surveyed = calibrate_well(
        ipr(), TUBING, make_tests(WIDE, bottomhole=True), OIL, GAS)
    plain = calibrate_well(ipr(), TUBING, make_tests(WIDE), OIL, GAS)
    worst = lambda r: max(abs(v) for v in r.correlations.values())
    assert worst(surveyed) < worst(plain)


def test_a_multiplier_at_its_bound_is_called_a_mismatch():
    """A correlation that needs scaling past 20% is the wrong correlation, or
    an input is wrong. Forced here by fitting against tests from a well whose
    tubing is far heavier than any multiplier can reach."""
    heavy = []
    for wellhead in WIDE:
        solved = nodal_analysis(ipr(TRUE_PI), TUBING, wellhead, 600.0, 0.25,
                                OIL, GAS, gravity_multiplier=1.6,
                                friction_multiplier=1.0)
        if solved.converged:
            heavy.append(WellTest(
                rate_stb_d=solved.liquid_rate_stb_d, wellhead_psia=wellhead,
                water_cut=0.25, gas_oil_ratio_scf_stb=600.0))

    result = calibrate_well(ipr(TRUE_PI), TUBING, heavy, OIL, GAS)
    assert GRAVITY in result.at_bound, result.parameters
    assert result.parameters[GRAVITY] == pytest.approx(MAX_MULTIPLIER, rel=1e-3)
    assert not result.trustworthy
    assert any("physics model mismatch" in note for note in result.notes)


def test_the_fit_never_leaves_the_model_worse():
    """The objective is the rate error only to first order -- the gap-to-rate
    conversion is a local slope -- so a fit can come out behind. When it
    does, the starting parameters come back rather than a worse model with a
    confident label on it."""
    for seed in range(6):
        tests = make_tests(WIDE, noise=0.08, seed=seed)
        result = calibrate_well(ipr(), TUBING, tests, OIL, GAS)
        assert result.rms_after <= result.rms_before + 1e-12, (
            "seed %d: %.3f -> %.3f" % (seed, result.rms_before,
                                       result.rms_after))


def test_a_pressure_gap_is_not_assumed_to_be_a_rate_error():
    """Regression for the same thing from the other side.

    Minimising the gap between the two curves while scoring rate let the
    optimiser trade a large rate error for a small pressure one: near a
    tangent intersection a few psi moves the operating point by hundreds of
    barrels. On one noise seed the rate error went from 28% to 51% and the
    fit called itself converged.
    """
    tests = make_tests(WIDE, noise=0.04, seed=3)
    result = calibrate_well(ipr(), TUBING, tests, OIL, GAS)
    assert result.rms_after < 0.15, result.rms_after
    assert result.improved


# ------------------------------------------------------------- the inputs --


def test_the_parameters_reach_the_model_they_calibrate():
    """A calibration whose numbers do not change the forward model is
    decoration. Checked by solving the well with and without them."""
    result = calibrate_well(ipr(), TUBING, make_tests(WIDE), OIL, GAS)
    raw = nodal_analysis(ipr(), TUBING, 400.0, 600.0, 0.25, OIL, GAS)
    tuned = nodal_analysis(
        ipr(result.parameters["productivity_index"]), TUBING, 400.0, 600.0,
        0.25, OIL, GAS, gravity_multiplier=result.parameters[GRAVITY],
        friction_multiplier=result.parameters[FRICTION])
    assert abs(tuned.liquid_rate_stb_d - raw.liquid_rate_stb_d) > 1.0


def test_the_callers_ipr_is_not_mutated_by_fitting():
    """Trial values from whichever iteration happened to run last must not
    leak back into the caller's object."""
    original = ipr()
    before = original.productivity_index
    calibrate_well(original, TUBING, make_tests(WIDE), OIL, GAS)
    assert original.productivity_index == before


def test_nonsense_tests_are_refused():
    with pytest.raises(ValueError, match="not an operating point"):
        WellTest(rate_stb_d=0.0, wellhead_psia=300.0)
    with pytest.raises(ValueError, match="wellhead pressure"):
        WellTest(rate_stb_d=500.0, wellhead_psia=0.0)
    with pytest.raises(ValueError, match="water cut"):
        WellTest(rate_stb_d=500.0, wellhead_psia=300.0, water_cut=1.0)
    with pytest.raises(ValueError, match="no well tests"):
        identifiability([])
    with pytest.raises(ValueError, match="unknown parameter"):
        identifiability(make_tests(WIDE), ["skin"])


def test_the_multipliers_are_bounded_where_the_physics_says():
    """A band wide enough for a correlation that is merely imperfect and
    narrow enough that a structural error cannot hide inside it."""
    assert MIN_MULTIPLIER == pytest.approx(0.80)
    assert MAX_MULTIPLIER == pytest.approx(1.20)
    result = calibrate_well(ipr(), TUBING, make_tests(WIDE), OIL, GAS)
    for name in (GRAVITY, FRICTION):
        assert MIN_MULTIPLIER <= result.parameters[name] <= MAX_MULTIPLIER
