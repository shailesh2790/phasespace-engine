"""The grey-box correction, and the three controls that decide whether it ships.

A learned layer inside a physics model is easy to build and hard to
justify. The justification here is not that it fits -- anything with eight
hidden units fits -- but that it behaves correctly in three cases where
the right answer is known in advance:

  physics already right      it must not invent a correction
  a uniform bias             two constants already handle it, so the
                             network must not win
  a condition-dependent bias two constants cannot express it, so the
                             network must win, and by a lot

The middle case is the one usually left out, and it is the one that
matters: a learned layer that beats the simpler model only because it has
more parameters has learned the training set.

This engine has shipped that verdict before. The hydrate predictor was
built, measured against tuned physics on 594 real instances, found to add
nothing, and not shipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.ude import (
    FEATURES,
    MAX_CORRECTION,
    MIN_WELLS,
    Correction,
    Survey,
    features_from,
    fit,
    fit_constants,
    traverse,
)
from phasespace.sim.wells import (
    ProductionStream,
    Tubing,
    beggs_brill_gradient,
    pressure_traverse,
)


def a_survey(depth=8000.0, diameter=2.441, oil_rate=800.0, water=200.0,
             gas_rate=600_000.0, wellhead=200.0):
    return Survey(
        well="W", wellhead_psia=wellhead, bottomhole_psia=0.0,
        stream=ProductionStream(
            oil_rate_stb_d=oil_rate, water_rate_stb_d=water,
            gas_rate_scf_d=gas_rate,
            oil=LiquidProperties(api_gravity=35.0),
            gas=GasProperties(gravity=0.65)),
        tubing=Tubing(depth_ft=depth, inner_diameter_in=diameter))


# --------------------------------------------- the same physics underneath --


def test_the_identity_correction_reproduces_the_engine_exactly():
    """This module marches its own traverse so it can see each segment's
    contribution, which the engine's does not expose. That duplication is
    only safe if the two agree, so they are compared bit for bit -- without
    this test the grey box would be a second physics implementation
    quietly drifting from the first."""
    survey = a_survey()
    mine = traverse(survey, None, (1.0, 1.0))["bottomhole_psia"]
    engine = pressure_traverse(
        survey.wellhead_psia, survey.stream, survey.tubing).pressures_psia[-1]
    assert mine == engine


def test_constants_reproduce_the_engine_s_multipliers_exactly():
    survey = a_survey()
    mine = traverse(survey, None, (1.1, 0.9))["bottomhole_psia"]
    engine = pressure_traverse(
        survey.wellhead_psia, survey.stream, survey.tubing,
        gravity_multiplier=1.1, friction_multiplier=0.9).pressures_psia[-1]
    assert mine == engine


def test_a_fresh_correction_is_the_identity():
    """Fitting starts from the published correlation, not from a random
    perturbation of it. Otherwise the physics is only recovered if the
    optimiser happens to find its way back."""
    correction = Correction.zero()
    assert correction.is_identity
    survey = a_survey()
    assert (traverse(survey, correction, (1.0, 1.0))["bottomhole_psia"]
            == traverse(survey, None, (1.0, 1.0))["bottomhole_psia"])


# -------------------------------------------------------------- the bound --


def test_no_network_output_can_move_a_term_past_the_bound():
    """The physics stays the referee. A correction is allowed the same 20%
    a fitted constant is allowed, and no more, whatever the weights say."""
    rng = np.random.default_rng(0)
    correction = Correction.zero(rng)
    correction.w2 = rng.normal(0.0, 50.0, correction.w2.shape)
    correction.b2 = rng.normal(0.0, 50.0, correction.b2.shape)
    values = correction.apply(rng.normal(0.0, 3.0, (500, len(FEATURES))))
    assert np.all(np.abs(values - 1.0) <= MAX_CORRECTION + 1e-12)


def test_features_are_dimensionless():
    """A network fed raw velocities and pressures learns the units of the
    training field and transfers to nothing. Doubling every absolute scale
    while holding the dimensionless groups fixed must not move them."""
    survey = a_survey()
    result = beggs_brill_gradient(500.0, 150.0, survey.stream, survey.tubing)
    values = features_from(result, survey.tubing)
    assert values.shape == (len(FEATURES),)
    assert 0.0 <= values[0] <= 1.0           # holdup
    assert 0.0 <= values[2] <= 1.0           # gas fraction
    assert values[3] == pytest.approx(1.0)   # vertical


# ------------------------------------------------------------ persistence --


def test_a_correction_round_trips_through_a_dict():
    rng = np.random.default_rng(1)
    correction = Correction.zero(rng)
    correction.w2 = rng.normal(0.0, 1.0, correction.w2.shape)
    restored = Correction.from_dict(correction.to_dict())
    rows = rng.normal(0.0, 1.0, (20, len(FEATURES)))
    assert np.allclose(restored.apply(rows), correction.apply(rows))


def test_a_correction_trained_on_other_features_is_refused():
    """Weights are meaningless against a different feature definition, and
    applying them anyway would silently corrupt every pressure in the
    field."""
    raw = Correction.zero().to_dict()
    raw["features"] = ["holdup", "something_else"]
    with pytest.raises(ValueError, match="refusing to apply it"):
        Correction.from_dict(raw)


# ----------------------------------------------------------- the controls --


def biased_march(survey, bias, segments=40):
    """Truth, generated by marching with a planted bias."""
    tubing = survey.tubing
    dz = tubing.depth_ft / segments
    pressure = survey.wellhead_psia
    depth = 0.0
    for _ in range(segments):
        t_f = tubing.temperature_f_at(depth + 0.5 * dz)
        midpoint = pressure
        for stage in (0, 1):
            result = beggs_brill_gradient(
                pressure if stage == 0 else midpoint, t_f,
                survey.stream, tubing)
            gravity, friction = bias(features_from(result, tubing))
            elevation = result["elevation_psi_ft"]
            friction_term = result["friction_psi_ft"]
            parts = elevation + friction_term
            extra = (result["gradient_psi_ft"] / parts) if parts > 0.0 else 1.0
            gradient = (elevation * gravity + friction_term * friction) * extra
            if stage == 0:
                midpoint = max(pressure + gradient * dz * 0.5, 1.0)
            else:
                pressure = max(pressure + gradient * dz, 1.0)
        depth += dz
    return pressure


def a_field(bias, wells=8, per_well=3, noise=3.0, seed=7):
    rng = np.random.default_rng(seed)
    surveys = []
    for index in range(wells):
        tubing = Tubing(depth_ft=6000.0 + 800.0 * (index % 5),
                        inner_diameter_in=[1.995, 2.441, 2.992][index % 3])
        oil = LiquidProperties(api_gravity=28.0 + 4.0 * (index % 4))
        gas = GasProperties(gravity=0.62 + 0.03 * (index % 3))
        for k in range(per_well):
            rate = 250.0 * (k + 1) * (1.0 + 0.3 * (index % 3))
            gor = 300.0 + 200.0 * (index % 4)
            cut = 0.1 * (index % 5)
            survey = Survey(
                well="W%02d" % index, wellhead_psia=150.0 + 40.0 * k,
                bottomhole_psia=0.0,
                stream=ProductionStream(
                    oil_rate_stb_d=rate * (1.0 - cut),
                    water_rate_stb_d=rate * cut,
                    gas_rate_scf_d=rate * gor, oil=oil, gas=gas),
                tubing=tubing)
            survey.bottomhole_psia = (biased_march(survey, bias)
                                      + rng.normal(0.0, noise))
            surveys.append(survey)
    return surveys


HELD_OUT = ["W06", "W07"]


def test_it_does_not_invent_a_correction_when_the_physics_is_right():
    """The control that stops this being a machine for manufacturing
    confidence. With nothing to learn, the network must come back level
    with the constants, not ahead of them."""
    result = fit(a_field(lambda x: (1.0, 1.0)), test_wells=HELD_OUT,
                 iterations=60, seed=1)
    assert result.recommendation == "ship the constants"
    assert result.rms_ude < 4.0 * result.rms_constants


def test_a_uniform_bias_is_left_to_the_two_constants():
    """The control usually left out. Two constants express a uniform bias
    exactly, so the network has nothing to add and must not appear to add
    anything -- a learned layer that wins here is winning on parameter
    count."""
    result = fit(a_field(lambda x: (1.08, 1.0)), test_wells=HELD_OUT,
                 iterations=60, seed=1)
    assert result.constants[0] == pytest.approx(1.08, abs=0.02)
    assert result.recommendation == "ship the constants"


def test_a_condition_dependent_bias_is_recovered():
    """What the grey box is for. A bias that varies with holdup cannot be
    written as two constants, and the network must both beat them and get
    back down to the measurement noise."""
    field = a_field(lambda x: (1.0 + 0.15 * (x[0] - 0.5),
                               1.0 - 0.10 * (x[0] - 0.5)))
    result = fit(field, test_wells=HELD_OUT, iterations=120, seed=1)
    assert result.recommendation == "ship the correction"
    assert result.gain_over_constants > 0.3
    assert any("real field-specific bias" in note for note in result.notes)


# ----------------------------------------------------------- the refusals --


def test_too_few_wells_is_refused():
    """The whole claim rests on a held-out comparison between wells. Below
    a handful of wells that comparison cannot be made, and a number from
    it would be theatre."""
    field = a_field(lambda x: (1.0, 1.0), wells=MIN_WELLS - 1)
    with pytest.raises(ValueError, match="cannot support this"):
        fit(field)


def test_the_held_out_wells_are_absent_from_training():
    result = fit(a_field(lambda x: (1.0, 1.0)), test_wells=HELD_OUT,
                 iterations=20, seed=1)
    assert set(result.test_wells).isdisjoint(result.train_wells)
    assert set(result.test_wells) == set(HELD_OUT)


def test_the_constants_baseline_is_bounded_like_the_network():
    """Both sides of the comparison get the same licence, or it is not a
    comparison."""
    gravity, friction = fit_constants(a_field(lambda x: (1.5, 1.5))[:6])
    assert 1.0 - MAX_CORRECTION <= gravity <= 1.0 + MAX_CORRECTION
    assert 1.0 - MAX_CORRECTION <= friction <= 1.0 + MAX_CORRECTION


def test_fitting_is_reproducible():
    field = a_field(lambda x: (1.0, 1.0))
    runs = [fit(field, test_wells=HELD_OUT, iterations=20, seed=1).rms_ude
            for _ in range(2)]
    assert runs[0] == runs[1]
