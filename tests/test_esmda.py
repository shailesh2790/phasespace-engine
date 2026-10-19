"""ES-MDA, checked against the one case where the right answer is known.

On a linear forward model with Gaussian priors and Gaussian noise the
posterior is available in closed form. ES-MDA must reproduce it, and that
is the test that would catch almost any error worth catching here -- a
missing (n-1) in a covariance, an unperturbed observation vector, a wrong
inflation, a transposed gain. Most of the rest of this file is about the
failure modes the closed form cannot show: bounds, collapse, and
parameters the data says nothing about.

Emerick & Reynolds, Computers & Geosciences 55 (2013) 3-15.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.esmda import (
    COLLAPSE_FRACTION,
    MIN_ENSEMBLE,
    Parameter,
    esmda,
)

DESIGN = np.array([[1.0, 0.5], [0.3, 1.0], [0.8, 0.8], [1.2, 0.2], [0.1, 1.4]])
TRUTH = np.array([2.0, 1.0])
NOISE = 0.05
PRIOR_MEAN = np.array([1.5, 1.5])
PRIOR_SD = np.array([0.8, 0.8])


def observations(seed=1):
    rng = np.random.default_rng(seed)
    return DESIGN @ TRUTH + rng.normal(0.0, NOISE, DESIGN.shape[0])


def analytic_posterior(observed):
    """The closed-form Gaussian posterior this must reproduce."""
    prior_cov = np.diag(PRIOR_SD ** 2)
    data_cov = np.diag(np.full(DESIGN.shape[0], NOISE ** 2))
    gain = prior_cov @ DESIGN.T @ np.linalg.inv(
        DESIGN @ prior_cov @ DESIGN.T + data_cov)
    mean = PRIOR_MEAN + gain @ (observed - DESIGN @ PRIOR_MEAN)
    cov = prior_cov - gain @ DESIGN @ prior_cov
    return mean, np.sqrt(np.diag(cov))


def linear_parameters():
    # Bounds set far outside the posterior so the transform is effectively
    # the identity here: this test is about the update, not the mapping.
    return [Parameter("a", PRIOR_MEAN[0], PRIOR_SD[0], lower=-50.0, upper=50.0),
            Parameter("b", PRIOR_MEAN[1], PRIOR_SD[1], lower=-50.0, upper=50.0)]


def linear_forward(values):
    return DESIGN @ np.array([values["a"], values["b"]])


# ------------------------------------------------- against the closed form --


@pytest.mark.parametrize("assimilations", [1, 4, 8])
def test_the_posterior_matches_the_analytic_one(assimilations):
    """The check that makes the rest believable. On a linear-Gaussian
    problem ES-MDA is not an approximation -- it must land on the exact
    Bayesian posterior, in spread as well as in mean."""
    observed = observations()
    mean, spread = analytic_posterior(observed)
    result = esmda(linear_parameters(), linear_forward, observed, NOISE,
                   members=4000, assimilations=assimilations, seed=3)

    assert result.mean["a"] == pytest.approx(mean[0], abs=0.01)
    assert result.mean["b"] == pytest.approx(mean[1], abs=0.01)
    assert result.sd["a"] == pytest.approx(spread[0], rel=0.08)
    assert result.sd["b"] == pytest.approx(spread[1], rel=0.08)


def test_the_number_of_assimilations_does_not_move_the_answer_on_a_linear_model():
    """Tempering buys accuracy on a *nonlinear* posterior. On a linear one
    it must change nothing, and a drift with N_a would mean the inflations
    are not being applied as Emerick & Reynolds specify."""
    observed = observations()
    runs = [esmda(linear_parameters(), linear_forward, observed, NOISE,
                  members=4000, assimilations=n, seed=3).mean["a"]
            for n in (1, 4, 8)]
    assert max(runs) - min(runs) < 0.01


def test_inflations_that_do_not_sum_correctly_are_refused():
    """sum(1/alpha) = 1 is what makes the tempered steps add to one full
    assimilation. Violating it does not fail loudly -- it quietly
    over-weights the data and returns a posterior that is too narrow."""
    with pytest.raises(ValueError, match=r"sum\(1/alpha\) = 1"):
        esmda(linear_parameters(), linear_forward, observations(), NOISE,
              members=50, inflations=[2.0, 2.0, 2.0])


def test_equal_inflations_are_accepted():
    esmda(linear_parameters(), linear_forward, observations(), NOISE,
          members=50, inflations=[4.0, 4.0, 4.0, 4.0])


# -------------------------------------------------------------- the bounds --


def test_a_bounded_parameter_never_reaches_its_bound():
    """Bounds are enforced by transforming to an unbounded space, not by
    clipping. Clipping piles probability mass exactly on the limit, and a
    posterior with a spike at the bound reads as confidence when it is the
    opposite."""
    observed = observations()
    result = esmda(
        [Parameter("a", 1.5, 0.8, lower=0.5, upper=2.5),
         Parameter("b", 1.5, 0.8, lower=0.5, upper=2.5)],
        linear_forward, observed, NOISE, members=2000, seed=5)
    for name in ("a", "b"):
        column = result.posterior[:, result.names.index(name)]
        assert column.min() > 0.5
        assert column.max() < 2.5
        # No pile-up at either edge.
        assert np.mean(column < 0.51) < 0.001
        assert np.mean(column > 2.49) < 0.001


def test_a_one_sided_bound_keeps_a_positive_parameter_positive():
    result = esmda(
        [Parameter("a", 1.5, 0.8, lower=0.0),
         Parameter("b", 1.5, 0.8, lower=0.0)],
        linear_forward, observations(), NOISE, members=1000, seed=7)
    assert result.posterior.min() > 0.0


def test_a_prior_mean_outside_its_bounds_is_refused():
    with pytest.raises(ValueError, match="outside its bounds"):
        Parameter("a", 3.0, 0.5, lower=0.0, upper=2.0)


def test_a_parameter_with_no_prior_spread_is_refused():
    with pytest.raises(ValueError, match="positive prior spread"):
        Parameter("a", 1.0, 0.0)


# ------------------------------------------------------- what was not learned --


def test_a_parameter_the_data_cannot_see_comes_back_as_its_prior():
    """The honest outcome, and the one a point estimate hides. A parameter
    the observations do not constrain must return its prior spread and say
    so, not a confident number."""
    def forward(values):
        # "b" does not appear: nothing can be learned about it.
        return np.full(DESIGN.shape[0], values["a"])

    result = esmda(linear_parameters(), forward,
                   np.full(DESIGN.shape[0], 2.0), NOISE,
                   members=500, seed=11)
    assert result.reduction["b"] < 0.1
    assert result.reduction["a"] > 0.5
    assert any("came back as its prior" in note for note in result.notes)


def test_collapse_is_reported_rather_than_read_as_precision():
    result = esmda(linear_parameters(), linear_forward, observations(),
                   NOISE, members=MIN_ENSEMBLE, assimilations=8, seed=13)
    if result.collapsed:
        assert any("collapsed" in note for note in result.notes)
        assert any("ensemble too small" in note for note in result.notes)
    else:
        spreads = result.posterior.std(axis=0, ddof=1)
        priors = result.prior.std(axis=0, ddof=1)
        assert (spreads / priors > COLLAPSE_FRACTION).all()


def test_a_posterior_that_still_misses_the_data_says_so():
    """Matching badly is a finding about the model, not a silent result."""
    observed = observations() + 50.0        # unreachable by any parameters
    result = esmda(linear_parameters(), linear_forward, observed, NOISE,
                   members=200, seed=17)
    assert result.data_misfit > 3.0
    assert any("misses the data" in note for note in result.notes)


def test_matching_closer_than_the_measurements_is_also_flagged():
    result = esmda(linear_parameters(), linear_forward, observations(),
                   observation_sd=5.0, members=200, seed=19)
    assert any("Suspect the error estimate" in note for note in result.notes)


# ---------------------------------------------------------- the refusals --


def test_the_measurement_error_is_required_to_be_positive():
    with pytest.raises(ValueError, match="positive measurement error"):
        esmda(linear_parameters(), linear_forward, observations(), 0.0,
              members=50)


def test_a_mismatched_error_vector_is_refused():
    with pytest.raises(ValueError, match="one value or one per observation"):
        esmda(linear_parameters(), linear_forward, observations(),
              [0.05, 0.05], members=50)


def test_too_small_an_ensemble_is_refused():
    with pytest.raises(ValueError, match="cannot support a usable sample"):
        esmda(linear_parameters(), linear_forward, observations(), NOISE,
              members=5)


def test_a_forward_returning_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="but 5 observations were given"):
        esmda(linear_parameters(), lambda v: [1.0, 2.0],
              observations(), NOISE, members=50)


def test_a_forward_that_fails_everywhere_is_refused_with_the_reason():
    """Saying "the prior is too wide" is actionable; a LinAlgError from
    inside a covariance is not."""
    with pytest.raises(ValueError, match="too wide rather than"):
        esmda(linear_parameters(),
              lambda v: np.full(DESIGN.shape[0], np.nan),
              observations(), NOISE, members=50)


def test_a_few_failed_members_are_tolerated_and_recorded():
    calls = {"n": 0}

    def flaky(values):
        calls["n"] += 1
        if calls["n"] % 11 == 0:
            return np.full(DESIGN.shape[0], np.nan)
        return linear_forward(values)

    result = esmda(linear_parameters(), flaky, observations(), NOISE,
                   members=200, seed=23)
    assert any("failed to run" in note for note in result.notes)
    assert np.isfinite(result.posterior).all()


# ------------------------------------------------------- the reproducibility --


def test_the_same_seed_gives_the_same_posterior():
    """The engine's provenance bundle requires it: a calibration quoted in
    a report has to be reproducible from its inputs."""
    runs = [esmda(linear_parameters(), linear_forward, observations(), NOISE,
                  members=100, seed=29).posterior for _ in range(2)]
    assert np.array_equal(runs[0], runs[1])


def test_different_seeds_disagree_by_less_than_the_posterior_spread():
    """Sampling error must be small against the thing being measured, or
    the number quoted is the seed rather than the well."""
    observed = observations()
    means = [esmda(linear_parameters(), linear_forward, observed, NOISE,
                   members=500, seed=seed).mean["a"] for seed in range(5)]
    _mean, spread = analytic_posterior(observed)
    assert np.std(means) < 0.3 * spread[0]


# -------------------------------------------------------------- on a well --


def test_it_runs_on_the_real_well_model_and_reports_what_was_learned():
    """Small ensemble, one assimilation: the nodal solve is the cost here,
    so this checks the wiring rather than the statistics."""
    from phasespace.sim.esmda import calibrate_well_ensemble
    from phasespace.sim.fluids import GasProperties, LiquidProperties
    from phasespace.sim.wellcalibration import WellTest
    from phasespace.sim.wells import CompositeIPR, Tubing

    tests = [WellTest(rate_stb_d=r, wellhead_psia=p)
             for r, p in [(1970, 120), (810, 480)]]
    result = calibrate_well_ensemble(
        CompositeIPR(3500.0, 2700.0, 2.0),
        Tubing(depth_ft=8000.0, inner_diameter_in=2.441),
        tests, LiquidProperties(api_gravity=35.0), GasProperties(gravity=0.65),
        members=MIN_ENSEMBLE, assimilations=1, seed=0)

    assert set(result.names) == {"productivity_index", "gravity_multiplier",
                                 "friction_multiplier"}
    bounds = result.interval(0.90)
    for low, high in bounds.values():
        assert high >= low
    assert 0.0 <= min(result.reduction.values()) <= 1.0
