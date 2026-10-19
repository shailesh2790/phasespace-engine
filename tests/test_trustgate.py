"""The gate, and the three places its obvious implementation is wrong.

A gate that passes everything is decoration, and one that fails everything
gets switched off in a week. Most of these tests are about the boundary:
what it must catch, and what it must not fire on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.trustgate import (
    CoverageMonitor,
    RefereeLog,
    TrainingEnvelope,
    TrustGate,
    check_conservation,
    check_range,
    check_sanity,
    check_spread,
)


def training_set(seed=0, n=600):
    """Rate and water cut, correlated the way a real field's are: the high
    rates are the dry wells."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(200.0, 2000.0, n)
    cut = np.clip(0.6 - rate / 4000.0 + rng.normal(0.0, 0.04, n), 0.0, 1.0)
    return np.column_stack([rate, cut])


def envelope(seed=0):
    return TrainingEnvelope.fit(training_set(seed), names=["rate", "water_cut"])


# ------------------------------------------------------------- the range --


def test_a_query_in_the_middle_of_training_passes():
    checks = check_range(envelope(), [1100.0, 0.33])
    assert all(c.passed for c in checks), [c.detail for c in checks]


def test_a_feature_outside_its_training_range_is_caught():
    """The easy case, and the one a joint distance can miss: Mahalanobis
    averages over directions, so one feature far out can be diluted."""
    checks = check_range(envelope(), [9000.0, 0.3])
    feature_check = next(c for c in checks if c.name == "range.features")
    assert not feature_check.passed
    assert "rate" in feature_check.detail


def test_an_unseen_combination_inside_every_range_is_caught():
    """The case a per-feature range check cannot see. High rate and high
    water cut are each ordinary here; together they never occur, because
    the training set pairs high rate with low cut."""
    checks = check_range(envelope(), [1900.0, 0.55])
    feature_check = next(c for c in checks if c.name == "range.features")
    joint_check = next(c for c in checks if c.name == "range.joint")
    assert feature_check.passed, "each feature really is inside its range"
    assert not joint_check.passed, "the combination is not one the model saw"


def test_the_joint_limit_comes_from_the_training_set_not_a_table():
    """Scored against the training set's own distance distribution, because
    the inputs are not Gaussian and a chi-squared quantile would be
    answering a different question. So about 1% of training rows should
    themselves fail, by construction."""
    data = training_set()
    env = TrainingEnvelope.fit(data, names=["rate", "water_cut"])
    failures = sum(1 for row in data if env.distance(row) > env.distance_limit)
    assert 0.003 <= failures / len(data) <= 0.02


def test_too_few_rows_for_a_covariance_is_refused():
    with pytest.raises(ValueError, match="not estimable"):
        TrainingEnvelope.fit([[1.0, 2.0], [2.0, 3.0]])


def test_collinear_features_do_not_blow_up_the_distance():
    """Real features are correlated, sometimes exactly. A plain inverse of
    a singular covariance returns infinities and the gate then rejects
    everything, which looks like a strict gate and is a broken one."""
    rng = np.random.default_rng(1)
    rate = rng.uniform(100.0, 900.0, 400)
    data = np.column_stack([rate, rate * 2.0, rng.normal(0, 1, 400)])
    env = TrainingEnvelope.fit(data)
    assert np.isfinite(env.distance([500.0, 1000.0, 0.0]))
    assert np.isfinite(env.distance_limit)


# ------------------------------------------------------------ the physics --


def test_an_impossible_answer_is_caught():
    checks = check_sanity({"bhp_psia": -40.0},
                          {"bhp_psia": (0.0, 20000.0)})
    assert not checks[0].passed
    assert "outside" in checks[0].detail


def test_conservation_is_relative_to_throughput():
    """10 scf/d of imbalance is nothing on a trunk and is the whole flow on
    a stripper, so an absolute tolerance is wrong at one end or the other."""
    assert check_conservation(50_000_000.0, 50_000_010.0).passed
    assert not check_conservation(100.0, 110.0).passed


def test_a_perfectly_balanced_answer_passes():
    assert check_conservation(1234.5, 1234.5).passed


# ----------------------------------------------------------- the ensemble --


def test_disagreeing_members_fail():
    assert not check_spread([100.0, 140.0, 90.0, 130.0, 80.0], tolerance=5.0).passed


def test_agreeing_members_pass():
    assert check_spread([100.0, 100.4, 99.7, 100.1, 99.9], tolerance=5.0).passed


def test_a_single_member_cannot_show_agreement():
    """One model has no spread, and reporting zero spread as agreement is
    the most flattering possible lie about an ensemble of one."""
    result = check_spread([100.0], tolerance=5.0)
    assert not result.passed
    assert "cannot show disagreement" in result.detail


# ----------------------------------------------------------- the coverage --


def test_a_well_behaved_model_does_not_trip_the_coverage_check():
    """The failure the PRD's fixed +/-2% band would cause. At n=50 and
    p=0.9 the sampling standard deviation of realised coverage is 4.2%, so
    a 2% band fires constantly on a model doing exactly what it promised."""
    rng = np.random.default_rng(3)
    tripped = 0
    for trial in range(60):
        monitor = CoverageMonitor(nominal=0.90, window=50)
        for _ in range(50):
            inside = rng.random() < 0.90
            monitor.record(0.0, -1.0, 1.0 if inside else -0.5)
        if not monitor.check().passed:
            tripped += 1
    assert tripped <= 6, (
        "a correctly calibrated model tripped %d times in 60; the band is "
        "ignoring sampling error" % tripped)


def test_real_drift_is_caught():
    monitor = CoverageMonitor(nominal=0.90, window=200)
    rng = np.random.default_rng(5)
    for _ in range(200):
        inside = rng.random() < 0.70        # badly drifted
        monitor.record(0.0, -1.0, 1.0 if inside else -0.5)
    result = monitor.check()
    assert not result.passed
    assert "drifted" in result.detail


def test_coverage_is_not_judged_on_too_few_results():
    monitor = CoverageMonitor(nominal=0.90, window=200)
    for _ in range(5):
        monitor.record(0.0, -1.0, -0.5)     # every one a miss
    result = monitor.check()
    assert result.passed
    assert "cannot be judged" in result.detail


def test_over_covering_is_not_a_failure():
    """Covering more than promised is conservative, not broken. A
    two-sided band would demand retraining for being too safe."""
    monitor = CoverageMonitor(nominal=0.90, window=100)
    for _ in range(100):
        monitor.record(0.0, -1.0, 1.0)
    assert monitor.check().passed


def test_the_window_rolls():
    monitor = CoverageMonitor(nominal=0.90, window=50)
    for _ in range(120):
        monitor.record(0.0, -1.0, 1.0)
    assert monitor.n == 50


# ------------------------------------------------------------- the referee --


def test_the_spot_check_samples_about_the_stated_fraction():
    log = RefereeLog(fraction=0.02, seed=7)
    sampled = sum(1 for _ in range(20_000) if log.should_check())
    assert 300 <= sampled <= 500


def test_the_sampler_is_reproducible():
    """An auditor replaying a run from the log has to re-solve the same
    queries, or the check cannot be repeated."""
    one, two = RefereeLog(seed=11), RefereeLog(seed=11)
    first = [one.should_check() for _ in range(500)]
    assert first == [two.should_check() for _ in range(500)]
    # And a different seed must give a different sequence, or the seed is
    # not doing anything and "reproducible" is just "constant".
    assert first != [RefereeLog(seed=12).should_check() for _ in range(500)]


def test_a_model_the_engine_disagrees_with_is_demoted():
    log = RefereeLog(tolerance=0.02)
    for _ in range(20):
        log.record(learned=110.0, engine=100.0)
    result = log.check()
    assert not result.passed
    assert "advisory" in result.detail


def test_the_referee_withholds_judgement_on_too_few_checks():
    log = RefereeLog()
    log.record(learned=200.0, engine=100.0)
    assert log.check().passed


# ---------------------------------------------------------------- the gate --


def gate() -> TrustGate:
    return TrustGate(
        model="well-surrogate", version="1.0.0", envelope=envelope(),
        bounds={"bhp_psia": (0.0, 20000.0)}, spread_tolerance=20.0)


def test_a_good_query_passes_the_whole_gate():
    verdict = gate().evaluate(
        [1100.0, 0.33], {"bhp_psia": 2400.0},
        ensemble=[2400.0, 2402.0, 2398.0, 2401.0, 2399.0],
        balance=(1000.0, 1000.2))
    assert verdict.passed, verdict.describe()
    assert not verdict.answer_with_engine


def test_one_failure_sends_the_query_to_the_engine():
    verdict = gate().evaluate(
        [9000.0, 0.33], {"bhp_psia": 2400.0},
        ensemble=[2400.0] * 5, balance=(1000.0, 1000.0))
    assert verdict.answer_with_engine
    assert any("outside the training range" in r for r in verdict.reasons)
    assert "The engine answers instead" in verdict.describe()


def test_the_verdict_names_the_model_and_every_failure():
    verdict = gate().evaluate(
        [9000.0, 0.9], {"bhp_psia": -5.0},
        ensemble=[2400.0, 2500.0, 2300.0, 2600.0, 2200.0],
        balance=(1000.0, 1200.0))
    assert verdict.model == "well-surrogate"
    names = {c.name for c in verdict.failed}
    assert "range.features" in names
    assert "sanity.bhp_psia" in names
    assert "conservation" in names
    assert "spread" in names
