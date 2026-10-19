"""Tests for the learned layer of card diagnosis.

The tests that matter here are not about accuracy -- accuracy is measured by
the training script against a held-out split, and asserting a number in a test
would just pin whatever the last run produced. These check the things that
silently destroy a learned layer's meaning: leakage across the split, a feature
vector that has drifted away from the weights, and a softmax that overflows on
exactly the predictions it is most confident about.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.rodpump.classifier import CardClassifier, evaluate, train_classifier
from phasespace.rodpump.diagnosis import FEATURE_NAMES
from phasespace.rodpump.synthetic import FAULTS, PumpSpec, downhole_card
from phasespace.rodpump.training import (
    balance_classes,
    build_dataset,
    sample_well,
    split_by_well,
)

SPEC = PumpSpec(plunger_diameter_in=1.5, net_lift_ft=4000.0,
                fluid_specific_gravity=0.9, stroke_in=100.0, strokes_per_minute=8.0)


@pytest.fixture(scope="module")
def small_dataset():
    return build_dataset(wells=14, severities=5, seed=3)


@pytest.fixture(scope="module")
def trained(small_dataset):
    # scikit-learn is needed to *train* the classifier and deliberately not
    # to run one -- `test_sklearn_stays_out_of_the_inference_path` below
    # asserts exactly that, and still runs without it. So a contributor who
    # installed only the engine's own dependencies gets a skip here rather
    # than a failure they did not cause.
    pytest.importorskip("sklearn",
                        reason="only training the classifier needs scikit-learn")
    train, test = split_by_well(small_dataset, holdout_fraction=0.3, seed=3)
    balanced = balance_classes(train, seed=3)
    model = train_classifier(balanced["features"], balanced["labels"],
                             hidden=(48, 24), seed=3, max_iter=400)
    return model, test


# ------------------------------------------------------------- dataset -----


def test_every_fault_is_represented(small_dataset):
    assert set(small_dataset["labels"].tolist()) == set(FAULTS)


def test_features_are_finite(small_dataset):
    """One NaN anywhere in a feature matrix poisons standardisation and turns
    every weight into nan, which shows up much later as a model that predicts
    one class."""
    assert np.all(np.isfinite(small_dataset["features"]))


def test_feature_width_matches_the_names(small_dataset):
    assert small_dataset["features"].shape[1] == len(FEATURE_NAMES)


def test_classes_are_balanced_after_balancing(small_dataset):
    """A full pump has no severity axis, so it arrives once per well against
    several times for every fault. Left alone the model pays almost nothing for
    getting the healthy case wrong -- and duly did, scoring below the rule layer
    on the one class where a false positive sends a crew to a well that does
    not need touching."""
    train, _ = split_by_well(small_dataset, holdout_fraction=0.3, seed=3)
    balanced = balance_classes(train, seed=3)
    _, counts = np.unique(balanced["labels"], return_counts=True)
    assert counts.max() / counts.min() < 1.35, counts


def test_split_shares_no_well(small_dataset):
    """The leakage test. Cards from one well at adjacent severities are near
    duplicates; a random row split puts each side of a pair across the split
    and reports memorisation as generalisation."""
    train, test = split_by_well(small_dataset, holdout_fraction=0.3, seed=3)
    assert set(train["groups"].tolist()).isdisjoint(set(test["groups"].tolist()))
    assert test["features"].shape[0] > 0
    assert train["features"].shape[0] > test["features"].shape[0]


def test_balancing_does_not_touch_the_held_out_split(small_dataset):
    """The reported score has to describe the data as it comes."""
    _, test = split_by_well(small_dataset, holdout_fraction=0.3, seed=3)
    before = np.unique(test["labels"], return_counts=True)[1]
    _ = balance_classes(test.copy(), seed=3)
    after = np.unique(test["labels"], return_counts=True)[1]
    assert np.array_equal(before, after)


def test_the_analyst_damping_disagrees_with_the_truth():
    """Damping is a judgement call on every real well. Training with it matched
    to the generator would be learning from an oracle the field never has."""
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(40):
        config = sample_well(rng)
        if abs(config.assumed_damping - config.true_damping) > 1e-9:
            mismatches += 1
    assert mismatches > 30, "%d of 40 wells had matched damping" % (40 - mismatches)


# ------------------------------------------------------------ inference ----


def test_softmax_survives_a_confident_prediction():
    """Exponentiating raw logits overflows to inf and then nan, which breaks
    precisely the predictions the model is most sure about -- the ones an
    operator would act on."""
    model = CardClassifier(
        weights=[np.eye(len(FEATURE_NAMES), 3) * 500.0],
        biases=[np.zeros(3)],
        mean=np.zeros(len(FEATURE_NAMES)), scale=np.ones(len(FEATURE_NAMES)),
        classes=["a", "b", "c"])
    probabilities = model.probabilities(np.full(len(FEATURE_NAMES), 1000.0))
    assert np.all(np.isfinite(probabilities))
    assert probabilities.sum() == pytest.approx(1.0)


def test_probabilities_sum_to_one(trained):
    model, test = trained
    probabilities = model.probabilities(test["features"][:25])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)


def test_prediction_reports_a_runner_up(trained):
    """A 0.45/0.43 split is a different situation from 0.95/0.02, and the top
    label alone hides which one you are in."""
    model, _ = trained
    prediction = model.predict(downhole_card("fluid_pound", SPEC, severity=0.8), SPEC)
    assert prediction.fault in FAULTS
    assert 0.0 <= prediction.confidence <= 1.0
    runner_up, probability = prediction.runner_up
    assert runner_up != prediction.fault
    assert probability <= prediction.confidence


def test_fillage_is_measured_not_predicted(trained):
    """Fillage is read off the card and converts straight into barrels. Asking
    a classifier to guess a quantity the card already states would add error
    for nothing."""
    from phasespace.rodpump.analysis import measure

    model, _ = trained
    card = downhole_card("fluid_pound", SPEC, severity=0.6)
    assert model.predict(card, SPEC).fillage == pytest.approx(
        measure(card, SPEC).fillage, rel=1e-12)


def test_inference_needs_only_numpy():
    """The deployment constraint. The serverless function carries numpy and
    nothing else, and a model that cannot run where the answer is needed is a
    research result rather than a feature."""
    import phasespace.rodpump.classifier as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    body = source.split("# ------------------------------------------------------------- training")[0]
    assert "sklearn" not in body, "sklearn leaked into the inference path"


# --------------------------------------------------------- persistence ----


def test_round_trips_through_disk(trained, tmp_path):
    model, test = trained
    path = tmp_path / "model.npz"
    model.save(path)
    reloaded = CardClassifier.load(path)
    assert np.allclose(reloaded.probabilities(test["features"][:20]),
                       model.probabilities(test["features"][:20]))
    assert reloaded.classes == model.classes


def test_a_model_trained_on_different_features_is_refused(trained, tmp_path):
    """The feature vector is computed by code that can change after a model is
    trained. Feeding a model a differently-ordered 71 features produces
    confident nonsense, so the names travel with the weights."""
    model, _ = trained
    path = tmp_path / "stale.npz"
    model.save(path)

    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    names = [str(n) for n in payload["feature_names"]]
    names[0] = "something_else"
    payload["feature_names"] = np.asarray(names)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="different feature vector"):
        CardClassifier.load(path)


def test_feature_width_mismatch_is_refused():
    with pytest.raises(ValueError, match="feature vector has changed"):
        CardClassifier(weights=[np.zeros((5, 2))], biases=[np.zeros(2)],
                       mean=np.zeros(5), scale=np.ones(5), classes=["a", "b"])


# ------------------------------------------------------------ learning ----


def test_the_model_learns_something_real(trained):
    """Not a pinned accuracy -- that would just record the last run. This asks
    whether it beats guessing the most common class by a wide margin, which is
    the floor below which the layer is worthless."""
    model, test = trained
    result = evaluate(model, test["features"], test["labels"])
    _, counts = np.unique(test["labels"], return_counts=True)
    majority = counts.max() / counts.sum()
    assert result["accuracy"] > max(0.70, 3.0 * majority), result["accuracy"]


def test_evaluate_reports_per_class_and_confusions(trained):
    """Per-class matters more than the headline: a model that is excellent on
    six faults and cannot separate pound from gas has failed the most valuable
    call a card offers while reporting a respectable average."""
    model, test = trained
    result = evaluate(model, test["features"], test["labels"])
    assert set(result["per_class"]).issubset(set(FAULTS))
    assert result["samples"] == test["labels"].size
    for key in result["confusions"]:
        assert " -> " in key
