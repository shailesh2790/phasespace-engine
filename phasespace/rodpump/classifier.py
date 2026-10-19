"""The learned layer of dynamometer card diagnosis.

The rule layer reads a card the way an operator does -- drooping top line,
lifting bottom line, late abrupt transfer -- and it can say *why*. It is also
a set of thresholds, and thresholds have a characteristic failure: they are
confident on either side and arbitrary near the line. Its errors are almost all
at mild severities, which is exactly where a fault is still cheap to fix.

This layer exists for those cases. It does not replace the rules. The rules run
always and supply the evidence, because a verdict an operator cannot check is
not usable where being confidently wrong costs a pulling unit.

Where the two disagree, that disagreement is the most informative thing the
system produces, and it is reported rather than resolved silently. A card that
the physics-derived thresholds and a model trained on physics-generated cards
read differently is a card worth a human looking at.

**Inference is pure numpy, deliberately.** Training uses scikit-learn, but the
fitted weights are exported to an npz and evaluated here as three matrix
multiplies. That keeps the model deployable into the serverless function, which
carries numpy and nothing else -- the same constraint that made the gas
surrogate go to ONNX in the browser. A model that cannot be deployed where the
answer is needed is a research result, not a feature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diagnosis import FEATURE_NAMES, feature_vector
from .synthetic import FAULT_LABELS, PumpSpec
from .wave import Card


@dataclass
class Prediction:
    """What the model thinks, with its own uncertainty."""

    fault: str
    confidence: float
    probabilities: dict
    fillage: float

    @property
    def label(self) -> str:
        return FAULT_LABELS.get(self.fault, self.fault)

    @property
    def runner_up(self) -> tuple:
        """The second-place class and its probability.

        Worth surfacing: a 0.45/0.43 split between fluid pound and gas
        interference is a completely different situation from 0.95/0.02, and
        the top-1 label alone hides which one you are in.
        """
        ranked = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        return ranked[1] if len(ranked) > 1 else (ranked[0][0], 0.0)


class CardClassifier:
    """A small multilayer perceptron over the 71 card features.

    Small on purpose. The features are already the physics -- fillage, load
    ratio, transfer sharpness, the normalised card shape -- so the model is
    asked to learn where the boundaries between fault classes actually fall,
    not to rediscover what a dynamometer card is. A large network on this
    feature set memorises wells.
    """

    def __init__(self, weights: list, biases: list, mean: np.ndarray,
                 scale: np.ndarray, classes: list, metadata: dict = None):
        if len(weights) != len(biases):
            raise ValueError("weights and biases must pair up: %d vs %d"
                             % (len(weights), len(biases)))
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.classes = list(classes)
        self.metadata = dict(metadata or {})

        if self.mean.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                "model expects %d features but this build produces %d -- the "
                "feature vector has changed since the model was trained, and "
                "the saved weights no longer mean anything"
                % (self.mean.shape[0], len(FEATURE_NAMES)))

    # -------------------------------------------------------- inference --

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Forward pass: standardise, ReLU stack, softmax."""
        x = (np.atleast_2d(features) - self.mean) / self.scale
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ weight + bias, 0.0)
        logits = x @ self.weights[-1] + self.biases[-1]
        # Subtract the row max before exponentiating. Without it a confident
        # prediction overflows to inf/nan, which turns the most certain
        # answers into the broken ones.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def predict(self, card: Card, spec: PumpSpec) -> Prediction:
        features = feature_vector(card, spec)
        probabilities = self.probabilities(features)[0]
        best = int(np.argmax(probabilities))
        return Prediction(
            fault=self.classes[best],
            confidence=float(probabilities[best]),
            probabilities={name: float(p)
                           for name, p in zip(self.classes, probabilities)},
            # Fillage is measured, not predicted: it is read directly off the
            # reconstructed card and converts straight into barrels. Asking a
            # classifier to guess a quantity the card already states would be
            # adding error for nothing.
            fillage=float(features[FEATURE_NAMES.index("fillage")]),
        )

    # ------------------------------------------------------ persistence --

    def save(self, path) -> None:
        payload = {
            "classes": np.asarray(self.classes),
            "mean": self.mean,
            "scale": self.scale,
            "layers": np.asarray([len(self.weights)]),
            "feature_names": np.asarray(FEATURE_NAMES),
        }
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload["weight_%d" % i] = weight
            payload["bias_%d" % i] = bias
        for key, value in self.metadata.items():
            payload["meta_%s" % key] = np.asarray(value)
        np.savez(str(path), **payload)

    @classmethod
    def load(cls, path) -> "CardClassifier":
        with np.load(str(path), allow_pickle=False) as data:
            layers = int(data["layers"][0])
            weights = [data["weight_%d" % i] for i in range(layers)]
            biases = [data["bias_%d" % i] for i in range(layers)]
            metadata = {key[5:]: data[key].tolist()
                        for key in data.files if key.startswith("meta_")}

            # The feature vector is computed by code that can change after a
            # model is trained. Silently feeding a 71-feature model a
            # differently-ordered 71 features would produce confident nonsense,
            # so the names travel with the weights and are checked.
            if "feature_names" in data.files:
                saved = [str(name) for name in data["feature_names"]]
                if saved != list(FEATURE_NAMES):
                    raise ValueError(
                        "this model was trained on a different feature vector "
                        "(first mismatch: %s). Retrain it against the current "
                        "one rather than loading it."
                        % _first_mismatch(saved, list(FEATURE_NAMES)))

            return cls(weights=weights, biases=biases, mean=data["mean"],
                       scale=data["scale"],
                       classes=[str(c) for c in data["classes"]],
                       metadata=metadata)


def _first_mismatch(a: list, b: list) -> str:
    for i in range(max(len(a), len(b))):
        left = a[i] if i < len(a) else "<missing>"
        right = b[i] if i < len(b) else "<missing>"
        if left != right:
            return "position %d, %r vs %r" % (i, left, right)
    return "lengths differ"


MODEL_FILENAME = "dynacard_classifier.npz"


def default_model_path() -> Path:
    """Where the shipped model lives.

    A sibling file wins over the repository's models/ directory, because the
    vendored copy inside the serverless function has no repository around it --
    it is a flat directory of modules plus the weights. Checking next to this
    file first means the same code loads the model in both places without the
    deployment needing to know it is a deployment.
    """
    beside = Path(__file__).resolve().parent / MODEL_FILENAME
    if beside.is_file():
        return beside
    return Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME


def load_default() -> CardClassifier:
    """The shipped model, or a clear error saying how to make one.

    Deliberately not a silent fallback to the rule layer. If a caller asked for
    the learned layer and there is no model, they should be told, not quietly
    handed something else that scores differently.
    """
    path = default_model_path()
    if not path.is_file():
        raise FileNotFoundError(
            "no trained card classifier at %s -- run "
            "`python scripts/train_dynacard.py` to build one" % path)
    return CardClassifier.load(path)


# ------------------------------------------------------------- training ----


def train_classifier(features: np.ndarray, labels: np.ndarray,
                     hidden: tuple = (96, 48), seed: int = 0,
                     max_iter: int = 900) -> CardClassifier:
    """Fit the network. Requires scikit-learn; inference never does.

    Standardisation statistics are taken from the training split only and
    carried inside the model, so the deployed path cannot accidentally
    standardise with numbers that saw the test set.
    """
    try:
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "training needs scikit-learn (inference does not). "
            "pip install scikit-learn") from exc

    mean = features.mean(axis=0)
    # A constant feature has zero spread; dividing by it produces inf and then
    # nan weights, which surfaces much later as a model that predicts one class.
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0

    standardised = (features - mean) / scale
    network = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                            solver="adam", alpha=1e-3, max_iter=max_iter,
                            random_state=seed, early_stopping=True,
                            n_iter_no_change=25, validation_fraction=0.15)
    network.fit(standardised, labels)

    return CardClassifier(
        weights=network.coefs_, biases=network.intercepts_,
        mean=mean, scale=scale, classes=list(network.classes_))


def evaluate(model: CardClassifier, features: np.ndarray,
             labels: np.ndarray) -> dict:
    """Accuracy overall and per class, plus the confusion pairs.

    Per-class matters more than the headline here. A model that is excellent on
    six faults and cannot tell fluid pound from gas interference has failed at
    the single most valuable call a card offers, while still reporting a
    respectable average.
    """
    probabilities = model.probabilities(features)
    predicted = np.asarray(model.classes)[probabilities.argmax(axis=1)]
    correct = predicted == labels

    per_class = {}
    for fault in sorted(set(labels.tolist())):
        mask = labels == fault
        per_class[fault] = float(correct[mask].mean()) if mask.any() else math.nan

    confusions = {}
    for truth, guess in zip(labels[~correct], predicted[~correct]):
        key = "%s -> %s" % (truth, guess)
        confusions[key] = confusions.get(key, 0) + 1

    return {
        "accuracy": float(correct.mean()),
        "per_class": per_class,
        "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
        "samples": int(labels.size),
    }
