"""Training data for the dynamometer card classifier.

Physics generates the labels, which is the whole reason a learned layer is
possible here without owning a fleet of instrumented wells: a fault is
described exactly at the pump, propagated up through the wave equation to the
surface card a dynamometer would record, and the classifier only ever sees
that surface card. The label is known by construction.

Three things about how the set is built matter more than the model that gets
trained on it.

**Every sample makes the full round trip.** The fault is built downhole, pushed
up to the surface, and pulled back down before any feature is measured. A
classifier trained on clean generated pump cards would learn the generator's
shapes and then meet cards that have been through a real reconstruction, with
its ringing and its damping error, and fail. Training on the round trip means
the reconstruction's artefacts are in the training distribution.

**The reconstruction damping deliberately disagrees with the truth.** Damping
is the one tuned parameter in the Gibbs method, and nobody knows the true value
on a real well -- an analyst picks it so the valve lines look right. So the
cards are generated at one damping and reconstructed at another, drawn nearby.
A model trained with them matched would be learning from an oracle it will
never have.

**The split is by well configuration, not by row.** Cards from the same well at
adjacent severities are near-duplicates; splitting randomly puts each side of a
near-duplicate pair in train and test, and reports a score that is mostly
memorisation. Held-out wells are the honest question: does this work on a well
the model has not seen?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnosis import feature_vector
from .rodstring import RodSection, RodString
from .synthetic import FAULTS, PumpSpec, surface_card_for, true_fillage
from .wave import Card, surface_to_downhole

# The taper patterns a real string is built from, as API rod numbers.
TAPERS = ((8, 6), (8, 7, 6), (7,), (8,), (9, 8, 7))

PLUNGERS = (1.06, 1.25, 1.5, 1.75, 2.0, 2.25, 2.75)


@dataclass
class WellConfig:
    """One well: the string, the pump, and how the analyst reconstructs it."""

    depth_ft: float
    taper: tuple
    plunger_in: float
    net_lift_ft: float
    fluid_sg: float
    stroke_in: float
    spm: float
    true_damping: float
    assumed_damping: float
    noise_lb_fraction: float

    def string(self) -> RodString:
        share = self.depth_ft / len(self.taper)
        return RodString([RodSection(api_number=n, length_ft=share)
                          for n in self.taper])

    def spec(self) -> PumpSpec:
        return PumpSpec(plunger_diameter_in=self.plunger_in,
                        net_lift_ft=self.net_lift_ft,
                        fluid_specific_gravity=self.fluid_sg,
                        stroke_in=self.stroke_in,
                        strokes_per_minute=self.spm)


def sample_well(rng: np.random.Generator) -> WellConfig:
    """Draw a plausible rod-pumped well.

    Ranges span shallow stripper wells through deep high-volume lift, because
    a classifier that has only seen one kind of well learns that well. The net
    lift is drawn as a fraction of depth rather than independently, since a
    pump cannot lift from below itself and independent draws produce wells that
    do not exist.
    """
    depth = float(rng.uniform(1200.0, 9000.0))
    taper = TAPERS[rng.integers(len(TAPERS))]
    true_damping = float(rng.uniform(0.15, 0.85))

    # The analyst's damping, off by up to about 40% either way. This is the
    # single most realistic thing in the generator: on a real well this number
    # is a judgement call, never a measurement.
    assumed = float(np.clip(true_damping * rng.uniform(0.6, 1.4), 0.05, 1.5))

    return WellConfig(
        depth_ft=depth,
        taper=taper,
        plunger_in=float(PLUNGERS[rng.integers(len(PLUNGERS))]),
        net_lift_ft=float(depth * rng.uniform(0.55, 0.98)),
        fluid_sg=float(rng.uniform(0.78, 1.05)),
        stroke_in=float(rng.uniform(48.0, 168.0)),
        spm=float(rng.uniform(4.0, 14.0)),
        true_damping=true_damping,
        assumed_damping=assumed,
        noise_lb_fraction=float(rng.choice([0.0, 0.01, 0.03, 0.06])),
    )


def sample_for(config: WellConfig, fault: str, severity: float,
               rng: np.random.Generator, assumed_damping: float = None) -> tuple:
    """One training row: (features, label, fillage) through the full round trip.

    Returns the true fillage alongside the class because fillage is the number
    an operator acts on -- it converts straight into barrels -- and a model that
    can only name a fault has answered the less useful half of the question.
    """
    string = config.string()
    spec = config.spec()

    surface, _ = surface_card_for(fault, string, spec, severity=severity,
                                  damping_per_s=config.true_damping, rng=rng)

    if config.noise_lb_fraction > 0.0:
        scale = config.noise_lb_fraction * spec.fluid_load_lb
        surface = Card(surface.position_in,
                       surface.load_lb + rng.normal(0.0, scale, surface.n_samples),
                       period_s=surface.period_s)

    # Reconstructed with the analyst's damping, not the true one.
    downhole = surface_to_downhole(
        surface, string,
        damping_per_s=config.assumed_damping if assumed_damping is None
        else assumed_damping)
    return feature_vector(downhole, spec), fault, true_fillage(fault, severity)


def build_dataset(wells: int = 120, severities: int = 9, seed: int = 0) -> dict:
    """Generate the full set, grouped by well so it can be split honestly.

    Severity starts above zero because a fault at severity zero *is* a full
    pump -- generating it under another label would be teaching the model that
    two identical cards have different names, which no model can learn and
    which quietly caps accuracy for every class.
    """
    rng = np.random.default_rng(seed)
    severity_grid = np.linspace(0.2, 1.0, severities)

    features, labels, fillages, groups = [], [], [], []
    for well_index in range(wells):
        config = sample_well(rng)
        for fault in FAULTS:
            for severity in severity_grid:
                # A full pump has no severity axis -- there is only one way for
                # a pump to be working correctly -- so it would arrive once per
                # well against `severities` times for every fault.
                #
                # Emitting it the same number of times, varying the analyst's
                # damping instead, is not padding. Damping is the one tuned
                # parameter in the reconstruction and it is a guess on every
                # real well; varying it produces genuinely different cards, and
                # what the model learns from them is to be invariant to that
                # guess. Oversampling by duplicating rows was tried first and
                # did nothing, because a duplicate carries no information.
                assumed = None
                if fault == "full_pump":
                    assumed = float(np.clip(
                        config.true_damping * rng.uniform(0.6, 1.4), 0.05, 1.5))
                x, y, fillage = sample_for(config, fault, float(severity), rng,
                                           assumed_damping=assumed)
                features.append(x)
                labels.append(y)
                fillages.append(fillage)
                groups.append(well_index)

    return {
        "features": np.asarray(features, dtype=float),
        "labels": np.asarray(labels),
        "fillages": np.asarray(fillages, dtype=float),
        "groups": np.asarray(groups, dtype=int),
    }


def split_by_well(dataset: dict, holdout_fraction: float = 0.25,
                  seed: int = 0) -> tuple:
    """Split on wells, never on rows.

    Cards from one well at adjacent severities are near-duplicates. A random
    row split puts one of each pair in train and the other in test, and the
    resulting score measures memorisation rather than generalisation -- it can
    look excellent while the model is useless on a new well.
    """
    rng = np.random.default_rng(seed)
    wells = np.unique(dataset["groups"])
    rng.shuffle(wells)
    cut = max(int(round(len(wells) * holdout_fraction)), 1)
    held_out = set(wells[:cut].tolist())

    mask = np.array([group in held_out for group in dataset["groups"]])
    train = {key: value[~mask] for key, value in dataset.items()}
    test = {key: value[mask] for key, value in dataset.items()}
    return train, test


def balance_classes(split: dict, seed: int = 0) -> dict:
    """Oversample under-represented classes up to the median class count.

    A full pump has no severity axis -- there is only one way for a pump to be
    working correctly -- so it arrives at one card per well against seven for
    every fault. Left alone, the model pays almost nothing for getting it wrong
    and duly does: it scored 87% on healthy pumps where the rule layer scored
    100%.

    That is the worst class to be weak on. Every other error sends a crew to
    the wrong repair; this one sends a crew to a well that does not need
    touching, which is how a diagnostic tool loses the crew's trust and stops
    being used at all.

    Applied to the training split only. The test split keeps the natural
    distribution, because the reported score has to describe the data as it
    comes rather than as it was convenient to train on.
    """
    rng = np.random.default_rng(seed)
    labels = split["labels"]
    classes, counts = np.unique(labels, return_counts=True)
    target = int(np.median(counts))

    indices = [np.arange(labels.size)]
    for name, count in zip(classes, counts):
        if count >= target:
            continue
        pool = np.nonzero(labels == name)[0]
        indices.append(rng.choice(pool, size=target - count, replace=True))

    order = np.concatenate(indices)
    rng.shuffle(order)
    return {key: value[order] for key, value in split.items()}
