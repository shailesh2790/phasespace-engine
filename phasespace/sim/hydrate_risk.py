"""Will a hydrate actually form, and when. **Measured answer: not from here.**

The Hammerschmidt curve answers a different question from the one an operator
has. It says whether the stream is inside the hydrate stability region, which
is a necessary condition for a plug and not a sufficient one -- measured on
Petrobras 3W, in five of seven real events the margin was still 25 to 65 degF
clear of the curve ten minutes before the label began, while in two other
instances the stream sat below the curve for over ten hours with nothing
happening. What the curve cannot express is kinetics.

That looked like the one place in this engine where a learned layer had a
measured reason to exist. It was built, trained on the 81 simulated instances
Petrobras ships for the purpose, and tested within each real instance so that
a well is compared against its own earlier hours rather than against other
wells -- the normal instances come from wells 1-8 and the events from wells
19-32, so a model scored across those groups could be recognising the well.

**It did not beat the physics, and the reason is worth more than the model
would have been.** Against a *tuned* margin threshold the learned layer was
identical at its best operating point -- 4 of 11 events caught, 1 false alarm
in 537 normal instances -- and worse at every tighter threshold.

Then the diagnostic. In the hour before onset, the tree pressure and
temperature do not move:

    instance                      -60 min   -30 min   -10 min    -1 min
    WELL-00019_20120601165020        49.9      52.1      50.9      54.0
    WELL-00019_20141117190526        65.5      65.9      65.6      65.7
    WELL-00019_20150614025713        63.6      63.8      63.8      63.9
    WELL-00025_20200629194141        38.5      38.9      33.2      24.7
    WELL-00027_20230918170106       -37.4     -37.3     -37.3     -37.2
    WELL-00030_20141108012623        31.2      26.2      25.8      26.5
    WELL-00032_20110830235321       -77.1     -77.1     -77.1     -77.1

The inputs are flat up to the event. No model on these two channels can
predict onset, because the information is not in them -- which is why the
learned layer matched the baseline exactly and why its lead time was always
precisely the window length: both were reading a constant.

**So no learned hydrate predictor ships.** What ships is the physics
detector, on its own evidence: concurrent detection of 7 of 7 real
production-line events, 1 false alarm in 537 normal instances.

What would change the answer is instrumentation, not architecture. These
events form somewhere the subsea tree does not see, and predicting them
needs a channel that moves beforehand -- water cut, inhibitor injection
rate, a downstream cold-spot temperature, or the flowline profile. That is a
sensing problem with a known shape, which is a better thing to hand a
customer than a model that cannot work.

The feature extraction below stays, because the margin, its trends and the
residence time are useful physics in their own right and are what any future
attempt would start from. RiskModel stays with it, unwired, as the record of
what was tried.

Units: pressure psia, temperature degF, time seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .thermal import (
    HAMMERSCHMIDT_MAX_PSIA,
    HAMMERSCHMIDT_MIN_PSIA,
    hydrate_temperature_r,
)

# How far ahead the question is asked. Thirty minutes is the shortest window
# in which an operator can act -- start methanol injection, open a choke,
# begin a depressurisation -- so a warning with less lead than this is a
# notification rather than a decision.
DEFAULT_HORIZON_S = 1800

# Trend windows, in seconds. A hydrate approach is slow; one minute of
# history says almost nothing, and an hour smooths the approach away.
TREND_WINDOWS = (300, 1800)

# How far the learned layer may move the physics view, in probability. The
# cap exists so a model trained on simulated instances cannot overrule the
# thermodynamics on a well it has never seen.
DEFAULT_CAP = 0.35

FEATURES = (
    "margin_f",
    "margin_rate_5m",
    "margin_rate_30m",
    "pressure_psia",
    "pressure_rate_30m",
    "temperature_f",
    "temperature_rate_30m",
    "seconds_in_region",
)


def hydrate_margin_f(pressure_psia, temperature_f):
    """Degrees above the hydrate curve. Negative means inside the region.

    Non-physical pressures give NaN rather than an exception, because this
    runs over whole instrument traces where a failed sensor is ordinary and
    must not stop the scan. 2.2% of tree-pressure readings in 3W are
    negative.
    """
    pressure = np.asarray(pressure_psia, dtype=float)
    temperature = np.asarray(temperature_f, dtype=float)
    usable = ((pressure > HAMMERSCHMIDT_MIN_PSIA)
              & (pressure < HAMMERSCHMIDT_MAX_PSIA)
              & np.isfinite(pressure) & np.isfinite(temperature))
    out = np.full(pressure.shape, np.nan)
    if usable.any():
        curve = np.array([hydrate_temperature_r(float(p))[0] - 459.67
                          for p in pressure[usable]])
        out[usable] = temperature[usable] - curve
    return out


def _rate_per_hour(values, window_s, sample_s=1.0):
    """Change per hour over a trailing window, NaN-safe.

    A plain difference against `values[i - window]` propagates a single NaN
    into the whole window. Differencing a forward-filled copy keeps the
    feature defined through short instrument dropouts, which are frequent.
    """
    series = np.asarray(values, dtype=float)
    filled = series.copy()
    missing = ~np.isfinite(filled)
    if missing.all():
        return np.full(series.shape, np.nan)
    index = np.arange(filled.size)
    filled[missing] = np.interp(index[missing], index[~missing],
                                filled[~missing])

    step = max(int(window_s / sample_s), 1)
    shifted = np.empty_like(filled)
    shifted[:step] = filled[0]
    shifted[step:] = filled[:-step]
    return (filled - shifted) / (window_s / 3600.0)


def _seconds_in_region(margin, sample_s=1.0):
    """How long the stream has been continuously below the curve.

    Time matters: a plug needs subcooling *and* residence. A stream that
    dipped below the curve for a minute is not the same as one that has been
    there for six hours, and the equilibrium margin cannot tell them apart.
    """
    series = np.asarray(margin, dtype=float)
    out = np.zeros(series.shape)
    running = 0.0
    for index, value in enumerate(series):
        running = running + sample_s if (np.isfinite(value) and value < 0.0) else 0.0
        out[index] = running
    return out


def features_from_trace(pressure_psia, temperature_f, sample_s=1.0) -> dict:
    """The feature set, every column physics-derived or directly measured."""
    margin = hydrate_margin_f(pressure_psia, temperature_f)
    pressure = np.asarray(pressure_psia, dtype=float)
    temperature = np.asarray(temperature_f, dtype=float)
    return {
        "margin_f": margin,
        "margin_rate_5m": _rate_per_hour(margin, TREND_WINDOWS[0], sample_s),
        "margin_rate_30m": _rate_per_hour(margin, TREND_WINDOWS[1], sample_s),
        "pressure_psia": pressure,
        "pressure_rate_30m": _rate_per_hour(pressure, TREND_WINDOWS[1], sample_s),
        "temperature_f": temperature,
        "temperature_rate_30m": _rate_per_hour(temperature, TREND_WINDOWS[1],
                                               sample_s),
        "seconds_in_region": _seconds_in_region(margin, sample_s),
    }


def matrix_from_trace(pressure_psia, temperature_f, sample_s=1.0):
    """(n, len(FEATURES)) array in a fixed column order."""
    columns = features_from_trace(pressure_psia, temperature_f, sample_s)
    return np.column_stack([columns[name] for name in FEATURES])


def label_from_trace(classes, horizon_s=DEFAULT_HORIZON_S, event_codes=(8, 108),
                     sample_s=1.0):
    """1 where an event begins within the horizon, 0 before that, NaN at and
    after onset.

    Samples at or after onset are excluded rather than labelled negative or
    positive. Including them would let a model score well by recognising an
    event already in progress, which is detection -- the physics already does
    that, and doing it again would not be prediction.
    """
    labels = np.asarray(classes, dtype=float)
    event = np.isin(labels, event_codes)
    target = np.zeros(labels.shape)
    if not event.any():
        return target                       # a normal instance: all negative

    onset = int(np.argmax(event))
    horizon = max(int(horizon_s / sample_s), 1)
    target[:] = 0.0
    target[max(onset - horizon, 0):onset] = 1.0
    target[onset:] = np.nan
    return target


# =============================================================== the model ==


@dataclass
class RiskModel:
    """Logistic regression on physics features, bounded against the physics.

    Logistic rather than deep, and that is a finding rather than a
    limitation. There are seven real hydrate events in the public record with
    usable instrumentation; a model with thousands of parameters would fit
    the simulator that generated the training instances, not the physics, and
    `evaluate` would not be able to tell. A linear model on eight physics
    features has coefficients an engineer can read and argue with, which is
    the same property the rest of this engine is built on.

    Weights are plain arrays, so inference is numpy and the air-gapped
    deployment story holds.
    """

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    cap: float = DEFAULT_CAP
    feature_names: tuple = FEATURES

    def _standardise(self, matrix):
        return (np.asarray(matrix, dtype=float) - self.mean) / self.scale

    def raw_probability(self, matrix):
        z = self._standardise(matrix) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    def physics_probability(self, matrix):
        """What the thermodynamics alone says, as a probability.

        A smooth ramp on the margin rather than a step, so the bound below is
        a comparison between two continuous quantities. Centred on the curve
        itself and scaled over 20 degF, which is roughly the spread between a
        stream comfortably clear of the region and one well inside it.
        """
        margin = np.asarray(matrix, dtype=float)[:, FEATURES.index("margin_f")]
        return 1.0 / (1.0 + np.exp(np.clip(margin / 20.0, -60.0, 60.0)))

    def predict(self, matrix):
        """Bounded risk, plus where the model wanted to go further.

        Returns (probability, capped) -- `capped` marks the samples where the
        learned layer disagreed with the physics by more than the cap and was
        held back. Those are the rows to show a user, because a model
        straining against the thermodynamics is either finding something the
        correlation misses or being asked about a well unlike anything it was
        trained on, and the two look identical from inside.
        """
        physics = self.physics_probability(matrix)
        learned = self.raw_probability(matrix)
        low, high = physics - self.cap, physics + self.cap
        bounded = np.clip(learned, low, high)
        return bounded, (learned < low) | (learned > high)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "cap": float(self.cap),
            "features": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiskModel":
        return cls(weights=np.array(data["weights"], dtype=float),
                   bias=float(data["bias"]),
                   mean=np.array(data["mean"], dtype=float),
                   scale=np.array(data["scale"], dtype=float),
                   cap=float(data.get("cap", DEFAULT_CAP)),
                   feature_names=tuple(data.get("features", FEATURES)))


def fit(matrix, target, l2=1.0, iterations=300, cap=DEFAULT_CAP) -> RiskModel:
    """Fit by Newton-Raphson on the penalised log-likelihood.

    numpy only, like the rest of the engine. Eight features and a few hundred
    thousand rows converge in a handful of Newton steps, so there is no case
    for a dependency or for stochastic gradient descent.

    The L2 penalty is not optional at this data size: hydrate instances are
    heavily correlated within an instance -- consecutive one-second samples
    are nearly identical -- so the effective sample count is far smaller than
    the row count, and an unpenalised fit reads that redundancy as evidence.
    """
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[keep], y[keep]
    if x.size == 0:
        raise ValueError("no usable rows after dropping missing values")
    if len(np.unique(y)) < 2:
        raise ValueError(
            "the training set has only one class, so nothing can be learned "
            "-- check the horizon and the event codes")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale

    design = np.column_stack([z, np.ones(z.shape[0])])
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[-1, -1] = 0.0                      # never penalise the intercept

    for _ in range(iterations):
        eta = np.clip(design @ beta, -60.0, 60.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        gradient = design.T @ (probability - y) + penalty @ beta
        weights = np.clip(probability * (1.0 - probability), 1e-9, None)
        hessian = (design * weights[:, None]).T @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break

    return RiskModel(weights=beta[:-1], bias=float(beta[-1]), mean=mean,
                     scale=scale, cap=cap)


# ============================================================== evaluation ==


@dataclass
class Score:
    name: str
    lead_minutes: list = field(default_factory=list)
    detected: int = 0
    events: int = 0
    false_alarm_instances: int = 0
    normal_instances: int = 0
    # Event instances where the alarm also fired during that same well's
    # earlier normal hours. A detector that is on all the time catches every
    # event and is worth nothing, and only a within-instance count exposes
    # that without the well confound.
    noisy_events: int = 0

    @property
    def recall(self) -> float:
        return self.detected / self.events if self.events else 0.0

    @property
    def false_alarm_rate(self) -> float:
        return (self.false_alarm_instances / self.normal_instances
                if self.normal_instances else 0.0)

    @property
    def median_lead(self) -> float:
        return float(np.median(self.lead_minutes)) if self.lead_minutes else 0.0

    def line(self) -> str:
        return ("%-20s caught %2d/%-2d  lead %6.1f min  always-on %2d/%-2d  "
                "false alarms %3d/%-4d"
                % (self.name, self.detected, self.events, self.median_lead,
                   self.noisy_events, self.events,
                   self.false_alarm_instances, self.normal_instances))
