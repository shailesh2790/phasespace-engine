"""Tests for the hydrate risk features, and for the bound on the model.

The learned layer this module was built for does not ship: measured against
a tuned margin threshold on Petrobras 3W it was identical at its best
operating point and worse elsewhere, because the tree pressure and
temperature are flat in the hour before an event and the information simply
is not in them. The module's own docstring carries the numbers.

What is tested here is what survives that: the physics feature extraction,
which is useful on its own and is what any future attempt would start from,
and the bound that keeps `RiskModel` from overruling thermodynamics if it is
ever wired in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.hydrate_risk import (
    DEFAULT_CAP,
    DEFAULT_HORIZON_S,
    FEATURES,
    RiskModel,
    fit,
    hydrate_margin_f,
    label_from_trace,
    matrix_from_trace,
)
from phasespace.sim.thermal import hydrate_temperature_r


# ------------------------------------------------------------- the margin --


def test_the_margin_agrees_with_the_curve_it_is_built_on():
    pressures = np.array([500.0, 1200.0, 2500.0])
    temperatures = np.array([80.0, 80.0, 80.0])
    margins = hydrate_margin_f(pressures, temperatures)
    for pressure, temperature, margin in zip(pressures, temperatures, margins):
        expected = temperature - (hydrate_temperature_r(float(pressure))[0] - 459.67)
        assert margin == pytest.approx(expected, abs=1e-9)


def test_pressures_outside_the_correlation_give_nan_not_an_exception():
    """This runs over whole instrument traces where a failed sensor is
    ordinary. 2.2% of tree-pressure readings in 3W are negative, and an
    exception would stop the scan on the first one."""
    margins = hydrate_margin_f([-100.0, 0.0, 1500.0, 99_000.0],
                               [60.0, 60.0, 60.0, 60.0])
    assert np.isnan(margins[0]) and np.isnan(margins[1])
    assert np.isfinite(margins[2])
    assert np.isnan(margins[3])


def test_a_cooling_stream_crosses_the_curve():
    pressure = np.full(600, 1500.0)
    temperature = np.linspace(120.0, 40.0, 600)
    margins = hydrate_margin_f(pressure, temperature)
    assert margins[0] > 0.0
    assert margins[-1] < 0.0
    assert np.all(np.diff(margins) < 0.0)


# ------------------------------------------------------------- the trends --


def test_trend_features_survive_instrument_dropouts():
    """A plain difference against a lagged sample propagates one NaN across a
    whole window. Dropouts are frequent, and a feature that vanishes whenever
    a sensor blinks is not usable on real traces."""
    pressure = np.full(4000, 1500.0)
    temperature = np.linspace(120.0, 70.0, 4000)
    temperature[1000:1010] = np.nan

    columns = matrix_from_trace(pressure, temperature)
    rate = columns[:, FEATURES.index("margin_rate_30m")]
    assert np.isfinite(rate[2000:]).all()
    assert rate[3000] < 0.0, "a cooling stream must show a negative margin rate"


def test_residence_time_counts_only_continuous_time_below_the_curve():
    """A stream that dipped below for a minute is not the same as one that
    has been there six hours, and the equilibrium margin cannot tell them
    apart."""
    pressure = np.full(300, 1500.0)
    temperature = np.full(300, 40.0)        # well inside the region
    temperature[100:110] = 200.0            # a brief excursion back out
    seconds = matrix_from_trace(pressure, temperature)[
        :, FEATURES.index("seconds_in_region")]

    assert seconds[0] == pytest.approx(1.0)
    assert seconds[99] == pytest.approx(100.0)
    assert seconds[105] == 0.0, "the counter must reset on leaving the region"
    assert seconds[-1] == pytest.approx(190.0)


# ------------------------------------------------------------- the labels --


def test_only_the_window_before_onset_is_positive():
    classes = np.zeros(5000)
    classes[4000:] = 8
    target = label_from_trace(classes, horizon_s=600)

    assert target[3300] == 0.0
    assert target[3500] == 1.0
    assert np.isnan(target[4200]), "samples after onset must be excluded"
    assert np.nansum(target) == pytest.approx(600.0)


def test_samples_during_the_event_are_excluded_not_labelled():
    """Including them would let a model score well by recognising an event
    already in progress. That is detection, which the physics already does,
    and it is not the question being asked."""
    classes = np.zeros(2000)
    classes[1000:] = 108
    target = label_from_trace(classes, horizon_s=DEFAULT_HORIZON_S)
    assert np.isnan(target[1000:]).all()
    assert np.isfinite(target[:1000]).all()


def test_a_normal_instance_is_all_negative():
    target = label_from_trace(np.zeros(500))
    assert (target == 0.0).all()


# --------------------------------------------------------------- the bound --


def synthetic_model():
    """A model deliberately trained to disagree with the physics, so the
    bound has something to hold back."""
    rng = np.random.default_rng(0)
    rows = 4000
    matrix = np.zeros((rows, len(FEATURES)))
    matrix[:, FEATURES.index("margin_f")] = rng.uniform(-80.0, 80.0, rows)
    for index in range(1, len(FEATURES)):
        matrix[:, index] = rng.normal(0.0, 1.0, rows)
    # Label the opposite way round from the thermodynamics.
    target = (matrix[:, FEATURES.index("margin_f")] > 0.0).astype(float)
    return fit(matrix, target), matrix


def test_the_learned_layer_cannot_overrule_the_thermodynamics():
    model, matrix = synthetic_model()
    bounded, capped = model.predict(matrix)
    physics = model.physics_probability(matrix)

    assert np.all(np.abs(bounded - physics) <= DEFAULT_CAP + 1e-9)
    assert capped.any(), "this model was built to strain against the bound"


def test_where_the_model_strains_is_reported_not_hidden():
    """A model pushing against the thermodynamics is either finding
    something the correlation misses or being asked about a well unlike
    anything it saw, and those look identical from inside. Either way the
    user needs to know which rows they were."""
    model, matrix = synthetic_model()
    _bounded, capped = model.predict(matrix)
    assert capped.dtype == bool
    assert capped.shape == (matrix.shape[0],)


def test_the_physics_probability_follows_the_margin():
    model, _ = synthetic_model()
    deep = np.zeros((1, len(FEATURES)))
    deep[0, FEATURES.index("margin_f")] = -60.0
    clear = np.zeros((1, len(FEATURES)))
    clear[0, FEATURES.index("margin_f")] = 60.0

    assert model.physics_probability(deep)[0] > 0.9
    assert model.physics_probability(clear)[0] < 0.1


def test_a_model_round_trips_through_a_dict():
    """Weights are plain arrays so inference stays numpy and the air-gapped
    deployment story holds."""
    model, matrix = synthetic_model()
    restored = RiskModel.from_dict(model.to_dict())
    assert np.allclose(restored.predict(matrix)[0], model.predict(matrix)[0])


def test_fitting_one_class_is_refused():
    matrix = np.zeros((100, len(FEATURES)))
    with pytest.raises(ValueError, match="only one class"):
        fit(matrix, np.ones(100))
