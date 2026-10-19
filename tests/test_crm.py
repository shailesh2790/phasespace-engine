"""Validation ledger for the capacitance-resistance model.

Structured after the build plan's section 3.3, with one substantive departure
that the plan did not anticipate and that changes what CRM output means.

The plan's connectivity test compares fitted ``f_ij`` against "known allocation
factors". There are two different things that phrase can mean, and they are not
close to each other:

    where the injected water ends up      a transport answer, from streamlines
    dq_j / di_i at steady state           a pressure answer, from diffusion

CRM fits the second. On the asymmetric case below, water from I-1 reaches P-1
(0.68) and never reaches P-3 (0.00) -- but turning I-1 *up* raises pressure
across the connected volume and a quarter of that increase comes out of P-3
(0.23). Both numbers are correct answers to different questions.

So the recovery test here compares against a step-response fixture, and a
separate test documents the gap against the tracer fixture, because an operator
reading a fitted f_ij as "the share of my water arriving there" would be wrong
by a factor of three on this perfectly ordinary case.

Truth comes from a finite-difference pressure-diffusion simulator with Peaceman
wells. It shares no mathematics with CRM -- no exponential, no time constant,
no least squares -- which is what makes it usable as a reference at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.reservoir.crm.fit import (
    bootstrap_connectivity,
    distance_prior,
    fit_crm,
    identifiability,
)
from phasespace.reservoir.crm.model import (
    CRMParameters,
    WellField,
    crm_predict,
    crmt_predict,
)
from phasespace.reservoir.synthetic import (
    Grid,
    Well,
    five_spot,
    simulate,
    step_response_connectivity,
)


def asymmetric_case(steps=120, days=3600.0, seed=0):
    """Two injectors, three producers, uncorrelated schedules."""
    rng = np.random.default_rng(seed)
    grid = Grid(nx=21, ny=15)
    injectors = [Well("I-1", 5, 7), Well("I-2", 15, 7)]
    producers = [Well("P-1", 2, 3), Well("P-2", 10, 11), Well("P-3", 18, 3)]
    schedule = np.clip(np.column_stack([
        1500 + 900 * np.sin(np.arange(steps) / 7.0) + rng.normal(0, 60, steps),
        2200 + 1100 * np.sin(np.arange(steps) / 11.0 + 2.0) + rng.normal(0, 60, steps),
    ]), 0.0, None)
    case = simulate(grid, injectors, producers, schedule, 2300.0,
                    days=days, steps=steps)
    return grid, injectors, producers, case


@pytest.fixture(scope="module")
def asymmetric():
    return asymmetric_case()


@pytest.fixture(scope="module")
def step_truth(asymmetric):
    grid, injectors, producers, _ = asymmetric
    return step_response_connectivity(grid, injectors, producers,
                                      np.array([1500.0, 2200.0]), 2300.0)


# --------------------------------------------- the forward model, exactly ---


def test_step_injection_matches_the_closed_form():
    """With constant injection and no pressure change the recursion collapses
    to q(t) = q0*exp(-t/tau) + f*i*(1 - exp(-t/tau)). Derived on paper,
    compared against the code, exact."""
    steps, dt, tau, gain, rate = 40, 30.0, 200.0, 0.7, 1000.0
    times = np.arange(1, steps + 1) * dt
    field = WellField(times_days=times,
                      injection=np.full((steps, 1), rate),
                      production=np.zeros((steps, 1)))
    parameters = CRMParameters(connectivity=np.array([[gain]]),
                               tau_days=np.array([tau]))
    predicted = crm_predict(field, parameters, initial_rates=np.array([500.0]))

    elapsed = np.arange(1, steps + 1) * dt
    expected = 500.0 * np.exp(-elapsed / tau) + gain * rate * (1 - np.exp(-elapsed / tau))
    assert predicted[:, 0] == pytest.approx(expected, rel=1e-10)


def test_zero_injection_reduces_to_exponential_decline():
    """CRM contains classical decline as a special case: with no injection it
    is Arps with b = 0 and D = 1/tau. This is how you explain the model to an
    engineer who has only ever used Arps, and it is an independent check
    because Arps is a different published model."""
    steps, dt, tau, q0 = 36, 30.0, 400.0, 800.0
    times = np.arange(1, steps + 1) * dt
    field = WellField(times_days=times,
                      injection=np.zeros((steps, 1)),
                      production=np.zeros((steps, 1)))
    predicted = crm_predict(field, CRMParameters(np.array([[0.0]]), np.array([tau])),
                            initial_rates=np.array([q0]))
    arps = q0 * np.exp(-times / tau)
    assert predicted[:, 0] == pytest.approx(arps, rel=1e-10)


def test_material_balance_is_a_property_of_the_parameters():
    parameters = CRMParameters(np.array([[0.5, 0.3], [0.2, 0.6]]),
                               np.array([100.0, 150.0]))
    assert parameters.injector_allocation == pytest.approx([0.8, 0.8])
    assert parameters.unallocated_fraction == pytest.approx([0.2, 0.2])


def test_pore_volume_needs_a_productivity_index():
    """tau = c_t*V_p/J, so without J there is no pore volume. Returning one
    anyway would invent the number an operator is most likely to sanity-check
    against their mapped drainage area."""
    parameters = CRMParameters(np.array([[1.0]]), np.array([200.0]))
    with pytest.raises(ValueError, match="productivity"):
        parameters.pore_volume_bbl(1.2e-5)

    with_j = CRMParameters(np.array([[1.0]]), np.array([200.0]),
                           productivity=np.array([5.0]))
    assert with_j.pore_volume_bbl(1e-5) == pytest.approx([200.0 * 5.0 / 1e-5])


def test_crmt_is_the_field_total():
    steps = 30
    times = np.arange(1, steps + 1) * 30.0
    field = WellField(times_days=times,
                      injection=np.full((steps, 2), 500.0),
                      production=np.full((steps, 3), 300.0))
    total = crmt_predict(field, tau_days=150.0, gain=1.0)
    assert total.shape == (steps,)
    assert total[-1] == pytest.approx(1000.0, rel=0.05)


# ------------------------------------------------- against the simulator ----


def test_the_simulator_reproduces_an_exact_symmetry():
    """A five-spot must allocate exactly a quarter to each corner producer.
    That is an analytic result from symmetry, and it is the check that the
    truth generator is worth trusting -- everything else here is measured
    against it."""
    grid, injectors, producers = five_spot(spacing_cells=6)
    steps = 80
    case = simulate(grid, injectors, producers, np.full((steps, 1), 2000.0),
                    2200.0, days=2400.0, steps=steps)
    assert case.true_connectivity[0] == pytest.approx([0.25] * 4, abs=1e-6)
    assert case.true_connectivity.sum() == pytest.approx(1.0, abs=1e-9)


def test_tracer_allocation_nearly_closes_on_a_sealed_grid():
    """Closed boundaries mean injected water must arrive somewhere, so the row
    sums should be one -- but not exactly.

    The tracer solve is steady state and the reservoir is not: while pressure
    is still falling, some injected volume goes into storage rather than out of
    a producer, and the balance misses by that amount. It is under a percent
    here and grows with depletion, which is the honest characterisation of the
    fixture rather than a bug to tune away.
    """
    _, _, _, case = asymmetric_case(steps=60, days=1800.0)
    sums = case.true_connectivity.sum(axis=1)
    assert np.all(sums <= 1.0 + 1e-3)
    assert np.all(sums > 0.99), sums


def test_crm_recovers_the_step_response(asymmetric, step_truth):
    """The headline test, and the one the build plan's kill criterion governs.

    Target was f_ij within 0.05, kill below 0.10. Against the quantity CRM
    actually fits, it lands far inside that.
    """
    _, _, _, case = asymmetric
    field = WellField(times_days=case.times_days, injection=case.injection,
                      production=case.production)
    fit = fit_crm(field, starts=14, seed=1)

    error = np.abs(fit.parameters.connectivity - step_truth).max()
    assert error < 0.05, (
        "max |f_ij error| %.3f\nfitted:\n%s\ntruth:\n%s"
        % (error, np.round(fit.parameters.connectivity, 3), np.round(step_truth, 3)))


def test_fitted_connectivity_is_not_where_the_water_goes(asymmetric, step_truth):
    """The finding that changes how output must be labelled.

    Both matrices below are correct. They answer different questions, and on an
    ordinary connected sand they differ by a factor of three. A report that
    prints a fitted f_ij under the heading "share of injected water arriving"
    is wrong, and no fit statistic will say so.
    """
    _, _, _, case = asymmetric
    tracer = case.true_connectivity
    gap = np.abs(tracer - step_truth).max()
    assert gap > 0.20, (
        "the two definitions agreed to %.3f here, so this case no longer "
        "demonstrates the distinction" % gap)

    # Water from I-1 never reaches P-3; a rate change at I-1 very much does.
    assert tracer[0, 2] < 0.02
    assert step_truth[0, 2] > 0.15


def test_history_match_quality_does_not_prove_connectivity(asymmetric, step_truth):
    """Why the previous test matters operationally. The fit reproduces the
    history to well under a percent whichever definition you believe, so fit
    quality cannot be used to argue that the connectivity means what you hoped
    it meant."""
    _, _, _, case = asymmetric
    field = WellField(times_days=case.times_days, injection=case.injection,
                      production=case.production)
    fit = fit_crm(field, starts=10, seed=1)
    assert fit.rmse / case.production.mean() < 0.02


def test_recovered_time_constant_is_physical_and_flags_its_own_resolution(asymmetric):
    """tau must be a real response time, and the fit must say when the data
    could not resolve it.

    This reservoir responds in about a month against monthly sampling, so tau
    lands near the sampling interval. That is not a fitting failure -- it is
    the honest answer, and the useful thing is that the fit reports tau as an
    upper bound rather than quoting 34.2 days as though the data supported the
    decimal.
    """
    _, _, _, case = asymmetric
    field = WellField(times_days=case.times_days, injection=case.injection,
                      production=case.production)
    fit = fit_crm(field, starts=10, seed=1)
    span = case.times_days[-1] - case.times_days[0]
    step = float(np.median(np.diff(case.times_days)))

    assert np.all(fit.parameters.tau_days > step)
    assert np.all(fit.parameters.tau_days < 0.5 * span)
    assert any("faster than the data resolves" in note for note in fit.notes)


def test_material_balance_holds_after_fitting(asymmetric):
    """The constraint is enforced during optimisation, not clipped afterwards,
    so it must hold to solver tolerance rather than approximately."""
    _, _, _, case = asymmetric
    field = WellField(times_days=case.times_days, injection=case.injection,
                      production=case.production)
    fit = fit_crm(field, starts=8, seed=2)
    assert np.all(fit.parameters.injector_allocation <= 1.0 + 1e-6)
    assert np.all(fit.parameters.connectivity >= -1e-9)


def test_held_out_forecast_is_reported(asymmetric):
    """Fit on the first 70%, predict the last 30%. The number goes in the
    output whatever it is."""
    _, _, _, case = asymmetric
    cut = int(0.7 * case.times_days.size)
    train = WellField(times_days=case.times_days[:cut],
                      injection=case.injection[:cut],
                      production=case.production[:cut])
    fit = fit_crm(train, starts=10, seed=3)

    full = WellField(times_days=case.times_days, injection=case.injection,
                     production=case.production)
    predicted = crm_predict(full, fit.parameters)
    observed = case.production[cut:]
    mape = np.mean(np.abs(predicted[cut:] - observed) / np.maximum(observed, 1.0)) * 100
    assert mape < 15.0, "held-out MAPE %.1f%%" % mape


# ------------------------------------------------------- identifiability ----


def test_correlated_injectors_are_reported_as_inseparable():
    """Two injectors turned up and down together contribute one direction to
    the objective. The split between them is the prior, not the data, and
    saying so is the difference between a model and a fit."""
    steps = 60
    times = np.arange(1, steps + 1) * 30.0
    base = 1000 + 400 * np.sin(np.arange(steps) / 6.0)
    field = WellField(times_days=times,
                      injection=np.column_stack([base, base * 1.3]),
                      production=np.full((steps, 2), 800.0))
    notes = identifiability(field)
    assert any("correlated rate histories" in note for note in notes)


def test_a_constant_injector_is_reported():
    steps = 40
    times = np.arange(1, steps + 1) * 30.0
    field = WellField(times_days=times,
                      injection=np.column_stack([np.full(steps, 900.0),
                                                 1000 + 300 * np.sin(np.arange(steps) / 5.0)]),
                      production=np.full((steps, 2), 700.0))
    assert any("constant rate" in note for note in identifiability(field))


def test_overparameterised_fits_say_so():
    """A 4x5 pattern on 12 months is 40 connectivity parameters against 60
    observations. The fit will succeed and mean nothing."""
    steps = 12
    rng = np.random.default_rng(0)
    times = np.arange(1, steps + 1) * 30.0
    field = WellField(times_days=times,
                      injection=rng.uniform(500, 1500, (steps, 4)),
                      production=rng.uniform(400, 900, (steps, 5)))
    fit = fit_crm(field, starts=3, seed=0)
    assert any("fit noise" in note for note in fit.notes)


def test_productivity_without_pressure_is_refused(asymmetric):
    """J only enters through the pressure term. Asking for it without
    bottomhole pressures must return a note, not a number."""
    _, _, _, case = asymmetric
    field = WellField(times_days=case.times_days, injection=case.injection,
                      production=case.production)
    fit = fit_crm(field, starts=3, seed=0, fit_productivity=True)
    assert fit.parameters.productivity is None
    assert any("not identifiable" in note for note in fit.notes)


# -------------------------------------------------------------- priors -----


def test_distance_prior_favours_near_wells_and_sums_to_one():
    injector_xy = np.array([[0.0, 0.0]])
    producer_xy = np.array([[100.0, 0.0], [2000.0, 0.0]])
    prior = distance_prior(injector_xy, producer_xy, length_scale=500.0)
    assert prior[0, 0] > prior[0, 1]
    assert prior.sum(axis=1) == pytest.approx([1.0])


def test_distance_prior_rejects_a_nonsense_length_scale():
    with pytest.raises(ValueError):
        distance_prior(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), 0.0)


# ----------------------------------------------------------- uncertainty ---


def test_bootstrap_gives_bands_that_bracket_the_fit(asymmetric):
    """Blocks rather than individual samples: a production series is heavily
    autocorrelated, and resampling single months destroys the time structure
    the time constant is fitted from."""
    _, _, _, case = asymmetric
    field = WellField(times_days=case.times_days[:60],
                      injection=case.injection[:60],
                      production=case.production[:60])
    fit = fit_crm(field, starts=4, seed=0)
    p10, p90 = bootstrap_connectivity(field, resamples=6, seed=0, starts=2)

    assert p10.shape == fit.parameters.connectivity.shape
    assert np.all(p90 >= p10 - 1e-9)


# -------------------------------------------------------------- guards -----


def test_bad_histories_are_rejected():
    times = np.arange(1, 11) * 30.0
    with pytest.raises(ValueError, match="non-finite"):
        WellField(times_days=times,
                  injection=np.full((10, 1), np.nan),
                  production=np.full((10, 1), 500.0))
    with pytest.raises(ValueError, match="negative"):
        WellField(times_days=times,
                  injection=np.full((10, 1), -5.0),
                  production=np.full((10, 1), 500.0))
    with pytest.raises(ValueError, match="increase"):
        WellField(times_days=times[::-1],
                  injection=np.full((10, 1), 500.0),
                  production=np.full((10, 1), 500.0))
    with pytest.raises(ValueError, match="three timesteps"):
        WellField(times_days=np.array([1.0, 2.0]),
                  injection=np.full((2, 1), 500.0),
                  production=np.full((2, 1), 500.0))


def test_negative_time_constants_are_refused():
    with pytest.raises(ValueError):
        CRMParameters(np.array([[0.5]]), np.array([-10.0]))
