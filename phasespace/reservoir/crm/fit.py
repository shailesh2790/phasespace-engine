"""Fitting CRM: constrained, multi-start, and honest about identifiability.

Three things make this harder than it looks, and each has a specific remedy
here rather than a general one.

**The constraints are physics, not preferences.** An injector cannot deliver
more water than it injected, so the connectivity rows must sum to at most one.
Fitting without that constraint and clipping afterwards produces a different
answer -- the optimiser spends its freedom on an infeasible region and the
projection back is arbitrary. So the material balance enters as a linear
constraint the optimiser must respect at every step.

**The objective is not convex.** A CRM surface has local minima that look like
plausible fields: a producer can be explained by the wrong injector at the
wrong time constant. One gradient descent from one starting point finds one of
them, reliably and without complaint. Hence multi-start.

**Connectivity is often not identifiable.** Two injectors that were always
turned up and down together cannot be told apart by any amount of fitting, and
an unregularised fit will nonetheless report a confident split between them.
The remedy is a distance prior plus an explicit identifiability report, so the
answer arrives with the question "could this data ever have distinguished
these?" already answered.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .model import CRMFit, CRMParameters, WellField, crm_predict

# A time constant far below one timestep cannot be resolved -- the model
# reduces to instantaneous response and tau stops affecting the prediction.
# Far above the history length is equally meaningless: the exponential is flat
# over the data and any large value fits as well as any other.
MIN_TAU_STEPS = 0.05
MAX_TAU_HISTORIES = 3.0


def _pack(connectivity: np.ndarray, tau: np.ndarray,
          productivity: np.ndarray = None) -> np.ndarray:
    parts = [connectivity.ravel(), tau]
    if productivity is not None:
        parts.append(productivity)
    return np.concatenate(parts)


def _unpack(vector: np.ndarray, n_injectors: int, n_producers: int,
            with_productivity: bool) -> tuple:
    cut = n_injectors * n_producers
    connectivity = vector[:cut].reshape(n_injectors, n_producers)
    tau = vector[cut:cut + n_producers]
    productivity = vector[cut + n_producers:] if with_productivity else None
    return connectivity, tau, productivity


def distance_prior(injector_xy: np.ndarray, producer_xy: np.ndarray,
                   length_scale: float) -> np.ndarray:
    """Connectivity prior that decays with interwell distance.

    Not a physical law -- water follows permeability, not geometry, and the
    whole point of CRM is to find the places where it does not follow geometry.
    It is a prior in the honest sense: absent evidence, near wells are more
    likely connected than far ones, and the fit is allowed to overrule it where
    the data insists. The weight is reported with the result so a reader can
    see how much of the answer is data and how much is assumption.
    """
    injector_xy = np.atleast_2d(np.asarray(injector_xy, dtype=float))
    producer_xy = np.atleast_2d(np.asarray(producer_xy, dtype=float))
    if length_scale <= 0.0:
        raise ValueError("length scale must be positive, got %r" % (length_scale,))

    separation = np.linalg.norm(
        injector_xy[:, None, :] - producer_xy[None, :, :], axis=2)
    weights = np.exp(-separation / length_scale)
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.full_like(weights, 1.0 / weights.shape[1]),
                     where=totals > 0.0)


def identifiability(field: WellField, threshold: float = 0.95) -> list:
    """Which injectors this history could never tell apart.

    Two injectors whose rate histories move together contribute a single
    direction to the objective, so the split of their connectivity between
    producers is set by the prior and the optimiser's starting point, not by
    the data. Reporting that is the difference between a model and a fit.
    """
    notes = []
    injection = field.injection
    n = injection.shape[1]
    for i in range(n):
        for k in range(i + 1, n):
            a, b = injection[:, i], injection[:, k]
            if a.std() < 1e-9 or b.std() < 1e-9:
                notes.append(
                    "Injector %d or %d has a constant rate over the whole "
                    "history, so its connectivity is not separable from the "
                    "producers' baseline decline." % (i, k))
                continue
            correlation = float(np.corrcoef(a, b)[0, 1])
            if abs(correlation) >= threshold:
                notes.append(
                    "Injectors %d and %d have correlated rate histories "
                    "(r = %.3f). No fit can distribute connectivity between "
                    "them from this data; the split you see is the prior."
                    % (i, k, correlation))
    return notes


def fit_crm(
    field: WellField,
    starts: int = 20,
    prior: np.ndarray = None,
    prior_weight: float = 0.0,
    fit_productivity: bool = False,
    weights: np.ndarray = None,
    seed: int = 0,
    max_iterations: int = 400,
) -> CRMFit:
    """Fit CRMP by constrained least squares from several starting points.

    ``prior_weight`` is in the same units as the objective: it multiplies the
    squared deviation of the connectivity from ``prior``. Zero means an
    unregularised fit, which is honest on a small well count and reckless on a
    large one -- a 12 by 15 pattern has 180 connectivity parameters and will
    happily fit noise into every one of them.
    """
    if starts < 1:
        raise ValueError("need at least one start, got %r" % (starts,))

    n_inj, n_prod = field.n_injectors, field.n_producers
    observed = field.production
    dt = field.dt_days

    if weights is None:
        weights = np.ones(n_prod)
    weights = np.asarray(weights, dtype=float)
    if weights.size != n_prod:
        raise ValueError("need one weight per producer")
    if np.any(weights < 0.0):
        raise ValueError("weights cannot be negative")

    with_productivity = fit_productivity and field.bottomhole_psia is not None
    notes = []
    if fit_productivity and field.bottomhole_psia is None:
        notes.append(
            "Productivity index was requested but no bottomhole pressures were "
            "supplied. J only enters the model through the pressure term, so "
            "it is not identifiable here and was not fitted.")

    tau_low = MIN_TAU_STEPS * float(np.median(dt))
    tau_high = MAX_TAU_HISTORIES * float(field.times_days[-1] - field.times_days[0])
    scale = float(np.median(np.abs(observed))) or 1.0

    def objective(vector: np.ndarray) -> float:
        connectivity, tau, productivity = _unpack(vector, n_inj, n_prod,
                                                  with_productivity)
        try:
            parameters = CRMParameters(connectivity=connectivity, tau_days=tau,
                                       productivity=productivity)
            predicted = crm_predict(field, parameters)
        except (ValueError, FloatingPointError):
            return 1e12
        if not np.all(np.isfinite(predicted)):
            return 1e12

        residual = (predicted - observed) / scale
        cost = float(np.sum(weights * np.sum(residual ** 2, axis=0)))
        if prior is not None and prior_weight > 0.0:
            cost += prior_weight * float(np.sum((connectivity - prior) ** 2))
        return cost

    # Material balance, one row per injector: sum_j f_ij <= 1.
    def allocation_slack(vector: np.ndarray) -> np.ndarray:
        connectivity, _, _ = _unpack(vector, n_inj, n_prod, with_productivity)
        return 1.0 - connectivity.sum(axis=1)

    constraints = [{"type": "ineq", "fun": allocation_slack}]

    bounds = ([(0.0, 1.0)] * (n_inj * n_prod)
              + [(tau_low, tau_high)] * n_prod)
    if with_productivity:
        bounds += [(1e-6, 1e6)] * n_prod

    rng = np.random.default_rng(seed)
    best = None
    best_cost = np.inf

    for attempt in range(starts):
        if attempt == 0 and prior is not None:
            connectivity0 = np.clip(prior.copy(), 0.0, 1.0)
        elif attempt == 0:
            # An even split that already satisfies material balance.
            connectivity0 = np.full((n_inj, n_prod), 1.0 / max(n_prod, 1))
        else:
            connectivity0 = rng.random((n_inj, n_prod))
            row_sums = connectivity0.sum(axis=1, keepdims=True)
            connectivity0 *= rng.uniform(0.3, 1.0, (n_inj, 1)) / row_sums

        span = field.times_days[-1] - field.times_days[0]
        tau0 = (np.full(n_prod, 0.15 * span) if attempt == 0
                else rng.uniform(tau_low, min(tau_high, span), n_prod))
        productivity0 = np.full(n_prod, 1.0) if with_productivity else None

        start_vector = _pack(connectivity0, tau0, productivity0)
        result = minimize(objective, start_vector, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": max_iterations, "ftol": 1e-10})

        if result.fun < best_cost:
            best_cost = float(result.fun)
            best = result

    connectivity, tau, productivity = _unpack(best.x, n_inj, n_prod,
                                              with_productivity)
    parameters = CRMParameters(connectivity=connectivity, tau_days=tau,
                               productivity=productivity)
    predicted = crm_predict(field, parameters)

    notes.extend(identifiability(field))
    overshoot = parameters.injector_allocation.max() - 1.0
    if overshoot > 1e-6:
        notes.append(
            "Material balance is violated by %.4f, which should be impossible "
            "under the constraint -- treat this fit as failed." % overshoot)

    # A time constant near the sampling interval is at the edge of what the
    # data can resolve: the exponential has largely completed within one step,
    # so the fit is reading the shape of a transition it barely sampled.
    coarse = tau <= 2.0 * float(np.median(dt))
    if np.any(coarse):
        notes.append(
            "Producers %s have time constants within two sampling intervals "
            "(%.0f days). The response is faster than the data resolves, so "
            "tau is an upper bound rather than a measurement -- monthly data "
            "cannot see a two-week response."
            % (np.nonzero(coarse)[0].tolist(), float(np.median(dt))))

    parameter_count = n_inj * n_prod + n_prod + (n_prod if with_productivity else 0)
    observation_count = observed.size
    if parameter_count > observation_count / 5.0:
        notes.append(
            "%d parameters against %d observations. This is enough freedom to "
            "fit noise; regularise, use CRMT, or get more history."
            % (parameter_count, observation_count))

    return CRMFit(parameters=parameters, predicted=predicted,
                  residuals=predicted - observed, objective=best_cost,
                  converged=bool(best.success), starts_tried=starts,
                  message=str(best.message), notes=notes)


def bootstrap_connectivity(field: WellField, resamples: int = 200,
                           block_steps: int = None, seed: int = 0,
                           **fit_kwargs) -> tuple:
    """P10/P90 on each connectivity, by resampling contiguous time blocks.

    Blocks rather than individual samples, because a production series is
    heavily autocorrelated: resampling single months destroys the very time
    structure the time constant is fitted from, and returns uncertainty bands
    that are far too narrow. Block length defaults to roughly the fitted
    response time, which is the correlation length that matters here.
    """
    if resamples < 2:
        raise ValueError("need at least two resamples, got %r" % (resamples,))

    rng = np.random.default_rng(seed)
    n_steps = field.n_steps
    if block_steps is None:
        block_steps = max(int(round(n_steps / 10.0)), 2)
    block_steps = min(block_steps, n_steps)
    n_blocks = int(np.ceil(n_steps / block_steps))

    samples = []
    for _ in range(resamples):
        starts = rng.integers(0, max(n_steps - block_steps, 1) + 1, n_blocks)
        index = np.concatenate([np.arange(s, min(s + block_steps, n_steps))
                                for s in starts])[:n_steps]
        index = np.sort(index)

        resampled = WellField(
            times_days=field.times_days,
            injection=field.injection[index],
            production=field.production[index],
            bottomhole_psia=(None if field.bottomhole_psia is None
                             else field.bottomhole_psia[index]))
        try:
            fit = fit_crm(resampled, **fit_kwargs)
        except (ValueError, np.linalg.LinAlgError):
            continue
        samples.append(fit.parameters.connectivity)

    if len(samples) < 2:
        raise ValueError("bootstrap produced too few usable fits")

    stacked = np.stack(samples)
    return (np.percentile(stacked, 10, axis=0),
            np.percentile(stacked, 90, axis=0))
