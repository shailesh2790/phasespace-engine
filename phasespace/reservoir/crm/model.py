"""The capacitance-resistance forward model.

A waterflood as a set of tanks. Each producer drains a pore volume that is
charged by some share of each injector's water and discharged through its own
productivity index. Two numbers describe a producer: how much of each
injector's water reaches it (``f_ij``, the connectivity) and how long it takes
to respond (``tau``, the time constant). That is the whole model, and it is why
it can be fitted from production data alone -- no grid, no geology, no seismic,
no simulation deck.

The discrete form, for producer j over a timestep of length dt:

    q_j(n) = q_j(n-1) * E
             + (1 - E) * [ sum_i f_ij * i_i(n) - J_j * tau_j * dp_wf/dt ]

    E = exp(-dt / tau_j)

Everything about the model's behaviour is in that exponential. E near 1 is a
large, slow tank that barely notices a change in injection; E near 0 is a
producer that tracks its injectors within a month. Fitting CRM is, in essence,
asking each producer how much of the reservoir it is talking to.

Three physical constraints are not optional, and are enforced in the fit rather
than applied afterwards by clipping:

    f_ij >= 0            water does not flow backwards up an injector
    sum_j f_ij <= 1      an injector cannot deliver more than it injected
    tau_j, J_j > 0       a negative time constant is not slow, it is unstable

The middle one earns its keep. A row summing to less than one means injected
water is leaving the pattern -- to an aquifer, across a lease line, or up a
thief zone -- and quantifying that is a defensible operational finding rather
than a modelling artefact.

Rates are volumes per unit time in whatever consistent units the caller uses
(bbl/day throughout this codebase); time is in days.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WellField:
    """The observed history: who injected what, who produced what, and where.

    ``injection`` is (n_steps, n_injectors) and ``production`` is
    (n_steps, n_producers), both as rates rather than volumes, sampled on the
    same time base.

    ``bottomhole_psia`` is optional because most operators do not have it. Its
    absence is not fatal -- it collapses the pressure term and the fit then
    attributes all rate change to injection, which is the standard "CRM without
    pressure" and is what the literature mostly reports. It does mean J is no
    longer identifiable, and the fit says so rather than returning a number.
    """

    times_days: np.ndarray
    injection: np.ndarray
    production: np.ndarray
    bottomhole_psia: np.ndarray = None
    injector_names: list = field(default_factory=list)
    producer_names: list = field(default_factory=list)
    injector_xy: np.ndarray = None
    producer_xy: np.ndarray = None

    def __post_init__(self):
        self.times_days = np.asarray(self.times_days, dtype=float)
        self.injection = np.atleast_2d(np.asarray(self.injection, dtype=float))
        self.production = np.atleast_2d(np.asarray(self.production, dtype=float))

        if self.times_days.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if self.injection.shape[0] != self.times_days.size:
            raise ValueError(
                "injection has %d rows but there are %d timesteps"
                % (self.injection.shape[0], self.times_days.size))
        if self.production.shape[0] != self.times_days.size:
            raise ValueError(
                "production has %d rows but there are %d timesteps"
                % (self.production.shape[0], self.times_days.size))
        if self.times_days.size < 3:
            raise ValueError("need at least three timesteps to fit a time constant")
        if np.any(np.diff(self.times_days) <= 0.0):
            raise ValueError("times must increase strictly")
        if np.any(self.injection < 0.0):
            raise ValueError("injection rates cannot be negative")
        if not np.all(np.isfinite(self.injection)):
            # A gap in a historian export arrives as NaN, and NaN comparisons
            # are false, so it passes every range check and then silently
            # poisons the objective into nan -- which the optimiser reports as
            # a converged fit at an arbitrary point.
            raise ValueError("injection contains non-finite values")
        if not np.all(np.isfinite(self.production)):
            raise ValueError("production contains non-finite values")

        if self.bottomhole_psia is not None:
            self.bottomhole_psia = np.atleast_2d(
                np.asarray(self.bottomhole_psia, dtype=float))
            if self.bottomhole_psia.shape != self.production.shape:
                raise ValueError(
                    "bottomhole pressure must match production shape %s, got %s"
                    % (self.production.shape, self.bottomhole_psia.shape))

    @property
    def n_steps(self) -> int:
        return self.times_days.size

    @property
    def n_injectors(self) -> int:
        return self.injection.shape[1]

    @property
    def n_producers(self) -> int:
        return self.production.shape[1]

    @property
    def dt_days(self) -> np.ndarray:
        """Step lengths, with the first inferred from the second.

        The first sample has no predecessor, so its step is undefined. Taking
        it from the next one is the least-wrong choice on a regular series and
        matters little on any series long enough to fit.
        """
        steps = np.diff(self.times_days)
        return np.concatenate([steps[:1], steps])


@dataclass
class CRMParameters:
    """What a fit recovers.

    ``connectivity`` is (n_injectors, n_producers). ``tau_days`` and
    ``productivity`` are per producer.
    """

    connectivity: np.ndarray
    tau_days: np.ndarray
    productivity: np.ndarray = None

    def __post_init__(self):
        self.connectivity = np.atleast_2d(np.asarray(self.connectivity, dtype=float))
        self.tau_days = np.atleast_1d(np.asarray(self.tau_days, dtype=float))
        if self.productivity is not None:
            self.productivity = np.atleast_1d(
                np.asarray(self.productivity, dtype=float))
        if np.any(self.tau_days <= 0.0):
            raise ValueError("time constants must be positive, got %r" % (self.tau_days,))
        if self.connectivity.shape[1] != self.tau_days.size:
            raise ValueError(
                "connectivity has %d producers but %d time constants were given"
                % (self.connectivity.shape[1], self.tau_days.size))

    @property
    def injector_allocation(self) -> np.ndarray:
        """Row sums: the share of each injector's water that is accounted for.

        Below one means water is leaving the pattern. This is the number worth
        putting in front of an operator, because it converts directly into
        barrels of injection doing no work.
        """
        return self.connectivity.sum(axis=1)

    @property
    def unallocated_fraction(self) -> np.ndarray:
        return np.clip(1.0 - self.injector_allocation, 0.0, 1.0)

    def pore_volume_bbl(self, total_compressibility_psi: float) -> np.ndarray:
        """Back out drainage pore volume from tau = c_t * V_p / J.

        A sanity check with real teeth: a time constant that implies a pore
        volume ten times the mapped drainage area is a fit that has found
        something other than the reservoir.
        """
        if self.productivity is None:
            raise ValueError(
                "pore volume needs the productivity index, which is only "
                "identifiable when bottomhole pressures were supplied")
        if total_compressibility_psi <= 0.0:
            raise ValueError("compressibility must be positive, got %r"
                             % (total_compressibility_psi,))
        return self.tau_days * self.productivity / total_compressibility_psi


def crm_predict(field: WellField, parameters: CRMParameters,
                initial_rates: np.ndarray = None) -> np.ndarray:
    """Run the forward model. Returns (n_steps, n_producers) of rates.

    The recursion is sequential in time and vectorised across producers, which
    is the only axis worth vectorising: a field has tens of producers and
    hundreds of timesteps, and the timestep loop carries a genuine dependency.
    """
    connectivity = parameters.connectivity
    if connectivity.shape[0] != field.n_injectors:
        raise ValueError(
            "connectivity has %d injectors but the field has %d"
            % (connectivity.shape[0], field.n_injectors))
    if connectivity.shape[1] != field.n_producers:
        raise ValueError(
            "connectivity has %d producers but the field has %d"
            % (connectivity.shape[1], field.n_producers))

    tau = parameters.tau_days
    dt = field.dt_days
    rates = np.empty((field.n_steps, field.n_producers), dtype=float)

    previous = (field.production[0].copy() if initial_rates is None
                else np.asarray(initial_rates, dtype=float).copy())
    if previous.size != field.n_producers:
        raise ValueError("initial rates must have one value per producer")

    # The pressure-support term, when bottomhole pressures are available.
    pressure_term = np.zeros((field.n_steps, field.n_producers))
    if field.bottomhole_psia is not None and parameters.productivity is not None:
        dp = np.diff(field.bottomhole_psia, axis=0)
        dp = np.vstack([np.zeros((1, field.n_producers)), dp])
        with np.errstate(divide="ignore", invalid="ignore"):
            pressure_term = (parameters.productivity * tau
                             * np.divide(dp, dt[:, None],
                                         out=np.zeros_like(dp),
                                         where=dt[:, None] != 0.0))

    for n in range(field.n_steps):
        decay = np.exp(-dt[n] / tau)
        support = field.injection[n] @ connectivity
        previous = previous * decay + (1.0 - decay) * (support - pressure_term[n])
        rates[n] = previous

    return rates


def crmt_predict(field: WellField, tau_days: float,
                 gain: float = 1.0, initial_rate: float = None) -> np.ndarray:
    """The whole-field tank: one time constant, all injection into one pot.

    Deliberately the first thing to run on any dataset. It has two parameters,
    so it cannot overfit anything, and its time constant is the right order of
    magnitude to start a CRMP fit from. If CRMT cannot follow the field total,
    a per-producer fit is fitting noise and the problem is the data.
    """
    if tau_days <= 0.0:
        raise ValueError("time constant must be positive, got %r" % (tau_days,))

    total_injection = field.injection.sum(axis=1, keepdims=True)
    total_production = field.production.sum(axis=1, keepdims=True)
    tank = WellField(times_days=field.times_days,
                     injection=total_injection,
                     production=total_production)
    parameters = CRMParameters(connectivity=np.array([[gain]]),
                               tau_days=np.array([tau_days]))
    start = None if initial_rate is None else np.array([initial_rate])
    return crm_predict(tank, parameters, initial_rates=start)[:, 0]


@dataclass
class CRMFit:
    """A fitted model and everything needed to judge it."""

    parameters: CRMParameters
    predicted: np.ndarray
    residuals: np.ndarray
    objective: float
    converged: bool
    starts_tried: int
    message: str = ""
    connectivity_p10: np.ndarray = None
    connectivity_p90: np.ndarray = None
    notes: list = field(default_factory=list)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2)))

    def mape(self, observed: np.ndarray, floor: float = 1.0) -> float:
        """Mean absolute percentage error, with a floor on the denominator.

        Without the floor a single near-zero observed rate -- a well down for a
        month -- contributes a percentage error in the thousands and swamps the
        statistic for the whole field.
        """
        observed = np.asarray(observed, dtype=float)
        denominator = np.maximum(np.abs(observed), floor)
        return float(np.mean(np.abs(self.predicted - observed) / denominator) * 100.0)
