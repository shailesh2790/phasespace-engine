"""Sharing a limited lift-gas supply between wells.

The question is always the same and the compressor never gets bigger: there is
so much gas a day, every well wants more, and more is not always better --
past a well's optimum the extra gas adds friction faster than it lightens the
column, and the well makes *less* oil for more gas.

Three steps, each checkable:

  1. Every well gets a gas-lift performance curve, oil against injection.
  2. Each curve is replaced by its upper concave envelope, so "one more unit of
     gas" has a well-defined, non-increasing marginal gain.
  3. Gas is handed out in increments to whichever well has the highest marginal
     gain, ending at the equal-slope condition: every well receiving gas has
     the same marginal oil per unit of gas. That common slope is the shadow
     price, and it is the most useful number the calculation produces -- it
     prices the next thousand cubic metres of compression.

Curves come from one of two places, and the difference matters:

  **Fitted to the well's own tests**, with the Alarcon et al. (2002) form
  q_o = a + b*q_i + c*q_i^2 + d*ln(q_i + 1). Linear in its coefficients, so
  the fit is a least-squares solve with no starting guess. It needs at least
  four tests and says so.

  **Computed from the engine's nodal model**, which needs no test history at
  all -- a new well, a well that has never been tested at more than one
  injection rate, or a well whose tests are suspect. This is the path a
  single-well tool cannot offer, and running both on the same well is a real
  cross-check: a fitted curve that disagrees with the physics is either a bad
  fit or a well that is not behaving like its model.

One honest limitation, stated here because the prototype this replaces claimed
otherwise. Greedy allocation on concave envelopes is *exactly* optimal when
wells can take any injection rate from zero upwards. Real wells have a minimum
stable injection: below it they head, slug, or simply do not lift, so a well is
either off or on at q_min or more. That makes the choice partly discrete, and
greedy is then a strong heuristic rather than a proof. `allocate` says so in
its result, and the tests bound the gap against dense search rather than
asserting optimality that no longer holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Four coefficients, so four tests is the arithmetic minimum for a fit with no
# degrees of freedom left over. Fewer is not a loose fit, it is an
# underdetermined system that will return something confident and meaningless.
MIN_TESTS = 4

# A fit whose residual exceeds this share of the well's largest tested oil rate
# is reported as loose. Not an error: a loose curve on a well with scattered
# tests is information about the tests.
LOOSE_FIT_FRACTION = 0.05


@dataclass
class GasLiftWell:
    """One well's response to lift gas, and the limits it must respect."""

    name: str
    # (injection, oil) pairs in whatever consistent units the caller uses.
    tests: list = field(default_factory=list)
    # Minimum stable injection. A well gets at least this much gas or none:
    # between zero and q_min it heads rather than lifts, and the curve there
    # describes a flow regime the well is not in.
    q_min: float = 0.0
    q_max: float = 30000.0
    current: float = None
    # Set by fit(), or supplied directly by a curve from the nodal model.
    coefficients: np.ndarray = field(default=None, repr=False)
    curve_points: list = field(default=None, repr=False)
    rms: float = 0.0
    source: str = "tests"
    tested_low: float = 0.0
    tested_high: float = 0.0

    def __post_init__(self):
        if self.q_max <= 0.0:
            raise ValueError("%s: maximum injection must be positive, got %r"
                             % (self.name, self.q_max))
        if self.q_min < 0.0:
            raise ValueError("%s: minimum injection cannot be negative, got %r"
                             % (self.name, self.q_min))
        if self.q_min > self.q_max:
            raise ValueError("%s: minimum injection %g exceeds maximum %g"
                             % (self.name, self.q_min, self.q_max))

    # ------------------------------------------------------------ fitting --

    def fit(self) -> "GasLiftWell":
        """Least squares on the Alarcon form, from this well's own tests."""
        if len(self.tests) < MIN_TESTS:
            raise ValueError(
                "%s: fitting the four-coefficient curve needs at least %d well "
                "tests, got %d. Fewer is not a loose fit, it is an "
                "underdetermined system that returns a confident meaningless "
                "answer." % (self.name, MIN_TESTS, len(self.tests)))

        injection = np.asarray([t[0] for t in self.tests], dtype=float)
        oil = np.asarray([t[1] for t in self.tests], dtype=float)
        if np.any(injection < 0.0):
            raise ValueError("%s: negative injection rate in the tests" % self.name)
        if not np.all(np.isfinite(injection)) or not np.all(np.isfinite(oil)):
            raise ValueError("%s: tests contain non-finite values" % self.name)

        design = np.column_stack([np.ones_like(injection), injection,
                                  injection ** 2, np.log(injection + 1.0)])
        self.coefficients, *_ = np.linalg.lstsq(design, oil, rcond=None)
        self.rms = float(np.sqrt(np.mean((design @ self.coefficients - oil) ** 2)))
        self.tested_low = float(injection.min())
        self.tested_high = float(injection.max())
        self.source = "tests"
        return self

    @classmethod
    def from_curve(cls, name: str, points: list, **kwargs) -> "GasLiftWell":
        """A well whose curve came from the nodal model rather than from tests.

        The points are used directly, with no functional form imposed on them.
        Fitting a four-parameter curve to a curve the physics already produced
        would only add error, and would hide the shape the model actually
        predicted -- including the sharp fall past the optimum that the Alarcon
        form, being quadratic, rounds off.
        """
        ordered = sorted((float(x), float(y)) for x, y in points)
        if len(ordered) < 2:
            raise ValueError("%s: a curve needs at least two points" % name)
        well = cls(name=name, tests=[], **kwargs)
        well.curve_points = ordered
        well.source = "nodal model"
        well.tested_low = ordered[0][0]
        well.tested_high = ordered[-1][0]
        return well

    # ----------------------------------------------------------- the curve --

    @property
    def ready(self) -> bool:
        return self.coefficients is not None or self.curve_points is not None

    def oil(self, injection) -> np.ndarray:
        """Oil rate at an injection rate. Never negative."""
        if not self.ready:
            raise ValueError("%s: fit the curve before evaluating it" % self.name)
        q = np.asarray(injection, dtype=float)

        if self.curve_points is not None:
            xs = np.asarray([p[0] for p in self.curve_points])
            ys = np.asarray([p[1] for p in self.curve_points])
            # Clamped, not extrapolated. Past the last computed point the
            # model has said nothing, and a straight line drawn onward is the
            # tool inventing deliverability.
            return np.maximum(0.0, np.interp(q, xs, ys))

        a, b, c, d = self.coefficients
        return np.maximum(0.0, a + b * q + c * q ** 2 + d * np.log(q + 1.0))

    def equation(self) -> str:
        if self.curve_points is not None:
            return ("tabulated from the nodal model, %d points over %g-%g"
                    % (len(self.curve_points), self.tested_low, self.tested_high))
        a, b, c, d = self.coefficients
        return ("q_o = %.4g + %.4g*q_i + %.4g*q_i^2 + %.4g*ln(q_i+1)"
                % (a, b, c, d))

    def warnings(self, allocation: float) -> list:
        """What to distrust about this well's answer."""
        notes = []
        if allocation > 0.0 and not (self.tested_low <= allocation <= self.tested_high):
            notes.append(
                "%g is outside the range the curve covers (%g to %g) — confirm "
                "with a test before setting it"
                % (allocation, self.tested_low, self.tested_high))
        if self.tests and self.rms > LOOSE_FIT_FRACTION * max(
                [t[1] for t in self.tests] + [1e-9]):
            notes.append("curve fit is loose (rms %.3g) — add tests" % self.rms)
        return notes


# ---------------------------------------------------------------- envelope --


def concave_envelope(x, y) -> tuple:
    """Upper concave hull of a curve, as (x, y) arrays.

    Marginal gain has to be non-increasing for a greedy allocation to be
    meaningful: if a well's curve has a flat spot followed by a steeper rise,
    greedy would stop at the flat spot and miss the better region beyond.
    Replacing the curve by its upper concave hull removes that trap -- the hull
    segments are the best achievable mixtures, and their slopes fall
    monotonically by construction.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size != ys.size:
        raise ValueError("x and y must be the same length")
    if xs.size == 0:
        raise ValueError("an envelope needs at least one point")

    hull = []
    for point in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # Cross product: drop the middle point when it sits under the chord.
            if (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1) >= 0.0:
                hull.pop()
            else:
                break
        hull.append(point)
    hx, hy = zip(*hull)
    return np.asarray(hx), np.asarray(hy)


# -------------------------------------------------------------- allocation --


@dataclass
class WellAllocation:
    name: str
    injection: float
    oil: float
    previous_injection: float
    previous_oil: float
    warnings: list = field(default_factory=list)

    @property
    def oil_change(self) -> float:
        return self.oil - self.previous_oil

    @property
    def injection_change(self) -> float:
        return self.injection - self.previous_injection


@dataclass
class Allocation:
    wells: list
    gas_available: float
    gas_used: float
    shadow_price: float
    exactly_optimal: bool
    notes: list = field(default_factory=list)

    @property
    def gas_unused(self) -> float:
        return self.gas_available - self.gas_used

    @property
    def total_oil(self) -> float:
        return sum(w.oil for w in self.wells)

    @property
    def previous_oil(self) -> float:
        return sum(w.previous_oil for w in self.wells)

    @property
    def oil_gain(self) -> float:
        return self.total_oil - self.previous_oil


def allocate(
    wells: list,
    gas_available: float,
    step: float = 250.0,
    oil_price: float = 1.0,
    gas_price: float = 0.0,
) -> Allocation:
    """Share the gas. Returns the allocation and the shadow price.

    ``gas_price`` in the same value units as ``oil_price`` turns the objective
    from volume into value, and with it a well stops receiving gas once the
    marginal oil no longer pays for the marginal gas. That is the difference
    between maximising production and maximising profit, and on a field with
    expensive compression they are not the same allocation.
    """
    if gas_available < 0.0:
        raise ValueError("gas available cannot be negative, got %r" % (gas_available,))
    if step <= 0.0:
        raise ValueError("step must be positive, got %r" % (step,))
    if not wells:
        raise ValueError("no wells supplied")

    for well in wells:
        if not well.ready:
            well.fit()

    envelopes = {}
    for well in wells:
        # The grid starts at zero and then jumps to q_min: there is no such
        # thing as injecting half the minimum stable rate, so no grid point
        # should imply there is.
        lowest = max(well.q_min, step)
        grid = np.concatenate([[0.0], np.arange(lowest, well.q_max + step / 2.0, step)])
        grid = grid[grid <= well.q_max + 1e-9]
        value = oil_price * well.oil(grid) - gas_price * grid
        value[0] = oil_price * float(well.oil(0.0))   # natural flow, no gas cost
        envelopes[well.name] = concave_envelope(grid, value)

    allocated = {w.name: 0.0 for w in wells}
    segment = {w.name: 0 for w in wells}
    remaining = float(gas_available)
    shadow_price = 0.0

    while remaining > 1e-9:
        best = None
        best_slope = 0.0
        for well in wells:
            hx, hy = envelopes[well.name]
            k = segment[well.name]
            if k + 1 >= len(hx):
                continue
            need = hx[k + 1] - hx[k]
            # A step that cannot be completed must not be taken partially when
            # doing so would leave the well below its minimum stable
            # injection. The prototype this replaces allocated the remainder
            # regardless, and handed back setpoints like 600 sm3/d on a well
            # whose minimum is 1500 -- a rate at which the well heads instead
            # of lifting, so the oil it predicted there was fiction.
            if need > remaining and hx[k] <= 0.0 and well.q_min > 0.0:
                continue
            slope = (hy[k + 1] - hy[k]) / need
            if slope > best_slope:
                best, best_slope = well, slope

        if best is None:
            break

        hx, _ = envelopes[best.name]
        k = segment[best.name]
        need = hx[k + 1] - hx[k]
        if need <= remaining:
            allocated[best.name] = float(hx[k + 1])
            segment[best.name] += 1
            remaining -= need
        else:
            allocated[best.name] += remaining
            remaining = 0.0
        shadow_price = best_slope

    results = []
    for well in wells:
        injection = allocated[well.name]
        previous = well.current if well.current is not None else 0.0
        results.append(WellAllocation(
            name=well.name,
            injection=injection,
            oil=float(well.oil(injection)),
            previous_injection=previous,
            previous_oil=float(well.oil(previous)),
            warnings=well.warnings(injection),
        ))

    # Greedy on concave envelopes is exactly optimal only when every well can
    # take any rate from zero. A minimum stable injection makes the first step
    # a discrete jump, and comparing its average slope against other wells'
    # marginals is a heuristic.
    constrained = any(w.q_min > 0.0 for w in wells)
    notes = []
    if constrained:
        notes.append(
            "Some wells have a minimum stable injection, so the first increment "
            "is a discrete jump rather than a marginal step. The allocation is "
            "a strong heuristic here, not a proof of optimality.")
    unused = gas_available - (gas_available - remaining)
    if remaining > 1e-6:
        notes.append(
            "%g of gas was left unallocated: no well could take it without "
            "falling below its minimum stable injection or exceeding its "
            "valve limit." % remaining)
    del unused

    return Allocation(wells=results, gas_available=gas_available,
                      gas_used=gas_available - remaining,
                      shadow_price=shadow_price,
                      exactly_optimal=not constrained, notes=notes)


# ------------------------------------------------- curves from the engine --


def curve_from_nodal(
    name: str,
    ipr,
    tubing,
    wellhead_pressure_psia: float,
    formation_gor_scf_stb: float,
    injection_rates,
    water_cut: float = 0.0,
    oil=None,
    gas=None,
    q_min: float = 0.0,
    q_max: float = None,
    current: float = None,
) -> GasLiftWell:
    """Build a well's gas-lift curve from the nodal model, not from tests.

    This is what a tool with no well model cannot do. A well with one test, or
    none, or tests an engineer does not trust, still has an IPR and a tubing
    string -- and the same nodal analysis that answers "what does this well
    make at this wellhead pressure" answers "what does it make at this
    injection rate". Allocation then covers the whole field rather than only
    the wells with a test history.
    """
    from .wells import gas_lift_performance_curve

    rates = [float(r) for r in injection_rates]
    curve = gas_lift_performance_curve(
        ipr=ipr, tubing=tubing, wellhead_pressure_psia=wellhead_pressure_psia,
        formation_gor_scf_stb=formation_gor_scf_stb,
        injection_rates_scf_d=rates, water_cut=water_cut, oil=oil, gas=gas)

    points = [(point.injection_rate_scf_d, max(point.oil_rate_stb_d, 0.0))
              for point in curve]
    return GasLiftWell.from_curve(
        name, points, q_min=q_min,
        q_max=q_max if q_max is not None else max(rates),
        current=current)


def compare_curves(fitted: GasLiftWell, modelled: GasLiftWell,
                   points: int = 12) -> dict:
    """How far a test-fitted curve sits from the physics for the same well.

    The cross-check the two sources make possible. A fitted curve that departs
    badly from the nodal model is either a bad fit or a well that is not
    behaving like its own model, and both are worth knowing before the curve
    is used to move gas around a field.
    """
    low = max(fitted.tested_low, modelled.tested_low)
    high = min(fitted.tested_high, modelled.tested_high)
    if high <= low:
        raise ValueError("the two curves do not overlap in injection rate")

    grid = np.linspace(low, high, points)
    a = fitted.oil(grid)
    b = modelled.oil(grid)
    scale = max(float(np.max(np.abs(b))), 1e-9)
    difference = np.abs(a - b)
    return {
        "low": float(low),
        "high": float(high),
        "mean_absolute": float(np.mean(difference)),
        "worst_absolute": float(np.max(difference)),
        "worst_fraction": float(np.max(difference) / scale),
        "agree": bool(np.max(difference) / scale < 0.15),
    }


def show_the_maths(wells: list, allocation: Allocation) -> list:
    """The working, as lines an engineer can check or print.

    Required by the product principle that every number names its equation,
    its inputs and its source. A result that cannot be reproduced by hand is
    not a glass box however much of the method is described in prose.
    """
    lines = [
        "Objective: maximise total oil (or oil value less gas value) subject to",
        "  sum of injection <= %g, and q_min <= q_i <= q_max per well."
        % allocation.gas_available,
        "",
        "Step 1 — a performance curve per well:",
    ]
    for well in wells:
        lines.append("    %-8s [%s]  %s" % (well.name, well.source, well.equation()))

    lines += [
        "",
        "Step 2 — each curve replaced by its upper concave envelope, so marginal",
        "  gain is non-increasing and 'one more unit of gas' is well defined.",
        "",
        "Step 3 — gas given in increments to the highest marginal gain, ending at",
        "  the equal-slope condition. Shadow price lambda = %.6g oil per unit of gas,"
        % allocation.shadow_price,
        "  i.e. the next unit of compression is worth that much oil.",
        "",
        "Result: %g of %g gas used, oil %.4g -> %.4g (%+.4g)."
        % (allocation.gas_used, allocation.gas_available,
           allocation.previous_oil, allocation.total_oil, allocation.oil_gain),
    ]
    if not allocation.exactly_optimal:
        lines.append(
            "  Note: with minimum stable injection the greedy step is a heuristic,"
            " not a proof of optimality.")
    return lines
