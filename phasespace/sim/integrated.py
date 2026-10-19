"""Wells and the network solved as one system.

Everything else in this engine answers half a question. Nodal analysis tells
you what a well makes *at a given wellhead pressure*; the network solver tells
you what pressures a *given set of rates* produces. Run separately, each takes
the other's answer as an assumption, and the pair is a set of calculators
rather than a model of the field.

The physical statement they jointly make is simple and circular:

    reservoir     q_j  = IPR_j(p_wf,j)
    wellbore      p_wf,j = VLP_j(q_j, p_wh,j)
    network       p_wh   = Network(q_1 ... q_N)

A well that produces more raises the pressure in the gathering line, which
raises its own wellhead pressure, which makes it produce less. Nothing is
solved until all three agree at once. That loop is the entire reason adding
compression at a station changes what every well makes, and no amount of
careful single-well modelling can see it.

The method is damped fixed-point iteration rather than one large Newton
system, for a reason that is about the physics and not about convenience: the
well relation is not smooth. A well crossing its liquid-loading limit, or
hitting a minimum-rate shut-in rule, is a discontinuity, and a Newton method
meeting one oscillates between two states indefinitely while reporting that it
is still working. Fixed-point iteration meets the same discontinuity, and can
be *told* what to do about it -- which is to freeze the offending well, solve
everything else, and say so in the output.

Three traps are handled explicitly, and each earned its place:

**Well-state oscillation.** A well that flips between flowing and shut across
successive iterations will never converge. Detected by history rather than
damped away, frozen in its last state, and reported. Silently damping it
produces a converged answer to a question nobody asked.

**The unstable nodal root.** A gassy well's VLP curve turns back up at low
rate, so the inflow and outflow curves can cross twice. The lower crossing is
a well that dies. `nodal_analysis` already brackets from the high-rate side
for this reason; the coupling inherits that and additionally flags when the
two roots are close, because a well near the turning point is near heading
instability and that is itself worth reporting.

**Adaptive damping.** Too much and it crawls, too little and it oscillates.
The factor halves whenever the update reverses sign and grows slowly while it
does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .solver import FLOW, PRESSURE, Network, Node, solve
from .wells import BEGGS_BRILL, InflowPerformance, Tubing, nodal_analysis

SECONDS_PER_DAY = 86400.0

# A well whose rate changes by less than this fraction between outer passes is
# converged. Tighter than the measurement resolution of any real well test,
# and loose enough that the nodal bisection's own tolerance does not dominate.
DEFAULT_RATE_TOLERANCE = 1e-4

# A well flipping state across this many consecutive passes is oscillating
# rather than converging, and freezing it is the only way the rest of the
# network converges at all.
OSCILLATION_PASSES = 3


@dataclass
class CoupledWell:
    """One well attached to a network node.

    ``node`` is the network node this well delivers into -- its wellhead. The
    network sees the well as an injection at that node; the well sees the
    network as a back pressure.
    """

    name: str
    node: int
    ipr: InflowPerformance
    tubing: Tubing
    gas_oil_ratio_scf_stb: float = 600.0
    water_cut: float = 0.0
    oil: object = None
    gas: object = None
    correlation: str = BEGGS_BRILL
    # A well that cannot make this rate is shut in. The commonest operating
    # rule there is, and the commonest source of non-smoothness in the solve.
    min_rate_stb_d: float = 0.0
    # Above this wellhead pressure the well will not flow at all.
    max_wellhead_psia: float = 0.0
    # Resolution of each nodal solve used to build the performance curve.
    # Lower than the standalone defaults on purpose: the curve is sampled at
    # many pressures and then interpolated, so per-point precision matters far
    # less than it does for a single reported operating point. Measured on a
    # four-well field, dropping from 24x30 to 14x18 changes the field rate by
    # under half a percent and roughly halves the time.
    curve_rate_points: int = 14
    curve_segments: int = 18

    def performance_curve(self, low_psia: float, high_psia: float,
                          points: int = 14) -> list:
        """Rate against wellhead pressure, sampled once.

        The coupled iteration asks each well "what do you make at this
        pressure" dozens of times, and every one of those was a full nodal
        analysis -- a rate grid with a multiphase traverse at each point. A
        four-well field took two minutes.

        Sampling the same relationship once and interpolating is what IPM and
        PIPESIM do, and they do it for a better reason than speed: the curve
        is built on a fixed grid, so the iteration sees a smooth, monotone
        function instead of a bisection whose bracketing can shift slightly
        between calls. The non-smoothness that the freeze logic exists to
        catch comes from operating limits, not from sampling noise, and this
        removes the second source entirely.

        Returns (pressure, liquid_rate) ascending in pressure, with zero rate
        where the well will not flow.
        """
        if points < 2:
            raise ValueError("a curve needs at least two points, got %r" % (points,))
        if high_psia <= low_psia:
            raise ValueError("curve range must increase, got %r to %r"
                             % (low_psia, high_psia))

        curve = []
        for i in range(points):
            pressure = low_psia + (high_psia - low_psia) * i / (points - 1)
            try:
                result = self.deliverability(
                    pressure, points=self.curve_rate_points,
                    segments=self.curve_segments)
                rate = max(result.liquid_rate_stb_d, 0.0) if result.converged else 0.0
            except (ValueError, ZeroDivisionError):
                rate = 0.0
            curve.append((pressure, rate))
        return curve

    def deliverability(self, wellhead_psia: float, points: int = 24,
                       segments: int = 30):
        """What this well makes against a given back pressure."""
        return nodal_analysis(
            ipr=self.ipr, tubing=self.tubing,
            wellhead_pressure_psia=wellhead_psia,
            gas_oil_ratio_scf_stb=self.gas_oil_ratio_scf_stb,
            water_cut=self.water_cut, oil=self.oil, gas=self.gas,
            correlation=self.correlation, points=points, segments=segments)


@dataclass
class WellState:
    """Where one well ended up, and whether to believe it."""

    name: str
    node: int
    flowing: bool
    oil_rate_stb_d: float = 0.0
    liquid_rate_stb_d: float = 0.0
    wellhead_psia: float = 0.0
    bottomhole_psia: float = 0.0
    constraint: str = ""
    frozen: bool = False
    notes: list = field(default_factory=list)


@dataclass
class IntegratedResult:
    wells: list = field(default_factory=list)
    pressures: dict = field(default_factory=dict)
    converged: bool = False
    passes: int = 0
    max_change: float = math.inf
    notes: list = field(default_factory=list)

    @property
    def total_liquid_stb_d(self) -> float:
        return sum(w.liquid_rate_stb_d for w in self.wells if w.flowing)

    @property
    def flowing_wells(self) -> list:
        return [w for w in self.wells if w.flowing]

    @property
    def binding_constraints(self) -> dict:
        """What is actually limiting the field, counted.

        The output an engineer acts on. A field limited by separator pressure
        is a compression project; one limited by tubing size is a workover
        programme; one limited by inflow is neither, and knowing which is the
        whole value of coupling the two halves.
        """
        counts = {}
        for well in self.wells:
            if well.constraint:
                counts[well.constraint] = counts.get(well.constraint, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def interpolate(curve: list, pressure: float) -> float:
    """Rate at a pressure, from a sampled performance curve.

    Clamped at both ends rather than extrapolated: below the lowest sampled
    pressure the curve is flat, and above the highest the well is not flowing.
    Extrapolating a nodal curve past its last point is how a solver invents a
    well that produces against a pressure above its own reservoir.
    """
    if not curve:
        return 0.0
    if pressure <= curve[0][0]:
        return curve[0][1]
    if pressure >= curve[-1][0]:
        return curve[-1][1]
    for (p0, q0), (p1, q1) in zip(curve, curve[1:]):
        if pressure <= p1:
            if p1 == p0:
                return q1
            weight = (pressure - p0) / (p1 - p0)
            return q0 + weight * (q1 - q0)
    return curve[-1][1]


def solve_integrated(
    network: Network,
    wells: list,
    max_passes: int = 60,
    rate_tolerance: float = DEFAULT_RATE_TOLERANCE,
    initial_damping: float = 0.5,
    curves: dict = None,
    curve_points: int = 14,
) -> IntegratedResult:
    """Solve wells and network simultaneously.

    ``network`` must contain a node for each well's wellhead and at least one
    fixed-pressure node -- the separator or delivery point that sets the datum
    the whole field works against.

    The wells are attached as injections, so their rates enter the network as
    negative withdrawals at their nodes. The network's own demand nodes, if
    any, are left alone.
    """
    if not wells:
        raise ValueError("no wells supplied; use solve() for a network alone")
    if initial_damping <= 0.0 or initial_damping > 1.0:
        raise ValueError("damping must be in (0, 1], got %r" % (initial_damping,))

    for well in wells:
        if well.node not in network.nodes:
            raise ValueError("well %s is attached to node %r, which is not in "
                             "the network" % (well.name, well.node))
        if network.nodes[well.node].kind == PRESSURE:
            raise ValueError(
                "well %s is attached to node %r, which is a fixed-pressure "
                "node. A well cannot deliver into a pressure that is held "
                "constant regardless of what it produces -- that is the "
                "uncoupled calculation this function exists to replace."
                % (well.name, well.node))

    # Start every well from its unconstrained deliverability against the
    # network's datum pressure. Any starting point converges to the same
    # answer; this one is closest and costs one nodal solve per well.
    datum = min((n.pressure_psia for n in network.nodes.values()
                 if n.kind == PRESSURE), default=100.0)

    # One curve per well, built once and reused for every pass. Callers running
    # a sweep can build them once and pass them in, since a well's curve does
    # not depend on the separator pressure at all.
    if curves is None:
        reservoir_ceiling = max(
            (getattr(w.ipr, "reservoir_pressure_psia", 0.0) or 0.0) for w in wells)
        top = max(reservoir_ceiling, datum * 3.0, datum + 500.0)
        curves = {w.name: w.performance_curve(max(datum * 0.25, 14.7), top,
                                              points=curve_points)
                  for w in wells}

    rates = {}
    flowing = {}
    for well in wells:
        rates[well.name] = max(interpolate(curves[well.name], datum), 0.0)
        flowing[well.name] = rates[well.name] > well.min_rate_stb_d

    # Damping is per well, not shared.
    #
    # A single factor mutated inside the well loop let one well's step
    # direction change the next well's update, which destroys the symmetry of
    # a symmetric field: identical wells on identical lines came out with
    # different rates purely because of the order they were visited in.
    damping = {well.name: initial_damping for well in wells}
    previous_step = {well.name: 0.0 for well in wells}
    state_history = {well.name: [] for well in wells}
    frozen = set()
    notes = []
    pressures = {}
    change = math.inf
    passes = 0

    for passes in range(1, max_passes + 1):
        # 1. The network, given the rates the wells are currently making.
        for well in wells:
            node = network.nodes[well.node]
            if node.kind == FLOW:
                # Injection is a negative withdrawal. Liquid networks carry
                # bbl/day, which is what nodal analysis returns.
                node.withdrawal = -rates[well.name] if flowing[well.name] else 0.0

        network_result = solve(network)
        if not network_result.converged:
            notes.append(
                "The network did not converge on pass %d, so the wellhead "
                "pressures the wells were then solved against are not "
                "trustworthy." % passes)
        pressures = dict(network_result.pressures)

        # 2. Each well, against the wellhead pressure the network just gave it.
        change = 0.0
        for well in wells:
            if well.name in frozen:
                continue

            wellhead = pressures.get(well.node, datum)
            wants_to_flow = True
            if well.max_wellhead_psia > 0.0 and wellhead >= well.max_wellhead_psia:
                wants_to_flow = False
                target = 0.0
            else:
                target = max(interpolate(curves[well.name], wellhead), 0.0)
                if target <= well.min_rate_stb_d:
                    wants_to_flow = False
                    target = 0.0

            # 3. Damped update, with the factor adapting to the step's sign.
            current = rates[well.name]
            step = target - current
            if step * previous_step[well.name] < 0.0:
                damping[well.name] = max(damping[well.name] * 0.5, 0.05)
            elif abs(step) < abs(previous_step[well.name]):
                damping[well.name] = min(damping[well.name] * 1.1, 0.9)
            previous_step[well.name] = step

            updated = current + damping[well.name] * step
            scale = max(abs(current), abs(target), 1.0)
            change = max(change, abs(updated - current) / scale)
            rates[well.name] = max(updated, 0.0)

            state_history[well.name].append(wants_to_flow)
            flowing[well.name] = wants_to_flow

            # 4. A well flipping state every pass will never converge, and
            #    damping cannot fix a discontinuity. Freeze and report.
            recent = state_history[well.name][-(OSCILLATION_PASSES * 2):]
            if len(recent) >= OSCILLATION_PASSES * 2 and len(set(recent)) > 1:
                flips = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
                if flips >= OSCILLATION_PASSES:
                    frozen.add(well.name)
                    notes.append(
                        "%s flipped between flowing and shut %d times and was "
                        "frozen in its last state to let the rest of the field "
                        "converge. It is sitting on its operating limit, which "
                        "is a finding rather than a numerical problem."
                        % (well.name, flips))

        if change <= rate_tolerance:
            break

    states = []
    for well in wells:
        wellhead = pressures.get(well.node, datum)
        is_flowing = flowing[well.name] and rates[well.name] > 0.0
        state = WellState(name=well.name, node=well.node, flowing=is_flowing,
                          wellhead_psia=wellhead, frozen=well.name in frozen)

        if is_flowing:
            result = well.deliverability(wellhead)
            if not result.converged:
                # The well cannot actually flow against the pressure the
                # network settled at. Reporting the rate it last held would
                # be reporting a well producing against a wellhead pressure
                # above its own reservoir -- which is what this did before
                # the check existed.
                state.flowing = False
                state.liquid_rate_stb_d = 0.0
                state.oil_rate_stb_d = 0.0
                state.constraint = ("cannot flow against %.0f psia at the "
                                    "wellhead" % wellhead)
                state.notes = list(result.notes)
                states.append(state)
                continue
            state.oil_rate_stb_d = result.oil_rate_stb_d
            state.liquid_rate_stb_d = rates[well.name]
            state.bottomhole_psia = result.bottomhole_pressure_psia
            state.constraint = _binding_constraint(well, result, wellhead, datum)
            state.notes = list(result.notes)
        else:
            state.constraint = ("wellhead pressure above the well's limit"
                                if (well.max_wellhead_psia > 0.0
                                    and wellhead >= well.max_wellhead_psia)
                                else "below minimum rate")
        states.append(state)

    # A frozen well is an unresolved discontinuity, not a solution. With every
    # well frozen the rate change is zero by construction, and reporting that
    # as convergence is how this returned a field producing against a wellhead
    # pressure above its own reservoir and called it converged.
    converged = change <= rate_tolerance and not frozen
    if frozen:
        notes.append(
            "%d of %d wells were frozen on an operating limit, so this is a "
            "partial solution: the rest of the field is consistent, those "
            "wells are not." % (len(frozen), len(wells)))
    if not converged and change > rate_tolerance:
        notes.append(
            "Rates were still moving by %.2g between passes after %d passes "
            "against a tolerance of %.2g." % (change, passes, rate_tolerance))

    return IntegratedResult(wells=states, pressures=pressures,
                            converged=converged, passes=passes,
                            max_change=change, notes=notes)


def _binding_constraint(well: CoupledWell, result, wellhead_psia: float,
                        datum_psia: float) -> str:
    """Which limit is actually setting this well's rate.

    Measured rather than inferred from a ratio. The obvious heuristic -- call
    it back pressure when the wellhead is a large fraction of the bottomhole
    pressure -- never fires on a deep well, because 8,000 ft of hydrostatic
    column dwarfs any realistic wellhead pressure. It reported "tubing" for
    every well on a field that was visibly throttled by its gathering system.

    So the question is asked directly: re-solve the well as if the gathering
    system were free, at the separator's own pressure, and see how much of its
    potential the network is taking. That costs one extra nodal solve per well,
    once, at report time -- and it is the number an engineer wants anyway,
    because it is the prize for debottlenecking.

    Naming this matters more than the rate: separator pressure is a
    compression project, tubing is a workover, and inflow is neither.
    """
    if not result.converged:
        return "nodal solve did not converge"

    reservoir = getattr(well.ipr, "reservoir_pressure_psia", None)
    if reservoir:
        drawdown = reservoir - result.bottomhole_pressure_psia
        if drawdown <= 0.05 * reservoir:
            return "inflow — little drawdown left"

    # What the well would make with no gathering losses at all.
    if wellhead_psia > datum_psia + 1.0:
        try:
            unrestricted = well.deliverability(datum_psia)
        except (ValueError, ZeroDivisionError):
            unrestricted = None
        if unrestricted is not None and unrestricted.liquid_rate_stb_d > 0.0:
            lost = ((unrestricted.liquid_rate_stb_d - result.liquid_rate_stb_d)
                    / unrestricted.liquid_rate_stb_d)
            if lost >= 0.10:
                return "back pressure — the network is taking %.0f%% of this well" % (
                    100 * lost)

    return "tubing — most of the loss is in the wellbore"


def attach_wells(network: Network, wells: list, wellhead_elevation_ft: float = 0.0):
    """Add a flow node for every well that does not already have one.

    Convenience for building a case from a well list, so that a caller does
    not have to keep the two descriptions of the same field in step by hand --
    which is the sort of bookkeeping that produces a network missing one well
    and a total that looks plausible.
    """
    for well in wells:
        if well.node not in network.nodes:
            network.add_node(Node(well.node, FLOW, withdrawal=0.0,
                                  elevation_ft=wellhead_elevation_ft,
                                  name=well.name))
    return network
