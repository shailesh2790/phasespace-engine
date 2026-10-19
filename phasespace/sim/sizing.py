"""Route definition and pipe-size recommendation.

The question this answers: *I have laid out a route -- these lengths, these
bends, this terrain -- and I need this much gas delivered above this
pressure. What is the smallest pipe that does it?*

Smallest, not merely adequate, because pipe cost rises steeply with
diameter and the temptation in hand calculations is to round up twice: once
for the correlation's uncertainty and once for comfort. Sizing against a
solved hydraulic model with the fittings actually counted removes the first
excuse and makes the second visible as a cost number.

A candidate is rejected for any of three reasons, and the report says which:

  * **Delivery pressure.** It cannot make the contract pressure at the far
    end.
  * **Erosional velocity.** API RP 14E's Ve = C/sqrt(rho). A line can be
    hydraulically fine and still be rejected in review on this alone, which
    is why it is checked here rather than left as a downstream surprise.
  * **Absolute velocity.** A practical ceiling for noise and vibration,
    separate from erosion.

Cost is reported as a relative index, not currency. Installed pipeline cost
scales roughly as diameter^1.4 per unit length across published estimating
data, but the absolute number depends on terrain, labour, and steel price on
the day -- none of which this module can know. A ratio between candidates is
defensible; a rupee figure would not be.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .components import WEYMOUTH, GasPipe
from .fittings import FittingCount, total_equivalent_length_ft
from .fluids import GasProperties
from .solver import FLOW, GAS, PRESSURE, Network, Node, solve
from .units import FT_PER_MI

# Standard-weight (STD) steel pipe: nominal size -> actual inside diameter.
# Sizing has to be done on the real bore, not the nominal label: an "8 inch"
# line is 7.981 in inside, and diameter enters the Weymouth equation to the
# 2.667 power, so treating nominal as actual overstates capacity by ~1.3%
# at 8 in and considerably more in small bore.
STD_PIPE_ID_IN = {
    2: 2.067,
    3: 3.068,
    4: 4.026,
    6: 6.065,
    8: 7.981,
    10: 10.020,
    12: 12.000,
    14: 13.250,
    16: 15.250,
    18: 17.250,
    20: 19.250,
    24: 23.250,
    30: 29.250,
    36: 35.250,
}

DEFAULT_CANDIDATES = (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36)


@dataclass
class RouteSegment:
    """One straight run of pipe, with whatever fittings sit on it."""

    length_ft: float
    elevation_change_ft: float = 0.0
    fittings: list = field(default_factory=list)
    roughness_in: float = 0.0007
    name: str = ""

    def __post_init__(self):
        # NaN has to be rejected explicitly: every comparison against it is
        # False, so a bare `length <= 0` check lets a NaN straight through and
        # it then silently poisons the whole solve. This is the normal state
        # of a half-typed row in a spreadsheet-style editor, so it reaches
        # here in ordinary use rather than only under abuse.
        if not math.isfinite(self.length_ft):
            raise ValueError("segment length must be a finite number, got %r"
                             % (self.length_ft,))
        if self.length_ft <= 0.0:
            raise ValueError("segment length must be positive, got %r" % (self.length_ft,))
        if not math.isfinite(self.elevation_change_ft):
            raise ValueError("segment elevation change must be a finite number, got %r"
                             % (self.elevation_change_ft,))
        for f in self.fittings:
            if not isinstance(f, FittingCount):
                raise TypeError("fittings must be FittingCount instances, got %r" % (f,))

    def fitting_equivalent_length_ft(self, diameter_in: float) -> float:
        return total_equivalent_length_ft(self.fittings, diameter_in, self.roughness_in)

    def effective_length_ft(self, diameter_in: float) -> float:
        return self.length_ft + self.fitting_equivalent_length_ft(diameter_in)

    def fitting_count(self) -> int:
        return sum(f.count for f in self.fittings)


@dataclass
class PipeRoute:
    """An ordered chain of segments from inlet to delivery point."""

    segments: list = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        if not self.segments:
            raise ValueError("a route needs at least one segment")

    @property
    def total_length_ft(self) -> float:
        return sum(s.length_ft for s in self.segments)

    @property
    def total_rise_ft(self) -> float:
        return sum(s.elevation_change_ft for s in self.segments)

    @property
    def total_fittings(self) -> int:
        return sum(s.fitting_count() for s in self.segments)

    def elevation_profile_ft(self) -> list:
        """Cumulative elevation at each node, starting at zero."""
        profile = [0.0]
        for s in self.segments:
            profile.append(profile[-1] + s.elevation_change_ft)
        return profile

    def distance_profile_ft(self) -> list:
        profile = [0.0]
        for s in self.segments:
            profile.append(profile[-1] + s.length_ft)
        return profile

    def total_effective_length_ft(self, diameter_in: float) -> float:
        return sum(s.effective_length_ft(diameter_in) for s in self.segments)


@dataclass
class RouteSolution:
    """The solved hydraulics of one route at one diameter and flow."""

    diameter_in: float
    nominal_size: float
    flow_scfd: float
    node_pressures_psia: list
    segment_velocities_ft_s: list
    segment_erosional_limits_ft_s: list
    converged: bool
    fitting_equivalent_length_ft: float

    @property
    def choked(self) -> bool:
        """True when pressure collapsed to the solver's physical floor.

        This is not a convergence failure and must not be reported as one.
        It is the model saying the line is too small to pass the requested
        flow at all: there is no positive pressure profile that satisfies
        the mass balance, so the honest verdict is "undersized", not
        "the numerics gave up".
        """
        return min(self.node_pressures_psia) <= 1.0001

    @property
    def delivery_pressure_psia(self) -> float:
        return self.node_pressures_psia[-1]

    @property
    def total_drop_psi(self) -> float:
        return self.node_pressures_psia[0] - self.node_pressures_psia[-1]

    @property
    def max_velocity_ft_s(self) -> float:
        return max(self.segment_velocities_ft_s) if self.segment_velocities_ft_s else 0.0

    @property
    def worst_erosional_ratio(self) -> float:
        """Highest velocity as a fraction of its local erosional limit."""
        pairs = zip(self.segment_velocities_ft_s, self.segment_erosional_limits_ft_s)
        return max((v / limit for v, limit in pairs if limit > 0.0), default=0.0)


def build_route_network(
    route: PipeRoute,
    diameter_in: float,
    inlet_pressure_psia: float,
    flow_scfd: float,
    gas: GasProperties,
    equation: str = WEYMOUTH,
    efficiency: float = 0.92,
) -> tuple:
    """Turn a route into a solvable Network: one node per segment boundary.

    Fittings enter as added equivalent length on the segment that carries
    them, which is what lets minor losses flow through every correlation in
    `components.py` without any of them knowing fittings exist.
    """
    net = Network(phase=GAS, fluid=gas)
    net.add_node(Node(0, PRESSURE, pressure_psia=inlet_pressure_psia, name="inlet"))

    elevation = 0.0
    pipes = []
    for i, segment in enumerate(route.segments):
        node_id = i + 1
        is_last = (i == len(route.segments) - 1)
        # All the offtake sits at the far end; intermediate nodes are pure
        # junctions, so the whole flow passes through every segment.
        withdrawal = flow_scfd if is_last else 0.0
        net.add_node(Node(node_id, FLOW, withdrawal=withdrawal,
                          name=segment.name or ("node %d" % node_id)))
        length_mi = segment.effective_length_ft(diameter_in) / FT_PER_MI
        pipe = GasPipe(
            a=node_id - 1,
            b=node_id,
            diameter_in=diameter_in,
            length_mi=length_mi,
            roughness_in=segment.roughness_in,
            efficiency=efficiency,
            elevation_a_ft=elevation,
            elevation_b_ft=elevation + segment.elevation_change_ft,
            equation=equation,
            name=segment.name or ("segment %d" % (i + 1)),
        )
        net.add_branch(pipe)
        pipes.append(pipe)
        elevation += segment.elevation_change_ft

    return net, pipes


def solve_route(
    route: PipeRoute,
    diameter_in: float,
    inlet_pressure_psia: float,
    flow_scfd: float,
    gas: GasProperties,
    equation: str = WEYMOUTH,
    efficiency: float = 0.92,
    nominal_size: float = 0.0,
) -> RouteSolution:
    """Solve one route at one diameter, carrying `flow_scfd` to the far end."""
    net, pipes = build_route_network(route, diameter_in, inlet_pressure_psia,
                                     flow_scfd, gas, equation, efficiency)
    result = solve(net)

    pressures = [result.pressures[i] for i in range(len(route.segments) + 1)]
    velocities = []
    erosional = []
    for pipe in pipes:
        p_mid = 0.5 * (result.pressures[pipe.a] + result.pressures[pipe.b])
        velocities.append(pipe.velocity_ft_s(flow_scfd, p_mid, gas))
        erosional.append(pipe.erosional_velocity_ft_s(p_mid, gas))

    return RouteSolution(
        diameter_in=diameter_in,
        nominal_size=nominal_size or diameter_in,
        flow_scfd=flow_scfd,
        node_pressures_psia=pressures,
        segment_velocities_ft_s=velocities,
        segment_erosional_limits_ft_s=erosional,
        converged=result.converged,
        fitting_equivalent_length_ft=route.total_effective_length_ft(diameter_in)
        - route.total_length_ft,
    )


def flow_at_pressures(
    route: PipeRoute,
    diameter_in: float,
    inlet_pressure_psia: float,
    outlet_pressure_psia: float,
    gas: GasProperties,
    equation: str = WEYMOUTH,
    efficiency: float = 0.92,
) -> float:
    """The other direction: fix both end pressures, get the flow.

    Pin both ends as pressure nodes and the flow is simply what the solve
    produces -- no search required, because the network has no unknown
    pressures left. This is the "what will this line actually pass?" question
    rather than "what do I need?".
    """
    net = Network(phase=GAS, fluid=gas)
    net.add_node(Node(0, PRESSURE, pressure_psia=inlet_pressure_psia, name="inlet"))

    elevation = 0.0
    n = len(route.segments)
    for i, segment in enumerate(route.segments):
        node_id = i + 1
        is_last = (i == n - 1)
        if is_last:
            net.add_node(Node(node_id, PRESSURE, pressure_psia=outlet_pressure_psia,
                              name="delivery"))
        else:
            net.add_node(Node(node_id, FLOW, withdrawal=0.0))
        net.add_branch(GasPipe(
            a=node_id - 1, b=node_id, diameter_in=diameter_in,
            length_mi=segment.effective_length_ft(diameter_in) / FT_PER_MI,
            roughness_in=segment.roughness_in, efficiency=efficiency,
            elevation_a_ft=elevation, elevation_b_ft=elevation + segment.elevation_change_ft,
            equation=equation))
        elevation += segment.elevation_change_ft

    result = solve(net)
    first_pipe = net.branches[0]
    return result.flows[first_pipe]


@dataclass
class SizingCandidate:
    """One nominal pipe size evaluated against the design case."""

    nominal_size: float
    inside_diameter_in: float
    solution: RouteSolution
    feasible: bool
    failures: list
    relative_cost: float

    @property
    def delivery_pressure_psia(self) -> float:
        return self.solution.delivery_pressure_psia

    @property
    def max_velocity_ft_s(self) -> float:
        return self.solution.max_velocity_ft_s

    @property
    def erosional_ratio(self) -> float:
        return self.solution.worst_erosional_ratio


@dataclass
class SizingReport:
    candidates: list
    recommended: SizingCandidate = None
    notes: list = field(default_factory=list)

    @property
    def feasible_candidates(self) -> list:
        return [c for c in self.candidates if c.feasible]

    def cost_premium_of(self, candidate: SizingCandidate) -> float:
        """How much more this size costs than the recommendation, as a
        fraction. The number that makes 'round up one size for comfort'
        an explicit decision rather than a reflex."""
        if self.recommended is None or self.recommended.relative_cost <= 0.0:
            return 0.0
        return candidate.relative_cost / self.recommended.relative_cost - 1.0


def relative_cost_index(nominal_size: float, length_ft: float, exponent: float = 1.4) -> float:
    """Relative installed cost, proportional to diameter^exponent * length.

    Deliberately unitless. Published estimating data puts the diameter
    exponent for installed onshore pipeline cost between about 1.0 and 1.5 --
    steel weight alone would argue for 2.0, but trenching, right-of-way and
    welding scale far more weakly, which pulls the total down. 1.4 is a
    middle default and is exposed so it can be replaced with a fitted value
    once real tender prices are available.
    """
    return (nominal_size ** exponent) * length_ft


def recommend_size(
    route: PipeRoute,
    inlet_pressure_psia: float,
    target_flow_scfd: float,
    minimum_delivery_psia: float,
    gas: GasProperties = None,
    candidates=DEFAULT_CANDIDATES,
    equation: str = WEYMOUTH,
    efficiency: float = 0.92,
    max_velocity_ft_s: float = 80.0,
    erosional_fraction: float = 1.0,
    cost_exponent: float = 1.4,
) -> SizingReport:
    """Evaluate every candidate size and recommend the cheapest that works.

    Returns the whole ladder, not just the winner. The sizes immediately
    above and below the recommendation are the ones an engineer argues
    about, and the argument is better with the delivery pressure and cost
    premium of each in front of them.
    """
    gas = gas or GasProperties()
    evaluated = []
    notes = []

    for nominal in candidates:
        inside = STD_PIPE_ID_IN.get(nominal)
        if inside is None:
            raise ValueError(
                "no standard-weight ID known for nominal size %r (known: %s)"
                % (nominal, ", ".join(str(k) for k in sorted(STD_PIPE_ID_IN))))

        solution = solve_route(route, inside, inlet_pressure_psia, target_flow_scfd,
                               gas, equation=equation, efficiency=efficiency,
                               nominal_size=nominal)

        failures = []
        if solution.choked:
            failures.append(
                "too small: pressure collapses before the delivery point, so this "
                "size cannot pass %.1f MMscfd at any delivery pressure"
                % (target_flow_scfd / 1e6,))
        elif not solution.converged:
            failures.append("hydraulic solve did not converge")
        elif solution.delivery_pressure_psia < minimum_delivery_psia:
            failures.append(
                "delivery %.1f psia is below the %.1f psia minimum"
                % (solution.delivery_pressure_psia, minimum_delivery_psia))
        if not solution.choked and solution.max_velocity_ft_s > max_velocity_ft_s:
            failures.append(
                "velocity %.1f ft/s exceeds the %.1f ft/s limit"
                % (solution.max_velocity_ft_s, max_velocity_ft_s))
        ratio = solution.worst_erosional_ratio
        if not solution.choked and ratio > erosional_fraction:
            failures.append(
                "velocity reaches %.0f%% of the API RP 14E erosional limit"
                % (100.0 * ratio))

        evaluated.append(SizingCandidate(
            nominal_size=nominal,
            inside_diameter_in=inside,
            solution=solution,
            feasible=not failures,
            failures=failures,
            relative_cost=relative_cost_index(nominal, route.total_length_ft, cost_exponent),
        ))

    feasible = [c for c in evaluated if c.feasible]
    recommended = min(feasible, key=lambda c: c.relative_cost) if feasible else None

    if recommended is None:
        notes.append(
            "no candidate size meets the design case. The binding constraint is "
            "usually inlet pressure or route length -- consider raising inlet "
            "pressure, adding compression, or shortening the route.")
    else:
        margin = recommended.delivery_pressure_psia - minimum_delivery_psia
        notes.append(
            "%g in nominal delivers %.1f psia, %.1f psi above the %.1f psia minimum, "
            "at %.0f%% of its erosional limit."
            % (recommended.nominal_size, recommended.delivery_pressure_psia, margin,
               minimum_delivery_psia, 100.0 * recommended.erosional_ratio))
        if margin < 0.05 * minimum_delivery_psia:
            notes.append(
                "that margin is thin (under 5%%): a modest demand increase or a "
                "colder-than-design day would breach it, so the next size up is "
                "worth pricing.")

    return SizingReport(candidates=evaluated, recommended=recommended, notes=notes)
