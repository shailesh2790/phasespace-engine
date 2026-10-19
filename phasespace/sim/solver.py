"""Steady-state network model and the deterministic Newton-Raphson solver.

This is the audit anchor for everything else in PhaseSpace (PRD Section 6):
the surrogate is trained against it, and the UI shows its answer alongside
every prediction. So it is the piece that has to be right.

What changed from `phasespace.physics.network`
----------------------------------------------
That solver had exactly one unknown type (pressure at a demand node) and
exactly one equation (Weymouth mass balance). It could not express a
compressor, a regulator, or a pump, because those devices do not have a
flow/pressure relation -- they impose a condition and pass whatever flow
the network needs.

So the unknown vector here is two-part:

    x = [ p_j  for each node with unknown pressure ]
        [ q_k  for each control device             ]

and the residual vector matches it:

    R_j = (flow into node j) - (withdrawal at node j)      mass balance
    R_k = device residual (e.g. p_discharge - setpoint)    device equation

Passive branches (pipes, valves) contribute flow and its pressure
derivatives to the mass-balance rows. Control devices contribute their own
flow *variable* to the two nodes they touch, plus one row of their own.
The square system that results is solved the same way as before.

Devices that switch state -- a regulator that runs out of upstream pressure,
a compressor asked for more ratio than it has -- cannot be written as a
single smooth residual. Those are handled by an outer loop that re-solves
after each state change, which is the standard treatment and converges in a
handful of passes because the state set is finite and small.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Sparse linear algebra if scipy is available, dense otherwise. A pipe-network
# Jacobian has only about degree+1 non-zeros per row, so a dense factorisation
# is O(n^3) work on a matrix that is ~99% zeros -- at 120 nodes that is already
# half the solve time, and it is the term that makes a real distribution
# network (hundreds of nodes) intractable.
#
# Optional rather than required: the deployed web bundle vendors this module
# with numpy alone, because scipy would blow Vercel's function size limit. Small
# networks do not need sparse anyway, so the dense path is a genuine fallback
# rather than a degraded one.
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    _HAVE_SPARSE = True
except ImportError:                                   # pragma: no cover
    _HAVE_SPARSE = False

# Below this, assembling a sparse matrix costs more than the dense solve saves.
SPARSE_THRESHOLD = 60

from .components import (
    PRESSURE_RATIO,
    Branch,
    Compressor,
    ControlBranch,
    Regulator,
)
from .fluids import GasProperties, LiquidProperties

PRESSURE = "pressure"  # fixed-pressure node (a source, a slack, a delivery held at pressure)
FLOW = "flow"          # known net withdrawal, unknown pressure

GAS = "gas"
LIQUID = "liquid"


@dataclass
class Node:
    id: int
    kind: str = FLOW
    pressure_psia: float = 0.0   # fixed value when kind == PRESSURE, else the solved result
    withdrawal: float = 0.0      # positive = taken out of the network; negative = injected
    elevation_ft: float = 0.0
    name: str = ""

    def __post_init__(self):
        if self.kind not in (PRESSURE, FLOW):
            raise ValueError("node kind must be %r or %r, got %r" % (PRESSURE, FLOW, self.kind))
        if self.kind == PRESSURE and self.pressure_psia <= 0.0:
            # A fixed-pressure node at zero is not a boundary condition, it is
            # a vacuum, and the solver will happily converge the whole network
            # onto its pressure floor and report success. Caught here rather
            # than in the UI because it is a physical impossibility, not an
            # input-validation preference.
            raise ValueError(
                "node %s is a fixed-pressure node with a pressure of %g psia. "
                "A source holds the network up; it needs a positive absolute "
                "pressure." % (self.name or self.id, self.pressure_psia))


@dataclass
class Network:
    """A fluid network: nodes, passive branches, and control devices.

    `fluid` is a GasProperties or LiquidProperties; `phase` says which
    equation family the branches belong to and therefore what unit the
    flows carry (scf/day for gas, bbl/day for liquid).
    """

    phase: str = GAS
    fluid: object = None
    nodes: dict = field(default_factory=dict)
    branches: list = field(default_factory=list)
    controls: list = field(default_factory=list)

    def __post_init__(self):
        if self.phase not in (GAS, LIQUID):
            raise ValueError("phase must be %r or %r, got %r" % (GAS, LIQUID, self.phase))
        if self.fluid is None:
            self.fluid = GasProperties() if self.phase == GAS else LiquidProperties()

    # -- construction -------------------------------------------------------

    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError("duplicate node id %r" % (node.id,))
        self.nodes[node.id] = node
        return node

    def add_branch(self, branch: Branch) -> Branch:
        self.branches.append(branch)
        return branch

    def add_control(self, control: ControlBranch) -> ControlBranch:
        self.controls.append(control)
        return control

    def add(self, component):
        """Add a branch or a control device, whichever it is."""
        if isinstance(component, ControlBranch):
            return self.add_control(component)
        return self.add_branch(component)

    # -- queries ------------------------------------------------------------

    def pressure_node_ids(self) -> list:
        return [n.id for n in self.nodes.values() if n.kind == PRESSURE]

    def unknown_node_ids(self) -> list:
        return [n.id for n in self.nodes.values() if n.kind == FLOW]

    def validate(self) -> list:
        """Return a list of structural problems, empty if the model is sound.

        Checked before solving rather than after: a network with no fixed
        pressure has no unique solution at all, and Newton will wander for
        `max_iter` iterations and then report a plausible-looking
        non-converged result instead of saying what is actually wrong.
        """
        problems = []
        if not self.nodes:
            problems.append("network has no nodes")
        if not self.pressure_node_ids():
            problems.append(
                "network has no fixed-pressure node -- pressures are only "
                "determined up to a constant without one"
            )

        known_ids = set(self.nodes)
        for component in list(self.branches) + list(self.controls):
            label = getattr(component, "name", "") or type(component).__name__
            for end in (component.a, component.b):
                if end not in known_ids:
                    problems.append("%s references unknown node id %r" % (label, end))
            if component.a == component.b:
                problems.append("%s connects node %r to itself" % (label, component.a))

        # Connectivity: every node must have a path to a fixed-pressure node,
        # or its own mass balance is unsolvable in isolation.
        adjacency = {nid: set() for nid in self.nodes}
        for component in list(self.branches) + list(self.controls):
            if component.a in adjacency and component.b in adjacency:
                adjacency[component.a].add(component.b)
                adjacency[component.b].add(component.a)
        seen = set()
        stack = list(self.pressure_node_ids())
        seen.update(stack)
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        orphans = sorted(set(self.nodes) - seen)
        if orphans:
            problems.append(
                "node(s) %s are not connected to any fixed-pressure node"
                % (", ".join(str(o) for o in orphans),)
            )
        return problems


@dataclass
class SolveResult:
    pressures: dict          # node id -> psia
    flows: dict              # branch object -> flow (scf/day or bbl/day, a -> b)
    converged: bool
    iterations: int
    max_residual: float
    outer_passes: int = 1
    notes: list = field(default_factory=list)

    def flow_by_nodes(self) -> dict:
        """Flows keyed by (a, b) -- the shape `physics.network.SolveResult`
        used, kept so existing report/plot code can consume either."""
        return {(br.a, br.b): q for br, q in self.flows.items()}

    def node_imbalance(self, net: Network) -> dict:
        """Net mass-balance error at every node, in the network's flow unit.

        The convergence flag says the solver stopped; this says how well the
        answer actually conserves mass, which is the number to put in front
        of an engineer being asked to trust it.
        """
        imbalance = {nid: -node.withdrawal for nid, node in net.nodes.items()}
        for br, q in self.flows.items():
            imbalance[br.a] = imbalance.get(br.a, 0.0) - q
            imbalance[br.b] = imbalance.get(br.b, 0.0) + q
        for nid, node in net.nodes.items():
            if node.kind == PRESSURE:
                # A fixed-pressure node supplies whatever the network draws,
                # so its "imbalance" is just that supply rate, not an error.
                imbalance[nid] = 0.0
        return imbalance


def _flow_scale(net: Network) -> float:
    """A representative flow magnitude, used to non-dimensionalise the system.

    Gas flows are in the millions of scf/day while pressures are in the
    hundreds of psia. Handing Newton a Jacobian spanning that much dynamic
    range is what made the Phase 0 solver need an SVD to survive; scaling
    the variables first is the actual fix, and it makes the ordinary
    least-squares solve well-conditioned instead of merely survivable.
    """
    total = sum(abs(n.withdrawal) for n in net.nodes.values())
    if total > 0.0:
        return total
    return 1e6 if net.phase == GAS else 1e3


def solve(
    net: Network,
    rel_tol: float = 1e-8,
    max_iter: int = 200,
    max_outer: int = 12,
    init_pressure: float = 0.0,
    verbose: bool = False,
    pressure_squared: bool = None,
    initial_pressures: dict = None,
) -> SolveResult:
    """Solve the network for steady-state pressures and flows.

    `rel_tol` is relative to total system throughput, not absolute: flow
    rates span orders of magnitude with pipe size and demand, and a fixed
    absolute tolerance that looks fine on one network silently fails to
    resolve on another an order of magnitude larger.

    `initial_pressures` is a per-node warm start: node id to psia, for any
    subset of the unknown nodes. Newton converges to the same answer from
    any starting point it converges from at all, so this changes the
    iteration count and nothing else -- which is the whole reason a
    learned or approximate guess is safe here. Nodes left out of the dict
    fall back to the scalar `init_pressure` as before.
    """
    problems = net.validate()
    if problems:
        raise ValueError("invalid network:\n  - " + "\n  - ".join(problems))

    notes = []
    result = None
    outer = 0

    for outer in range(1, max_outer + 1):
        result = _solve_once(net, rel_tol, max_iter, init_pressure, verbose,
                             pressure_squared,
                             initial_pressures if outer == 1 else None)
        changed = _update_device_states(net, result, notes)
        if not changed:
            break
        # Warm-start the next pass from where this one landed: a device state
        # change usually perturbs the solution locally, so the previous
        # answer is a far better starting point than a flat guess.
        init_pressure = 0.0

    result.outer_passes = outer
    result.notes = notes
    return result


def _solve_once(net, rel_tol, max_iter, init_pressure, verbose,
                pressure_squared=None, initial_pressures=None) -> SolveResult:
    # Gas flow is driven by the difference of *squared* pressures, so solving
    # for u = P^2 rather than P is the natural variable: the residual is far
    # closer to linear in u, and Newton spends less of its budget undoing the
    # curvature that the chain rule through P introduces. Every branch keeps
    # its pressure-space derivative -- since u = P^2, du = 2P dP, so
    # dq/du = (dq/dP) / (2P), which needs no change to any component.
    #
    # Only for gas. Liquid hydraulics is driven by a plain pressure difference,
    # where squaring would introduce curvature rather than remove it.
    # Off by default, and kept only so the experiment is reproducible.
    # Solving for u = P^2 looked promising -- gas flow is driven by squared
    # pressure differences -- but it changes nothing, because Newton's method
    # is invariant under a smooth reparameterisation of its variables. Measured
    # on 120-node networks the two agree to the iteration: 59.9 either way.
    if pressure_squared is None:
        pressure_squared = False
    if net.phase != GAS:
        pressure_squared = False

    unknown_ids = net.unknown_node_ids()
    n_p = len(unknown_ids)
    p_index = {nid: i for i, nid in enumerate(unknown_ids)}

    active_controls = [c for c in net.controls if _is_active(c)]
    n_q = len(active_controls)
    q_index = {id(c): n_p + k for k, c in enumerate(active_controls)}

    # Regulators that gave up control act as plain wide-open restrictions.
    passive = list(net.branches) + [
        c.passthrough() for c in net.controls if isinstance(c, Regulator) and not c.controlling
    ]

    n = n_p + n_q
    fixed = {nid: net.nodes[nid].pressure_psia for nid in net.pressure_node_ids()}

    if init_pressure <= 0.0:
        init_pressure = min(fixed.values()) * 0.9 if fixed else 500.0

    x = np.empty(n, dtype=float)
    x[:n_p] = init_pressure ** 2 if pressure_squared else init_pressure
    if initial_pressures:
        # A guess is only ever a starting point, so it is clamped to
        # something physical rather than trusted: a predictor that returns
        # a negative pressure would otherwise put Newton's first step
        # somewhere the correlations are undefined.
        for nid, guess in initial_pressures.items():
            if nid not in p_index:
                continue
            value = float(guess)
            if not np.isfinite(value) or value <= 0.0:
                continue
            x[p_index[nid]] = value ** 2 if pressure_squared else value
    q_scale = _flow_scale(net)
    x[n_p:] = q_scale * 0.1

    p_scale = max(max(fixed.values()) if fixed else 1.0, 1.0)
    # The pressure unknowns are u = P^2 when enabled, so their scale has to be
    # squared too or the column scaling silently stops doing its job.
    unknown_scale = p_scale ** 2 if pressure_squared else p_scale
    col_scale = (np.concatenate([np.full(n_p, unknown_scale), np.full(n_q, q_scale)])
                 if n else np.zeros(0))
    # Pressures stay physical, expressed in whichever variable is being solved.
    floor = 1.0 if pressure_squared else 1.0

    tol = rel_tol * q_scale

    def pressure_of(nid, x_vec):
        """Physical pressure in psia, whatever variable the solve is using."""
        if nid in p_index:
            value = x_vec[p_index[nid]]
            if pressure_squared:
                return float(np.sqrt(max(value, floor)))
            return value
        return fixed[nid]

    def assemble(x_vec):
        r = np.zeros(n)
        jac = np.zeros((n, n))

        for br in passive:
            p_a = pressure_of(br.a, x_vec)
            p_b = pressure_of(br.b, x_vec)
            q, dq_da, dq_db = br.flow_and_grad(p_a, p_b, net.fluid)
            if pressure_squared:
                # du = 2P dP, so dq/du = (dq/dP) / (2P). The branch keeps its
                # own derivative; only the chain rule changes.
                dq_da = dq_da / (2.0 * max(p_a, 1e-9))
                dq_db = dq_db / (2.0 * max(p_b, 1e-9))
            ia = p_index.get(br.a)
            ib = p_index.get(br.b)
            if ia is not None:
                r[ia] -= q
                jac[ia, ia] -= dq_da
                if ib is not None:
                    jac[ia, ib] -= dq_db
            if ib is not None:
                r[ib] += q
                jac[ib, ib] += dq_db
                if ia is not None:
                    jac[ib, ia] += dq_da

        for c in active_controls:
            k = q_index[id(c)]
            q = x_vec[k]
            p_a = pressure_of(c.a, x_vec)
            p_b = pressure_of(c.b, x_vec)
            ia = p_index.get(c.a)
            ib = p_index.get(c.b)
            if ia is not None:
                r[ia] -= q
                jac[ia, k] -= 1.0
            if ib is not None:
                r[ib] += q
                jac[ib, k] += 1.0

            res, dr_da, dr_db, dr_dq = c.residual_and_grad(p_a, p_b, q, net.fluid)
            if pressure_squared:
                # A device residual is stated in pressure (e.g. p_out - setpoint),
                # so its derivative needs the same chain rule as a branch flow.
                dr_da = dr_da / (2.0 * max(p_a, 1e-9))
                dr_db = dr_db / (2.0 * max(p_b, 1e-9))
            r[k] = res
            if ia is not None:
                jac[k, ia] = dr_da
            if ib is not None:
                jac[k, ib] = dr_db
            jac[k, k] = dr_dq

        for nid, i in p_index.items():
            r[i] -= net.nodes[nid].withdrawal

        return r, jac

    def clamp(x_vec):
        if n_p:
            x_vec = x_vec.copy()
            # Stay physical: P > 0, and equivalently u = P^2 > 0.
            x_vec[:n_p] = np.maximum(x_vec[:n_p], floor)
        return x_vec

    def merit(x_vec):
        """Scalar measure of how wrong a state is, used by the line search.

        Mass rows are flows and device rows are pressures, so they are each
        divided by their own scale before being combined -- otherwise a
        network carrying millions of scf/day would drown out a compressor
        missing its setpoint by 50 psi, and the line search would happily
        accept steps that wreck the device equations.
        """
        r_vec = assemble(x_vec)[0]
        total = 0.0
        if n_p:
            total += float(np.sum((r_vec[:n_p] / q_scale) ** 2))
        if n_q:
            total += float(np.sum((r_vec[n_p:] / p_scale) ** 2))
        return total

    converged = False
    iterations = 0
    max_res = 0.0

    for iterations in range(1, max_iter + 1):
        r, jac = assemble(x)
        # Mass-balance rows are compared against the flow tolerance; device
        # rows are pressures (psia) and get a pressure-scaled tolerance, so
        # neither kind can hide a real error behind the other's units.
        mass_res = float(np.max(np.abs(r[:n_p]))) if n_p else 0.0
        dev_res = float(np.max(np.abs(r[n_p:]))) if n_q else 0.0
        max_res = mass_res
        if mass_res < tol and dev_res < max(rel_tol * p_scale, 1e-7):
            converged = True
            break
        if n == 0:
            converged = True
            break

        # Column scaling turns the raw system (pressures in the hundreds
        # next to flows in the millions) into a well-conditioned one before
        # the linear solve; row equilibration does the same for the
        # residuals, which mix flow units and pressure units by construction.
        jac_scaled = jac * col_scale
        row_norm = np.maximum(np.max(np.abs(jac_scaled), axis=1), 1e-30)
        jac_scaled = jac_scaled / row_norm[:, None]
        rhs = -r / row_norm

        delta_scaled = _linear_solve(jac_scaled, rhs)

        delta = delta_scaled * col_scale

        # Cap the pressure step so a large early correction cannot throw a
        # node to a non-physical pressure and strand the iteration somewhere
        # the correlations are not even defined.
        damping = 1.0
        if n_p:
            step = float(np.max(np.abs(delta[:n_p])))
            cap = 0.5 * unknown_scale
            if step > cap:
                damping = cap / step

        # Backtracking line search. A full Newton step is not safe here: in a
        # large-bore, low-drop line dq/dp is enormous and q ~ sqrt(|dP^2|)
        # has infinite slope at zero differential, so the undamped step
        # overshoots and the iteration settles into a limit cycle -- bouncing
        # between two wrong states forever while reporting a huge residual.
        # Requiring each step to actually reduce the merit function turns
        # that cycle into convergence, and costs only a residual evaluation
        # per backtrack.
        current = merit(x)
        alpha = damping
        x_next = clamp(x + alpha * delta)
        for _ in range(40):
            if merit(x_next) < current * (1.0 - 1e-4 * alpha):
                break
            alpha *= 0.5
            x_next = clamp(x + alpha * delta)
        else:
            # No downhill step exists along this direction: we are either at
            # a solution already or on a kink the Jacobian cannot see. Take
            # the smallest step and let the convergence test decide.
            pass
        x = x_next

        if verbose:
            print("  iter %3d  mass_res=%.4g  dev_res=%.4g  alpha=%.4g"
                  % (iterations, mass_res, dev_res, alpha))

    pressures = {nid: fixed[nid] for nid in fixed}
    for nid, i in p_index.items():
        pressures[nid] = (float(np.sqrt(max(x[i], floor))) if pressure_squared
                          else float(x[i]))

    flows = {}
    for br in net.branches:
        flows[br] = br.flow_and_grad(pressures[br.a], pressures[br.b], net.fluid)[0]
    for c in net.controls:
        if _is_active(c):
            flows[c] = float(x[q_index[id(c)]])
        else:
            through = c.passthrough()
            flows[c] = through.flow_and_grad(pressures[c.a], pressures[c.b], net.fluid)[0]

    return SolveResult(
        pressures=pressures,
        flows=flows,
        converged=converged,
        iterations=iterations,
        max_residual=max_res,
    )


def _linear_solve(jacobian: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """One Newton step, sparse where that is cheaper.

    A singular Jacobian is a real modelling condition -- two controls fighting
    over the same pressure, an isolated loop -- rather than necessarily a fatal
    one, so both paths fall back to least squares, which takes the minimum-norm
    step and lets the iteration continue instead of aborting the solve.
    """
    n = jacobian.shape[0]
    if _HAVE_SPARSE and n >= SPARSE_THRESHOLD:
        try:
            with np.errstate(all="ignore"):
                solution = spsolve(csr_matrix(jacobian), rhs)
            # spsolve signals a singular matrix by returning NaNs rather than
            # raising, so the result has to be checked rather than trusted.
            if np.all(np.isfinite(solution)):
                return solution
        except Exception:                              # noqa: BLE001
            pass
    try:
        return np.linalg.solve(jacobian, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(jacobian, rhs, rcond=None)[0]


def _is_active(control) -> bool:
    if isinstance(control, Regulator):
        return control.controlling
    return True


def _update_device_states(net: Network, result: SolveResult, notes: list) -> bool:
    """Switch devices between their operating states; True if anything moved.

    Two switches matter physically:

    * A regulator can only *reduce* pressure. Once upstream pressure has
      fallen to its setpoint the valve is wide open and the setpoint is
      simply not being held any more -- reporting the setpoint regardless
      would be the model lying about a low-pressure event, which is exactly
      the condition an engineer runs this simulation to find.
    * A compressor has a finite maximum ratio. Asked for more, the honest
      answer is that it delivers its maximum ratio and the downstream
      pressure falls short, not that it magically meets the setpoint.
    """
    changed = False

    for c in net.controls:
        if isinstance(c, Regulator):
            p_up = result.pressures.get(c.a, 0.0)
            if c.controlling and p_up <= c.setpoint * 1.0001:
                c.controlling = False
                changed = True
                notes.append(
                    "regulator %s lost control: upstream %.1f psia is at or below "
                    "its %.1f psia setpoint, so it is wide open"
                    % (c.name or "(unnamed)", p_up, c.setpoint)
                )
            elif not c.controlling and p_up > c.setpoint * 1.02:
                c.controlling = True
                changed = True
                notes.append(
                    "regulator %s regained control (upstream back to %.1f psia)"
                    % (c.name or "(unnamed)", p_up)
                )
        elif isinstance(c, Compressor):
            p_a = result.pressures.get(c.a, 0.0)
            p_b = result.pressures.get(c.b, 0.0)
            if c.mode != PRESSURE_RATIO and p_a > 0.0 and p_b / p_a > c.max_ratio * 1.0001:
                notes.append(
                    "compressor %s hit its %.2f max ratio (setpoint needed %.2f); "
                    "held at maximum ratio instead"
                    % (c.name or "(unnamed)", c.max_ratio, p_b / p_a)
                )
                c.mode = PRESSURE_RATIO
                c.setpoint = c.max_ratio
                changed = True

    return changed
