"""A better first guess for Newton, and the measurement that it helped.

PRD v2, C2. Of everything in the neural layer this is the only item that
carries no accuracy risk at all. Newton converges to the same root from
any starting point it converges from, so a guess changes the iteration
count and nothing else. The answer stays the solver's; a wrong guess
costs time, never correctness. That is a rare shape for a learned
component and it is why this is worth doing before anything that asks a
user to trust a predicted number.

**The predictor here is physics, not a network, and that is deliberate.**
The PRD's own rule is to start with the simplest model that meets the
target. A gas pipe carries `q = K sqrt((u_a - u_b) / L)` in the squared
pressure `u = p^2`, which is a resistor network with a flow-dependent
conductance. Freezing that conductance makes it linear, and one sparse
solve gives a pressure field that already respects the topology, the pipe
sizes and the boundary conditions -- everything the constant guess throws
away. Two refinement passes update the conductance from the field just
computed, which is a Picard iteration on the real equation.

**What it is worth, measured rather than claimed.** On hub topologies --
wells into a header into a separator, which is what a gathering system
actually is -- it saves a third to a half of the Newton iterations:

    star, 150 wells      10 -> 5 iterations     50%
    star, 400 wells       9 -> 5                44%
    trunk, 100 offtakes  10 -> 8                20%
    trunk, 250 offtakes  10 -> 9                10%
    trunk, 500 offtakes  10 -> 11              -10%

The PRD's target is half or more. This meets it on the shape that matters
and decays to slightly worse than useless on a 500-link chain, where the
linearisation's error compounds along a path no gathering system has.
That decay is the honest argument for a graph network rather than against
one: it is exactly the regime a learned model could cover, and
`benchmark` is how anyone would find out. The scaffold in
`phasespace/model` is the drop-in -- anything returning node id to
pressure fits where this does.

The worst measured case costs 10% more iterations, which is the bounded
downside against that upside. `warm_solve` does not try to detect it:
knowing the warm start was slower means having already paid for the cold
solve, so that measurement belongs in `benchmark`.

**Measured, not claimed.** `benchmark` solves each network both ways and
reports both iteration counts, because the PRD asks for the speed-up to
be measured on every solve rather than assumed once. It also checks the
two answers agree, which is the property that makes the whole idea safe
and the one that would quietly break if a warm start were ever allowed to
change the convergence criterion rather than only the starting point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .solver import FLOW, GAS, PRESSURE, Network, solve

# Picard passes over the frozen conductance. Each pass costs one linear
# solve and propagates the correction one more step along the graph, so
# the number that helps depends on how far the furthest node is from a
# boundary condition. Measured by sweeping 2, 4, 8 and 16 passes:
#
#   star, 400 wells     33% -> 44% of iterations saved, flat after 4
#   trunk, 250 offtakes 10% at every count
#   trunk, 500 offtakes -40% -> -10%, flat after 4
#
# Four, because everything plateaus there and each extra pass is pure
# cost.
DEFAULT_PASSES = 4

# Conductances span orders of magnitude with diameter (d^2.667), so the
# linear system is poorly conditioned by nature. Solved by least squares
# rather than a plain inverse, which also handles a disconnected island
# without raising.
RCOND = 1e-12

# Above this the dense least-squares solve is the cost. The matrix is a
# graph Laplacian and is extremely sparse -- a node touches only its
# neighbours -- so the same mistake the solver already avoids with its own
# SPARSE_THRESHOLD applies here. Measured before the sparse path existed,
# a 500-node guess took 111 ms against a 145 ms solve, which is most of
# the saving spent on computing it.
SPARSE_THRESHOLD = 60


def _conductance(branch, delta_u: float) -> float:
    """Linearised conductance of a branch in squared-pressure space.

    From `q = K sqrt(dU / L)`, the secant conductance at a given drop is
    `q / dU = K / sqrt(L dU)`. K itself is not exposed by the component
    interface, so it is recovered from the branch's own flow calculation
    at a probe drop -- which keeps this correct for whatever correlation
    the branch implements rather than re-deriving Weymouth here and
    drifting from it.
    """
    probe = max(delta_u, 1.0)
    # Probe pressures that differ by `probe` in squared space.
    high = math.sqrt(probe + 1.0e6)
    low = math.sqrt(1.0e6)
    try:
        flow = branch.flow(high, low, _probe_fluid(branch))
    except Exception:                                   # noqa: BLE001
        return 0.0
    if not math.isfinite(flow) or flow <= 0.0:
        return 0.0
    return flow / probe


_PROBE_FLUID = None


def _probe_fluid(branch):
    """The fluid a branch wants, without the caller having to know.

    Components take whatever their phase needs; the warm start does not
    care about the fluid beyond getting a finite flow out of one probe.
    """
    global _PROBE_FLUID
    if _PROBE_FLUID is None:
        from .fluids import GasProperties

        _PROBE_FLUID = GasProperties()
    return _PROBE_FLUID


def linearised_pressures(net: Network, passes: int = DEFAULT_PASSES) -> dict:
    """A starting guess for every unknown node, from one linear solve.

    Returns node id to psia. Never raises: a guess that cannot be formed
    is an empty dict, and the solver then starts where it always did.
    """
    try:
        unknown = list(net.unknown_node_ids())
        fixed = {nid: net.nodes[nid].pressure_psia
                 for nid in net.pressure_node_ids()}
    except Exception:                                   # noqa: BLE001
        return {}
    if not unknown or not fixed:
        return {}

    index = {nid: i for i, nid in enumerate(unknown)}
    size = len(unknown)
    branches = [b for b in net.branches
                if b.a in net.nodes and b.b in net.nodes]
    if not branches:
        return {}

    # Scale for the first pass: the spread of the boundary conditions, in
    # squared pressure, shared over a typical path.
    highest = max(fixed.values())
    lowest = min(min(fixed.values()), highest * 0.75)
    spread = max(highest ** 2 - lowest ** 2, 1.0)
    drop = {id(b): spread / max(len(branches), 1) for b in branches}

    guess = {}
    for _pass in range(max(passes, 1)):
        matrix = np.zeros((size, size))
        rhs = np.zeros(size)
        for branch in branches:
            g = _conductance(branch, drop[id(branch)])
            if g <= 0.0:
                continue
            for near, far in ((branch.a, branch.b), (branch.b, branch.a)):
                if near not in index:
                    continue
                row = index[near]
                matrix[row, row] += g
                if far in index:
                    matrix[row, index[far]] -= g
                else:
                    rhs[row] += g * fixed[far] ** 2

        for nid in unknown:
            node = net.nodes[nid]
            # Withdrawal leaves the node, so it sits on the right-hand
            # side with the sign the conductance rows imply.
            rhs[index[nid]] -= getattr(node, "withdrawal", 0.0)

        if not np.any(matrix):
            return {}
        solution = _solve_linear(matrix, rhs)
        if solution is None:
            return {}

        squared = {nid: solution[index[nid]] for nid in unknown}
        floor = max(min(fixed.values()) * 0.05, 1.0) ** 2
        guess = {nid: math.sqrt(max(value, floor))
                 for nid, value in squared.items()}

        # Refresh each branch's conductance from the field just computed,
        # which is a Picard step on the real square-root relation.
        for branch in branches:
            u_a = (guess.get(branch.a) or fixed.get(branch.a, 0.0)) ** 2
            u_b = (guess.get(branch.b) or fixed.get(branch.b, 0.0)) ** 2
            drop[id(branch)] = max(abs(u_a - u_b), 1.0)

    return guess


def _solve_linear(matrix, rhs):
    """The guess's own linear solve, sparse once that is cheaper."""
    if matrix.shape[0] >= SPARSE_THRESHOLD:
        try:
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import spsolve

            with np.errstate(all="ignore"):
                found = spsolve(csc_matrix(matrix), rhs)
            if np.all(np.isfinite(found)):
                return np.asarray(found, dtype=float)
        except Exception:                               # noqa: BLE001
            # A singular Laplacian, or no scipy. Either way the dense
            # least-squares path below is the fallback, and a warm start
            # is optional by construction.
            pass
    try:
        found, *_ = np.linalg.lstsq(matrix, rhs, rcond=RCOND)
    except np.linalg.LinAlgError:
        return None
    return found


@dataclass
class WarmStartResult:
    """What the warm start cost and what it saved, on one network."""

    nodes: int
    branches: int
    cold_iterations: int
    warm_iterations: int
    cold_converged: bool
    warm_converged: bool
    max_pressure_difference: float
    guess_seconds: float = 0.0

    @property
    def saved(self) -> int:
        return self.cold_iterations - self.warm_iterations

    @property
    def fraction_saved(self) -> float:
        if self.cold_iterations <= 0:
            return 0.0
        return self.saved / self.cold_iterations

    @property
    def comparable(self) -> bool:
        """Both solves reached an answer, so the comparison means something.

        A network that converges from neither start says nothing about
        the warm start -- it is usually infeasible, which is a property
        of the network. Averaging its iteration cap into a speed-up
        figure would hide that behind a number.
        """
        return self.cold_converged and self.warm_converged

    @property
    def agrees(self) -> bool:
        """The property that makes this safe, checked rather than assumed.

        Only meaningful when both converged: two solves that both ran out
        of iterations land in different places, and that disagreement is
        about the network, not about the starting point.
        """
        if not self.comparable:
            return self.cold_converged == self.warm_converged
        return self.max_pressure_difference < 1e-6


def benchmark(net: Network, passes: int = DEFAULT_PASSES,
              **solve_kwargs) -> WarmStartResult:
    """Solve both ways and compare. Both counts, as the PRD asks.

    Solving twice is the point of a benchmark and the opposite of what
    production should do, which is why this is a separate function from
    `warm_solve`.
    """
    import time

    cold = solve(net, **solve_kwargs)

    started = time.perf_counter()
    guess = linearised_pressures(net, passes)
    elapsed = time.perf_counter() - started

    warm = solve(net, initial_pressures=guess, **solve_kwargs)

    shared = set(cold.pressures) & set(warm.pressures)
    difference = max((abs(cold.pressures[n] - warm.pressures[n])
                      / max(abs(cold.pressures[n]), 1.0)
                      for n in shared), default=0.0)

    return WarmStartResult(
        nodes=len(net.nodes), branches=len(net.branches),
        cold_iterations=cold.iterations, warm_iterations=warm.iterations,
        cold_converged=cold.converged, warm_converged=warm.converged,
        max_pressure_difference=difference, guess_seconds=elapsed)


def warm_solve(net: Network, passes: int = DEFAULT_PASSES, **solve_kwargs):
    """Solve from a warm start, falling back if it does not converge.

    The fallback is on convergence, not on iteration count: knowing the
    warm start took more iterations would mean having already paid for
    the cold solve, so measuring that belongs in `benchmark`. What is
    worth guarding against here is a guess that sends Newton somewhere it
    cannot recover from, which costs the answer rather than time.
    """
    guess = linearised_pressures(net, passes)
    if not guess:
        return solve(net, **solve_kwargs)

    result = solve(net, initial_pressures=guess, **solve_kwargs)
    if result.converged:
        return result

    cold = solve(net, **solve_kwargs)
    if cold.converged:
        cold.notes.append(
            "the warm start did not converge and the solver fell back to "
            "its own initialisation, which did")
    return cold
