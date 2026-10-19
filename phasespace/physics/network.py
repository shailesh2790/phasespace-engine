"""Steady-state gas pipeline network model + deterministic Newton-Raphson solver.

This is the "deterministic fallback engine" from PRD Section 5.1: it is both
the ground-truth generator for synthetic training data (Section 5.2.1) and
the always-shown audit companion in the UI (Section 6) — never hidden behind
the surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .weymouth import GasProperties, weymouth_flow, weymouth_flow_and_grad

SOURCE = "source"
DEMAND = "demand"


@dataclass
class Node:
    id: int
    kind: str  # SOURCE or DEMAND
    pressure_psia: float | None = None  # fixed for SOURCE nodes, solved for DEMAND nodes
    demand_scfd: float = 0.0  # consumption at DEMAND nodes (0 for pure junctions)


@dataclass
class Edge:
    a: int  # node id
    b: int  # node id
    diameter_in: float
    length_mi: float
    roughness_in: float = 0.0007  # stored for future friction-factor use; unused by Weymouth


@dataclass
class Network:
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    gas: GasProperties = field(default_factory=GasProperties)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def source_ids(self) -> list[int]:
        return [n.id for n in self.nodes.values() if n.kind == SOURCE]

    def demand_ids(self) -> list[int]:
        return [n.id for n in self.nodes.values() if n.kind == DEMAND]

    def edges_at(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges if e.a == node_id or e.b == node_id]


@dataclass
class SolveResult:
    pressures: dict[int, float]  # psia, all nodes
    flows: dict[tuple[int, int], float]  # (a, b) -> scf/day, positive = a -> b
    converged: bool
    iterations: int
    max_residual: float


def solve_network(
    net: Network,
    rel_tol: float = 1e-6,
    max_iter: int = 100,
    init_pressure: float | None = None,
) -> SolveResult:
    """Newton-Raphson solve for unknown pressures at DEMAND nodes.

    Unknowns: pressure at every DEMAND node. Residual at node j:
        R_j = (sum of flow into j across incident edges) - demand_j
    Jacobian is assembled from the Weymouth equation's exact partial
    derivatives (see weymouth_flow_and_grad), not a finite-difference
    approximation: near a loop tie's near-zero flow, any finite-difference
    step is either too coarse to resolve the true local slope or fine
    enough to hit floating-point noise, which stalls Newton in a limit
    cycle instead of converging.

    Gas flow rates span many orders of magnitude with pipe size and demand,
    so the convergence tolerance is relative to total system throughput
    rather than a fixed absolute constant — a fixed absolute tolerance that
    looks fine for one network can silently fail to resolve on another an
    order of magnitude larger or smaller.
    """
    demand_ids = net.demand_ids()
    n = len(demand_ids)
    idx = {nid: i for i, nid in enumerate(demand_ids)}

    if init_pressure is None:
        source_pressures = [net.nodes[i].pressure_psia for i in net.source_ids()]
        init_pressure = min(source_pressures) * 0.9 if source_pressures else 500.0

    p = np.full(n, init_pressure, dtype=float)

    total_demand = sum(abs(net.nodes[nid].demand_scfd) for nid in demand_ids)
    scale = max(total_demand, 1.0)
    tol = rel_tol * scale

    def pressures_from(p_vec: np.ndarray) -> dict[int, float]:
        pr = {nid: node.pressure_psia for nid, node in net.nodes.items() if node.kind == SOURCE}
        for nid, i in idx.items():
            pr[nid] = p_vec[i]
        return pr

    def edge_flow(e: Edge, pr: dict[int, float]) -> float:
        return weymouth_flow(pr[e.a], pr[e.b], e.diameter_in, e.length_mi, net.gas)

    def residuals_and_jacobian(p_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pr = pressures_from(p_vec)
        r = np.zeros(n)
        jac = np.zeros((n, n))
        for e in net.edges:
            q_ab, dq_da, dq_db = weymouth_flow_and_grad(pr[e.a], pr[e.b], e.diameter_in, e.length_mi, net.gas)
            if e.b in idx:
                j = idx[e.b]
                r[j] += q_ab
                if e.a in idx:
                    jac[j, idx[e.a]] += dq_da
                jac[j, j] += dq_db
            if e.a in idx:
                i = idx[e.a]
                r[i] -= q_ab
                jac[i, i] -= dq_da
                if e.b in idx:
                    jac[i, idx[e.b]] -= dq_db
        for nid, i in idx.items():
            r[i] -= net.nodes[nid].demand_scfd
        return r, jac

    converged = False
    iterations = 0
    max_res = float("inf")

    for iterations in range(1, max_iter + 1):
        r, jac = residuals_and_jacobian(p)
        max_res = float(np.max(np.abs(r))) if n else 0.0
        if max_res < tol:
            converged = True
            break

        # lstsq (SVD-based) rather than solve (LU-based): a Jacobian with
        # extreme dynamic range (see residuals_and_jacobian) is exactly the
        # case where direct LU factorization is fragile -- it's what
        # crashed the native LAPACK call outright on a real network with
        # several near-simultaneous demand-to-demand ties. SVD degrades
        # gracefully (a near-singular direction just gets a small-but-finite
        # contribution) instead of failing catastrophically. The network is
        # at most 15 nodes, so the extra SVD cost over LU is negligible.
        delta = np.linalg.lstsq(jac, -r, rcond=None)[0]

        # Damp large steps to keep pressures positive and iteration stable.
        step_norm = np.max(np.abs(delta)) if n else 0.0
        damping = 1.0 if step_norm < 200 else 200.0 / step_norm
        p = p + damping * delta
        p = np.maximum(p, 1.0)  # pressures must stay physical (> 0 psia)

    pr = pressures_from(p)
    flows = {(e.a, e.b): edge_flow(e, pr) for e in net.edges}
    return SolveResult(pressures=pr, flows=flows, converged=converged, iterations=iterations, max_residual=max_res)
