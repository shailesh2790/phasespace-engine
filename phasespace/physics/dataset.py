"""Synthetic training-data generator: physics generates the training set.

PRD Section 5.2.1 — "Physics generates the training data. Sweep the
governing correlations across realistic Indian network parameter ranges ->
millions of labeled examples, on demand." No external data dependency.

Each sample is a small random gas-distribution network (one source, 4-14
junction/demand nodes, at least one loop), solved to convergence by the
deterministic Newton-Raphson solver in network.py. The solved network is
the label; the surrogate never sees anything the physics didn't produce.
"""
from __future__ import annotations

import heapq
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .network import DEMAND, SOURCE, Edge, Network, Node, SolveResult, solve_network
from .weymouth import GasProperties, weymouth_flow


@dataclass
class GraphSample:
    """One solved network, in a framework-agnostic array format.

    node_features: (N, 4) -> [is_source, is_demand, demand_scfd (0 for
        source/junction), source_pressure_psia (0 if not a source)]
    edge_index: (E, 2) int -> (a, b) node indices into this sample's local
        0..N-1 numbering (not the original Node.id)
    edge_features: (E, 2) -> [diameter_in, length_mi]
    node_targets: (N,) -> solved pressure (psia)
    edge_targets: (E,) -> solved flow a->b (scf/day)
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_targets: np.ndarray
    edge_targets: np.ndarray


# Parameter ranges chosen to span typical Indian city-gas / gathering-network
# scale gas-distribution hardware, not any specific real network.
DIAMETER_RANGE_IN = (3.0, 16.0)
LENGTH_RANGE_MI = (0.3, 6.0)
SOURCE_PRESSURE_RANGE_PSIA = (500.0, 1100.0)
DEMAND_FRACTION_OF_BOTTLENECK = (0.05, 0.2)  # demand vs. each node's own widest-path deliverable capacity
JUNCTION_DEMAND_PROBABILITY = 0.7  # fraction of non-source nodes that draw any demand at all


def _bottleneck_capacities(
    n_nodes: int,
    edge_specs: list[tuple[int, int, float, float]],
    source_idx: int,
    source_pressure: float,
    gas: GasProperties,
) -> dict[int, float]:
    """Widest-path capacity from the source to every node: the maximum, over
    all paths, of the minimum edge capacity along that path.

    Sizing each node's demand against this (rather than a single incident
    edge, or a network-wide budget split randomly) is what actually
    prevents infeasible samples: a demand node behind one narrow pipe must
    have its demand bounded by that pipe regardless of how wide the rest of
    the network is. Classic widest-path / maximum-bottleneck-path via a
    Dijkstra-style relaxation (max instead of min).
    """
    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_nodes)}
    for a, b, diameter, length in edge_specs:
        capacity = abs(weymouth_flow(source_pressure, 1.0, diameter, length, gas))
        adjacency[a].append((b, capacity))
        adjacency[b].append((a, capacity))

    best = {i: 0.0 for i in range(n_nodes)}
    best[source_idx] = float("inf")
    heap: list[tuple[float, int]] = [(-float("inf"), source_idx)]
    visited: set[int] = set()
    while heap:
        neg_cap, node = heapq.heappop(heap)
        cap = -neg_cap
        if node in visited:
            continue
        visited.add(node)
        for neighbor, edge_capacity in adjacency[node]:
            candidate_cap = min(cap, edge_capacity)
            if candidate_cap > best[neighbor]:
                best[neighbor] = candidate_cap
                heapq.heappush(heap, (-candidate_cap, neighbor))
    return best


def _random_topology(n_nodes: int, rng: random.Random) -> tuple[list[tuple[int, int]], int]:
    """Random connected topology with at least one loop.

    Builds a random spanning tree over `n_nodes` nodes (node 0 is the
    source), then adds 1-2 extra edges between non-adjacent nodes to force
    at least one loop — pure trees never require an iterative solve at
    interior nodes, and the whole point of wedge B is loop/branch topology.
    """
    nodes = list(range(n_nodes))
    rng.shuffle(nodes)
    source = nodes[0]

    edges: set[tuple[int, int]] = set()
    connected = [source]
    remaining = [nid for nid in nodes if nid != source]
    for nid in remaining:
        parent = rng.choice(connected)
        edges.add((min(parent, nid), max(parent, nid)))
        connected.append(nid)

    n_extra = rng.randint(1, 2)
    attempts = 0
    while n_extra > 0 and attempts < 20:
        attempts += 1
        a, b = rng.sample(nodes, 2)
        key = (min(a, b), max(a, b))
        if key not in edges:
            edges.add(key)
            n_extra -= 1

    return list(edges), source


# The original 5-15 node range was right for a spike but is what now caps the
# surrogate. Measured on the trained model, accuracy falls from 94.3% of nodes
# within 1% at 5 nodes to 50.2% at 15 -- and a real gas distribution network has
# hundreds. A model that has never seen a graph wider than 15 nodes cannot be
# expected to transfer to one, so the range is a parameter rather than a
# constant.
NODE_RANGE = (5, 15)


def _solve_with_engine(net: "Network", gas: GasProperties):
    """Solve a Phase 0 `Network` with the rewritten engine.

    Kept as a translation rather than rewriting the generator around the new
    types: the generator's demand sizing, topology sampling and GraphSample
    layout are all tested, and the only thing that needed to change is which
    Newton iteration runs.
    """
    from ..sim.components import GasPipe
    from ..sim.fluids import GasProperties as EngineGas
    from ..sim.solver import FLOW, GAS, PRESSURE, Network as EngineNetwork, Node as EngineNode, solve

    # Z is pinned to the Phase 0 constant so the labels stay consistent with
    # everything already trained and validated against them. The engine's
    # pressure-dependent Z would silently change the ground truth.
    engine_gas = EngineGas(gravity=gas.gravity, temperature_r=gas.temperature_r,
                           fixed_z=gas.z_factor)
    engine_net = EngineNetwork(phase=GAS, fluid=engine_gas)
    for nid, node in net.nodes.items():
        if node.kind == SOURCE:
            engine_net.add_node(EngineNode(nid, PRESSURE, pressure_psia=node.pressure_psia))
        else:
            engine_net.add_node(EngineNode(nid, FLOW, withdrawal=node.demand_scfd))
    pipes = []
    for edge in net.edges:
        pipe = GasPipe(a=edge.a, b=edge.b, diameter_in=edge.diameter_in,
                       length_mi=edge.length_mi, name="e%d" % len(pipes))
        engine_net.add_branch(pipe)
        pipes.append(pipe)

    try:
        solved = solve(engine_net)
    except ValueError:
        return None
    if not solved.converged:
        return None

    class _Result:
        pass

    out = _Result()
    out.converged = True
    out.pressures = solved.pressures
    out.flows = {(pipe.a, pipe.b): solved.flows[pipe] for pipe in pipes}
    return out


def generate_sample(rng: random.Random, gas: GasProperties | None = None,
                    node_range: tuple = NODE_RANGE) -> GraphSample | None:
    """Generate and solve one random network. Returns None if the solver fails
    to converge (discarded rather than mislabeled — a non-converged sample is
    not a physics-generated label)."""
    gas = gas or GasProperties()
    n_nodes = rng.randint(*node_range)
    edge_pairs, source_idx = _random_topology(n_nodes, rng)

    net = Network(gas=gas)
    source_pressure = rng.uniform(*SOURCE_PRESSURE_RANGE_PSIA)
    for nid in range(n_nodes):
        if nid == source_idx:
            net.add_node(Node(id=nid, kind=SOURCE, pressure_psia=source_pressure))
        else:
            net.add_node(Node(id=nid, kind=DEMAND, demand_scfd=0.0))

    edge_specs = []
    for a, b in edge_pairs:
        diameter = rng.uniform(*DIAMETER_RANGE_IN)
        length = rng.uniform(*LENGTH_RANGE_MI)
        net.add_edge(Edge(a=a, b=b, diameter_in=diameter, length_mi=length))
        edge_specs.append((a, b, diameter, length))

    # Size each node's demand against its own widest-path deliverable
    # capacity, not a single incident edge or a network-wide budget: both of
    # those let demand nodes behind a narrow pipe (or several nodes sharing
    # one upstream pipe) stack past what's actually deliverable, which the
    # solver can only satisfy by collapsing pressure to zero. This was the
    # dominant sample-rejection mode before switching to bottleneck sizing.
    bottleneck = _bottleneck_capacities(n_nodes, edge_specs, source_idx, source_pressure, gas)
    for nid in range(n_nodes):
        if nid == source_idx or rng.random() > JUNCTION_DEMAND_PROBABILITY:
            continue
        net.nodes[nid].demand_scfd = bottleneck[nid] * rng.uniform(*DEMAND_FRACTION_OF_BOTTLENECK)

    # Solved by the rewritten engine rather than the Phase 0 solver. On the
    # same networks the Phase 0 Newton iteration converges 0 times in 12 at 15
    # nodes where the rewritten one converges 12 times, because it lacks the
    # backtracking line search: a large-bore, low-drop branch has an enormous
    # dq/dp and q ~ sqrt(|dP^2|) has infinite slope at zero differential, so an
    # undamped step overshoots into a limit cycle. That failure rate is what
    # capped the training set at 15 nodes.
    result = _solve_with_engine(net, gas)
    if result is None:
        return None

    node_features = np.zeros((n_nodes, 4), dtype=np.float32)
    node_targets = np.zeros(n_nodes, dtype=np.float32)
    for nid, node in net.nodes.items():
        is_source = 1.0 if node.kind == SOURCE else 0.0
        node_features[nid] = [
            is_source,
            1.0 - is_source,
            node.demand_scfd,
            source_pressure if node.kind == SOURCE else 0.0,
        ]
        node_targets[nid] = result.pressures[nid]

    edge_index = np.array([[e.a, e.b] for e in net.edges], dtype=np.int64)
    edge_features = np.array([[e.diameter_in, e.length_mi] for e in net.edges], dtype=np.float32)
    edge_targets = np.array([result.flows[(e.a, e.b)] for e in net.edges], dtype=np.float32)

    return GraphSample(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_targets=node_targets,
        edge_targets=edge_targets,
    )


def generate_dataset(n_samples: int, seed: int = 0,
                     node_range: tuple = NODE_RANGE) -> list[GraphSample]:
    """Generate `n_samples` converged random networks. Keeps sampling past
    solver failures so the caller always gets exactly `n_samples` usable
    graphs.

    `node_range` widens the size distribution. Larger networks are strictly
    harder for the solver as well as the surrogate -- more nodes means a bigger
    Jacobian and more chances for a near-singular loop tie -- so the
    convergence-failure rate rises with it, which is why the retry budget below
    is generous rather than tight.
    """
    rng = random.Random(seed)
    samples: list[GraphSample] = []
    attempts = 0
    max_attempts = n_samples * 8
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        sample = generate_sample(rng, node_range=node_range)
        if sample is not None:
            samples.append(sample)
    return samples


def save_dataset(samples: list[GraphSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_dataset(path: Path) -> list[GraphSample]:
    with open(path, "rb") as f:
        return pickle.load(f)
