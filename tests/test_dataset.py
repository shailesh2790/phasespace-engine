"""Sanity tests for the synthetic dataset generator.

The generator's only job is to hand the surrogate nothing the deterministic
solver didn't produce (PRD 5.2.1) — these tests check that every generated
sample is a genuinely solved, mass-conserving network, not just "some
numbers that didn't crash."
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phasespace.physics.dataset import generate_dataset, generate_sample


def test_generate_sample_produces_valid_graph_shapes():
    rng = random.Random(0)
    sample = None
    for _ in range(50):
        sample = generate_sample(rng)
        if sample is not None:
            break
    assert sample is not None

    n_nodes = sample.node_features.shape[0]
    n_edges = sample.edge_index.shape[0]
    assert 5 <= n_nodes <= 15
    assert sample.node_targets.shape == (n_nodes,)
    assert sample.edge_features.shape == (n_edges, 2)
    assert sample.edge_targets.shape == (n_edges,)
    assert sample.edge_index.min() >= 0
    assert sample.edge_index.max() < n_nodes
    # Exactly one source node per sample (column 0 of node_features).
    assert sample.node_features[:, 0].sum() == 1.0


def test_generated_samples_conserve_mass_at_every_demand_node():
    samples = generate_dataset(30, seed=7)
    assert len(samples) == 30
    for sample in samples:
        n_nodes = sample.node_features.shape[0]
        balance = -sample.node_features[:, 2]  # start at -demand
        for (a, b), flow in zip(sample.edge_index, sample.edge_targets):
            balance[a] -= flow
            balance[b] += flow
        is_demand = sample.node_features[:, 1] == 1.0
        max_imbalance = abs(balance[is_demand]).max() if is_demand.any() else 0.0
        # Flows are ~1e6-1e8 scf/day stored as float32 (~7 significant
        # digits), so a few units of absolute imbalance is expected
        # rounding noise, not a solver defect — tolerance is relative to
        # the sample's own flow scale.
        flow_scale = max(abs(sample.edge_targets).max(), 1.0)
        assert max_imbalance < 1e-5 * flow_scale


def test_generate_dataset_is_reproducible_given_a_seed():
    a = generate_dataset(20, seed=123)
    b = generate_dataset(20, seed=123)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert (sa.node_targets == sb.node_targets).all()
        assert (sa.edge_targets == sb.edge_targets).all()
