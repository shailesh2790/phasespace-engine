"""Who else is affected when one thing changes.

An engineer asking "can I take this well up 20%" is not really asking what
that well will do. They are asking which *other* wells will notice, and
whether any of them is near a limit. A solved network answers the first
question and is silent on the second, so the question gets answered by
intuition or by not asking it.

The method is deliberately plain: solve, perturb, solve again, difference the
two. Exact, fast, and reliable. An asynchronous local-propagation solver was
measured against this first and was a thousand times more expensive on a cold
solve; it only pays when a converged state is being nudged continuously, which
is a monitor's problem and not a web request's. The interesting result from
that experiment was not the algorithm, it was what the differences look like.

**Two measurements shape what this reports.**

Gas networks are far more sparse than intuition suggests. Moving one well's
demand by half shifted only a fraction of a percent of a 400-node field by
more than a hundredth of a psi. Most of a network genuinely does not care, and
a tool that lists every node as "affected by 0.0003 psi" buries the three that
matter.

And precision past the instrument is waste. On the same case, tightening the
solver tolerance by four orders of magnitude cost three million times the
arithmetic and moved the answer by 0.03 psi -- against a field transducer that
reads to about 2 psi. So the threshold here is the gauge, not the float: a
node counts as affected when it moves by more than the reading could resolve,
and everything else is reported as unaffected rather than as a very small
number.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .solver import FLOW, PRESSURE, solve

# What a decent field pressure transducer resolves, in psi. Typical
# accuracy is 0.25% of span, so a 1,000 psi gauge is good to about 2.5 psi.
# Changes below this are real in the model and invisible on the instrument,
# which is the honest place to put the cut.
DEFAULT_GAUGE_RESOLUTION_PSI = 2.0


@dataclass
class NodeInfluence:
    node: int
    name: str
    base_psia: float
    changed_psia: float
    is_source: bool = False

    @property
    def delta_psi(self) -> float:
        return self.changed_psia - self.base_psia

    @property
    def magnitude(self) -> float:
        return abs(self.delta_psi)


@dataclass
class InfluenceReport:
    """What moved, what did not, and how confidently we can say so."""

    source_node: int
    source_name: str
    change_fraction: float = 0.0
    affected: list = field(default_factory=list)
    unaffected_count: int = 0
    gauge_resolution_psi: float = DEFAULT_GAUGE_RESOLUTION_PSI
    base_converged: bool = True
    changed_converged: bool = True
    notes: list = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return len(self.affected) + self.unaffected_count

    @property
    def others(self) -> list:
        """The answer to the question actually asked.

        The perturbed node's own pressure drop is kept in `affected` because
        it is worth seeing, but it is never news -- of course the well you
        turned up moved. Everything that matters is here.
        """
        return [e for e in self.affected if not e.is_source]

    @property
    def sparsity(self) -> float:
        """Share of the rest of the network that actually responds."""
        responding = len(self.others)
        population = self.total_nodes - 1
        return responding / population if population > 0 else 0.0

    @property
    def worst(self) -> NodeInfluence:
        """The most affected node other than the one that was changed."""
        rest = self.others
        return rest[0] if rest else None

    @property
    def reliable(self) -> bool:
        return self.base_converged and self.changed_converged


def influence_of_demand(
    network,
    node: int,
    change_fraction: float = 0.20,
    gauge_resolution_psi: float = DEFAULT_GAUGE_RESOLUTION_PSI,
) -> InfluenceReport:
    """Which nodes move when one node's demand changes.

    ``change_fraction`` is relative: 0.20 is a 20% increase, -0.5 a halving.
    The network is not modified; the perturbation is applied to a copy.
    """
    if node not in network.nodes:
        raise ValueError("node %r is not in the network" % (node,))
    if network.nodes[node].kind != FLOW:
        raise ValueError(
            "node %r is a fixed-pressure node. Its demand is not an input, so "
            "there is nothing to perturb -- perturb its pressure instead."
            % (node,))
    if gauge_resolution_psi <= 0.0:
        raise ValueError("gauge resolution must be positive, got %r"
                         % (gauge_resolution_psi,))

    base = solve(network)

    changed_net = copy.deepcopy(network)
    target = changed_net.nodes[node]
    if target.withdrawal == 0.0:
        raise ValueError(
            "node %r withdraws nothing, so a relative change to its demand is "
            "still nothing. Give it a demand, or perturb a node that has one."
            % (node,))
    target.withdrawal *= (1.0 + change_fraction)
    changed = solve(changed_net)

    report = InfluenceReport(
        source_node=node,
        source_name=network.nodes[node].name or str(node),
        change_fraction=change_fraction,
        gauge_resolution_psi=gauge_resolution_psi,
        base_converged=base.converged,
        changed_converged=changed.converged)

    for other_id, other in network.nodes.items():
        if other.kind == PRESSURE:
            # A fixed-pressure node is held there by definition; reporting it
            # as unaffected would be reporting the boundary condition back.
            continue
        entry = NodeInfluence(
            node=other_id, name=other.name or str(other_id),
            base_psia=base.pressures.get(other_id, 0.0),
            changed_psia=changed.pressures.get(other_id, 0.0),
            is_source=(other_id == node))
        if entry.magnitude >= gauge_resolution_psi:
            report.affected.append(entry)
        else:
            report.unaffected_count += 1

    report.affected.sort(key=lambda e: -e.magnitude)

    if not report.reliable:
        report.notes.append(
            "One of the two solves did not converge, so this comparison is "
            "between an answer and a guess.")
    elif not report.others:
        report.notes.append(
            "Nothing else moves by more than the %.3g psi a gauge could read. "
            "This change is contained; it is real in the model and invisible "
            "on the rest of the field's instruments."
            % gauge_resolution_psi)
    else:
        report.notes.append(
            "%d of the other %d nodes respond above gauge resolution. The rest "
            "move by less than %.3g psi and are reported as unaffected rather "
            "than as very small numbers."
            % (len(report.others), report.total_nodes - 1,
               gauge_resolution_psi))
    return report


def influence_matrix(
    network,
    nodes: list = None,
    change_fraction: float = 0.20,
    gauge_resolution_psi: float = DEFAULT_GAUGE_RESOLUTION_PSI,
) -> dict:
    """Run the same question for several nodes.

    Costs one solve per node plus one, so it is linear rather than free. On a
    large field this belongs in a scheduled job rather than a request, which
    is exactly the regime where an incremental solver would earn its keep --
    noted here because it is the honest boundary of the simple approach, not
    because anything needs changing yet.
    """
    candidates = nodes if nodes is not None else [
        i for i, n in network.nodes.items()
        if n.kind == FLOW and n.withdrawal != 0.0]
    return {
        node: influence_of_demand(network, node, change_fraction,
                                  gauge_resolution_psi)
        for node in candidates
    }
