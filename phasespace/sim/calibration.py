"""Calibrate the model to measured data, then explain what is left over.

This is the piece that turns a simulator into something worth buying. A
study-mode model is built once and drifts; a model re-fitted to yesterday's
SCADA every morning tells you something new every morning. And once the
model matches measured reality, the *residual* is physical -- it is gas
going somewhere nobody metered.

The method rests on one asymmetry, and everything here follows from it:

    A leak breaks the metered mass balance. Loss of hydraulic efficiency
    does not.

Fouling, wax, a part-closed valve -- all of these lower downstream pressure
while conserving mass. A leak lowers downstream pressure *and* puts the
inlet meter permanently above the sum of the outlet meters. So:

  * The **metered imbalance** gives the size of the loss, and it is a
    direct measurement, not a model output. It survives being wrong about
    the hydraulics entirely.
  * The **pressure profile** gives the position, because a leak steepens the
    gradient upstream of itself and relaxes it downstream, which no uniform
    efficiency change can imitate.

Using both is what separates "your pipe is fouling" from "you are losing
gas", and that distinction is the product.

Honesty about identifiability
-----------------------------
With a handful of meters you cannot fit an efficiency for every segment and
a leak at every node -- there are more unknowns than independent
measurements, and an optimiser asked to do it will return a confident,
meaningless answer. `identifiability_report` checks this *before* fitting
and says what the data can and cannot support. A calibration that quietly
over-fits is exactly the confidently-wrong failure PRD Section 6 rules out,
and it is the easiest mistake to make in this whole module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .scada import FLOW_TAG, PRESSURE_TAG, ScadaHistory
from .solver import FLOW, PRESSURE, solve

# Efficiency is physically bounded. Letting the optimiser wander outside this
# band lets it "explain" a leak by inventing a pipe that flows better than
# physics allows, which is how a fitted model ends up hiding the very defect
# it was built to find.
MIN_EFFICIENCY = 0.50
MAX_EFFICIENCY = 1.00


@dataclass
class MassBalance:
    """The metered imbalance over one snapshot or a whole history."""

    supply_scfd: float
    metered_offtake_scfd: float

    @property
    def imbalance_scfd(self) -> float:
        return self.supply_scfd - self.metered_offtake_scfd

    @property
    def imbalance_fraction(self) -> float:
        if self.supply_scfd <= 0.0:
            return 0.0
        return self.imbalance_scfd / self.supply_scfd


def metered_mass_balance(history: ScadaHistory, supply_tags: list,
                         offtake_tags: list) -> MassBalance:
    """Average supply and metered offtake across the history.

    Averaged over time on purpose. An instantaneous imbalance on a gas
    network is dominated by line pack -- the line breathes with demand, so a
    single snapshot can show a large apparent loss or gain that is simply
    storage. Only the time average over a period long enough to return to
    similar pack is a real loss.
    """
    supply_total = 0.0
    offtake_total = 0.0
    count = 0
    for snapshot in history.snapshots:
        if not all(tag in snapshot.readings for tag in supply_tags + offtake_tags):
            continue
        supply_total += sum(snapshot.readings[tag] for tag in supply_tags)
        offtake_total += sum(snapshot.readings[tag] for tag in offtake_tags)
        count += 1
    if count == 0:
        return MassBalance(0.0, 0.0)
    return MassBalance(supply_total / count, offtake_total / count)


@dataclass
class CalibrationResult:
    efficiencies: dict            # branch name -> fitted efficiency
    initial_efficiencies: dict
    rms_pressure_error_psi: float
    max_pressure_error_psi: float
    per_tag_error_psi: dict
    iterations: int
    converged: bool
    notes: list = field(default_factory=list)

    def drift(self) -> dict:
        """Fractional change from the as-designed efficiency, per branch.

        The number an operator reads: -0.23 means that segment is passing
        gas as though it were 23% less efficient than design, which is a
        maintenance work order, not a modelling curiosity.
        """
        return {
            name: (self.efficiencies[name] / self.initial_efficiencies[name] - 1.0)
            for name in self.efficiencies
            if self.initial_efficiencies.get(name)
        }


def _apply_state(net, demands: dict, source_pressures: dict, efficiencies: dict,
                 extra_withdrawals: dict = None):
    for node_id, value in demands.items():
        if node_id in net.nodes and net.nodes[node_id].kind == FLOW:
            net.nodes[node_id].withdrawal = value
    for node_id, value in (extra_withdrawals or {}).items():
        if node_id in net.nodes and net.nodes[node_id].kind == FLOW:
            net.nodes[node_id].withdrawal += value
    for node_id, value in source_pressures.items():
        if node_id in net.nodes and net.nodes[node_id].kind == PRESSURE:
            net.nodes[node_id].pressure_psia = value
    for branch in net.branches:
        name = getattr(branch, "name", "")
        if name in efficiencies:
            branch.efficiency = efficiencies[name]


def _subsample(snapshots: list, limit: int) -> list:
    """Evenly spaced subset, preserving coverage of the whole period."""
    if limit <= 0 or len(snapshots) <= limit:
        return list(snapshots)
    step = len(snapshots) / float(limit)
    return [snapshots[int(i * step)] for i in range(limit)]


def _snapshot_inputs(snapshot, meters):
    """Split a snapshot's readings into model inputs and model targets.

    Flow meters and source-pressure meters are *inputs* -- they set the
    boundary conditions. Interior pressure meters are *targets*: the model
    predicts them and calibration makes the prediction match. Confusing the
    two would make the fit trivially perfect and completely uninformative.
    """
    demands = {}
    source_pressures = {}
    targets = {}
    for meter in meters:
        value = snapshot.readings.get(meter.tag)
        if value is None:
            continue
        if meter.kind == FLOW_TAG and meter.node_id >= 0:
            demands[meter.node_id] = value
        elif meter.kind == PRESSURE_TAG:
            targets[meter.tag] = (meter.node_id, value)
    return demands, source_pressures, targets


def calibrate_efficiency(
    build_network,
    history: ScadaHistory,
    branch_names: list = None,
    source_pressure_tags: dict = None,
    max_iterations: int = 40,
    tolerance: float = 1e-4,
    extra_withdrawals: dict = None,
    max_snapshots: int = 48,
) -> CalibrationResult:
    """Fit per-branch hydraulic efficiency so predicted pressures match measured.

    Flow meters set the boundary conditions; interior pressure meters are
    what the fit is scored against. Solved by Levenberg-Marquardt through
    scipy, with the network solve as the forward model -- the parameter count
    is small (one per branch) and each forward evaluation is milliseconds, so
    a finite-difference Jacobian is entirely affordable and avoids having to
    differentiate through the Newton solve.

    `max_snapshots` evenly subsamples the history. A real historian holds 90
    days of five-minute data -- around 26,000 snapshots -- and every one of
    them costs a network solve on every optimiser iteration. Calibrating on
    all of them is both intractable and pointless: what a fit needs is
    *distinct operating points*, and consecutive five-minute samples are
    nearly identical. Even spacing across the whole period is deliberate, so
    the subsample still spans the full demand swing rather than one quiet
    night.
    """
    from scipy.optimize import least_squares

    source_pressure_tags = source_pressure_tags or {}
    template = build_network()
    all_names = [getattr(b, "name", "") for b in template.branches]
    if any(not name for name in all_names):
        raise ValueError(
            "every branch must have a name before calibration -- fitted "
            "efficiencies are reported per branch and an unnamed branch "
            "cannot be reported or acted on")
    if len(set(all_names)) != len(all_names):
        raise ValueError("branch names must be unique for calibration: %s" % (all_names,))

    names = list(branch_names) if branch_names else all_names
    initial = {}
    for branch in template.branches:
        initial[getattr(branch, "name", "")] = branch.efficiency
    start = np.array([initial[name] for name in names], dtype=float)

    usable = []
    for snapshot in _subsample(history.snapshots, max_snapshots):
        demands, _, targets = _snapshot_inputs(snapshot, history.meters)
        source_pressures = {
            node_id: snapshot.readings[tag]
            for tag, node_id in source_pressure_tags.items()
            if tag in snapshot.readings
        }
        # A source's own pressure meter is a boundary condition, not a target.
        for tag in source_pressure_tags:
            targets.pop(tag, None)
        if targets:
            usable.append((demands, source_pressures, targets))

    notes = []
    if not usable:
        return CalibrationResult({}, initial, 0.0, 0.0, {}, 0, False,
                                 ["no snapshot had an interior pressure measurement "
                                  "to calibrate against"])

    def residuals(theta):
        efficiencies = dict(zip(names, theta))
        out = []
        for demands, source_pressures, targets in usable:
            net = build_network()
            _apply_state(net, demands, source_pressures, efficiencies, extra_withdrawals)
            try:
                result = solve(net)
            except ValueError:
                out.extend([1e3] * len(targets))
                continue
            if not result.converged:
                # A large but finite penalty, not inf: least_squares needs a
                # usable gradient, and inf makes the whole step undefined.
                out.extend([1e3] * len(targets))
                continue
            for node_id, measured in targets.values():
                out.append(result.pressures.get(node_id, 0.0) - measured)
        return np.array(out)

    fit = least_squares(
        residuals, start,
        bounds=(MIN_EFFICIENCY, MAX_EFFICIENCY),
        xtol=tolerance, ftol=tolerance, max_nfev=max_iterations * (len(names) + 1),
    )

    fitted = dict(zip(names, [float(v) for v in fit.x]))
    final = residuals(fit.x)
    rms = float(np.sqrt(np.mean(final ** 2))) if final.size else 0.0
    worst = float(np.max(np.abs(final))) if final.size else 0.0

    per_tag = {}
    index = 0
    for _, _, targets in usable:
        for tag in targets:
            per_tag.setdefault(tag, []).append(abs(final[index]))
            index += 1
    per_tag = {tag: float(np.mean(values)) for tag, values in per_tag.items()}

    for name, value in fitted.items():
        if value >= MAX_EFFICIENCY - 1e-6:
            notes.append(
                "%s fitted at the upper efficiency bound (%.2f). The model still "
                "cannot lose as much pressure as measured, which usually means a "
                "leak or a restriction the topology does not contain."
                % (name, MAX_EFFICIENCY))
        elif value <= MIN_EFFICIENCY + 1e-6:
            notes.append(
                "%s fitted at the lower efficiency bound (%.2f) -- treat as "
                "unidentified rather than as a measurement." % (name, MIN_EFFICIENCY))

    return CalibrationResult(
        efficiencies=fitted,
        initial_efficiencies={name: initial[name] for name in names},
        rms_pressure_error_psi=rms,
        max_pressure_error_psi=worst,
        per_tag_error_psi=per_tag,
        iterations=int(fit.nfev),
        converged=bool(fit.success),
        notes=notes,
    )


@dataclass
class LeakHypothesis:
    node_id: int
    rate_scfd: float
    rms_pressure_error_psi: float
    efficiencies: dict = field(default_factory=dict)
    efficiency_distortion: float = 0.0

    @property
    def rate_mmscfd(self) -> float:
        return self.rate_scfd / 1e6


@dataclass
class LeakFinding:
    detected: bool
    imbalance: MassBalance
    best: LeakHypothesis = None
    ranked: list = field(default_factory=list)
    baseline_rms_psi: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def confidence_margin(self) -> float:
        """How much better the best location fits than the runner-up.

        Localisation is only meaningful when the winner is clearly better.
        On a network whose meters cannot tell two nodes apart this comes out
        near zero, and the honest output is then "we know how much, not
        where" -- still commercially useful, and far better than a confident
        wrong node.
        """
        if len(self.ranked) < 2 or self.ranked[0].rms_pressure_error_psi <= 0.0:
            return 0.0
        second = self.ranked[1].rms_pressure_error_psi
        first = self.ranked[0].rms_pressure_error_psi
        return (second - first) / max(second, 1e-9)


def localise_leak(
    build_network,
    history: ScadaHistory,
    supply_tags: list,
    offtake_tags: list,
    candidate_nodes: list = None,
    source_pressure_tags: dict = None,
    detection_threshold_fraction: float = 0.005,
    branch_names: list = None,
    max_snapshots: int = 48,
) -> LeakFinding:
    """Size the loss from the meters, then find where it best explains pressures.

    Two separate steps with different evidence, because they have different
    reliability. The *size* comes from metered mass balance alone: arithmetic
    on instruments, independent of the hydraulic model, and it stands even if
    the model is badly calibrated. The *location* comes from the pressure
    profile and does depend on the model, so it is reported with a margin
    saying how much better the best candidate fits than the next.

    Efficiency is re-fitted **inside each hypothesis**, and that is the whole
    trick. Calibrating once on leaking data and then testing leak locations
    against those efficiencies does not work: the calibration has already
    absorbed the leak as fake efficiency loss, so adding the leak back
    double-counts it and every location looks worse than assuming no leak at
    all. Re-fitting per hypothesis asks the right question instead -- *if the
    leak were here, would the pipe efficiencies come out sane?* At the true
    node they return close to design; at a wrong node the optimiser has to
    distort them to compensate, and both the residual and the distortion give
    it away.

    Cost is one calibration per candidate node. Networks have few nodes and a
    calibration is seconds, so this stays practical; it would need revisiting
    for a network with hundreds of candidate locations.
    """
    source_pressure_tags = source_pressure_tags or {}
    notes = []

    balance = metered_mass_balance(history, supply_tags, offtake_tags)

    if balance.supply_scfd <= 0.0:
        return LeakFinding(False, balance, notes=["no supply measurement in the history"])

    if balance.imbalance_fraction <= detection_threshold_fraction:
        notes.append(
            "metered imbalance is %.2f%% of supply, at or below the %.2f%% "
            "detection threshold -- within what meter accuracy alone can explain."
            % (100.0 * balance.imbalance_fraction,
               100.0 * detection_threshold_fraction))
        return LeakFinding(False, balance, notes=notes)

    template = build_network()
    if candidate_nodes is None:
        candidate_nodes = [n.id for n in template.nodes.values() if n.kind == FLOW]

    has_pressure_target = any(m.kind == PRESSURE_TAG and m.tag not in source_pressure_tags
                              for m in history.meters)
    if not has_pressure_target:
        notes.append(
            "imbalance of %.2f%% detected, but no interior pressure measurement "
            "is available to locate it. The size is trustworthy; the location is "
            "not determinable from this instrument set."
            % (100.0 * balance.imbalance_fraction))
        return LeakFinding(True, balance, notes=notes)

    def distortion(result):
        """How far this hypothesis had to bend the pipe efficiencies to fit."""
        drift = result.drift()
        if not drift:
            return 0.0
        return float(np.sqrt(np.mean(np.array(list(drift.values())) ** 2)))

    common = dict(source_pressure_tags=source_pressure_tags, branch_names=branch_names,
                  max_snapshots=max_snapshots)
    baseline = calibrate_efficiency(build_network, history, **common)

    rate = balance.imbalance_scfd
    hypotheses = []
    for node_id in candidate_nodes:
        fit = calibrate_efficiency(build_network, history,
                                   extra_withdrawals={node_id: rate}, **common)
        hypotheses.append(LeakHypothesis(
            node_id=node_id,
            rate_scfd=rate,
            rms_pressure_error_psi=fit.rms_pressure_error_psi,
            efficiencies=fit.efficiencies,
            efficiency_distortion=distortion(fit),
        ))

    hypotheses.sort(key=lambda h: h.rms_pressure_error_psi)
    best = hypotheses[0] if hypotheses else None

    finding = LeakFinding(
        detected=True,
        imbalance=balance,
        best=best,
        ranked=hypotheses,
        baseline_rms_psi=baseline.rms_pressure_error_psi,
        notes=notes,
    )

    finding.notes.append(
        "metered imbalance is %.2f%% of supply (%.3f MMscfd unaccounted)."
        % (100.0 * balance.imbalance_fraction, balance.imbalance_scfd / 1e6))

    if best is not None and math.isfinite(best.rms_pressure_error_psi):
        finding.notes.append(
            "node %d is the best-fitting location (%.3f psi RMS, and it leaves the "
            "pipe efficiencies within %.1f%% of design)."
            % (best.node_id, best.rms_pressure_error_psi,
               100.0 * best.efficiency_distortion))
        if finding.confidence_margin < 0.10:
            finding.notes.append(
                "it fits only %.0f%% better than the next candidate, so these "
                "meters cannot reliably separate those locations. Treat the size "
                "as firm and the location as a shortlist."
                % (100.0 * finding.confidence_margin))
        else:
            finding.notes.append(
                "that is %.0f%% better than the next candidate, which is a clear "
                "separation." % (100.0 * finding.confidence_margin))

    return finding


@dataclass
class IdentifiabilityReport:
    n_parameters: int
    n_pressure_tags: int
    demand_swing: float
    supported: bool
    notes: list = field(default_factory=list)


def identifiability_report(build_network, history: ScadaHistory,
                           branch_names: list = None,
                           minimum_swing: float = 0.05,
                           source_pressure_tags: dict = None) -> IdentifiabilityReport:
    """Can this instrument set support fitting this many parameters?

    Run before calibrating, not after. An optimiser handed more unknowns than
    independent measurements will still return numbers, and they will look as
    authoritative as a well-posed fit.

    Two conditions, and both must hold:

      * **Enough independent targets.** At least one *interior* pressure
        measurement per fitted parameter. Fewer, and infinitely many
        parameter sets reproduce the data equally well -- the fit slides
        along a ridge, trading one segment's efficiency against another's,
        and reports whichever point on that ridge the optimiser stopped at.

        A source's own pressure gauge does not count. It sets a boundary
        condition, so the model reproduces it exactly by construction and it
        constrains nothing. Counting it was a real bug here: a network with
        gauges at the gate and two delivery points looked like three targets
        for three parameters ("supported") when it had two, and the fit
        duly invented a 13% efficiency loss on the trunk that was not there.
      * **Enough excitation.** The network has to be worked across a range of
        flows. At a single operating point, a slightly worse efficiency and a
        slightly higher demand are indistinguishable, so the fit is
        determined by the starting guess rather than by the data.

    Excitation is measured as the relative spread of total metered offtake
    rather than by counting distinct values. Counting distinct readings looks
    reasonable and is wrong: meter noise alone makes every reading unique, so
    a dead-flat history scores as maximally informative. This function
    originally did exactly that and had to be fixed.
    """
    template = build_network()
    names = list(branch_names) if branch_names else [
        getattr(b, "name", "") for b in template.branches]
    n_parameters = len(names)

    source_pressure_tags = source_pressure_tags or {}
    pressure_tags = [m.tag for m in history.meters
                     if m.kind == PRESSURE_TAG and m.tag not in source_pressure_tags]
    flow_tags = [m.tag for m in history.meters if m.kind == FLOW_TAG]

    totals = []
    for snapshot in history.snapshots:
        values = [snapshot.readings[tag] for tag in flow_tags if tag in snapshot.readings]
        if values:
            totals.append(sum(values))

    if len(totals) >= 2:
        mean = float(np.mean(totals))
        swing = (float(np.max(totals)) - float(np.min(totals))) / abs(mean) if mean else 0.0
    else:
        swing = 0.0

    notes = []
    enough_targets = len(pressure_tags) >= n_parameters
    enough_swing = swing >= minimum_swing
    supported = enough_targets and enough_swing

    notes.append(
        "%d parameter(s) to fit against %d interior pressure tag(s) — boundary "
        "gauges excluded — over %d snapshots spanning a %.0f%% swing in total "
        "metered offtake."
        % (n_parameters, len(pressure_tags), len(history.snapshots), 100.0 * swing))
    if not enough_targets:
        notes.append(
            "fewer interior pressure measurements (%d) than parameters (%d): the "
            "fit is under-determined. It will still return numbers, and they will "
            "look authoritative, but they are a point on a ridge of equally good "
            "answers rather than a measurement — expect spurious drift on one "
            "segment offset by the opposite on another. Add a gauge at an "
            "un-instrumented junction, or group segments that share a condition "
            "and fit one efficiency for the group."
            % (len(pressure_tags), n_parameters))
    if not enough_swing:
        notes.append(
            "the network sits at essentially one operating point (%.1f%% swing, "
            "below the %.0f%% minimum), so efficiency and demand cannot be "
            "separated: many parameter sets fit equally well. Calibrate across "
            "a demand swing."
            % (100.0 * swing, 100.0 * minimum_swing))

    return IdentifiabilityReport(
        n_parameters=n_parameters,
        n_pressure_tags=len(pressure_tags),
        demand_swing=swing,
        supported=supported,
        notes=notes,
    )
