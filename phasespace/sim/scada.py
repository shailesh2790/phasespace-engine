"""Synthetic SCADA history: what a customer's historian would actually hand you.

Calibration and leak detection cannot be trusted until they have been run
against data where the answer is known. Real operating data never comes with
a label -- nobody can tell you the true leak rate on a live line -- so the
only way to establish that the method works is to plant a defect, generate
the measurements it would produce, and check the method finds it.

This generator is therefore a test instrument, not a demo prop. It models
the three things that make real historian data hard, all of which will
defeat a naive calibration:

  * **Measurement noise.** Every reading is a sample, not a value.
  * **Meter bias.** Real meters drift, and a biased meter looks exactly like
    a small leak if you only check mass balance. This is the single most
    common false positive in unaccounted-for-gas work.
  * **Demand variation.** A network only reveals its parameters when it is
    exercised. Calibrating on flat overnight demand is close to useless,
    because many parameter combinations fit an unstressed network equally
    well.

Swapping to a real historian export later means replacing `generate_history`
and nothing else: everything downstream consumes `ScadaHistory`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .solver import FLOW, solve

PRESSURE_TAG = "pressure"
FLOW_TAG = "flow"


@dataclass
class Meter:
    """One instrument on the network.

    `noise_std` is in the meter's own units (psia or scf/day). `bias_factor`
    is multiplicative and is the thing that makes leak detection hard: a
    meter reading 1% low is indistinguishable from a 1% leak on a mass
    balance alone.
    """

    tag: str
    kind: str                 # PRESSURE_TAG or FLOW_TAG
    node_id: int = -1         # for pressure meters, and for flow at a node
    branch_name: str = ""     # for flow meters on a branch
    noise_std: float = 0.0
    bias_factor: float = 1.0

    def __post_init__(self):
        if self.kind not in (PRESSURE_TAG, FLOW_TAG):
            raise ValueError("meter kind must be %r or %r, got %r"
                             % (PRESSURE_TAG, FLOW_TAG, self.kind))
        if self.noise_std < 0.0:
            raise ValueError("noise_std cannot be negative, got %r" % (self.noise_std,))


@dataclass
class Snapshot:
    """All meter readings at one instant, plus the true state behind them."""

    timestamp_s: float
    readings: dict = field(default_factory=dict)        # tag -> measured value
    truth: dict = field(default_factory=dict)           # tag -> noise-free value
    converged: bool = True


@dataclass
class ScadaHistory:
    meters: list
    snapshots: list
    notes: list = field(default_factory=list)

    def tags(self) -> list:
        return [m.tag for m in self.meters]

    def series(self, tag: str) -> list:
        return [s.readings[tag] for s in self.snapshots if tag in s.readings]

    def truth_series(self, tag: str) -> list:
        return [s.truth[tag] for s in self.snapshots if tag in s.truth]

    def to_csv(self, path) -> None:
        """Write in the long format historians actually export."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("timestamp_s,tag,value\n")
            for snapshot in self.snapshots:
                for tag, value in sorted(snapshot.readings.items()):
                    handle.write("%.1f,%s,%.6g\n" % (snapshot.timestamp_s, tag, value))


def load_history(rows, meters: list) -> ScadaHistory:
    """Build a history from long-format historian rows.

    `rows` is any iterable of (timestamp, tag, value) triples -- a parsed CSV,
    a database cursor, an OPC-UA subscription buffer. Long format is what
    every historian exports, because a wide table breaks the moment one
    instrument is added or renamed.

    Only tags named in `meters` are kept. A real export carries hundreds of
    tags, most of them irrelevant (valve positions, alarms, ambient
    temperature), and silently ingesting them would let an unrelated tag with
    a familiar-looking name join the calibration as though it were an
    instrument.

    Rows are grouped into snapshots by exact timestamp. Instruments in a real
    historian do not report on a shared clock, so a caller feeding raw
    sub-minute data should resample to a common grid first. This function
    reports how many snapshots came out incomplete rather than quietly
    interpolating across the gaps -- an interpolated pressure that never
    existed is indistinguishable downstream from a measured one, and it would
    be calibrated against as if it were real.
    """
    wanted = {m.tag: m for m in meters}
    by_time = {}
    unknown_tags = set()

    for timestamp, tag, value in rows:
        if tag not in wanted:
            unknown_tags.add(tag)
            continue
        try:
            t = float(timestamp)
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(t) and math.isfinite(v)):
            continue
        by_time.setdefault(t, {})[tag] = v

    snapshots = [Snapshot(timestamp_s=t, readings=readings)
                 for t, readings in sorted(by_time.items())]

    notes = []
    if unknown_tags:
        notes.append("ignored %d tag(s) not declared as meters (e.g. %s)"
                     % (len(unknown_tags), ", ".join(sorted(unknown_tags)[:4])))
    incomplete = sum(1 for s in snapshots if len(s.readings) < len(wanted))
    if incomplete:
        notes.append(
            "%d of %d snapshots are missing at least one instrument. They are "
            "used where they still carry a usable measurement and skipped where "
            "they do not -- no value is interpolated across a gap."
            % (incomplete, len(snapshots)))
    missing = sorted(set(wanted) - {tag for s in snapshots for tag in s.readings})
    if missing:
        notes.append("no data at all for: %s" % ", ".join(missing))

    return ScadaHistory(meters=meters, snapshots=snapshots, notes=notes)


def read_csv_rows(path):
    """Yield (timestamp, tag, value) from a long-format CSV with a header.

    Column names are matched loosely, because no two historians spell them
    the same way and asking a customer to rename columns before their first
    look at the tool is a poor way to start a pilot.
    """
    import csv

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return
        lowered = [h.strip().lower() for h in header]
        try:
            t_i = next(i for i, h in enumerate(lowered) if "time" in h or "stamp" in h)
            tag_i = next(i for i, h in enumerate(lowered) if "tag" in h or "name" in h)
            val_i = next(i for i, h in enumerate(lowered) if "value" in h or "reading" in h)
        except StopIteration:
            raise ValueError(
                "could not find timestamp / tag / value columns in header %r -- "
                "expected long format, one row per instrument per timestamp"
                % (header,))
        for row in reader:
            if len(row) > max(t_i, tag_i, val_i):
                yield row[t_i], row[tag_i].strip(), row[val_i]


# ------------------------------------------------------------- defects ------


@dataclass
class LeakDefect:
    """Gas leaving the network at a node, seen by no meter.

    This is the defect that matters commercially. It shows up two ways at
    once, and using both is what makes localisation possible: it breaks the
    metered mass balance (giving its size) and it steepens the pressure
    gradient upstream of itself while relaxing it downstream (giving its
    position).
    """

    node_id: int
    rate_scfd: float


@dataclass
class EfficiencyDefect:
    """A branch losing hydraulic efficiency -- fouling, wax, partial blockage,
    or a valve someone left part-closed.

    The confuser for leak detection: it also lowers downstream pressure, but
    it conserves mass. Any method that only watches pressures will call this
    a leak.
    """

    branch_name: str
    factor: float  # multiplies the branch's efficiency, e.g. 0.8


def generate_history(
    build_network,
    demand_schedule,
    meters: list,
    duration_s: float = 86400.0,
    interval_s: float = 3600.0,
    leak: LeakDefect = None,
    efficiency_defect: EfficiencyDefect = None,
    seed: int = 0,
) -> ScadaHistory:
    """Simulate the network through time and record what the meters would see.

    `build_network` is a zero-argument callable returning a fresh Network --
    fresh because control devices mutate their own state when they switch
    mode, so reusing one instance across snapshots would leak yesterday's
    regulator position into today's solve.

    `demand_schedule` maps (timestamp, node_id) -> withdrawal, so a daily
    profile, a weekday/weekend pattern, or a step test are all the same call.
    """
    rng = np.random.default_rng(seed)
    snapshots = []
    notes = []

    n_steps = int(round(duration_s / interval_s))
    for step in range(n_steps + 1):
        t = step * interval_s
        net = build_network()

        for node in net.nodes.values():
            if node.kind == FLOW:
                node.withdrawal = demand_schedule(t, node.id)

        # The planted leak is an extra withdrawal that no meter sees. That is
        # precisely what makes it a leak rather than an offtake.
        if leak is not None:
            net.nodes[leak.node_id].withdrawal += leak.rate_scfd

        if efficiency_defect is not None:
            for branch in net.branches:
                if getattr(branch, "name", "") == efficiency_defect.branch_name:
                    branch.efficiency = branch.efficiency * efficiency_defect.factor

        result = solve(net)
        if not result.converged:
            notes.append("snapshot at t=%.0f s did not converge; skipped" % t)
            continue

        flows_by_name = {getattr(b, "name", ""): q for b, q in result.flows.items()}

        truth = {}
        readings = {}
        for meter in meters:
            if meter.kind == PRESSURE_TAG:
                value = result.pressures[meter.node_id]
            elif meter.branch_name:
                value = flows_by_name.get(meter.branch_name, 0.0)
            else:
                # A flow meter at a delivery node reads that node's offtake --
                # the *metered* offtake, deliberately excluding any leak, since
                # a leak is by definition gas that no meter accounts for.
                value = demand_schedule(t, meter.node_id)

            truth[meter.tag] = value
            noisy = value * meter.bias_factor
            if meter.noise_std > 0.0:
                noisy += rng.normal(0.0, meter.noise_std)
            readings[meter.tag] = noisy

        snapshots.append(Snapshot(timestamp_s=t, readings=readings, truth=truth))

    return ScadaHistory(meters=meters, snapshots=snapshots, notes=notes)


def daily_demand_profile(base_by_node: dict, swing: float = 0.35,
                         peak_hour: float = 19.0):
    """A realistic city-gas demand shape: a morning and an evening peak.

    Returned as a schedule callable. Demand variation is not decoration --
    a network only reveals its parameters when it is exercised across a
    range of flows, so a generator that produced flat demand would make the
    calibration below look far better than it deserves to.
    """
    def schedule(t_s: float, node_id: int) -> float:
        base = base_by_node.get(node_id, 0.0)
        if base == 0.0:
            return 0.0
        hour = (t_s / 3600.0) % 24.0
        evening = math.exp(-((hour - peak_hour) ** 2) / 8.0)
        morning = 0.7 * math.exp(-((hour - 8.0) ** 2) / 6.0)
        shape = (evening + morning) / 1.7
        return base * (1.0 - swing + 2.0 * swing * shape)

    return schedule
