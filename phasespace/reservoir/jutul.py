"""Reading a JutulDarcy run, in the units it actually produced.

JutulDarcy is an MIT-licensed reservoir simulator from SINTEF. This reads
what `_jutul_worker` wrote and converts it into the field units the rest of
the engine works in. It imports no Julia, so it is testable without the
1.8 GB depot the simulator needs.

**The unit situation is measured, not documented, and it is not uniform.**
Three different conventions come back from one result:

  summary rates and pressures    field units already -- psia, stb/d,
                                 Mscf/d. Established by SPE9 reporting
                                 `WBHP` of exactly 1000, which is that
                                 benchmark's prescribed minimum bottomhole
                                 pressure of 1000 psia, and by `FOPR`
                                 opening at 36,203 against the deck's 25
                                 producers capped at 1,500 stb/d each.

  per-cell state arrays          SI. `Pressure` came back as 21.8e6 to
                                 31.2e6, which is pascals; the same cells
                                 are 3,161 to 4,521 psia.

  summary cumulatives            **86,400x too large.** Measured to
                                 machine precision on four independent
                                 rate/cumulative pairs: the reported
                                 cumulative equals the integral of the
                                 reported rate times elapsed *seconds*,
                                 while the rate itself is per *day*. So
                                 SPE9's 900-day oil production reports as
                                 1.7057e12 where the rate integrates to
                                 19,741,876 stb -- and 19.7 million
                                 barrels at an average 21,935 stb/d is
                                 what the rate curve actually describes.

That last one matters more than the others combined, because cumulative
production is the input to material balance and to every reserves number.
Wrong by four and a half orders of magnitude, it does not look like an
error; it looks like a larger field. `CUMULATIVE_SECOND_FACTOR` corrects
it, `tests/test_jutul.py` pins the correction against the integral of the
rates, and if a future version of JutulDarcy fixes the output upstream
that test fails loudly rather than the correction silently double-counting.

**Nothing here is validated.** SPE9 agreeing with SPE9 is verification
that the install and this reader work. The simulator's own authors report
matching AD-GPRS and MRST, which is code-to-code comparison and so also
verification. Whether JutulDarcy describes a particular reservoir is a
question only measured field data answers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PA_PER_PSI = 6894.757

# Cumulative summary curves are the integral of a per-day rate against
# elapsed seconds, so dividing by the seconds in a day recovers the rate's
# own volume unit. Measured, not assumed: see the module docstring and
# `test_cumulatives_are_the_integral_of_the_rates`.
CUMULATIVE_SECOND_FACTOR = 86400.0

# Eclipse summary mnemonics are positional: the first letter is the scope
# (F field, W well), the rest names the quantity. A trailing T is a
# cumulative total and a trailing R a rate, which is the whole basis for
# knowing which curves need the correction above.
CUMULATIVE_SUFFIXES = ("PT", "IT")          # FOPT, FGIT, WWIT ...


def is_cumulative(mnemonic: str) -> bool:
    """Whether a summary key is a cumulative total rather than a rate.

    Checked on the mnemonic's own grammar rather than a hand-maintained
    list, because a list would silently miss any key a future version
    adds -- and the failure mode of missing one is a number 86,400x out.
    """
    name = mnemonic.rsplit(".", 1)[-1].upper()
    return any(name.endswith(suffix) for suffix in CUMULATIVE_SUFFIXES)


def is_pressure(mnemonic: str) -> bool:
    name = mnemonic.rsplit(".", 1)[-1].upper()
    return name.endswith("BHP") or name in ("FPR", "FPRH", "FPRP")


@dataclass
class ReservoirRun:
    """One simulation, in field units, with its provenance."""

    days: object = None                   # ndarray, days since start
    field_curves: dict = field(default_factory=dict)
    well_curves: dict = field(default_factory=dict)   # well -> key -> array
    states: dict = field(default_factory=dict)        # key -> (steps, cells)
    meta: dict = field(default_factory=dict)
    corrections: list = field(default_factory=list)

    @property
    def wells(self) -> list:
        return sorted(self.well_curves)

    @property
    def steps(self) -> int:
        return 0 if self.days is None else int(len(self.days))

    @property
    def cells(self) -> int:
        pressure = self.states.get("Pressure")
        return 0 if pressure is None else int(pressure.shape[-1])

    def describe(self) -> str:
        lines = [
            "%d steps, %.0f to %.0f days, %d wells, %d cells"
            % (self.steps,
               float(self.days[0]) if self.steps else 0.0,
               float(self.days[-1]) if self.steps else 0.0,
               len(self.wells), self.cells),
        ]
        if self.meta.get("solve_seconds"):
            lines.append("solved in %.1f s" % self.meta["solve_seconds"])
        lines.extend(self.corrections)
        lines.append(
            "Verification only: a benchmark reproducing itself says the "
            "install and this reader work, not that the simulator describes "
            "any particular reservoir.")
        return "\n".join(lines)


def load(npz_path, meta_path="") -> ReservoirRun:
    """Read a worker's output and convert it to field units."""
    import numpy as np

    with np.load(str(npz_path)) as data:
        arrays = {key: data[key] for key in data.files}

    meta = {}
    if meta_path and os.path.exists(str(meta_path)):
        try:
            with open(str(meta_path)) as handle:
                meta = json.load(handle)
        except ValueError:
            meta = {}

    run = ReservoirRun(meta=meta)
    run.days = arrays.get("DAYS")

    cumulatives, pressures = 0, 0
    for key, values in arrays.items():
        if key == "DAYS":
            continue

        if key.startswith("STATE."):
            name = key.split(".", 1)[1]
            if name == "Pressure":
                # The one place SI arrives, and the only state array with a
                # pressure dimension.
                run.states[name] = values / PA_PER_PSI
                pressures += 1
            else:
                # Saturations are dimensionless; Rs is left as the simulator
                # gave it, because its unit was not established by
                # measurement here and a guessed conversion is worse than a
                # labelled raw number.
                run.states[name] = values
            continue

        converted = values
        if is_cumulative(key):
            converted = values / CUMULATIVE_SECOND_FACTOR
            cumulatives += 1

        if key.startswith("FIELD."):
            run.field_curves[key.split(".", 1)[1]] = converted
        elif key.startswith("WELL."):
            _, well, mnemonic = key.split(".", 2)
            run.well_curves.setdefault(well, {})[mnemonic] = converted

    if cumulatives:
        run.corrections.append(
            "%d cumulative curves divided by %.0f: the simulator integrates "
            "a per-day rate against elapsed seconds, so its totals arrive "
            "%.0fx too large. Measured against the integral of the rates, "
            "not assumed." % (cumulatives, CUMULATIVE_SECOND_FACTOR,
                              CUMULATIVE_SECOND_FACTOR))
    if pressures:
        run.corrections.append(
            "per-cell pressure converted from pascals to psia; the summary "
            "pressures were already psia and were left alone")
    run.corrections.append(
        "Rs and saturations are passed through unconverted -- saturations "
        "are dimensionless, and Rs's unit was not established by "
        "measurement, so it is labelled rather than guessed at")
    return run


def simulate(deck, out_dir, python=None, timeout=3600) -> ReservoirRun:
    """Run a deck in a subprocess and read the result.

    A subprocess rather than an import, and that is not caution: on the
    first run here SPE9 solved correctly and the process then died with
    `ConcurrencyViolationError("Vector can not be resized concurrently")`
    inside PythonCall's deallocator -- Julia's garbage collector finalising
    Python objects during interpreter teardown. It is a fatal Julia error,
    not a Python exception, so nothing in Python catches it.

    A worker that writes its answer and exits is immune: the results are
    already on disk when teardown happens, and the process was going to
    die anyway. Anything long-lived that imports `juliacall` is not immune,
    which is why no other module in this repository imports it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "run.npz"
    meta = out_dir / "run.json"

    command = [python or sys.executable, "-m",
               "phasespace.reservoir._jutul_worker",
               str(deck), str(npz), str(meta)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=timeout,
                               cwd=str(Path(__file__).resolve().parents[2]))

    if not npz.exists():
        detail = ""
        if meta.exists():
            try:
                with open(meta) as handle:
                    detail = json.load(handle).get("error", "")
            except ValueError:
                detail = ""
        raise RuntimeError(
            "the reservoir worker produced no output.\n%s\n%s"
            % (detail or completed.stderr[-2000:], completed.stdout[-500:]))

    return load(npz, meta)
