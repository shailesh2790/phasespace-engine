"""Does the Hammerschmidt hydrate curve find real hydrates in field data?

Petrobras 3W 2.0.0 labels real hydrate events on real subsea wells, and
carries the subsea-tree pressure and temperature alongside. That is enough to
ask the only question that matters about a correlation: applied to the
measurement, does it say what happened?

Nothing is trained, fitted or tuned here. The curve is T[degF] = 8.9 p^0.285,
published by Hammerschmidt in 1934, with no free parameters. That is what
makes this a validation rather than a benchmark -- there is no train/test
split to get wrong, no leakage to worry about, and an auditor can reproduce
any single row with a calculator.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import the engine from this checkout, wherever it was cloned.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phasespace.sim.thermal import (
    HAMMERSCHMIDT_MAX_PSIA,
    HAMMERSCHMIDT_MIN_PSIA,
    hydrate_temperature_r,
)

PA_PER_PSI = 6894.757
# Where fetch_3w.py put the dataset. `data/` is gitignored, so the
# corpus never lands in a commit.
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/3w")

# Subsea tree pressure and temperature. The tree is the cold, high-pressure
# point in the production path and the place a subsea well forms hydrates, so
# it is the physically right pair -- not the one that scores best.
PRESSURE = "P-TPT"
TEMPERATURE = "T-TPT"

NORMAL, PRODUCTION_LINE, SERVICE_LINE = 0, 8, 9
TRANSIENT_OFFSET = 100


def margin_degf(pressure_pa, temperature_c):
    """Degrees above the hydrate curve. Negative means hydrates form.

    Non-physical pressures are excluded rather than clipped. 8.9% of the
    P-TPT readings in this dataset are *negative* -- an instrument fault, not
    a reading -- and the engine refuses them outright, which is how they were
    found. Clipping them to a small positive number would have produced a
    hydrate temperature near zero and a spuriously comfortable margin on
    exactly the samples where the instrument had failed.
    """
    psia = np.asarray(pressure_pa, dtype=float) / PA_PER_PSI
    fahrenheit = np.asarray(temperature_c, dtype=float) * 9.0 / 5.0 + 32.0
    physical = psia > 0.0
    hydrate_f = np.full(psia.shape, np.nan)
    hydrate_f[physical] = np.array(
        [hydrate_temperature_r(float(p))[0] - 459.67 for p in psia[physical]])
    return fahrenheit - hydrate_f, psia, physical


def instances(cls):
    return sorted(glob.glob(str(ROOT / str(cls) / "*.parquet")))


def well_of(path):
    name = Path(path).stem
    return name.split("_")[0] if name.startswith("WELL") else name


def scan(classes):
    rows = []
    uninstrumented = {cls: 0 for cls in classes}
    available = {cls: len(instances(cls)) for cls in classes}
    for cls in classes:
        for path in instances(cls):
            frame = pd.read_parquet(path)
            if (PRESSURE not in frame or TEMPERATURE not in frame
                    or frame[PRESSURE].notna().sum() == 0
                    or frame[TEMPERATURE].notna().sum() == 0):
                # No tree pressure or temperature anywhere in the file. Not a
                # missed detection -- there is no instrument reading to apply
                # the correlation to -- but it has to appear in the
                # denominator or the result reads better than the data is.
                uninstrumented[cls] += 1
                continue
            frame = frame.dropna(subset=["class", PRESSURE, TEMPERATURE])
            if frame.empty:
                continue
            margins, psia, physical = margin_degf(frame[PRESSURE],
                                                  frame[TEMPERATURE])
            inside = (physical
                      & (psia >= HAMMERSCHMIDT_MIN_PSIA)
                      & (psia <= HAMMERSCHMIDT_MAX_PSIA))
            labels = frame["class"].to_numpy()
            rows.append({
                "file": Path(path).name,
                "well": well_of(path),
                "folder": cls,
                "margins": margins,
                "labels": labels,
                "inside": inside,
                "psia": psia,
                "physical": physical,
            })
    return rows, uninstrumented, available


def report(scanned):
    rows, uninstrumented, available = scanned
    counted = [r for r in rows if r["physical"].size]
    if not counted:
        # The common case here is running this before the download, so say
        # that rather than dividing by zero three lines further down.
        print("No instances found under %s." % ROOT)
        print("Download the corpus first:")
        print("    python validation/fetch_3w.py")
        print("or pass the directory holding the numbered class folders:")
        print("    python validation/hydrate_3w.py /path/to/3w/dataset")
        return
    print("Instances scanned: %d, from %d distinct wells"
          % (len(counted), len({r["well"] for r in counted})))
    total = sum(r["physical"].size for r in rows)
    faulty = sum((~r["physical"]).sum() for r in rows)
    usable = sum(r["inside"].sum() for r in rows)
    print("Observations: %d total, %d (%.1f%%) with a non-physical negative "
          "tree pressure" % (total, faulty, 100.0 * faulty / total))
    print("Usable inside Hammerschmidt's %.0f-%.0f psia range: %d (%.1f%%)"
          % (HAMMERSCHMIDT_MIN_PSIA, HAMMERSCHMIDT_MAX_PSIA, usable,
             100.0 * usable / total))
    print()

    buckets = {}
    for row in rows:
        for label in np.unique(row["labels"]):
            keep = (row["labels"] == label) & row["inside"]
            if keep.sum() == 0:
                continue
            buckets.setdefault(int(label), []).append(row["margins"][keep])

    print("%-28s %10s %9s %9s %9s %9s"
          % ("label", "n", "mean", "p5", "median", "p95"))
    names = {0: "normal", 8: "hydrate, production line",
             9: "hydrate, service line", 108: "  -> its transient",
             109: "  -> its transient"}
    for label in sorted(buckets):
        values = np.concatenate(buckets[label])
        print("%-28s %10d %9.1f %9.1f %9.1f %9.1f"
              % (names.get(label, str(label)), len(values), values.mean(),
                 np.percentile(values, 5), np.median(values),
                 np.percentile(values, 95)))
    print()

    # Per-instance detection, with missing data separated from missed
    # detection. Conflating the two is the easy mistake and it understates
    # the result in a way that is still dishonest: five of the fourteen real
    # production-line instances carry no tree pressure or temperature at all,
    # and a correlation cannot be charged with failing to read an instrument
    # that is not there.
    print("%-30s %9s %9s %9s %9s %9s"
          % ("", "detected", "usable", "no range", "no sensor", "in 3W"))
    for cls, title in ((NORMAL, "normal operation"),
                       (PRODUCTION_LINE, "hydrate, production line"),
                       (SERVICE_LINE, "hydrate, service line")):
        subset = [r for r in rows if r["folder"] == cls]
        if not subset:
            continue
        detected = usable = 0
        for row in subset:
            if cls == NORMAL:
                window = row["inside"]
            else:
                window = row["inside"] & np.isin(
                    row["labels"], [cls, cls + TRANSIENT_OFFSET])
            if not window.any():
                continue
            usable += 1
            detected += bool((row["margins"][window] < 0.0).any())
        out_of_range = len(subset) - usable
        blind = uninstrumented.get(cls, 0)
        rate = ("%.0f%%" % (100.0 * detected / usable)) if usable else "--"
        print("%-30s %9d %9d %9d %9d %9d   %s"
              % (title, detected, usable, out_of_range, blind,
                 available.get(cls, 0), rate))

    print()
    print("A normal instance counts as detected only if the curve goes "
          "negative anywhere in it -- i.e. a false alarm.")


if __name__ == "__main__":
    report(scan([NORMAL, PRODUCTION_LINE, SERVICE_LINE]))
