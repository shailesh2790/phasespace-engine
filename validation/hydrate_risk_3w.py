"""Can a learned layer beat the hydrate curve at predicting, not detecting?

Protocol, fixed before any number was looked at:

* **Train on simulated instances only.** Petrobras ships them for this; the
  real production-line events with usable instrumentation are too few to
  train on and are kept untouched as a test set.
* **Test within each real instance.** Two facts about this dataset force
  that choice. Normal operation comes from wells 1-8 and the hydrate events
  from wells 19-32, almost disjoint, so a model scored across those groups
  could be recognising *which well it is* rather than whether a hydrate is
  coming. And every simulated instance places onset at exactly 1800 s, so a
  thirty-minute question has no negatives in the training set at all.

  The within-instance test removes the first confound entirely: each real
  event instance carries hours of its own normal operation before the event,
  so the comparison is a well against itself. False alarms are counted
  separately on the normal instances.
* **Score against a *tuned* physics baseline.** The margin threshold is swept
  and the best one reported, because beating a strawman is the single most
  common way this literature overstates itself (McGreivy & Hakim, *Nature
  Machine Intelligence* 2024: 79% of ML-for-fluid-PDE papers compared against
  a weak baseline).
* **Prediction, not detection.** Samples at and after the labelled onset are
  excluded entirely. The physics already detects a hydrate in progress; doing
  that again is not the question.

A model that does not beat the tuned baseline does not ship. That outcome is
a result, not a failure.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phasespace.sim.hydrate_risk import (
    DEFAULT_HORIZON_S,
    FEATURES,
    Score,
    fit,
    label_from_trace,
    matrix_from_trace,
)

PA_PER_PSI = 6894.757
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
PRESSURE, TEMPERATURE = "P-TPT", "T-TPT"

# Training horizon. Shorter than the question being asked, because every
# simulated instance puts onset at exactly 1800 s: at a 1800 s horizon the
# whole prefix is positive and there is nothing to discriminate against. At
# 900 s the model learns to separate the final quarter-hour from the
# quarter-hour before it, which is a harder problem than separating an event
# from unrelated normal operation and does not depend on the well.
TRAIN_HORIZON_S = 900

# One sample per this many seconds. The traces are 1 Hz and consecutive
# seconds are nearly identical, so thinning costs no information and makes
# the fit tractable; the trend features are computed before thinning, on the
# full-rate trace, so no dynamics are lost.
STRIDE = 30


def load(path, horizon_s=DEFAULT_HORIZON_S):
    frame = pd.read_parquet(path)
    if PRESSURE not in frame or TEMPERATURE not in frame:
        return None
    frame = frame.dropna(subset=["class"])
    if frame.empty or frame[PRESSURE].notna().sum() == 0:
        return None
    pressure = frame[PRESSURE].to_numpy(dtype=float) / PA_PER_PSI
    temperature = frame[TEMPERATURE].to_numpy(dtype=float) * 9.0 / 5.0 + 32.0
    return {
        "name": Path(path).stem,
        "well": Path(path).stem.split("_")[0],
        "matrix": matrix_from_trace(pressure, temperature),
        "target": label_from_trace(frame["class"].to_numpy(),
                                   horizon_s=horizon_s),
        "has_event": bool(np.isin(frame["class"].to_numpy(), [8, 108]).any()),
    }


def gather(pattern, horizon_s=DEFAULT_HORIZON_S):
    out = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        record = load(path, horizon_s)
        if record is not None:
            out.append(record)
    return out


def score_instances(records, risk_of, threshold, name):
    """Walk each instance once and count events caught and alarms raised.

    For an event instance the alarm must fire inside the pre-onset window to
    count, and the lead time is measured from the first firing to onset.
    Everything at or after onset is excluded, so an alarm that only fires
    once the hydrate is already forming scores nothing -- the physics
    already does that.
    """
    score = Score(name=name)
    for record in records:
        risk = risk_of(record["matrix"])
        target = record["target"]
        usable = np.isfinite(risk)

        if record["has_event"]:
            score.events += 1
            window = usable & (target == 1.0)
            if window.any():
                fired = np.where(window & (risk >= threshold))[0]
                if fired.size:
                    score.detected += 1
                    onset = int(np.argmax(np.isnan(target)))
                    score.lead_minutes.append((onset - fired[0]) / 60.0)
            # Within-instance false alarm: did it also fire during this same
            # well's own earlier normal hours? A detector that is simply on
            # all the time catches every event and is worth nothing.
            quiet = usable & (target == 0.0)
            if quiet.any() and (risk[quiet] >= threshold).any():
                score.noisy_events += 1
        else:
            score.normal_instances += 1
            if (usable & (risk >= threshold)).any():
                score.false_alarm_instances += 1
    return score


def physics_risk(threshold_f):
    """Alarm when the margin falls below a threshold. The thing we ship now."""
    index = FEATURES.index("margin_f")

    def risk_of(matrix):
        margin = matrix[:, index]
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(margin),
                            (margin < threshold_f).astype(float), np.nan)
    return risk_of


def main():
    simulated = gather("sim8/*.parquet", TRAIN_HORIZON_S)
    real_events = [r for r in gather("8/*.parquet") if r["has_event"]]
    normal = gather("0/*.parquet")

    print("training on %d simulated instances" % len(simulated))
    print("testing on %d real events and %d normal instances, %d wells"
          % (len(real_events), len(normal),
             len({r["well"] for r in real_events + normal})))
    print("horizon %d min, samples strided 1:%d"
          % (DEFAULT_HORIZON_S // 60, STRIDE))
    print()

    if not simulated:
        raise SystemExit("no simulated instances found -- run fetch_3w.py")

    x = np.vstack([r["matrix"][::STRIDE] for r in simulated])
    y = np.concatenate([r["target"][::STRIDE] for r in simulated])
    usable = np.isfinite(y) & np.isfinite(x).all(axis=1)
    print("training rows %d (%.1f%% positive)"
          % (usable.sum(), 100.0 * np.nanmean(y[usable])))

    model = fit(x, y)
    print()
    print("fitted coefficients, standardised (sign and size are readable):")
    order = np.argsort(-np.abs(model.weights))
    for index in order:
        print("   %-22s %+8.3f" % (FEATURES[index], model.weights[index]))
    print()

    # Tuned physics baseline: sweep the threshold rather than assuming zero.
    print("%s" % "physics baseline, threshold swept".upper())
    best = None
    for threshold_f in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0):
        score = score_instances(real_events + normal, physics_risk(threshold_f),
                                0.5, "margin < %+.0f F" % threshold_f)
        print("   " + score.line())
        quality = (score.recall, -score.false_alarm_rate)
        if best is None or quality > best[0]:
            best = (quality, score)
    print()

    print("LEARNED LAYER, bounded against the physics")
    for threshold in (0.5, 0.6, 0.7, 0.8):
        def risk_of(matrix, threshold=threshold):
            bounded, _capped = model.predict(matrix)
            return np.where(np.isfinite(matrix).all(axis=1), bounded, np.nan)
        score = score_instances(real_events + normal, risk_of, threshold,
                                "risk >= %.2f" % threshold)
        print("   " + score.line())
    print()
    print("Best physics baseline: %s" % best[1].line())


if __name__ == "__main__":
    main()
