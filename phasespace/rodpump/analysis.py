"""Measurements read off a downhole card.

A card is not just a picture -- it is a production measurement. Fillage
converts directly into barrels, and the load levels say which valve is
holding and which is not. These are the numbers an operator acts on, and they
are computed here deterministically so a diagnosis can be explained rather
than merely asserted.

Everything is measured on the *downhole* card. A surface card carries the rod
string's own dynamics on top of the pump's behaviour, so measuring fillage
from one is measuring the wrong thing -- which is precisely why the wave
equation has to come first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synthetic import PumpSpec
from .wave import Card


@dataclass
class CardMeasurements:
    """What the card says about the pump."""

    fluid_load_lb: float
    upper_load_lb: float
    lower_load_lb: float
    net_stroke_in: float
    fillage: float
    displacement_bbl_per_day: float
    upstroke_droop_fraction: float
    downstroke_lift_fraction: float
    transfer_sharpness: float
    pickup_spread: float
    peak_compression_lb: float
    area_lb_in: float

    def as_dict(self) -> dict:
        return {
            "fluid_load_lb": self.fluid_load_lb,
            "upper_load_lb": self.upper_load_lb,
            "lower_load_lb": self.lower_load_lb,
            "net_stroke_in": self.net_stroke_in,
            "fillage": self.fillage,
            "displacement_bbl_per_day": self.displacement_bbl_per_day,
            "upstroke_droop_fraction": self.upstroke_droop_fraction,
            "downstroke_lift_fraction": self.downstroke_lift_fraction,
            "transfer_sharpness": self.transfer_sharpness,
            "pickup_spread": self.pickup_spread,
            "peak_compression_lb": self.peak_compression_lb,
            "area_lb_in": self.area_lb_in,
        }


def _split_strokes(card: Card) -> tuple:
    """Indices of the upstroke and the downstroke.

    Split on where the plunger actually turns around rather than at the
    halfway sample: a real pumping unit is not symmetric in time, and
    assuming it is puts part of the downstroke into the upstroke's
    measurements.
    """
    top = int(np.argmax(card.position_in))
    bottom = int(np.argmin(card.position_in))
    n = card.n_samples
    if top == bottom:
        return np.arange(n // 2), np.arange(n // 2, n)
    up = np.arange(bottom, top) if bottom < top else np.concatenate(
        [np.arange(bottom, n), np.arange(0, top)])
    down = np.arange(top, bottom) if top < bottom else np.concatenate(
        [np.arange(top, n), np.arange(0, bottom)])
    return up, down


def measure(card: Card, spec: PumpSpec) -> CardMeasurements:
    """Read the standard quantities off a downhole card."""
    load = card.load_lb
    position = card.position_in

    # Percentiles rather than min/max: a single noisy sample or a tagging
    # spike would otherwise set the whole load scale and corrupt every
    # measurement derived from it.
    upper = float(np.percentile(load, 90))
    lower = float(np.percentile(load, 10))
    fluid_load = max(upper - lower, 1e-9)
    net_stroke = max(float(position.max() - position.min()), 1e-9)

    up_idx, down_idx = _split_strokes(card)

    # Fillage: coming down, the plunger falls through gas or vapour until it
    # reaches liquid and the traveling valve can open. The position at which
    # the load transfers is the height of liquid in the barrel.
    threshold = lower + 0.5 * fluid_load
    down_load = load[down_idx]
    down_position = position[down_idx]
    below = np.nonzero(down_load < threshold)[0]
    if below.size:
        transfer_position = float(down_position[below[0]])
        fillage = (transfer_position - float(position.min())) / net_stroke
    else:
        # The load never sets down: the pump is not transferring at all.
        fillage = 1.0
    fillage = float(min(max(fillage, 0.0), 1.0))

    # A traveling valve that leaks bleeds load away as the plunger lifts, so
    # the top of the card slopes down across the upstroke.
    up_load = load[up_idx]
    if up_load.size >= 8:
        carrying = up_load[up_load > lower + 0.5 * fluid_load]
        if carrying.size >= 4:
            first = float(np.mean(carrying[: max(carrying.size // 4, 1)]))
            last = float(np.mean(carrying[-max(carrying.size // 4, 1):]))
            droop = (first - last) / fluid_load
        else:
            droop = 0.0
    else:
        droop = 0.0

    # A standing valve that leaks lets the barrel lose fluid on the way down,
    # so the rods never fully set the load down and the bottom line lifts.
    if down_load.size >= 8:
        resting = down_load[down_load < lower + 0.5 * fluid_load]
        if resting.size >= 4:
            early = float(np.mean(resting[: max(resting.size // 4, 1)]))
            late = float(np.mean(resting[-max(resting.size // 4, 1):]))
            lift = (late - early) / fluid_load
        else:
            lift = 0.0
    else:
        lift = 0.0

    # How abruptly the load transfers on the downstroke. This is the one
    # measurement that separates fluid pound from gas interference: liquid is
    # incompressible so the transfer is sudden, gas compresses so it is
    # gradual. The two look alike otherwise and call for different remedies.
    transfer_sharpness = _transfer_sharpness(down_position, down_load, fluid_load, net_stroke)

    # How much of the stroke the plunger travels while picking the load up.
    # On an anchored string the transfer is nearly vertical on the card. When
    # the tubing is free it stretches as it takes the load and recoils as it
    # sheds it, so both transfers lean and the rectangle shears into a
    # parallelogram -- which is the only signature that separates unanchored
    # tubing from a pump fault.
    pickup_spread = _pickup_spread(position[up_idx], load[up_idx], lower,
                                   fluid_load, net_stroke)

    return CardMeasurements(
        fluid_load_lb=fluid_load,
        upper_load_lb=upper,
        lower_load_lb=lower,
        net_stroke_in=net_stroke,
        fillage=fillage,
        displacement_bbl_per_day=spec.displacement_bbl_per_day(fillage),
        upstroke_droop_fraction=float(droop),
        downstroke_lift_fraction=float(lift),
        transfer_sharpness=float(transfer_sharpness),
        pickup_spread=float(pickup_spread),
        peak_compression_lb=float(min(load.min() - lower, 0.0)),
        area_lb_in=float(abs(card.area_lb_in())),
    )


def _pickup_spread(position: np.ndarray, load: np.ndarray, lower: float,
                   fluid_load: float, net_stroke: float) -> float:
    """Stroke fraction covered by the *rising* load transfer, 10% to 90% of Fo.

    Measured as the first crossing of each level, not the extent of the band.
    Taking the band's full width instead conflates this with any card whose
    load droops: a leaking traveling valve spends most of the upstroke passing
    back down through that band, and scored 0.999 -- indistinguishable from
    unanchored tubing, which is the one thing this is supposed to isolate.
    """
    if position.size < 8:
        return 0.0
    low = lower + 0.1 * fluid_load
    high = lower + 0.9 * fluid_load

    above_low = np.nonzero(load >= low)[0]
    if above_low.size == 0:
        return 0.0
    start = int(above_low[0])

    above_high = np.nonzero(load[start:] >= high)[0]
    if above_high.size == 0:
        # The load never reaches the top of the band on this stroke, so there
        # is no clean transfer to measure.
        return 0.0
    end = start + int(above_high[0])

    span = abs(float(position[end] - position[start]))
    return span / net_stroke


def _first_sustained(mask: np.ndarray, samples: int) -> int:
    """First index where `mask` holds for `samples` consecutive points.

    A single-sample crossing test is what a threshold measurement gets wrong
    on real data: one noise spike below the level starts the transfer early
    and the measured span balloons. Requiring the crossing to persist costs
    nothing on a clean card and is the difference between a measurement that
    survives field noise and one that inverts under it.
    """
    if mask.size < samples:
        return -1
    run = np.convolve(mask.astype(int), np.ones(samples, dtype=int), mode="valid")
    hit = np.nonzero(run == samples)[0]
    return int(hit[0]) if hit.size else -1


def _transfer_sharpness(position: np.ndarray, load: np.ndarray,
                        fluid_load: float, net_stroke: float,
                        persistence: int = 3) -> float:
    """Steepness of the downstroke load transfer, in fluid loads per stroke.

    Large means the load drops over a short distance (liquid, which does not
    compress); small means it slides away gradually (gas compressing). This is
    the measurement that separates fluid pound from gas interference, and the
    remedies for those differ completely -- so it has to hold up on a card with
    noise on it, not only on a synthetic one.

    Measured as a *span*: the stroke fraction covered while the load falls from
    80% to 20% of the fluid load. The obvious alternative -- a high percentile
    of |dL/dx| -- differentiates the signal and so amplifies noise directly
    into the answer. It inverts at 4% rms noise: gas scored 18.1 against fluid
    pound's 15.5, which is a confident wrong call on the most valuable
    distinction a card offers. The span form is integral rather than
    differential and holds separation to 9% rms noise.
    """
    if position.size < 8:
        return 0.0
    lower = float(np.percentile(load, 10))
    high = lower + 0.8 * fluid_load
    low = lower + 0.2 * fluid_load

    start = _first_sustained(load <= high, persistence)
    if start < 0:
        return 0.0
    end = _first_sustained(load[start:] <= low, persistence)
    if end < 0:
        # The load never sets down on this stroke: no transfer to measure.
        return 0.0

    span = abs(float(position[start + end] - position[start])) / net_stroke
    # 0.6 is the load fraction crossed; dividing by the stroke fraction it
    # took gives fluid loads per stroke, the same units as before.
    return 0.6 / max(span, 1e-3)


def shape_features(card: Card, points: int = 32) -> np.ndarray:
    """A scale-free description of the card's shape, for a classifier.

    Position and load are each normalised to [0, 1] before sampling, so the
    features describe the *shape* and not the well: a deep well with a big
    plunger and a shallow one with a small plunger produce the same feature
    vector for the same fault. Without that, a classifier learns which well it
    is looking at rather than what is wrong with it.
    """
    position = card.position_in
    load = card.load_lb
    span = max(float(position.max() - position.min()), 1e-9)
    load_span = max(float(np.percentile(load, 98) - np.percentile(load, 2)), 1e-9)

    x = (position - position.min()) / span
    y = (load - np.percentile(load, 2)) / load_span

    # Resample on a common stroke-position grid, once for the upstroke and
    # once for the downstroke, so the two halves stay distinguishable -- the
    # difference between them is where most faults live.
    up_idx, down_idx = _split_strokes(card)
    grid = np.linspace(0.0, 1.0, points)

    def resample(idx):
        xi, yi = x[idx], y[idx]
        order = np.argsort(xi)
        return np.interp(grid, xi[order], yi[order])

    return np.concatenate([resample(up_idx), resample(down_idx)])
