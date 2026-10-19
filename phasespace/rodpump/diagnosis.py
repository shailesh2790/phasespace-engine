"""What is wrong with the pump.

Two layers, and the order matters.

The **rule layer** encodes what an operator reads off a card: a drooping top
line is a traveling valve passing fluid, a lifting bottom line is a standing
valve passing fluid, a late abrupt transfer is fluid pound, a late gradual one
is gas. It is deterministic and it can say *why*, which is what makes a
recommendation actionable rather than an oracle's verdict.

The **learned layer** sits on top for the cases rules handle badly: faults in
combination, mild severities near a threshold, and real cards with noise and
unit dynamics the rules were never written for. It is trained on
physics-generated cards, so the labels are exact.

The rule layer is not a fallback for when the model is unavailable. It is the
explanation, and it runs always. A diagnosis that cannot be checked against
the measurement that produced it is not usable in a field where being
confidently wrong costs a pulling unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import CardMeasurements, measure, shape_features
from .synthetic import (
    FAULT_LABELS,
    FLUID_POUND,
    FULL_PUMP,
    GAS_INTERFERENCE,
    PUMP_TAGGING,
    STANDING_VALVE_LEAK,
    TRAVELING_VALVE_LEAK,
    TUBING_MOVEMENT,
    WORN_PUMP,
    PumpSpec,
)
from .wave import Card

# A pump carrying less than this share of the fluid load it should be carrying
# is not lifting what the well can give it.
WORN_LOAD_RATIO = 0.80
# Below this fillage the pump is not filling, whatever the cause.
INCOMPLETE_FILLAGE = 0.92
# Fluid transfers in a stroke-fraction; gas slides away over a large part of
# one. This is the threshold that separates the two, in fluid loads per stroke.
SHARP_TRANSFER = 6.0
# On an anchored string the load transfer is nearly vertical on the card.
# Above this share of the stroke, the tubing is moving with the load.
SHEARED_PICKUP = 0.25


@dataclass
class Finding:
    fault: str
    confidence: float
    evidence: str

    @property
    def label(self) -> str:
        return FAULT_LABELS.get(self.fault, self.fault)


@dataclass
class Diagnosis:
    findings: list = field(default_factory=list)
    measurements: CardMeasurements = None
    expected_fluid_load_lb: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def primary(self) -> Finding:
        return self.findings[0] if self.findings else Finding(FULL_PUMP, 0.0, "no card")

    @property
    def load_ratio(self) -> float:
        """Measured fluid load against what the well should produce.

        The measurement that exposes a worn pump. Everything else on a card is
        read relative to the load the card itself shows, which hides a pump
        that is simply not carrying what it should -- the shape can look
        perfectly normal at 60% of the expected load.
        """
        if self.expected_fluid_load_lb <= 0.0:
            return 1.0
        return self.measurements.fluid_load_lb / self.expected_fluid_load_lb


def diagnose(card: Card, spec: PumpSpec) -> Diagnosis:
    """Rule-based diagnosis of a downhole card, with its evidence."""
    m = measure(card, spec)
    expected = spec.fluid_load_lb
    result = Diagnosis(measurements=m, expected_fluid_load_lb=expected)
    ratio = result.load_ratio
    findings = []

    if m.peak_compression_lb < -0.5 * m.fluid_load_lb:
        findings.append(Finding(
            PUMP_TAGGING,
            min(abs(m.peak_compression_lb) / m.fluid_load_lb, 1.0),
            "compressive spike of %.0f lb against a %.0f lb fluid load -- the "
            "plunger is striking bottom" % (m.peak_compression_lb, m.fluid_load_lb)))

    if m.fillage < INCOMPLETE_FILLAGE:
        shortfall = (INCOMPLETE_FILLAGE - m.fillage) / INCOMPLETE_FILLAGE
        if m.transfer_sharpness >= SHARP_TRANSFER:
            findings.append(Finding(
                FLUID_POUND, min(shortfall * 1.6, 1.0),
                "fillage %.0f%%, and the load transfers abruptly (%.1f fluid "
                "loads per stroke) -- liquid does not compress, so the plunger "
                "is striking a liquid level part-way down"
                % (100 * m.fillage, m.transfer_sharpness)))
        else:
            findings.append(Finding(
                GAS_INTERFERENCE, min(shortfall * 1.6, 1.0),
                "fillage %.0f%%, but the load slides away gradually (%.1f fluid "
                "loads per stroke) -- gas compressing rather than a liquid "
                "impact" % (100 * m.fillage, m.transfer_sharpness)))

    if m.pickup_spread > SHEARED_PICKUP:
        findings.append(Finding(
            TUBING_MOVEMENT, min((m.pickup_spread - SHEARED_PICKUP) * 3.0, 1.0),
            "the load takes %.0f%% of the stroke to transfer -- the tubing is "
            "stretching and recoiling with it, which shears the card into a "
            "parallelogram" % (100 * m.pickup_spread)))

    if m.upstroke_droop_fraction > 0.10:
        findings.append(Finding(
            TRAVELING_VALVE_LEAK, min(m.upstroke_droop_fraction * 3.0, 1.0),
            "load bleeds off %.0f%% across the upstroke -- fluid is slipping "
            "back past the plunger or traveling valve"
            % (100 * m.upstroke_droop_fraction)))

    if m.downstroke_lift_fraction > 0.08:
        findings.append(Finding(
            STANDING_VALVE_LEAK, min(m.downstroke_lift_fraction * 4.0, 1.0),
            "the load does not set fully down -- it lifts %.0f%% across the "
            "downstroke, so the barrel is losing fluid back through the "
            "standing valve" % (100 * m.downstroke_lift_fraction)))

    if ratio < WORN_LOAD_RATIO:
        findings.append(Finding(
            WORN_PUMP, min((WORN_LOAD_RATIO - ratio) * 3.0, 1.0),
            "carrying %.0f%% of the %.0f lb fluid load this well should "
            "produce -- slippage past a worn plunger or barrel"
            % (100 * ratio, expected)))

    if not findings:
        findings.append(Finding(
            FULL_PUMP, 1.0,
            "fillage %.0f%%, load %.0f lb against an expected %.0f lb, valve "
            "lines flat -- the pump is filling and both valves are holding"
            % (100 * m.fillage, m.fluid_load_lb, expected)))

    findings.sort(key=lambda f: -f.confidence)
    result.findings = findings

    if m.fillage < INCOMPLETE_FILLAGE:
        result.notes.append(
            "Pump is displacing about %.0f bbl/day against %.0f bbl/day at full "
            "fillage." % (m.displacement_bbl_per_day,
                          spec.displacement_bbl_per_day(1.0)))
    return result


def feature_vector(card: Card, spec: PumpSpec) -> np.ndarray:
    """Features for the learned layer: card shape plus the scalar measurements.

    The shape half is normalised and so says nothing about the well; the
    scalar half deliberately does, because the load a pump *should* carry is
    exactly the context a shape cannot supply. A worn pump traces a perfectly
    normal rectangle at 60% of the load it ought to.
    """
    m = measure(card, spec)
    expected = max(spec.fluid_load_lb, 1e-9)
    scalars = np.array([
        m.fillage,
        m.upstroke_droop_fraction,
        m.downstroke_lift_fraction,
        np.tanh(m.transfer_sharpness / 20.0),
        m.peak_compression_lb / expected,
        m.fluid_load_lb / expected,
        m.area_lb_in / (expected * max(m.net_stroke_in, 1e-9)),
    ])
    return np.concatenate([shape_features(card), scalars])


FEATURE_NAMES = (
    ["shape_up_%02d" % i for i in range(32)]
    + ["shape_down_%02d" % i for i in range(32)]
    + ["fillage", "upstroke_droop", "downstroke_lift", "transfer_sharpness",
       "peak_compression_ratio", "load_ratio", "normalised_area"]
)
