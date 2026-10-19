"""Physics-generated dynamometer cards with known faults.

This is what makes a rod-pump classifier possible without owning a fleet of
wells. A fault has a characteristic shape *at the pump*, and `wave.py` runs in
both directions -- so a fault can be described exactly as a downhole card and
then propagated up to the surface card a dynamometer would actually record.
The label is known by construction, which is the same strategy the gas
surrogate uses (PRD 5.2.1: physics generates the training data).

The card is traversed in physical order: bottom of stroke, up (the plunger
lifts fluid), top, down, back to bottom. Position is inches from the bottom of
the plunger stroke; load is pounds force on the plunger.

An ideal full pump traces a rectangle. Every fault below is a specific,
recognisable distortion of that rectangle, and the distortion is what an
operator is really reading when they look at a card:

  full_pump            the rectangle
  fluid_pound          barrel not filled; load transfer is late on the
                       downstroke and abrupt when the plunger hits liquid
  gas_interference     same late transfer, but gas compresses so the corner
                       is rounded rather than sharp -- this is the pair that
                       matters most, because the remedy differs completely
  traveling_valve_leak fluid slips back past the plunger, so load bleeds off
                       during the upstroke and the top line droops
  standing_valve_leak  the barrel loses fluid on the downstroke, so the
                       bottom line lifts
  tubing_movement      unanchored tubing stretches and recoils with the load,
                       shearing the rectangle into a parallelogram
  pump_tagging         the plunger strikes the bottom, giving a load spike
                       against an otherwise normal card
  worn_pump            a general loss of load and rounding of corners
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .wave import Card

# Fluid pressure gradient of fresh water, psi per foot. Multiplied by specific
# gravity it gives the gradient of whatever the well is lifting.
WATER_GRADIENT_PSI_FT = 0.433

FULL_PUMP = "full_pump"
FLUID_POUND = "fluid_pound"
GAS_INTERFERENCE = "gas_interference"
TRAVELING_VALVE_LEAK = "traveling_valve_leak"
STANDING_VALVE_LEAK = "standing_valve_leak"
TUBING_MOVEMENT = "tubing_movement"
PUMP_TAGGING = "pump_tagging"
WORN_PUMP = "worn_pump"

FAULTS = (
    FULL_PUMP,
    FLUID_POUND,
    GAS_INTERFERENCE,
    TRAVELING_VALVE_LEAK,
    STANDING_VALVE_LEAK,
    TUBING_MOVEMENT,
    PUMP_TAGGING,
    WORN_PUMP,
)

FAULT_LABELS = {
    FULL_PUMP: "Full pump",
    FLUID_POUND: "Fluid pound (incomplete fillage)",
    GAS_INTERFERENCE: "Gas interference",
    TRAVELING_VALVE_LEAK: "Traveling valve / plunger leak",
    STANDING_VALVE_LEAK: "Standing valve leak",
    TUBING_MOVEMENT: "Unanchored tubing movement",
    PUMP_TAGGING: "Pump tagging bottom",
    WORN_PUMP: "Worn pump",
}


@dataclass
class PumpSpec:
    """The pump and the fluid column it lifts."""

    plunger_diameter_in: float = 1.5
    net_lift_ft: float = 4000.0
    fluid_specific_gravity: float = 0.9
    stroke_in: float = 100.0
    strokes_per_minute: float = 8.0

    def __post_init__(self):
        if self.plunger_diameter_in <= 0.0:
            raise ValueError("plunger diameter must be positive, got %r"
                             % (self.plunger_diameter_in,))
        if self.stroke_in <= 0.0:
            raise ValueError("stroke must be positive, got %r" % (self.stroke_in,))
        if self.strokes_per_minute <= 0.0:
            raise ValueError("SPM must be positive, got %r" % (self.strokes_per_minute,))

    @property
    def plunger_area_in2(self) -> float:
        return math.pi * self.plunger_diameter_in ** 2 / 4.0

    @property
    def fluid_load_lb(self) -> float:
        """Fo, the weight of the fluid column standing on the plunger.

        This single number sets the height of the card. It is the load the
        rods pick up when the traveling valve closes, and the load they set
        down when it opens.
        """
        return (WATER_GRADIENT_PSI_FT * self.fluid_specific_gravity
                * self.net_lift_ft * self.plunger_area_in2)

    @property
    def period_s(self) -> float:
        return 60.0 / self.strokes_per_minute

    def displacement_bbl_per_day(self, fillage: float = 1.0) -> float:
        """Theoretical pump displacement at a given fillage.

        The reason fillage is worth measuring at all: it converts directly
        into barrels, so a card is a production measurement and not merely a
        picture.
        """
        volume_in3 = self.plunger_area_in2 * self.stroke_in * fillage
        strokes_per_day = self.strokes_per_minute * 60.0 * 24.0
        return volume_in3 * strokes_per_day / 9702.0  # in^3 -> bbl


def _stroke_phase(samples: int) -> np.ndarray:
    """Fraction of the cycle, 0 at the bottom of the stroke."""
    return np.arange(samples) / samples


def _position(spec: PumpSpec, samples: int) -> np.ndarray:
    """Plunger position over one cycle: up on the first half, down on the second.

    A simple harmonic stroke rather than a full kinematic model of the
    pumping unit's geometry. The unit's crank/pitman linkage does distort the
    velocity profile, and that matters for a specific well, but the fault
    *shapes* -- which is what a diagnosis reads -- are carried by the load,
    not by the exact position profile.
    """
    phase = _stroke_phase(samples)
    return spec.stroke_in / 2.0 * (1.0 - np.cos(2.0 * np.pi * phase))


def _phase_at_downstroke_position(fillage: float) -> float:
    """Cycle phase at which the plunger, coming down, reaches `fillage` of the
    stroke above bottom.

    Fillage is a *position* fraction -- the share of the barrel holding liquid
    -- so the plunger falls through (1 - fillage) of the stroke before it
    reaches liquid and the traveling valve can open. Position is sinusoidal in
    phase, not linear, so the phase of that event is not simply the fillage:
    at 0.58 fillage the transfer happens at phase 0.79, and using one for the
    other mislabels the training data by a wide margin.
    """
    fillage = float(min(max(fillage, 0.0), 1.0))
    return 1.0 - math.acos(min(max(1.0 - 2.0 * fillage, -1.0), 1.0)) / (2.0 * math.pi)


def _transition(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Smooth 0->1 step. Real load transfers take finite time; a true step
    would put energy in every harmonic and make the propagated surface card
    ring with content no dynamometer ever recorded."""
    return 0.5 * (1.0 + np.tanh((x - centre) / max(width, 1e-6)))


def downhole_card(
    fault: str,
    spec: PumpSpec = None,
    samples: int = 256,
    severity: float = 0.5,
    rng: np.random.Generator = None,
) -> Card:
    """Build the pump card for one fault.

    `severity` runs 0 (barely present) to 1 (severe) and means something
    different per fault -- fillage fraction for fluid pound, leak rate for a
    valve leak, tubing stretch for unanchored tubing. That is deliberate:
    a single "severity" axis per fault is what lets a training set span the
    range an operator actually sees rather than only textbook extremes.
    """
    if fault not in FAULTS:
        raise ValueError("unknown fault %r (expected one of %s)"
                         % (fault, ", ".join(FAULTS)))
    spec = spec or PumpSpec()
    rng = rng or np.random.default_rng(0)
    severity = float(min(max(severity, 0.0), 1.0))

    phase = _stroke_phase(samples)
    position = _position(spec, samples)
    fo = spec.fluid_load_lb

    # Baseline: load picked up just after the bottom of the stroke, set down
    # just after the top. Transfers are fast but not instantaneous.
    pickup, setdown = 0.03, 0.53
    edge = 0.02
    load = fo * (_transition(phase, pickup, edge) - _transition(phase, setdown, edge))

    if fault == FULL_PUMP:
        pass

    elif fault == FLUID_POUND:
        # The barrel does not fill, so the traveling valve cannot open at the
        # top -- the rods carry fluid load part-way down until the plunger
        # reaches liquid, then transfer abruptly.
        fillage = 1.0 - 0.65 * severity
        load = fo * (_transition(phase, pickup, edge)
                     - _transition(phase, _phase_at_downstroke_position(fillage), 0.01))

    elif fault == GAS_INTERFERENCE:
        # Same late transfer, but gas in the barrel compresses instead of
        # transferring load at once, so the corner is rounded. Distinguishing
        # this from fluid pound is the single most valuable call on a card:
        # pound calls for pump-off control, gas calls for a different fix
        # entirely.
        fillage = 1.0 - 0.65 * severity
        load = fo * (_transition(phase, pickup, edge)
                     - _transition(phase, _phase_at_downstroke_position(fillage),
                                   0.05 + 0.10 * severity))

    elif fault == TRAVELING_VALVE_LEAK:
        # Fluid slips back past the plunger while it lifts, so the load bleeds
        # away through the upstroke and the top of the card droops.
        bleed = 0.5 * severity
        upstroke = np.clip((phase - pickup) / (setdown - pickup), 0.0, 1.0)
        load = load * (1.0 - bleed * upstroke)

    elif fault == STANDING_VALVE_LEAK:
        # The barrel loses fluid back to the formation on the downstroke, so
        # the rods never fully set the load down and the bottom line lifts.
        downstroke = np.clip((phase - setdown) / (1.0 - setdown), 0.0, 1.0)
        load = load + fo * 0.35 * severity * downstroke

    elif fault == TUBING_MOVEMENT:
        # Unanchored tubing stretches as it takes load and recoils as it
        # sheds it, so the vertical load-transfer lines lean and the
        # rectangle shears into a parallelogram.
        shear = 0.25 * severity
        load = fo * (_transition(phase, pickup + shear * 0.5, edge + shear)
                     - _transition(phase, setdown + shear * 0.5, edge + shear))

    elif fault == PUMP_TAGGING:
        # The plunger strikes the bottom of the barrel: a sharp compressive
        # spike at the bottom of the stroke on an otherwise normal card.
        spike = np.exp(-((phase - 0.98) / 0.012) ** 2)
        spike += np.exp(-((phase - (-0.02)) / 0.012) ** 2)
        load = load - fo * (0.8 + 0.7 * severity) * spike

    elif fault == WORN_PUMP:
        # General slippage: less load carried and every corner rounded.
        load = load * (1.0 - 0.45 * severity)
        load = fo * (1.0 - 0.45 * severity) * (
            _transition(phase, pickup, edge + 0.08 * severity)
            - _transition(phase, setdown, edge + 0.08 * severity))

    return Card(position_in=position, load_lb=load, period_s=spec.period_s)


def true_fillage(fault: str, severity: float) -> float:
    """The pump fillage the generator actually built, as a position fraction.

    Carried alongside the fault label because fillage is the number an
    operator acts on -- it converts straight into barrels -- so it is worth
    predicting as a quantity, not just classifying the fault that caused it.
    """
    severity = float(min(max(severity, 0.0), 1.0))
    if fault in (FLUID_POUND, GAS_INTERFERENCE):
        return 1.0 - 0.65 * severity
    if fault == WORN_PUMP:
        # A worn pump still fills; it simply fails to carry the full load.
        return 1.0
    return 1.0


def surface_card_for(
    fault: str,
    string,
    spec: PumpSpec = None,
    samples: int = 256,
    severity: float = 0.5,
    damping_per_s: float = 0.4,
    rng: np.random.Generator = None,
) -> tuple:
    """The surface card a dynamometer would record for this fault.

    Returns (surface, downhole). Both are kept because the downhole card is
    the label's ground truth -- a diagnosis is judged on whether it recovers
    the pump condition, not on whether it matches a surface shape.
    """
    from .wave import downhole_to_surface

    down = downhole_card(fault, spec=spec, samples=samples, severity=severity, rng=rng)
    surface = downhole_to_surface(down, string, damping_per_s=damping_per_s)
    return surface, down
