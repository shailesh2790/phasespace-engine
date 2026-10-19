"""Tests for synthetic cards, card measurements, and diagnosis.

The important test here is the end-to-end one: a fault is described at the
pump, propagated *up* to the surface card a dynamometer would record, then
propagated back down and diagnosed. The diagnosis never sees the card it was
built from, so it has to survive a full round trip through the wave equation
-- which is exactly what happens on a real well.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.rodpump.analysis import measure, shape_features
from phasespace.rodpump.diagnosis import diagnose, feature_vector, FEATURE_NAMES
from phasespace.rodpump.rodstring import taper_86
from phasespace.rodpump.synthetic import (
    FAULTS,
    FLUID_POUND,
    FULL_PUMP,
    GAS_INTERFERENCE,
    PUMP_TAGGING,
    STANDING_VALVE_LEAK,
    TRAVELING_VALVE_LEAK,
    TUBING_MOVEMENT,
    WORN_PUMP,
    PumpSpec,
    downhole_card,
    surface_card_for,
    true_fillage,
)
from phasespace.rodpump.wave import surface_to_downhole

SPEC = PumpSpec(plunger_diameter_in=1.5, net_lift_ft=4000.0,
                fluid_specific_gravity=0.9, stroke_in=100.0, strokes_per_minute=8.0)
STRING = taper_86(4500.0)
DAMPING = 0.4


def round_tripped(fault, severity, spec=SPEC):
    """The card as a well would actually deliver it: fault -> surface -> back."""
    surface, _ = surface_card_for(fault, STRING, spec, severity=severity,
                                  damping_per_s=DAMPING)
    return surface_to_downhole(surface, STRING, damping_per_s=DAMPING)


# ------------------------------------------------------------ pump spec ----


def test_fluid_load_matches_the_hand_calc():
    """Fo = 0.433 * SG * lift * plunger area. This one number sets the height
    of every card, so it is worth checking against arithmetic."""
    expected = 0.433 * 0.9 * 4000.0 * (np.pi * 1.5 ** 2 / 4.0)
    assert SPEC.fluid_load_lb == pytest.approx(expected, rel=1e-9)


def test_displacement_is_proportional_to_fillage():
    full = SPEC.displacement_bbl_per_day(1.0)
    assert SPEC.displacement_bbl_per_day(0.5) == pytest.approx(full / 2.0)
    assert 150.0 < full < 300.0  # a 1.5 in plunger on a 100 in stroke at 8 SPM


def test_pump_spec_rejects_nonsense():
    for kwargs in ({"plunger_diameter_in": 0.0}, {"stroke_in": 0.0},
                   {"strokes_per_minute": 0.0}):
        with pytest.raises(ValueError):
            PumpSpec(**kwargs)


# ------------------------------------------------------------- synthetic ---


def test_every_fault_produces_a_distinct_card():
    """If two faults traced the same card, no method could tell them apart and
    the taxonomy would be lying about what is diagnosable."""
    cards = {f: downhole_card(f, SPEC, severity=0.7).load_lb for f in FAULTS}
    for i, first in enumerate(FAULTS):
        for second in FAULTS[i + 1:]:
            difference = np.abs(cards[first] - cards[second]).max()
            assert difference > 0.01 * SPEC.fluid_load_lb, (
                "%s and %s trace nearly the same card" % (first, second))


def test_full_pump_traces_a_rectangle():
    card = downhole_card(FULL_PUMP, SPEC, severity=0.0)
    m = measure(card, SPEC)
    assert m.fillage > 0.95
    assert abs(m.upstroke_droop_fraction) < 0.1
    assert m.fluid_load_lb == pytest.approx(SPEC.fluid_load_lb, rel=0.02)


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError):
        downhole_card("gremlins", SPEC)


def test_fillage_label_is_a_position_fraction():
    """Fillage is the share of the barrel holding liquid, so the load must
    transfer at that height. Position is sinusoidal in phase, so labelling it
    by phase instead mislabels the data badly -- 0.58 fillage transfers at
    phase 0.79."""
    for severity in (0.0, 0.3, 0.6, 0.9):
        card = downhole_card(FLUID_POUND, SPEC, severity=severity)
        m = measure(card, SPEC)
        assert m.fillage == pytest.approx(true_fillage(FLUID_POUND, severity), abs=0.02)


# ----------------------------------------------------------- measurement ---


def test_measured_fillage_converts_to_barrels():
    """The reason fillage is worth measuring: it is a production number."""
    card = downhole_card(FLUID_POUND, SPEC, severity=0.6)
    m = measure(card, SPEC)
    assert m.displacement_bbl_per_day == pytest.approx(
        SPEC.displacement_bbl_per_day(m.fillage), rel=1e-9)
    assert m.displacement_bbl_per_day < SPEC.displacement_bbl_per_day(1.0)


def test_sharpness_separates_fluid_pound_from_gas():
    """The most valuable single call on a card. Liquid does not compress so
    the transfer is abrupt; gas does, so it slides. The remedies differ
    completely, which is why confusing them is expensive."""
    for severity in (0.4, 0.6, 0.8):
        pound = measure(downhole_card(FLUID_POUND, SPEC, severity=severity), SPEC)
        gas = measure(downhole_card(GAS_INTERFERENCE, SPEC, severity=severity), SPEC)
        assert pound.transfer_sharpness > 2.0 * gas.transfer_sharpness, (
            "severity %.1f: pound %.1f vs gas %.1f"
            % (severity, pound.transfer_sharpness, gas.transfer_sharpness))


def test_valve_leaks_move_the_lines_in_opposite_directions():
    """A traveling valve bleeds load off the top; a standing valve lifts the
    bottom. Getting these confused sends a crew after the wrong valve."""
    traveling = measure(downhole_card(TRAVELING_VALVE_LEAK, SPEC, severity=0.8), SPEC)
    standing = measure(downhole_card(STANDING_VALVE_LEAK, SPEC, severity=0.8), SPEC)
    assert traveling.upstroke_droop_fraction > 0.15
    assert standing.downstroke_lift_fraction > 0.10
    assert traveling.upstroke_droop_fraction > standing.upstroke_droop_fraction
    assert standing.downstroke_lift_fraction > traveling.downstroke_lift_fraction


def test_pickup_spread_isolates_tubing_movement():
    """Measured as the first crossing of each level rather than the width of
    the band. Using the band's extent instead scored a leaking traveling valve
    at 0.999, indistinguishable from unanchored tubing."""
    tubing = measure(downhole_card(TUBING_MOVEMENT, SPEC, severity=0.7), SPEC)
    others = [measure(downhole_card(f, SPEC, severity=0.7), SPEC).pickup_spread
              for f in FAULTS if f != TUBING_MOVEMENT]
    assert tubing.pickup_spread > 0.3
    assert tubing.pickup_spread > 2.0 * max(others)


def test_tagging_shows_compression():
    """A plunger striking bottom puts the rods briefly into compression --
    a load below the resting line, which nothing else on this list does."""
    tagging = measure(downhole_card(PUMP_TAGGING, SPEC, severity=0.8), SPEC)
    assert tagging.peak_compression_lb < -0.5 * SPEC.fluid_load_lb


def test_shape_features_are_scale_free():
    """A deep well with a big plunger and a shallow one with a small plunger
    must give the same features for the same fault. Otherwise a classifier
    learns which well it is looking at rather than what is wrong."""
    small = PumpSpec(plunger_diameter_in=1.25, net_lift_ft=2500.0,
                     stroke_in=64.0, strokes_per_minute=11.0)
    large = PumpSpec(plunger_diameter_in=2.25, net_lift_ft=7000.0,
                     stroke_in=144.0, strokes_per_minute=5.0)
    for fault in FAULTS:
        a = shape_features(downhole_card(fault, small, severity=0.7))
        b = shape_features(downhole_card(fault, large, severity=0.7))
        assert np.abs(a - b).max() < 1e-6, fault


def test_feature_vector_has_named_columns():
    card = downhole_card(FULL_PUMP, SPEC)
    assert feature_vector(card, SPEC).shape[0] == len(FEATURE_NAMES)
    assert np.all(np.isfinite(feature_vector(card, SPEC)))


# ------------------------------------------------------------- diagnosis ---


def test_full_pump_is_not_flagged():
    """The false-positive case. A tool that finds a fault on a healthy pump
    sends crews to wells that do not need them, and gets switched off."""
    diagnosis = diagnose(round_tripped(FULL_PUMP, 0.0), SPEC)
    assert diagnosis.primary.fault == FULL_PUMP
    assert "filling" in diagnosis.primary.evidence


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_fault_is_identified_when_clearly_present(fault):
    """Through the whole pipeline: pump card -> surface card -> wave equation
    -> diagnosis. The diagnosis never sees the card the fault was built from."""
    diagnosis = diagnose(round_tripped(fault, 0.8), SPEC)
    assert diagnosis.primary.fault == fault, (
        "diagnosed %s, evidence: %s" % (diagnosis.primary.fault,
                                        diagnosis.primary.evidence))


def test_diagnosis_carries_its_evidence():
    """A verdict an operator cannot check is not usable in a field where being
    confidently wrong costs a pulling unit."""
    diagnosis = diagnose(round_tripped(FLUID_POUND, 0.8), SPEC)
    assert diagnosis.primary.evidence
    assert "fillage" in diagnosis.primary.evidence
    assert diagnosis.notes and "bbl/day" in diagnosis.notes[0]


def test_worn_pump_needs_the_expected_load_not_just_the_shape():
    """A worn pump traces a normal-looking rectangle at a reduced load, so it
    is invisible to any measurement normalised by the card's own load. It is
    only detectable against the load the well *should* produce."""
    diagnosis = diagnose(round_tripped(WORN_PUMP, 0.8), SPEC)
    assert diagnosis.load_ratio < 0.8
    assert diagnosis.primary.fault == WORN_PUMP


def test_severity_raises_confidence():
    """A barely-present fault should not be reported as forcefully as a severe
    one."""
    mild = diagnose(round_tripped(FLUID_POUND, 0.35), SPEC)
    severe = diagnose(round_tripped(FLUID_POUND, 0.95), SPEC)
    assert severe.primary.confidence > mild.primary.confidence


def test_pipeline_accuracy_across_severities():
    """The headline number, measured rather than asserted. Mild faults near a
    threshold are genuinely ambiguous, so this is not expected to be perfect --
    it is the floor that the learned layer has to beat to be worth adding."""
    hits = total = 0
    for fault in FAULTS:
        for severity in (0.3, 0.5, 0.7, 0.9, 1.0):
            hits += diagnose(round_tripped(fault, severity), SPEC).primary.fault == fault
            total += 1
    assert hits / total >= 0.85, "rule layer scored %d/%d" % (hits, total)


def test_diagnosis_survives_a_different_well():
    """Thresholds must not be tuned to one pump. A deeper well with a larger
    plunger and a different string should diagnose the same faults."""
    spec = PumpSpec(plunger_diameter_in=2.0, net_lift_ft=6500.0,
                    fluid_specific_gravity=0.85, stroke_in=120.0,
                    strokes_per_minute=6.0)
    for fault in (FULL_PUMP, FLUID_POUND, GAS_INTERFERENCE, PUMP_TAGGING):
        surface, _ = surface_card_for(fault, taper_86(7000.0), spec, severity=0.8,
                                      damping_per_s=DAMPING)
        inferred = surface_to_downhole(surface, taper_86(7000.0), damping_per_s=DAMPING)
        assert diagnose(inferred, spec).primary.fault == fault, fault


def test_sharpness_survives_field_noise():
    """The pound-vs-gas call is the most valuable one a card offers, and a real
    dynamometer trace is not clean. Measuring sharpness as a percentile of
    |dL/dx| differentiates the signal and so amplifies noise straight into the
    answer -- it inverted at 4% rms, scoring gas (18.1) sharper than fluid
    pound (15.5), which is a confident wrong call sending a crew after the
    wrong remedy. The span form holds to 9%.
    """
    for noise_lb in (0.0, 0.02, 0.05, 0.09):
        rms = noise_lb * SPEC.fluid_load_lb
        for seed in range(6):
            rng = np.random.default_rng(seed)

            def noisy(fault):
                surface, _ = surface_card_for(fault, STRING, SPEC, severity=0.7,
                                              damping_per_s=DAMPING)
                if rms:
                    surface = type(surface)(
                        surface.position_in,
                        surface.load_lb + rng.normal(0.0, rms, surface.n_samples),
                        period_s=surface.period_s)
                return measure(surface_to_downhole(surface, STRING,
                                                   damping_per_s=DAMPING), SPEC)

            pound = noisy(FLUID_POUND).transfer_sharpness
            gas = noisy(GAS_INTERFERENCE).transfer_sharpness
            assert pound > 2.0 * gas, (
                "%.0f%% rms, seed %d: pound %.1f vs gas %.1f"
                % (100 * noise_lb, seed, pound, gas))


def test_diagnosis_holds_under_noise():
    """End to end with noise on the surface card, which is the only condition
    that matters -- a clean card is not a card from a well."""
    rms = 0.05 * SPEC.fluid_load_lb
    hits = total = 0
    for fault in FAULTS:
        for seed in range(4):
            rng = np.random.default_rng(seed)
            surface, _ = surface_card_for(fault, STRING, SPEC, severity=0.8,
                                          damping_per_s=DAMPING)
            surface = type(surface)(
                surface.position_in,
                surface.load_lb + rng.normal(0.0, rms, surface.n_samples),
                period_s=surface.period_s)
            inferred = surface_to_downhole(surface, STRING, damping_per_s=DAMPING)
            hits += diagnose(inferred, SPEC).primary.fault == fault
            total += 1
    assert hits / total >= 0.85, "scored %d/%d on noisy cards" % (hits, total)
