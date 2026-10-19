"""Tests for the rod-string wave equation.

This is the audit anchor for rod-pump diagnostics: every downhole card, every
pump-fillage number and every fault call rests on it, so it is checked against
things known without it -- the acoustic velocity of steel, the buoyant weight
of the string, conservation of energy under damping, and exact reversibility.

Two real bugs were caught here and both are pinned below. The second is worth
knowing about: a wrong sign in the position/load coupling still round-trips
perfectly, because the inverse of a wrong transfer is still its inverse. Only
an energy argument exposes it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.rodpump.rodstring import (
    API_ROD_DIAMETER_IN,
    RodSection,
    RodString,
    taper_86,
    uniform_string,
)
from phasespace.rodpump.wave import Card, downhole_to_surface, surface_to_downhole


def make_card(spm=8.0, stroke_in=100.0, samples=256, mean_lb=12000.0, amplitude_lb=3000.0):
    t = np.arange(samples) / samples
    position = stroke_in / 2.0 * (1.0 - np.cos(2.0 * np.pi * t))
    load = mean_lb + amplitude_lb * np.sin(2.0 * np.pi * t)
    return Card(position, load, period_s=60.0 / spm)


# ----------------------------------------------------------- rod string ----


def test_acoustic_velocity_matches_steel():
    """sqrt(E g / rho) for steel is about 16,000-17,000 ft/s. Every reflection
    time in the model scales with this, so an error here mistimes the whole
    downhole card."""
    section = RodSection(api_number=7, length_ft=1000.0)
    assert 15500.0 < section.acoustic_velocity_ft_s < 17500.0


def test_acoustic_velocity_is_independent_of_rod_size():
    """A taper changes acoustic impedance, not wave speed. If these differed,
    the section-by-section propagation would be wrong for the wrong reason."""
    speeds = [RodSection(n, 100.0).acoustic_velocity_ft_s for n in API_ROD_DIAMETER_IN]
    assert max(speeds) - min(speeds) < 1e-6


def test_api_rod_number_is_eighths_of_an_inch():
    assert RodSection(8, 100.0).diameter_in == pytest.approx(1.0)
    assert RodSection(6, 100.0).diameter_in == pytest.approx(0.75)


def test_rod_weight_is_physical():
    """A 7/8 in steel rod weighs about 2.2 lb/ft; a 6,000 ft string of it is
    roughly 12,000 lb, which is the dominant term in any surface card."""
    section = RodSection(api_number=7, length_ft=1.0)
    assert 1.9 < section.weight_lb_per_ft < 2.4
    assert 11000.0 < uniform_string(7, 6000.0).total_weight_lb < 13500.0


def test_buoyancy_reduces_weight_by_about_a_ninth():
    """Steel is ~7.85 times water; in 0.85 SG fluid the rods lose about 11% of
    their weight. Ignoring this overstates static load by that much and shifts
    every load threshold read off the card."""
    string = uniform_string(7, 6000.0)
    ratio = string.buoyant_weight_lb(0.85) / string.total_weight_lb
    assert 0.87 < ratio < 0.92


def test_tapered_string_is_heavier_at_the_top():
    string = taper_86(6000.0)
    assert string.sections[0].weight_lb_per_ft > string.sections[1].weight_lb_per_ft
    assert string.total_length_ft == pytest.approx(6000.0)


def test_rod_string_rejects_nonsense():
    with pytest.raises(ValueError):
        RodSection(api_number=99, length_ft=100.0)
    with pytest.raises(ValueError):
        RodSection(api_number=7, length_ft=0.0)
    with pytest.raises(ValueError):
        RodString([])


# ----------------------------------------------------------------- card ----


def test_card_validates_its_inputs():
    with pytest.raises(ValueError):
        Card(np.zeros(10), np.zeros(9), period_s=7.5)
    with pytest.raises(ValueError):
        Card(np.zeros(4), np.zeros(4), period_s=7.5)
    with pytest.raises(ValueError):
        Card(np.zeros(16), np.zeros(16), period_s=0.0)


def test_card_area_is_the_enclosed_loop():
    """A unit rectangle traversed anticlockwise encloses unit area. The card
    area is a work measurement downhole, so the sign and scale must be right."""
    position = np.array([0.0, 1.0, 1.0, 0.0] * 4)
    load = np.array([0.0, 0.0, 1.0, 1.0] * 4)
    card = Card(position, load, period_s=1.0)
    assert abs(abs(card.area_lb_in()) - 4.0) < 1e-9


# ------------------------------------------------------------ transfer ----


def test_quasi_static_limit_is_the_buoyant_rod_weight():
    """The one case with an answer known without the wave equation: pumped
    slowly enough, the downhole card is the surface card less the weight of
    rod hanging between them."""
    string = uniform_string(7, 6000.0)
    card = make_card(spm=0.05)
    downhole = surface_to_downhole(card, string, damping_per_s=0.4)
    drop = card.load_lb.mean() - downhole.load_lb.mean()
    assert drop == pytest.approx(string.buoyant_weight_lb(0.85), rel=1e-6)


def test_mean_load_drop_holds_at_normal_pumping_speed():
    """The static term carries the rod weight at every speed -- a harmonic
    carries no net force, so nothing else can."""
    string = uniform_string(7, 6000.0)
    for spm in (2.0, 8.0, 15.0):
        card = make_card(spm=spm)
        downhole = surface_to_downhole(card, string, damping_per_s=0.4)
        drop = card.load_lb.mean() - downhole.load_lb.mean()
        assert drop == pytest.approx(string.buoyant_weight_lb(0.85), rel=1e-6)


@pytest.mark.parametrize("spm", [2.0, 8.0, 15.0])
def test_round_trip_recovers_the_original_card(spm):
    """Down then back up must return what it started with. This caught a real
    bug: the static stretch used the same expression in both directions, but
    average tension is the local load minus half the section weight going down
    and plus half going up. The position came back 43 in out on a 100 in
    stroke."""
    string = uniform_string(7, 6000.0)
    card = make_card(spm=spm)
    back = downhole_to_surface(surface_to_downhole(card, string, 0.4), string, 0.4)
    assert np.abs(back.load_lb - card.load_lb).max() < 1e-6
    assert np.abs(back.position_in - card.position_in).max() < 1e-9


def test_round_trip_holds_through_a_taper():
    """Each taper change is an impedance discontinuity. Getting the section
    ordering wrong on the way back up would show up here and nowhere else."""
    string = RodString([RodSection(8, 2000.0), RodSection(7, 2000.0), RodSection(6, 2000.0)])
    card = make_card(spm=8.0)
    back = downhole_to_surface(surface_to_downhole(card, string, 0.4), string, 0.4)
    assert np.abs(back.load_lb - card.load_lb).max() < 1e-6
    assert np.abs(back.position_in - card.position_in).max() < 1e-9


def test_damping_dissipates_work_going_down():
    """The check that exposes a sign error the round trip cannot.

    A wrong sign in the position/load coupling still round-trips exactly,
    because the inverse of a wrong transfer is still its inverse. Only energy
    tells them apart: for a fixed surface card, more rod damping means more
    work lost in the string, so less reaches the pump. With the sign wrong,
    more damping delivered *more* work to the pump.
    """
    string = uniform_string(7, 6000.0)
    card = make_card(spm=8.0)
    areas = [abs(surface_to_downhole(card, string, damping_per_s=c).area_lb_in())
             for c in (0.0, 0.1, 0.3, 0.6, 1.0)]
    for lighter, heavier in zip(areas, areas[1:]):
        assert heavier < lighter, "damping added work at the pump: %s" % areas
    assert areas[-1] < 0.75 * areas[0]


def test_damping_demands_more_work_at_surface():
    """The same statement read the other way: driving a fixed pump card
    through a more damped string costs more at the polished rod."""
    string = uniform_string(7, 6000.0)
    samples = 256
    t = np.arange(samples) / samples
    card = Card(50.0 * (1.0 - np.cos(2.0 * np.pi * t)),
                6000.0 + 2500.0 * np.sin(2.0 * np.pi * t),
                period_s=60.0 / 8.0)
    areas = [abs(downhole_to_surface(card, string, damping_per_s=c).area_lb_in())
             for c in (0.0, 0.1, 0.3, 0.6, 1.0)]
    for lighter, heavier in zip(areas, areas[1:]):
        assert heavier > lighter, "damping reduced the work required: %s" % areas


def test_undamped_string_conserves_work():
    """With no damping the string stores and returns energy but dissipates
    none, so the work at both ends should be close. It is not identical --
    rod inertia shifts phase -- but a large gap would mean the lossless case
    is losing energy somewhere it should not."""
    string = uniform_string(7, 4000.0)
    card = make_card(spm=6.0)
    downhole = surface_to_downhole(card, string, damping_per_s=0.0)
    ratio = abs(downhole.area_lb_in()) / abs(card.area_lb_in())
    assert 0.9 < ratio < 1.1


def test_negative_damping_is_rejected():
    """Negative damping would add energy to a passive steel rod."""
    string = uniform_string(7, 6000.0)
    card = make_card()
    with pytest.raises(ValueError):
        surface_to_downhole(card, string, damping_per_s=-0.1)
    with pytest.raises(ValueError):
        downhole_to_surface(card, string, damping_per_s=-0.1)


def test_card_metadata_survives_propagation():
    string = uniform_string(7, 6000.0)
    card = make_card(spm=9.0, samples=128)
    downhole = surface_to_downhole(card, string, 0.4)
    assert downhole.n_samples == card.n_samples
    assert downhole.period_s == pytest.approx(card.period_s)
    assert downhole.strokes_per_minute == pytest.approx(9.0)


def test_longer_string_shifts_load_further():
    """More rod between the two cards means a larger static offset. A model
    that ignored string length would fail this."""
    card = make_card(spm=8.0)
    drops = []
    for length in (2000.0, 4000.0, 8000.0):
        string = uniform_string(7, length)
        downhole = surface_to_downhole(card, string, 0.4)
        drops.append(card.load_lb.mean() - downhole.load_lb.mean())
    assert drops[0] < drops[1] < drops[2]
