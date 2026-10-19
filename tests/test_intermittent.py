"""Intermittent gas lift: the cycle, the optimum, and what it will not claim.

The well used throughout is the one the module exists for -- 1200 psia
reservoir, 0.25 stb/d/psi, 5000 ft of lift -- because the behaviour worth
testing only appears on a well that cannot sustain a flowing column.

The tests that matter most here are the ones about limits. A cycle model
that returns a confident rate for a slug too thin to hold together, or that
recommends continuous lift on a well that would head, is worse than no
model, because the number it returns looks exactly like a good one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.intermittent import (
    FALLBACK_AT_OPTIMUM,
    MIN_COMPETENT_SLUG_FT,
    OPTIMUM_SLUG_VELOCITY_FT_MIN,
    IntermittentWell,
    calibrate_fallback,
    compare_with_continuous,
    cycle,
    fallback_fraction,
    optimise_cycles,
    screen,
    tubing_capacity_bbl_ft,
)
from phasespace.sim.wells import ProductivityIndexIPR, Tubing

OIL = LiquidProperties(api_gravity=32.0, water_cut=0.3)
GAS = GasProperties(gravity=0.7)


def low_pressure_well(**overrides):
    settings = dict(
        ipr=ProductivityIndexIPR(1200.0, 0.25),
        depth_ft=5000.0, tubing_id_in=2.441, wellhead_psia=80.0,
        liquid=OIL, gas=GAS)
    settings.update(overrides)
    return IntermittentWell(**settings)


# ------------------------------------------------------------- the geometry --


def test_tubing_capacity_matches_the_standard_formula():
    """C = d^2 / 1029.4 bbl/ft. 2.441 in tubing is a hair under 0.0058."""
    assert tubing_capacity_bbl_ft(2.441) == pytest.approx(0.005788, abs=1e-6)
    assert tubing_capacity_bbl_ft(1.995) == pytest.approx(0.003866, abs=1e-6)


def test_a_zero_diameter_is_refused():
    with pytest.raises(ValueError, match="diameter must be positive"):
        tubing_capacity_bbl_ft(0.0)


# ------------------------------------------------------------- the fallback --


def test_fallback_is_least_at_the_design_velocity():
    """Brown's optimum is about 1000 ft/min, and the penalty is symmetric
    enough either side that halving or doubling costs about the same."""
    best = fallback_fraction(OPTIMUM_SLUG_VELOCITY_FT_MIN, 5000.0)
    slow = fallback_fraction(OPTIMUM_SLUG_VELOCITY_FT_MIN / 2, 5000.0)
    fast = fallback_fraction(OPTIMUM_SLUG_VELOCITY_FT_MIN * 2, 5000.0)
    assert best < slow and best < fast
    assert slow == pytest.approx(fast, rel=1e-9)


def test_fallback_at_the_optimum_is_brown_s_anchor():
    """5% per 1000 ft, so 25% over 5000 ft of lift."""
    assert fallback_fraction(OPTIMUM_SLUG_VELOCITY_FT_MIN, 5000.0) == \
        pytest.approx(FALLBACK_AT_OPTIMUM * 5.0)


def test_fallback_grows_with_depth():
    shallow = fallback_fraction(OPTIMUM_SLUG_VELOCITY_FT_MIN, 2000.0)
    deep = fallback_fraction(OPTIMUM_SLUG_VELOCITY_FT_MIN, 8000.0)
    assert deep > shallow


def test_fallback_cannot_exceed_the_whole_slug():
    assert fallback_fraction(OPTIMUM_SLUG_VELOCITY_FT_MIN, 100_000.0) < 1.0


# ---------------------------------------------------------------- the cycle --


def test_a_cycle_too_fast_for_the_hardware_is_refused_not_approximated():
    """Lift plus blowdown is a floor on the period. Returning a rate for a
    cycle that cannot physically complete would be a number with nothing
    behind it."""
    well = low_pressure_well()
    result = cycle(well, 300.0)
    assert not result.feasible
    assert result.oil_stb_d == 0.0
    assert "cannot cycle this fast" in result.notes[0]


def test_the_accumulating_column_chokes_the_inflow():
    """The mechanism that makes slow cycling bad, and it is worth testing
    directly because the plausible explanation -- film loss on a short
    slug -- is *not* what this model does. Left long enough, the column
    drives bottomhole pressure to reservoir pressure and the slug stops
    growing entirely."""
    well = low_pressure_well()
    slow = cycle(well, 2.0)
    quicker = cycle(well, 4.0)
    assert slow.final_bottomhole_psia == pytest.approx(1200.0, rel=0.02)
    # Twice the waiting buys almost no extra liquid.
    assert slow.slug_bbl < quicker.slug_bbl * 1.1


def test_more_cycles_eventually_means_less_oil():
    well = low_pressure_well()
    assert cycle(well, 120.0).oil_stb_d < cycle(well, 36.0).oil_stb_d


def test_a_slug_too_thin_to_hold_together_is_flagged():
    """Gas channels through a short slug instead of lifting it, and a
    fallback expressed per 1000 ft cannot represent that. The rate is
    returned because refusing it would be worse, but it is labelled."""
    well = low_pressure_well()
    fast = cycle(well, 150.0)
    if fast.feasible and 0.0 < fast.final_column_ft < MIN_COMPETENT_SLUG_FT:
        assert any("channels through" in note for note in fast.notes)
    else:
        pytest.skip("this geometry does not reach the thin-slug regime")


def test_gas_per_cycle_rises_with_a_heavier_slug():
    """The gas has to carry the slug's weight, so a longer one costs more."""
    well = low_pressure_well()
    small = cycle(well, 60.0)
    large = cycle(well, 8.0)
    assert large.gas_per_cycle_scf > small.gas_per_cycle_scf


def test_gas_per_barrel_is_worse_at_high_frequency():
    """The economic reason not to simply cycle as fast as possible."""
    well = low_pressure_well()
    assert cycle(well, 120.0).gas_liquid_ratio > cycle(well, 20.0).gas_liquid_ratio


# -------------------------------------------------------------- the optimum --


def test_there_is_an_interior_optimum():
    """Neither the slowest nor the fastest cycle is best, which is the
    whole reason this needs solving rather than setting."""
    well = low_pressure_well()
    best = optimise_cycles(well)
    assert 5.0 < best.cycles_per_day < 100.0
    assert best.best.oil_stb_d > cycle(well, 4.0).oil_stb_d
    assert best.best.oil_stb_d > cycle(well, 120.0).oil_stb_d


def test_pricing_the_gas_slows_the_optimum_down():
    """Gas per barrel climbs steeply on the fast side, so an operator
    paying for compression should cycle more slowly than one maximising
    volume. The two optima are different wells to run."""
    well = low_pressure_well()
    volume = optimise_cycles(well, oil_price=1.0, gas_price=0.0)
    priced = optimise_cycles(well, oil_price=1.0, gas_price=0.0005)
    assert priced.cycles_per_day < volume.cycles_per_day
    assert priced.best.oil_stb_d <= volume.best.oil_stb_d


def test_the_optimum_carries_the_fallback_caveat():
    """The dominant uncertainty must travel with the number, not sit in
    documentation nobody opens."""
    assert any("fallback" in note for note in optimise_cycles(low_pressure_well()).notes)


def test_the_curve_is_in_the_form_the_allocator_consumes():
    """An intermittent well competes for the same compressor as the
    continuous ones. The allocator wants oil against gas."""
    best = optimise_cycles(low_pressure_well())
    gas, oil = best.performance_curve()
    assert gas.size == oil.size and gas.size > 5
    assert (gas > 0).all() and (oil > 0).all()


def test_a_well_too_deep_to_cycle_once_a_day_is_refused():
    well = low_pressure_well(depth_ft=2_000_000.0, slug_velocity_ft_min=1.0)
    with pytest.raises(ValueError, match="cannot complete"):
        optimise_cycles(well)


# ----------------------------------------------------------- the calibration --


def test_fallback_can_be_backed_out_of_one_measured_rate():
    """The one parameter the correlation cannot supply, and the only one
    in the cycle carrying real uncertainty. A round trip: set a fallback,
    predict a rate, recover the fallback from it."""
    well = low_pressure_well(fallback_per_1000ft=0.062)
    predicted = cycle(well, 24.0).oil_stb_d
    recovered = calibrate_fallback(low_pressure_well(), 24.0, predicted)
    assert recovered == pytest.approx(0.062, abs=1e-6)


def test_a_rate_the_formation_cannot_feed_is_refused():
    """A measured rate above what the inflow can supply means the model or
    the measurement is wrong, and silently fitting a negative fallback
    would hide both."""
    well = low_pressure_well()
    feedable = cycle(well, 24.0).slug_bbl * 24.0
    with pytest.raises(ValueError, match="can only feed"):
        calibrate_fallback(well, 24.0, feedable * 1.5)


# -------------------------------------------------------------- the choosing --


def test_a_low_productivity_well_is_screened_in():
    flagged, reasons = screen(ProductivityIndexIPR(1200.0, 0.25), 1200.0)
    assert flagged
    assert any("productivity index" in r for r in reasons)


def test_a_strong_well_is_not_screened_in():
    flagged, _reasons = screen(ProductivityIndexIPR(3000.0, 3.0), 3000.0)
    assert not flagged


def test_the_comparison_runs_both_models_and_picks_one():
    tubing = Tubing(depth_ft=5000.0, inner_diameter_in=2.441)
    choice = compare_with_continuous(
        low_pressure_well(), tubing, OIL, GAS,
        gas_available_scf_d=400_000.0, formation_gor=300.0,
        reservoir_psia=1200.0)
    assert choice.recommendation in ("continuous", "intermittent")
    assert choice.intermittent_oil_stb_d > 0.0


def test_recommending_continuous_on_a_screened_well_carries_the_stability_caveat():
    """The bias that would otherwise make this feature actively
    misleading. A steady-state intersection does not test whether the well
    heads, and heading is the failure that sends a well to intermittent
    lift in the first place -- so the wells where the answer matters most
    are the ones where the continuous figure is least trustworthy."""
    tubing = Tubing(depth_ft=5000.0, inner_diameter_in=2.441)
    choice = compare_with_continuous(
        low_pressure_well(), tubing, OIL, GAS,
        gas_available_scf_d=400_000.0, formation_gor=300.0,
        reservoir_psia=1200.0)
    if choice.recommendation != "continuous":
        pytest.skip("this case did not choose continuous")
    assert any("Asheim" in r for r in choice.reasons)
    assert any("upper bound" in r for r in choice.reasons)


def test_the_comparison_is_deterministic():
    tubing = Tubing(depth_ft=5000.0, inner_diameter_in=2.441)
    runs = [compare_with_continuous(
        low_pressure_well(), tubing, OIL, GAS, gas_available_scf_d=400_000.0,
        formation_gor=300.0, reservoir_psia=1200.0).intermittent_oil_stb_d
        for _ in range(2)]
    assert runs[0] == runs[1]


# ------------------------------------------------------- the stability gate --


def heading_inputs():
    """A low-productivity well on a generous orifice: the configuration
    that heads, which is the one continuous lift is wrongly recommended
    for."""
    from phasespace.sim.stability import StabilityInputs

    return StabilityInputs(
        lift_gas_scf_d=300_000.0, liquid_stb_d=186.0,
        productivity_stb_d_psi=0.25, orifice_diameter_in=0.25,
        injection_pressure_psia=900.0, injection_temperature_r=600.0,
        depth_to_valve_ft=5000.0, tubing_head_psia=80.0,
        tubing_volume_above_valve_bbl=29.0,
        annulus_volume_to_valve_bbl=70.0)


def test_a_heading_well_is_sent_to_intermittent_despite_more_oil_on_paper():
    """The whole point of implementing Asheim here. The continuous model
    computes a higher rate, and the well cannot hold it -- so a comparison
    that only reads the rates recommends the method that does not work."""
    from phasespace.sim.stability import asheim_stability

    tubing = Tubing(depth_ft=5000.0, inner_diameter_in=2.441)
    verdict = asheim_stability(heading_inputs())
    assert not verdict.stable, "this fixture is meant to fail both criteria"

    without = compare_with_continuous(
        low_pressure_well(), tubing, OIL, GAS, gas_available_scf_d=400_000.0,
        formation_gor=300.0, reservoir_psia=1200.0)
    with_stability = compare_with_continuous(
        low_pressure_well(), tubing, OIL, GAS, gas_available_scf_d=400_000.0,
        formation_gor=300.0, reservoir_psia=1200.0,
        stability_inputs=heading_inputs())

    assert without.recommendation == "continuous"
    assert without.continuous_oil_stb_d > without.intermittent_oil_stb_d
    assert with_stability.recommendation == "intermittent"
    # The rate itself must not change -- only the reading of it.
    assert (with_stability.continuous_oil_stb_d
            == pytest.approx(without.continuous_oil_stb_d))
    assert any("will not hold" in r for r in with_stability.reasons)


def test_the_verdict_is_attached_to_the_choice():
    """`stability is None` means it was not evaluated, which is a
    different thing from stable, and a caller must be able to tell."""
    tubing = Tubing(depth_ft=5000.0, inner_diameter_in=2.441)
    assert compare_with_continuous(
        low_pressure_well(), tubing, OIL, GAS, gas_available_scf_d=400_000.0,
        formation_gor=300.0, reservoir_psia=1200.0).stability is None
    assert compare_with_continuous(
        low_pressure_well(), tubing, OIL, GAS, gas_available_scf_d=400_000.0,
        formation_gor=300.0, reservoir_psia=1200.0,
        stability_inputs=heading_inputs()).stability is not None
