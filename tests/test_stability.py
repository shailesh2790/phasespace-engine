"""Asheim's gas-lift stability criteria, and the limits of this transcription.

The criteria are from Asheim, SPE 16468-PA (1988), by way of Durham's
SPE-EuALF 2021 presentation on applying them. The original was not
available, so what is tested here is split deliberately in two.

The *sensitivities* are testable and are tested: they are stated in words
in the source ("stability promoted by high PI, small orifice port size"
for F1; "shallow injection point, higher THP" for F2), and a transcription
error in the algebra would break them. These are the tests that would
actually catch a mistake.

The *absolute calibration* is not verified, and no test here pretends
otherwise. The published worked example omits the injection-point PVT that
F1 depends on and the tubing and annulus volumes that F2 depends on, so it
cannot be reproduced. `test_the_published_case_is_bracketed_not_matched`
records exactly that, so the gap stays visible instead of being quietly
forgotten the next time someone reads an F number off a screen.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.stability import (
    MARGINAL_BAND,
    StabilityInputs,
    asheim_stability,
    depletion_response,
    inflow_response,
)


def base(**overrides) -> StabilityInputs:
    settings = dict(
        lift_gas_scf_d=1.0e6, liquid_stb_d=3000.0,
        productivity_stb_d_psi=5.0, orifice_diameter_in=0.25,
        injection_pressure_psia=1500.0, injection_temperature_r=620.0,
        gas_gravity=0.7, z_factor=0.88,
        depth_to_valve_ft=5000.0, tubing_head_psia=200.0,
        tubing_volume_above_valve_bbl=30.0,
        annulus_volume_to_valve_bbl=80.0)
    settings.update(overrides)
    return StabilityInputs(**settings)


# -------------------------------------------------------- the sensitivities --


def test_a_higher_productivity_index_stabilises_the_inflow_response():
    """F1 is proportional to J: a reservoir that refills the tubing faster
    is harder to run away from."""
    assert inflow_response(base(productivity_stb_d_psi=10.0)) == \
        pytest.approx(2.0 * inflow_response(base(productivity_stb_d_psi=5.0)))


def test_a_smaller_orifice_stabilises_the_inflow_response():
    """F1 goes as 1/A^2, so halving the port diameter is a sixteen-fold
    increase. This is why orifice sizing is the usual fix for a heading
    well, and why F1 is sensitive to the assumed orifice efficiency."""
    small = inflow_response(base(orifice_diameter_in=0.125))
    large = inflow_response(base(orifice_diameter_in=0.25))
    assert small == pytest.approx(16.0 * large)


def test_a_shallower_injection_point_stabilises_the_depletion_response():
    shallow = depletion_response(base(depth_to_valve_ft=3000.0), f1=0.5)
    deep = depletion_response(base(depth_to_valve_ft=9000.0), f1=0.5)
    assert shallow > deep


def test_a_higher_tubing_head_pressure_stabilises_the_depletion_response():
    high = depletion_response(base(tubing_head_psia=400.0), f1=0.5)
    low = depletion_response(base(tubing_head_psia=100.0), f1=0.5)
    assert high > low


def test_a_bigger_annulus_destabilises_the_depletion_response():
    """F2 is the tubing-to-annulus volume ratio. A large annulus holds
    enough gas to keep feeding the surge instead of throttling it."""
    small = depletion_response(base(annulus_volume_to_valve_bbl=40.0), f1=0.5)
    large = depletion_response(base(annulus_volume_to_valve_bbl=160.0), f1=0.5)
    assert small > large


def test_both_criteria_are_dimensionless():
    """The check worth doing on a transcribed formula. F1 resolves to
    kg/(s^2.Pa.m) and Pa is kg/(m.s^2), so it cancels -- if it did not, a
    unit slip somewhere would show up as a wildly scaled number that still
    looked plausible."""
    value = inflow_response(base())
    assert 1e-6 < value < 1e6, (
        "F1 = %g is not a dimensionless group's plausible magnitude; "
        "suspect a unit conversion" % value)


# -------------------------------------------------------------- the verdict --


def test_either_criterion_above_one_is_stable():
    """Asheim's condition is a disjunction, not a conjunction. Requiring
    both would call a large fraction of working wells unstable."""
    verdict = asheim_stability(base(productivity_stb_d_psi=400.0))
    assert verdict.f1 > 1.0
    assert verdict.stable


def test_failing_both_is_heading():
    verdict = asheim_stability(base(
        productivity_stb_d_psi=0.2, orifice_diameter_in=0.5,
        depth_to_valve_ft=9000.0, tubing_head_psia=40.0,
        annulus_volume_to_valve_bbl=300.0))
    assert verdict.f1 < 1.0 and verdict.f2 < 1.0
    assert not verdict.stable
    assert "heading" in verdict.describe()


def test_f2_is_not_evaluated_against_a_stable_f1():
    """The (1 - F1) term is the inflow response F2 arrests. With F1 above
    1 there is nothing to arrest, and the expression would go negative --
    which would read as 'unstable' on a well that is stable."""
    verdict = asheim_stability(base(productivity_stb_d_psi=400.0))
    assert math.isinf(verdict.f2)
    assert verdict.stable


def test_a_result_near_the_threshold_is_called_marginal():
    """Given that the absolute calibration is unverified, a number just
    either side of 1 is not a decision."""
    found = None
    for productivity in [p / 100.0 for p in range(10, 4000)]:
        verdict = asheim_stability(base(productivity_stb_d_psi=productivity))
        if abs(verdict.f1 - 1.0) < MARGINAL_BAND / 2.0:
            found = verdict
            break
    assert found is not None
    assert found.marginal
    assert "not been verified" in " ".join(found.reasons)


def test_f2_is_nan_rather_than_guessed_when_its_inputs_are_missing():
    """Tubing and annulus volumes are often not to hand. Defaulting them
    would produce an F2 that looks like a measurement."""
    verdict = asheim_stability(base(
        productivity_stb_d_psi=0.2, annulus_volume_to_valve_bbl=0.0))
    assert verdict.f2 != verdict.f2            # NaN
    assert any("could not be evaluated" in r for r in verdict.reasons)
    assert any("pessimistic half" in r for r in verdict.reasons)


def test_every_verdict_carries_the_assumptions():
    """Six assumptions, each violable in a real well. They travel with the
    number rather than living in documentation."""
    verdict = asheim_stability(base())
    assert len(verdict.assumptions) == 6
    assert any("multi-pointing" in a for a in verdict.assumptions)


# ----------------------------------------------------------- the provenance --


def test_the_published_case_is_bracketed_not_matched():
    """The limit of this transcription, asserted so it stays visible.

    Durham's presentation reports F1 = 0.76 for a well with J = 18
    stb/d/psi, 2.5 MMscf/d of lift gas, 14,000 stb/d of liquid and a 15/64
    in orifice near 6000 ft. It does not give the injection-point
    pressure, temperature, Z or gas gravity, all of which F1 depends on
    through B_g. Reconstructing those brackets their figure rather than
    reproducing it, which is evidence the algebra is transcribed correctly
    and is not a verification of the absolute value.
    """
    def f1_at(pressure):
        return inflow_response(StabilityInputs(
            lift_gas_scf_d=2.5e6, liquid_stb_d=14000.0,
            productivity_stb_d_psi=18.0, orifice_diameter_in=15.0 / 64.0,
            injection_pressure_psia=pressure, injection_temperature_r=639.67,
            gas_gravity=0.7, z_factor=0.86))

    low, high = f1_at(3500.0), f1_at(2000.0)
    assert low < high, "F1 must fall as injection pressure rises"
    assert low < 1.0 < high, (
        "the published 0.76 should sit inside the range the unstated "
        "injection PVT can produce; got %.2f to %.2f" % (low, high))


def test_inputs_that_cannot_give_an_answer_are_refused():
    with pytest.raises(ValueError, match="orifice diameter must be positive"):
        base(orifice_diameter_in=0.0)
    with pytest.raises(ValueError, match="liquid rate must be positive"):
        base(liquid_stb_d=0.0)
    with pytest.raises(ValueError, match="productivity index must be positive"):
        base(productivity_stb_d_psi=0.0)
