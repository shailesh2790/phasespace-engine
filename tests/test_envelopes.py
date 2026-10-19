"""Tests for correlation applicability.

The point of this module is that it fires on ordinary field conditions, not on
absurd ones. A validity check that only complains about a 400 in pipe teaches
people that it never matters; one that says "this 3.5 in tubing is outside the
1-1.5 in Hagedorn-Brown was fitted on, and the error goes in the direction of
overstating bottomhole pressure" changes how the answer is read.

So several tests below assert that a perfectly normal well trips a warning.
That is the intended behaviour, not a false positive.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.envelopes import (
    MARGINAL_FRACTION,
    Envelope,
    beggs_brill_applicability,
    hagedorn_brown_applicability,
    turner_applicability,
    worst,
    z_factor_applicability,
)
from phasespace.sim.fluids import GasProperties
from phasespace.sim.units import f_to_r


# ----------------------------------------------------------- the mechanism --


def test_inside_a_range_is_silent():
    envelope = Envelope("Width", "in", 1.0, 2.0, "basis", "consequence")
    check = envelope.check(1.5)
    assert check.inside and check.severity == "inside"


def test_severity_is_graded_by_distance_not_merely_by_being_outside():
    """A 1.6 in tubing against a 1.5 in basis is a footnote; a 7 in casing is a
    different flow regime. Treating them the same trains people to ignore the
    warning entirely."""
    envelope = Envelope("Pipe diameter", "in", 1.0, 1.5, "basis", "consequence")
    assert envelope.check(1.5 * (1.0 + MARGINAL_FRACTION / 2)).severity == "marginal"
    assert envelope.check(7.0).severity == "outside"


def test_a_violation_carries_its_direction_source_and_consequence():
    """"Outside 1.0-1.5 in" is not actionable. Which way the error goes is."""
    envelope = Envelope("Pipe diameter", "in", 1.0, 1.5,
                        "fitted on acrylic pipe", "holdup over-predicts")
    message = envelope.check(4.0).describe()
    assert "above" in message
    assert "fitted on acrylic pipe" in message
    assert "holdup over-predicts" in message
    assert "4" in message and "1.5" in message


def test_below_a_range_reads_as_below():
    envelope = Envelope("Depth", "ft", 1000.0, 5000.0, "basis", "consequence")
    assert "below" in envelope.check(200.0).describe()


# ------------------------------------------------------------ Beggs-Brill --


def test_beggs_brill_flags_field_pressure():
    """The largest extrapolation in ordinary use, and the one nobody mentions:
    the correlation was fitted on an air-water loop at 35-95 psia, and is then
    applied at two thousand."""
    result = beggs_brill_applicability(diameter_in=2.441, inclination_deg=90.0,
                                       pressure_psia=2000.0)
    quantities = [v.envelope.quantity for v in result.violations]
    assert "Pressure" in quantities
    assert result.severity == "outside"
    assert any("gas density" in m for m in result.messages())


def test_beggs_brill_is_content_with_its_own_test_loop():
    """Inside the published basis it must say nothing at all, or the warnings
    are noise."""
    result = beggs_brill_applicability(diameter_in=1.25, inclination_deg=45.0,
                                       pressure_psia=60.0,
                                       liquid_viscosity_cp=1.0)
    assert result.ok, result.messages()
    assert result.severity == "inside"


def test_beggs_brill_accepts_any_real_angle():
    """The angle coverage is the correlation's genuine strength, and the reason
    it is the default here. It must not be flagged for using it."""
    for angle in (-90.0, -30.0, 0.0, 45.0, 90.0):
        result = beggs_brill_applicability(1.25, angle, 60.0)
        assert "Inclination" not in [v.envelope.quantity for v in result.violations]


def test_beggs_brill_flags_viscous_oil():
    result = beggs_brill_applicability(1.25, 90.0, 60.0, liquid_viscosity_cp=40.0)
    assert "Liquid viscosity" in [v.envelope.quantity for v in result.violations]
    assert any("under-predicts holdup" in m for m in result.messages())


# -------------------------------------------------------- Hagedorn-Brown --


def test_hagedorn_brown_flags_a_deviated_well_loudly():
    """The mistake this is here to catch. Hagedorn-Brown carries no inclination
    term at all, so a deviated section is computed as though vertical."""
    result = hagedorn_brown_applicability(diameter_in=2.441, inclination_deg=55.0,
                                          depth_ft=8000.0)
    assert "Inclination" in [v.envelope.quantity for v in result.violations]
    assert any("no inclination term" in m for m in result.messages())


def test_hagedorn_brown_is_happy_vertical():
    result = hagedorn_brown_applicability(1.5, 90.0, 1200.0, liquid_viscosity_cp=20.0)
    assert result.ok, result.messages()


def test_inclination_is_compared_on_magnitude():
    """A well flowing downhill at -90 degrees is as vertical as one at +90; the
    sign is a direction, not a deviation."""
    assert hagedorn_brown_applicability(1.5, -90.0, 1000.0).ok


def test_a_normal_deep_well_trips_several_at_once():
    """The honest headline for most real wells: a 2.4 in, 8,000 ft producer is
    outside the basis of both correlations on diameter and depth. That is worth
    saying once, plainly, rather than implying the answer is exact."""
    result = hagedorn_brown_applicability(2.441, 90.0, 8000.0)
    quantities = {v.envelope.quantity for v in result.violations}
    assert {"Pipe diameter", "Depth"} <= quantities


# ------------------------------------------------------------- Z factor ----


def test_z_factor_envelope_uses_pseudo_reduced_conditions():
    gas = GasProperties(gravity=0.65, temperature_r=f_to_r(100.0))
    p_pc, t_pc = gas.pseudo_criticals()
    assert z_factor_applicability(1000.0, f_to_r(100.0), p_pc, t_pc).ok


def test_z_factor_flags_a_cold_gas():
    """Below Tpr = 1 the gas may be two phase, which a single-phase Z cannot
    represent at all -- a much more serious statement than being off a chart."""
    gas = GasProperties(gravity=0.65, temperature_r=f_to_r(-120.0))
    p_pc, t_pc = gas.pseudo_criticals()
    result = z_factor_applicability(600.0, f_to_r(-120.0), p_pc, t_pc)
    assert not result.ok
    assert any("two-phase" in m for m in result.messages())


def test_z_factor_rejects_impossible_criticals():
    with pytest.raises(ValueError):
        z_factor_applicability(1000.0, 560.0, 0.0, 350.0)


# --------------------------------------------------------------- Turner ----


def test_turner_flags_a_wet_gas_well():
    assert not turner_applicability(400.0).ok
    assert turner_applicability(20.0).ok


# ------------------------------------------------------------- combining ---


def test_worst_picks_the_weakest_link():
    """A result computed through several correlations is only as well supported
    as its weakest. Reporting them separately invites reading the reassuring
    one and stopping there."""
    clean = beggs_brill_applicability(1.25, 45.0, 60.0)
    broken = hagedorn_brown_applicability(7.0, 30.0, 12000.0)
    assert worst(clean, broken) is broken
    assert worst(clean, clean) is clean


def test_worst_prefers_the_larger_excess_among_equals():
    mild = turner_applicability(140.0)
    severe = turner_applicability(900.0)
    assert worst(mild, severe) is severe


def test_worst_needs_something_to_compare():
    with pytest.raises(ValueError):
        worst(None)


def test_as_dict_is_serialisable_and_complete():
    """The API hands this to a browser, so it has to survive JSON with every
    field the UI needs to render a badge and a reason."""
    import json

    payload = beggs_brill_applicability(4.0, 90.0, 2000.0, 40.0).as_dict()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["severity"] == "outside"
    assert round_tripped["reference"].startswith("Beggs & Brill")
    for violation in round_tripped["violations"]:
        assert set(violation) == {"quantity", "value", "low", "high", "units",
                                  "severity", "message"}
