"""Tests for gas-lift valve capacity and the compressor constraint.

Two kinds of check.

Against independent physics: the critical pressure ratio has a closed form, a
choked port must stop responding to downstream pressure, and the static gas
column has an exponential solution. None of those depend on the correlation
being right -- they are properties any correct implementation must have.

Against the consequence: an allocation that ignores injection pressure returns
setpoints nobody can execute. The tests below put a compressor limit on a real
allocation and check that the result changes, that no well is handed more than
its valve can pass, and that a well the compressor cannot reach is dropped
rather than quietly given gas it will never see.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.gaslift import GasLiftWell, allocate
from phasespace.sim.gaslift_valve import (
    GAS_HEAT_CAPACITY_RATIO,
    SCF_PER_MSCF,
    SM3_PER_MSCF,
    apply_injection_limits,
    casing_pressure_at_depth,
    critical_pressure_ratio,
    injection_capacity,
    thornhill_craver_rate,
)

QUARTER_INCH_AREA = math.pi * 0.25 ** 2 / 4.0

# The allocation tests below work in sm3/day throughout, because the wells do.
#
# The first version of this file converted valve capacity to scf/day while the
# wells carried sm3/day limits -- a factor of 35 that made every limit look
# non-binding and three tests fail for a reason that had nothing to do with
# the physics. That is exactly the error `injection_capacity` takes an
# explicit `units_per_mscf` to prevent, and it caught its own author.
ALLOCATION_UNITS = SM3_PER_MSCF


def well(name, q_min=1000.0, q_max=25000.0, current=9000.0, uplift=5.0,
         response=1.5e-4, friction=2.0e-5):
    rates = np.array([0, 3000, 6000, 9000, 13000, 18000], dtype=float)
    oil = uplift * (1.0 - np.exp(-response * rates)) - friction * rates * 1e-2
    tests = [(float(q), float(max(0.0, o))) for q, o in zip(rates, oil)]
    return GasLiftWell(name, tests, q_min=q_min, q_max=q_max,
                       current=current).fit()


# ------------------------------------------------- against the closed form --


def test_critical_ratio_matches_the_textbook_value():
    """(2/(k+1))^(k/(k-1)) at k = 1.27 is 0.551, which is where the familiar
    'choked below about 0.55' comes from. Checked against the arithmetic."""
    k = GAS_HEAT_CAPACITY_RATIO
    assert critical_pressure_ratio() == pytest.approx(
        (2.0 / (k + 1.0)) ** (k / (k - 1.0)), rel=1e-12)
    assert critical_pressure_ratio() == pytest.approx(0.5512, abs=0.001)


def test_a_choked_port_stops_caring_about_downstream_pressure():
    """The defining property of choked flow, and the reason a gas-lift valve's
    capacity is set by the casing side rather than the tubing side over most of
    its range."""
    rates = [thornhill_craver_rate(QUARTER_INCH_AREA, 1000.0, p, 580.0)
             for p in (500.0, 400.0, 200.0, 50.0, 1.0)]
    assert max(rates) - min(rates) < 1e-9, rates

    # Above the critical ratio it does respond.
    assert (thornhill_craver_rate(QUARTER_INCH_AREA, 1000.0, 800.0, 580.0)
            < thornhill_craver_rate(QUARTER_INCH_AREA, 1000.0, 600.0, 580.0))


def test_no_differential_passes_no_gas():
    """Tubing at or above casing pressure means the valve passes nothing.
    Returned as zero rather than as the negative root the expression would
    otherwise take."""
    assert thornhill_craver_rate(QUARTER_INCH_AREA, 900.0, 900.0, 580.0) == 0.0
    assert thornhill_craver_rate(QUARTER_INCH_AREA, 900.0, 1200.0, 580.0) == 0.0


def test_capacity_scales_with_port_area_not_diameter():
    """Doubling the port diameter quadruples the area and so the capacity. A
    system sized on diameter rather than area is out by a factor of four."""
    small = thornhill_craver_rate(math.pi * 0.125 ** 2 / 4.0, 1000.0, 400.0, 580.0)
    large = thornhill_craver_rate(math.pi * 0.250 ** 2 / 4.0, 1000.0, 400.0, 580.0)
    assert large == pytest.approx(4.0 * small, rel=1e-9)


def test_capacity_is_a_plausible_field_number():
    """A quarter-inch port at 1,000 psia passing something over a megascuff a
    day. Gas-lift injection rates run a few hundred Mscf/d to a couple of
    MMscf/d, so this is the right order; a result in the wrong decade would be
    a units error in the constant."""
    rate = thornhill_craver_rate(QUARTER_INCH_AREA, 1000.0, 400.0, 580.0)
    assert 500.0 < rate < 3000.0, rate


def test_derating_only_reduces_and_is_bounded():
    full = thornhill_craver_rate(QUARTER_INCH_AREA, 1000.0, 400.0, 580.0)
    assert thornhill_craver_rate(QUARTER_INCH_AREA, 1000.0, 400.0, 580.0,
                                 derating=0.7) == pytest.approx(0.7 * full)
    for bad in (0.0, -0.2, 1.4):
        with pytest.raises(ValueError, match="derating"):
            thornhill_craver_rate(QUARTER_INCH_AREA, 1000.0, 400.0, 580.0,
                                  derating=bad)


# ------------------------------------------------------- the gas column ----


def test_casing_pressure_rises_with_depth_exponentially():
    """p = p0 * exp(0.01877 * G * D / (T Z)), checked against the arithmetic
    and against the magnitude an engineer expects: 15 to 20 percent on a
    6,000 ft well."""
    surface, depth, t, g, z = 1200.0, 6000.0, 580.0, 0.65, 0.9
    expected = surface * math.exp(0.01877 * g * depth / (t * z))
    assert casing_pressure_at_depth(surface, depth, t, g, z) == pytest.approx(
        expected, rel=1e-12)

    uplift = (expected - surface) / surface
    assert 0.10 < uplift < 0.25, uplift


def test_zero_depth_is_surface_pressure():
    assert casing_pressure_at_depth(900.0, 0.0, 580.0) == pytest.approx(900.0)


def test_a_heavier_gas_builds_more_column():
    light = casing_pressure_at_depth(1000.0, 6000.0, 580.0, gas_gravity=0.60)
    heavy = casing_pressure_at_depth(1000.0, 6000.0, 580.0, gas_gravity=0.85)
    assert heavy > light


# ------------------------------------------------------ what it means ------


def test_deeper_injection_helps_through_the_column():
    """More annulus means more static head, so the same compressor delivers a
    higher pressure at the valve and the port passes more."""
    shallow = injection_capacity("A", 1200.0, 3000.0, 700.0, 0.25, 580.0)
    deep = injection_capacity("A", 1200.0, 8000.0, 700.0, 0.25, 580.0)
    assert deep.rate > shallow.rate
    assert deep.casing_at_depth_psia > shallow.casing_at_depth_psia


def test_a_well_the_compressor_cannot_reach_takes_nothing():
    limit = injection_capacity("A", 600.0, 2000.0, 1400.0, 0.25, 580.0)
    assert limit.rate == 0.0
    assert "cannot take gas" in limit.describe()


def test_units_are_explicit_because_the_error_is_invisible():
    """Mscf/day into scf/day or sm3/day. Mixing the two is a factor of about
    35 and produces a number that looks entirely plausible on a page."""
    scf = injection_capacity("A", 1200.0, 6000.0, 700.0, 0.25, 580.0,
                             units_per_mscf=SCF_PER_MSCF)
    sm3 = injection_capacity("A", 1200.0, 6000.0, 700.0, 0.25, 580.0,
                             units_per_mscf=SM3_PER_MSCF)
    assert scf.rate / sm3.rate == pytest.approx(SCF_PER_MSCF / SM3_PER_MSCF)


def test_nonsense_inputs_are_refused():
    with pytest.raises(ValueError, match="port diameter"):
        injection_capacity("A", 1200.0, 6000.0, 700.0, 0.0, 580.0)
    with pytest.raises(ValueError, match="surface pressure"):
        casing_pressure_at_depth(0.0, 6000.0, 580.0)
    with pytest.raises(ValueError, match="depth"):
        casing_pressure_at_depth(900.0, -10.0, 580.0)
    with pytest.raises(ValueError, match="specific heats"):
        critical_pressure_ratio(k=1.0)


# -------------------------------------------- the constraint on allocation --


def test_a_compressor_limit_changes_the_allocation():
    """The point of the whole module. Without it the allocator can hand a well
    a rate the compressor cannot deliver, and the plan is unexecutable."""
    wells = [well("W-01"), well("W-02", uplift=7.0), well("W-03", uplift=3.0)]
    free = allocate(wells, 45000.0)

    limited = [well("W-01"), well("W-02", uplift=7.0), well("W-03", uplift=3.0)]
    limits = [injection_capacity(w.name, 950.0, 5000.0, 800.0, 0.1875, 580.0,
                                 units_per_mscf=ALLOCATION_UNITS)
              for w in limited]
    apply_injection_limits(limited, limits)
    constrained = allocate(limited, 45000.0)

    assert any(limit.binding for limit in limits)
    assert constrained.total_oil <= free.total_oil + 1e-9
    for row, w in zip(constrained.wells, limited):
        assert row.injection <= w.q_max + 1e-6


def test_no_well_is_allocated_more_than_its_valve_can_pass():
    wells = [well("W-%02d" % i, uplift=4.0 + i) for i in range(1, 5)]
    limits = [injection_capacity(w.name, 1000.0, 5500.0, 750.0, 0.125, 580.0,
                                 units_per_mscf=ALLOCATION_UNITS)
              for w in wells]
    apply_injection_limits(wells, limits)
    result = allocate(wells, 200000.0)

    capacity = {limit.well: limit.rate for limit in limits}
    for row in result.wells:
        assert row.injection <= capacity[row.name] + 1e-6, (
            "%s allocated %g against a valve capacity of %g"
            % (row.name, row.injection, capacity[row.name]))


def test_a_well_below_its_own_minimum_after_limiting_is_dropped():
    """If the compressor cannot even reach a well's minimum stable injection,
    the well is not a candidate for gas at all. Leaving q_min above q_max would
    be an invalid well; zeroing both is the honest encoding of 'cannot be
    lifted today'."""
    wells = [well("W-01", q_min=20000.0, q_max=25000.0)]
    limits = [injection_capacity("W-01", 820.0, 2500.0, 760.0, 0.0625, 580.0,
                                 units_per_mscf=ALLOCATION_UNITS)]
    apply_injection_limits(wells, limits)

    assert wells[0].q_max == 0.0
    assert wells[0].q_min == 0.0
    result = allocate(wells, 50000.0)
    assert result.wells[0].injection == 0.0


def test_limits_for_unknown_wells_are_refused():
    wells = [well("W-01")]
    stray = injection_capacity("ghost", 1200.0, 6000.0, 700.0, 0.25, 580.0)
    with pytest.raises(ValueError, match="unknown well"):
        apply_injection_limits(wells, [stray])


def test_a_bigger_port_relieves_a_valve_limited_well():
    """Which constraint binds decides what to buy. A well held back by its
    port needs wireline; one held back by compressor pressure needs a
    machine. The two must be distinguishable."""
    small = injection_capacity("A", 1100.0, 6000.0, 800.0, 0.125, 580.0)
    large = injection_capacity("A", 1100.0, 6000.0, 800.0, 0.3125, 580.0)
    assert large.rate > 5.0 * small.rate

    weak = injection_capacity("A", 900.0, 6000.0, 800.0, 0.25, 580.0)
    strong = injection_capacity("A", 1400.0, 6000.0, 800.0, 0.25, 580.0)
    assert strong.rate > weak.rate
