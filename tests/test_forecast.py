"""Tests for the coupled production forecast.

Each forecast step is a full well-and-network solve, so these use small fields
and short horizons and assert *properties* rather than numbers: pressure and
rate must fall together, cumulative must match rate times time, the answer
must not depend on how the stepping got there, and the same call twice must
give the same answer.

Two of them are regression tests for bugs that produced confident nonsense.
The productivity index was being rescaled relative to its previous value
instead of its initial one, so it compounded and the field rate *rose* as the
reservoir depleted. And the forecast depleted the wells it was handed without
putting them back, so calling it twice with identical arguments returned
different reserves.

The last one is the capability the module exists for: a decline curve cannot
tell you what dropping separator pressure is worth, because separator pressure
is not one of its parameters.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.components import LiquidPipe
from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.forecast import (
    DAYS_PER_YEAR,
    Compartment,
    baseline_inflow,
    forecast,
    forecast_single_tank,
)
from phasespace.sim.materialbalance import GasTank, OilTank
from phasespace.sim.pvt_oil import OilPVT
from phasespace.sim.solver import FLOW, LIQUID, PRESSURE, Network, Node
from phasespace.sim.wells import CompositeIPR, GasBackpressureIPR, Tubing

# Loose on purpose. These tests are checking structure, and the step-size
# study that justifies the default lives in the module rather than here.
FAST = 0.30


def oil_pvt():
    return OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0,
                  rsb_scf_stb=600.0)


def field(separator_psia=250.0, ooip_stb=40e6, well_count=2):
    """A small field: wells on their own flowlines into one separator."""
    pvt = oil_pvt()
    tank = OilTank(ooip_stb=ooip_stb, initial_psia=4000.0, pvt=pvt)
    oil = LiquidProperties(api_gravity=35.0, water_cut=0.2, temperature_r=660.0,
                           gas_gravity=0.70,
                           bubble_point_psia=pvt.bubble_point_psia)
    gas = GasProperties(gravity=0.70, temperature_r=660.0)

    wells = []
    net = Network(phase=LIQUID, fluid=oil)
    net.add_node(Node(0, PRESSURE, pressure_psia=separator_psia, name="Separator"))
    for index in range(well_count):
        node = index + 1
        net.add_node(Node(node, FLOW, name="W-%d wh" % node))
        net.add_branch(LiquidPipe(a=node, b=0, diameter_in=4.0, length_mi=1.0,
                                  name="W-%d line" % node))
        wells.append(_well("W-%d" % node, node, 2.5 - 0.3 * index, pvt, oil, gas))
    return net, wells, tank


def _well(name, node, productivity_index, pvt, oil, gas):
    from phasespace.sim.integrated import CoupledWell
    return CoupledWell(
        name=name, node=node,
        ipr=CompositeIPR(reservoir_pressure_psia=4000.0,
                         bubble_point_psia=pvt.bubble_point_psia,
                         productivity_index=productivity_index),
        tubing=Tubing(depth_ft=8000.0, inner_diameter_in=2.992),
        gas_oil_ratio_scf_stb=600.0, water_cut=0.2, oil=oil, gas=gas)


def run(years=2.0, **kwargs):
    net, wells, tank = field(**{k: v for k, v in kwargs.items()
                                if k in ("separator_psia", "ooip_stb",
                                         "well_count")})
    return forecast_single_tank(
        net, wells, tank, years=years, step_days=182.625,
        produced_gor_scf_stb=600.0, economic_limit_stb_d=30.0,
        max_step_error=FAST, max_steps=40)


# ------------------------------------------------------------- the physics --


def test_pressure_and_rate_both_fall_as_the_field_produces():
    """The sign of the whole exercise. A forecast in which rate rises while
    pressure falls is not a forecast of anything."""
    result = run(years=3.0)
    pressures = [step.compartment_psia["field"] for step in result.steps]
    assert pressures == sorted(pressures, reverse=True), pressures

    # Rate is compared only while the same wells are flowing: a well dying
    # drops the field rate for a reason that is not depletion.
    constant = [step for step in result.steps if step.flowing_count == 2]
    rates = [step.field_oil_stb_d for step in constant]
    assert rates == sorted(rates, reverse=True), rates


def test_the_productivity_index_is_rescaled_from_its_initial_value():
    """Regression. The mobility correction was applied to whatever J held at
    the time, so it compounded: on an undersaturated oil, where viscosity
    genuinely falls on the way down to the bubble point, J grew about 23% per
    step and the field rate rose as the reservoir depleted. The physics was
    right and the bookkeeping was wrong.

    Checked by asserting that J never leaves the band the mobility ratio can
    justify -- the compounding version left it within two steps.
    """
    net, wells, tank = field()
    baseline = baseline_inflow(wells)
    result = run(years=3.0)
    assert result.steps
    for well in wells:
        start = baseline[well.name]["productivity_index"]
        assert 0.5 * start <= well.ipr.productivity_index <= 1.5 * start


def test_cumulative_production_matches_rate_times_time():
    """An independent check on the integration: the cumulative the forecast
    carries must be close to the trapezoid of its own reported rates. Not
    exact -- it is built from midpoint rates, and this is the trapezoid -- but
    a mismatch beyond a few percent would mean the two are not the same
    quantity."""
    result = run(years=3.0)
    steps = result.steps
    integral = 0.0
    for before, after in zip(steps, steps[1:]):
        span = after.day - before.day
        integral += 0.5 * (before.field_oil_stb_d + after.field_oil_stb_d) * span
    reported = steps[-1].cumulative["field"]
    assert reported == pytest.approx(integral, rel=0.08), (reported, integral)


def test_more_wells_on_one_tank_deplete_it_faster():
    """Interference, which is the reason wells share a compartment. Two wells
    on one tank take the pressure down faster than one does, and each sees the
    other's production as its own decline."""
    one = run(years=2.0, well_count=1)
    two = run(years=2.0, well_count=2)
    assert (two.final.compartment_psia["field"]
            < one.final.compartment_psia["field"])
    assert two.cumulative_oil_stb > one.cumulative_oil_stb


def test_a_bigger_tank_holds_its_pressure_longer():
    small = run(years=2.0, ooip_stb=20e6)
    large = run(years=2.0, ooip_stb=80e6)
    assert (large.final.compartment_psia["field"]
            > small.final.compartment_psia["field"])


# ------------------------------------------------------------- the stepping --


def test_the_step_adapts_rather_than_staying_uniform():
    """An undersaturated oil field loses a third of its pressure in the first
    months and then flattens. A uniform step either gets the early years wrong
    or spends the whole budget on the flat tail."""
    result = run(years=3.0)
    spans = [after.day - before.day
             for before, after in zip(result.steps, result.steps[1:])]
    assert len(set(round(span, 3) for span in spans)) > 1, spans
    # Compared against the longest step rather than the last one: the last is
    # clipped to the horizon, so it is short for a reason that has nothing to
    # do with the controller. Asserting against it failed on a run whose
    # stepping was working exactly as intended.
    assert spans[0] < max(spans), spans
    assert max(spans) == pytest.approx(182.625, rel=1e-6), spans


def test_a_tighter_step_target_barely_moves_the_answer():
    """The step controller aims at an indicator that overstates the real error
    by about two orders of magnitude, so tightening it should cost time and
    change little. If this ever fails the default is wrong."""
    loose = run(years=2.0)
    net, wells, tank = field()
    tight = forecast_single_tank(
        net, wells, tank, years=2.0, step_days=182.625,
        produced_gor_scf_stb=600.0, economic_limit_stb_d=30.0,
        max_step_error=0.05, max_steps=80)
    assert len(tight.steps) > len(loose.steps)
    assert tight.cumulative_oil_stb == pytest.approx(
        loose.cumulative_oil_stb, rel=0.03)


def test_the_step_count_is_capped_and_says_so():
    net, wells, tank = field()
    result = forecast_single_tank(
        net, wells, tank, years=30.0, step_days=30.0,
        produced_gor_scf_stb=600.0, economic_limit_stb_d=30.0,
        max_step_error=FAST, max_steps=4)
    assert len(result.steps) == 4
    assert any("step limit" in note
               for step in result.steps for note in step.notes)


def test_a_step_error_target_without_the_corrector_is_refused():
    """Without the corrector there is no error estimate, so there is nothing
    for the adaptive step to adapt to. Said plainly rather than quietly taking
    fixed steps while appearing to honour a tolerance."""
    net, wells, tank = field()
    compartment = Compartment(name="field", tank=tank,
                              wells=[w.name for w in wells],
                              produced_gor_scf_stb=600.0)
    with pytest.raises(ValueError, match="contradictory"):
        forecast(net, wells, [compartment], years=1.0, corrector=False)


# --------------------------------------------------------------- the events --


def test_a_well_that_cannot_lift_is_reported_with_its_date():
    """The forecast's most useful output and the one a decline curve cannot
    produce: when the well stops flowing, and why.

    These wells die at about 1,950 psia because 8,000 ft of 2.992 in tubing
    with 20% water needs more bottomhole pressure than the reservoir can
    supply at any rate -- natural-flow death, not inflow.
    """
    result = run(years=9.0)
    assert result.shut_ins, "no well died in nine years, so this field is wrong"
    years = [year for year, _name in result.shut_ins]
    assert years == sorted(years)
    assert math.isfinite(result.natural_flow_ends_year)
    assert result.natural_flow_ends_year >= years[-1]
    assert any("stopped flowing" in note for note in result.notes), result.notes

    # And the same run shows that a death is not counted as integration
    # error. A step across a discontinuity reports a near-total disagreement
    # between its explicit and midpoint estimates, and no step size integrates
    # across that -- counting it made every forecast that ran a well to death
    # declare itself unreliable. Checked here rather than in a test of its own
    # because it needs this same expensive nine-year run.
    assert result.events
    assert result.worst_step_error < 1.0
    assert all(step.step_error_fraction >= 0.0 for step in result.steps)


def test_the_forecast_stops_when_the_field_does():
    result = run(years=40.0)
    assert result.final.flowing_count == 0
    assert result.final.year < 40.0
    assert any("finished before the horizon" in note
               for step in result.steps for note in step.notes)


# ---------------------------------------------------------- the caller's wells


def test_the_same_call_twice_gives_the_same_answer():
    """Regression. A forecast depletes the wells it is given; without putting
    them back, the second call snapshotted the depleted IPRs as its starting
    point and returned 4.9% less oil from identical arguments."""
    net, wells, tank = field()
    kwargs = dict(years=2.0, step_days=182.625, produced_gor_scf_stb=600.0,
                  economic_limit_stb_d=30.0, max_step_error=FAST, max_steps=40)
    first = forecast_single_tank(net, wells, tank, **kwargs)
    second = forecast_single_tank(net, wells, tank, **kwargs)
    assert first.cumulative_oil_stb == second.cumulative_oil_stb


def test_the_wells_are_handed_back_as_they_arrived():
    net, wells, tank = field()
    before = [(well.ipr.productivity_index, well.ipr.reservoir_pressure_psia,
               well.min_rate_stb_d) for well in wells]
    forecast_single_tank(net, wells, tank, years=2.0, step_days=182.625,
                         produced_gor_scf_stb=600.0,
                         economic_limit_stb_d=30.0, max_step_error=FAST,
                         max_steps=40)
    after = [(well.ipr.productivity_index, well.ipr.reservoir_pressure_psia,
              well.min_rate_stb_d) for well in wells]
    assert before == after


# ------------------------------------------------------------- the inputs ----


def test_a_well_in_no_compartment_is_refused():
    """Nothing would deplete as it produced, so it would hold its initial
    reservoir pressure forever -- a well with infinite reserves, reported
    without comment."""
    net, wells, tank = field()
    compartment = Compartment(name="field", tank=tank, wells=[wells[0].name])
    with pytest.raises(ValueError, match="belong to no compartment"):
        forecast(net, wells, [compartment], years=1.0)


def test_a_well_in_two_compartments_is_refused():
    net, wells, tank = field()
    pvt = oil_pvt()
    other = OilTank(ooip_stb=10e6, initial_psia=4000.0, pvt=pvt)
    names = [well.name for well in wells]
    with pytest.raises(ValueError, match="drained by both"):
        forecast(net, wells,
                 [Compartment(name="a", tank=tank, wells=names),
                  Compartment(name="b", tank=other, wells=names)],
                 years=1.0)


def test_a_compartment_with_no_wells_is_refused():
    pvt = oil_pvt()
    with pytest.raises(ValueError, match="no wells"):
        Compartment(name="empty",
                    tank=OilTank(ooip_stb=1e6, initial_psia=3000.0, pvt=pvt),
                    wells=[])


def test_a_dry_gas_well_is_refused_rather_than_mis_scaled():
    """GasBackpressureIPR returns scf/day while every stage downstream carries
    STB/day, so the rate would be wrong by five orders of magnitude and would
    still solve. The fix is a gas-rate coupled path, not a conversion factor,
    so this is refused with the reason."""
    net, wells, tank = field(well_count=1)
    wells[0].ipr = GasBackpressureIPR(reservoir_pressure_psia=4000.0, c=1e-3)
    compartment = Compartment(name="field", tank=tank, wells=[wells[0].name])
    with pytest.raises(ValueError, match="dry-gas"):
        forecast(net, wells, [compartment], years=1.0)


def test_a_gas_compartment_accumulates_associated_gas():
    """A gas compartment here holds the gas produced with the oil, so its
    cumulative must be the oil times the gas-oil ratio."""
    net, wells, tank = field(well_count=1)
    gas_tank = GasTank(giip_scf=50e9, initial_psia=4000.0,
                       gas=GasProperties(gravity=0.70, temperature_r=660.0))
    compartment = Compartment(name="gas", tank=gas_tank,
                              wells=[wells[0].name])
    result = forecast(net, wells, [compartment], years=1.0, step_days=182.625,
                      max_step_error=FAST, max_steps=20)
    produced_gas = result.cumulative_gas_scf
    assert produced_gas > 0.0
    assert result.cumulative_oil_stb == 0.0
    # Every barrel carried 600 scf with it.
    steps = result.steps
    integral = 0.0
    for before, after in zip(steps, steps[1:]):
        span = after.day - before.day
        integral += 0.5 * (before.field_oil_stb_d + after.field_oil_stb_d) * span
    assert produced_gas == pytest.approx(integral * 600.0, rel=0.10)


def test_a_zero_or_negative_horizon_is_refused():
    net, wells, tank = field()
    with pytest.raises(ValueError, match="horizon must be positive"):
        forecast_single_tank(net, wells, tank, years=0.0)
    with pytest.raises(ValueError, match="step must be positive"):
        forecast_single_tank(net, wells, tank, years=1.0, step_days=0.0)


# ------------------------------------------------- what this is actually for --


def test_dropping_separator_pressure_extends_the_field():
    """The capability that justifies coupling the reservoir to the network.

    A decline curve fitted to this field's history cannot answer what a lower
    separator pressure is worth, because separator pressure is not one of its
    parameters -- it can only extrapolate the trend the field happens to be
    on. Here the same reservoir, wells and tubing at 400 psia and at 120 psia
    give materially different lives and materially different reserves, and the
    difference comes from physics that is in the model.
    """
    high = run(years=9.0, separator_psia=400.0)
    low = run(years=9.0, separator_psia=120.0)

    assert low.natural_flow_ends_year > high.natural_flow_ends_year
    assert low.cumulative_oil_stb > high.cumulative_oil_stb * 1.5
    assert low.recovery_factor > high.recovery_factor
