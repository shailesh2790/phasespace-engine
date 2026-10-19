"""Run a cross-section of the engine and print every number it produced.

Invoked as a subprocess by `test_reproducibility.py`, twice, under different
PYTHONHASHSEED values. Anything that depends on set or dict iteration order,
on address-based hashing, or on the environment shows up as a differing line.

Printed at full float precision on purpose: `repr` of a float round-trips
exactly, so a difference in the last bit is visible rather than rounded away.
A reproducibility guarantee that only holds to six decimal places is not one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phasespace.sim.components import GasPipe, LiquidPipe
from phasespace.sim.decline import ArpsDecline, fit_arps
from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.materialbalance import GasTank, OilTank, fit_gas_balance
from phasespace.sim.pvt_oil import OilPVT
from phasespace.sim.solver import FLOW, GAS, LIQUID, PRESSURE, Network, Node, solve
from phasespace.sim.wellcalibration import WellTest, calibrate_well
from phasespace.sim.wells import CompositeIPR, Tubing, nodal_analysis

GAS_65 = GasProperties(gravity=0.65, temperature_r=540.0)


def emit(label, value):
    print("%s=%r" % (label, value))


def gas_network():
    net = Network(phase=GAS, fluid=GAS_65)
    net.add_node(Node(0, PRESSURE, pressure_psia=900.0, name="src"))
    for index in range(1, 10):
        net.add_node(Node(index, FLOW, withdrawal=3.0e5 + 1.7e4 * index,
                          name="n%d" % index))
    edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (3, 6),
             (2, 7), (7, 8), (8, 9), (6, 9)]
    for a, b in edges:
        net.add_branch(GasPipe(a=a, b=b, diameter_in=6.0, length_mi=1.4,
                               name="p%d%d" % (a, b)))
    result = solve(net)
    emit("net.converged", result.converged)
    emit("net.iterations", result.iterations)
    for node_id in sorted(result.pressures):
        emit("net.p[%d]" % node_id, result.pressures[node_id])
    for component, flow in sorted(result.flows.items(),
                                  key=lambda kv: getattr(kv[0], "name", "")):
        emit("net.q[%s]" % getattr(component, "name", "?"), flow)


def well():
    oil = LiquidProperties(api_gravity=35.0, water_cut=0.25,
                           temperature_r=660.0, gas_gravity=0.70,
                           bubble_point_psia=2700.0)
    gas = GasProperties(gravity=0.70, temperature_r=660.0)
    tubing = Tubing(depth_ft=8000.0, inner_diameter_in=2.992)
    ipr = CompositeIPR(3500.0, 2700.0, 2.0)
    for wellhead in (150.0, 400.0, 800.0):
        solved = nodal_analysis(ipr, tubing, wellhead, 600.0, 0.25, oil, gas)
        emit("well.q[%g]" % wellhead, solved.liquid_rate_stb_d)
        emit("well.pwf[%g]" % wellhead, solved.bottomhole_pressure_psia)


def pvt_and_balance():
    pvt = OilPVT(api=35.0, gas_gravity=0.70, temperature_f=200.0,
                 rsb_scf_stb=600.0)
    emit("pvt.pb", pvt.bubble_point_psia)
    for pressure in (800.0, 2000.0, 3500.0):
        emit("pvt.rs[%g]" % pressure, pvt.rs(pressure))
        emit("pvt.bo[%g]" % pressure, pvt.bo(pressure))
        emit("pvt.mu[%g]" % pressure, pvt.viscosity(pressure))

    tank = OilTank(ooip_stb=40e6, initial_psia=4000.0, pvt=pvt)
    for fraction in (0.01, 0.05, 0.10):
        emit("tank.p[%g]" % fraction,
             tank.pressure_at(fraction * 40e6, 900.0))

    gas_tank = GasTank(giip_scf=100e9, initial_psia=4000.0, gas=GAS_65)
    cumulatives = [0.0, 10e9, 25e9, 45e9, 60e9]
    pressures = [gas_tank.pressure_at(g) for g in cumulatives]
    for value in pressures:
        emit("gastank.p", value)
    fit = fit_gas_balance(pressures, cumulatives, GAS_65)
    emit("gasfit.giip", fit.giip_scf)
    emit("gasfit.drive", fit.drive)


def decline():
    true = ArpsDecline(qi=1200.0, di_per_year=0.55, b=0.8,
                       d_min_per_year=0.05)
    days = [i * 30.44 for i in range(30)]
    rates = [true.rate(d / 365.25) for d in days]
    fit = fit_arps(days, rates, q_abandon=20.0)
    emit("arps.b", fit.decline.b)
    emit("arps.qi", fit.decline.qi)
    emit("arps.di", fit.decline.di_per_year)
    emit("arps.eur", fit.eur)
    emit("arps.eur_low", fit.eur_low)
    emit("arps.eur_high", fit.eur_high)


def calibration():
    oil = LiquidProperties(api_gravity=35.0, water_cut=0.25,
                           temperature_r=660.0, gas_gravity=0.70,
                           bubble_point_psia=2700.0)
    gas = GasProperties(gravity=0.70, temperature_r=660.0)
    tubing = Tubing(depth_ft=8000.0, inner_diameter_in=2.992)
    tests = [
        WellTest(rate_stb_d=3807.4, wellhead_psia=150.0, water_cut=0.25,
                 gas_oil_ratio_scf_stb=600.0, bottomhole_psia=2385.4),
        WellTest(rate_stb_d=3303.0, wellhead_psia=300.0, water_cut=0.25,
                 gas_oil_ratio_scf_stb=600.0, bottomhole_psia=2543.1),
        WellTest(rate_stb_d=2412.9, wellhead_psia=500.0, water_cut=0.25,
                 gas_oil_ratio_scf_stb=600.0, bottomhole_psia=2804.0),
    ]
    result = calibrate_well(CompositeIPR(3500.0, 2700.0, 2.0), tubing, tests,
                            oil, gas)
    for key in sorted(result.parameters):
        emit("calib.%s" % key, result.parameters[key])
    emit("calib.rms_after", result.rms_after)
    for pair in sorted(result.correlations):
        emit("calib.corr[%s]" % pair, result.correlations[pair])


def liquid_network():
    oil = LiquidProperties(api_gravity=32.0, water_cut=0.2,
                           temperature_r=620.0)
    net = Network(phase=LIQUID, fluid=oil)
    net.add_node(Node(0, PRESSURE, pressure_psia=250.0, name="sep"))
    for index in (1, 2, 3):
        net.add_node(Node(index, FLOW, withdrawal=-1200.0 * index,
                          name="w%d" % index))
        net.add_branch(LiquidPipe(a=index, b=0, diameter_in=4.0,
                                  length_mi=1.0 + 0.3 * index,
                                  name="fl%d" % index))
    result = solve(net)
    emit("liq.converged", result.converged)
    for node_id in sorted(result.pressures):
        emit("liq.p[%d]" % node_id, result.pressures[node_id])


if __name__ == "__main__":
    gas_network()
    well()
    pvt_and_balance()
    decline()
    calibration()
    liquid_network()
