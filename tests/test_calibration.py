"""Tests for SCADA calibration and unaccounted-for-gas localisation.

These are the tests that decide whether the commercial claim is true. The
claim is: *given your operating data, this tells you how much gas you are
losing and where.* So every test here plants a known defect, generates the
measurements it would produce, and checks the method recovers it -- including
the two ways the method can be embarrassingly wrong:

  * calling normal meter noise a leak (false positive), and
  * calling pipe fouling a leak, because both lower downstream pressure.

The second is the hard one, and it is the reason the method uses metered
mass balance for size rather than pressure alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.calibration import (
    calibrate_efficiency,
    identifiability_report,
    localise_leak,
    metered_mass_balance,
)
from phasespace.sim.components import GasPipe
from phasespace.sim.fluids import GasProperties
from phasespace.sim.scada import (
    FLOW_TAG,
    PRESSURE_TAG,
    EfficiencyDefect,
    LeakDefect,
    Meter,
    daily_demand_profile,
    generate_history,
)
from phasespace.sim.solver import FLOW, GAS, PRESSURE, Network, Node

GAS_PROPS = GasProperties(gravity=0.65, temperature_r=520.0)
BASE_DEMAND = {2: 9.0e6, 3: 6.0e6}
DESIGN_EFFICIENCY = 0.92


def build():
    """A small city-gas spur: gate -> trunk -> two laterals."""
    net = Network(phase=GAS, fluid=GAS_PROPS)
    net.add_node(Node(0, PRESSURE, pressure_psia=620.0, name="city gate"))
    net.add_node(Node(1, FLOW, name="junction"))
    net.add_node(Node(2, FLOW, name="industrial"))
    net.add_node(Node(3, FLOW, name="domestic"))
    net.add_branch(GasPipe(a=0, b=1, diameter_in=10.02, length_mi=6.0, name="trunk"))
    net.add_branch(GasPipe(a=1, b=2, diameter_in=6.065, length_mi=4.0, name="lateral-ind"))
    net.add_branch(GasPipe(a=1, b=3, diameter_in=6.065, length_mi=5.0, name="lateral-dom"))
    return net


METERS = [
    Meter("PT-gate", PRESSURE_TAG, node_id=0, noise_std=0.5),
    Meter("PT-junction", PRESSURE_TAG, node_id=1, noise_std=0.5),
    Meter("PT-ind", PRESSURE_TAG, node_id=2, noise_std=0.5),
    Meter("PT-dom", PRESSURE_TAG, node_id=3, noise_std=0.5),
    Meter("FT-gate", FLOW_TAG, branch_name="trunk", noise_std=40_000.0),
    Meter("FT-ind", FLOW_TAG, node_id=2, noise_std=25_000.0),
    Meter("FT-dom", FLOW_TAG, node_id=3, noise_std=25_000.0),
]
SUPPLY_TAGS = ["FT-gate"]
OFFTAKE_TAGS = ["FT-ind", "FT-dom"]
SOURCE_TAGS = {"PT-gate": 0}
SCHEDULE = daily_demand_profile(BASE_DEMAND, swing=0.35)

# Kept small on purpose: these run a network solve per snapshot per optimiser
# step, and a leak search re-calibrates once per candidate node.
FAST = dict(duration_s=2 * 86400.0, interval_s=4 * 3600.0)
FIT = dict(source_pressure_tags=SOURCE_TAGS, max_snapshots=12)


def history_for(leak=None, defect=None, seed=7):
    return generate_history(build, SCHEDULE, METERS, leak=leak,
                            efficiency_defect=defect, seed=seed, **FAST)


# ---------------------------------------------------------------- generator --


def test_generator_produces_noisy_readings_around_the_truth():
    history = history_for()
    assert history.snapshots
    measured = history.series("PT-ind")
    truth = history.truth_series("PT-ind")
    assert len(measured) == len(truth)
    assert measured != truth, "readings are noise-free -- the generator is not testing anything"
    errors = [abs(m - t) for m, t in zip(measured, truth)]
    assert max(errors) < 5.0, "noise is implausibly large for a 0.5 psi instrument"


def test_demand_profile_actually_swings():
    """A network only reveals its parameters when it is exercised. A flat
    history would make calibration look far better than it deserves."""
    history = history_for()
    flows = history.truth_series("FT-ind")
    assert max(flows) > 1.25 * min(flows)


def test_history_exports_to_long_format_csv(tmp_path):
    history = history_for()
    path = tmp_path / "scada.csv"
    history.to_csv(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "timestamp_s,tag,value"
    assert len(lines) == 1 + len(history.snapshots) * len(METERS)


def test_meters_are_validated():
    with pytest.raises(ValueError):
        Meter("bad", "temperature", node_id=1)
    with pytest.raises(ValueError):
        Meter("bad", PRESSURE_TAG, node_id=1, noise_std=-1.0)


# -------------------------------------------------------------- calibration --


def test_calibration_recovers_design_efficiency_on_a_healthy_network():
    """With no defect, the fit must land back on the efficiency the network
    was built with. If it cannot do this, no drift it reports means anything."""
    history = history_for()
    result = calibrate_efficiency(build, history, **FIT)
    assert result.converged
    for name, efficiency in result.efficiencies.items():
        assert efficiency == pytest.approx(DESIGN_EFFICIENCY, abs=0.03), (
            "%s fitted to %.4f, expected ~%.2f" % (name, efficiency, DESIGN_EFFICIENCY))
    assert result.rms_pressure_error_psi < 2.0


def test_calibration_measures_the_size_of_a_planted_efficiency_loss():
    """The maintenance signal: a 25% efficiency loss on one segment must come
    back as a 25% drift on that segment and near-zero on the others."""
    history = history_for(defect=EfficiencyDefect("lateral-dom", 0.75))
    result = calibrate_efficiency(build, history, **FIT)
    assert result.converged
    drift = result.drift()
    assert drift["lateral-dom"] == pytest.approx(-0.25, abs=0.04)
    assert abs(drift["trunk"]) < 0.05
    assert abs(drift["lateral-ind"]) < 0.05


def test_calibration_needs_named_branches():
    """A fitted efficiency that cannot be attributed to a named segment is
    not actionable, so this is refused rather than silently reported."""
    def unnamed():
        net = build()
        net.branches[0].name = ""
        return net

    with pytest.raises(ValueError):
        calibrate_efficiency(unnamed, history_for(), **FIT)


def test_identifiability_flags_a_flat_history():
    """All snapshots at one operating point cannot separate efficiency from
    demand. Saying so beforehand is the difference between a tool an engineer
    trusts twice and one they trust once."""
    flat = generate_history(build, lambda t, n: BASE_DEMAND.get(n, 0.0), METERS,
                            duration_s=86400.0, interval_s=6 * 3600.0, seed=3)
    report = identifiability_report(build, flat)
    assert not report.supported
    assert report.demand_swing < 0.05
    assert any("one operating point" in n for n in report.notes)


def test_identifiability_is_not_fooled_by_meter_noise():
    """Counting distinct readings to measure excitation looks reasonable and
    is wrong: noise alone makes every reading unique, so a dead-flat history
    scores as maximally informative. This check exists because the first
    implementation did exactly that."""
    noisy_flat_meters = [
        Meter(m.tag, m.kind, node_id=m.node_id, branch_name=m.branch_name,
              noise_std=m.noise_std * 4.0)
        for m in METERS
    ]
    flat = generate_history(build, lambda t, n: BASE_DEMAND.get(n, 0.0),
                            noisy_flat_meters, duration_s=86400.0,
                            interval_s=3 * 3600.0, seed=11)
    report = identifiability_report(build, flat)
    assert not report.supported, (
        "noise was mistaken for genuine demand variation (swing %.3f)"
        % report.demand_swing)


def test_identifiability_accepts_a_well_exercised_history():
    report = identifiability_report(build, history_for(),
                                    source_pressure_tags=SOURCE_TAGS)
    assert report.supported
    assert report.n_pressure_tags >= report.n_parameters
    assert report.demand_swing > 0.05


def test_a_source_gauge_does_not_count_as_a_target():
    """A source's own pressure gauge sets a boundary condition, so the model
    reproduces it exactly by construction and it constrains nothing. Counting
    it was a real bug: a network with gauges at the gate and two delivery
    points looked like three targets for three parameters, and the
    under-determined fit invented a 13% efficiency loss on a healthy trunk."""
    history = history_for()
    counted = identifiability_report(build, history)
    excluded = identifiability_report(build, history, source_pressure_tags=SOURCE_TAGS)
    assert excluded.n_pressure_tags == counted.n_pressure_tags - len(SOURCE_TAGS)


def test_under_instrumented_network_is_refused_not_fitted_anyway():
    """Two interior gauges cannot pin three segment efficiencies. The fit
    slides along a ridge, trading one segment against another, and returns a
    confident-looking point on it. Saying so beforehand is the whole purpose
    of this check."""
    thin = [m for m in METERS if m.tag != "PT-junction"]
    history = generate_history(build, SCHEDULE, thin, seed=7, **FAST)
    report = identifiability_report(build, history, source_pressure_tags=SOURCE_TAGS)
    assert not report.supported
    assert report.n_pressure_tags < report.n_parameters
    assert any("under-determined" in n for n in report.notes)


def test_a_fully_instrumented_healthy_network_shows_no_spurious_drift():
    """The false-positive that the missing junction gauge produced: a healthy
    network reporting double-digit fouling on one segment offset by gains on
    the others. With every node gauged, drift must stay near zero."""
    result = calibrate_efficiency(build, history_for(), **FIT)
    for name, value in result.drift().items():
        assert abs(value) < 0.05, "%s drifted %+.1f%% on a healthy network" % (
            name, 100 * value)


# ----------------------------------------------------------- mass balance ---


def test_healthy_network_balances():
    balance = metered_mass_balance(history_for(), SUPPLY_TAGS, OFFTAKE_TAGS)
    assert abs(balance.imbalance_fraction) < 0.005


def test_leak_shows_up_in_the_metered_balance_at_the_right_size():
    """The size claim, and the one that does not depend on the model being
    right: it is arithmetic on instruments."""
    history = history_for(leak=LeakDefect(node_id=3, rate_scfd=900_000.0))
    balance = metered_mass_balance(history, SUPPLY_TAGS, OFFTAKE_TAGS)
    assert balance.imbalance_scfd == pytest.approx(900_000.0, rel=0.05)
    assert balance.imbalance_fraction > 0.05


def test_efficiency_loss_does_not_break_mass_balance():
    """The asymmetry the whole method rests on. Fouling lowers pressure but
    conserves mass; if this ever failed, leak and fouling would be
    indistinguishable and the product claim would collapse."""
    history = history_for(defect=EfficiencyDefect("lateral-dom", 0.75))
    balance = metered_mass_balance(history, SUPPLY_TAGS, OFFTAKE_TAGS)
    assert abs(balance.imbalance_fraction) < 0.005


# ------------------------------------------------------------- leak search --


def test_no_leak_is_reported_on_a_healthy_network():
    """The false-positive test. A tool that cries leak on meter noise gets
    switched off after the second wasted crew dispatch."""
    finding = localise_leak(build, history_for(), SUPPLY_TAGS, OFFTAKE_TAGS, **FIT)
    assert not finding.detected
    assert finding.notes


def test_fouling_is_not_reported_as_a_leak():
    """The expensive false positive: fouling lowers downstream pressure
    exactly as a leak does. Any method watching pressure alone calls this a
    leak and sends a crew to dig up a healthy pipe."""
    history = history_for(defect=EfficiencyDefect("lateral-dom", 0.75))
    finding = localise_leak(build, history, SUPPLY_TAGS, OFFTAKE_TAGS, **FIT)
    assert not finding.detected


def test_leak_is_detected_sized_and_located():
    history = history_for(leak=LeakDefect(node_id=3, rate_scfd=900_000.0))
    finding = localise_leak(build, history, SUPPLY_TAGS, OFFTAKE_TAGS, **FIT)
    assert finding.detected
    assert finding.imbalance.imbalance_scfd == pytest.approx(900_000.0, rel=0.05)
    assert finding.best.node_id == 3, (
        "ranked %s" % [(h.node_id, round(h.rms_pressure_error_psi, 3))
                       for h in finding.ranked])
    # The true location explains the data better than assuming no leak.
    assert finding.best.rms_pressure_error_psi < finding.baseline_rms_psi


def test_the_true_location_leaves_efficiencies_undistorted():
    """The sharpest discriminator, and the reason efficiency is re-fitted
    inside each hypothesis rather than once beforehand: at the true node the
    pipe efficiencies come back at design, because the leak explains the
    pressures on its own. At a wrong node the optimiser has to bend them to
    compensate."""
    history = history_for(leak=LeakDefect(node_id=3, rate_scfd=900_000.0))
    finding = localise_leak(build, history, SUPPLY_TAGS, OFFTAKE_TAGS, **FIT)
    best = finding.best
    others = [h for h in finding.ranked if h.node_id != best.node_id]
    assert best.efficiency_distortion < 0.05
    assert all(h.efficiency_distortion > best.efficiency_distortion for h in others)


def test_a_leak_and_fouling_at_once_are_separated():
    """The case that decides whether this is a product. A real network has
    both: gas going missing somewhere and a segment quietly fouling. The
    method must report the leak's size and location *and* attribute the
    pressure loss on the fouled segment separately."""
    history = history_for(leak=LeakDefect(node_id=2, rate_scfd=700_000.0),
                          defect=EfficiencyDefect("trunk", 0.85))
    finding = localise_leak(build, history, SUPPLY_TAGS, OFFTAKE_TAGS, **FIT)
    assert finding.detected
    assert finding.imbalance.imbalance_scfd == pytest.approx(700_000.0, rel=0.06)
    assert finding.best.node_id == 2, (
        "ranked %s" % [(h.node_id, round(h.rms_pressure_error_psi, 3))
                       for h in finding.ranked])
    # And the fouling is still visible in that hypothesis's fitted efficiencies.
    assert finding.best.efficiencies["trunk"] < 0.9 * DESIGN_EFFICIENCY


def test_a_bigger_leak_is_easier_to_see():
    previous = None
    for rate in (600_000.0, 1_200_000.0, 2_000_000.0):
        history = history_for(leak=LeakDefect(node_id=3, rate_scfd=rate))
        balance = metered_mass_balance(history, SUPPLY_TAGS, OFFTAKE_TAGS)
        if previous is not None:
            assert balance.imbalance_fraction > previous
        previous = balance.imbalance_fraction


def test_size_is_reported_even_when_location_is_undeterminable():
    """With no interior pressure instruments the size still stands -- it is
    metered arithmetic -- but the location does not. Reporting a location
    anyway would be inventing precision the data does not contain."""
    flow_only = [m for m in METERS if m.kind == FLOW_TAG]
    history = generate_history(build, SCHEDULE, flow_only,
                               leak=LeakDefect(node_id=3, rate_scfd=900_000.0),
                               seed=7, **FAST)
    finding = localise_leak(build, history, SUPPLY_TAGS, OFFTAKE_TAGS,
                            source_pressure_tags={}, max_snapshots=12)
    assert finding.detected
    assert finding.best is None
    assert any("not determinable" in n for n in finding.notes)


def test_a_biased_meter_is_the_known_false_positive():
    """A meter reading 3% low is arithmetically identical to a 3% leak on a
    mass balance. The tool cannot distinguish them and must not pretend to --
    this test documents the limitation rather than asserting it away."""
    biased = []
    for meter in METERS:
        if meter.tag == "FT-ind":
            biased.append(Meter(meter.tag, meter.kind, node_id=meter.node_id,
                                branch_name=meter.branch_name,
                                noise_std=meter.noise_std, bias_factor=0.97))
        else:
            biased.append(meter)
    history = generate_history(build, SCHEDULE, biased, seed=7, **FAST)
    balance = metered_mass_balance(history, SUPPLY_TAGS, OFFTAKE_TAGS)
    # There is no leak, yet the imbalance is real and looks like one.
    assert balance.imbalance_fraction > 0.01
