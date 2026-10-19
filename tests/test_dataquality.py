"""What the quality gate catches, and -- as carefully -- what it does not.

The checks here are cheap to write and easy to write badly. A frozen-tag
test that fires on every quantised transmitter, or a unit test that converts
on a guess, makes the data worse while reporting that it made it better. So
most of these tests are about the false positives rather than the true ones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.dataquality import (
    MIN_SAMPLES,
    check,
    check_tests,
    check_tag,
    require_usable,
)
from phasespace.sim.units import PSI_PER_BAR, PSI_PER_KPA


def quantised(values, step):
    """What a transmitter of this resolution would have reported."""
    return np.round(np.asarray(values) / step) * step


# ------------------------------------------------------------- the frozen --


def test_a_latched_transmitter_is_blocked():
    quality = check_tag("PI-101", np.full(200, 1013.7), "pressure_psia")
    assert not quality.usable
    assert [f.code for f in quality.findings] == ["frozen"]


def test_a_quantised_but_live_tag_is_not_called_frozen():
    """The false positive that matters. A steady well on a 12-bit
    transmitter over a 1500 psi range moves in 0.37 psi steps and repeats
    constantly; a run-length test that ignores resolution flags it, and the
    engineer learns to ignore the report."""
    rng = np.random.default_rng(7)
    step = 1500.0 / 4096.0
    truth = 1000.0 + np.cumsum(rng.normal(0.0, 0.05, 600))
    quality = check_tag("PI-102", quantised(truth, step), "pressure_psia")
    assert quality.usable, quality.describe()
    assert not any(f.code == "frozen" for f in quality.findings)


def test_a_long_flat_run_inside_a_live_window_is_amber_not_red():
    """A transmitter that stuck for an hour and recovered is a different
    problem from one that is dead, and blocking the whole window for it
    throws away the part that is good."""
    rng = np.random.default_rng(11)
    values = 800.0 + rng.normal(0.0, 3.0, 400)
    values[100:260] = 800.0
    quality = check_tag("PI-103", values, "pressure_psia")
    assert quality.severity == "amber"
    assert quality.usable
    assert any(f.code == "stale" for f in quality.findings)


def walk(seed, noise_per_step, step=0.366, samples=600):
    """A random walk as a transmitter of this resolution would report it."""
    rng = np.random.default_rng(seed)
    truth = 1000.0 + np.cumsum(rng.normal(0.0, noise_per_step * step, samples))
    return quantised(truth, step)


def test_the_stale_test_separates_live_from_stuck_where_it_claims_to():
    """The thresholds are not taste. Above RESOLVED_MOVEMENT the live and
    stuck populations have to stay on opposite sides of STALE_RUN_RATIO,
    and this is the sweep that says they do."""
    for noise_per_step in (0.6, 1.0, 3.0, 10.0):
        for seed in range(6):
            live = walk(seed, noise_per_step)
            assert not any(f.code == "stale" for f
                           in check_tag("live", live, "pressure_psia").findings), (
                "false positive at noise/step %s seed %s" % (noise_per_step, seed))

            stuck = live.copy()
            stuck[200:360] = stuck[200]
            assert any(f.code == "stale" for f
                       in check_tag("stuck", stuck, "pressure_psia").findings), (
                "missed a 160-sample stick at noise/step %s seed %s"
                % (noise_per_step, seed))


def test_a_tag_too_coarse_to_judge_says_so_rather_than_guessing():
    """Below RESOLVED_MOVEMENT the two populations invert -- a live tag
    scored 110x and a stuck one 10x -- so a verdict either way would be
    worse than none."""
    quality = check_tag("PI-106", walk(0, 0.1), "pressure_psia")
    assert any(f.code == "unresolved" for f in quality.findings)
    assert not any(f.code in ("frozen", "stale") for f in quality.findings)
    assert quality.usable


def test_freezing_is_not_judged_on_too_short_a_window():
    quality = check_tag("PI-104", np.full(MIN_SAMPLES - 1, 500.0),
                        "pressure_psia")
    assert not any(f.code == "frozen" for f in quality.findings)
    assert any(f.code == "short" for f in quality.findings)


# --------------------------------------------------------- the impossible --


def test_a_water_cut_above_one_is_impossible():
    values = np.full(100, 0.4)
    values[:20] = 1.6
    quality = check_tag("WC-01", values, "water_cut")
    assert not quality.usable
    assert any("above 1" in f.detail for f in quality.findings)


def test_a_few_bad_readings_do_not_block_the_window():
    """Instrument glitches are ordinary. Blocking a 400-sample window for
    two spikes means nobody can calibrate anything."""
    values = np.full(400, 250.0)
    values[17] = -40.0
    values[203] = -40.0
    quality = check_tag("PI-105", values, "pressure_psia")
    assert quality.usable
    assert any(f.code == "impossible" and f.severity == "amber"
               for f in quality.findings)


def test_negative_absolute_pressure_is_caught_but_gauge_pressure_is_allowed():
    """-5 psig is a vacuum and perfectly real; -5 psia is not a pressure."""
    absolute = check_tag("A", np.full(100, -5.0), "pressure_psia")
    gauge = check_tag("G", np.full(100, -5.0), "pressure_psig")
    assert any(f.code == "impossible" for f in absolute.findings)
    assert not any(f.code == "impossible" for f in gauge.findings)


# --------------------------------------------------------------- the units --


def test_a_pressure_tag_in_kpa_is_caught_and_named():
    rng = np.random.default_rng(3)
    psia = 1000.0 + rng.normal(0.0, 10.0, 200)
    quality = check_tag("PI-201", psia / PSI_PER_KPA, "pressure_psia")
    assert not quality.usable
    finding = next(f for f in quality.findings if f.code == "units")
    assert "kPa" in finding.detail


def test_bar_and_ksc_are_reported_together_rather_than_guessed_between():
    """They differ by 2%. One tag cannot separate them, and a model that
    picks one is wrong by 2% on every pressure in the field while looking
    certain."""
    rng = np.random.default_rng(5)
    quality = check_tag("PI-202", 7.0 + rng.normal(0.0, 0.05, 200),
                        "pressure_psia")
    finding = next(f for f in quality.findings if f.code == "units")
    assert "bar" in finding.detail and "kg/cm2" in finding.detail
    assert "cannot be told apart" in finding.detail


def test_a_unit_error_that_stays_inside_the_plausible_band_is_missed():
    """The recorded limit of this check, asserted so it does not become
    folklore. 1000 psia recorded in bar reads 68.9, which is a believable
    low-pressure psia reading. Separating those needs a second constraint --
    a known setpoint or a hydrostatic check -- which is not available from
    one tag's magnitude."""
    rng = np.random.default_rng(13)
    psia = 1000.0 + rng.normal(0.0, 5.0, 200)
    quality = check_tag("PI-203", psia / PSI_PER_BAR, "pressure_psia")
    assert quality.usable
    assert not any(f.code == "units" for f in quality.findings)


def test_rates_are_never_unit_checked():
    """A stripper well making 3 stb/d is not a unit error, and flagging it
    as one would fire on the wells that need the attention most."""
    quality = check_tag("FI-01", np.full(200, 3.0), "rate_stb_d")
    assert not any(f.code in ("units", "out_of_range")
                   for f in quality.findings)


def test_a_pressure_that_matches_no_conversion_is_amber_not_named():
    quality = check_tag("PI-204", np.linspace(90_000.0, 95_000.0, 200),
                        "pressure_psig")
    codes = {f.code for f in quality.findings}
    assert "out_of_range" in codes or "impossible" in codes
    assert "units" not in codes


# ---------------------------------------------------------------- the gaps --


def test_missing_samples_are_reported_and_nothing_is_filled_in():
    values = np.full(200, 300.0)
    values[50:70] = np.nan
    quality = check_tag("PI-301", values, "pressure_psia")
    gaps = next(f for f in quality.findings if f.code == "gaps")
    assert gaps.samples == 20
    assert quality.finite == 180


def test_a_mostly_missing_tag_is_blocked():
    values = np.full(200, np.nan)
    values[:40] = 300.0
    assert not check_tag("PI-302", values, "pressure_psia").usable


def test_shut_in_zeros_are_recorded_as_real_not_as_gaps():
    """A shut-in day is a measurement. Pooling it with missing data is what
    makes a forecast quietly lose a well's downtime."""
    rate = np.full(200, 450.0)
    rate[100:140] = 0.0
    quality = check_tag("FI-02", rate, "rate_stb_d")
    zeros = next(f for f in quality.findings if f.code == "zeros")
    assert zeros.samples == 40
    assert zeros.severity == "info"
    assert quality.usable
    assert not any(f.code == "gaps" for f in quality.findings)


def test_an_empty_tag_is_blocked_rather_than_passed_as_clean():
    assert not check_tag("PI-303", [], "pressure_psia").usable


# ---------------------------------------------------------------- the gate --


def test_one_red_tag_blocks_the_whole_set():
    report = check(
        {"good": np.linspace(900.0, 1000.0, 200),
         "latched": np.full(200, 640.0)},
        kinds={"good": "pressure_psia", "latched": "pressure_psia"})
    assert not report.may_fit
    assert [t.tag for t in report.blocked] == ["latched"]


def test_a_clean_set_passes_the_gate():
    rng = np.random.default_rng(17)
    report = check(
        {"PI-1": 1200.0 + rng.normal(0.0, 8.0, 300),
         "TI-1": 140.0 + rng.normal(0.0, 1.0, 300)},
        kinds={"PI-1": "pressure_psia", "TI-1": "temperature_f"})
    assert report.may_fit, report.summary()
    require_usable(report)


def test_the_refusal_names_the_tag_and_the_reason():
    """An error that says "data quality check failed" sends the engineer
    looking through a historian by hand."""
    report = check({"PI-9": np.full(200, 410.0)},
                   kinds={"PI-9": "pressure_psia"})
    with pytest.raises(ValueError, match="PI-9"):
        require_usable(report, action="calibrate")
    with pytest.raises(ValueError, match="latched"):
        require_usable(report, action="calibrate")


def test_checking_is_deterministic():
    """The gate is part of the audit trail, so two runs over one export must
    produce the same verdict and the same wording."""
    rng = np.random.default_rng(23)
    values = 700.0 + rng.normal(0.0, 4.0, 300)
    values[20:60] = np.nan
    first = check_tag("PI-401", values, "pressure_psia")
    second = check_tag("PI-401", values, "pressure_psia")
    assert first.describe() == second.describe()


# ------------------------------------------------------- the well-test sheet --


def sheet(pressures, rates=None):
    from phasespace.sim.wellcalibration import WellTest
    rates = rates or [400.0 + 90.0 * i for i in range(len(pressures))]
    return [WellTest(rate_stb_d=r, wellhead_psia=p, label="T%d" % i)
            for i, (r, p) in enumerate(zip(rates, pressures))]


def test_a_carried_forward_pressure_column_is_blocked():
    """Six tests, three pressures. The rates are genuine and the residuals
    will be small, which is exactly why this has to be caught before the
    fit rather than read off it afterwards."""
    report = check_tests(sheet([1200.0, 1200.0, 1050.0, 1050.0, 900.0, 900.0]))
    assert not report.may_fit
    assert report.tag("wellhead_psia").severity == "red"
    assert "carried forward" in report.tag("wellhead_psia").describe()


def test_identifiability_passes_the_sheet_that_this_catches():
    """The gap being closed. `identifiability` asks about rate spread, which
    this sheet has; the pressures are the copied column."""
    from phasespace.sim.wellcalibration import identifiability
    tests = sheet([1200.0, 1200.0, 1050.0, 1050.0, 900.0, 900.0])
    assert identifiability(tests).supported
    assert not check_tests(tests).may_fit


def test_genuinely_distinct_tests_pass():
    report = check_tests(sheet([1240.0, 1160.0, 1085.0, 1010.0, 940.0, 870.0]))
    assert report.may_fit, report.summary()


def test_repeated_rates_are_recorded_but_not_blocking():
    """Two tests a month apart on a stable well read the same rate. That is
    a real measurement, not a copied one."""
    tests = sheet([1240.0, 1160.0, 1085.0, 1010.0],
                  rates=[500.0, 500.0, 610.0, 720.0])
    report = check_tests(tests)
    assert report.may_fit
    assert report.tag("rate_stb_d").severity == "info"


def test_a_low_pressure_test_sheet_in_ksc_is_caught():
    """A separator-pressure column in kg/cm2 lands below the band a field
    operates in, so the ratio test sees it."""
    report = check_tests(sheet([7.4, 7.1, 6.8, 6.5, 6.2, 5.9]))
    assert not report.may_fit
    assert "kg/cm2" in report.tag("wellhead_psia").describe()


def test_a_flowing_test_sheet_in_ksc_is_missed_for_the_documented_reason():
    """The same limitation as the single-tag case, reached from the other
    direction: 87 kg/cm2 is 1237 psia, and read as 73.6 psia it is an
    entirely believable low-pressure well. Nothing in the column itself
    gives it away."""
    report = check_tests(sheet([87.0, 81.5, 76.2, 71.0, 66.1, 61.3]))
    assert report.may_fit
    assert not any(f.code == "units"
                   for f in report.tag("wellhead_psia").findings)


def test_an_empty_sheet_is_not_a_pass_or_a_crash():
    assert check_tests([]).tags == []
