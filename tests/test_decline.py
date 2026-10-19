"""Tests for decline-curve analysis.

Arps has closed-form cumulatives for every case, so most of these check the
code against arithmetic rather than against its own previous output: the
exponential, harmonic and hyperbolic integrals, the terminal-decline switch,
and the time to reach a rate.

The rest check the honesty, which is the part of this module that earns its
keep. A fit to noisy history must report a reserve *range*, that range has to
narrow as history accumulates, and a fit pinned against the b ceiling has to
say so rather than quoting the ceiling as a measurement.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.decline import (
    DAYS_PER_YEAR,
    DEFAULT_TERMINAL_DECLINE,
    MAX_FITTED_B,
    ArpsDecline,
    DuongDecline,
    fit_arps,
    fit_duong,
)


# ----------------------------------------------- against the closed forms ---


def test_exponential_rate_and_cumulative_match_the_analytic_result():
    """q = qi exp(-Di t), Np = (qi - q)/Di. Both exact, so the tolerance is
    floating point rather than engineering."""
    decline = ArpsDecline(qi=1000.0, di_per_year=0.4, b=0.0, d_min_per_year=0.0)
    for t in (0.0, 0.5, 1.0, 5.0):
        assert decline.rate(t) == pytest.approx(1000.0 * math.exp(-0.4 * t), rel=1e-12)
        expected = (1000.0 - 1000.0 * math.exp(-0.4 * t)) / 0.4 * DAYS_PER_YEAR
        assert decline.cumulative(t) == pytest.approx(expected, rel=1e-12)


def test_harmonic_cumulative_is_logarithmic_not_the_hyperbolic_formula():
    """At b = 1 the hyperbolic cumulative is 0/0 and its limit is a logarithm.
    Evaluating the general formula near b = 1 instead loses precision exactly
    where harmonic wells are fitted."""
    decline = ArpsDecline(qi=800.0, di_per_year=0.5, b=1.0,
                          d_min_per_year=1e-9)
    for t in (0.5, 2.0, 10.0):
        q = 800.0 / (1.0 + 0.5 * t)
        assert decline.rate(t) == pytest.approx(q, rel=1e-10)
        expected = 800.0 / 0.5 * math.log(800.0 / q) * DAYS_PER_YEAR
        assert decline.cumulative(t) == pytest.approx(expected, rel=1e-6)


def test_hyperbolic_cumulative_matches_the_integral():
    decline = ArpsDecline(qi=1200.0, di_per_year=0.6, b=0.5,
                          d_min_per_year=0.0)
    for t in (0.25, 1.0, 4.0):
        q = decline.rate(t)
        expected = (1200.0 / ((1.0 - 0.5) * 0.6)
                    * (1.0 - (q / 1200.0) ** 0.5) * DAYS_PER_YEAR)
        assert decline.cumulative(t) == pytest.approx(expected, rel=1e-10)


def test_cumulative_agrees_with_numerically_integrating_the_rate():
    """An independent check on all three branches at once, including across
    the terminal-decline switch, where the analytic cumulative is assembled
    from two pieces and a mistake in the join would not show up anywhere
    else."""
    decline = ArpsDecline(qi=1000.0, di_per_year=0.7, b=0.9,
                          d_min_per_year=0.06)
    assert 0.0 < decline.switch_year < 20.0
    horizon = decline.switch_year + 8.0
    t = np.linspace(0.0, horizon, 400_001)
    rates = decline.rates(t)
    integral = np.trapezoid(rates, t) * DAYS_PER_YEAR
    assert decline.cumulative(horizon) == pytest.approx(float(integral), rel=1e-5)


def test_time_to_rate_inverts_the_rate_curve():
    for b, terminal in ((0.0, 0.0), (0.6, 0.05), (1.2, 0.05)):
        decline = ArpsDecline(qi=900.0, di_per_year=0.5, b=b,
                              d_min_per_year=terminal)
        for target in (700.0, 300.0, 50.0, 5.0):
            t = decline.time_to_rate(target)
            assert decline.rate(t) == pytest.approx(target, rel=1e-6), (b, target)


# ------------------------------------------------- the terminal exponential --


def test_the_switch_happens_where_the_decline_rate_reaches_the_terminal_value():
    """D(t) = Di/(1 + b Di t), so the switch is at (Di/Dmin - 1)/(b Di)."""
    decline = ArpsDecline(qi=1000.0, di_per_year=0.8, b=1.1,
                          d_min_per_year=0.08)
    expected = (0.8 / 0.08 - 1.0) / (1.1 * 0.8)
    assert decline.switch_year == pytest.approx(expected)
    assert decline.decline_at(decline.switch_year - 0.01) > 0.08
    assert decline.decline_at(decline.switch_year + 5.0) == pytest.approx(0.08)


def test_the_rate_is_continuous_across_the_switch():
    """A step at the join would put a discontinuity into every cumulative and
    every forecast built on it."""
    decline = ArpsDecline(qi=1000.0, di_per_year=0.9, b=1.3,
                          d_min_per_year=0.07)
    switch = decline.switch_year
    assert decline.rate(switch - 1e-6) == pytest.approx(
        decline.rate(switch + 1e-6), rel=1e-6)


def test_harmonic_and_above_without_a_terminal_decline_is_refused():
    """b >= 1 integrates to infinite cumulative production. A tool that books
    those reserves silently is worse than one that refuses the input."""
    for b in (1.0, 1.5, 2.0):
        with pytest.raises(ValueError, match="infinite reserves"):
            ArpsDecline(qi=1000.0, di_per_year=0.5, b=b, d_min_per_year=0.0)


def test_the_terminal_switch_makes_reserves_finite():
    unbounded = ArpsDecline(qi=1000.0, di_per_year=0.5, b=0.95,
                            d_min_per_year=1e-12)
    bounded = ArpsDecline(qi=1000.0, di_per_year=0.5, b=0.95,
                          d_min_per_year=0.05)
    assert bounded.eur(max_years=200.0) < unbounded.eur(max_years=200.0)
    assert math.isfinite(bounded.eur(max_years=1000.0))


def test_eur_stops_at_the_abandonment_rate():
    decline = ArpsDecline(qi=1000.0, di_per_year=0.5, b=0.5,
                          d_min_per_year=0.05)
    assert decline.eur(q_abandon=100.0) < decline.eur(q_abandon=10.0)
    # An abandonment rate above the initial rate means the well never pays.
    assert decline.eur(q_abandon=2000.0) == 0.0
    at_limit = decline.eur(q_abandon=100.0)
    assert at_limit == pytest.approx(
        decline.cumulative(decline.time_to_rate(100.0)))


def test_nonsense_inputs_are_refused():
    with pytest.raises(ValueError, match="initial rate"):
        ArpsDecline(qi=-5.0, di_per_year=0.5)
    with pytest.raises(ValueError, match="initial decline"):
        ArpsDecline(qi=100.0, di_per_year=-0.1)
    with pytest.raises(ValueError, match="b cannot be negative"):
        ArpsDecline(qi=100.0, di_per_year=0.5, b=-0.2)
    decline = ArpsDecline(qi=100.0, di_per_year=0.5, b=0.5)
    with pytest.raises(ValueError, match="time cannot be negative"):
        decline.rate(-1.0)
    with pytest.raises(ValueError, match="horizon"):
        decline.eur(max_years=0.0)


# ==================================================================== fits ==


def history(decline, months, noise=0.0, seed=0):
    days = np.arange(0.0, months * 30.4, 30.4)
    rates = decline.rates(days / DAYS_PER_YEAR)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        rates = rates * np.exp(rng.normal(0.0, noise, rates.size))
    return days, rates


def test_a_noiseless_history_recovers_its_own_parameters():
    """The fit is exact for fixed b because q^-b is linear in t, so a clean
    history must come back to machine precision -- anything looser would mean
    the b search, not the data, is the limit."""
    for b in (0.0, 0.3, 0.8, 1.4):
        true = ArpsDecline(qi=1200.0, di_per_year=0.55, b=b,
                           d_min_per_year=DEFAULT_TERMINAL_DECLINE)
        fit = fit_arps(*history(true, 36))
        assert fit.decline.b == pytest.approx(b, abs=0.01), b
        assert fit.decline.qi == pytest.approx(1200.0, rel=1e-3), b
        assert fit.decline.di_per_year == pytest.approx(0.55, rel=1e-3), b
        assert fit.rms_log_residual < 1e-6, b


def test_the_reserve_range_narrows_as_history_accumulates():
    """The central claim of this module. One or two years of data cannot
    distinguish the curves that matter for reserves; five years can. If the
    range did not narrow, it would be decoration.

    Asserted as short-versus-long rather than as a monotone sequence, because
    it is not monotone and the reason is worth recording: at twelve months the
    best fit is pinned against the b ceiling, so the acceptable range is
    *truncated* by the cap and comes out narrower than the twenty-four-month
    range that straddles it. A short history can therefore understate its own
    uncertainty, which is the opposite of the error this module exists to
    prevent -- and the reason the ceiling is reported as a note rather than
    left to show up in the spread.
    """
    true = ArpsDecline(qi=1200.0, di_per_year=0.55, b=0.9,
                       d_min_per_year=DEFAULT_TERMINAL_DECLINE)
    spreads = {}
    for months in (12, 24, 60):
        fit = fit_arps(*history(true, months, noise=0.08, seed=7),
                       q_abandon=20.0)
        spreads[months] = fit.eur_spread_fraction
    assert spreads[60] < 0.3, spreads
    assert spreads[60] < spreads[24], spreads
    assert spreads[60] < spreads[12], spreads


def test_the_reserve_range_contains_the_truth():
    """A range that excludes the right answer is worse than no range."""
    true = ArpsDecline(qi=1200.0, di_per_year=0.55, b=0.9,
                       d_min_per_year=DEFAULT_TERMINAL_DECLINE)
    truth = true.eur(q_abandon=20.0)
    for seed in range(6):
        fit = fit_arps(*history(true, 36, noise=0.08, seed=seed), q_abandon=20.0)
        assert fit.eur_low <= truth <= fit.eur_high, (
            seed, fit.eur_low, truth, fit.eur_high)


def test_a_short_history_says_so_and_leans_high():
    """Early production is usually transient flow, which fits a flat decline.
    The forecast is then optimistic, and a tool that does not say so is
    selling the optimism."""
    true = ArpsDecline(qi=1200.0, di_per_year=0.55, b=0.9,
                       d_min_per_year=DEFAULT_TERMINAL_DECLINE)
    short = fit_arps(*history(true, 10, noise=0.08, seed=3), q_abandon=20.0)
    long = fit_arps(*history(true, 72, noise=0.08, seed=3), q_abandon=20.0)
    assert any("months of history" in note for note in short.notes), short.notes
    assert short.eur > long.eur


def test_a_fit_against_the_b_ceiling_is_declared():
    """Reporting b = 2.00 from a bounded search reads as a measurement. It is
    the bound."""
    flat = ArpsDecline(qi=1000.0, di_per_year=0.3, b=MAX_FITTED_B + 1.5,
                       d_min_per_year=0.05)
    fit = fit_arps(*history(flat, 18), q_abandon=10.0)
    assert fit.decline.b == pytest.approx(MAX_FITTED_B, abs=1e-3)
    assert any("ceiling" in note for note in fit.notes), fit.notes


def test_downtime_is_dropped_and_reported():
    true = ArpsDecline(qi=1000.0, di_per_year=0.5, b=0.6,
                       d_min_per_year=DEFAULT_TERMINAL_DECLINE)
    days, rates = history(true, 24)
    rates[4] = 0.0
    rates[9] = 0.0
    fit = fit_arps(days, rates)
    assert fit.points_used == rates.size - 2
    assert any("downtime" in note for note in fit.notes), fit.notes
    # Dropping them must not move the answer: they were never decline.
    assert fit.decline.b == pytest.approx(0.6, abs=0.02)


def test_too_little_history_is_refused_rather_than_fitted():
    true = ArpsDecline(qi=1000.0, di_per_year=0.5, b=0.6,
                       d_min_per_year=DEFAULT_TERMINAL_DECLINE)
    days, rates = history(true, 4)
    with pytest.raises(ValueError, match="at least"):
        fit_arps(days[:3], rates[:3])


def test_the_fit_is_in_log_space_so_the_tail_is_not_ignored():
    """An absolute-error fit is dominated by the first months, where the rates
    are an order of magnitude larger. Since the tail is the part being
    extrapolated, weighting it to nothing is the wrong trade -- so the late
    points must be matched about as well as the early ones."""
    true = ArpsDecline(qi=2000.0, di_per_year=0.9, b=0.8,
                       d_min_per_year=DEFAULT_TERMINAL_DECLINE)
    days, rates = history(true, 60)
    fit = fit_arps(days, rates)
    predicted = fit.decline.rates(days / DAYS_PER_YEAR)
    relative = np.abs(predicted - rates) / rates
    assert relative[-10:].max() < 1e-4, relative[-10:]


def test_the_band_is_not_called_a_confidence_interval():
    """It is the scatter of the data about the fitted curve, which is mostly
    downtime. Reading it as forecast uncertainty is how reserves get quoted
    with a tight band around the wrong number, so the EUR range is the
    uncertainty this module stands behind."""
    true = ArpsDecline(qi=1000.0, di_per_year=0.5, b=0.7,
                       d_min_per_year=DEFAULT_TERMINAL_DECLINE)
    fit = fit_arps(*history(true, 36, noise=0.10, seed=1), q_abandon=10.0)
    t = np.array([1.0, 3.0, 10.0])
    low, middle, high = fit.band(t)
    assert np.all(low < middle) and np.all(middle < high)
    # The band is a constant ratio, because the residual is constant in log
    # space -- it does not widen with time, which is exactly why it is not a
    # forecast interval.
    ratios = high / middle
    assert ratios.max() - ratios.min() < 1e-9
    with pytest.raises(ValueError, match="confidence"):
        fit.band(t, confidence=1.0)


# =================================================================== Duong ==


def test_duong_round_trips_its_own_parameters():
    true = DuongDecline(q1=900.0, a=1.6, m=1.25)
    days = np.arange(30.0, 2000.0, 30.4)
    rates = true.rates(days)
    cumulative = np.array([true.cumulative(float(t)) for t in days])
    fit = fit_duong(days, rates, cumulative)
    assert fit.decline.m == pytest.approx(1.25, rel=1e-6)
    assert fit.decline.a == pytest.approx(1.6, rel=1e-6)
    assert fit.decline.q1 == pytest.approx(900.0, rel=1e-6)
    assert not fit.suspect


def test_duong_cumulative_is_the_relation_the_method_is_built_on():
    """q/Gp = a t^-m is not a consequence of Duong's rate equation, it is its
    premise. The two must agree or the fit is solving a different problem from
    the forecast."""
    decline = DuongDecline(q1=900.0, a=1.6, m=1.25)
    for t in (10.0, 100.0, 1000.0):
        assert decline.rate(t) / decline.cumulative(t) == pytest.approx(
            1.6 * t ** -1.25, rel=1e-10)


def test_duong_refuses_m_at_or_below_one():
    with pytest.raises(ValueError, match="m must exceed 1"):
        DuongDecline(q1=900.0, a=1.6, m=1.0)


def test_a_boundary_dominated_well_fits_duong_badly_and_says_so():
    """Duong assumes long-term linear flow, and nothing in its arithmetic
    notices when that is false.

    This was first written expecting a refusal, on the reasoning that an
    exponential decline would fit m <= 1. It does not: it fits m = 1.58 and
    returns parameters that look like parameters while missing the rates by
    27%. Fitting the wrong model is not a structural error the fit can detect,
    so the misfit is measured and reported instead -- which is also the only
    form of the check that does not refuse genuinely noisy shale data.
    """
    arps = ArpsDecline(qi=1000.0, di_per_year=0.6, b=0.0, d_min_per_year=0.0)
    days = np.arange(30.0, 2000.0, 30.4)
    fit = fit_duong(days, arps.rates(days / DAYS_PER_YEAR))
    assert fit.suspect
    assert fit.rms_log_residual > 0.2
    assert any("linear flow" in note for note in fit.notes), fit.notes
