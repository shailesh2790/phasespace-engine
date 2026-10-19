"""Does the coverage guarantee actually hold?

Conformal prediction's selling point is a theorem, and a theorem is only
worth shipping if the implementation satisfies it. The central tests here
run the whole procedure many times over fresh random draws and check that
the realised coverage lands where the theory says, including the upper
bound -- which is the half people forget, and the half that catches an
implementation quietly producing intervals wider than it should.

The tests are deliberately adversarial about the model: coverage must hold
with a *biased* predictor, a *heteroscedastic* one, and one fitted to
completely the wrong functional form. That is the entire claim --
distribution free, model free -- and an implementation that only covers when
the model is good has silently assumed something.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from phasespace.sim.conformal import (
    MIN_CALIBRATION,
    ConformalInterval,
    StratifiedConformal,
    calibrate,
    calibrate_quantiles,
    empirical_coverage,
    required_calibration,
)


def draw(rng, n, noise=1.0, bias=0.0, heteroscedastic=False):
    x = rng.uniform(0.0, 10.0, n)
    scale = noise * (1.0 + x / 5.0) if heteroscedastic else noise
    y = 3.0 * x + 2.0 + rng.normal(0.0, scale, n)
    prediction = 3.0 * x + 2.0 + bias
    return x, y, prediction


# ------------------------------------------------- the guarantee itself ----


@pytest.mark.parametrize("confidence", [0.80, 0.90, 0.95])
def test_coverage_lands_inside_the_theoretical_band(confidence):
    """P(Y in C) >= 1-alpha, and <= 1-alpha+1/(n+1).

    Checked over 400 independent repetitions so the Monte Carlo error is
    small enough to see a systematic miss. The upper bound matters as much
    as the lower: an implementation that drops the (n+1) correction, or one
    that is simply too conservative, over-covers and produces intervals
    wider than they need to be.
    """
    rng = np.random.default_rng(7)
    n_cal = 200
    hits = []
    for _ in range(400):
        _, y_cal, p_cal = draw(rng, n_cal)
        interval = calibrate(p_cal, y_cal, confidence)
        _, y_test, p_test = draw(rng, 200)
        low, high = interval.apply(p_test)
        hits.append(empirical_coverage(low, high, y_test))

    realised = float(np.mean(hits))
    upper = confidence + 1.0 / (n_cal + 1)
    assert realised >= confidence - 0.01, (realised, confidence)
    assert realised <= upper + 0.01, (realised, upper)


def test_coverage_holds_with_a_badly_biased_model():
    """Distribution free and *model* free. A predictor 10 units off must
    still produce covering intervals -- they will simply be wide, which is
    the correct behaviour and a useful diagnostic."""
    rng = np.random.default_rng(11)
    _, y_cal, p_cal = draw(rng, 300, bias=10.0)
    interval = calibrate(p_cal, y_cal, 0.90)
    _, y_test, p_test = draw(rng, 2000, bias=10.0)
    low, high = interval.apply(p_test)

    assert empirical_coverage(low, high, y_test) >= 0.88
    assert interval.quantile > 9.0, "a biased model should give wide intervals"


def test_coverage_holds_when_the_model_has_the_wrong_shape():
    """Fitted as a straight line to a quadratic truth.

    Averaged over calibration draws, because the guarantee is marginal over
    the calibration set as well as over the test point. A single draw can
    and does undercover -- this test failed at 0.865 on one draw before it
    was written properly -- and reading one realisation as the guarantee is
    the same error as reading one well's outcome as a 90% interval.
    """
    rng = np.random.default_rng(3)

    def quadratic(n):
        x = rng.uniform(0.0, 10.0, n)
        y = 0.5 * x ** 2 + rng.normal(0.0, 1.0, n)
        return y, 5.0 * x - 8.0            # a line through it, badly

    hits = []
    for _ in range(200):
        y_cal, p_cal = quadratic(400)
        interval = calibrate(p_cal, y_cal, 0.90)
        y_test, p_test = quadratic(500)
        low, high = interval.apply(p_test)
        hits.append(empirical_coverage(low, high, y_test))

    assert float(np.mean(hits)) >= 0.89, float(np.mean(hits))
    # And the spread across draws is real, which is the point above.
    assert float(np.std(hits)) > 0.005


def test_the_n_plus_one_correction_is_present():
    """Dropping it undercovers at every sample size, and it is the single
    most commonly omitted detail in implementations of this method."""
    rng = np.random.default_rng(5)
    scores = rng.uniform(0.0, 1.0, 20)
    interval = calibrate(np.zeros(20), scores, 0.90)

    rank = math.ceil((20 + 1) * 0.90)
    assert interval.quantile == pytest.approx(float(np.sort(scores)[rank - 1]))
    # The plain empirical quantile would be smaller, i.e. undercover.
    assert interval.quantile >= float(np.quantile(scores, 0.90))


# ------------------------------------------------------------------ CQR ----


def test_cqr_keeps_the_guarantee_and_adapts_the_width():
    """The reason to prefer CQR: on heteroscedastic data a constant-width
    interval must be wide enough for the worst region everywhere, while CQR
    narrows where the data is quiet -- with the same guarantee."""
    rng = np.random.default_rng(13)

    def sample(n):
        x = rng.uniform(0.0, 10.0, n)
        scale = 1.0 + x
        y = 3.0 * x + rng.normal(0.0, scale, n)
        # A quantile model that knows the shape but not the calibration.
        return x, y, 3.0 * x - 1.2 * scale, 3.0 * x + 1.2 * scale

    _, y_cal, lo_cal, hi_cal = sample(500)
    interval = calibrate_quantiles(lo_cal, hi_cal, y_cal, 0.90)

    x_test, y_test, lo_test, hi_test = sample(4000)
    low, high = interval.apply_quantiles(lo_test, hi_test)
    assert empirical_coverage(low, high, y_test) >= 0.88

    widths = high - low
    quiet = widths[x_test < 2.0].mean()
    busy = widths[x_test > 8.0].mean()
    assert busy > 2.0 * quiet, (quiet, busy)


def test_cqr_can_tighten_an_over_conservative_model():
    """The conformal quantile is allowed to be negative. If the underlying
    quantile regression was too wide, conformal corrects it downward and the
    guarantee still holds -- an implementation that clamps at zero gives up
    efficiency for nothing."""
    rng = np.random.default_rng(17)
    y = rng.normal(0.0, 1.0, 500)
    interval = calibrate_quantiles(np.full(500, -20.0), np.full(500, 20.0),
                                   y, 0.90)
    assert interval.quantile < 0.0

    low, high = interval.apply_quantiles(np.full(2000, -20.0),
                                         np.full(2000, 20.0))
    assert (high - low)[0] < 40.0
    assert empirical_coverage(low, high, rng.normal(0.0, 1.0, 2000)) >= 0.88


def test_crossed_quantiles_are_refused():
    with pytest.raises(ValueError, match="below lower"):
        calibrate_quantiles([5.0, 6.0], [4.0, 7.0], [5.0, 6.5], 0.90)


# ------------------------------------------- what it is not allowed to claim


def test_too_few_points_is_reported_rather_than_guessed():
    """With n points the tightest expressible miscoverage is 1/(n+1). Asking
    for 95% from 10 points is asking for something the order statistics
    cannot express, and the honest answer is an unbounded interval and a
    note, not a confident number."""
    assert required_calibration(0.90) == 9
    assert required_calibration(0.95) == 19

    rng = np.random.default_rng(23)
    _, y, p = draw(rng, 10)
    interval = calibrate(p, y, 0.95)
    assert not interval.guaranteed
    assert math.isinf(interval.quantile)
    assert "not supported by" in interval.describe()
    assert "%.1f%%" % (interval.achievable_confidence * 100.0) in \
        interval.describe()


def test_a_supported_interval_says_so_plainly():
    rng = np.random.default_rng(29)
    _, y, p = draw(rng, 200)
    interval = calibrate(p, y, 0.90)
    assert interval.guaranteed
    assert "marginal coverage" in interval.describe()
    assert "90%" in interval.describe()


def test_the_guarantee_is_marginal_and_the_code_says_so():
    """The oversold claim. Pooled calibration can hold 90% overall while a
    sub-population sits far below it -- and that sub-population is exactly
    the well someone is asking about."""
    rng = np.random.default_rng(31)
    # Two regimes, one far noisier, pooled 50/50.
    quiet_y = rng.normal(0.0, 1.0, 400)
    noisy_y = rng.normal(0.0, 8.0, 400)
    truths = np.concatenate([quiet_y, noisy_y])
    predictions = np.zeros(truths.size)

    pooled = calibrate(predictions, truths, 0.90)
    low, high = pooled.apply(np.zeros(noisy_y.size))
    noisy_coverage = empirical_coverage(low, high, noisy_y)
    assert noisy_coverage < 0.85, (
        "this test needs a regime the pooled interval under-covers; got %.2f"
        % noisy_coverage)


def test_stratifying_restores_coverage_within_each_regime():
    rng = np.random.default_rng(37)
    quiet = rng.normal(0.0, 1.0, 400)
    noisy = rng.normal(0.0, 8.0, 400)
    truths = np.concatenate([quiet, noisy])
    strata = np.array(["quiet"] * 400 + ["noisy"] * 400)

    model = StratifiedConformal(0.90).fit(strata, np.zeros(truths.size), truths)
    for name, sample in (("quiet", rng.normal(0.0, 1.0, 2000)),
                         ("noisy", rng.normal(0.0, 8.0, 2000))):
        interval = model.interval_for(name)
        low, high = interval.apply(np.zeros(sample.size))
        assert empirical_coverage(low, high, sample) >= 0.87, name
        assert interval.stratum == name


def test_a_thin_stratum_falls_back_and_says_why():
    """Saying 'not enough data about wells like this' is information.
    Quoting a guarantee the data cannot support is not."""
    rng = np.random.default_rng(41)
    truths = np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(0.0, 1.0, 4)])
    strata = np.array(["common"] * 300 + ["rare"] * 4)

    model = StratifiedConformal(0.90).fit(strata, np.zeros(truths.size), truths)
    assert "common" in model.intervals
    assert "rare" not in model.intervals
    assert model.interval_for("rare") is model.fallback
    assert any("rare has 4 calibration points" in note for note in model.notes)


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="same length"):
        calibrate([1.0, 2.0], [1.0], 0.90)
    with pytest.raises(ValueError, match="confidence must be in"):
        required_calibration(1.0)
    with pytest.raises(ValueError, match="no finite"):
        calibrate([np.nan, np.nan], [np.nan, np.nan], 0.90)


def test_empirical_coverage_is_what_actually_happened():
    low = np.array([0.0, 0.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 1.0, 1.0])
    truths = np.array([0.5, 0.5, 0.5, 2.0])
    assert empirical_coverage(low, high, truths) == pytest.approx(0.75)


# ------------------------------------------- the decline application ------


def test_the_decline_band_becomes_a_guarantee():
    """What this module was built for: `ArpsFit.band` is scatter about a
    curve and says so; this is the calibrated version, and it needs other
    wells to exist at all.

    Coverage is checked on a completely separate field, because calibrating
    and scoring on the same wells would measure nothing.
    """
    from phasespace.sim.conformal import empirical_coverage
    from phasespace.sim.decline import (
        DAYS_PER_YEAR, ArpsDecline, conformal_band, fit_arps)

    def field(count, seed):
        rng = np.random.default_rng(seed)
        wells = []
        for index in range(count):
            true = ArpsDecline(qi=rng.uniform(300.0, 2500.0),
                               di_per_year=rng.uniform(0.35, 0.85),
                               b=rng.uniform(0.3, 1.2), d_min_per_year=0.05)
            days = np.arange(0, 42) * 30.44
            rates = true.rates(days / DAYS_PER_YEAR) * np.exp(
                rng.normal(0.0, 0.08, 42))
            wells.append(("W%02d" % index, days, rates))
        return wells

    interval, diagnostics = conformal_band(field(60, 1), confidence=0.90,
                                           horizon_months=12)
    assert diagnostics["wells_calibrated"] == 60
    assert interval.guaranteed
    assert 0.05 < interval.quantile < 0.6, interval.quantile

    lows, highs, truths = [], [], []
    for _name, days, rates in field(60, 99):
        cut = int(len(days) * 0.6)
        fit = fit_arps(days[:cut], rates[:cut])
        origin = days[:cut][rates[:cut] > 0.0][0]
        ahead = (days[cut:] - days[cut - 1] <= 12 * 30.44) & (rates[cut:] > 0.0)
        forecast = fit.decline.rates((days[cut:][ahead] - origin) / DAYS_PER_YEAR)
        usable = forecast > 0.0
        low, high = interval.apply(np.log(forecast[usable]))
        lows.extend(low)
        highs.extend(high)
        truths.extend(np.log(rates[cut:][ahead][usable]))

    coverage = empirical_coverage(lows, highs, truths)
    assert coverage >= 0.88, coverage
    # It errs wide, because well-months cluster and the effective sample is
    # nearer the well count than the residual count. Safe direction, but
    # empirical rather than guaranteed -- which is why it is measured here.
    assert coverage <= 0.98, coverage


def test_a_field_too_short_to_fit_and_score_is_refused():
    from phasespace.sim.decline import conformal_band

    with pytest.raises(ValueError, match="enough history"):
        conformal_band([("W1", np.arange(5) * 30.44, np.full(5, 100.0))])
