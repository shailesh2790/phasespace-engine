"""What is wrong with this tag, and may a model use it.

PRD FR-2: flag frozen tags, unit errors, impossible values and gaps before
any model uses them, and show what was corrected and why.

Every other module in this engine assumes its inputs are numbers somebody
meant. On a real historian that assumption is false several times a day, and
the failures are not noisy -- they are *plausible*. A transmitter that has
latched still reports a believable pressure. A tag configured in kPa reports
a believable number too, just one seven times smaller than the engineer
reading the screen assumes. Neither looks like missing data, and both will
be fitted to, optimised on and reported from without complaint.

So this module's job is not to clean data. It is to decide what the engine
is **allowed to use**, and to say why in terms an engineer can check. The
distinction matters because the usual approach -- silently repair and carry
on -- destroys the evidence that something is wrong with the instrument,
which is often worth more than the reading.

**Four checks, and the trap in each one.**

*Frozen.* The naive test is "the same value N times in a row". It is wrong,
and wrong in the direction that matters: a stable well on a coarsely
quantised transmitter repeats legitimately, so the naive test fires on the
best-behaved wells in the field. A live signal sitting on a quantisation
step still *transitions* between adjacent steps; a latched one never does.
`_quantisation` estimates the step from the data, and the run length is then
judged against what that step and the tag's own variability would produce.

*Units.* Physics gives the plausible range; the ratio to it gives the
candidate unit. What physics will not give is a unique answer, and the
honest output says so: bar and kg/cm2 differ by 2%, which no amount of
looking at one pressure tag can resolve. Reporting "about 14x out, consistent
with bar or ksc, cannot distinguish" is useful. Guessing one of them, writing
it into the model, and being wrong by 2% on every pressure in the field is
not. Nothing here converts anything. It reports.

The bound on this check is worth stating plainly, because it is not small.
It can only see a unit error that pushes the tag *out of* the band a field
operates in. A well flowing at 1000 psia recorded in bar reads 68.9, which
is a thoroughly believable low-pressure psia reading, and no amount of
looking at that one tag will reveal it. Catching that case needs a second
constraint -- a known setpoint, or hydrostatic consistency down a wellbore
with a survey -- which belongs with the field model and not here.
`test_dataquality.py` asserts the miss, so it stays a recorded limit rather
than becoming folklore about what the gate catches.

*Impossible.* Bounds that come from the physics rather than from a style
guide: absolute pressure below vacuum, a water cut outside [0, 1], a
temperature below the coldest Joule-Thomson expansion that could produce it,
a negative rate on a tag that cannot flow backwards.

*Gaps.* The trap is that a shut-in well and a dead sensor look identical at
the tag -- both report nothing, or zero -- and the difference decides whether
interpolating across the hole invents production that never happened. A gap
is a hole in *time*; a zero is a measurement. They are counted separately
and never merged, and no value is ever filled in here.

**The gate.** `TagQuality.usable` and `may_fit` are the whole point. A result
computed from red data is worse than no result, because it arrives with the
same formatting and the same confident interval as a good one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import ATM_PSIA, PSI_PER_BAR, PSI_PER_KPA, PSI_PER_KSC

# A run this long is suspicious even on a quiet tag, and below it there is
# not enough evidence to call anything. Ten samples is the shortest run that
# is unlikely to be a genuinely steady process on a quantised instrument.
MIN_FROZEN_RUN = 10

# A tag that is flat for this fraction of the window is reported as frozen
# regardless of run structure. A real process control loop moves.
FROZEN_FRACTION = 0.90

# How many times its own typical pause a tag must hold still before the
# pause is called stale. See `_check_frozen` for why the comparison is
# against the tag itself rather than against a fixed number of samples.
STALE_RUN_RATIO = 20.0

# Below this much movement per sample, measured in quantisation steps, no
# run-length test means anything and this module says so instead of
# answering. Swept over noise-to-resolution ratios from 0.05 to 10 with a
# 160-sample stick planted in each trace, the live and stuck populations
# separate cleanly above 0.5 (worst live 10.7x, best stuck 53.3x) and
# *invert* below 0.2 (live 110x against stuck 10x), because a tag that
# coarse lingers in one bin for hundreds of samples on its own.
RESOLVED_MOVEMENT = 0.5

# Below this many finite samples nothing is decided. Statements about
# freezing, drift or spread need a window to be made over, and a short
# window produces confident nonsense.
MIN_SAMPLES = 24

# Conversions a tag might plausibly be configured in when the model wants
# psia. Ordered so the report reads from the largest discrepancy down.
PRESSURE_UNITS = (
    ("kPa", PSI_PER_KPA),
    ("bar", PSI_PER_BAR),
    ("kg/cm2", PSI_PER_KSC),
)

# Physical bounds, in field units, for the kinds this module knows about.
# `None` means unbounded on that side. These are deliberately generous: the
# job is to catch the impossible, not to enforce a house view of the typical.
BOUNDS = {
    "pressure_psia": (0.0, 20_000.0),
    "pressure_psig": (-ATM_PSIA, 20_000.0),
    "temperature_f": (-100.0, 800.0),
    "rate_stb_d": (0.0, 200_000.0),
    "rate_mscf_d": (0.0, 500_000.0),
    "water_cut": (0.0, 1.0),
    "gas_oil_ratio_scf_stb": (0.0, 200_000.0),
    "fraction": (0.0, 1.0),
}

# Kinds whose plausible operating magnitude is known well enough to make a
# unit check meaningful. A rate tag has no such magnitude -- a well can
# legitimately make 3 stb/d or 30,000 -- so rates are never unit-checked.
# Claiming otherwise would flag every low-rate stripper well in the field.
UNIT_CHECKED = {"pressure_psia", "pressure_psig"}

# The window a pressure tag in an active field is expected to live in. Used
# only as the denominator of the ratio test, never as a validity bound.
TYPICAL_PRESSURE_PSIA = (20.0, 6_000.0)


@dataclass
class Finding:
    """One thing wrong with one tag, in the engineer's terms."""

    code: str
    severity: str          # "red" bars use, "amber" warns, "info" records
    detail: str
    samples: int = 0

    @property
    def blocking(self) -> bool:
        return self.severity == "red"


@dataclass
class TagQuality:
    """The verdict on one tag, and the evidence behind it."""

    tag: str
    kind: str
    samples: int
    finite: int
    findings: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether a model may read this tag at all."""
        return not any(f.blocking for f in self.findings)

    @property
    def severity(self) -> str:
        for level in ("red", "amber", "info"):
            if any(f.severity == level for f in self.findings):
                return level
        return "green"

    @property
    def coverage(self) -> float:
        return self.finite / self.samples if self.samples else 0.0

    def describe(self) -> str:
        if not self.findings:
            return "%s: %d samples, nothing found" % (self.tag, self.finite)
        return "%s [%s]: %s" % (
            self.tag, self.severity,
            "; ".join(f.detail for f in self.findings))


@dataclass
class QualityReport:
    """Every tag's verdict, and the one question a caller actually asks."""

    tags: list = field(default_factory=list)

    @property
    def may_fit(self) -> bool:
        """Whether the whole set is fit to calibrate or optimise against.

        One red tag stops the lot. That is deliberate and it is the
        difference between a quality report and a quality gate: a report
        that everyone may override is read once and then switched off.
        """
        return all(tag.usable for tag in self.tags)

    @property
    def blocked(self) -> list:
        return [tag for tag in self.tags if not tag.usable]

    def tag(self, name: str) -> TagQuality:
        for entry in self.tags:
            if entry.tag == name:
                return entry
        raise KeyError(name)

    def summary(self) -> str:
        if self.may_fit:
            return "%d tags checked, none blocking" % len(self.tags)
        return ("%d of %d tags cannot be used: %s"
                % (len(self.blocked), len(self.tags),
                   ", ".join(t.tag for t in self.blocked)))


def _quantisation(values) -> float:
    """The smallest real step this instrument resolves.

    A 12-bit transmitter on a 0-1500 psi range moves in 0.37 psi steps, and
    every genuine change is a multiple of that. Taking the median of the
    non-zero absolute differences recovers it without assuming the range or
    the word length, and it degrades gracefully: on a finely resolved tag the
    median difference is just the typical sample-to-sample movement, which is
    the right thing to compare run lengths against anyway.
    """
    diffs = np.abs(np.diff(values))
    moving = diffs[diffs > 0.0]
    if moving.size == 0:
        return 0.0
    return float(np.median(moving))


def _runs(values, step: float):
    """Lengths of every stretch with no transition, at this resolution.

    Equality is the wrong comparison on floats that have been through a
    historian's compression and back, so "no transition" means "moved by less
    than half a quantisation step" -- which is exactly the movement the
    instrument could not have reported.
    """
    if values.size == 0:
        return np.zeros(0, dtype=int)
    moved = np.abs(np.diff(values)) > max(step / 2.0, 0.0)
    return np.diff(np.flatnonzero(
        np.concatenate(([True], moved, [True]))))


def _check_frozen(values, quality: TagQuality) -> None:
    if values.size < MIN_SAMPLES:
        return
    step = _quantisation(values)

    if step == 0.0:
        quality.findings.append(Finding(
            code="frozen",
            severity="red",
            detail=("every one of %d readings is identical (%.4g) -- the "
                    "instrument is latched, not the process"
                    % (values.size, float(values[0]))),
            samples=int(values.size)))
        return

    runs = _runs(values, step)
    run = int(runs.max())
    fraction = run / values.size
    if fraction >= FROZEN_FRACTION and run >= MIN_FROZEN_RUN:
        quality.findings.append(Finding(
            code="frozen",
            severity="red",
            detail=("%d of %d readings sit inside one %.4g resolution step; "
                    "a live tag at this resolution would have moved"
                    % (run, values.size, step)),
            samples=run))
        return

    # Whether a run-length test can say anything at all about this tag.
    movement = float(np.std(np.diff(values))) / step
    if movement < RESOLVED_MOVEMENT:
        quality.findings.append(Finding(
            code="unresolved",
            severity="info",
            detail=("this tag moves %.2f of a %.4g resolution step per "
                    "sample, too little for a run-length test to tell a "
                    "stuck transmitter from a quiet one -- freezing is not "
                    "being judged here" % (movement, step)),
            samples=int(values.size)))
        return

    # Absolute run length cannot separate a stuck transmitter from a quiet
    # one, because how long a live tag sits still depends entirely on its
    # noise relative to its resolution: a well-resolved, slow-moving tag
    # genuinely holds one step for a minute at a time. What does separate
    # them is the tag against itself. A live tag's longest pause is a few
    # multiples of its typical pause; a stuck one's is orders of magnitude
    # longer. Measured on the two cases in the tests, a steady well on a
    # 12-bit transmitter runs at 1.1x and an hour-long stick at 164x.
    typical = max(float(np.median(runs)), 1.0)
    ratio = run / typical
    if run >= MIN_FROZEN_RUN and ratio >= STALE_RUN_RATIO:
        quality.findings.append(Finding(
            code="stale",
            severity="amber",
            detail=("%d consecutive readings with no movement at the %.4g "
                    "resolution of this tag, against a typical %.0f -- %.0fx "
                    "longer than this tag's own habit. Check the transmitter "
                    "before fitting to this window"
                    % (run, step, typical, ratio)),
            samples=run))


def _check_impossible(values, kind: str, quality: TagQuality) -> None:
    if kind not in BOUNDS:
        return
    low, high = BOUNDS[kind]

    below = int(np.sum(values < low)) if low is not None else 0
    above = int(np.sum(values > high)) if high is not None else 0
    if not (below or above):
        return

    count = below + above
    # A handful of bad readings is an instrument glitch and the rest of the
    # window is still usable; a window that is mostly impossible is a
    # configuration error and nothing in it can be trusted.
    severity = "red" if count > values.size * 0.05 else "amber"
    parts = []
    if below:
        parts.append("%d below %g" % (below, low))
    if above:
        parts.append("%d above %g" % (above, high))
    quality.findings.append(Finding(
        code="impossible",
        severity=severity,
        detail=("%s of %d readings are outside what %s can physically be (%s)"
                % (" and ".join(parts), values.size, kind,
                   "excluded from any fit" if severity == "amber"
                   else "the tag is unusable")),
        samples=count))


def _check_units(values, kind: str, quality: TagQuality) -> None:
    """Is this tag plausibly in the units the model will read it as?

    The test is a ratio, not a bound, because a pressure tag in kPa is not
    an *impossible* psi reading -- 1030 kPa is a perfectly believable 1030
    psi, which is the whole difficulty. What gives it away is that the whole
    tag sits a constant factor away from where a field's pressures live, and
    that factor matches a conversion.
    """
    if kind not in UNIT_CHECKED:
        return
    median = float(np.median(values))
    if median <= 0.0:
        return

    low, high = TYPICAL_PRESSURE_PSIA
    if low <= median <= high:
        return

    # The tag has left the band. Ask which conversion would bring it back:
    # a window recorded in kPa and read as psia reads ~6.9x high, and one
    # recorded in bar or kg/cm2 reads ~14x low.
    unique = [name for name, psi_per_unit in PRESSURE_UNITS
              if low <= median * psi_per_unit <= high]
    if unique:
        detail = ("median %.4g is outside the %g-%g psia a field operates in, "
                  "and converts into it from %s -- the tag is probably "
                  "configured in %s"
                  % (median, low, high, " or ".join(unique),
                     " or ".join(unique)))
        if len(unique) > 1:
            detail += (". These cannot be told apart from one pressure tag "
                       "(they differ by %.0f%%); confirm against a known "
                       "setpoint before converting."
                       % (abs(PSI_PER_BAR / PSI_PER_KSC - 1.0) * 100.0))
        quality.findings.append(Finding(
            code="units", severity="red", detail=detail,
            samples=int(values.size)))
    else:
        quality.findings.append(Finding(
            code="out_of_range", severity="amber",
            detail=("median %.4g is outside the %g-%g psia a field operates "
                    "in and does not match a unit conversion -- check the "
                    "tag mapping" % (median, low, high)),
            samples=int(values.size)))


def _check_gaps(raw, quality: TagQuality, zero_is_real: bool) -> None:
    """Missing samples, counted apart from zeros, and never filled.

    A gas-lift well that was shut in for a day reports zero and that zero is
    a measurement: averaging it in is correct, and interpolating across it
    invents a day of production. A sensor that was offline for a day reports
    nothing, and averaging *that* in is what silently drops the well's rate.
    They are the same shape on a chart and opposite in meaning, so they are
    never pooled here.
    """
    missing = int(np.sum(~np.isfinite(raw)))
    if missing:
        fraction = missing / raw.size
        quality.findings.append(Finding(
            code="gaps",
            severity="red" if fraction > 0.5 else "amber",
            detail=("%d of %d samples are missing (%.0f%%); no value has "
                    "been filled in" % (missing, raw.size, fraction * 100.0)),
            samples=missing))

    finite = raw[np.isfinite(raw)]
    zeros = int(np.sum(finite == 0.0))
    if zeros and zero_is_real:
        quality.findings.append(Finding(
            code="zeros",
            severity="info",
            detail=("%d readings are exactly zero and are being read as "
                    "real -- shut-in, not missing" % zeros),
            samples=zeros))


def check_tag(tag: str, values, kind: str = "",
              zero_is_real: bool = True) -> TagQuality:
    """Run every check that applies to one tag's window.

    `kind` names what the tag is supposed to be, which is what makes the
    bounds and the unit test physical rather than statistical. An unnamed
    kind still gets the checks that need no physics -- freezing and gaps --
    because those are the two that fire most often.
    """
    raw = np.asarray(values, dtype=float).ravel()
    finite = raw[np.isfinite(raw)]
    quality = TagQuality(tag=tag, kind=kind or "unknown",
                         samples=int(raw.size), finite=int(finite.size))

    if raw.size == 0:
        quality.findings.append(Finding(
            code="empty", severity="red", detail="the tag has no samples"))
        return quality

    _check_gaps(raw, quality, zero_is_real)

    if finite.size == 0:
        quality.findings.append(Finding(
            code="empty", severity="red",
            detail="every sample is missing or non-numeric",
            samples=int(raw.size)))
        return quality

    if finite.size < MIN_SAMPLES:
        quality.findings.append(Finding(
            code="short", severity="amber",
            detail=("%d usable samples; %d are needed before freezing or "
                    "drift can be judged" % (finite.size, MIN_SAMPLES)),
            samples=int(finite.size)))

    _check_impossible(finite, kind, quality)
    _check_units(finite, kind, quality)
    _check_frozen(finite, quality)
    return quality


def check(tags: dict, kinds: dict = None,
          zero_is_real: dict = None) -> QualityReport:
    """Check a whole historian export. `tags` maps tag name to its window."""
    kinds = kinds or {}
    zero_is_real = zero_is_real or {}
    return QualityReport(tags=[
        check_tag(name, values, kinds.get(name, ""),
                  zero_is_real.get(name, True))
        for name, values in sorted(tags.items())
    ])


def check_tests(tests: list) -> QualityReport:
    """The same gate, applied to a well-test sheet instead of a tag window.

    This is the form the Calibrator actually reads, and the sheet has its
    own characteristic failure that a time-series check would never see.
    When a test separator is not available the previous test's wellhead
    pressure is carried forward, so a sheet of six "independent" tests
    carries three distinct pressures. Fitting the tubing side to that is
    fitting to one measurement repeated, and the result looks like a
    well-determined calibration because the residuals really are small.

    `identifiability` does not catch it. It asks whether the *rates* span a
    range, which they do -- the rates are genuine. It is the pressures that
    were copied, and a well with real rate spread and no pressure spread is
    the case it passes and should not.

    Bounds are not re-checked here: `WellTest` refuses an impossible row at
    construction, so anything reaching this point is already inside them.
    """
    if not tests:
        return QualityReport(tags=[])

    columns = {
        "rate_stb_d": ("rate_stb_d", [t.rate_stb_d for t in tests]),
        "wellhead_psia": ("pressure_psia", [t.wellhead_psia for t in tests]),
        "water_cut": ("fraction", [t.water_cut for t in tests]),
    }
    surveys = [t.bottomhole_psia for t in tests if t.bottomhole_psia > 0.0]
    if surveys:
        columns["bottomhole_psia"] = ("pressure_psia", surveys)

    report = QualityReport()
    for name, (kind, values) in sorted(columns.items()):
        series = np.asarray(values, dtype=float)
        quality = TagQuality(tag=name, kind=kind, samples=int(series.size),
                             finite=int(np.isfinite(series).sum()))
        _check_units(series[np.isfinite(series)], kind, quality)
        _check_repeated(series, name, len(tests), quality)
        report.tags.append(quality)
    return report


def _check_repeated(series, name: str, tests: int,
                    quality: TagQuality) -> None:
    """How many of these readings are actually distinct."""
    finite = series[np.isfinite(series)]
    if finite.size < 3:
        return
    distinct = int(np.unique(np.round(finite, 6)).size)
    if distinct >= finite.size:
        return
    if distinct == 1 and float(finite[0]) == 0.0:
        # An all-zero column was not filled in. Saying it repeats is true
        # and useless, and it would say it on every fit, which is how a
        # quality report trains people to stop reading quality reports.
        return

    # Rates repeat legitimately -- two tests a month apart on a stable well
    # genuinely read the same. A pressure that repeats across tests is
    # almost always carried forward, and it is the pressures that carry the
    # tubing-side information.
    severity = "red" if (name.endswith("_psia")
                         and distinct <= max(2, finite.size // 2)) else "info"
    quality.findings.append(Finding(
        code="repeated",
        severity=severity,
        detail=("%d of %d tests share only %d distinct value%s of %s"
                % (finite.size, tests, distinct,
                   "" if distinct == 1 else "s", name)
                + (" -- this column was carried forward, and fitting the "
                   "tubing side to it fits one measurement repeated"
                   if severity == "red" else "")),
        samples=finite.size - distinct))


def require_usable(report: QualityReport, action: str = "fit") -> None:
    """Raise unless every tag may be used. The gate, in one call.

    Callers that can degrade gracefully should read `may_fit` instead. This
    exists for the callers that cannot -- a calibration or an optimisation,
    where a blocked tag means the answer would be confidently wrong rather
    than merely wide.
    """
    if report.may_fit:
        return
    raise ValueError(
        "refusing to %s on this data: %s. Fix the instrument or exclude the "
        "tag; the model will not average over it."
        % (action, "; ".join(t.describe() for t in report.blocked)))
