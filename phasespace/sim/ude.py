"""A learned correction to the tubing correlation, kept inside the physics.

Rackauckas et al., "Universal Differential Equations for Scientific Machine
Learning" (arXiv 2001.04385). Tier 1 of the reading list, and the piece the
list argues is the commercial differentiator: physics answer, plus a learned
field-specific correction, with the physics still deciding.

The gradient this corrects is the one the engine already integrates:

    dp/dL = rho_m g sin(theta) + f_m rho_m v_m^2 / (2 D) + acceleration

Beggs-Brill supplies the mixture density through a liquid holdup and the
friction factor. Both were fitted on a test loop in the 1970s, on fluids
that are not this field's, and on a particular well a holdup correlation
is routinely ten or twenty percent out -- not uniformly, which is the
point: the gravity term dominates at low rate, friction at high rate.

`wellcalibration` already handles that with two constants, one on each
term. A UDE is the same idea with the constants replaced by a small
network of the local flow conditions, so the correction can differ between
the gassy top of the string and the liquid-loaded bottom, which two
constants cannot express.

**What keeps this from being a neural network with a physics-flavoured
name.** Three things, and all three are the reason it is allowed near a
production forecast.

*The correction is bounded.* Each term is multiplied by `1 + 0.2 tanh(z)`,
so no network output can move a term by more than 20% -- the same band
`wellcalibration` allows its constants. A network that wants more is
telling you the correlation is wrong for this well, which is a finding to
report, not a licence to overwrite the physics.

*Zero is the default and must be earned.* The loss is regularised towards
no correction at all, so the published correlation is what comes back
unless the data pays for moving away from it.

*It is scored against the model it replaces, not against raw Beggs-Brill.*
Beating the uncorrected correlation is easy and means nothing -- two
constants already do that. `compare_with_constants` fits the two-constant
baseline on the same split and reports both, and if the network does not
beat it on held-out wells the honest result is to ship the constants. The
engine has done exactly that once already, with the hydrate predictor.

**The split is by well, never by point.** Points down one string share a
fluid, a geometry and a measurement campaign, so a point-wise split leaks
the answer across the fold and reports a skill the model does not have.

**On the gradient.** The forward model is exact: the traverse marches with
the correction applied at every segment. The *derivative* used for fitting
treats each segment's elevation and friction contributions as fixed while
the correction varies, ignoring that changing the pressure profile also
changes the properties at the next segment down. That makes the optimiser's
steps approximate and its fixed point correct, which is the right trade --
an exact adjoint through a predictor-corrector march is a great deal of
machinery for a slightly better step direction.

numpy only. Training and inference both, so a customer can refit on their
own wells inside their own network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .wells import BEGGS_BRILL, beggs_brill_gradient, hagedorn_brown_gradient

# The correction band, matching `wellcalibration.MIN/MAX_MULTIPLIER`. A
# learned layer is not allowed more licence than a fitted constant.
MAX_CORRECTION = 0.20

# Small on purpose. Four features and a handful of hidden units is already
# more freedom than a few dozen gradient surveys can pay for, and the
# failure mode of a bigger network here is a confident wrong answer on a
# well unlike the training set.
HIDDEN_UNITS = 8

FEATURES = (
    "holdup",                 # liquid holdup, 0-1
    "froude",                 # log10 mixture Froude number
    "gas_fraction",           # superficial gas velocity share, 0-1
    "sin_inclination",        # 1 vertical, 0 horizontal
)

# Below this the fit has fewer independent wells than it has freedom, and
# the held-out comparison it rests on cannot be made at all.
MIN_WELLS = 4


def _gradient_fn(correlation: str):
    return (hagedorn_brown_gradient if correlation and correlation != BEGGS_BRILL
            else beggs_brill_gradient)


def features_from(result: dict, tubing) -> np.ndarray:
    """Dimensionless description of the flow at one point in the string.

    Dimensionless on purpose: a network fed raw velocities and pressures
    learns the units of the training field and transfers to nothing.
    """
    v_m = float(result.get("mixture_velocity_ft_s", 0.0))
    v_sg = float(result.get("superficial_gas_velocity_ft_s", 0.0))
    diameter = max(tubing.diameter_ft, 1e-6)
    froude = v_m * v_m / (32.174 * diameter)
    return np.array([
        float(result.get("holdup", 1.0)),
        math.log10(froude + 1e-6),
        v_sg / v_m if v_m > 1e-9 else 0.0,
        math.sin(math.radians(tubing.inclination_deg)),
    ], dtype=float)


@dataclass
class Correction:
    """The small network, and the bound it cannot leave."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    max_correction: float = MAX_CORRECTION

    @classmethod
    def zero(cls, rng=None, hidden: int = HIDDEN_UNITS) -> "Correction":
        """A correction that does nothing, which is where fitting starts.

        The output layer begins at exactly zero so the first forward pass
        reproduces the published correlation bit for bit. Starting from a
        random correction would mean the physics is only recovered if the
        optimiser happens to find its way back.
        """
        rng = rng or np.random.default_rng(0)
        n = len(FEATURES)
        return cls(
            w1=rng.normal(0.0, 1.0 / math.sqrt(n), (n, hidden)),
            b1=np.zeros(hidden),
            w2=np.zeros((hidden, 2)),
            b2=np.zeros(2),
            feature_mean=np.zeros(n),
            feature_scale=np.ones(n))

    def _hidden(self, x):
        return np.tanh((x - self.feature_mean) / self.feature_scale @ self.w1
                       + self.b1)

    def apply(self, x) -> np.ndarray:
        """(gravity, friction) multipliers for one or many feature rows."""
        single = np.asarray(x, dtype=float).ndim == 1
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        raw = self._hidden(rows) @ self.w2 + self.b2
        out = 1.0 + self.max_correction * np.tanh(raw)
        return out[0] if single else out

    @property
    def is_identity(self) -> bool:
        return not np.any(self.w2) and not np.any(self.b2)

    def to_dict(self) -> dict:
        return {"w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": self.b2.tolist(),
                "feature_mean": self.feature_mean.tolist(),
                "feature_scale": self.feature_scale.tolist(),
                "max_correction": self.max_correction,
                "features": list(FEATURES)}

    @classmethod
    def from_dict(cls, raw: dict) -> "Correction":
        if list(raw.get("features", FEATURES)) != list(FEATURES):
            raise ValueError(
                "this correction was trained on features %s but this engine "
                "computes %s; refusing to apply it"
                % (raw.get("features"), list(FEATURES)))
        return cls(
            w1=np.array(raw["w1"], dtype=float), b1=np.array(raw["b1"], dtype=float),
            w2=np.array(raw["w2"], dtype=float), b2=np.array(raw["b2"], dtype=float),
            feature_mean=np.array(raw["feature_mean"], dtype=float),
            feature_scale=np.array(raw["feature_scale"], dtype=float),
            max_correction=float(raw.get("max_correction", MAX_CORRECTION)))


@dataclass
class Survey:
    """One measured bottomhole pressure, and the conditions it was taken at."""

    well: str
    wellhead_psia: float
    bottomhole_psia: float
    stream: object                  # wells.ProductionStream
    tubing: object                  # wells.Tubing
    correlation: str = BEGGS_BRILL


def traverse(survey: Survey, correction: Correction = None,
             constants: tuple = (1.0, 1.0), segments: int = 40) -> dict:
    """March the string with the correction applied, keeping the segments.

    Deliberately the same midpoint march as `wells.pressure_traverse`, over
    the same gradient function, because the point of a grey box is that the
    physics underneath is unchanged. `test_ude.py` asserts the two agree to
    floating point when the correction is the identity; without that test
    this would be a second implementation quietly drifting from the first.
    """
    tubing = survey.tubing
    gradient_fn = _gradient_fn(survey.correlation)
    dz = tubing.depth_ft / segments
    pressure = survey.wellhead_psia
    depth = 0.0
    rows = []

    for _ in range(segments):
        t_f = tubing.temperature_f_at(depth + 0.5 * dz)
        first = gradient_fn(pressure, t_f, survey.stream, tubing)
        gravity, friction = _multipliers(first, tubing, correction, constants)
        pressure_mid = max(pressure + _scaled(first, gravity, friction) * dz * 0.5, 1.0)

        second = gradient_fn(pressure_mid, t_f, survey.stream, tubing)
        gravity, friction = _multipliers(second, tubing, correction, constants)
        pressure = max(pressure + _scaled(second, gravity, friction) * dz, 1.0)
        depth += dz

        elevation = second["elevation_psi_ft"]
        friction_term = second["friction_psi_ft"]
        parts = elevation + friction_term
        extra = (second["gradient_psi_ft"] / parts) if parts > 0.0 else 1.0
        rows.append({
            "features": features_from(second, tubing),
            # How much the bottomhole pressure moves per unit of correction
            # on each term. The physics half of the chain rule.
            "d_gravity": elevation * extra * dz,
            "d_friction": friction_term * extra * dz,
        })

    return {"bottomhole_psia": pressure, "segments": rows}


def _multipliers(result, tubing, correction, constants):
    if correction is None:
        return constants
    pair = correction.apply(features_from(result, tubing))
    return float(pair[0]) * constants[0], float(pair[1]) * constants[1]


def _scaled(result, gravity, friction) -> float:
    elevation = result["elevation_psi_ft"]
    friction_term = result["friction_psi_ft"]
    parts = elevation + friction_term
    if parts <= 0.0:
        return result["gradient_psi_ft"]
    extra = result["gradient_psi_ft"] / parts
    return (elevation * gravity + friction_term * friction) * extra


@dataclass
class FitResult:
    """A fitted correction, and whether it earned its place."""

    correction: Correction
    constants: tuple
    train_wells: list
    test_wells: list
    rms_physics: float               # uncorrected correlation, held out
    rms_constants: float             # two fitted constants, held out
    rms_ude: float                   # the network, held out
    history: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def beats_constants(self) -> bool:
        return self.rms_ude < self.rms_constants

    @property
    def gain_over_constants(self) -> float:
        if self.rms_constants <= 0.0:
            return 0.0
        return 1.0 - self.rms_ude / self.rms_constants

    @property
    def recommendation(self) -> str:
        if not self.beats_constants:
            return "ship the constants"
        if self.gain_over_constants < 0.05:
            return "ship the constants"
        return "ship the correction"

    def describe(self) -> str:
        return ("held out %d well(s): correlation %.1f psi, two constants "
                "%.1f psi, learned correction %.1f psi -- %s"
                % (len(self.test_wells), self.rms_physics,
                   self.rms_constants, self.rms_ude, self.recommendation))


def _rms(surveys, correction, constants) -> float:
    errors = [traverse(s, correction, constants)["bottomhole_psia"]
              - s.bottomhole_psia for s in surveys]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0


def fit_constants(surveys: list, bounds=(1.0 - MAX_CORRECTION,
                                         1.0 + MAX_CORRECTION),
                  steps: int = 11) -> tuple:
    """The baseline the network has to beat: one constant on each term.

    A coarse grid then a local refinement. Two parameters over a bounded
    box does not need an optimiser, and a grid cannot report a local
    minimum as the answer.
    """
    low, high = bounds
    best, best_rms = (1.0, 1.0), math.inf
    span = (high - low) / 2.0
    centre = (low + high) / 2.0
    for _refinement in range(3):
        grid_g = np.linspace(max(centre - span, low), min(centre + span, high), steps)
        grid_f = np.linspace(max(centre - span, low), min(centre + span, high), steps)
        for gravity in grid_g:
            for friction in grid_f:
                score = _rms(surveys, None, (float(gravity), float(friction)))
                if score < best_rms:
                    best_rms, best = score, (float(gravity), float(friction))
        centre = (best[0] + best[1]) / 2.0
        span = (grid_g[1] - grid_g[0]) * 1.5 if grid_g.size > 1 else span
    return best


def fit(surveys: list, test_wells: list = None, learning_rate: float = 0.02,
        iterations: int = 300, regularisation: float = 1e-3,
        seed: int = 0) -> FitResult:
    """Fit the correction, and report whether it beats two constants.

    The held-out set is whole wells. Splitting by point would put surveys
    from one string on both sides of the fold -- same fluid, same geometry,
    same campaign -- and report a skill that does not survive a new well,
    which is the only well anyone cares about.

    **Adam rather than plain gradient descent, and not for fashion.** The
    derivative of bottomhole pressure with respect to a correction is the
    whole hydrostatic column -- on an 8000 ft well, about 1500 psi per unit.
    Against a residual of a few psi that is a gradient in the thousands, and
    a fixed step saturates the output tanh on the first iteration: the
    correction pins at its bound and stays there. Measured on the planted-bias
    case, plain descent at 0.05 returned 286 psi of held-out error against
    the constants' 2.6. Adam's per-parameter normalisation removes the scale
    entirely, which is what this problem needs.

    **A third split, for stopping.** The test wells answer "did it work".
    They cannot also choose when to stop, or the answer is contaminated, so
    a quarter of the training wells are held back again as a validation set
    and the iterate with the best validation error is the one returned.
    """
    wells = sorted({s.well for s in surveys})
    if len(wells) < MIN_WELLS:
        raise ValueError(
            "%d well(s) cannot support this: the correction has to be shown "
            "to beat two constants on wells it never saw, and %d are needed "
            "before that comparison means anything"
            % (len(wells), MIN_WELLS))

    if test_wells is None:
        rng = np.random.default_rng(seed)
        held = max(1, len(wells) // 4)
        test_wells = sorted(rng.choice(wells, size=held, replace=False).tolist())
    test_wells = list(test_wells)
    train = [s for s in surveys if s.well not in test_wells]
    test = [s for s in surveys if s.well in test_wells]
    if not train or not test:
        raise ValueError("the split left one side empty; check `test_wells`")

    constants = fit_constants(train)

    train_wells = sorted({s.well for s in train})
    rng_split = np.random.default_rng(seed + 1)
    held = max(1, len(train_wells) // 4)
    validation_wells = sorted(
        rng_split.choice(train_wells, size=held, replace=False).tolist())
    inner = [s for s in train if s.well not in validation_wells]
    validation = [s for s in train if s.well in validation_wells]
    if not inner:
        inner, validation = train, train

    rng = np.random.default_rng(seed)
    correction = Correction.zero(rng)
    sample = np.array([row["features"]
                       for s in inner
                       for row in traverse(s, None, constants)["segments"]])
    correction.feature_mean = sample.mean(axis=0)
    correction.feature_scale = np.where(sample.std(axis=0) > 1e-9,
                                        sample.std(axis=0), 1.0)

    history = []
    # Adam. Betas and epsilon are Kingma & Ba's defaults; nothing here is
    # sensitive to them, unlike the step size, which the per-parameter
    # normalisation is what makes safe.
    moments = {k: np.zeros_like(v) for k, v in
               (("w1", correction.w1), ("b1", correction.b1),
                ("w2", correction.w2), ("b2", correction.b2))}
    velocities = {k: np.zeros_like(v) for k, v in moments.items()}
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    best_score = _rms(validation, correction, constants)
    best = (correction.w1.copy(), correction.b1.copy(),
            correction.w2.copy(), correction.b2.copy())

    for step in range(iterations):
        grad_w1 = np.zeros_like(correction.w1)
        grad_b1 = np.zeros_like(correction.b1)
        grad_w2 = np.zeros_like(correction.w2)
        grad_b2 = np.zeros_like(correction.b2)
        total = 0.0

        for survey in inner:
            run = traverse(survey, correction, constants)
            residual = run["bottomhole_psia"] - survey.bottomhole_psia
            total += residual ** 2
            rows = run["segments"]
            x = np.array([r["features"] for r in rows])
            physics = np.array([[r["d_gravity"], r["d_friction"]] for r in rows])

            normalised = (x - correction.feature_mean) / correction.feature_scale
            pre = normalised @ correction.w1 + correction.b1
            hidden = np.tanh(pre)
            raw = hidden @ correction.w2 + correction.b2
            # d(multiplier)/d(raw) for 1 + k tanh(raw)
            slope = correction.max_correction * (1.0 - np.tanh(raw) ** 2)

            # dL/d(raw) = 2 * residual * d(bhp)/d(multiplier) * d(mult)/d(raw)
            upstream = 2.0 * residual * physics * slope / len(inner)
            grad_w2 += hidden.T @ upstream
            grad_b2 += upstream.sum(axis=0)
            back = (upstream @ correction.w2.T) * (1.0 - hidden ** 2)
            grad_w1 += normalised.T @ back
            grad_b1 += back.sum(axis=0)

        grad_w2 += 2.0 * regularisation * correction.w2
        grad_b2 += 2.0 * regularisation * correction.b2

        grads = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
        for key, grad in grads.items():
            moments[key] = beta1 * moments[key] + (1.0 - beta1) * grad
            velocities[key] = (beta2 * velocities[key]
                               + (1.0 - beta2) * grad * grad)
            corrected_m = moments[key] / (1.0 - beta1 ** (step + 1))
            corrected_v = velocities[key] / (1.0 - beta2 ** (step + 1))
            setattr(correction, key,
                    getattr(correction, key)
                    - learning_rate * corrected_m
                    / (np.sqrt(corrected_v) + epsilon))

        history.append(math.sqrt(total / len(inner)))
        score = _rms(validation, correction, constants)
        if score < best_score:
            best_score = score
            best = (correction.w1.copy(), correction.b1.copy(),
                    correction.w2.copy(), correction.b2.copy())

    # The iterate that was best on wells it was not fitted to, not the last.
    correction.w1, correction.b1, correction.w2, correction.b2 = best

    result = FitResult(
        correction=correction, constants=constants,
        train_wells=[w for w in wells if w not in test_wells],
        test_wells=test_wells,
        rms_physics=_rms(test, None, (1.0, 1.0)),
        rms_constants=_rms(test, None, constants),
        rms_ude=_rms(test, correction, constants),
        history=history)

    if not result.beats_constants:
        result.notes.append(
            "the learned correction does not beat two fitted constants on "
            "the held-out wells (%.1f psi against %.1f). Ship the constants: "
            "the extra freedom bought nothing and will not transfer."
            % (result.rms_ude, result.rms_constants))
    elif result.gain_over_constants < 0.05:
        result.notes.append(
            "the learned correction beats two constants by %.1f%%, which is "
            "inside the noise of this comparison. Ship the constants."
            % (result.gain_over_constants * 100.0))
    else:
        result.notes.append(
            "the learned correction beats two constants by %.1f%% on wells "
            "it never saw. That is a real field-specific bias in the "
            "correlation, and worth carrying."
            % (result.gain_over_constants * 100.0))

    reach = float(np.max(np.abs(result.correction.apply(
        np.array([row["features"] for s in test
                  for row in traverse(s, correction, constants)["segments"]]))
        - 1.0)))
    if reach > 0.95 * MAX_CORRECTION:
        result.notes.append(
            "the correction is pressed against its %.0f%% bound. The "
            "correlation is further out on this field than a grey box is "
            "allowed to fix, which is a finding about the correlation "
            "rather than a reason to widen the bound."
            % (MAX_CORRECTION * 100.0))
    return result
