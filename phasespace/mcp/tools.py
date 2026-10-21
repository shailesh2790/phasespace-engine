"""The engine, as tools an agent may call.

Seven tools, chosen because each answers a question an engineer actually
asks and each is backed by something in the engine that has tests behind
it. Nothing here computes physics: every handler marshals arguments, calls
the engine, and reports what came back together with whatever the engine
said about its own limits.

Two conventions run through all of them.

**Field units at the boundary.** The engine works in field units
internally, so the tool surface does too, and every schema states the unit
in the property description. A tool that accepted "pressure" without
saying which unit would be the single most likely source of a wrong answer
on this whole surface -- a ksc figure read as psia is out by a factor of
fourteen, and the result looks entirely plausible.

**The engine's own caveats are the tool's output.** Where a module returns
notes -- a correlation outside its range, a test with too little drawdown,
a water analysis it was never given -- those notes are passed through
rather than summarised away. They are usually the most useful part of the
answer and they are what stops an agent presenting a number with more
confidence than it has.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile

from phasespace.sim.actions import (
    MEASURED,
    MIN_ACTIONS_FOR_A_RATIO,
    Action,
    Measurement,
    Scorecard,
    Window,
)
from phasespace.sim.decline import ArpsDecline, fit_arps
from phasespace.sim.fluids import GasProperties, LiquidProperties
from phasespace.sim.gaslift import GasLiftWell, allocate
from phasespace.sim.integrity import assess, scale_inputs
from phasespace.sim.dataquality import check
from phasespace.sim.wells import (BEGGS_BRILL, HAGEDORN_BROWN,
                                  CompositeIPR, FetkovichIPR,
                                  ProductionStream, ProductivityIndexIPR,
                                  Tubing, VogelIPR, nodal_analysis,
                                  pressure_traverse)

from .registry import Refusal, Registry, Tool

# ----------------------------------------------------------------- helpers --

PRESSURE = {"type": "number", "description": "psia (absolute, not gauge)"}
DEPTH = {"type": "number", "description": "feet"}
TEMPERATURE = {"type": "number", "description": "degrees Fahrenheit"}


def _gas(args: dict) -> GasProperties:
    return GasProperties(
        gravity=float(args.get("gas_gravity", 0.65)),
        temperature_r=float(args.get("temperature_f", 100.0)) + 459.67,
        y_co2=float(args.get("co2_fraction", 0.0)),
        y_h2s=float(args.get("h2s_fraction", 0.0)),
        y_n2=float(args.get("n2_fraction", 0.0)),
    )


def _require(args: dict, *names):
    missing = [n for n in names if args.get(n) is None]
    if missing:
        raise ValueError("missing required argument(s): %s" % ", ".join(missing))


def _correlation(args: dict) -> str:
    """Resolve the correlation name, and say when the default is a poor fit.

    The default is Beggs-Brill, which is the right general choice and the
    wrong one for a vertical dry gas well: on 150 measured gradient surveys
    it gave 13.0% median absolute error against Hagedorn-Brown's 3.9%. The
    engine does not silently switch, so the tool flags it instead.
    """
    name = str(args.get("correlation", BEGGS_BRILL)).lower()
    if name not in (BEGGS_BRILL, HAGEDORN_BROWN):
        raise ValueError("correlation must be %r or %r, got %r"
                         % (BEGGS_BRILL, HAGEDORN_BROWN, name))
    return name


# ------------------------------------------------------------ well traverse --

TRAVERSE_SCHEMA = {
    "type": "object",
    "properties": {
        "depth_ft": DEPTH,
        "tubing_id_in": {"type": "number", "description": "tubing inner diameter, inches"},
        "wellhead_pressure_psia": PRESSURE,
        "gas_rate_scf_d": {"type": "number", "description": "gas rate, scf/day"},
        "oil_rate_stb_d": {"type": "number", "description": "oil rate, stb/day"},
        "water_rate_stb_d": {"type": "number", "description": "water rate, stb/day"},
        "gas_gravity": {"type": "number", "description": "air = 1.0"},
        "api_gravity": {"type": "number", "description": "oil API gravity"},
        "wellhead_temperature_f": TEMPERATURE,
        "bottomhole_temperature_f": TEMPERATURE,
        "inclination_deg": {"type": "number",
                            "description": "from horizontal; 90 is vertical"},
        "correlation": {"type": "string", "enum": [BEGGS_BRILL, HAGEDORN_BROWN],
                        "description": "default beggs_brill"},
    },
    "required": ["depth_ft", "tubing_id_in", "wellhead_pressure_psia"],
}


def traverse(args: dict) -> dict:
    _require(args, "depth_ft", "tubing_id_in", "wellhead_pressure_psia")
    correlation = _correlation(args)
    tubing = Tubing(
        depth_ft=float(args["depth_ft"]),
        inner_diameter_in=float(args["tubing_id_in"]),
        inclination_deg=float(args.get("inclination_deg", 90.0)),
        wellhead_temperature_f=float(args.get("wellhead_temperature_f", 100.0)),
        bottomhole_temperature_f=float(args.get("bottomhole_temperature_f", 180.0)),
    )
    stream = ProductionStream(
        oil_rate_stb_d=float(args.get("oil_rate_stb_d", 0.0)),
        water_rate_stb_d=float(args.get("water_rate_stb_d", 0.0)),
        gas_rate_scf_d=float(args.get("gas_rate_scf_d", 0.0)),
        oil=LiquidProperties(api_gravity=float(args.get("api_gravity", 35.0))),
        gas=_gas(args),
    )
    if (stream.liquid_rate_stb_d <= 0.0 and stream.gas_rate_scf_d <= 0.0):
        raise Refusal(
            "the well is producing nothing -- give a gas rate, an oil rate "
            "or a water rate. A traverse at zero flow is the static "
            "gradient, which is a different calculation.")

    result = pressure_traverse(float(args["wellhead_pressure_psia"]),
                               stream, tubing, correlation=correlation)

    notes = []
    if correlation == BEGGS_BRILL and stream.liquid_rate_stb_d == 0.0:
        notes.append(
            "this is a dry gas well on Beggs-Brill, which was developed on "
            "an inclined air-water loop. On 150 measured gradient surveys "
            "Hagedorn-Brown gave 3.9% median absolute error against "
            "Beggs-Brill's 13.0% for this well type -- consider "
            "correlation='hagedorn_brown'.")
    notes.append(
        "gravity accounts for %.0f%% of the pressure drop and friction "
        "%.0f%%" % (result.elevation_fraction * 100.0,
                    result.friction_fraction * 100.0))

    return {
        "summary": ("Bottomhole pressure %.1f psia at %.0f ft, from %.1f psia "
                    "at the wellhead (%s)."
                    % (result.bottomhole_pressure_psia, tubing.depth_ft,
                       result.wellhead_pressure_psia, correlation)),
        "outputs": {
            "bottomhole_pressure_psia": result.bottomhole_pressure_psia,
            "elevation_fraction": result.elevation_fraction,
            "friction_fraction": result.friction_fraction,
            "depths_ft": list(result.depths_ft),
            "pressures_psia": list(result.pressures_psia),
            "flow_regimes": list(result.regimes),
        },
        "notes": notes,
        "correlations": [correlation, "Dranchuk-Abou-Kassem Z",
                         "Sutton pseudo-criticals"],
    }


# ----------------------------------------------------------------- nodal --

NODAL_SCHEMA = {
    "type": "object",
    "properties": {
        "reservoir_pressure_psia": PRESSURE,
        "max_rate_stb_d": {"type": "number",
                           "description": "absolute open flow, stb/day (Vogel)"},
        "productivity_index": {"type": "number",
                               "description": "stb/day/psi (straight-line IPR, alternative to max_rate_stb_d)"},
        "depth_ft": DEPTH,
        "tubing_id_in": {"type": "number", "description": "inches"},
        "wellhead_pressure_psia": PRESSURE,
        "gas_oil_ratio_scf_stb": {"type": "number", "description": "scf/stb"},
        "water_cut": {"type": "number", "description": "fraction, 0 to 1"},
        "api_gravity": {"type": "number"},
        "gas_gravity": {"type": "number"},
        "correlation": {"type": "string", "enum": [BEGGS_BRILL, HAGEDORN_BROWN]},
    },
    "required": ["reservoir_pressure_psia", "depth_ft", "tubing_id_in",
                 "wellhead_pressure_psia"],
}


def nodal(args: dict) -> dict:
    _require(args, "reservoir_pressure_psia", "depth_ft", "tubing_id_in",
             "wellhead_pressure_psia")
    reservoir = float(args["reservoir_pressure_psia"])

    if args.get("productivity_index") is not None:
        ipr = ProductivityIndexIPR(
            reservoir_pressure_psia=reservoir,
            productivity_index=float(args["productivity_index"]))
    elif args.get("max_rate_stb_d") is not None:
        ipr = VogelIPR(reservoir_pressure_psia=reservoir,
                       max_rate=float(args["max_rate_stb_d"]))
    else:
        raise Refusal(
            "an inflow curve needs either max_rate_stb_d (Vogel) or "
            "productivity_index (straight line). Without one of them there "
            "is no inflow relation and nothing to intersect the tubing "
            "curve with.")

    result = nodal_analysis(
        ipr=ipr,
        tubing=Tubing(depth_ft=float(args["depth_ft"]),
                      inner_diameter_in=float(args["tubing_id_in"])),
        wellhead_pressure_psia=float(args["wellhead_pressure_psia"]),
        gas_oil_ratio_scf_stb=float(args.get("gas_oil_ratio_scf_stb", 0.0)),
        water_cut=float(args.get("water_cut", 0.0)),
        oil=LiquidProperties(api_gravity=float(args.get("api_gravity", 35.0))),
        gas=_gas(args),
        correlation=_correlation(args),
    )

    if not result.converged:
        raise Refusal(
            "the inflow and tubing curves do not intersect over the rate "
            "range searched, so this well has no operating point under these "
            "conditions. That is usually a real result -- the well is dead "
            "at this wellhead pressure -- rather than a numerical failure. "
            + "; ".join(result.notes))

    notes = list(result.notes)
    notes.append(
        "the traverse underlying this is validated to 2.9% median absolute "
        "error on 150 measured gradient surveys; the intersection itself -- "
        "predicting the rate -- is NOT validated, because that needs a well "
        "where inflow and outflow were measured together.")

    return {
        "summary": ("Operating point %.0f stb/d liquid (%.0f stb/d oil) at "
                    "%.1f psia bottomhole."
                    % (result.liquid_rate_stb_d, result.oil_rate_stb_d,
                       result.bottomhole_pressure_psia)),
        "outputs": {
            "liquid_rate_stb_d": result.liquid_rate_stb_d,
            "oil_rate_stb_d": result.oil_rate_stb_d,
            "bottomhole_pressure_psia": result.bottomhole_pressure_psia,
            "converged": result.converged,
        },
        "notes": notes,
        "envelopes": ["nodal intersection is verified, not validated"],
    }


# ------------------------------------------------------------- gas lift --

ALLOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "wells": {
            "type": "array",
            "description": "one entry per well, each with its lift tests",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tests": {
                        "type": "array",
                        "description": "at least 4 [injection_rate, oil_rate] pairs",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "q_min": {"type": "number",
                              "description": "minimum stable injection; below it the well heads"},
                    "q_max": {"type": "number"},
                    "current": {"type": "number",
                                "description": "injection today, for the gain comparison"},
                },
                "required": ["name", "tests"],
            },
        },
        "gas_available": {"type": "number",
                          "description": "total lift gas to share out, same units as the tests"},
        "step": {"type": "number", "description": "allocation increment, default 250"},
        "oil_price": {"type": "number"},
        "gas_price": {"type": "number",
                      "description": "same value units as oil_price; non-zero maximises value rather than volume"},
    },
    "required": ["wells", "gas_available"],
}


def gas_lift_allocate(args: dict) -> dict:
    _require(args, "wells", "gas_available")
    raw = args["wells"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("wells must be a non-empty array")

    wells = []
    for entry in raw:
        tests = [(float(a), float(b)) for a, b in entry.get("tests", [])]
        if len(tests) < 4:
            raise Refusal(
                "%s has %d lift tests. Fitting the four-coefficient response "
                "curve needs at least 4; fewer is not a loose fit but an "
                "underdetermined system that returns a confident meaningless "
                "answer." % (entry.get("name", "?"), len(tests)))
        wells.append(GasLiftWell(
            name=str(entry["name"]), tests=tests,
            q_min=float(entry.get("q_min", 0.0)),
            q_max=float(entry.get("q_max", 30000.0)),
            current=(float(entry["current"])
                     if entry.get("current") is not None else None)))

    result = allocate(wells, float(args["gas_available"]),
                      step=float(args.get("step", 250.0)),
                      oil_price=float(args.get("oil_price", 1.0)),
                      gas_price=float(args.get("gas_price", 0.0)))

    notes = list(result.notes)
    notes.append(
        "the shadow price is the marginal oil per unit of lift gas at the "
        "optimum: every well receiving gas shares it. It is what another "
        "unit of compression is worth.")
    if not result.exactly_optimal:
        notes.append(
            "at least one response curve is not concave, so greedy "
            "allocation is a strong heuristic here rather than provably "
            "optimal.")

    return {
        "summary": ("Allocated %.0f of %.0f available gas across %d wells for "
                    "%.1f total oil (%+.1f against current). Shadow price "
                    "%.5f oil per unit gas."
                    % (result.gas_used, result.gas_available,
                       len(result.wells), result.total_oil, result.oil_gain,
                       result.shadow_price)),
        "outputs": {
            "shadow_price": result.shadow_price,
            "total_oil": result.total_oil,
            "oil_gain": result.oil_gain,
            "gas_used": result.gas_used,
            "gas_unused": result.gas_unused,
            "exactly_optimal": result.exactly_optimal,
            "wells": [{"name": w.name, "injection": w.injection,
                       "oil": w.oil,
                       "injection_change": w.injection_change,
                       "oil_change": w.oil_change,
                       "warnings": list(w.warnings)}
                      for w in result.wells],
        },
        "notes": notes,
        "correlations": ["Alarcon gas-lift response form",
                         "greedy allocation to the equal-slope condition"],
        "envelopes": ["gas-lift optimisation is not validated: no open "
                      "lift-test data exists to validate it against"],
    }


# -------------------------------------------------------------- decline --

DECLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "days": {"type": "array", "items": {"type": "number"},
                 "description": "days since first production"},
        "rates": {"type": "array", "items": {"type": "number"},
                  "description": "production rate at each day, any consistent unit"},
        "terminal_decline_per_year": {"type": "number",
                                      "description": "switch to exponential below this, default 0.05"},
        "abandonment_rate": {"type": "number",
                             "description": "stop EUR integration here"},
    },
    "required": ["days", "rates"],
}


def decline(args: dict) -> dict:
    _require(args, "days", "rates")
    days = [float(d) for d in args["days"]]
    rates = [float(r) for r in args["rates"]]
    if len(days) != len(rates):
        raise ValueError("days and rates must be the same length, got %d and %d"
                         % (len(days), len(rates)))

    try:
        fit = fit_arps(days, rates,
                       d_min_per_year=float(args.get("terminal_decline_per_year", 0.05)),
                       q_abandon=float(args.get("abandonment_rate", 0.0)))
    except ValueError as exc:
        raise Refusal(
            "this history cannot be fitted with Arps: %s. A flat or rising "
            "history is not a decline curve, and forcing one would produce a "
            "forecast with nothing behind it." % exc)

    notes = list(fit.notes)
    notes.append(
        "b = %.2f over the range %.2f to %.2f. The EUR spread of %.0f%% "
        "comes from that range, not from a confidence interval on the data."
        % (fit.decline.b, fit.b_low, fit.b_high,
           fit.eur_spread_fraction * 100.0))

    return {
        "summary": ("Arps fit: qi %.1f, Di %.4f/yr, b %.2f. EUR %.0f (%.0f to "
                    "%.0f), from %d points."
                    % (fit.decline.qi, fit.decline.di_per_year, fit.decline.b,
                       fit.eur, fit.eur_low, fit.eur_high, fit.points_used)),
        "outputs": {
            "qi": fit.decline.qi,
            "di_per_year": fit.decline.di_per_year,
            "b": fit.decline.b,
            "b_low": fit.b_low, "b_high": fit.b_high,
            "eur": fit.eur, "eur_low": fit.eur_low, "eur_high": fit.eur_high,
            "eur_spread_fraction": fit.eur_spread_fraction,
            "rms_log_residual": fit.rms_log_residual,
            "points_used": fit.points_used,
        },
        "notes": notes,
        "correlations": ["Arps (1945)"],
    }


# --------------------------------------------------------- data quality --

QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "object",
            "description": "tag name to its series of values, newest last",
            "additionalProperties": {"type": "array", "items": {"type": "number"}},
        },
        "kinds": {
            "type": "object",
            "description": "optional tag name to physical kind, e.g. pressure, rate, temperature",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["tags"],
}


def data_quality(args: dict) -> dict:
    _require(args, "tags")
    tags = args["tags"]
    if not isinstance(tags, dict) or not tags:
        raise ValueError("tags must be a non-empty object of name to series")
    series = {str(k): [float(v) for v in vals] for k, vals in tags.items()}

    report = check(series, kinds=args.get("kinds") or None)

    # `blocked` is the list of unusable tags and `may_fit` the gate. One red
    # tag stops the lot, which is the difference between a quality report
    # and a quality gate -- a report everyone may override gets switched
    # off, so the boolean an agent reads is `may_fit`.
    notes = [tag.describe() for tag in report.tags if not tag.usable]
    if report.may_fit:
        notes.append("no blocking data-quality problem found")
    else:
        notes.append(
            "do not calibrate or optimise on this data until the tags above "
            "are resolved. Fitting a well model to a latched instrument "
            "anchors it to a number the process never had.")

    return {
        "summary": report.summary(),
        "outputs": {
            "may_fit": report.may_fit,
            "blocked_tags": [tag.tag for tag in report.blocked],
            "tags": [{"tag": tag.tag,
                      "kind": tag.kind,
                      "usable": tag.usable,
                      "severity": tag.severity,
                      "samples": tag.samples,
                      "finite": tag.finite,
                      "coverage": tag.coverage,
                      "findings": [{"code": f.code, "severity": f.severity,
                                    "detail": f.detail, "samples": f.samples}
                                   for f in tag.findings]}
                     for tag in report.tags],
        },
        "notes": notes,
        "correlations": ["frozen-tag detection against each tag's own "
                         "typical run length"],
    }


# ------------------------------------------------------------- integrity --

INTEGRITY_SCHEMA = {
    "type": "object",
    "properties": {
        "depths_ft": {"type": "array", "items": {"type": "number"}},
        "temperatures_f": {"type": "array", "items": {"type": "number"}},
        "pressures_psia": {"type": "array", "items": {"type": "number"}},
        "co2_fraction": {"type": "number", "description": "mole fraction"},
        "water_cut": {"type": "number", "description": "fraction, 0 to 1"},
        "velocity_ft_s": {"type": "number"},
        "h2s_fraction": {"type": "number", "description": "mole fraction"},
    },
    "required": ["depths_ft", "temperatures_f", "pressures_psia",
                 "co2_fraction", "water_cut"],
}


def integrity(args: dict) -> dict:
    _require(args, "depths_ft", "temperatures_f", "pressures_psia",
             "co2_fraction", "water_cut")
    report = assess(
        [float(d) for d in args["depths_ft"]],
        [float(t) for t in args["temperatures_f"]],
        [float(p) for p in args["pressures_psia"]],
        co2_fraction=float(args["co2_fraction"]),
        water_cut=float(args["water_cut"]),
        velocity_ft_s=float(args.get("velocity_ft_s", 0.0)),
        h2s_fraction=float(args.get("h2s_fraction", 0.0)),
    )

    if not report.usable:
        raise Refusal(report.describe())

    worst = report.worst
    return {
        "summary": report.describe(),
        "outputs": {
            "worst_depth_ft": worst.depth_ft,
            "worst_temperature_f": worst.temperature_f,
            "worst_rate_mm_yr": worst.rate_mm_yr,
            "worst_rate_mpy": worst.rate_mpy,
            "co2_partial_bar": worst.co2_partial_bar,
            "film_factor": worst.film_factor,
            "oil_wetted_throughout": report.oil_wetted_throughout,
            "basis": worst.basis,
        },
        "notes": list(report.notes),
        "correlations": ["de Waard & Milliams (1991) with siderite film"],
        "envelopes": ["an upper bound for ranking and siting inspections, "
                      "never a remaining-life figure; never validated "
                      "against a caliper log or a coupon"],
    }


# --------------------------------------------------- what it will not do --

SCALE_SCHEMA = {"type": "object", "properties": {}}


def scale_prediction(_args: dict) -> dict:
    """Deliberately refuses, and says what it would need.

    Present as a tool because an agent asked about scale will otherwise
    reason its way to an answer from nothing. A tool that refuses in the
    agent's own vocabulary is a stronger guard than the absence of a tool.
    """
    raise Refusal(
        "scale saturation is not implemented, and deliberately so. Calcite "
        "and barite indices need a water analysis -- %s -- and no archive "
        "this engine reads carries one. A saturation index computed from "
        "assumed chemistry is a number about the assumption. Ask the "
        "operator for these values." % "; ".join(scale_inputs()))


# ------------------------------------------------- what it actually did --

MEASURE_SCHEMA = {
    "type": "object",
    "properties": {
        "well": {"type": "string"},
        "change": {"type": "string",
                   "description": "what was done, in words"},
        "forecast_gain": {
            "type": "number",
            "description": "what the optimiser expected, per producing day"},
        "approved_by": {
            "type": "string",
            "description": "the engineer who approved it. Required, and a "
                           "person rather than a role: a record saying "
                           "'approved' without saying by whom cannot answer "
                           "the question an audit asks of one."},
        "before": {
            "type": "object",
            "description": "the window before the change",
            "properties": {
                "days": {"type": "number"},
                "total_volume": {"type": "number"},
                "days_on_production": {
                    "type": "number",
                    "description": "separates a real gain from a well that "
                                   "was simply up more"},
            },
            "required": ["days", "total_volume"],
        },
        "after": {
            "type": "object",
            "description": "the window after it",
            "properties": {
                "days": {"type": "number"},
                "total_volume": {"type": "number"},
                "days_on_production": {"type": "number"},
            },
            "required": ["days", "total_volume"],
        },
        "decline": {
            "type": "object",
            "description": "the well's own fitted decline, as returned by "
                           "decline_curve_fit. Without it the baseline is "
                           "held flat, which biases the answer in a "
                           "direction that depends on the well.",
            "properties": {
                "qi": {"type": "number"},
                "di_per_year": {"type": "number"},
                "b": {"type": "number"},
            },
        },
        "other_actions_on_this_well": {
            "type": "array", "items": {"type": "string"},
            "description": "ids of any other change made to this well in "
                           "the same window",
        },
    },
    "required": ["well", "forecast_gain", "approved_by", "before", "after"],
}


def _window(raw, which: str) -> Window:
    if not isinstance(raw, dict):
        raise ValueError("%s must be an object with days and total_volume"
                         % which)
    return Window(days=float(raw.get("days", 0.0)),
                  total_volume=float(raw.get("total_volume", 0.0)),
                  days_on_production=float(raw.get("days_on_production", 0.0)))


def measure_recommendation(args: dict) -> dict:
    _require(args, "well", "forecast_gain", "approved_by", "before", "after")

    action = Action(action_id=str(args.get("action_id", "measured")),
                    well=str(args["well"]),
                    change=str(args.get("change", "")),
                    forecast_gain=float(args["forecast_gain"]),
                    proposed_by="optimiser")
    try:
        action.approve(by=str(args["approved_by"])).execute()
    except ValueError as exc:
        raise Refusal(str(exc))

    decline = None
    raw = args.get("decline")
    if isinstance(raw, dict) and raw.get("qi"):
        decline = ArpsDecline(qi=float(raw["qi"]),
                              di_per_year=float(raw.get("di_per_year", 0.0)),
                              b=float(raw.get("b", 0.0)))

    # Only their existence matters here, not their contents: the rule is
    # that a second change to the same well in the same window makes
    # neither separately creditable.
    others = [Action(action_id=str(other), well=str(args["well"]),
                     change="", forecast_gain=0.0)
              for other in (args.get("other_actions_on_this_well") or [])]

    measurement = action.measure(_window(args["before"], "before"),
                                 _window(args["after"], "after"),
                                 decline=decline,
                                 other_actions_in_window=others)

    if not measurement.usable:
        raise Refusal(measurement.describe())

    return {
        "summary": ("%s: forecast %+.1f, realised %+.1f -- %.0f%% of "
                    "forecast."
                    % (action.well, measurement.forecast_gain,
                       measurement.realised_gain,
                       measurement.realised_fraction * 100.0)),
        "outputs": {
            "forecast_gain": measurement.forecast_gain,
            "realised_gain": measurement.realised_gain,
            "realised_fraction": measurement.realised_fraction,
            "baseline_rate": measurement.baseline_rate,
            "expected_rate_without_action":
                measurement.expected_rate_without_action,
            "actual_rate": measurement.actual_rate,
            "approved_by": action.decided_by,
            "basis": measurement.basis,
        },
        "notes": list(measurement.notes),
        "correlations": (["baseline carried forward along the well's own "
                          "Arps decline"] if decline else
                         ["baseline held flat -- no decline curve supplied"]),
        "envelopes": [measurement.basis],
    }


SCORECARD_SCHEMA = {
    "type": "object",
    "properties": {
        "measured": {
            "type": "array",
            "description": "one entry per measured action",
            "items": {
                "type": "object",
                "properties": {
                    "well": {"type": "string"},
                    "forecast_gain": {"type": "number"},
                    "realised_gain": {"type": "number"},
                },
                "required": ["forecast_gain", "realised_gain"],
            },
        },
        "not_measurable": {
            "type": "integer",
            "description": "actions that could not be scored, counted so "
                           "they cannot be quietly dropped"},
        "rejected": {
            "type": "integer",
            "description": "proposals an engineer declined"},
    },
    "required": ["measured"],
}


def recommendation_scorecard(args: dict) -> dict:
    """Field arithmetic over measurements the caller already took.

    This does not re-measure anything. Each realised gain was computed by
    `measure_recommendation`, with its own decline correction and its own
    refusals; what is added here is the field total and the one rule that
    only makes sense across actions -- that too few of them is not a track
    record.
    """
    _require(args, "measured")
    entries = args["measured"]
    if not isinstance(entries, list):
        raise ValueError("measured must be an array")
    if not entries:
        raise Refusal("no measured actions were supplied, so there is "
                      "nothing to score.")

    card = Scorecard()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("entry %d is not an object" % (index + 1))
        action = Action(action_id="A-%d" % (index + 1),
                        well=str(entry.get("well", "W-%d" % (index + 1))),
                        change="",
                        forecast_gain=float(entry["forecast_gain"]),
                        proposed_by="optimiser")
        action.approve(by="recorded").execute()
        action.measurement = Measurement(
            forecast_gain=float(entry["forecast_gain"]),
            realised_gain=float(entry["realised_gain"]),
            baseline_rate=0.0, expected_rate_without_action=0.0,
            actual_rate=0.0)
        action.state = MEASURED
        card.actions.append(action)

    # Counted rather than described, because the only thing the field
    # arithmetic needs from them is that they existed.
    for index in range(int(args.get("not_measurable") or 0)):
        blocked = Action(action_id="unmeasurable-%d" % index, well="-",
                         change="", forecast_gain=0.0)
        blocked.approve(by="recorded").execute()
        blocked.measurement = Measurement(
            forecast_gain=0.0, realised_gain=0.0, baseline_rate=0.0,
            expected_rate_without_action=0.0, actual_rate=0.0,
            usable=False,
            notes=["reported as not measurable by the caller"])
        card.actions.append(blocked)

    for index in range(int(args.get("rejected") or 0)):
        card.actions.append(
            Action(action_id="R-%d" % index, well="-", change="",
                   forecast_gain=0.0).reject(by="recorded"))

    fraction = card.realised_fraction()
    measured = [a for a in card.actions
                if a.measurement is not None and a.measurement.usable]

    notes = []
    if fraction is None:
        notes.append(
            "no field ratio is reported: there are %d measured actions, and "
            "fewer than %d is a story about individual wells rather than a "
            "track record -- it would be quoted as one."
            % (len(measured), MIN_ACTIONS_FOR_A_RATIO))
    else:
        notes.append(
            "losses are inside this total rather than excluded as outliers: "
            "%d of %d actions lost production against the well's own "
            "decline." % (card.losses, len(measured)))
    if card.rejected:
        notes.append(
            "%d of %d proposals were rejected by an engineer. How often "
            "that happens is a measure of the optimiser, which is why "
            "rejections are kept." % (card.rejected, card.proposed))

    summary = (("%.0f%% of forecast realised across %d measured actions"
                % (fraction * 100.0, len(measured))) if fraction is not None
               else ("%d measured actions -- too few for a field ratio"
                     % len(measured)))

    return {
        "summary": summary,
        "outputs": {
            "proposed": card.proposed,
            "approved": card.approved,
            "rejected": card.rejected,
            "approval_rate": card.approval_rate,
            "measured": len(measured),
            "not_measurable": len(card.actions) - len(measured)
                              - card.rejected,
            "total_forecast": card.total_forecast,
            "total_realised": card.total_realised,
            "realised_fraction": fraction,
            "losses": card.losses,
        },
        "notes": notes,
        "envelopes": ["association rather than causation: these wells did "
                      "better after the changes, which is not the same as "
                      "better because of them. Only a comparable well left "
                      "alone would support the stronger word."],
    }


# --------------------------------------------------- the reservoir itself --

RESERVOIR_SCHEMA = {
    "type": "object",
    "properties": {
        "deck": {
            "type": "string",
            "description": "path to an Eclipse-format .DATA deck with a "
                           "corner-point grid"},
        "approved_by": {
            "type": "string",
            "description": "the engineer approving the compute spend. "
                           "Required: a reservoir run is minutes to hours, "
                           "not milliseconds, and launching one is a "
                           "decision rather than a query."},
        "minutes_budget": {
            "type": "number",
            "description": "how long the run may take before it is killed. "
                           "A 9,000-cell benchmark takes about a minute "
                           "here; a 44,000-cell field model took 259 s in "
                           "the simulator's own published benchmark."},
        "out_dir": {
            "type": "string",
            "description": "where to write the results; defaults to a "
                           "temporary directory"},
    },
    "required": ["deck", "approved_by", "minutes_budget"],
}

# A reservoir run is the one tool on this surface that cannot answer inside
# a request. The others are milliseconds; this is minutes at best. So it
# needs both an approval and a budget, which is also what the product
# requirement says about launching runs above a compute budget.
MAX_MINUTES_BUDGET = 240.0


def reservoir_simulate(args: dict) -> dict:
    """Run a reservoir deck through JutulDarcy, in a subprocess.

    Refuses rather than fails where it cannot work, and the commonest such
    place is the deployed web application: JutulDarcy needs a 1.8 GB Julia
    depot and a ten-minute first-run install, so a serverless function
    cannot host it at all. Saying that plainly is more useful than a
    timeout.
    """
    _require(args, "deck", "approved_by", "minutes_budget")

    if not str(args["approved_by"]).strip():
        raise Refusal(
            "a reservoir run needs a named engineer's approval. It costs "
            "minutes to hours of compute rather than milliseconds, so "
            "launching one is a decision, and a decision nobody signed "
            "cannot answer the question an audit asks of it.")

    budget = float(args["minutes_budget"])
    if budget <= 0.0 or budget > MAX_MINUTES_BUDGET:
        raise Refusal(
            "the compute budget must be between 0 and %.0f minutes, got "
            "%.1f. An unbounded reservoir run is how someone's cluster "
            "disappears for a weekend."
            % (MAX_MINUTES_BUDGET, budget))

    if importlib.util.find_spec("jutuldarcy") is None:
        raise Refusal(
            "JutulDarcy is not installed in this deployment, and on a "
            "serverless host it cannot be: it needs a 1.8 GB Julia depot "
            "and a first-run install that took 639 s when measured. "
            "Reservoir simulation is a self-hosted capability -- an edge "
            "box or an on-premise server -- and this tool works there. "
            "Nothing is wrong with the request.")


    deck = str(args["deck"])
    if not os.path.isfile(deck):
        raise Refusal(
            "no deck at %s. This needs an Eclipse-format .DATA file with a "
            "corner-point grid; the simulator also ships SPE1, SPE9, EGG, "
            "OLYMPUS and NORNE as downloadable benchmarks." % deck)

    from phasespace.reservoir.jutul import simulate

    out_dir = args.get("out_dir") or tempfile.mkdtemp(prefix="phasespace-res-")
    try:
        run = simulate(deck, out_dir, timeout=budget * 60.0)
    except subprocess.TimeoutExpired:
        raise Refusal(
            "the run exceeded its %.0f-minute budget and was stopped. That "
            "is the budget working rather than the simulator failing; a "
            "larger model needs a larger budget, and the published "
            "benchmark for a 44,000-cell field is 259 s serial."
            % budget)
    except RuntimeError as exc:
        raise Refusal(str(exc))

    curves = run.field_curves
    summary = ["%d steps over %.0f days, %d wells, %d cells"
               % (run.steps, float(run.days[-1]) if run.steps else 0.0,
                  len(run.wells), run.cells)]
    outputs = {
        "steps": run.steps,
        "days": float(run.days[-1]) if run.steps else 0.0,
        "wells": run.wells,
        "cells": run.cells,
        "output_dir": str(out_dir),
        "solve_seconds": run.meta.get("solve_seconds"),
    }
    for key in ("FOPT", "FGPT", "FWPT", "FOPR", "FGPR", "FPR"):
        if key in curves:
            outputs[key] = float(curves[key][-1])
    if "FOPT" in outputs:
        summary.append("cumulative oil %.0f, field pressure %.0f psia at the "
                       "end" % (outputs["FOPT"], outputs.get("FPR", 0.0)))

    return {
        "summary": " -- ".join(summary),
        "outputs": outputs,
        "notes": list(run.corrections) + [
            "approved by %s against a %.0f-minute budget"
            % (args["approved_by"], budget),
            "the full result is on disk at %s: per-step field and well "
            "curves, and per-cell pressure, saturations and Rs" % out_dir],
        "correlations": ["JutulDarcy (SINTEF, MIT) black-oil / compositional "
                         "finite-volume solver"],
        "envelopes": [
            "verification only: the simulator's authors report matching "
            "AD-GPRS and MRST, which is code-to-code comparison and so says "
            "they share assumptions. Whether it describes a particular "
            "reservoir is a question only measured field data answers."],
    }


# ------------------------------------------------------------- building --

def build_registry() -> Registry:
    registry = Registry()
    registry.add(Tool(
        "well_pressure_traverse",
        "March the pressure down a well from the wellhead at a known rate, "
        "returning bottomhole pressure plus the pressure, holdup and flow "
        "regime at every depth. Validated to 2.9% median absolute error on "
        "150 measured gradient surveys.",
        TRAVERSE_SCHEMA, traverse))
    registry.add(Tool(
        "well_nodal_analysis",
        "Find a well's operating point where the inflow curve meets the "
        "tubing curve. Returns rate and bottomhole pressure. The traverse is "
        "validated; predicting the rate is not.",
        NODAL_SCHEMA, nodal))
    registry.add(Tool(
        "gas_lift_allocate",
        "Share a limited volume of lift gas across wells to maximise oil or "
        "value, respecting each well's stable-injection limits. Returns the "
        "allocation, the gain against current injection, and the shadow "
        "price of lift gas.",
        ALLOCATE_SCHEMA, gas_lift_allocate))
    registry.add(Tool(
        "decline_curve_fit",
        "Fit Arps to a production history and forecast EUR, with the spread "
        "that the range of admissible b exponents implies. Refuses a flat or "
        "rising history rather than forcing a fit.",
        DECLINE_SCHEMA, decline))
    registry.add(Tool(
        "data_quality_check",
        "Screen measurement series for frozen tags, impossible values, unit "
        "mismatches and gaps before any model uses them. Says whether the "
        "data may be fitted on.",
        QUALITY_SCHEMA, data_quality))
    registry.add(Tool(
        "well_integrity_assess",
        "Predict where CO2 corrosion attacks a string fastest, given the "
        "pressure and temperature traverse. Reports the depth, not just a "
        "rate, because the worst point is often not the hottest. Refuses "
        "sour gas.",
        INTEGRITY_SCHEMA, integrity))
    registry.add(Tool(
        "measure_recommendation",
        "Compare what a change actually delivered against what was "
        "forecast, with the well's own decline subtracted. Refuses "
        "when another change touched the same well, when uptime "
        "moved, or when the window is too short to mean anything.",
        MEASURE_SCHEMA, measure_recommendation))
    registry.add(Tool(
        "recommendation_scorecard",
        "The field track record across measured recommendations. "
        "Losses count, refusals stay visible, and below five "
        "measured actions it reports no ratio at all.",
        SCORECARD_SCHEMA, recommendation_scorecard))
    registry.add(Tool(
        "reservoir_simulate",
        "Run an Eclipse-format reservoir deck through JutulDarcy "
        "(SINTEF, MIT) in a subprocess. Minutes to hours rather "
        "than milliseconds, so it needs a named approval and a "
        "compute budget. Refuses on a serverless host, where the "
        "1.8 GB Julia depot cannot be installed.",
        RESERVOIR_SCHEMA, reservoir_simulate))
    registry.add(Tool(
        "scale_saturation",
        "Not implemented. Call it to learn precisely which water-analysis "
        "values an operator must supply before scale can be predicted at "
        "all.",
        SCALE_SCHEMA, scale_prediction))
    return registry
