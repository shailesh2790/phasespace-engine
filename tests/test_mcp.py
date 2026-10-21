"""The agent-facing surface, and the guarantee it is supposed to carry.

Two things are tested here and the second is the point of the module.

The protocol: a client that speaks MCP can hand this server a session and
get valid JSON-RPC back, including when the request is nonsense.

The guarantee: *every* number that leaves this server carries a provenance
block with the engine fingerprint, and a tool that cannot answer refuses in
a sentence an agent can act on rather than raising or, worse, guessing. The
PRD's agent layer rests on "they never compute physics in their head", and
a prompt cannot enforce that. These tests are what enforcement looks like.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.mcp.protocol import PROTOCOL_VERSION, Server, serve
from phasespace.mcp.registry import Refusal, Registry, Tool
from phasespace.mcp.tools import build_registry
from phasespace.sim.provenance import engine_fingerprint


@pytest.fixture(scope="module")
def registry():
    return build_registry()


def call(registry, name, arguments):
    return registry.call(name, arguments)


def body(result):
    return result["content"][0]["text"]


def bundle(result):
    return result["_meta"]["provenance"]


# ------------------------------------------------------------- protocol --


def test_initialize_negotiates_a_version(registry):
    server = Server(registry)
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": PROTOCOL_VERSION}})
    result = reply["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["capabilities"]["tools"] == {"listChanged": False}
    assert result["serverInfo"]["name"] == "phasespace"


def test_tools_are_refused_before_the_handshake(registry):
    """Answering before initialize would let a client skip negotiation and
    then fail later in a way that looks like our fault."""
    server = Server(registry)
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "error" in reply
    assert "initialize" in reply["error"]["message"]


def test_a_notification_is_never_answered(registry):
    """A reply to a notification is a protocol violation and some clients
    treat it as fatal."""
    server = Server(registry)
    server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {}})
    assert server.handle({"jsonrpc": "2.0",
                          "method": "notifications/initialized"}) is None


def test_an_unknown_method_is_a_jsonrpc_error(registry):
    server = Server(registry)
    server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {}})
    reply = server.handle({"jsonrpc": "2.0", "id": 2, "method": "nope"})
    assert reply["error"]["code"] == -32601


def test_every_tool_publishes_a_schema_with_its_required_arguments(registry):
    schemas = registry.schemas()
    assert len(schemas) >= 7
    for schema in schemas:
        assert schema["name"]
        assert len(schema["description"]) > 40, schema["name"]
        assert schema["inputSchema"]["type"] == "object"


def test_a_whole_session_round_trips_over_stdio(registry):
    """End to end through the real transport, because the framing is where
    a stdio server usually breaks: one JSON object per line, flushed."""
    import io

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": PROTOCOL_VERSION}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "well_pressure_traverse",
                    "arguments": {"depth_ft": 8000.0, "tubing_id_in": 2.441,
                                  "wellhead_pressure_psia": 400.0,
                                  "gas_rate_scf_d": 1.5e6,
                                  "correlation": "hagedorn_brown"}}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    serve(registry, stdin=stdin, stdout=stdout)

    lines = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
    # Three requests carried an id; the notification must not have replied.
    assert [m["id"] for m in lines] == [1, 2, 3]
    assert all("\n" not in json.dumps(m) for m in lines)
    answer = lines[-1]["result"]
    assert answer["isError"] is False
    assert "510" in answer["content"][0]["text"]


def test_a_malformed_line_does_not_end_the_session(registry):
    import io
    stdin = io.StringIO('not json\n'
                        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n')
    stdout = io.StringIO()
    serve(registry, stdin=stdin, stdout=stdout)
    lines = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
    assert lines[0]["error"]["code"] == -32700
    assert lines[1]["result"]["serverInfo"]["name"] == "phasespace"


def test_what_a_tool_prints_cannot_corrupt_the_stream():
    """stdout belongs to the protocol. An engine module that grows a stray
    print must not take the server down, and the failure it would cause --
    a client that cannot parse our reply -- is miserable to diagnose."""
    import io

    def noisy(_args):
        print("this would corrupt the protocol stream")
        return {"summary": "fine", "outputs": {"value": 1}}

    registry = Registry()
    registry.add(Tool("noisy", "Prints to stdout, which it must not.",
                      {"type": "object", "properties": {}}, noisy))

    stdin = io.StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/call",'
        '"params":{"name":"noisy","arguments":{}}}\n')
    stdout = io.StringIO()
    serve(registry, stdin=stdin, stdout=stdout)

    for line in stdout.getvalue().splitlines():
        if line.strip():
            json.loads(line)          # every line is still valid JSON
    assert "corrupt" not in stdout.getvalue()


# -------------------------------------------- the provenance guarantee --


def test_every_tool_that_answers_carries_the_engine_fingerprint(registry):
    """The guarantee, checked across the whole surface rather than on one
    tool: a number without a provenance block did not come from here."""
    answered = 0
    for name, arguments in WORKING_CALLS.items():
        result = call(registry, name, arguments)
        assert result["isError"] is False, "%s: %s" % (name, body(result))
        provenance = bundle(result)
        assert provenance["engine_fingerprint"] == engine_fingerprint(), name
        assert provenance["calculation"] == name
        assert provenance["inputs"] == arguments, name
        assert provenance["outputs"], name
        assert engine_fingerprint() in body(result), name
        answered += 1
    assert answered >= 5


def test_the_bundle_is_byte_stable_so_two_runs_can_be_compared(registry):
    """An auditor's test. The same question twice must produce the same
    bundle digest, or the digest means nothing."""
    arguments = WORKING_CALLS["well_pressure_traverse"]
    first = call(registry, "well_pressure_traverse", arguments)
    second = call(registry, "well_pressure_traverse", arguments)
    assert json.dumps(bundle(first), sort_keys=True) == \
           json.dumps(bundle(second), sort_keys=True)


def test_a_refusal_is_a_readable_result_not_an_exception(registry):
    """An agent can reason about a sentence. It cannot reason about a
    transport error, and what it does instead is retry or invent."""
    result = call(registry, "scale_saturation", {})
    assert result["isError"] is True
    text = body(result)
    assert "not implemented" in text
    # And it names what it would need, so the agent can ask for it.
    for needed in ("calcium", "barium", "sulphate"):
        assert needed in text


def test_sour_gas_is_refused_through_the_tool_surface(registry):
    result = call(registry, "well_integrity_assess", {
        "depths_ft": [0.0, 5000.0], "temperatures_f": [100.0, 180.0],
        "pressures_psia": [2000.0, 3000.0], "co2_fraction": 0.02,
        "water_cut": 0.5, "h2s_fraction": 0.001})
    assert result["isError"] is True
    assert "MR0175" in body(result)


def test_a_rising_history_is_refused_rather_than_fitted(registry):
    result = call(registry, "decline_curve_fit",
                  {"days": [0, 30, 60, 90, 120, 150, 180],
                   "rates": [100, 110, 120, 130, 140, 150, 160]})
    assert result["isError"] is True
    assert "decline" in body(result).lower()


def test_a_well_producing_nothing_is_refused(registry):
    result = call(registry, "well_pressure_traverse",
                  {"depth_ft": 8000.0, "tubing_id_in": 2.441,
                   "wellhead_pressure_psia": 400.0})
    assert result["isError"] is True
    assert "producing nothing" in body(result)


def test_an_unknown_tool_lists_the_real_ones(registry):
    result = registry.call("make_me_money", {})
    assert result["isError"] is True
    assert "well_nodal_analysis" in body(result)


def test_bad_arguments_report_the_expected_schema(registry):
    """So an agent's second attempt can be better than its first."""
    result = call(registry, "decline_curve_fit",
                  {"days": [1, 2, 3], "rates": [1, 2]})
    assert result["isError"] is True
    assert "same length" in body(result)


# ------------------------------------------------- the physics it reports --


def test_the_traverse_tool_warns_when_the_default_correlation_is_wrong(registry):
    """The 150-survey result made this concrete: on a vertical dry gas well
    Beggs-Brill gave 13.0% median absolute error against Hagedorn-Brown's
    3.9%. The engine does not silently switch, so the tool has to say so --
    otherwise an agent picks the default and is quietly four times worse."""
    result = call(registry, "well_pressure_traverse",
                  {"depth_ft": 8000.0, "tubing_id_in": 2.441,
                   "wellhead_pressure_psia": 400.0, "gas_rate_scf_d": 1.5e6})
    assert result["isError"] is False
    assert "hagedorn_brown" in body(result)
    assert "13.0%" in body(result)


def test_nodal_says_the_intersection_is_not_validated(registry):
    """The distinction the validation record insists on travels with the
    answer, so it cannot be dropped between the engine and the user."""
    result = call(registry, "well_nodal_analysis",
                  WORKING_CALLS["well_nodal_analysis"])
    assert result["isError"] is False
    assert "NOT validated" in body(result)


def test_gas_lift_allocation_reports_a_shadow_price(registry):
    """FR-5 asks for the gain per action and the shadow price of each
    constraint. The shadow price is the whole economic answer: it is what
    another unit of compression is worth."""
    result = call(registry, "gas_lift_allocate",
                  WORKING_CALLS["gas_lift_allocate"])
    assert result["isError"] is False
    outputs = bundle(result)["outputs"]
    assert outputs["shadow_price"] > 0.0
    assert outputs["gas_used"] <= outputs["gas_available"] + 1e-9 \
        if "gas_available" in outputs else True
    assert len(outputs["wells"]) == 2


def test_gas_lift_refuses_a_well_with_too_few_tests(registry):
    """Four coefficients from three points is not a loose fit, it is an
    underdetermined system that returns a confident meaningless answer."""
    result = call(registry, "gas_lift_allocate", {
        "wells": [{"name": "A", "tests": [[0, 100], [500, 140], [1000, 160]]}],
        "gas_available": 1000.0})
    assert result["isError"] is True
    assert "at least 4" in body(result)


def test_data_quality_blocks_a_frozen_tag(registry):
    """FR-2 before FR-3: a frozen tag must stop a calibration rather than
    silently anchor it."""
    result = call(registry, "data_quality_check",
                  {"tags": {"WHP": [250.0] * 60,
                            "RATE": [100.0 + i * 0.7 for i in range(60)]}})
    assert result["isError"] is False
    outputs = bundle(result)["outputs"]
    # `may_fit` is the gate an agent reads: one red tag stops the lot.
    assert outputs["may_fit"] is False
    assert outputs["blocked_tags"] == ["WHP"]
    assert any("latched" in n for n in bundle(result)["notes"])


def test_integrity_reports_where_not_just_whether(registry):
    """The result worth having. On a real well the peak sat at 3,194 ft
    while the bottom was 9,747 ft, because the protective film forms above
    60 degC -- so an inspection at total depth looks in the wrong place."""
    result = call(registry, "well_integrity_assess", {
        "depths_ft": [0.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0],
        "temperatures_f": [100.0, 130.0, 160.0, 185.0, 205.0, 215.0],
        "pressures_psia": [3000.0, 3200.0, 3400.0, 3600.0, 3800.0, 4000.0],
        "co2_fraction": 0.02, "water_cut": 0.6})
    assert result["isError"] is False
    outputs = bundle(result)["outputs"]
    assert outputs["worst_depth_ft"] < 10000.0
    assert "upper bound" in outputs["basis"]


# A call per tool that should succeed, used by the sweep above so that a new
# tool cannot be added without someone deciding what a working call to it
# looks like.
WORKING_CALLS = {
    "well_pressure_traverse": {
        "depth_ft": 8000.0, "tubing_id_in": 2.441,
        "wellhead_pressure_psia": 400.0, "gas_rate_scf_d": 1.5e6,
        "correlation": "hagedorn_brown",
    },
    "well_nodal_analysis": {
        "reservoir_pressure_psia": 3000.0, "max_rate_stb_d": 2000.0,
        "depth_ft": 8000.0, "tubing_id_in": 2.441,
        "wellhead_pressure_psia": 200.0,
        "gas_oil_ratio_scf_stb": 500.0, "water_cut": 0.2,
    },
    "gas_lift_allocate": {
        "wells": [
            {"name": "A", "tests": [[0, 100], [500, 140], [1000, 160],
                                    [1500, 168], [2000, 170]],
             "current": 500.0, "q_max": 3000.0},
            {"name": "B", "tests": [[0, 80], [500, 125], [1000, 150],
                                    [1500, 162], [2000, 165]],
             "current": 500.0, "q_max": 3000.0},
        ],
        "gas_available": 2500.0,
    },
    "decline_curve_fit": {
        "days": [0, 60, 120, 180, 240, 300, 360, 420, 480, 540],
        "rates": [1000.0, 862.0, 771.0, 704.0, 652.0, 610.0, 575.0,
                  545.0, 519.0, 497.0],
    },
    "data_quality_check": {
        "tags": {"WHP": [250.0 + i * 0.3 for i in range(60)],
                 "RATE": [100.0 + i * 0.7 for i in range(60)]},
    },
    "well_integrity_assess": {
        "depths_ft": [0.0, 5000.0, 9000.0],
        "temperatures_f": [100.0, 170.0, 205.0],
        "pressures_psia": [2000.0, 2800.0, 3400.0],
        "co2_fraction": 0.02, "water_cut": 0.5,
    },
}


# ------------------------------------------- measuring what was delivered --

CLEAN_BEFORE = {"days": 30.0, "total_volume": 3000.0}
CLEAN_AFTER = {"days": 30.0, "total_volume": 3240.0}
DECLINE = {"qi": 100.0, "di_per_year": 0.30, "b": 0.5}


def test_measuring_a_recommendation_subtracts_the_wells_decline(registry):
    """The bias runs the way people do not expect. Without the correction
    this reads as +8.0; the well would have fallen to 97.6 on its own, so
    the action is worth 10.4 against where it was actually heading."""
    result = call(registry, "measure_recommendation", {
        "well": "W-1", "forecast_gain": 10.0, "approved_by": "S. Tripathi",
        "before": CLEAN_BEFORE, "after": CLEAN_AFTER, "decline": DECLINE})

    assert result["isError"] is False
    outputs = bundle(result)["outputs"]
    assert outputs["expected_rate_without_action"] == pytest.approx(97.6, abs=0.1)
    assert outputs["realised_gain"] == pytest.approx(10.4, abs=0.1)
    assert outputs["realised_fraction"] > 1.0
    assert "association, not causation" in outputs["basis"]


def test_an_unsigned_approval_is_refused(registry):
    """A record saying 'approved' without saying by whom cannot answer the
    only question an audit asks of one afterwards."""
    result = call(registry, "measure_recommendation", {
        "well": "W-1", "forecast_gain": 10.0, "approved_by": "",
        "before": CLEAN_BEFORE, "after": CLEAN_AFTER})
    assert result["isError"] is True
    assert "named person" in body(result)


def test_a_second_change_on_the_same_well_is_refused(registry):
    result = call(registry, "measure_recommendation", {
        "well": "W-1", "forecast_gain": 10.0, "approved_by": "S. Tripathi",
        "before": CLEAN_BEFORE, "after": CLEAN_AFTER,
        "other_actions_on_this_well": ["A-7"]})
    assert result["isError"] is True
    assert "A-7" in body(result)
    assert "separately credited" in body(result)


def test_uptime_drift_is_refused_through_the_tool(registry):
    result = call(registry, "measure_recommendation", {
        "well": "W-1", "forecast_gain": 10.0, "approved_by": "S. Tripathi",
        "before": {"days": 30.0, "total_volume": 1500.0,
                   "days_on_production": 15.0},
        "after": {"days": 30.0, "total_volume": 3000.0,
                  "days_on_production": 30.0}})
    assert result["isError"] is True
    assert "uptime moved" in body(result)


def test_the_scorecard_refuses_a_ratio_from_too_few_actions(registry):
    result = call(registry, "recommendation_scorecard", {
        "measured": [{"forecast_gain": 10.0, "realised_gain": 8.0},
                     {"forecast_gain": 10.0, "realised_gain": 9.0}]})
    assert result["isError"] is False
    assert bundle(result)["outputs"]["realised_fraction"] is None
    assert "too few" in body(result)


def test_the_scorecard_counts_losses_in_the_total(registry):
    """Four actions at +8 and one at -10 against 50 forecast is 44%, not
    the 64% you get by dropping the loss as an outlier."""
    result = call(registry, "recommendation_scorecard", {
        "measured": [{"forecast_gain": 10.0, "realised_gain": 8.0}] * 4
                    + [{"forecast_gain": 10.0, "realised_gain": -10.0}]})
    outputs = bundle(result)["outputs"]
    assert outputs["losses"] == 1
    assert outputs["total_realised"] == pytest.approx(22.0)
    assert outputs["realised_fraction"] == pytest.approx(0.44)


def test_the_scorecard_keeps_refusals_and_rejections_visible(registry):
    """An unmeasurable action that vanished would flatter the ratio by
    selection, and the rejection count is a measure of the optimiser."""
    result = call(registry, "recommendation_scorecard", {
        "measured": [{"forecast_gain": 10.0, "realised_gain": 8.0}] * 5,
        "not_measurable": 2, "rejected": 3})
    outputs = bundle(result)["outputs"]
    assert outputs["not_measurable"] == 2
    assert outputs["rejected"] == 3
    assert outputs["approval_rate"] == pytest.approx(7 / 10)
    assert any("rejected by an engineer" in n for n in bundle(result)["notes"])


def test_an_empty_scorecard_refuses_rather_than_reporting_zero(registry):
    result = call(registry, "recommendation_scorecard", {"measured": []})
    assert result["isError"] is True
    assert "nothing to score" in body(result)


# ------------------------------------------------- the reservoir, guarded --


def test_a_reservoir_run_needs_a_named_approval(registry):
    """It costs minutes to hours of compute rather than milliseconds, so
    launching one is a decision and not a query."""
    result = call(registry, "reservoir_simulate",
                  {"deck": "x.DATA", "approved_by": "", "minutes_budget": 10})
    assert result["isError"] is True
    assert "named engineer's approval" in body(result)


@pytest.mark.parametrize("budget", [0.0, -5.0, 10_000.0])
def test_an_unbounded_compute_budget_is_refused(registry, budget):
    result = call(registry, "reservoir_simulate",
                  {"deck": "x.DATA", "approved_by": "S. Tripathi",
                   "minutes_budget": budget})
    assert result["isError"] is True
    assert "compute budget must be between" in body(result)


def test_a_missing_deck_names_the_benchmarks_that_do_exist(registry):
    """An agent that cannot find a deck can at least be told which ones
    ship with the simulator."""
    result = call(registry, "reservoir_simulate",
                  {"deck": "nowhere.DATA", "approved_by": "S. Tripathi",
                   "minutes_budget": 10})
    assert result["isError"] is True
    assert "SPE9" in body(result)


def test_a_host_without_julia_says_so_rather_than_timing_out(registry,
                                                             monkeypatch):
    """The deployed web application is exactly this case: JutulDarcy needs
    a 1.8 GB Julia depot and a first-run install measured at 639 s, so a
    serverless function cannot host it at all. Saying that is more useful
    than a timeout, and the check comes before the deck check because
    naming the absent capability beats naming a file it was never going
    to find.
    """
    import importlib.util as util
    real = util.find_spec

    def absent(name, *args, **kwargs):
        if name == "jutuldarcy":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(util, "find_spec", absent)
    result = call(registry, "reservoir_simulate",
                  {"deck": "nowhere.DATA", "approved_by": "S. Tripathi",
                   "minutes_budget": 10})
    assert result["isError"] is True
    assert "self-hosted capability" in body(result)
    assert "Nothing is wrong with the request" in body(result)


def test_the_reservoir_tool_declares_its_compute_cost_in_the_schema(registry):
    """So a model choosing between tools knows this one is not free."""
    schema = next(t for t in registry.schemas()
                  if t["name"] == "reservoir_simulate")
    assert "minutes_budget" in schema["inputSchema"]["required"]
    assert "approved_by" in schema["inputSchema"]["required"]
    assert "milliseconds" in schema["description"]
