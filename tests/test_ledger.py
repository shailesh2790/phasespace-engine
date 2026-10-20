"""The append-only record, and the limits of what a hash chain proves.

Most of these tests are about detection rather than storage. Writing a line
to a file is not interesting; what matters is that an edited, truncated or
reordered ledger stops verifying, and that the failure says where.

The last group is the one worth reading. A hash chain is tamper-*evident*,
not tamper-proof, and the test that demonstrates a clean rewrite passing
verification is there deliberately. A guarantee nobody has written down the
limits of gets quoted without them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.ledger import GENESIS, Ledger
from phasespace.sim.provenance import Provenance


def bundle(name="well_pressure_traverse", value=510.8):
    return Provenance(calculation=name,
                      inputs={"depth_ft": 8000.0},
                      outputs={"bottomhole_pressure_psia": value},
                      correlations=["hagedorn_brown"])


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "evidence.jsonl")


# ------------------------------------------------------------- writing --


def test_a_provenance_object_can_be_appended_directly(ledger):
    """Accepting the object as well as the dict keeps the call site honest.
    The alternative is every caller remembering to serialise, and the one
    that forgets stores a repr."""
    record = ledger.append(bundle())
    assert record.index == 0
    assert record.previous == GENESIS
    assert record.calculation == "well_pressure_traverse"
    assert len(record.digest) == 64


def test_records_chain_to_one_another(ledger):
    first = ledger.append(bundle(value=1.0))
    second = ledger.append(bundle(value=2.0))
    third = ledger.append(bundle(value=3.0))

    assert [r.index for r in (first, second, third)] == [0, 1, 2]
    assert second.previous == first.digest
    assert third.previous == second.digest
    assert ledger.head() == third.digest


def test_the_file_is_one_json_object_per_line(tmp_path):
    """The format an auditor can read without asking for the software."""
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    led.append(bundle(value=1.0))
    led.append(bundle(value=2.0))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_an_empty_ledger_reads_as_empty_rather_than_failing(ledger):
    assert len(ledger) == 0
    assert ledger.head() == GENESIS
    assert ledger.verify().ok
    assert ledger.export() == ""


def test_a_record_must_be_a_mapping(ledger):
    with pytest.raises(TypeError):
        ledger.append("a number I liked")


# ------------------------------------------------------------ detection --


def test_an_intact_ledger_verifies(ledger):
    for i in range(5):
        ledger.append(bundle(value=float(i)))
    verdict = ledger.verify()
    assert verdict.ok
    assert verdict.records == 5
    assert "5 records" in verdict.describe()


def test_editing_a_value_breaks_the_chain(tmp_path):
    """The case this exists for: someone improves a number after the fact."""
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    for i in range(4):
        led.append(bundle(value=float(i)))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    doctored = json.loads(lines[1])
    doctored["payload"]["outputs"]["bottomhole_pressure_psia"] = 9999.0
    lines[1] = json.dumps(doctored, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = led.verify()
    assert not verdict.ok
    assert verdict.first_bad_index == 1
    assert "do not match" in verdict.reason


def test_removing_a_record_from_the_middle_is_caught(tmp_path):
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    for i in range(4):
        led.append(bundle(value=float(i)))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    del lines[2]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = led.verify()
    assert not verdict.ok
    assert "removed or reordered" in verdict.reason


def test_truncating_the_beginning_is_caught(tmp_path):
    """A fixed genesis digest is what makes this detectable. With an empty
    string for the first `previous`, a file with its head cut off would
    read as a shorter intact ledger."""
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    for i in range(4):
        led.append(bundle(value=float(i)))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    path.write_text("\n".join(lines[2:]) + "\n", encoding="utf-8")

    verdict = led.verify()
    assert not verdict.ok
    assert verdict.first_bad_index == 2


def test_reordering_two_records_is_caught(tmp_path):
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    for i in range(4):
        led.append(bundle(value=float(i)))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert not led.verify().ok


def test_a_corrupt_line_raises_rather_than_being_skipped(tmp_path):
    """Skipping an unparseable line would make a corrupt ledger read as a
    shorter intact one."""
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    led.append(bundle())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")

    with pytest.raises(ValueError, match="line 2"):
        list(led.records())
    assert not led.verify().ok


# ------------------------------------------------- the limits, on purpose --


def test_a_whole_rewrite_verifies_cleanly_and_that_is_the_known_limit(tmp_path):
    """Tamper-*evidence*, not tamper-proofing.

    Anyone who can write the file can rebuild the chain from their edit
    onwards, and the result verifies. The chain raises the cost of a quiet
    change from editing one number to rewriting every record after it, and
    it makes a careless edit certain to be caught. It is not a vault, and
    this test is here so that nobody quotes the guarantee without it.

    What closes the gap is anchoring `head()` somewhere the writer cannot
    reach -- which is why that method exists and is documented as the thing
    to publish.
    """
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    for i in range(3):
        led.append(bundle(value=float(i)))
    original_head = led.head()

    # A determined editor: change record 1, then re-append the rest.
    path.unlink()
    rebuilt = Ledger(path)
    rebuilt.append(bundle(value=0.0))
    rebuilt.append(bundle(value=9999.0))          # the altered one
    rebuilt.append(bundle(value=2.0))

    assert rebuilt.verify().ok, "a full rewrite is self-consistent"
    # And this is how it is caught: the head no longer matches what was
    # published elsewhere.
    assert rebuilt.head() != original_head


def test_the_head_is_what_an_operator_publishes(ledger):
    ledger.append(bundle())
    first = ledger.head()
    ledger.append(bundle(value=1.0))
    assert ledger.head() != first
    assert len(ledger.head()) == 64


# ---------------------------------------------------------------- export --


def test_export_returns_the_lines_as_written(tmp_path):
    """Byte-identical rather than re-serialised, so what an auditor
    verifies is what was stored."""
    path = tmp_path / "evidence.jsonl"
    led = Ledger(path)
    for i in range(3):
        led.append(bundle(value=float(i)))

    assert led.export() == path.read_text(encoding="utf-8")


def test_export_can_start_from_a_point(ledger):
    for i in range(5):
        ledger.append(bundle(value=float(i)))
    tail = ledger.export(since_index=3)
    assert len(tail.strip().split("\n")) == 2


def test_a_summary_says_what_is_in_here(ledger):
    ledger.append(bundle("well_pressure_traverse"))
    ledger.append(bundle("well_pressure_traverse"))
    ledger.append(bundle("gas_lift_allocate"))

    summary = ledger.summary()
    assert summary["records"] == 3
    assert summary["by_calculation"]["well_pressure_traverse"] == 2
    assert summary["by_calculation"]["gas_lift_allocate"] == 1
    assert summary["head"] == ledger.head()
    # Every bundle carries the engine that produced it, so a ledger
    # spanning an upgrade shows both and an auditor knows to expect two
    # sets of numbers.
    assert len(summary["engines"]) == 1


def test_a_ledger_spanning_an_engine_change_shows_both(ledger):
    first = bundle()
    first.engine = "aaaaaaaaaaaa"
    second = bundle()
    second.engine = "bbbbbbbbbbbb"
    ledger.append(first)
    ledger.append(second)
    assert ledger.summary()["engines"] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]


def test_the_digest_does_not_depend_on_key_order(tmp_path):
    """Two runs that produced the same bundle must hash the same, or the
    digest is a hash of dictionary ordering rather than of the evidence."""
    a = Ledger(tmp_path / "a.jsonl")
    b = Ledger(tmp_path / "b.jsonl")
    stamp = "2026-01-01T00:00:00Z"

    one = a.append({"calculation": "x", "alpha": 1, "beta": 2}, recorded_at=stamp)
    two = b.append({"calculation": "x", "beta": 2, "alpha": 1}, recorded_at=stamp)
    assert one.digest == two.digest


# ------------------------------------------- wired into the tool surface --


def test_a_registry_with_a_ledger_records_every_answer(tmp_path):
    """The point of the whole file: evidence that outlives the reply."""
    from phasespace.mcp.tools import build_registry

    led = Ledger(tmp_path / "calls.jsonl")
    registry = build_registry()
    registry.ledger = led

    registry.call("well_pressure_traverse",
                  {"depth_ft": 8000.0, "tubing_id_in": 2.441,
                   "wellhead_pressure_psia": 400.0, "gas_rate_scf_d": 1.5e6,
                   "gas_gravity": 0.60, "correlation": "hagedorn_brown"})

    assert len(led) == 1
    record = next(led.records())
    assert record.calculation == "well_pressure_traverse"
    assert record.payload["outputs"]["bottomhole_pressure_psia"] == \
        pytest.approx(510.8, abs=0.5)
    assert led.verify().ok


def test_a_refusal_is_recorded_too(tmp_path):
    """What was asked and why it could not be answered is exactly what an
    auditor reconstructing a decision needs. Recording only the successes
    would leave the most interesting moments missing."""
    from phasespace.mcp.tools import build_registry

    led = Ledger(tmp_path / "calls.jsonl")
    registry = build_registry()
    registry.ledger = led

    result = registry.call("scale_saturation", {})
    assert result["isError"] is True

    assert len(led) == 1
    record = next(led.records())
    assert record.calculation == "scale_saturation"
    assert "refused" in record.payload["outputs"]


def test_the_answer_reports_the_digest_it_was_recorded_under(tmp_path):
    """So a caller can cite the record without reading the file."""
    from phasespace.mcp.tools import build_registry

    led = Ledger(tmp_path / "calls.jsonl")
    registry = build_registry()
    registry.ledger = led

    result = registry.call("scale_saturation", {})
    assert result["_meta"]["ledger_digest"] == led.head()


def test_without_a_ledger_nothing_is_recorded_and_nothing_breaks(tmp_path):
    """The default. A serverless deployment has nowhere durable to write."""
    from phasespace.mcp.tools import build_registry

    registry = build_registry()
    assert registry.ledger is None
    result = registry.call("scale_saturation", {})
    assert result["_meta"]["ledger_digest"] == ""


def test_a_broken_ledger_does_not_swallow_the_answer(tmp_path):
    """The caller asked a question and the engine answered it. A storage
    failure must be reported, not allowed to discard the answer -- but it
    must be visible rather than silent."""
    from phasespace.mcp.tools import build_registry

    class Broken:
        def append(self, _bundle):
            raise OSError("read-only filesystem")

    registry = build_registry()
    registry.ledger = Broken()
    result = registry.call("scale_saturation", {})

    assert "NOT recorded in the audit ledger" in result["content"][0]["text"]
    assert "read-only filesystem" in result["content"][0]["text"]
    # The engine's own answer survives alongside the warning.
    assert "calcium" in result["content"][0]["text"]


def test_a_session_of_calls_verifies_as_one_chain(tmp_path):
    from phasespace.mcp.tools import build_registry

    led = Ledger(tmp_path / "calls.jsonl")
    registry = build_registry()
    registry.ledger = led

    registry.call("scale_saturation", {})
    registry.call("well_pressure_traverse",
                  {"depth_ft": 8000.0, "tubing_id_in": 2.441,
                   "wellhead_pressure_psia": 400.0, "gas_rate_scf_d": 1.5e6})
    registry.call("data_quality_check",
                  {"tags": {"WHP": [250.0 + i * 0.3 for i in range(60)]}})

    verdict = led.verify()
    assert verdict.ok and verdict.records == 3
    assert led.summary()["by_calculation"]["scale_saturation"] == 1
