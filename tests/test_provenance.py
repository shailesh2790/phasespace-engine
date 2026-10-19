"""The fingerprint has to describe the code beside it, or it describes nothing.

`provenance.py` exists so that a number computed eighteen months ago can be
traced to the equations that produced it, and the whole mechanism rests on
one property: the stamp in `_fingerprint.txt` is the hash of the engine
sitting next to it. `verify_fingerprint` was written to check exactly that
and then was never called from anywhere -- not from a test, not from the
application, not from the sync script.

It had been wrong for several commits. The source tree stamped
`da3d2f10dbda` while hashing to `b65ab5696742`, so every result the engine
produced outside the deployed copy named the wrong version of itself, and
the one function written to notice was never run.

That is the characteristic way a provenance claim fails. It does not break
loudly; it quietly stops being true, and because nothing reads it, nothing
says so. These tests read it.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from phasespace.sim.provenance import (
    Provenance,
    compute_fingerprint,
    engine_fingerprint,
    verify_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "web" / "api" / "_engine"


def test_the_source_stamp_matches_the_source():
    """Run `python scripts/sync_engine.py` if this fails. It is not a
    cosmetic mismatch: the stamp is what every result claims about which
    equations ran."""
    agrees, stamped, computed = verify_fingerprint()
    assert agrees, (
        "phasespace/sim/_fingerprint.txt says %s but the engine hashes to "
        "%s -- re-run scripts/sync_engine.py" % (stamped, computed))


@pytest.mark.skipif(not VENDORED.exists(), reason="no vendored engine")
def test_the_vendored_stamp_matches_the_vendored_engine():
    """Checked in a subprocess because the vendored copy imports itself as
    `_engine`, and importing both trees into one interpreter would leave it
    ambiguous which `provenance` module answered."""
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, '.');"
         "from _engine.provenance import verify_fingerprint;"
         "a, s, c = verify_fingerprint();"
         "print(a, s, c)"],
        cwd=str(VENDORED.parent), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    agrees, stamped, computed = result.stdout.split()
    assert agrees == "True", (
        "the deployed engine is stamped %s but hashes to %s -- a module was "
        "edited without re-running scripts/sync_engine.py, and the hosted "
        "app is reporting the wrong fingerprint" % (stamped, computed))


def test_the_two_trees_are_allowed_to_differ_but_not_to_be_wrong():
    """The vendored copy carries a subset of the modules, so the two
    fingerprints differ by design. Asserting they are equal would be the
    obvious test to write and it would be the wrong one -- it would start
    failing the moment a module is deliberately kept out of the deployment,
    and the fix would be to weaken it."""
    stamp = (VENDORED / "_fingerprint.txt")
    if not stamp.exists():
        pytest.skip("no vendored engine")
    assert len(stamp.read_text().strip()) == len(engine_fingerprint())


def test_changing_a_correlation_changes_the_fingerprint(tmp_path):
    """The property that makes the stamp worth reading. If a line of physics
    can change without the fingerprint moving, two runs can report the same
    version and different equations."""
    import phasespace.sim.provenance as prov

    before = compute_fingerprint()
    target = ROOT / "phasespace" / "sim" / "fluids.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# a correlation changed here\n")
        assert compute_fingerprint() != before
    finally:
        target.write_bytes(original)
    assert compute_fingerprint() == before


def test_an_evidence_bundle_is_byte_stable():
    """An auditor archives this and replays it later. Two serialisations of
    one result must be the same bytes, or the archive cannot be compared."""
    inputs = {"reservoir_psia": 3500.0, "pi": 2.0}
    outputs = {"rate_stb_d": 912.4}
    first = Provenance("nodal", inputs=inputs, outputs=outputs).to_json()
    second = Provenance("nodal",
                        inputs=dict(reversed(list(inputs.items()))),
                        outputs=outputs).to_json()
    assert first == second


def test_the_bundle_carries_the_fingerprint_it_was_computed_under():
    bundle = Provenance("nodal", inputs={"a": 1.0},
                        outputs={"b": 2.0}).to_dict()
    assert bundle["engine_fingerprint"] == engine_fingerprint()
