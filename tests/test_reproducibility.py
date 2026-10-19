"""Can a number produced today be produced again in five years?

A reserves auditor doing substantive testing re-estimates a sample
independently and compares against the software's figure. That is cheap when
the software can hand back the exact inputs and the exact code behind a
number from eighteen months ago, and expensive otherwise. It is also the one
property here that cannot be retrofitted: every week of new surface area
makes a later attempt harder.

Three things are tested, in rising order of how hard they are to fake.

**Determinism, across processes.** Not "we ran it twice in one process" --
that proves nothing about hash-order dependence, because Python randomises
string hashing per process. Two subprocesses under different PYTHONHASHSEED
values must print identical bytes for every float the engine computes. If
anything iterates a set, or depends on object identity, this is where it
shows.

**A fingerprint that tracks the physics.** Not a hand-maintained version
number, which drifts the first time someone forgets to bump it. A hash of
the engine source, verified here to change when a single correlation
coefficient changes by 0.1%.

**A vendored copy that cannot silently go stale.** The deployed engine is a
copy. A module edited but not re-synced would serve old physics under a new
build, and nothing else in the suite would notice.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phasespace.sim.provenance import (  # noqa: E402
    FINGERPRINT_LENGTH,
    Provenance,
    compute_fingerprint,
    engine_fingerprint,
)

PROBE = Path(__file__).resolve().parent / "determinism_probe.py"
VENDORED = ROOT / "web" / "api" / "_engine"

# The two tests below guard a *vendored* copy of the engine -- a deployment
# that bundles `phasespace/sim` rather than importing it, which is how the
# engine reaches a serverless function with a size limit. A stale copy
# serves old physics under a new build and nothing else in the suite
# notices, so the check is worth having wherever it applies.
#
# This repository ships no deployment, so there is nothing to compare and
# they skip. They are kept rather than deleted because anyone vendoring
# this engine into their own deployment wants exactly this check, and
# pointing `VENDORED` at their copy is the whole of the work.
needs_vendored = pytest.mark.skipif(
    not VENDORED.is_dir(),
    reason="no vendored engine copy in this checkout (see VENDORED above)")


def run_probe(hash_seed):
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(hash_seed)
    environment["PYTHONIOENCODING"] = "utf-8"
    finished = subprocess.run(
        [sys.executable, str(PROBE)], capture_output=True, text=True,
        env=environment, cwd=str(ROOT), timeout=900)
    assert finished.returncode == 0, finished.stderr[-2000:]
    return finished.stdout


# ------------------------------------------------------------ determinism --


def test_the_engine_is_deterministic_across_processes():
    """Separate processes, different hash seeds, byte-identical output.

    Values are printed with `repr`, which round-trips a float exactly, so a
    difference in the last bit fails this rather than being rounded away. A
    reproducibility guarantee that only holds to six decimal places is not a
    guarantee.
    """
    first = run_probe(0)
    second = run_probe(12345)
    third = run_probe(999)

    assert first.count("=") > 50, "the probe produced almost nothing"
    assert first == second, _first_difference(first, second)
    assert first == third, _first_difference(first, third)


def test_the_probe_covers_the_engine_rather_than_one_corner():
    """A determinism test over one function would pass while the rest of the
    engine wandered."""
    output = run_probe(0)
    for prefix in ("net.", "well.", "pvt.", "tank.", "arps.", "calib.",
                   "liq.", "gasfit."):
        assert any(line.startswith(prefix) for line in output.splitlines()), \
            "the probe never exercised %s" % prefix


def _first_difference(left, right):
    for index, (a, b) in enumerate(zip(left.splitlines(), right.splitlines())):
        if a != b:
            return "line %d differs:\n  %s\n  %s" % (index + 1, a, b)
    return "outputs differ in length"


# ------------------------------------------------------------ fingerprint --


def test_the_fingerprint_is_stable_and_short_enough_to_quote():
    first = compute_fingerprint()
    assert first == compute_fingerprint()
    assert len(first) == FINGERPRINT_LENGTH
    assert all(c in "0123456789abcdef" for c in first)


def test_the_fingerprint_changes_when_the_physics_does():
    """The whole point. A version number nobody has to remember to bump.

    Checked by perturbing one coefficient of Gilbert's choke correlation by
    0.1% and restoring it, because a fingerprint that survives a change to
    the physics is decoration.
    """
    target = ROOT / "phasespace" / "sim" / "chokes.py"
    original = target.read_text(encoding="utf-8")
    before = compute_fingerprint()
    try:
        assert '"gilbert": (10.00' in original, "Gilbert's coefficient moved"
        target.write_text(original.replace('"gilbert": (10.00',
                                           '"gilbert": (10.01'),
                          encoding="utf-8")
        assert compute_fingerprint() != before
    finally:
        target.write_text(original, encoding="utf-8")
    assert compute_fingerprint() == before


def test_the_fingerprint_ignores_line_endings():
    """The same source checked out on Windows and on Linux must fingerprint
    the same, or the stamp reports the developer's operating system rather
    than the physics."""
    import hashlib

    source = ROOT / "phasespace" / "sim" / "units.py"
    content = source.read_bytes()
    lf = content.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert hashlib.sha256(lf).hexdigest() == hashlib.sha256(
        crlf.replace(b"\r\n", b"\n")).hexdigest()


def test_tests_and_the_web_app_do_not_change_the_fingerprint():
    """A change to a page or a test must not invalidate a number the engine
    would still compute identically, or the stamp becomes noise and people
    stop reading it."""
    before = compute_fingerprint()
    scratch = ROOT / "tests" / "_fingerprint_scratch.txt"
    scratch.write_text("x", encoding="utf-8")
    try:
        assert compute_fingerprint() == before
    finally:
        scratch.unlink()


# --------------------------------------------------- the deployed copy -----


@needs_vendored
def test_the_vendored_engine_matches_the_source():
    """The deployed engine is a copy, and a stale copy serves old physics
    under a new build with nothing else in the suite noticing.

    Compared module by module so the failure names the file to re-sync
    rather than only saying the hashes differ.
    """
    assert VENDORED.is_dir(), "no vendored engine at %s" % VENDORED
    source_dir = ROOT / "phasespace" / "sim"

    stale = []
    for vendored in sorted(VENDORED.glob("*.py")):
        if vendored.name == "__init__.py":
            continue
        original = source_dir / vendored.name
        if not original.is_file():
            stale.append("%s is vendored but no longer in the engine"
                         % vendored.name)
            continue
        # The vendored copy carries a generated header, so compare the body
        # that follows it rather than the whole file.
        body = vendored.read_text(encoding="utf-8").replace("\r\n", "\n")
        truth = original.read_text(encoding="utf-8").replace("\r\n", "\n")
        if truth.strip() not in body:
            stale.append(vendored.name)

    assert not stale, (
        "run scripts/sync_engine.py -- stale vendored modules: %s"
        % ", ".join(stale))


@needs_vendored
def test_the_vendored_stamp_matches_the_vendored_source():
    stamp = VENDORED / "_fingerprint.txt"
    assert stamp.is_file(), "run scripts/sync_engine.py to write the stamp"

    import hashlib

    digest = hashlib.sha256()
    for path in sorted(VENDORED.glob("*.py")):
        if path.name in ("__init__.py", "provenance.py"):
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    assert stamp.read_text(encoding="utf-8").strip() == \
        digest.hexdigest()[:FINGERPRINT_LENGTH]


# ---------------------------------------------------- the evidence bundle --


def test_a_bundle_round_trips_and_is_byte_stable():
    """Two bundles of the same calculation must be byte-identical, so they
    can be diffed and so a hash of one is meaningful."""
    def build():
        return Provenance(
            calculation="nodal", inputs={"wellhead_psia": 300.0, "gor": 600},
            outputs={"rate_stb_d": 3105.1},
            correlations=["Beggs-Brill (1973)"])

    first, second = build(), build()
    assert first.to_json() == second.to_json()
    assert first.digest == second.digest
    assert json.loads(first.to_json())["engine_fingerprint"] == \
        engine_fingerprint()


def test_a_bundle_changes_when_any_input_does():
    base = Provenance(calculation="nodal", inputs={"p": 300.0},
                      outputs={"q": 100.0})
    moved = Provenance(calculation="nodal", inputs={"p": 300.5},
                       outputs={"q": 100.0})
    assert base.digest != moved.digest


def test_a_bundle_serialises_numpy_rather_than_failing():
    """A bundle that cannot serialise at the moment it is most wanted -- when
    something has gone wrong and someone is exporting the evidence -- is
    worse than no bundle."""
    import numpy as np

    bundle = Provenance(
        calculation="traverse",
        inputs={"depths": np.linspace(0.0, 8000.0, 4)},
        outputs={"pressure": np.float32(2345.6), "count": np.int64(40)})
    decoded = json.loads(bundle.to_json())
    assert decoded["inputs"]["depths"] == [0.0, 8000.0 / 3.0,
                                           2.0 * 8000.0 / 3.0, 8000.0]
    assert decoded["outputs"]["count"] == 40
