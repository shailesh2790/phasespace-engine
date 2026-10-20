"""What produced this number, and can it be produced again.

A reserves auditor performing substantive testing re-estimates a sample
independently and compares. That is cheap if the software can hand back the
exact inputs and the exact code version behind a figure from eighteen months
ago, and expensive otherwise -- and the SPE auditing standard already tells
them to be cautious of "proprietary or commercial software without a full
understanding of the internal mathematical algorithms and correlations."
Reproducibility is the part of that caution a vendor can actually answer.

It is also the one property that cannot be retrofitted. Every week of new
surface area makes a later attempt harder, which is why this lands before
the features that will depend on it.

Three things, in order of how hard they are to fake:

**A fingerprint of the physics.** Not a hand-maintained version number --
those drift the moment someone forgets to bump one. A SHA-256 over the engine
source itself, so it is impossible to change a correlation without the
fingerprint changing. If two runs report the same fingerprint, they ran the
same equations.

**Determinism, demonstrated rather than asserted.** `tests/
test_reproducibility.py` runs a cross-section of the engine in separate
processes under different hash seeds and compares every float bit for bit.
A version stamp on a nondeterministic engine would be worse than none: it
would promise something the code does not deliver.

**An evidence bundle.** Inputs, fingerprint and outputs in one JSON document
an auditor can take away, archive, and replay against a future build.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# Only the physics. Tests, scripts and the web application are excluded
# deliberately: a change to a test or a page must not invalidate a number
# that the engine would still compute identically, or the fingerprint becomes
# noise and people stop reading it.
ENGINE_DIR = Path(__file__).resolve().parent

# Written by scripts/sync_engine.py at vendoring time and read at import, so
# a serverless cold start does not re-hash the source on every container.
# `verify_fingerprint` checks the two agree, which is what catches a stale
# vendored copy -- the failure this file exists to make impossible.
STAMP_FILE = ENGINE_DIR / "_fingerprint.txt"

FINGERPRINT_LENGTH = 12


# Modules that are not equations. The fingerprint answers one question --
# "did the physics change?" -- and a stamp that also moves when the
# bookkeeping around the physics changes answers it less well: every false
# alarm is a reason for someone to stop reading the stamp. `provenance`
# itself and `ledger` record and store answers; neither computes one.
#
# This list is a judgement with a cost, so it is stated rather than
# implied: a change to either of these modules is invisible to the
# fingerprint, and a bug in how a result is *recorded* would not be caught
# by it. `tests/test_reproducibility.py` covers them instead.
NOT_PHYSICS = ("__init__.py", "provenance.py", "ledger.py")


def _engine_sources():
    return sorted(
        path for path in ENGINE_DIR.glob("*.py")
        if path.name not in NOT_PHYSICS
    )


def compute_fingerprint() -> str:
    """SHA-256 over the engine's own source, truncated for readability.

    Both the name and the bytes of each module go in, so renaming a module
    changes the fingerprint even when its contents do not. Files are read as
    bytes and line endings normalised, because the same source checked out on
    Windows and Linux must fingerprint the same -- otherwise the stamp
    reports the developer's operating system rather than the physics.
    """
    digest = hashlib.sha256()
    for path in _engine_sources():
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:FINGERPRINT_LENGTH]


def _read_stamp():
    try:
        return STAMP_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def engine_fingerprint() -> str:
    """The fingerprint this build reports. Stamped if present, else computed."""
    return _read_stamp() or compute_fingerprint()


def write_stamp() -> str:
    fingerprint = compute_fingerprint()
    STAMP_FILE.write_text(fingerprint + "\n", encoding="utf-8")
    return fingerprint


def verify_fingerprint() -> tuple:
    """(ok, stamped, actual). False means a vendored copy has gone stale."""
    stamped = _read_stamp()
    actual = compute_fingerprint()
    return (stamped == actual, stamped, actual)


@dataclass
class Provenance:
    """Everything needed to produce one number again.

    Deliberately not a log line. A log says what happened; this is the
    *input* to a re-run, which is the thing an auditor needs and the thing
    nobody keeps unless the software keeps it for them.
    """

    calculation: str
    inputs: dict
    outputs: dict
    engine: str = field(default_factory=engine_fingerprint)
    correlations: list = field(default_factory=list)
    envelopes: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "calculation": self.calculation,
            "engine_fingerprint": self.engine,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "correlations": list(self.correlations),
            "envelopes": list(self.envelopes),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        """Stable JSON: sorted keys, no whitespace drift.

        So two bundles of the same calculation are byte-identical and can be
        diffed, and so a hash of the bundle is itself meaningful.
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=2,
                          default=_plain)

    @property
    def digest(self) -> str:
        """A short hash of the whole bundle. Two runs that agree share it."""
        return hashlib.sha256(
            self.to_json().encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def _plain(value):
    """Make numpy scalars and arrays serialise as plain JSON.

    Without this a bundle can fail to serialise at the moment it is most
    wanted -- when something has gone wrong and someone is trying to export
    the evidence.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
