"""An append-only record of every answer the engine gave.

A provenance bundle that exists only in one reply is evidence nobody can
retrieve. The claim worth being able to make is that any number this
software produced can be re-derived from the exact build that produced it,
by someone who does not trust the vendor, eighteen months later. That needs
the bundles kept somewhere, and kept in a way that makes quiet editing
detectable.

**The format is one JSON object per line and that is a deliberate choice.**
Not a database. The engine runs on numpy and the standard library so that
it deploys to an edge box inside a customer's network, and a ledger that
needed a database driver would not be usable in the deployments this is
aimed at. A text file is also the only format an auditor can read without
asking for the software, which is the point of giving it to them.

**Each record carries the digest of the one before it.** Change a record
and every digest after it stops matching, so `verify` finds the first point
where the file stops being self-consistent. This is called tamper-evidence
and the word *evident* is doing the work.

**What it does not do, stated here rather than discovered later.**

  It does not prevent tampering. Anyone who can write the file can rewrite
  it whole and recompute the chain from the edit onwards, and the result
  verifies cleanly. The chain raises the cost of a quiet edit from "change
  one number" to "rewrite every record after it", and it makes a careless
  edit certain to be caught. It is not a vault.

  It is not a signature. It says the file is internally consistent, not who
  wrote it. Proving authorship needs a key, which needs somewhere to keep a
  key, which is a different project with different failure modes.

  It does not timestamp in any provable sense. The time in a record is the
  time the machine thought it was.

Anchoring the head digest somewhere outside the file -- another system's
log, a daily email, a counterparty -- is what closes the rewrite gap, and
costs nothing but a line of text a day. `head()` returns exactly that line.

**Concurrency.** One writer. Appends are a single `write` of one line
opened in append mode, which most filesystems keep intact for short lines,
but two processes appending at once can still interleave on some of them,
and the chain would then record a `previous` that was not the previous. A
serverless deployment, where every request is its own process, is exactly
the case this does not cover -- and on those platforms the filesystem is
ephemeral anyway, so the ledger belongs on the customer's own server or
beside their historian rather than in a function. `verify` detects the
damage; it cannot undo it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# The digest of the record before the first one. A fixed, visible constant
# rather than an empty string, so a truncated file cannot pass as a fresh
# one: a ledger whose first record chains from anything else has had its
# beginning removed.
GENESIS = "0" * 64

SCHEMA = 1


@dataclass
class Record:
    """One answer, and its place in the chain."""

    index: int
    recorded_at: str
    payload: dict
    previous: str
    digest: str

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "index": self.index,
                "recorded_at": self.recorded_at, "payload": self.payload,
                "previous": self.previous, "digest": self.digest}

    @property
    def calculation(self) -> str:
        return str(self.payload.get("calculation", ""))

    @property
    def engine_fingerprint(self) -> str:
        return str(self.payload.get("engine_fingerprint", ""))


def _digest(index: int, recorded_at: str, payload: dict, previous: str) -> str:
    """Hash everything that matters, in a fixed order.

    `sort_keys` is what makes this reproducible: two runs that produced the
    same bundle must hash the same, or the digest is a hash of dictionary
    ordering rather than of the evidence. The separators matter for the
    same reason.
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)
    material = "%d\n%s\n%s\n%s" % (index, recorded_at, previous, body)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class Verdict:
    """What `verify` found, in enough detail to act on."""

    ok: bool
    records: int
    first_bad_index: int = -1
    reason: str = ""

    def describe(self) -> str:
        if self.ok:
            return ("%d records, chain intact" % self.records
                    if self.records else "empty ledger")
        return ("chain broken at record %d: %s"
                % (self.first_bad_index, self.reason))


class Ledger:
    """Append-only storage for provenance bundles."""

    def __init__(self, path):
        self.path = Path(path)

    # --------------------------------------------------------- writing --

    def append(self, payload, recorded_at: str = "") -> Record:
        """Add one bundle and return the record that was written.

        `payload` is a `Provenance`, its `to_dict()`, or any JSON-safe
        mapping. Accepting the object as well as the dict keeps the call
        site honest -- the alternative is every caller remembering to
        serialise, and the one that forgets stores a repr.
        """
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        if not isinstance(payload, dict):
            raise TypeError("a ledger record must be a mapping, got %s"
                            % type(payload).__name__)

        previous, index = self._head()
        stamp = recorded_at or time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())
        record = Record(index=index, recorded_at=stamp, payload=payload,
                        previous=previous,
                        digest=_digest(index, stamp, payload, previous))

        line = json.dumps(record.to_dict(), sort_keys=True,
                          separators=(",", ":"), default=str)
        if "\n" in line:
            raise ValueError("a record cannot contain a newline")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode, one write, flushed and synced. Opening with "a" on
        # every call rather than holding a handle is what lets an operator
        # rotate or copy the file from under a long-running process.
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    # --------------------------------------------------------- reading --

    def records(self):
        """Every record, in order. Malformed lines raise rather than skip.

        Skipping a line nobody can parse would mean a corrupt ledger reads
        as a shorter intact one, which is the failure this file exists to
        make impossible.
        """
        if not self.path.is_file():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except ValueError as exc:
                    raise ValueError("ledger line %d is not JSON: %s"
                                     % (number, exc))
                try:
                    yield Record(index=int(raw["index"]),
                                 recorded_at=str(raw["recorded_at"]),
                                 payload=raw["payload"],
                                 previous=str(raw["previous"]),
                                 digest=str(raw["digest"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("ledger line %d is malformed: %s"
                                     % (number, exc))

    def __len__(self) -> int:
        return sum(1 for _ in self.records())

    def head(self) -> str:
        """The digest to anchor somewhere this file cannot reach.

        Publishing this one line daily -- to another system, a colleague, a
        counterparty -- is what turns tamper-evidence into something a
        rewrite cannot defeat, because the rewritten chain would have to
        match a digest that is already elsewhere.
        """
        return self._head()[0]

    def _head(self):
        previous, index = GENESIS, 0
        for record in self.records():
            previous, index = record.digest, record.index + 1
        return previous, index

    # -------------------------------------------------------- checking --

    def verify(self) -> Verdict:
        """Recompute the whole chain and report the first break."""
        previous, expected = GENESIS, 0
        count = 0
        try:
            for record in self.records():
                if record.index != expected:
                    return Verdict(False, count, record.index,
                                   "expected index %d, found %d -- records "
                                   "have been removed or reordered"
                                   % (expected, record.index))
                if record.previous != previous:
                    return Verdict(False, count, record.index,
                                   "does not chain to the record before it")
                recomputed = _digest(record.index, record.recorded_at,
                                     record.payload, record.previous)
                if recomputed != record.digest:
                    return Verdict(False, count, record.index,
                                   "contents do not match the stored digest")
                previous, expected = record.digest, record.index + 1
                count += 1
        except ValueError as exc:
            return Verdict(False, count, count, str(exc))
        return Verdict(True, count)

    # --------------------------------------------------------- export --

    def export(self, since_index: int = 0) -> str:
        """The records as the file holds them, for someone to take away.

        Byte-identical lines rather than a re-serialisation, so what an
        auditor verifies is what was written.
        """
        lines = [json.dumps(r.to_dict(), sort_keys=True,
                            separators=(",", ":"), default=str)
                 for r in self.records() if r.index >= since_index]
        return "\n".join(lines) + ("\n" if lines else "")

    def summary(self) -> dict:
        """What is in here, for an operator rather than an auditor."""
        counts, engines, first, last = {}, set(), "", ""
        total = 0
        for record in self.records():
            total += 1
            counts[record.calculation] = counts.get(record.calculation, 0) + 1
            if record.engine_fingerprint:
                engines.add(record.engine_fingerprint)
            first = first or record.recorded_at
            last = record.recorded_at
        return {"records": total, "by_calculation": counts,
                "engines": sorted(engines), "first": first, "last": last,
                "head": self.head()}
