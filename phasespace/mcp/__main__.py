"""Run the PhaseSpace MCP server over stdio.

    python -m phasespace.mcp

Add it to a client's configuration like any other stdio MCP server, for
example in Claude Desktop or Claude Code:

    {
      "mcpServers": {
        "phasespace": {
          "command": "python",
          "args": ["-m", "phasespace.mcp"],
          "cwd": "/path/to/PhaseSpace"
        }
      }
    }

`--list` prints the catalogue and exits, which is the quickest way to check
an installation without a client attached.
"""
from __future__ import annotations

import json
import os
import sys

from phasespace.sim.ledger import Ledger
from phasespace.sim.provenance import engine_fingerprint, verify_fingerprint

from .protocol import serve
from .tools import build_registry

# Where to keep an append-only record of every call. Unset means keep none,
# which is right for a scratch session and wrong for a deployment anyone
# will be audited on. An environment variable rather than a flag because
# the client launches this process and usually will not let you add
# arguments.
LEDGER_ENV = "PHASESPACE_LEDGER"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    registry = build_registry()

    ledger_path = os.environ.get(LEDGER_ENV, "").strip()
    if ledger_path:
        registry.ledger = Ledger(ledger_path)

    if "--verify" in argv:
        if not ledger_path:
            print("set %s to the ledger you want checked" % LEDGER_ENV)
            return 2
        verdict = registry.ledger.verify()
        summary = registry.ledger.summary()
        print("ledger: %s" % ledger_path)
        print(verdict.describe())
        if verdict.ok and summary["records"]:
            print("head:   %s" % summary["head"])
            print("engines: %s" % ", ".join(summary["engines"]))
            for name, count in sorted(summary["by_calculation"].items()):
                print("  %-26s %d" % (name, count))
            print("")
            print("Publish the head digest somewhere this file cannot "
                  "reach -- another system, a colleague, a counterparty. "
                  "A hash chain detects a careless edit; an anchored head "
                  "is what a determined rewrite cannot match.")
        return 0 if verdict.ok else 1

    if "--list" in argv:
        agrees, stamped, computed = verify_fingerprint()
        print("PhaseSpace MCP server")
        print("engine fingerprint: %s" % engine_fingerprint())
        if not agrees:
            print("WARNING: stamp %s but source hashes to %s -- this build's "
                  "provenance is wrong" % (stamped, computed))
        print("%d tools:" % len(registry.names()))
        for schema in registry.schemas():
            required = schema["inputSchema"].get("required", [])
            print("\n  %s" % schema["name"])
            print("    %s" % schema["description"])
            print("    required: %s" % (", ".join(required) or "none"))
        return 0

    if "--schema" in argv:
        print(json.dumps(registry.schemas(), indent=2))
        return 0

    # stderr, so it cannot touch the protocol stream on stdout.
    # The ledger state belongs in the banner: a deployment that believes
    # it is recording and is not would otherwise look identical to one
    # that is, until someone asks for the evidence.
    sys.stderr.write(
        "phasespace mcp: engine %s, %d tools, ledger %s, ready\n"
        % (engine_fingerprint(), len(registry.names()),
           ledger_path or ("off -- set %s to record calls" % LEDGER_ENV)))
    sys.stderr.flush()
    serve(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
