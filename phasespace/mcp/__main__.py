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
import sys

from phasespace.sim.provenance import engine_fingerprint, verify_fingerprint

from .protocol import serve
from .tools import build_registry


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    registry = build_registry()

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
    sys.stderr.write("phasespace mcp: engine %s, %d tools, ready\n"
                     % (engine_fingerprint(), len(registry.names())))
    sys.stderr.flush()
    serve(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
