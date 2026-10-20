# The PhaseSpace MCP server

The engine as tools an agent may call, over the Model Context Protocol.

```bash
python -m phasespace.mcp --list      # the catalogue
python -m phasespace.mcp             # serve on stdio
```

Client configuration, for Claude Desktop, Claude Code, or anything else
that speaks MCP over stdio:

```json
{
  "mcpServers": {
    "phasespace": {
      "command": "python",
      "args": ["-m", "phasespace.mcp"],
      "cwd": "/path/to/PhaseSpace"
    }
  }
}
```

## Why this exists

The rule an engineering agent has to obey is one sentence:

> *"They never compute physics in their head. Every number an agent shows
> comes from a tool call to the engine, the optimiser or a surrogate, and
> the call is logged."*

A prompt cannot enforce that. A language model told not to do arithmetic
will do arithmetic anyway, and the failure is silent — a plausible number
with nothing behind it, indistinguishable in a transcript from a computed
one. That is the single thing that makes an engineering agent unusable, and
it is not solved by better instructions.

So it is enforced mechanically here instead.

**Every answer carries a provenance block.** Inputs, outputs, correlations
used, and the SHA-256 fingerprint of the engine source that produced them.
A number quoted without one did not come from this server, and that is
checkable after the fact rather than a matter of trust:

```
provenance: engine 61c9045ea5ee, bundle 26ecc471e9d5
```

The bundle is stable, sorted JSON, so the same question twice gives a
byte-identical bundle and the digest means something. Eighteen months from
now an auditor can take that fingerprint, check out the matching engine,
and re-derive the figure.

**Every answer states its own limits, in the same payload as the number.**
Not as a footnote that gets dropped on the way to a user. Ask for a nodal
operating point and the answer says, unprompted:

> the traverse underlying this is validated to 2.9% median absolute error
> on 150 measured gradient surveys; the intersection itself — predicting
> the rate — is NOT validated

**A refusal is a result, not an exception.** An agent can read a sentence
and act on it. It cannot reason about a transport error, and what it does
instead is retry or fill the gap itself. So when the engine cannot answer,
it returns `isError` with the reason:

> scale saturation is not implemented, and deliberately so. Calcite and
> barite indices need a water analysis — calcium, mg/l; bicarbonate, mg/l;
> barium, mg/l; sulphate, mg/l; … — and no archive this engine reads
> carries one. A saturation index computed from assumed chemistry is a
> number about the assumption. Ask the operator for these values.

## The tools

| tool | answers | refuses when |
|---|---|---|
| `well_pressure_traverse` | bottomhole pressure, and pressure, holdup and regime at every depth | the well is producing nothing |
| `well_nodal_analysis` | the operating point where inflow meets outflow | the curves do not intersect — the well is dead at that wellhead pressure |
| `gas_lift_allocate` | how to share limited lift gas, the gain, and the shadow price | a well has fewer than 4 lift tests |
| `decline_curve_fit` | Arps fit, EUR and the spread the admissible b range implies | the history is flat, rising or too short |
| `data_quality_check` | whether a set of tags may be fitted or optimised on | — (it reports; `may_fit` is the gate) |
| `well_integrity_assess` | *where* CO₂ corrosion attacks fastest, not just how fast | the gas is sour, or no CO₂ fraction was given |
| `scale_saturation` | nothing — it names the water analysis it would need | always, by design |

That last row is deliberate. An agent asked about scale with no tool
available will reason its way to an answer from nothing. A tool that
refuses in the agent's own vocabulary is a stronger guard than the absence
of a tool.

## Units

Field units at the boundary, and every schema states the unit in its
property description. This is the most likely source of a badly wrong
answer on the whole surface: a ksc figure read as psia is out by a factor
of fourteen and the result looks entirely plausible.

- pressures **psia**, absolute, not gauge
- depths **feet**, temperatures **°F**
- gas **scf/day**, liquid **stb/day**
- `inclination_deg` is from *horizontal* — 90 is a vertical well

## No dependencies

The protocol is implemented in `protocol.py` against the standard library.
There is an official SDK and this does not use it, for a product reason
rather than a preference: this has to run on an edge box, an on-premise
server and a private cloud, fully offline. A server that runs from a stock
Python with no `pip install` is one fewer thing to get through someone's IT
security review, and on an air-gapped site it is the difference between
working and not.

`stdout` belongs to the protocol. `serve()` redirects `sys.stdout` for the
duration of a session, so an engine module that grows a stray `print`
cannot corrupt the stream — a failure that otherwise looks like a client
mysteriously unable to parse a reply.

## Recording what was answered

A provenance bundle that exists only in one reply is evidence nobody can
retrieve. Point the server at a file and every call lands in an
append-only ledger:

```bash
PHASESPACE_LEDGER=evidence.jsonl python -m phasespace.mcp
PHASESPACE_LEDGER=evidence.jsonl python -m phasespace.mcp --verify
```

```
2 records, chain intact
head:   70f468b66b04477ac7f92fb788b9aa7daaa5352bf3bff62abddf165d39adc9b0
engines: 61c9045ea5ee
  scale_saturation           1
  well_pressure_traverse     1
```

One JSON object per line, each carrying the digest of the record before
it. Not a database: the engine is meant to run on an edge box inside a
customer's network, and a text file is the only format an auditor can read
without asking for the software. **Refusals are recorded too** — what was
asked and why it could not be answered is what someone reconstructing a
decision needs most.

It is off unless you set the variable, because a serverless filesystem is
ephemeral and recording evidence that silently disappears is worse than
recording none. The startup banner says which state you are in, so a
deployment that believes it is recording and is not does not look
identical to one that is.

**What the chain does and does not prove.** It detects an edited value, a
deleted record, a reordering, and a removed beginning — `verify` names the
first record where the file stops being self-consistent. It does *not*
prevent tampering: anyone who can write the file can rewrite it whole and
recompute the chain, and the result verifies cleanly. There is a test that
demonstrates exactly that, so the guarantee does not get quoted without
its limit. The chain raises the cost of a quiet edit from changing one
number to rewriting every record after it, and makes a careless edit
certain to be caught.

What closes the gap costs nothing: publish the head digest somewhere the
writer cannot reach — another system, a colleague, a counterparty. A
rewritten chain cannot match a digest that is already elsewhere.

It is also not a signature, and the timestamps are whatever the machine
believed.


## What is not here yet

- **An approval gate.** Nothing in this surface writes anything, so
  there is no approval gate to build yet. When a tool can change a
  setpoint, it needs one first.
- **Somewhere durable to keep the ledger on a serverless host.** The
  ledger above covers a self-hosted deployment, which is the one this is
  aimed at. On a platform with an ephemeral filesystem it needs object
  storage or the customer's own historian behind it, and that is not
  written. Note also that `phasespace/sim/audit.py` is unrelated despite
  the name -- it is a production-history audit, and the collision is how
  the absence of a real ledger stayed invisible for a while.
- **The agents themselves.** This is the tool surface they stand on. The
  agents that would use it -- a data steward, a calibrator, an
  optimisation analyst, a study runner, a reporter -- are orchestration
  over these calls.
- **Surrogates.** Deliberately absent for now. A field solve is 0.83 s
  measured, so nothing here needs speeding up. Surrogates belong in the
  optimiser's inner loop, where thousands of evaluations turn 0.83 s into
  hours, not in single answers where the saving is invisible.
