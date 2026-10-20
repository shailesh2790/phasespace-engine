"""The tool catalogue, and the rule that an agent cannot compute.

The rule an engineering agent has to obey is one sentence: *"They never compute physics in
their head. Every number an agent shows comes from a tool call to the engine,
the optimiser or a surrogate, and the call is logged."*

A prompt cannot enforce that. A language model asked not to do arithmetic
will still do arithmetic, and the failure is silent -- a plausible number
with nothing behind it, indistinguishable in the transcript from a computed
one. So the enforcement lives here instead, in three mechanical properties:

**Every result carries a provenance block.** Not as decoration. The block
holds the inputs, the outputs, the correlations used, and the SHA-256
fingerprint of the engine source that produced them, so the figure can be
re-derived from the exact build eighteen months later. A number an agent
quotes without a provenance block did not come from this server, and that is
checkable after the fact rather than a matter of trust.

**Every result states its own limits.** `notes` and `envelopes` travel with
the answer. Where a correlation is being used outside the range it was
fitted on, the tool says so in the same payload as the number, so the
caveat cannot be dropped on the way to the user the way a footnote can.

**A refusal is a result, not an exception.** When an input is missing or a
test is unusable, the tool returns `isError` with a sentence an agent can
read and act on. Raising instead would give the agent a transport failure,
which it cannot reason about -- and the realistic agent behaviour then is to
retry, or to fill the gap itself.

The text block is for the model to read; the JSON block beside it is for a
program to parse and for the audit log to keep.
"""
from __future__ import annotations

import json
import traceback

from phasespace.sim.provenance import Provenance, engine_fingerprint


def text_content(text: str, is_error: bool = False) -> dict:
    """A tool result carrying one text block.

    Defined here rather than in `protocol`, because the shape of a result
    belongs to the tool surface and not to the wire: the HTTP transport
    returns these too, and it has no stdio protocol to import. The
    dependency runs transport -> results, never the other way.
    """
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class Tool:
    """One callable, its schema, and the physics it is allowed to speak for."""

    def __init__(self, name, description, schema, handler, correlations=None):
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = handler
        self.correlations = list(correlations or [])

    def as_schema(self) -> dict:
        return {"name": self.name,
                "description": self.description,
                "inputSchema": self.schema}


class Refusal(Exception):
    """The tool ran and declined, with a reason worth reading.

    Distinct from a bug. A missing water analysis, a test with no drawdown,
    a history that does not decline -- these are the engine saying the
    question cannot be answered from what it was given, which is the most
    useful thing it can say and must not be dressed up as a number.
    """


class Registry:
    """Holds the tools and turns each call into a provenance-stamped answer."""

    def __init__(self, ledger=None):
        self._tools = {}
        # Optional, and off unless a deployment supplies one. A bundle
        # that exists only in one reply is evidence nobody can retrieve,
        # so a self-hosted deployment passes a `Ledger` here and every
        # answer -- refusals included -- lands in an append-only file an
        # auditor can take away. Not the default, because a serverless
        # filesystem is ephemeral: recording evidence that silently
        # disappears is worse than recording none.
        self.ledger = ledger

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list:
        return sorted(self._tools)

    def schemas(self) -> list:
        return [self._tools[n].as_schema() for n in self.names()]

    # --------------------------------------------------------------- calling --

    def call(self, name: str, arguments: dict) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return text_content(
                "no such tool: %s. Available: %s"
                % (name, ", ".join(self.names())), is_error=True)

        try:
            outcome = tool.handler(arguments)
        except Refusal as exc:
            # A refusal still earns a provenance block: what was asked, and
            # why it could not be answered, is itself auditable.
            return self._render(tool, arguments,
                                {"refused": str(exc)},
                                ["the engine declined to answer: %s" % exc],
                                is_error=True)
        except (ValueError, KeyError, TypeError) as exc:
            # Bad arguments. The agent can fix these, so the message says
            # what was wrong rather than only that something was.
            return text_content(
                "%s could not run: %s: %s\n\nExpected arguments:\n%s"
                % (name, type(exc).__name__, exc,
                   json.dumps(tool.schema.get("properties", {}), indent=2)),
                is_error=True)
        except Exception as exc:                              # noqa: BLE001
            return text_content(
                "%s failed unexpectedly: %s: %s\n%s"
                % (name, type(exc).__name__, exc,
                   traceback.format_exc(limit=4)), is_error=True)

        summary = outcome.get("summary", "")
        outputs = outcome.get("outputs", {})
        notes = list(outcome.get("notes", []))
        envelopes = list(outcome.get("envelopes", []))
        correlations = outcome.get("correlations", tool.correlations)
        return self._render(tool, arguments, outputs, notes,
                            summary=summary, envelopes=envelopes,
                            correlations=correlations)

    def _render(self, tool, arguments, outputs, notes, summary="",
                envelopes=None, correlations=None, is_error=False) -> dict:
        """Assemble the answer: prose for the model, JSON for the record."""
        bundle = Provenance(
            calculation=tool.name,
            inputs=dict(arguments),
            outputs=outputs,
            correlations=list(correlations if correlations is not None
                              else tool.correlations),
            envelopes=list(envelopes or []),
            notes=list(notes),
        )

        parts = []
        if summary:
            parts.append(summary)
        if notes:
            parts.append("\n".join("- %s" % n for n in notes))
        parts.append(
            "provenance: engine %s, bundle %s"
            % (bundle.engine, bundle.digest))
        parts.append("```json\n%s\n```" % bundle.to_json())

        # Recorded here rather than at each call site because every path
        # through this class converges on `_render`, answers and refusals
        # alike. A refusal is evidence too: what was asked, and why it
        # could not be answered, is what an auditor reconstructing a
        # decision needs most.
        recorded = ""
        if self.ledger is not None:
            try:
                recorded = self.ledger.append(bundle).digest
            except (OSError, ValueError, TypeError) as exc:
                # A ledger that cannot write must not swallow the answer:
                # the caller asked a question and the engine answered it.
                # The failure goes into the payload so it is visible.
                parts.insert(-1, "- this answer was NOT recorded in the "
                                 "audit ledger: %s" % exc)

        return {"content": [{"type": "text", "text": "\n\n".join(parts)}],
                "isError": is_error,
                # Structured mirror of the same answer, for the audit log and
                # for any caller that would rather parse than read.
                "_meta": {"provenance": bundle.to_dict(),
                          "engine_fingerprint": engine_fingerprint(),
                          "ledger_digest": recorded}}
