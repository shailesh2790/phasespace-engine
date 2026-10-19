"""The MCP wire protocol, in the standard library and nothing else.

There is an official SDK. This does not use it, and the reason is a product
requirement rather than a preference: this has to run on an edge box, an
on-premise server and a private cloud, fully offline. A server an operator
can run from a stock Python with no `pip install` is one fewer thing to get
approved by someone's IT security review, and on an air-gapped site it is
the difference between working and not.

So this is JSON-RPC 2.0 over newline-delimited stdio, which is all the stdio
transport is. The pieces:

  initialize                  handshake; we answer with our protocol version
                              and the one capability we have, tools
  notifications/initialized   client says it is ready; no reply permitted
  tools/list                  the catalogue, with a JSON Schema per tool
  tools/call                  run one, return content blocks
  ping                        liveness

**Framing.** One JSON object per line. The stdio transport forbids embedded
newlines in a message, which is why every response is written with
`separators` that produce no newlines and flushed immediately. A buffered
write here deadlocks the client, because it is waiting on a line that is
sitting in our buffer.

**stdout belongs to the protocol.** Anything that prints to stdout corrupts
the stream, and the failure looks like a client that mysteriously cannot
parse our reply. `serve()` therefore redirects `sys.stdout` for the duration
and hands tools a devnull, so an engine module that grows a stray `print`
cannot take the server down. Diagnostics go to stderr.

**Errors.** Two kinds, and the distinction matters to an agent. A malformed
request is a JSON-RPC error: the call never happened. A tool that ran and
failed returns a normal result with `isError` true and the reason in the
content, because the agent needs to read it, reason about it, and try
something else -- which it cannot do with a transport-level error.
"""
from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 2.0 reserved codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _write(stream, payload: dict) -> None:
    """One message, one line, flushed.

    `separators` without spaces is not for compactness: it guarantees no
    newline can appear inside the encoded object, which is what the
    line-delimited framing requires.
    """
    stream.write(json.dumps(payload, separators=(",", ":"),
                            default=_plain) + "\n")
    stream.flush()


def _plain(value):
    """Make numpy scalars and arrays serialise.

    The engine returns numpy types in places. Without this a tool call fails
    at the moment of replying, which is the least debuggable moment
    available.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def response(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def text_content(text: str, is_error: bool = False) -> dict:
    """A tools/call result carrying one text block."""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class Server:
    """Dispatch for one stdio session.

    Holds no state beyond the registry and whether the handshake happened,
    so a session is cheap and a crashed client costs nothing.
    """

    def __init__(self, registry, name: str = "phasespace",
                 version: str = "1.0.0"):
        self.registry = registry
        self.name = name
        self.version = version
        self.initialised = False

    # ------------------------------------------------------------ handlers --

    def _initialize(self, params: dict) -> dict:
        """Answer the handshake.

        We echo the client's protocol version when we recognise it and offer
        ours otherwise, which is what the specification asks for: version
        negotiation, not an assertion.
        """
        asked = (params or {}).get("protocolVersion")
        self.initialised = True
        return {
            "protocolVersion": asked if asked == PROTOCOL_VERSION
                               else PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": self.version},
            # Shown to the operator by clients that display it. It says the
            # one thing someone needs to know before trusting the output.
            "instructions": (
                "Every number this server returns was computed by the "
                "PhaseSpace physics engine, not inferred. Each result "
                "carries a provenance block with the engine's source "
                "fingerprint so the figure can be re-derived from the exact "
                "build that produced it. Where a correlation is being used "
                "outside its validity range, or an input is missing, the "
                "tool says so instead of returning a number."),
        }

    def _tools_list(self) -> dict:
        return {"tools": self.registry.schemas()}

    def _tools_call(self, params: dict) -> dict:
        params = params or {}
        name = params.get("name")
        if not isinstance(name, str):
            return text_content("tools/call needs a tool name", is_error=True)
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return text_content("arguments must be an object", is_error=True)
        return self.registry.call(name, arguments)

    # ------------------------------------------------------------ dispatch --

    def handle(self, message: dict):
        """One request in, one response out, or None for a notification."""
        if message.get("jsonrpc") != "2.0":
            return error(message.get("id"), INVALID_REQUEST,
                         "jsonrpc must be '2.0'")

        method = message.get("method")
        request_id = message.get("id")
        is_notification = "id" not in message

        if not isinstance(method, str):
            return None if is_notification else error(
                request_id, INVALID_REQUEST, "method must be a string")

        # Notifications get no reply, ever -- a reply to one is a protocol
        # violation that some clients treat as fatal.
        if is_notification:
            return None

        params = message.get("params") or {}

        if method == "initialize":
            return response(request_id, self._initialize(params))
        if method == "ping":
            return response(request_id, {})

        # Everything else requires the handshake. Answering tools/list before
        # initialize would let a client skip version negotiation and then
        # fail later in a way that looks like our bug.
        if not self.initialised:
            return error(request_id, INVALID_REQUEST,
                         "initialize must be called first")

        if method == "tools/list":
            return response(request_id, self._tools_list())
        if method == "tools/call":
            return response(request_id, self._tools_call(params))

        return error(request_id, METHOD_NOT_FOUND, "no such method: %s" % method)


def serve(registry, stdin=None, stdout=None, name: str = "phasespace",
          version: str = "1.0.0") -> None:
    """Run a session until stdin closes.

    `sys.stdout` is swapped for the duration so that nothing the engine
    prints can reach the protocol stream. The real stdout is captured first
    and used only for protocol writes.
    """
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout

    server = Server(registry, name=name, version=version)

    guarded = _DevNull()
    real_stdout = sys.stdout
    sys.stdout = guarded
    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError as exc:
                _write(sink, error(None, PARSE_ERROR, "invalid JSON: %s" % exc))
                continue
            if not isinstance(message, dict):
                _write(sink, error(None, INVALID_REQUEST,
                                   "a message must be a JSON object"))
                continue
            try:
                reply = server.handle(message)
            except Exception as exc:                      # noqa: BLE001
                # A crash in dispatch must not end the session: the client
                # has no way to distinguish our death from a hang.
                reply = error(message.get("id"), INTERNAL_ERROR,
                              "%s: %s" % (type(exc).__name__, exc))
            if reply is not None:
                _write(sink, reply)
    finally:
        sys.stdout = real_stdout


class _DevNull:
    """Absorbs anything a tool prints, so it cannot corrupt the stream."""

    def write(self, _text):
        return len(_text) if _text else 0

    def flush(self):
        pass

    def isatty(self):
        return False
