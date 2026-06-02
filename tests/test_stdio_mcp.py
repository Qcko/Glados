"""Tests for the stdio MCP transport.

Two layers:
- StdioServer / StdioToolProxy unit tests using a tiny in-test stdio
  helper subprocess (Python `-c` one-liner) so no test depends on the
  full toy_stdio_server.py.
- Integration test that spawns the real `scripts/toy_stdio_server.py`
  via `build_app()`'s lifespan and calls `toy_stdio.add` end-to-end
  through the MCPRegistry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glados.mcp.registry import CallEnvelope
from glados.mcp.stdio_client import StdioServer, StdioServerError, StdioToolProxy


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_inline_server(body: str, server_id: str = "inline") -> StdioServer:
    """Spawn a one-shot Python subprocess that runs `body` as its main
    loop. `body` should read stdin lines and write JSON-RPC responses.
    `server_id` becomes the `server` prefix on every tool's qualified
    name (real MCP doesn't carry server identity on the wire)."""
    return StdioServer(sys.executable, ["-c", body], server_id=server_id)


# Minimal real-MCP echo server inline. Implements `initialize`,
# `tools/list`, `tools/call` for one `echo` tool. Drops notifications
# silently (rid=None). Content rides as a JSON-encoded text block per
# real-MCP shape; StdioServer parses it back into a dict.
_INLINE_SERVER = """
import json, sys
TOOLS = [{"name": "echo", "description": "echo",
          "inputSchema": {"type": "object",
                          "properties": {"text": {"type": "string"}}}}]
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    rid = req.get("id")
    m = req.get("method")
    if rid is None:
        continue  # notification, no response
    if m == "initialize":
        out = {"jsonrpc":"2.0","id":rid,"result":{
            "protocolVersion":"2024-11-05",
            "capabilities":{"tools":{}},
            "serverInfo":{"name":"inline","version":"0.1"}}}
    elif m == "tools/list":
        out = {"jsonrpc":"2.0","id":rid,"result":{"tools":TOOLS}}
    elif m == "tools/call":
        args = req["params"]["arguments"]
        body = json.dumps({"text": args["text"]})
        out = {"jsonrpc":"2.0","id":rid,"result":{
            "content":[{"type":"text","text":body}],
            "isError":False}}
    else:
        out = {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"unknown"}}
    sys.stdout.write(json.dumps(out)+"\\n"); sys.stdout.flush()
"""


async def test_stdio_server_initialize_and_list_tools() -> None:
    server = _make_inline_server(_INLINE_SERVER)
    await server.start()
    try:
        await server.initialize()
        specs = await server.list_tools()
        assert len(specs) == 1
        assert specs[0].qualified == "inline.echo"
    finally:
        await server.aclose()


async def test_stdio_server_call_tool_round_trip() -> None:
    server = _make_inline_server(_INLINE_SERVER)
    await server.start()
    try:
        result = await server.call_tool("echo", {"text": "hi"})
        assert result.ok
        assert result.content == {"text": "hi"}
    finally:
        await server.aclose()


async def test_stdio_server_proxy_routes_envelope() -> None:
    server = _make_inline_server(_INLINE_SERVER)
    await server.start()
    try:
        specs = await server.list_tools()
        proxy = StdioToolProxy(server, specs[0])
        env = CallEnvelope(
            session_id="s1", room_id="desk", speaker_id="u1"
        )
        result = await proxy.call({"text": "round"}, env)
        assert result.ok
        assert result.content == {"text": "round"}
    finally:
        await server.aclose()


# Server that serves a `memory://lessons` resource over `resources/read`.
# Any other resource URI, or any other method, errors.
_LESSONS_SERVER = """
import json, sys
LESSONS = "# Lessons\\nSearch the broad noun, read volume from each result."
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    rid = req.get("id")
    m = req.get("method")
    if rid is None:
        continue
    if m == "initialize":
        out = {"jsonrpc":"2.0","id":rid,"result":{
            "protocolVersion":"2024-11-05","capabilities":{},
            "serverInfo":{"name":"lessons","version":"0.1"}}}
    elif m == "resources/read" and req["params"]["uri"] == "memory://lessons":
        out = {"jsonrpc":"2.0","id":rid,"result":{"contents":[
            {"uri":"memory://lessons","mimeType":"text/markdown","text":LESSONS}]}}
    else:
        out = {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"not found"}}
    sys.stdout.write(json.dumps(out)+"\\n"); sys.stdout.flush()
"""


async def test_read_lessons_returns_markdown() -> None:
    server = _make_inline_server(_LESSONS_SERVER, server_id="lessons")
    await server.start()
    try:
        await server.initialize()
        text = await server.read_lessons()
        assert text is not None
        assert "Search the broad noun" in text
    finally:
        await server.aclose()


async def test_read_lessons_none_when_unsupported() -> None:
    # The echo server has no resources support — replies with a JSON-RPC
    # error, which read_lessons swallows into None (the common case).
    server = _make_inline_server(_INLINE_SERVER)
    await server.start()
    try:
        await server.initialize()
        assert await server.read_lessons() is None
    finally:
        await server.aclose()


async def test_stdio_server_dies_fails_pending_calls() -> None:
    # Subprocess that handles `initialize` then exits, simulating a crash
    # in the middle of a session.
    body = """
import json, sys
line = sys.stdin.readline()
req = json.loads(line)
sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":{
    "protocolVersion":"2024-11-05",
    "capabilities":{"tools":{}},
    "serverInfo":{"name":"dying","version":"0.1"}}})+"\\n")
sys.stdout.flush()
sys.exit(0)
"""
    server = _make_inline_server(body)
    await server.start()
    try:
        await server.initialize()  # this one succeeds
        # Wait briefly for the reader to observe EOF after subprocess exit,
        # then confirm the next call returns a usable error rather than
        # hanging. `call_tool` deliberately converts StdioServerError into
        # MCPCallResult(ok=False, ...) so the LLM sees a clean tool failure.
        import asyncio
        for _ in range(20):
            if server._dead:  # noqa: SLF001 - white-box check, no public surface yet
                break
            await asyncio.sleep(0.05)
        result = await server.call_tool("anything", {})
        assert not result.ok
        assert result.error is not None
    finally:
        await server.aclose()


# ---- Auto-restart / circuit-breaker ------------------------------------


# Subprocess that handles initialize + tools/call normally, but exits on
# a "die" call. Used to verify the next call_tool transparently respawns.
# Real-MCP shape: content rides as JSON-encoded text blocks; notifications
# (rid=None) get no response.
_RESTARTABLE_SERVER = """
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    rid = req.get("id")
    m = req.get("method")
    if rid is None:
        continue
    if m == "initialize":
        sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":rid,"result":{
            "protocolVersion":"2024-11-05",
            "capabilities":{"tools":{}},
            "serverInfo":{"name":"restartable","version":"0.1"}}})+"\\n")
        sys.stdout.flush()
    elif m == "tools/call":
        name = req["params"]["name"]
        if name == "die":
            sys.exit(1)
        body = json.dumps({"name":name})
        sys.stdout.write(json.dumps({
            "jsonrpc":"2.0","id":rid,
            "result":{"content":[{"type":"text","text":body}],"isError":False}})+"\\n")
        sys.stdout.flush()
"""


async def test_stdio_server_auto_restarts_after_crash() -> None:
    """A crash + a follow-up call should transparently respawn the
    subprocess and the second call should succeed against the fresh
    instance."""
    import asyncio

    server = StdioServer(
        sys.executable,
        ["-c", _RESTARTABLE_SERVER],
        restart_backoff_s=0.01,
    )
    await server.start()
    try:
        await server.initialize()
        # First call kills the subprocess.
        result = await server.call_tool("die", {})
        assert not result.ok
        # Wait for the reader to observe EOF.
        for _ in range(40):
            if server._dead:  # noqa: SLF001
                break
            await asyncio.sleep(0.02)
        assert server._dead  # noqa: SLF001
        # Second call should trigger restart and answer cleanly.
        result = await server.call_tool("hello", {})
        assert result.ok, f"expected ok after restart, got error={result.error!r}"
        assert result.content == {"name": "hello"}
        assert len(server._restart_attempts) == 1  # noqa: SLF001
    finally:
        await server.aclose()


async def test_stdio_server_circuit_breaker_after_max_restarts() -> None:
    """A subprocess that exits immediately after initialize should
    exhaust the restart budget and then return a clean circuit-open
    error rather than retrying forever."""
    body = """
import json, sys
line = sys.stdin.readline()
req = json.loads(line)
sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":{
    "protocolVersion":"2024-11-05",
    "capabilities":{"tools":{}},
    "serverInfo":{"name":"dying","version":"0.1"}}})+"\\n")
sys.stdout.flush()
sys.exit(0)
"""
    server = StdioServer(
        sys.executable,
        ["-c", body],
        max_restarts=2,
        restart_backoff_s=0.01,
    )
    import asyncio

    await server.start()
    try:
        await server.initialize()
        for _ in range(40):
            if server._dead:  # noqa: SLF001
                break
            await asyncio.sleep(0.02)
        # Each call_tool triggers a restart attempt; subprocess dies again
        # before tools/call can be sent, so call_tool returns ok=False.
        # After max_restarts attempts the circuit opens.
        for _ in range(server._max_restarts):  # noqa: SLF001
            result = await server.call_tool("x", {})
            assert not result.ok
        # Budget spent: next call should report circuit-open without
        # spawning another subprocess.
        attempts_before = len(server._restart_attempts)  # noqa: SLF001
        result = await server.call_tool("x", {})
        assert not result.ok
        assert "circuit open" in result.error
        assert len(server._restart_attempts) == attempts_before  # noqa: SLF001
    finally:
        await server.aclose()


# ---- End-to-end via build_app() spawning the real toy_stdio_server -----


@pytest.fixture
def stdio_app():
    """Build a fresh app with the real configs/servers.toml in play."""
    import os

    os.environ["GLADOS_CONFIG_DIR"] = str(_REPO_ROOT / "configs")
    from glados.core.server import build_app

    app = build_app()
    with TestClient(app) as client:
        yield client


def test_toy_stdio_tools_registered_via_lifespan(stdio_app: TestClient) -> None:
    """After lifespan startup the registry should know about toy_stdio.add."""
    registry = stdio_app.app.state.mcp
    spec = registry.spec_for("toy_stdio", "add")
    assert spec is not None
    assert spec.description.startswith("Return the sum")


def test_toy_stdio_overlay_applies_requires_confirmation(stdio_app: TestClient) -> None:
    """servers.toml `[server.tool_overlays.roll_dice]` should flip the
    spec's `requires_confirmation` to True after `tools/list` — the wire
    schema doesn't carry that flag, so the overlay is the only place it
    can come from for a third-party server."""
    registry = stdio_app.app.state.mcp
    roll = registry.spec_for("toy_stdio", "roll_dice")
    assert roll is not None
    assert roll.requires_confirmation is True
    # Sibling without an overlay entry stays at the wire default.
    add = registry.spec_for("toy_stdio", "add")
    assert add is not None
    assert add.requires_confirmation is False


async def test_toy_stdio_dispatch_round_trip(stdio_app: TestClient) -> None:
    """Dispatch a real call through the registry into the subprocess."""
    registry = stdio_app.app.state.mcp
    env = CallEnvelope(session_id="s1", room_id="desk", speaker_id="u1")
    result = await registry.dispatch("toy_stdio", "add", {"a": 2, "b": 3}, env)
    assert result.ok
    assert result.content == {"sum": 5}
