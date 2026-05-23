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


def _make_inline_server(body: str) -> StdioServer:
    """Spawn a one-shot Python subprocess that runs `body` as its main
    loop. `body` should read stdin lines and write JSON-RPC responses."""
    return StdioServer(sys.executable, ["-c", body])


# Minimal stdio echo server inline. Implements only `tools/list` (returns
# one tool spec) and `tools/call` for `echo`.
_INLINE_SERVER = """
import json, sys
TOOLS = [{"server": "inline", "name": "echo", "description": "echo",
          "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}]
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    rid = req.get("id")
    m = req.get("method")
    if m == "initialize":
        out = {"jsonrpc":"2.0","id":rid,"result":{}}
    elif m == "tools/list":
        out = {"jsonrpc":"2.0","id":rid,"result":{"tools":TOOLS}}
    elif m == "tools/call":
        args = req["params"]["arguments"]
        out = {"jsonrpc":"2.0","id":rid,"result":{"ok":True,"content":{"text":args["text"]}}}
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


async def test_stdio_server_dies_fails_pending_calls() -> None:
    # Subprocess that handles `initialize` then exits, simulating a crash
    # in the middle of a session.
    body = """
import json, sys
line = sys.stdin.readline()
req = json.loads(line)
sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":{}})+"\\n")
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


async def test_toy_stdio_dispatch_round_trip(stdio_app: TestClient) -> None:
    """Dispatch a real call through the registry into the subprocess."""
    registry = stdio_app.app.state.mcp
    env = CallEnvelope(session_id="s1", room_id="desk", speaker_id="u1")
    result = await registry.dispatch("toy_stdio", "add", {"a": 2, "b": 3}, env)
    assert result.ok
    assert result.content == {"sum": 5}
