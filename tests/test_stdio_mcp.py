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

import asyncio
import logging
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
    # The echo server has no resources support -- replies with a JSON-RPC
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


# ---- Lazy spawn + idle reap (ARCH section 13) ---------------------------------


async def test_stdio_server_sleeps_then_wakes_on_call() -> None:
    """A dormant server has no child but isn't closed; the next call_tool
    transparently wakes it (start + initialize) and serves the call."""
    server = _make_inline_server(_INLINE_SERVER)
    await server.start()
    try:
        await server.initialize()
        await server.list_tools()
        await server.sleep()
        assert not server.is_resident()  # child gone, but reanimatable
        result = await server.call_tool("echo", {"text": "wake"})
        assert result.ok and result.content == {"text": "wake"}
        assert server.is_resident()  # woke for the call
    finally:
        await server.aclose()


async def test_wake_does_not_consume_restart_budget() -> None:
    """Sleep/wake is an intended transition, so cycling it more times than
    `max_restarts` must keep working -- it never touches the crash circuit."""
    server = StdioServer(
        sys.executable, ["-c", _INLINE_SERVER], max_restarts=1, server_id="inline"
    )
    await server.start()
    try:
        await server.initialize()
        for i in range(4):  # well past max_restarts=1
            await server.sleep()
            result = await server.call_tool("echo", {"text": f"n{i}"})
            assert result.ok and result.content == {"text": f"n{i}"}
        assert server._restart_attempts == []  # noqa: SLF001 - budget untouched
    finally:
        await server.aclose()


async def test_reap_idle_servers_sleeps_only_idle_resident() -> None:
    """The reaper helper sleeps a resident server past its idle window, but
    leaves a recently-active one alone and skips an already-dormant one."""
    from glados.core.server import _reap_idle_servers

    server = _make_inline_server(_INLINE_SERVER)
    await server.start()
    try:
        await server.initialize()
        # Fresh call stamps activity 'now'; with a tiny window it reads idle.
        await server.call_tool("echo", {"text": "x"})
        now = server._last_activity  # noqa: SLF001
        # Active within window -> not reaped.
        assert _reap_idle_servers([(server, 300.0)], now) == []
        # Past the window -> one sleep() coroutine returned; await it.
        coros = _reap_idle_servers([(server, 300.0)], now + 301.0)
        assert len(coros) == 1
        for c in coros:
            await c
        assert not server.is_resident()
        # Already dormant -> skipped (not resident).
        assert _reap_idle_servers([(server, 300.0)], now + 999.0) == []
    finally:
        await server.aclose()


async def test_sleep_refuses_while_call_in_flight() -> None:
    """The reaper must not sleep a server a dispatch just woke. `sleep()`
    refuses while `_active_calls` is non-zero, even before the RPC reaches
    `_pending` -- closing the wake-then-dispatch race."""
    server = _make_inline_server(_INLINE_SERVER)
    await server.start()
    try:
        await server.initialize()
        # Simulate a call in its wake-then-dispatch span (woken, RPC not yet
        # enqueued, so _pending is still empty).
        server._active_calls = 1  # noqa: SLF001
        await server.sleep()
        assert server.is_resident()  # refused -- call still in flight
        server._active_calls = 0  # noqa: SLF001
        await server.sleep()
        assert not server.is_resident()  # now idle -- sleeps
    finally:
        await server.aclose()


# ---- Abandoned calls: dispatch timeout / interrupt (DESIGN-dispatch-cancellation) ----


# Answers `initialize` promptly, then takes a full second over any
# `tools/call` -- long enough for the caller to give up first. Single
# threaded on purpose: a later request queues behind the slow one exactly
# as the real Dunnes server queues on its `lock (_gate)`.
_SLOW_SERVER = """
import json, sys, time
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
            "protocolVersion":"2024-11-05",
            "capabilities":{"tools":{}},
            "serverInfo":{"name":"slow","version":"0.1"}}}
    elif m == "tools/call":
        time.sleep(1.0)
        out = {"jsonrpc":"2.0","id":rid,"result":{
            "content":[{"type":"text","text":"{}"}],
            "isError":False}}
    else:
        out = {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"unknown"}}
    print(json.dumps(out)); sys.stdout.flush()
"""


# Answers `initialize`, then stops reading stdin entirely while staying
# alive. This is the pipe-buffer wedge, and it cannot be reproduced with an
# in-memory transport double.
_DEAF_SERVER = """
import json, sys, time
line = sys.stdin.readline()
req = json.loads(line)
out = {"jsonrpc":"2.0","id":req["id"],"result":{
    "protocolVersion":"2024-11-05",
    "capabilities":{"tools":{}},
    "serverInfo":{"name":"deaf","version":"0.1"}}}
print(json.dumps(out)); sys.stdout.flush()
time.sleep(30)
"""


async def test_timeout_abandons_the_call_instead_of_leaking_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller that gives up leaves a request the child is still running.

    It must land in `_abandoned` -- not leak in `_pending` (which silently
    disables idle reap forever) and not vanish (which would let the reaper
    kill the child mid-write). While it is outstanding the server is
    degraded, and the late response is what clears it.
    """
    from glados.core.server import _reap_idle_servers

    loop = asyncio.get_running_loop()
    server = _make_inline_server(_SLOW_SERVER, server_id="slow")
    await server.start()
    try:
        with caplog.at_level(logging.WARNING):
            await server.initialize()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(server.call_tool("slow_write", {}), 0.2)

            assert server._pending == {}  # noqa: SLF001 - no leak
            assert server._active_calls == 0  # noqa: SLF001
            assert not server._write_lock.locked()  # noqa: SLF001
            abandoned = server._abandoned  # noqa: SLF001
            assert [tool for tool, _ in abandoned.values()] == ["slow_write"]

            # The reaper must not kill a child that is mid-write.
            await server.sleep()
            assert server.is_resident()
            for coro in _reap_idle_servers([(server, 0.0)], loop.time() + 999.0):
                await coro
            assert server.is_resident()

            # A later call fails fast rather than queueing behind the zombie
            # and burning another full budget.
            started = loop.time()
            degraded = await server.call_tool("read_state", {})
            assert not degraded.ok
            assert "abandoned operation" in degraded.error
            assert "slow_write" in degraded.error
            assert loop.time() - started < 0.2

            # The late response is the only evidence that ever arrives.
            for _ in range(60):
                if not server._abandoned:  # noqa: SLF001
                    break
                await asyncio.sleep(0.05)
            assert server._abandoned == {}  # noqa: SLF001

        assert "late response for abandoned slow_write" in caplog.text
        assert "isError=False" in caplog.text
        # Cleared state means the server is usable again.
        recovered = await server.call_tool("slow_write", {})
        assert recovered.ok
    finally:
        await server.aclose()


async def test_death_clears_abandoned_calls() -> None:
    """No late response can arrive from a child that is gone, so a dead
    server must not stay degraded (or unreapable) forever."""
    server = _make_inline_server(_SLOW_SERVER, server_id="slow")
    await server.start()
    try:
        await server.initialize()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.call_tool("slow_write", {}), 0.2)
        assert server._abandoned  # noqa: SLF001
        server._mark_dead("test")  # noqa: SLF001
        assert server._abandoned == {}  # noqa: SLF001
    finally:
        await server.aclose()


async def test_write_to_a_child_that_stopped_reading_is_bounded() -> None:
    """An unbounded drain against a wedged child holds the write lock
    forever, hanging every later call outside any dispatch timeout.

    The bytes are already buffered when the drain gives up, so the child may
    still run the request whenever it resumes reading -- that makes this an
    abandoned call, not a clean failure.
    """
    loop = asyncio.get_running_loop()
    server = StdioServer(
        sys.executable, ["-c", _DEAF_SERVER], server_id="deaf", write_timeout_s=0.5
    )
    await server.start()
    try:
        await server.initialize()
        started = loop.time()
        result = await server.call_tool("echo", {"text": "x" * (2 * 1024 * 1024)})
        elapsed = loop.time() - started
        assert not result.ok
        assert "stopped reading stdin" in result.error
        assert elapsed < 5.0
        assert not server._write_lock.locked()  # noqa: SLF001
        assert server._pending == {}  # noqa: SLF001
        assert [tool for tool, _ in server._abandoned.values()] == ["echo"]  # noqa: SLF001
    finally:
        await server.aclose()


async def test_a_call_that_never_left_the_client_is_not_abandoned() -> None:
    """A write that gave up waiting for the lock put nothing on the wire, so
    the child cannot be running it and the server must not degrade."""
    server = StdioServer(
        sys.executable, ["-c", _SLOW_SERVER], server_id="slow", write_timeout_s=0.05
    )
    await server.start()
    try:
        await server.initialize()
        await server._write_lock.acquire()  # noqa: SLF001 - simulate a busy writer
        try:
            result = await server.call_tool("never_sent", {})
        finally:
            server._write_lock.release()  # noqa: SLF001
        assert not result.ok
        assert "write lock still held" in result.error
        assert server._abandoned == {}  # noqa: SLF001
        assert server._pending == {}  # noqa: SLF001
    finally:
        await server.aclose()


async def test_an_abandoned_call_that_never_answers_expires(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The late response is the intended exit from degraded, but a child that
    hangs without dying never sends one -- and a server that fast-fails every
    call and can never be reaped is worse than the leak this replaced."""
    server = StdioServer(
        sys.executable, ["-c", _SLOW_SERVER], server_id="slow", abandon_ttl_s=0.05
    )
    await server.start()
    try:
        with caplog.at_level(logging.WARNING):
            await server.initialize()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(server.call_tool("slow_write", {}), 0.2)
            assert server._abandoned  # noqa: SLF001
            await asyncio.sleep(0.1)
            await server.sleep()
        assert server._abandoned == {}  # noqa: SLF001
        assert not server.is_resident()  # reapable again
        assert "never answered" in caplog.text
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
    spec's `requires_confirmation` to True after `tools/list` -- the wire
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
