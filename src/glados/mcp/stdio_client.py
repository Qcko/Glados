"""Stdio JSON-RPC client for MCP-shape subprocess servers.

A `StdioServer` owns one child process, multiplexes concurrent
`call(name, args)` requests over the single stdin/stdout pair using
incrementing JSON-RPC ids and a futures dict, and exposes a clean
`aclose()` for the lifespan to call on shutdown.

A `StdioToolProxy` wraps `(StdioServer, ToolSpec)` and implements the
existing `Tool` Protocol so the `MCPRegistry` doesn't care whether a
tool runs in-process or in a subprocess.

Crash semantics: when the reader observes EOF or the subprocess exits,
every pending future fails with `StdioServerError`. The next `call_tool`
invocation triggers a bounded auto-restart — up to `max_restarts` within
`restart_window_s` with exponential backoff. Once the budget is spent
the circuit stays open: subsequent calls return a clean
`MCPCallResult(ok=False, error=...)` until `aclose()` (no manual reset
yet; field signal will tell us if we need one). The bound exists so a
hard-crashing server (Selenium browser process gone for good) can't burn
the event loop in a tight respawn loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Sequence

from ..core.adapters import ToolSpec
from .registry import CallEnvelope, MCPCallResult


_log = logging.getLogger(__name__)

# MCP protocol version we advertise in the initialize handshake. Servers
# may pin to an older version in their response; we don't gate on the
# match because today's set of methods (initialize, tools/list,
# tools/call) is stable across the 2024-11-05 / 2025-03-26 line. Bump
# when we adopt a method that requires a newer protocol.
_MCP_PROTOCOL_VERSION = "2024-11-05"


class StdioServerError(RuntimeError):
    pass


class StdioServer:
    def __init__(
        self,
        command: str,
        args: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        server_id: str | None = None,
        max_restarts: int = 3,
        restart_window_s: float = 60.0,
        restart_backoff_s: float = 0.5,
    ) -> None:
        self._command = command
        self._args = list(args)
        self._env = env
        self._cwd = cwd
        self.server_id = server_id or command
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._next_id = 1
        # Serialises writes so two concurrent `call()` invocations cannot
        # interleave bytes on stdin. Reads are owned by the single reader
        # task so no read-side lock is needed.
        self._write_lock = asyncio.Lock()
        self._dead = False
        self._died_reason: str | None = None
        self._closed = False
        # Auto-restart budget: up to `max_restarts` attempts within a
        # rolling `restart_window_s` window. Successful restarts also
        # count — three crashes in a minute is a sign of a deeper problem
        # and we'd rather surface "circuit open" to the LLM than chew
        # CPU respawning forever.
        self._max_restarts = max_restarts
        self._restart_window_s = restart_window_s
        self._restart_backoff_s = restart_backoff_s
        self._restart_attempts: list[float] = []
        self._restart_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._proc is not None:
            return
        # On Windows the default stdin/stdout encoding for a subprocess is
        # the OEM code page, which mangles non-ASCII tool args. We don't
        # care for the toy server but Dunnes will scrape pages with
        # non-ASCII text — set PYTHONIOENCODING and pass the env through.
        env = dict(self._env or {})
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # stderr piped + drained into the logger so server diagnostics
        # land in glados.log regardless of how the parent process was
        # launched. A dedicated drain task keeps the pipe from filling
        # (Windows pipe buffers cap around 64KB).
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**_os_environ(), **env},
            cwd=self._cwd,
        )
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"stdio-reader-{self.server_id}"
        )
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(), name=f"stdio-stderr-{self.server_id}"
        )

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        child_log = logging.getLogger(f"mcp.stdio.{self.server_id}")
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    child_log.info(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never let the drain task tear down the server.
            _log.exception("stdio %s: stderr drain crashed", self.server_id)

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    self._mark_dead("subprocess closed stdout (EOF)")
                    return
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    # Garbage from a misbehaving server. Log to stderr by
                    # re-raising on the future so the LLM sees a usable
                    # error instead of silent stalls.
                    continue
                rid = msg.get("id")
                fut = self._pending.pop(rid, None)
                if fut is None:
                    # Stray response — id never matched a pending call.
                    # Could be a protocol bug in the server or a parse-
                    # error response with id=null. Log so a stalled call
                    # is debuggable post-mortem.
                    _log.debug("stdio %s: dropping unmatched response id=%r", self.server_id, rid)
                    continue
                if not fut.done():
                    fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - blanket guard for reader thread
            self._mark_dead(f"reader crashed: {type(e).__name__}: {e}")

    def _mark_dead(self, reason: str) -> None:
        if self._dead:
            return
        self._dead = True
        self._died_reason = reason
        # Logged at error level each time a server dies. With auto-restart
        # the same server can die multiple times across a session; each
        # death is interesting on its own. /healthz could surface a
        # persistent dead state in a follow-up.
        _log.error("stdio server %s died: %s", self.server_id, reason)
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(StdioServerError(reason))
        self._pending.clear()

    async def _send_notification(self, method: str, params: dict | None = None) -> None:
        """JSON-RPC notification — no id, no response expected. Used by
        the MCP `notifications/initialized` handshake step."""
        if self._closed or self._dead or self._proc is None or self._proc.stdin is None:
            return
        req: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        payload = (json.dumps(req) + "\n").encode("utf-8")
        try:
            async with self._write_lock:
                self._proc.stdin.write(payload)
                await self._proc.stdin.drain()
        except Exception as e:  # noqa: BLE001 - pipe broken, reader will mark dead
            # Notification failures are non-fatal (the recipient doesn't
            # ack) but worth a debug line so a partial-restart "succeeded
            # then died" is traceable later.
            _log.debug(
                "stdio %s: notification %s failed: %s",
                self.server_id,
                method,
                e,
            )

    async def _call_method(self, method: str, params: dict | None = None) -> dict:
        if self._closed:
            raise StdioServerError(f"stdio server {self.server_id} is closed")
        if self._dead:
            raise StdioServerError(self._died_reason or "stdio server is dead")
        if self._proc is None or self._proc.stdin is None:
            raise StdioServerError("stdio server not started")
        rid = self._next_id
        self._next_id += 1
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        payload = (json.dumps(req) + "\n").encode("utf-8")
        try:
            async with self._write_lock:
                self._proc.stdin.write(payload)
                await self._proc.stdin.drain()
        except Exception as e:  # noqa: BLE001 - pipe broken, server gone
            # Pop the future so it doesn't leak in `_pending`; the reader
            # task may also independently call `_mark_dead` shortly.
            self._pending.pop(rid, None)
            raise StdioServerError(f"write failed: {type(e).__name__}: {e}") from e
        return await fut

    async def initialize(self) -> dict:
        """Real MCP initialize handshake. Sends protocolVersion +
        clientInfo, awaits the server's serverInfo/capabilities, then
        fires the `notifications/initialized` notification per the MCP
        spec. The server is allowed to refuse tool calls until that
        notification arrives."""
        resp = await self._call_method(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "glados", "version": "0.1"},
            },
        )
        result = _result_or_raise(resp)
        await self._send_notification("notifications/initialized")
        return result

    async def list_tools(self) -> list[ToolSpec]:
        """Fetch the server's tool catalogue and translate real-MCP shape
        (`{name, description, inputSchema}`) into GLaDOS's `ToolSpec`
        (`{server, name, description, parameters, ...}`). The `server`
        field is injected from our configured `server_id` because real
        MCP doesn't carry a per-tool server identifier — server identity
        is implicit in the subprocess. GLaDOS-only flags (`untrusted`,
        `requires_confirmation`, `timeout_s`) are NOT on the wire; they
        live in `servers.toml` `tool_overlays` and are applied after this
        method returns."""
        resp = await self._call_method("tools/list")
        result = _result_or_raise(resp)
        specs: list[ToolSpec] = []
        for t in result.get("tools", []):
            name = t.get("name")
            if not isinstance(name, str) or not name:
                _log.warning("stdio %s: skipping tool with missing name", self.server_id)
                continue
            specs.append(
                ToolSpec(
                    server=self.server_id,
                    name=name,
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema") or {"type": "object", "properties": {}},
                )
            )
        return specs

    async def call_tool(self, name: str, args: dict) -> MCPCallResult:
        if self._dead and not self._closed:
            ok = await self._try_restart()
            if not ok:
                return MCPCallResult(
                    ok=False,
                    error=self._died_reason or "stdio server unavailable",
                )
        try:
            resp = await self._call_method(
                "tools/call", {"name": name, "arguments": args}
            )
        except StdioServerError as e:
            return MCPCallResult(ok=False, error=str(e))
        if "error" in resp:
            err = resp["error"]
            return MCPCallResult(ok=False, error=err.get("message", "rpc error"))
        return _translate_tool_result(resp.get("result") or {})

    async def _try_restart(self) -> bool:
        """Bounded auto-restart. Returns True if the server is live after
        this call. Idempotent — concurrent callers serialise on the lock
        and only one respawn actually fires per dead-window."""
        async with self._restart_lock:
            if not self._dead or self._closed:
                # Another caller already brought it back, or aclose ran.
                return not self._closed and not self._dead
            now = asyncio.get_running_loop().time()
            cutoff = now - self._restart_window_s
            self._restart_attempts = [t for t in self._restart_attempts if t > cutoff]
            if len(self._restart_attempts) >= self._max_restarts:
                self._died_reason = (
                    f"stdio server {self.server_id} circuit open: "
                    f"{self._max_restarts} restart attempts in the last "
                    f"{self._restart_window_s:.0f}s"
                )
                return False
            backoff = self._restart_backoff_s * (2 ** len(self._restart_attempts))
            await asyncio.sleep(backoff)
            self._restart_attempts.append(now)
            # Drain whatever's left of the dead instance. _mark_dead has
            # already failed _pending and cancelled nothing; we leave the
            # old proc handle alone (it's already exited) and just null it
            # so start() respawns a fresh one.
            self._proc = None
            self._reader_task = None
            self._stderr_task = None
            self._dead = False
            self._died_reason = None
            try:
                await self.start()
                await self.initialize()
            except Exception as e:  # noqa: BLE001
                # initialize() raises StdioServerError if the new subprocess
                # also dies before responding; _mark_dead has already set
                # _died_reason in that case.
                if not self._dead:
                    self._mark_dead(f"restart failed: {type(e).__name__}: {e}")
                return False
            _log.info(
                "stdio server %s restarted (%d/%d in window)",
                self.server_id,
                len(self._restart_attempts),
                self._max_restarts,
            )
            return True

    async def aclose(self) -> None:
        # Mark closed first so a concurrent call_tool doesn't kick off an
        # auto-restart against the proc we're tearing down.
        self._closed = True
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._proc is not None:
            try:
                if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                    self._proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        self._mark_dead("server closed")


class StdioToolProxy:
    """Implements the `Tool` Protocol by delegating to a `StdioServer`."""

    def __init__(self, server: StdioServer, spec: ToolSpec) -> None:
        self._server = server
        self.spec = spec

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        # TODO(envelope): pass session_id/room_id/speaker_id through to
        # the subprocess (audit logging, per-user routing in Dunnes).
        # Today's toy server doesn't read them.
        return await self._server.call_tool(self.spec.name, args)


def _translate_tool_result(result: dict) -> MCPCallResult:
    """Translate a real-MCP `tools/call` result into `MCPCallResult`.

    Real MCP carries an array of typed content blocks
    (`[{"type":"text","text":"..."}, ...]`) plus an `isError` flag.
    GLaDOS uses a flat `{ok, content (dict|None), error (str|None)}`.

    Translation rules:
    - Concatenate all text-typed blocks into one string. Non-text blocks
      (images, embedded resources) are dropped today — bring back when a
      real consumer needs them.
    - If `isError`, return ok=False with that text as the error.
    - Otherwise, try to parse the text as JSON. If it's a JSON object,
      use it directly as `content`. This is how MCP servers ship
      structured data (the spec has no first-class object response —
      everything rides in `text` blocks). Falling back to wrapping the
      raw string as `{"text": "..."}` keeps non-JSON tools usable.
    """
    is_error = bool(result.get("isError"))
    content_items = result.get("content") or []
    text_parts: list[str] = []
    for item in content_items:
        if isinstance(item, dict) and item.get("type") == "text":
            t = item.get("text")
            if isinstance(t, str):
                text_parts.append(t)
    text = "\n".join(text_parts)
    if is_error:
        return MCPCallResult(ok=False, error=text or "tool error")
    if text:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return MCPCallResult(ok=True, content=parsed)
        if parsed is not None:
            # Valid JSON but not an object — list, scalar, bool, null.
            # MCPCallResult.content is dict-only, so wrap under "value"
            # rather than discarding the parse and re-stringifying.
            return MCPCallResult(ok=True, content={"value": parsed})
    return MCPCallResult(ok=True, content={"text": text})


def _result_or_raise(resp: dict) -> dict:
    if "error" in resp:
        err = resp["error"]
        raise StdioServerError(err.get("message", "rpc error"))
    return resp.get("result") or {}


def _os_environ() -> dict[str, str]:
    return dict(os.environ)
