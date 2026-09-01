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
invocation triggers a bounded auto-restart -- up to `max_restarts` within
`restart_window_s` with exponential backoff. Once the budget is spent
the circuit stays open: subsequent calls return a clean
`MCPCallResult(ok=False, error=...)` until `aclose()` (no manual reset
yet; field signal will tell us if we need one). The bound exists so a
hard-crashing server (Selenium browser process gone for good) can't burn
the event loop in a tight respawn loop.

Abandonment semantics: a caller that stops waiting -- a dispatch timeout or
a user interrupt -- cancels the Python await, not the child's work. That
request moves to `_abandoned` and the server is degraded until it answers:
later calls fail fast instead of queueing behind it, and the idle reaper
refuses to sleep (and so to kill) a child that is still mid-write. See
`DESIGN-dispatch-cancellation.md`.
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

# Well-known resource URI for a server's co-located lessons memory (ARCH
# section 14). A uniform constant across servers -- namespace by document kind if
# a server ever ships several (`memory://quirks`), never by server name
# (that's redundant with the connection and breaks the constant).
_LESSONS_URI = "memory://lessons"

# Cap on remembered abandoned requests. A permanently wedged server would
# otherwise accumulate one entry per timed-out call for the life of the
# process; the oldest is forgotten first, since the newest abandoned write
# is the one whose outcome is still in question.
_MAX_ABANDONED = 16

# Every write to the child is bounded. A child that has stopped reading
# stdin fills the pipe buffer, `drain()` never returns, and the write lock
# is held forever -- hanging every later call in the write, outside any
# timeout the registry applies.
_WRITE_TIMEOUT_S = 5.0

# How long a server stays degraded waiting for an abandoned call to answer.
# Without a bound, a child that hangs without dying takes the server out
# permanently: every call fast-fails and the reaper never sleeps it. Long
# enough that a slow browser write is still expected to arrive first.
_ABANDON_TTL_S = 300.0

# How long a child gets to exit on its own after stdin closes, and then again
# after kill(). Bounded twice: this runs inside a dispatch the registry is
# already timing, so an unbounded wait here becomes a hung turn.
_STOP_GRACE_S = 2.0


class StdioServerError(RuntimeError):
    pass


class _WriteAttempt:
    """Whether a message reached the child's pipe before the write failed.

    A cancelled or timed-out `drain()` does not unwrite the bytes already
    handed to the transport, so the child may still execute the request. The
    caller needs that distinction to tell a call that never left from one that
    is now running unattended.
    """

    def __init__(self) -> None:
        self.reached_the_child = False


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
        max_abandoned: int = _MAX_ABANDONED,
        write_timeout_s: float = _WRITE_TIMEOUT_S,
        abandon_ttl_s: float = _ABANDON_TTL_S,
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
        # Requests whose caller stopped waiting (dispatch timeout, user
        # interrupt) but which the child is still executing. `_pending` is
        # wrong for these: the caller is gone, yet the work is not. Keyed by
        # request id, carrying the tool name and start time so the late
        # response can be logged with something a human can act on --
        # `_pending` carries neither. While this map is non-empty the server
        # is DEGRADED: it may not be slept (a `kill()` mid-write turns an
        # unknown outcome into a partially-applied one) and later calls
        # fast-fail rather than queueing behind the zombie.
        self._abandoned: dict[int, tuple[str, float]] = {}
        self._next_id = 1
        # Lazy-spawn state (ARCH section 13 "lazy MCP child spawn + idle reap"). A
        # dormant server has no child process but is NOT closed: the next
        # `call_tool` wakes it via a clean start()+initialize(), and the
        # idle reaper sleeps it again after `idle_timeout_s` of no calls.
        # Distinct from `_dead` (crash) and `_closed` (permanent shutdown):
        # waking is an intended transition, so it does NOT count against the
        # crash-restart budget. `_last_activity` stamps the monotonic clock
        # on every call_tool so the reaper never sleeps an active server.
        self._dormant = False
        self._last_activity = 0.0
        # In-flight call_tool count. The reaper must not sleep a server while a
        # dispatch is mid-wake-then-RPC: `_pending` is only populated once the
        # RPC is actually sent (after `_wake` returns), so a check on `_pending`
        # alone leaves a window where a just-woken server gets reaped out from
        # under the call. This counter spans the whole call_tool body, closing
        # that window -- `sleep()` refuses while it's non-zero.
        self._active_calls = 0
        # Serialises writes so two concurrent `call()` invocations cannot
        # interleave bytes on stdin. Reads are owned by the single reader
        # task so no read-side lock is needed.
        self._write_lock = asyncio.Lock()
        self._dead = False
        self._died_reason: str | None = None
        self._closed = False
        # True only for the duration of `_stop_child`. A child that exits of
        # its own accord inside that window makes the reader observe EOF, which
        # is `_read_loop`'s death signal -- and a death we asked for is not a
        # death worth reporting.
        #
        # Its one live effect is the idle reap: of the three callers of
        # `_stop_child`, `aclose` clears this before its own deliberate
        # `_mark_dead`, and `_try_restart` only runs when `_dead` is already set
        # (where `_mark_dead` was a no-op regardless). That leaves `sleep`,
        # where the payoff is not state -- `sleep` resets `_dead` itself -- but
        # the log: reaping an idle server must not leave a
        # `died: subprocess closed stdout (EOF)` at ERROR for somebody to chase.
        self._stopping = False
        # Auto-restart budget: up to `max_restarts` attempts within a
        # rolling `restart_window_s` window. Successful restarts also
        # count -- three crashes in a minute is a sign of a deeper problem
        # and we'd rather surface "circuit open" to the LLM than chew
        # CPU respawning forever.
        self._max_restarts = max_restarts
        self._restart_window_s = restart_window_s
        self._restart_backoff_s = restart_backoff_s
        self._restart_attempts: list[float] = []
        self._restart_lock = asyncio.Lock()
        self._max_abandoned = max_abandoned
        self._write_timeout_s = write_timeout_s
        self._abandon_ttl_s = abandon_ttl_s

    async def start(self) -> None:
        if self._proc is not None:
            return
        # On Windows the default stdin/stdout encoding for a subprocess is
        # the OEM code page, which mangles non-ASCII tool args. We don't
        # care for the toy server but Dunnes will scrape pages with
        # non-ASCII text -- set PYTHONIOENCODING and pass the env through.
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
                    if self._claim_late_response(rid, msg):
                        continue
                    # Stray response -- id never matched a pending call.
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

    def _claim_late_response(self, rid: object, msg: dict) -> bool:
        """Match a response against the abandoned map. True if it was one.

        This is the only evidence that ever arrives about an abandoned call:
        whether the write landed, and how long the child really took. It also
        ends the degraded state -- the child has finished and is reading again.
        """
        entry = self._abandoned.pop(rid, None) if isinstance(rid, int) else None
        if entry is None:
            return False
        tool, started = entry
        elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000.0
        result = msg.get("result")
        if not isinstance(result, dict):
            result = {}
        _log.warning(
            "stdio %s: late response for abandoned %s after %.0fms "
            "(isError=%s, rpc_error=%s) -- the call the model was told failed "
            "may have landed",
            self.server_id,
            tool,
            elapsed_ms,
            bool(result.get("isError")),
            "error" in msg,
        )
        return True

    def _mark_dead(self, reason: str) -> None:
        if self._dead or self._stopping:
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
        # The child is gone, so no late response can ever arrive to clear
        # these. Holding them would wedge the reaper and the degraded gate on
        # a server that no longer exists.
        self._abandoned.clear()

    async def _write_payload(
        self, payload: bytes, what: str, attempt: _WriteAttempt | None = None
    ) -> None:
        """Write one framed message to the child, bounded at both steps.

        Both the lock acquire and the drain can block forever against a child
        that has stopped reading stdin, and neither is covered by the caller's
        dispatch timeout -- so a single wedged child would hang every later
        call in the write. Failure is always a `StdioServerError` naming the
        message that could not be sent.
        """
        stdin = self._proc.stdin if self._proc is not None else None
        if stdin is None:
            raise StdioServerError(f"stdio server not started, cannot send {what}")
        try:
            await asyncio.wait_for(self._write_lock.acquire(), self._write_timeout_s)
        except asyncio.TimeoutError as e:
            raise StdioServerError(
                f"write lock still held after {self._write_timeout_s:.0f}s "
                f"sending {what} to {self.server_id}"
            ) from e
        try:
            stdin.write(payload)
            if attempt is not None:
                attempt.reached_the_child = True
            await asyncio.wait_for(stdin.drain(), self._write_timeout_s)
        except asyncio.TimeoutError as e:
            raise StdioServerError(
                f"child stopped reading stdin: drain stalled {self._write_timeout_s:.0f}s "
                f"sending {what} to {self.server_id}"
            ) from e
        except Exception as e:  # noqa: BLE001 - pipe broken, server gone
            raise StdioServerError(
                f"write failed sending {what} to {self.server_id}: {type(e).__name__}: {e}"
            ) from e
        finally:
            self._write_lock.release()

    async def _send_notification(self, method: str, params: dict | None = None) -> None:
        """JSON-RPC notification -- no id, no response expected. Used by
        the MCP `notifications/initialized` handshake step."""
        if self._closed or self._dead or self._proc is None or self._proc.stdin is None:
            return
        req: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        payload = (json.dumps(req) + "\n").encode("utf-8")
        try:
            await self._write_payload(payload, f"notification {method}")
        except StdioServerError as e:
            # Notification failures are non-fatal (the recipient doesn't
            # ack) but worth a debug line so a partial-restart "succeeded
            # then died" is traceable later.
            _log.debug("stdio %s: %s", self.server_id, e)

    async def _call_method(
        self, method: str, params: dict | None = None, *, label: str | None = None
    ) -> dict:
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
        attempt = _WriteAttempt()
        try:
            await self._write_payload(payload, f"{method} id={rid}", attempt)
            return await fut
        except (asyncio.CancelledError, StdioServerError):
            # Two ways to reach here with the child still working: the caller
            # stopped waiting (dispatch timeout, user interrupt), or the write
            # timed out after the bytes were already buffered. Both are
            # abandoned calls rather than failed ones. A write that never
            # reached the pipe is a clean failure, and a future carrying an
            # answer means the response beat the cancellation -- degrading the
            # server on either would be a lie in the other direction.
            if attempt.reached_the_child and not _answered(fut):
                self._abandon(rid, label or method)
            elif _answered(fut):
                _log.warning(
                    "stdio %s: %s answered as the caller gave up; the result is "
                    "discarded and the call reported as a failure",
                    self.server_id,
                    label or method,
                )
            raise
        finally:
            # Leaving the entry here on a cancelled call silently disables idle
            # reap for the server: `sleep()` refuses while `_pending` is
            # non-empty, and nothing else ever pops it.
            self._pending.pop(rid, None)

    def _abandon(self, rid: int, tool: str) -> None:
        """Remember a request whose caller gave up while the child works on."""
        if len(self._abandoned) >= self._max_abandoned:
            forgotten_rid, (forgotten_tool, _) = next(iter(self._abandoned.items()))
            self._abandoned.pop(forgotten_rid)
            _log.warning(
                "stdio %s: abandoned-call map full, forgetting id=%s (%s)",
                self.server_id,
                forgotten_rid,
                forgotten_tool,
            )
        self._abandoned[rid] = (tool, asyncio.get_running_loop().time())
        _log.warning(
            "stdio %s: abandoned %s (id=%s) still running in the child; "
            "server degraded until it answers",
            self.server_id,
            tool,
            rid,
        )

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
        MCP doesn't carry a per-tool server identifier -- server identity
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

    async def read_lessons(self) -> str | None:
        """Fetch the server's co-located lessons memory (ARCH section 14 layer 1).

        Reads the well-known resource `memory://lessons` (`text/markdown`)
        over MCP `resources/read` and returns the concatenated text, or
        None if the server exposes no such resource / returns no text /
        errors. The URI is a uniform constant across servers -- provenance
        comes from the connection's `server_id`, stamped by the caller when
        it wraps the content, not from the URI. Best-effort: any failure
        means "this server has no lessons to inject", never a hard error,
        because a missing resource is the common case (most servers ship
        none) and must not break startup.
        """
        try:
            resp = await self._call_method(
                "resources/read", {"uri": _LESSONS_URI}
            )
        except StdioServerError as e:
            _log.debug("stdio %s: read_lessons failed: %s", self.server_id, e)
            return None
        if "error" in resp:
            # Servers that don't implement resources reply with a JSON-RPC
            # error (method not found / resource not found). Expected for
            # any server that ships no memory; debug, not warn.
            _log.debug(
                "stdio %s: no %s resource: %s",
                self.server_id,
                _LESSONS_URI,
                (resp.get("error") or {}).get("message", "rpc error"),
            )
            return None
        return _lessons_text(resp.get("result") or {})

    async def call_tool(self, name: str, args: dict) -> MCPCallResult:
        self._forget_expired_abandoned()
        if self._abandoned:
            # A rejected call is still traffic: without this the server reads
            # as maximally idle the instant it recovers, and the next reaper
            # tick sleeps a child that has been asked for work throughout.
            self._last_activity = asyncio.get_running_loop().time()
            # Queueing behind an abandoned operation burns another full
            # timeout and, on a mutating tool, would inherit an indeterminate
            # outcome it has not earned. Fail fast and deterministically
            # instead; the late response clears this state.
            return MCPCallResult(
                ok=False,
                error=(
                    f"{self.server_id} is busy with an abandoned operation "
                    f"({self._abandoned_summary()}); its state is unknown"
                ),
            )
        # Hold the in-flight guard across the whole call -- wake, restart, AND
        # the RPC -- so the reaper can't sleep the server mid-dispatch (the
        # window between a successful `_wake` and `_call_method` populating
        # `_pending`). Stamp activity too so a fresh call also reads non-idle.
        self._active_calls += 1
        try:
            self._last_activity = asyncio.get_running_loop().time()
            if self._dormant and not self._closed:
                ok = await self._wake()
                if not ok:
                    return MCPCallResult(
                        ok=False,
                        error=self._died_reason or "stdio server failed to wake",
                    )
            if self._dead and not self._closed:
                ok = await self._try_restart()
                if not ok:
                    return MCPCallResult(
                        ok=False,
                        error=self._died_reason or "stdio server unavailable",
                    )
            try:
                resp = await self._call_method(
                    "tools/call", {"name": name, "arguments": args}, label=name
                )
            except StdioServerError as e:
                return MCPCallResult(ok=False, error=str(e))
            if "error" in resp:
                err = resp["error"]
                return MCPCallResult(ok=False, error=err.get("message", "rpc error"))
            return _translate_tool_result(resp.get("result") or {})
        finally:
            self._active_calls -= 1

    def _forget_expired_abandoned(self) -> None:
        """Stop degrading the server on a call that will never be answered.

        The late response is the intended exit, but a child that hangs without
        dying never sends one -- and then every call fast-fails and the reaper
        never sleeps it. Forgetting the entry returns the server to ordinary
        service; what actually happened to that write stays unknown.
        """
        now = asyncio.get_running_loop().time()
        for rid, (tool, started) in list(self._abandoned.items()):
            if now - started < self._abandon_ttl_s:
                continue
            self._abandoned.pop(rid, None)
            _log.warning(
                "stdio %s: abandoned %s (id=%s) never answered in %.0fs; "
                "no longer degraded, outcome unknown",
                self.server_id,
                tool,
                rid,
                now - started,
            )

    def _abandoned_summary(self) -> str:
        return ", ".join(tool for tool, _ in self._abandoned.values())

    async def _try_restart(self) -> bool:
        """Bounded auto-restart. Returns True if the server is live after
        this call. Idempotent -- concurrent callers serialise on the lock
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
            # Tear the dead instance down before respawning. "Dead" has two
            # causes and only one of them means the child exited: EOF on stdout
            # does, a crashed reader task does NOT -- there the process is still
            # alive with nobody reading it. Nulling the handle there orphans a
            # live child for the rest of the session, which for a Selenium
            # server is a leaked browser. `_stop_child` is idempotent and
            # returns immediately for a process that has already gone.
            await self._stop_child()
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

    async def _stop_child(self) -> None:
        """Tear down the child process and its reader/stderr tasks. Cancels
        the reader BEFORE closing stdin so the resulting EOF can't race
        `_mark_dead`, and holds `_stopping` for the whole teardown so a child
        that exits on its own inside the window is not reported as a death
        either. Leaves `_proc = None`. Shared by `aclose` (permanent), `sleep`
        (dormant) and `_try_restart` (replacing a dead instance) -- the caller
        sets the resulting state flag.

        `_stopping` is a plain bool rather than a depth count, which is only
        safe because every one of those three callers holds `_restart_lock`
        across the await. A fourth caller added without it would let an inner
        teardown clear the flag out from under an outer one."""
        self._stopping = True
        try:
            await _settle(self._reader_task)
            await _settle(self._stderr_task)
            if self._proc is not None:
                try:
                    if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                        self._proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=_STOP_GRACE_S)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=_STOP_GRACE_S)
                    except asyncio.TimeoutError:
                        # A Chrome tree that TerminateProcess did not reap would
                        # otherwise hang here forever -- and `_try_restart` reaches
                        # this from inside a dispatch the registry is timing.
                        _log.warning(
                            "stdio %s: child survived kill(); abandoning the handle",
                            self.server_id,
                        )
        finally:
            # Cleared before the caller runs, so `aclose`'s own
            # `_mark_dead("server closed")` still lands and still fails every
            # pending caller. The handles are dropped here too: a teardown that
            # raised part-way still leaves no half-owned child behind.
            self._stopping = False
            self._proc = None
            self._reader_task = None
            self._stderr_task = None

    async def sleep(self) -> None:
        """Put a lazy server dormant: stop the child but stay reanimatable.

        Idempotent and serialised on `_restart_lock` so it can't interleave
        with a concurrent `_wake` / `_try_restart`. Refuses to sleep while a
        call is in flight -- either an RPC already on the wire (`_pending`) or a
        `call_tool` anywhere in its wake-then-dispatch span (`_active_calls`).
        Together these close the window where the reaper would reap a server a
        dispatch just woke.

        It also refuses while a call is abandoned. Nobody is awaiting that one,
        so neither of the other two conditions sees it -- and `_stop_child()`
        would `kill()` the child mid-write, turning an unknown outcome into a
        possibly partially-applied one."""
        self._forget_expired_abandoned()
        async with self._restart_lock:
            if (
                self._closed
                or self._proc is None
                or self._pending
                or self._active_calls
                or self._abandoned
            ):
                return
            await self._stop_child()
            self._dormant = True
            self._dead = False
            self._died_reason = None
            _log.info("stdio server %s sleeping (idle reap)", self.server_id)

    async def _wake(self) -> bool:
        """Bring a dormant server back. Unlike `_try_restart` this is an
        intended transition, so it does not consume the crash-restart budget.
        Returns True if the server is live afterward."""
        async with self._restart_lock:
            if self._closed:
                return False
            if not self._dormant and self._proc is not None and not self._dead:
                return True  # another caller already woke it
            self._dormant = False
            self._dead = False
            self._died_reason = None
            # A fresh child cannot answer the previous one's requests.
            self._abandoned.clear()
            try:
                await self.start()
                await self.initialize()
            except Exception as e:  # noqa: BLE001
                if not self._dead:
                    self._mark_dead(f"wake failed: {type(e).__name__}: {e}")
                return False
            _log.info("stdio server %s woke from dormant", self.server_id)
            return True

    def is_resident(self) -> bool:
        """True if a child process is currently running (not dormant/closed)."""
        return self._proc is not None and not self._dormant and not self._closed

    def idle_seconds(self, now: float) -> float:
        return now - self._last_activity

    async def aclose(self) -> None:
        # Mark closed first so a concurrent call_tool doesn't kick off an
        # auto-restart (or a wake) against the proc we're tearing down.
        self._closed = True
        async with self._restart_lock:
            await self._stop_child()
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
      (images, embedded resources) are dropped today -- bring back when a
      real consumer needs them.
    - If `isError`, return ok=False with that text as the error.
    - Otherwise, try to parse the text as JSON. If it's a JSON object,
      use it directly as `content`. This is how MCP servers ship
      structured data (the spec has no first-class object response --
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
            # Valid JSON but not an object -- list, scalar, bool, null.
            # MCPCallResult.content is dict-only, so wrap under "value"
            # rather than discarding the parse and re-stringifying.
            return MCPCallResult(ok=True, content={"value": parsed})
    return MCPCallResult(ok=True, content={"text": text})


def _lessons_text(result: dict) -> str | None:
    """Pull markdown out of a `resources/read` result.

    MCP returns `{"contents": [{"uri", "mimeType", "text"|"blob"}, ...]}`.
    We concatenate the `text` of every text-bearing content block (binary
    `blob` blocks are skipped -- lessons are markdown). Empty/whitespace
    text collapses to None so the caller treats it as "no lessons".
    """
    parts: list[str] = []
    for item in result.get("contents") or []:
        if isinstance(item, dict):
            t = item.get("text")
            if isinstance(t, str):
                parts.append(t)
    text = "\n".join(parts).strip()
    return text or None


def _result_or_raise(resp: dict) -> dict:
    if "error" in resp:
        err = resp["error"]
        raise StdioServerError(err.get("message", "rpc error"))
    return resp.get("result") or {}


async def _settle(task: asyncio.Task | None) -> None:
    """Cancel a task if it is still running, and swallow however it ended.

    A task that died on its own holds its exception until somebody asks for it,
    and asyncio logs an alarming "never retrieved" at GC if nobody does -- for
    a death `_mark_dead` has already reported properly.
    """
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def _answered(fut: asyncio.Future[dict]) -> bool:
    """True only if the child's response actually landed on the future."""
    return fut.done() and not fut.cancelled()


def _os_environ() -> dict[str, str]:
    return dict(os.environ)
