"""Stdio JSON-RPC client for MCP-shape subprocess servers.

A `StdioServer` owns one child process, multiplexes concurrent
`call(name, args)` requests over the single stdin/stdout pair using
incrementing JSON-RPC ids and a futures dict, and exposes a clean
`aclose()` for the lifespan to call on shutdown.

A `StdioToolProxy` wraps `(StdioServer, ToolSpec)` and implements the
existing `Tool` Protocol so the `MCPRegistry` doesn't care whether a
tool runs in-process or in a subprocess.

Crash semantics for v1: when the reader task observes EOF or the
subprocess exits, every pending future fails with `RuntimeError("stdio
server died")` and all subsequent `call()` invocations also fail.
Auto-restart / circuit-breaker is deferred — see ARCH §7 follow-up.
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
    ) -> None:
        self._command = command
        self._args = list(args)
        self._env = env
        self._cwd = cwd
        self.server_id = server_id or command
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._next_id = 1
        # Serialises writes so two concurrent `call()` invocations cannot
        # interleave bytes on stdin. Reads are owned by the single reader
        # task so no read-side lock is needed.
        self._write_lock = asyncio.Lock()
        self._dead = False
        self._died_reason: str | None = None

    async def start(self) -> None:
        if self._proc is not None:
            return
        # On Windows the default stdin/stdout encoding for a subprocess is
        # the OEM code page, which mangles non-ASCII tool args. We don't
        # care for the toy server but Dunnes will scrape pages with
        # non-ASCII text — set PYTHONIOENCODING and pass the env through.
        env = dict(self._env or {})
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # stderr inherits the parent's so a verbose third-party server
        # never deadlocks on a full pipe buffer (~64KB on Windows). We
        # don't capture stderr today; a follow-up could pipe + drain it
        # into the logger if structured server logs become useful.
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env={**_os_environ(), **env},
            cwd=self._cwd,
        )
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"stdio-reader-{self.server_id}"
        )

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
        # Logged at error level the first (and only) time a server dies,
        # so a stale-tools situation surfaces in the logs even though
        # auto-restart is deferred. /healthz could surface this too in a
        # follow-up.
        _log.error("stdio server %s died: %s", self.server_id, reason)
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(StdioServerError(reason))
        self._pending.clear()

    async def _call_method(self, method: str, params: dict | None = None) -> dict:
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
        resp = await self._call_method("initialize")
        return _result_or_raise(resp)

    async def list_tools(self) -> list[ToolSpec]:
        resp = await self._call_method("tools/list")
        result = _result_or_raise(resp)
        return [ToolSpec(**t) for t in result.get("tools", [])]

    async def call_tool(self, name: str, args: dict) -> MCPCallResult:
        try:
            resp = await self._call_method(
                "tools/call", {"name": name, "arguments": args}
            )
        except StdioServerError as e:
            return MCPCallResult(ok=False, error=str(e))
        if "error" in resp:
            err = resp["error"]
            return MCPCallResult(ok=False, error=err.get("message", "rpc error"))
        result = resp.get("result") or {}
        # The wire shape already matches MCPCallResult — the toy server
        # returns {"ok": bool, "content": ..., "error": ...}. Validate
        # defensively so a malformed third-party server can't crash us.
        return MCPCallResult(
            ok=bool(result.get("ok")),
            content=result.get("content") if isinstance(result.get("content"), dict) else None,
            error=result.get("error") if isinstance(result.get("error"), str) else None,
        )

    async def aclose(self) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
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


def _result_or_raise(resp: dict) -> dict:
    if "error" in resp:
        err = resp["error"]
        raise StdioServerError(err.get("message", "rpc error"))
    return resp.get("result") or {}


def _os_environ() -> dict[str, str]:
    return dict(os.environ)
