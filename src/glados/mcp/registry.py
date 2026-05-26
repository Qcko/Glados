"""Tool registry + dispatcher.

Holds both in-process tools and `StdioToolProxy` instances for subprocess
MCP servers — the dispatcher doesn't care which transport sits behind a
spec. Per-tool timeout overrides (`ToolSpec.timeout_s`) win over the
dispatch default; slow tools (Selenium page loads) carry their own
budget that way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Protocol

from pydantic import BaseModel

from ..core.adapters import ToolSpec


_log = logging.getLogger(__name__)


# Keep error-message arg dumps short so the LLM-visible failure stays
# readable even when a tool was called with a large payload.
_ARG_DUMP_MAX = 120


def _format_args(args: dict) -> str:
    try:
        rendered = json.dumps(args, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = repr(args)
    if len(rendered) > _ARG_DUMP_MAX:
        rendered = rendered[: _ARG_DUMP_MAX - 1] + "…"
    return rendered


class CallEnvelope(BaseModel):
    session_id: str
    room_id: str
    speaker_id: str


class MCPCallResult(BaseModel):
    ok: bool
    content: dict | None = None
    error: str | None = None


class Tool(Protocol):
    spec: ToolSpec

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult: ...


class MCPRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.qualified] = tool

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def spec_for(self, server: str, name: str) -> ToolSpec | None:
        tool = self._tools.get(f"{server}.{name}")
        return tool.spec if tool is not None else None

    async def dispatch(
        self,
        server: str,
        name: str,
        args: dict,
        envelope: CallEnvelope,
        *,
        timeout: float = 8.0,
    ) -> MCPCallResult:
        key = f"{server}.{name}"
        tool = self._tools.get(key)
        if tool is None:
            return MCPCallResult(ok=False, error=f"unknown tool: {key}")
        # Per-tool override on the spec wins over the dispatch default. Lets
        # slow tools (Selenium page loads) carry their own budget instead of
        # forcing every call site to know.
        effective = tool.spec.timeout_s if tool.spec.timeout_s is not None else timeout
        # Same truncated rendering used for entry, exit, timeout, and error
        # lines — avoids a multi-KB tool argument flooding the log.
        rendered_args = _format_args(args)
        _log.info("dispatch %s args=%s timeout=%ss", key, rendered_args, effective)
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(tool.call(args, envelope), effective)
            _log.info("dispatch %s ok=%s elapsed=%dms", key, result.ok, int((time.monotonic() - started) * 1000))
            return result
        except asyncio.TimeoutError:
            _log.warning("dispatch %s timeout after %ss", key, effective)
            return MCPCallResult(ok=False, error=f"timeout after {effective}s calling {key}({rendered_args})")
        except Exception as e:  # noqa: BLE001 - report any tool error to the LLM
            _log.exception("dispatch %s raised", key)
            return MCPCallResult(ok=False, error=f"{type(e).__name__} calling {key}({rendered_args}): {e}")
