"""Tool registry + dispatcher.

Holds both in-process tools and `StdioToolProxy` instances for subprocess
MCP servers — the dispatcher doesn't care which transport sits behind a
spec. Per-tool timeout overrides (`ToolSpec.timeout_s`) win over the
dispatch default; slow tools (Selenium page loads) carry their own
budget that way.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from pydantic import BaseModel

from ..core.adapters import ToolSpec


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
        try:
            return await asyncio.wait_for(tool.call(args, envelope), effective)
        except asyncio.TimeoutError:
            return MCPCallResult(ok=False, error=f"timeout after {effective}s")
        except Exception as e:  # noqa: BLE001 - report any tool error to the LLM
            return MCPCallResult(ok=False, error=f"{type(e).__name__}: {e}")
