"""Tool registry + dispatcher.

v0 step 2: in-process tools only. Real MCP stdio transport lands in v3 once
the protocol shape is stable.
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
        try:
            return await asyncio.wait_for(tool.call(args, envelope), timeout)
        except asyncio.TimeoutError:
            return MCPCallResult(ok=False, error=f"timeout after {timeout}s")
        except Exception as e:  # noqa: BLE001 - report any tool error to the LLM
            return MCPCallResult(ok=False, error=f"{type(e).__name__}: {e}")
