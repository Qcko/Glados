"""Dummy `time.now` tool -- proves the LLM<->MCP loop end-to-end."""

from __future__ import annotations

from datetime import datetime

from ..core.adapters import ToolSpec
from ..mcp.registry import CallEnvelope, MCPCallResult


class NowTool:
    spec = ToolSpec(
        server="time",
        name="now",
        description="Return the current local time.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        now = datetime.now()
        return MCPCallResult(
            ok=True,
            content={
                "iso": now.isoformat(timespec="seconds"),
                "human": now.strftime("%A %H:%M"),
            },
        )
