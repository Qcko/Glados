"""Generic toy tools -- stand-in MCP surface for the demo while real
third-party servers (LifeQuests, etc.) are still gated.

Three tools, all in-process:
- toy.echo: returns the input string
- toy.add: returns a + b
- toy.roll_dice: rolls `count` dice with `sides` faces
"""

from __future__ import annotations

import random

from ..core.adapters import ToolSpec
from ..mcp.registry import CallEnvelope, MCPCallResult


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


class EchoTool:
    spec = ToolSpec(
        server="toy",
        name="echo",
        description="Echo a string back to the caller.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        text = args.get("text")
        if not isinstance(text, str):
            return MCPCallResult(ok=False, error="`text` must be a string")
        return MCPCallResult(ok=True, content={"text": text})


class AddTool:
    spec = ToolSpec(
        server="toy",
        name="add",
        description="Return the sum of two numbers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        a, b = args.get("a"), args.get("b")
        if not _is_number(a) or not _is_number(b):
            return MCPCallResult(ok=False, error="`a` and `b` must be numbers")
        return MCPCallResult(ok=True, content={"sum": a + b})


class RollDiceTool:
    spec = ToolSpec(
        server="toy",
        name="roll_dice",
        description="Roll `count` dice with `sides` faces each.",
        parameters={
            "type": "object",
            "properties": {
                "sides": {"type": "integer", "minimum": 2, "maximum": 1000},
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["sides"],
            "additionalProperties": False,
        },
    )

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        sides = args.get("sides")
        count = args.get("count", 1)
        if not _is_int(sides) or not 2 <= sides <= 1000:
            return MCPCallResult(ok=False, error="`sides` must be an integer in [2, 1000]")
        if not _is_int(count) or not 1 <= count <= 100:
            return MCPCallResult(ok=False, error="`count` must be an integer in [1, 100]")
        rolls = [random.randint(1, sides) for _ in range(count)]
        return MCPCallResult(ok=True, content={"rolls": rolls, "total": sum(rolls)})


TOY_TOOLS = (EchoTool(), AddTool(), RollDiceTool())
