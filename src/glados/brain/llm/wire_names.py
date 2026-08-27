"""Tool-name sanitisation shared by every adapter that speaks a function-calling
wire format.

Our tool identity is a `server.name` pair; the wire wants one flat string. The
mapping has to be INJECTIVE or an attacker-supplied name could collide with a
real one, so `__` is reserved as the separator and refused inside either half.

The reverse direction is the security-relevant one. `build_name_map` is built
from the tools OFFERED THIS TURN, and `resolve` routes anything not in it to
`server="unknown"` -- which `MCPRegistry` answers with "unknown tool". That is
the offered-tools allowlist, and it is the ONLY place it lives: `spec_for`
consults the full registry rather than the per-turn offered set, so an adapter
that resolves names any other way silently widens what the model can reach.
"""

from __future__ import annotations

from ...core.adapters import ToolSpec

UNKNOWN_SERVER = "unknown"


def sanitise_pair(server: str, name: str) -> str:
    if "__" in server or "__" in name:
        raise ValueError(
            f"server/tool names must not contain '__' (reserved separator): {server}.{name}"
        )
    return f"{server}__{name}"


def sanitise(spec: ToolSpec) -> str:
    return sanitise_pair(spec.server, spec.name)


def build_name_map(tools: list[ToolSpec]) -> dict[str, ToolSpec]:
    return {sanitise(t): t for t in tools}


def resolve(
    wire_name: str, name_map: dict[str, ToolSpec]
) -> tuple[str, str]:
    """Wire name -> (server, name). An unoffered name becomes the `unknown`
    sentinel rather than an exception: the model then sees the registry's error
    and can correct itself, and the dropped name lands in the trace."""
    spec = name_map.get(wire_name)
    if spec is None:
        return UNKNOWN_SERVER, wire_name or "unnamed"
    return spec.server, spec.name
