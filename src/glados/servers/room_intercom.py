"""The `room.speak_into` declaration -- and nothing that performs it.

A tool whose effect is SOUND in another room is a server-side egress
capability, not a request/response tool (ARCHITECTURE.md section 13). So the
spec is registered -- that is the only way the model can see the capability,
namespace it, and have it count against the registry cap -- while the effect
stays in the Organizer, which is where egress routing already lives. See
`DESIGN-cross-room-delivery.md`.

The flags below are deliberately set in code. Every other mutating tool gets
`mutating` / `requires_confirmation` from the per-machine `servers.toml`
overlay, which cannot reach an in-process spec: `ServerEntry.apply_flags` runs
only on the stdio registration path. A capability that can put attacker-chosen
audio into a room nobody is watching must not have its gate live in a file that
may not exist on the next machine.
"""

from __future__ import annotations

from typing import Sequence

from ..core.adapters import ToolSpec
from ..mcp.registry import CallEnvelope, MCPCallResult

SPEAK_INTO = "room.speak_into"

# The spoken message is model-authored from an utterance that may itself have
# been shaped by untrusted bytes, and a long one holds the target room's
# speaker for minutes. Enforced in the Organizer as well: a JSON-schema
# maxLength is a hint to the model, not a boundary.
MAX_MESSAGE_CHARS = 200


class SpeakIntoTool:
    """Spec holder. `call` exists only to satisfy the `Tool` protocol and to
    fail loudly if the Organizer's interception is ever bypassed -- reaching a
    dispatch here would mean the capability had been demoted to an ordinary
    tool, and silence would be the wrong answer to that."""

    def __init__(self, rooms: Sequence[str]) -> None:
        self.spec = _spec(rooms)

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        return MCPCallResult(
            ok=False,
            error=(
                f"{SPEAK_INTO} is answered by the Organizer and must never be "
                "dispatched; nothing was spoken"
            ),
        )


def _spec(rooms: Sequence[str]) -> ToolSpec:
    return ToolSpec(
        server="room",
        name="speak_into",
        description=(
            "Speak a short message aloud on the speaker in ANOTHER room of the "
            "house, for passing a message to someone who is not with you. The "
            "message is handed to that room, not spoken immediately: say that "
            "you have passed it on, never that they have heard it. You cannot "
            "hear any reply."
        ),
        parameters={
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "description": "The room to speak into.",
                    # Pinned so the model cannot invent a room. Derived from
                    # configured rooms rather than connected clients, so a
                    # speaker that is briefly offline is still addressable and
                    # fails as a runtime refusal instead of an unknown name.
                    "enum": list(rooms),
                },
                "message": {
                    "type": "string",
                    "description": "What to say. One short sentence.",
                    "maxLength": MAX_MESSAGE_CHARS,
                },
            },
            "required": ["room", "message"],
            "additionalProperties": False,
        },
        mutating=True,
        requires_confirmation=True,
    )
