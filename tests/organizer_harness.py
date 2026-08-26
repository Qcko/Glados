"""One Organizer, built the way unit tests need it.

Three files were each standing up the same seven-argument Organizer -- a
sink-capturing `send`, a single desk binding, an empty registry, a temp
TraceStore -- and diverging in small ways as they went. The wiring is not what
any of them is testing, so it lives here once and the tests say only what they
vary.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel

from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry

CLIENT_ID = "desk-ui"
ROOM_ID = "desk"

DESK_BINDING = ClientBinding(
    client_id=CLIENT_ID, room_id=ROOM_ID, role="ui", default_user="qcko"
)


@dataclass
class Harness:
    """The organizer under test plus every frame it sent."""

    org: Organizer
    sink: list[tuple[str, dict]] = field(default_factory=list)

    def messages(self, kind: str, *more: str) -> list[dict]:
        return [m for _, m in self.sink if m.get("type") in (kind, *more)]

    def session_id(self) -> str:
        return next(iter(self.messages("welcome")))["session_id"]


@asynccontextmanager
async def desk_organizer(
    tmp: Path, *, llm: Any, mcp: MCPRegistry | None = None, **kwargs: Any
) -> AsyncIterator[Harness]:
    """An Organizer bound to the single desk client, closed on the way out.

    Single-binding by construction: tests needing several clients or rooms
    build their own, and this grows a `bindings` parameter the day one of
    them is migrated rather than in advance of it.

    `kwargs` goes straight to the constructor, so a test needing a
    `tool_router` or a `specialist_llm` names just that.
    """
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    org = Organizer(
        llm=llm,
        mcp=mcp if mcp is not None else MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=lambda cid: DESK_BINDING if cid == CLIENT_ID else None,
        clients_in_room=lambda rid: [CLIENT_ID] if rid == ROOM_ID else [],
        **kwargs,
    )
    try:
        yield Harness(org=org, sink=sink)
    finally:
        await org.close()


def trace_events(tmp: Path) -> list[dict]:
    """Every event the TraceStore wrote under `tmp`, in file order."""
    out: list[dict] = []
    for path in sorted(tmp.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            out.append(json.loads(line))
    return out
