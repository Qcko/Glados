"""A barge-in must not erase the reasoning the turn had already produced.

Reasoning is aggregated per LLM pass and written to the trace once the stream
finishes. That left the interrupted turn -- the single case where "where did
the budget go?" is the question actually being asked -- as the one turn with no
record of it, because cancellation propagates out of the streaming loop and
skips anything sitting after the block.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMText, LLMThinking
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry

REASONING = "Let me work through what the user actually wants here."


class _ThinksThenHangs:
    """Reasons, then stalls mid-stream so the turn can be interrupted while
    the reasoning is collected but not yet flushed to the trace."""

    def __init__(self) -> None:
        self.streaming = asyncio.Event()

    async def chat(self, messages, tools):
        yield LLMThinking(text=REASONING)
        self.streaming.set()
        await asyncio.sleep(3600)
        yield LLMText(text="never reached")


def _events(tmp: Path) -> list[dict]:
    out: list[dict] = []
    for path in tmp.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            out.append(json.loads(line))
    return out


@pytest.mark.asyncio
async def test_reasoning_is_traced_even_when_the_turn_is_cut_off(
    tmp_path: Path,
) -> None:
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    binding = ClientBinding(
        client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
    )
    llm = _ThinksThenHangs()
    org = Organizer(
        llm=llm,
        mcp=MCPRegistry(),
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={"desk-ui": binding}.get,
        clients_in_room=lambda r: ["desk-ui"] if r == "desk" else [],
    )
    try:
        await org.handle_user_text("desk-ui", "what's on sale")
        await asyncio.wait_for(llm.streaming.wait(), timeout=5)
        session_id = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        await org.handle_interrupt("desk-ui", session_id)
        # Bounded because the fake LLM parks for an hour: if a regression stops
        # the interrupt from cancelling, this must fail red rather than wedge
        # the runner until the sleep expires.
        await asyncio.wait_for(org.flush(), timeout=5)
    finally:
        await org.close()

    assert any(m["type"] == "cancelled" for _, m in sink), "the turn was not cut off"
    thinking = [e for e in _events(tmp_path) if e.get("event") == "assistant_thinking"]
    assert thinking, "the interrupted turn left no record of its reasoning"
    # Exactly one: the turn made a single LLM pass, and the aggregation is per
    # pass -- so a second event here would mean a double-write, not more detail.
    assert len(thinking) == 1
    assert thinking[0]["text"] == REASONING
    assert thinking[0]["chars"] == len(REASONING)
