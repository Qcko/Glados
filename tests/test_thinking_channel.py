"""Reasoning tokens are recorded, never spoken.

Reasoning models emit `thinking` alongside their answer. Two things must hold
at the organizer seam, and they pull in opposite directions: the reasoning must
NOT reach the spoken channel (it would be read aloud), and it must NOT be
silently dropped either -- reasoning that consumes the whole `num_predict`
budget is what starves the reply, and a turn whose most expensive part leaves
no record cannot be diagnosed (2026-08-25)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMText, LLMThinking
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry

REASONING = "The user asked about sales. Let me read the tool result carefully."
SPOKEN = "Onions are on sale."


class _ThinkingLLM:
    async def chat(self, messages, tools):
        yield LLMThinking(text=REASONING)
        yield LLMText(text=SPOKEN)


@asynccontextmanager
async def _make(tmp: Path):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    binding = ClientBinding(
        client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
    )
    org = Organizer(
        llm=_ThinkingLLM(),
        mcp=MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={"desk-ui": binding}.get,
        clients_in_room=lambda r: ["desk-ui"] if r == "desk" else [],
    )
    try:
        yield org, sink
    finally:
        await org.close()


async def _run(tmp_path: Path):
    async with _make(tmp_path) as (org, sink):
        await org.handle_user_text("desk-ui", "what's on sale")
        await org.flush()
    events = []
    for path in tmp_path.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return sink, events


@pytest.mark.asyncio
async def test_reasoning_never_reaches_the_spoken_channel(tmp_path: Path) -> None:
    sink, _ = await _run(tmp_path)
    deltas = [m["text"] for _, m in sink if m.get("type") == "assistant_delta"]
    assert deltas == [SPOKEN]
    assert not any(REASONING in text for text in deltas)


@pytest.mark.asyncio
async def test_reasoning_never_reaches_tts(tmp_path: Path) -> None:
    # The delta assertion above is the contract; this one guards the surface it
    # exists to protect, so a future change that speaks reasoning some other way
    # still fails a test.
    sink, _ = await _run(tmp_path)
    spoken_payloads = [
        json.dumps(m) for _, m in sink if m.get("type") in {"tts_chunk", "assistant_delta"}
    ]
    assert not any(REASONING in payload for payload in spoken_payloads)


@pytest.mark.asyncio
async def test_reasoning_is_recorded_in_the_trace(tmp_path: Path) -> None:
    _, events = await _run(tmp_path)
    thinking = [e for e in events if e.get("event") == "assistant_thinking"]
    assert len(thinking) == 1
    assert thinking[0]["text"] == REASONING
    assert thinking[0]["chars"] == len(REASONING)


@pytest.mark.asyncio
async def test_no_thinking_event_when_the_model_did_not_reason(tmp_path: Path) -> None:
    # A non-reasoning model must not litter every trace with an empty event.
    class _PlainLLM:
        async def chat(self, messages, tools):
            yield LLMText(text=SPOKEN)

    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    binding = ClientBinding(
        client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
    )
    org = Organizer(
        llm=_PlainLLM(),
        mcp=MCPRegistry(),
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={"desk-ui": binding}.get,
        clients_in_room=lambda r: ["desk-ui"] if r == "desk" else [],
    )
    try:
        await org.handle_user_text("desk-ui", "what's on sale")
        await org.flush()
    finally:
        await org.close()

    for path in tmp_path.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert json.loads(line).get("event") != "assistant_thinking"
