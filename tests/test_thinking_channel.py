"""Reasoning tokens are recorded, never spoken.

Reasoning models emit `thinking` alongside their answer. Two things must hold
at the organizer seam, and they pull in opposite directions: the reasoning must
NOT reach the spoken channel (it would be read aloud), and it must NOT be
silently dropped either -- reasoning that consumes the whole `num_predict`
budget is what starves the reply, and a turn whose most expensive part leaves
no record cannot be diagnosed (2026-08-25)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glados.core.adapters import LLMText, LLMThinking

from .organizer_harness import CLIENT_ID, desk_organizer, trace_events

REASONING = "The user asked about sales. Let me read the tool result carefully."
SPOKEN = "Onions are on sale."


class _ThinkingLLM:
    async def chat(self, messages, tools):
        yield LLMThinking(text=REASONING)
        yield LLMText(text=SPOKEN)


class _PlainLLM:
    async def chat(self, messages, tools):
        yield LLMText(text=SPOKEN)


async def _run(tmp_path: Path, llm) -> list[tuple[str, dict]]:
    async with desk_organizer(tmp_path, llm=llm) as h:
        await h.org.handle_user_text(CLIENT_ID, "what's on sale")
        await h.org.flush()
        return h.sink


@pytest.mark.asyncio
async def test_reasoning_never_reaches_the_spoken_channel(tmp_path: Path) -> None:
    sink = await _run(tmp_path, _ThinkingLLM())
    deltas = [m["text"] for _, m in sink if m.get("type") == "assistant_delta"]
    assert deltas == [SPOKEN]
    assert not any(REASONING in text for text in deltas)


@pytest.mark.asyncio
async def test_reasoning_never_reaches_tts(tmp_path: Path) -> None:
    # The delta assertion above is the contract; this one guards the surface it
    # exists to protect, so a future change that speaks reasoning some other way
    # still fails a test.
    sink = await _run(tmp_path, _ThinkingLLM())
    spoken_payloads = [
        json.dumps(m) for _, m in sink if m.get("type") in {"tts_chunk", "assistant_delta"}
    ]
    assert not any(REASONING in payload for payload in spoken_payloads)


@pytest.mark.asyncio
async def test_reasoning_is_recorded_in_the_trace(tmp_path: Path) -> None:
    await _run(tmp_path, _ThinkingLLM())
    thinking = [
        e for e in trace_events(tmp_path) if e.get("event") == "assistant_thinking"
    ]
    assert len(thinking) == 1
    assert thinking[0]["text"] == REASONING
    assert thinking[0]["chars"] == len(REASONING)


@pytest.mark.asyncio
async def test_no_thinking_event_when_the_model_did_not_reason(tmp_path: Path) -> None:
    # A non-reasoning model must not litter every trace with an empty event.
    await _run(tmp_path, _PlainLLM())
    assert not any(
        e.get("event") == "assistant_thinking" for e in trace_events(tmp_path)
    )
