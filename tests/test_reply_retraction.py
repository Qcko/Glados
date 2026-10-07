"""A re-driven or scrubbed turn retracts the reply it already streamed.

Prod 07-10-2026: "book some slot for Sunday" streamed "Slot booked" twice --
once from the primary, once from a specialist that aliased it."""

from __future__ import annotations

from pathlib import Path

import pytest

from glados.brain.router import Router
from glados.core.adapters import LLMText, LLMToolCall
from glados.core.config import ClientBinding
from glados.core.organizer import _CONFABULATION_REPLIES
from tests.test_router import ScriptedLLM, _organizer
from tests.test_step2 import _FakeTool, _make_organizer

_DESK = [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")]


def _frames(sink: list[tuple[str, dict]]) -> list[tuple[str, str]]:
    return [
        (m["type"], m.get("text") or m.get("target") or "")
        for _, m in sink
        if m["type"] in ("assistant_delta", "reply_retracted", "route_notice")
    ]


@pytest.mark.asyncio
async def test_escalation_retracts_the_primary_reply_first(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        specialist_llm=ScriptedLLM("specialist"),
        primary=ScriptedLLM("primary", fail=True),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()

    assert _frames(sink) == [
        ("route_notice", "primary"),
        ("assistant_delta", "primary reply"),
        ("reply_retracted", ""),
        ("route_notice", "specialist"),
        ("assistant_delta", "specialist reply"),
    ]


@pytest.mark.asyncio
async def test_a_specialist_aliasing_the_primary_never_escalates(tmp_path: Path) -> None:
    same = ScriptedLLM("primary", fail=True)
    async with _organizer(
        router=Router(), specialist_llm=same, primary=same, tmp_path=tmp_path
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()

    assert [m["target"] for _, m in sink if m["type"] == "route_notice"] == ["primary"]
    assert not [m for _, m in sink if m["type"] == "reply_retracted"]
    outcome = next(m for _, m in sink if m["type"] == "turn_outcome")
    assert outcome["outcome"] == "failed"


@pytest.mark.asyncio
async def test_nothing_is_retracted_when_nothing_streamed(tmp_path: Path) -> None:
    class SilentFailLLM:
        async def chat(self, messages, tools):
            if messages[-1].role != "tool":
                yield LLMToolCall(call_id="x", server="nope", name="nope", args={})

    async with _organizer(
        router=Router(),
        specialist_llm=ScriptedLLM("specialist"),
        primary=SilentFailLLM(),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()

    assert not [m for _, m in sink if m["type"] == "reply_retracted"]


class _ClaimsThenDoesItLLM:
    def __init__(self, *, comply: bool) -> None:
        self.passes = 0
        self._comply = comply

    async def chat(self, messages, tools):
        self.passes += 1
        if self.passes in (1, 3):
            yield LLMToolCall(call_id=f"c{self.passes}", server="dunnes",
                              name="view_cart", args={})
        elif self.passes == 2 or not self._comply:
            yield LLMText(text="Milk removed from cart.")
        elif self.passes == 4:
            yield LLMToolCall(call_id="c4", server="dunnes",
                              name="remove_from_cart_by_name", args={"name": "milk"})
        else:
            yield LLMText(text="Removed the milk from your cart.")


async def _run_claims_then(tmp_path: Path, *, comply: bool) -> list[tuple[str, str]]:
    async with _make_organizer(
        _DESK, tmp_path, llm=_ClaimsThenDoesItLLM(comply=comply),
        extra_tools=[
            _FakeTool("dunnes", "view_cart"),
            _FakeTool("dunnes", "remove_from_cart_by_name", mutating=True),
        ],
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "show me my cart and then remove the milk")
        await org.flush()
    return _frames(sink)


@pytest.mark.asyncio
async def test_a_confabulation_retry_retracts_the_false_claim(tmp_path: Path) -> None:
    frames = await _run_claims_then(tmp_path, comply=True)
    assert frames == [
        ("assistant_delta", "Milk removed from cart."),
        ("reply_retracted", ""),
        ("assistant_delta", "Removed the milk from your cart."),
    ]


@pytest.mark.asyncio
async def test_a_scrubbed_retry_is_retracted_before_the_canned_line(tmp_path: Path) -> None:
    frames = await _run_claims_then(tmp_path, comply=False)
    assert frames[:4] == [
        ("assistant_delta", "Milk removed from cart."),
        ("reply_retracted", ""),
        ("assistant_delta", "Milk removed from cart."),
        ("reply_retracted", ""),
    ]
    assert frames[4][1] in _CONFABULATION_REPLIES
