"""A barge-in must not erase the reasoning the turn had already produced.

Reasoning is aggregated per LLM pass and written to the trace once the stream
finishes. That left the interrupted turn -- the single case where "where did
the budget go?" is the question actually being asked -- as the one turn with no
record of it, because cancellation propagates out of the streaming loop and
skips anything sitting after the block.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from glados.core.adapters import LLMText, LLMThinking

from .organizer_harness import CLIENT_ID, desk_organizer, trace_events

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


@pytest.mark.asyncio
async def test_reasoning_is_traced_even_when_the_turn_is_cut_off(
    tmp_path: Path,
) -> None:
    llm = _ThinksThenHangs()
    async with desk_organizer(tmp_path, llm=llm) as h:
        await h.org.handle_user_text(CLIENT_ID, "what's on sale")
        await asyncio.wait_for(llm.streaming.wait(), timeout=5)
        await h.org.handle_interrupt(CLIENT_ID, h.session_id())
        # Bounded because the fake LLM parks for an hour: if a regression stops
        # the interrupt from cancelling, this must fail red rather than wedge
        # the runner until the sleep expires.
        await asyncio.wait_for(h.org.flush(), timeout=5)
        assert h.messages("cancelled"), "the turn was not cut off"

    thinking = [
        e for e in trace_events(tmp_path) if e.get("event") == "assistant_thinking"
    ]
    assert thinking, "the interrupted turn left no record of its reasoning"
    # Exactly one: the turn made a single LLM pass, and the aggregation is per
    # pass -- so a second event here would mean a double-write, not more detail.
    assert len(thinking) == 1
    assert thinking[0]["text"] == REASONING
    assert thinking[0]["chars"] == len(REASONING)
