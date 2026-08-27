"""Streaming behaviour when tool calls arrive as TEXT.

The adapter has to withhold text that might still become a `[TOOL_CALLS]`
marker without ever losing, duplicating, or reordering what it withheld. These
drive real chunk boundaries through the transport, because the interesting
failures are all splits: a marker torn across two chunks, content riding in on
the `done` chunk, a reply that only turns out to be ordinary speech late.
"""

from __future__ import annotations

import json

import httpx
import pytest

from glados.brain.llm.ollama import OllamaLLM
from glados.core.adapters import LLMMessage, LLMText, LLMThinking, LLMToolCall, ToolSpec

TOOL = ToolSpec(
    server="dunnes",
    name="view_cart",
    description="Show the cart.",
    parameters={"type": "object", "properties": {}},
)
WIRE = "dunnes__view_cart"


def _chunks(*texts: str, tool_calls: list[dict] | None = None) -> list[bytes]:
    """One NDJSON line per text fragment, then a terminal `done` line."""
    lines = [
        json.dumps({"message": {"content": t}, "done": False}).encode() for t in texts
    ]
    final: dict = {"message": {"content": ""}, "done": True, "done_reason": "stop"}
    if tool_calls:
        final["message"]["tool_calls"] = tool_calls
    final["prompt_eval_count"] = 10
    final["eval_count"] = 5
    lines.append(json.dumps(final).encode())
    return lines


def _llm(lines: list[bytes], **kw) -> OllamaLLM:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\n".join(lines))

    return OllamaLLM(
        model="ministral3:test",
        text_tool_format="mistral_v13",
        transport=httpx.MockTransport(handler),
        **kw,
    )


async def _run(llm: OllamaLLM) -> list:
    events = []
    async for ev in llm.chat([LLMMessage(role="user", content="hi")], [TOOL]):
        events.append(ev)
    await llm.aclose()
    return events


def _spoken(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, LLMText))


def _calls(events: list) -> list[tuple[str, str]]:
    return [(e.server, e.name) for e in events if isinstance(e, LLMToolCall)]


async def test_ordinary_speech_is_not_swallowed():
    events = await _run(_llm(_chunks("The capital ", "of France ", "is Paris.")))
    assert _spoken(events) == "The capital of France is Paris."
    assert _calls(events) == []


async def test_marker_split_across_chunks_still_dispatches():
    # The split that defeats a naive per-chunk check.
    events = await _run(_llm(_chunks("[TOOL", "_CALLS]", WIRE, "[ARGS]", "{}")))
    assert _calls(events) == [("dunnes", "view_cart")]
    assert _spoken(events) == ""


async def test_call_arriving_whole_in_one_chunk():
    events = await _run(_llm(_chunks(f"[TOOL_CALLS]{WIRE}[ARGS]{{}}")))
    assert _calls(events) == [("dunnes", "view_cart")]


async def test_text_that_only_looks_like_a_marker_is_released():
    # Starts with "[" but settles into prose -- must not be held forever.
    events = await _run(_llm(_chunks("[not", " a marker] hello")))
    assert _spoken(events) == "[not a marker] hello"


async def test_content_on_the_done_chunk_is_not_lost():
    lines = [
        json.dumps({"message": {"content": "[TOOL_CALLS]"}, "done": False}).encode(),
        json.dumps(
            {
                "message": {"content": f"{WIRE}[ARGS]{{}}"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        ).encode(),
    ]
    events = await _run(_llm(lines))
    assert _calls(events) == [("dunnes", "view_cart")]


async def test_reasoning_never_reaches_the_spoken_channel():
    events = await _run(_llm(_chunks("[THINK]I should ", "check.[/THINK]", "Paris.")))
    assert _spoken(events) == "Paris."
    assert any(isinstance(e, LLMThinking) for e in events)
    assert "[THINK]" not in _spoken(events)


async def test_native_tool_call_still_works_alongside_held_text():
    lines = _chunks(
        "[TOOL", "_CALLS]" + WIRE + "[ARGS]{}",
        tool_calls=[{"function": {"name": WIRE, "arguments": {}}}],
    )
    events = await _run(_llm(lines))
    # Both paths fire; neither loses its call.
    assert len(_calls(events)) == 2


async def test_recovered_call_is_marked_as_text_parsed():
    events = await _run(_llm(_chunks(f"[TOOL_CALLS]{WIRE}[ARGS]{{}}")))
    call = next(e for e in events if isinstance(e, LLMToolCall))
    assert call.from_text is True


async def test_format_off_leaves_the_marker_as_plain_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\n".join(_chunks(f"[TOOL_CALLS]{WIRE}[ARGS]{{}}")))

    llm = OllamaLLM(
        model="qwen3:8b", text_tool_format=None,
        transport=httpx.MockTransport(handler),
    )
    events = await _run(llm)
    assert _calls(events) == []
    assert "[TOOL_CALLS]" in _spoken(events)
