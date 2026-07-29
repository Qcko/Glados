"""Anthropic cloud LLM adapter: message translation + SSE stream parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from glados.brain.llm.anthropic import AnthropicLLM
from glados.core.adapters import LLMMessage, LLMToolCall, ToolSpec


def test_translation_folds_system_and_batches_tool_results() -> None:
    messages = [
        LLMMessage(role="system", content="You are GLaDOS."),
        LLMMessage(role="user", content="add milk"),
        LLMMessage(
            role="assistant",
            content="on it",
            tool_calls=[
                LLMToolCall(call_id="c1", server="dunnes", name="add", args={"q": "milk"}),
                LLMToolCall(call_id="c2", server="dunnes", name="view", args={}),
            ],
        ),
        LLMMessage(role="tool", tool_call_id="c1", content="added"),
        LLMMessage(role="tool", tool_call_id="c2", content="cart has milk"),
    ]
    system, api = AnthropicLLM._to_anthropic_messages(messages)
    assert system == "You are GLaDOS."
    assert api[0] == {"role": "user", "content": "add milk"}
    # Assistant turn carries a text block + two tool_use blocks with sanitised names.
    assistant = api[1]
    assert assistant["role"] == "assistant"
    kinds = [b["type"] for b in assistant["content"]]
    assert kinds == ["text", "tool_use", "tool_use"]
    assert assistant["content"][1]["name"] == "dunnes__add"
    # Both tool results collapse into a single user turn (roles must alternate).
    assert api[2]["role"] == "user"
    assert [b["tool_use_id"] for b in api[2]["content"]] == ["c1", "c2"]


def _sse(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


@pytest.mark.asyncio
async def test_streaming_parses_text_and_tool_call() -> None:
    spec = ToolSpec(server="time", name="now", description="", parameters={})
    stream = _sse(
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "there"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "time__now"},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"tz":'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"utc"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "k"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["tools"][0]["name"] == "time__now"
        return httpx.Response(200, content=stream)

    llm = AnthropicLLM(api_key="k", transport=httpx.MockTransport(handler))
    events = [
        e async for e in llm.chat([LLMMessage(role="user", content="hi")], [spec])
    ]
    await llm.aclose()

    texts = [e.text for e in events if e.type == "text"]
    calls = [e for e in events if e.type == "tool_call"]
    assert "".join(texts) == "Hi there"
    assert len(calls) == 1
    assert calls[0].server == "time" and calls[0].name == "now"
    assert calls[0].args == {"tz": "utc"}


@pytest.mark.asyncio
async def test_midstream_error_event_raises() -> None:
    # Anthropic sends mid-stream failures as a 200-OK SSE `error` event, so
    # raise_for_status passes -- the adapter must surface it, not end silently.
    stream = _sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}},
    )
    llm = AnthropicLLM(
        api_key="k",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=stream)),
    )
    with pytest.raises(RuntimeError, match="overloaded"):
        async for _ in llm.chat([LLMMessage(role="user", content="x")], []):
            pass
    await llm.aclose()


@pytest.mark.asyncio
async def test_streaming_hallucinated_tool_name_surfaces_unknown() -> None:
    stream = _sse(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t1", "name": "made__up"},
        },
        {"type": "content_block_stop", "index": 0},
    )
    llm = AnthropicLLM(
        api_key="k",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=stream)),
    )
    events = [e async for e in llm.chat([LLMMessage(role="user", content="x")], [])]
    await llm.aclose()
    assert events[0].type == "tool_call"
    assert events[0].server == "unknown"
