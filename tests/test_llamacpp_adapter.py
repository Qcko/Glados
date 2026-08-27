"""Unit tests for LlamaCppLLM against a mocked SSE transport.

The fixtures here are hand-built, which is a real limitation: they encode what
we BELIEVE llama-server's delta shape is, and that belief is the untested
thing. `test_llamacpp_integration.py` (skipped when llama-server is unreachable)
is what checks the belief; these check the assembly logic around it.

Bias of the cases: the interesting failures are all in ACCUMULATION, because
arguments arrive as string fragments keyed by `index` rather than as a whole
dict. A truncated or interleaved assembly is routine on a fragment stream where
it is near-impossible on Ollama's, so "refuses rather than dispatches" is the
property most of these pin down.
"""

from __future__ import annotations

import json

import httpx
import pytest

from glados.brain.llm.llamacpp import LlamaCppLLM
from glados.core.adapters import (
    LLMMessage,
    LLMText,
    LLMThinking,
    LLMToolCall,
    ToolSpec,
)


def _sse(*chunks: dict) -> bytes:
    lines = []
    for c in chunks:
        lines.append(f"data: {json.dumps(c)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _delta(delta: dict, *, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _mock_transport(body: bytes, *, captured: list[dict] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(request.content))
        return httpx.Response(200, content=body)

    return httpx.MockTransport(handler)


def _cart_spec() -> ToolSpec:
    return ToolSpec(
        server="shop",
        name="add_to_cart",
        description="add an item",
        parameters={"type": "object"},
        mutating=True,
    )


def _llm(body: bytes, *, captured: list[dict] | None = None) -> LlamaCppLLM:
    return LlamaCppLLM(
        model="ministral",
        max_tokens=4096,
        repeat_penalty=1.1,
        transport=_mock_transport(body, captured=captured),
    )


async def _collect(llm: LlamaCppLLM, tools: list[ToolSpec] | None = None) -> list:
    return [
        e
        async for e in llm.chat([LLMMessage(role="user", content="hi")], tools or [])
    ]


async def test_text_streams_through() -> None:
    body = _sse(_delta({"content": "Hello "}), _delta({"content": "there"}))
    events = await _collect(_llm(body))
    assert [e.text for e in events if isinstance(e, LLMText)] == ["Hello ", "there"]


async def test_reasoning_goes_to_the_thinking_channel() -> None:
    """Speaking reasoning aloud is the failure LLMThinking exists to stop."""
    body = _sse(_delta({"reasoning_content": "hmm"}), _delta({"content": "hi"}))
    events = await _collect(_llm(body))
    assert isinstance(events[0], LLMThinking)
    assert isinstance(events[1], LLMText)


async def test_tool_call_assembles_from_fragments() -> None:
    body = _sse(
        _delta({"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "shop__add_to_cart"}}]}),
        _delta({"tool_calls": [{"index": 0, "function": {"arguments": '{"q": '}}]}),
        _delta({"tool_calls": [{"index": 0, "function": {"arguments": '"milk"}'}}]}, finish="tool_calls"),
    )
    events = await _collect(_llm(body), [_cart_spec()])
    call = next(e for e in events if isinstance(e, LLMToolCall))
    assert (call.server, call.name, call.args) == ("shop", "add_to_cart", {"q": "milk"})
    assert call.call_id == "call_a"


async def test_from_text_is_false_for_a_structured_call() -> None:
    """The field means "recovered from assistant text". Faking it to trip the
    older confirmation arm would make every trace and log line lie; the
    untrusted-context gate is what covers this path instead."""
    body = _sse(
        _delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "shop__add_to_cart", "arguments": "{}"}}]}, finish="tool_calls"),
    )
    events = await _collect(_llm(body), [_cart_spec()])
    call = next(e for e in events if isinstance(e, LLMToolCall))
    assert call.from_text is False


async def test_content_then_call_yields_both() -> None:
    """llama.cpp accepts content-then-calls by design -- this is exactly what
    rule 2 refused on the text path, and the reason it had to be replaced
    rather than simply dropped."""
    body = _sse(
        _delta({"content": "Sure, adding that. "}),
        _delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "shop__add_to_cart", "arguments": "{}"}}]}, finish="tool_calls"),
    )
    events = await _collect(_llm(body), [_cart_spec()])
    assert any(isinstance(e, LLMText) for e in events)
    assert any(isinstance(e, LLMToolCall) for e in events)


async def test_parallel_calls_emit_in_index_order() -> None:
    """Fragments interleave; arrival order is not call order."""
    spec_b = ToolSpec(
        server="shop", name="view_cart", description="view", parameters={"type": "object"}
    )
    body = _sse(
        _delta({"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "shop__view_cart", "arguments": "{}"}}]}),
        _delta({"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "shop__add_to_cart"}}]}),
        _delta({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, finish="tool_calls"),
    )
    events = await _collect(_llm(body), [_cart_spec(), spec_b])
    calls = [e for e in events if isinstance(e, LLMToolCall)]
    assert [c.name for c in calls] == ["add_to_cart", "view_cart"]


async def test_unoffered_name_routes_to_unknown() -> None:
    """The offered-tools allowlist. MCPRegistry answers `unknown` with "unknown
    tool", so the model sees an error instead of reaching a tool it was never
    handed."""
    body = _sse(
        _delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "shop__drop_database", "arguments": "{}"}}]}, finish="tool_calls"),
    )
    events = await _collect(_llm(body), [_cart_spec()])
    call = next(e for e in events if isinstance(e, LLMToolCall))
    assert call.server == "unknown"


async def test_truncated_arguments_are_refused_not_emptied() -> None:
    """The important one. Dispatching a MUTATING tool with `args={}` because the
    JSON did not parse is the worst available reading of a truncated stream."""
    body = _sse(
        _delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "shop__add_to_cart"}}]}),
        _delta({"tool_calls": [{"index": 0, "function": {"arguments": '{"q": "mi'}}]}),
    )
    events = await _collect(_llm(body), [_cart_spec()])
    call = next(e for e in events if isinstance(e, LLMToolCall))
    assert call.server == "unknown"
    assert call.args == {}


async def test_non_object_arguments_are_refused() -> None:
    body = _sse(
        _delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "shop__add_to_cart", "arguments": '"milk"'}}]}, finish="tool_calls"),
    )
    events = await _collect(_llm(body), [_cart_spec()])
    call = next(e for e in events if isinstance(e, LLMToolCall))
    assert call.server == "unknown"


async def test_call_with_no_name_is_dropped() -> None:
    """A stream that died before the name arrived has nothing dispatchable in
    it -- emitting a half-built call would dispatch arguments at nothing."""
    body = _sse(
        _delta({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
    )
    events = await _collect(_llm(body), [_cart_spec()])
    assert not [e for e in events if isinstance(e, LLMToolCall)]


async def test_calls_are_capped_per_turn() -> None:
    """The same amplification cap tool_text.py applies: one echoed injection
    should not fan out into an unbounded run of mutating calls."""
    chunks = [
        _delta(
            {
                "tool_calls": [
                    {
                        "index": i,
                        "id": f"c{i}",
                        "function": {"name": "shop__add_to_cart", "arguments": "{}"},
                    }
                ]
            }
        )
        for i in range(10)
    ]
    events = await _collect(_llm(_sse(*chunks)), [_cart_spec()])
    assert len([e for e in events if isinstance(e, LLMToolCall)]) == 3


async def test_samplers_are_sent_explicitly() -> None:
    """An omitted sampler belongs to the RUNTIME, and this adapter exists to
    make a runtime comparison. llama-server's --n-predict defaults to unbounded
    where Ollama ran under num_predict."""
    captured: list[dict] = []
    await _collect(_llm(_sse(_delta({"content": "hi"})), captured=captured))
    sent = captured[0]
    assert sent["max_tokens"] == 4096
    assert sent["repeat_penalty"] == 1.1
    assert sent["temperature"] == 0.0
    assert sent["stream_options"] == {"include_usage": True}


async def test_sse_noise_lines_are_ignored() -> None:
    """aiter_lines gives lines, not events: comments, blanks and [DONE] all
    arrive here and none of them are JSON."""
    body = b': keep-alive\n\ndata: {"choices": [{"delta": {"content": "hi"}}]}\n\n\ndata: [DONE]\n\n'
    events = await _collect(_llm(body))
    assert [e.text for e in events if isinstance(e, LLMText)] == ["hi"]


async def test_unknown_call_replays_into_history_without_raising() -> None:
    """An `unknown` call carries the raw wire name, which contains the reserved
    `__` by construction. Sanitising it on the way back into history would raise
    on attacker-chosen bytes and kill the turn."""
    msg = LLMMessage(
        role="assistant",
        content="",
        tool_calls=[
            LLMToolCall(
                call_id="c", server="unknown", name="shop__drop_database", args={}
            )
        ],
    )
    wire = LlamaCppLLM._to_wire_msg(msg)
    assert wire["tool_calls"][0]["function"]["name"] == "shop__drop_database"
