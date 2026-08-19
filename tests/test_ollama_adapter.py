"""Unit tests for OllamaLLM with a mocked HTTP transport.

A separate integration test (skipped if Ollama isn't reachable) exercises the
real backend.
"""

from __future__ import annotations

import json

import httpx
import pytest

from glados.brain.llm.ollama import OllamaLLM
from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, ToolSpec


def _ndjson(*chunks: dict) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


def _mock_transport(body: bytes, *, captured: list[dict] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(request.content))
        return httpx.Response(200, content=body)

    return httpx.MockTransport(handler)


def _now_spec() -> ToolSpec:
    return ToolSpec(
        server="time",
        name="now",
        description="current time",
        parameters={"type": "object", "properties": {}},
    )


async def _collect(adapter: OllamaLLM, messages, tools):
    return [e async for e in adapter.chat(messages, tools)]


@pytest.mark.asyncio
async def test_streams_text_chunks() -> None:
    body = _ndjson(
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo."}, "done": True},
    )
    adapter = OllamaLLM(transport=_mock_transport(body))
    events = await _collect(
        adapter, [LLMMessage(role="user", content="hi")], []
    )
    assert [e.type for e in events] == ["text", "text"]
    assert "".join(e.text for e in events if isinstance(e, LLMText)) == "Hello."


@pytest.mark.asyncio
async def test_keep_alive_numeric_coerced_to_int() -> None:
    """A numeric keep_alive ("-1") must ride as an int -- Ollama rejects the
    bare-number STRING "-1" with 400 but accepts the int sentinel for resident-
    forever. A unit-duration string ("30m") passes through unchanged."""
    body = _ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True})
    captured: list[dict] = []
    adapter = OllamaLLM(
        keep_alive="-1", transport=_mock_transport(body, captured=captured)
    )
    await _collect(adapter, [LLMMessage(role="user", content="hi")], [])
    assert captured[0]["keep_alive"] == -1

    captured.clear()
    adapter = OllamaLLM(
        keep_alive="30m", transport=_mock_transport(body, captured=captured)
    )
    await _collect(adapter, [LLMMessage(role="user", content="hi")], [])
    assert captured[0]["keep_alive"] == "30m"


async def test_emits_tool_call() -> None:
    body = _ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "time__now", "arguments": {}}}
                ],
            },
            "done": True,
        }
    )
    adapter = OllamaLLM(transport=_mock_transport(body))
    events = await _collect(
        adapter, [LLMMessage(role="user", content="time?")], [_now_spec()]
    )
    tool_calls = [e for e in events if isinstance(e, LLMToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].server == "time" and tool_calls[0].name == "now"


@pytest.mark.asyncio
async def test_unknown_tool_surfaces_as_sentinel_call() -> None:
    body = _ndjson(
        {
            "message": {
                "tool_calls": [
                    {"function": {"name": "bogus_tool", "arguments": {}}}
                ]
            },
            "done": True,
        }
    )
    adapter = OllamaLLM(transport=_mock_transport(body))
    events = await _collect(
        adapter, [LLMMessage(role="user", content="?")], [_now_spec()]
    )
    tool_calls = [e for e in events if isinstance(e, LLMToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].server == "unknown"
    assert tool_calls[0].name == "bogus_tool"


@pytest.mark.asyncio
async def test_text_and_tool_call_in_same_chunk() -> None:
    body = _ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "let me check.",
                "tool_calls": [
                    {"function": {"name": "time__now", "arguments": {}}}
                ],
            },
            "done": True,
        }
    )
    adapter = OllamaLLM(transport=_mock_transport(body))
    events = await _collect(
        adapter, [LLMMessage(role="user", content="time?")], [_now_spec()]
    )
    assert [e.type for e in events] == ["text", "tool_call"]
    assert isinstance(events[1], LLMToolCall)
    assert (events[1].server, events[1].name) == ("time", "now")


@pytest.mark.asyncio
async def test_http_error_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"server boom")

    adapter = OllamaLLM(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await _collect(
            adapter, [LLMMessage(role="user", content="hi")], []
        )


@pytest.mark.asyncio
async def test_chunk_missing_message_is_ignored() -> None:
    body = (
        json.dumps({"done": False}) + "\n" + json.dumps({"message": {"content": "hi"}, "done": True}) + "\n"
    ).encode()
    adapter = OllamaLLM(transport=_mock_transport(body))
    events = await _collect(
        adapter, [LLMMessage(role="user", content="?")], []
    )
    assert [e.type for e in events] == ["text"]
    assert events[0].text == "hi"


def test_sanitise_pair_rejects_double_underscore() -> None:
    with pytest.raises(ValueError, match="reserved separator"):
        OllamaLLM._sanitise_pair("life__quests", "create")


@pytest.mark.asyncio
async def test_sanitised_name_uses_double_underscore() -> None:
    captured: list[dict] = []
    body = _ndjson({"message": {"content": "ok"}, "done": True})
    adapter = OllamaLLM(transport=_mock_transport(body, captured=captured))
    spec = ToolSpec(
        server="time_zone",
        name="now",
        description="",
        parameters={"type": "object"},
    )
    await _collect(adapter, [LLMMessage(role="user", content="hi")], [spec])
    assert captured[0]["tools"][0]["function"]["name"] == "time_zone__now"


@pytest.mark.asyncio
async def test_tool_args_string_decoded() -> None:
    body = _ndjson(
        {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "time__now",
                            "arguments": '{"foo": 1}',
                        }
                    }
                ]
            },
            "done": True,
        }
    )
    adapter = OllamaLLM(transport=_mock_transport(body))
    events = await _collect(
        adapter, [LLMMessage(role="user", content="?")], [_now_spec()]
    )
    assert events[0].args == {"foo": 1}


@pytest.mark.asyncio
async def test_client_reused_across_chat_calls() -> None:
    """Pooling: each chat() must reuse the same AsyncClient so httpx keep-alive
    sockets to Ollama survive across turns, not be torn down per call."""
    body = _ndjson({"message": {"content": "ok"}, "done": True})
    adapter = OllamaLLM(transport=_mock_transport(body))
    await _collect(adapter, [LLMMessage(role="user", content="a")], [])
    client_after_first = adapter._client
    await _collect(adapter, [LLMMessage(role="user", content="b")], [])
    assert adapter._client is client_after_first is not None
    await adapter.aclose()
    assert adapter._client is None


@pytest.mark.asyncio
async def test_aclose_is_idempotent() -> None:
    adapter = OllamaLLM(transport=_mock_transport(_ndjson()))
    await adapter.aclose()  # never used
    await adapter.aclose()  # second call must not raise


@pytest.mark.asyncio
async def test_request_payload_shape() -> None:
    captured: list[dict] = []
    body = _ndjson({"message": {"content": "ok"}, "done": True})
    adapter = OllamaLLM(
        model="qwen3:4b",
        transport=_mock_transport(body, captured=captured),
    )
    msgs = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
    ]
    await _collect(adapter, msgs, [_now_spec()])

    payload = captured[0]
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is True
    assert payload["tools"][0]["function"]["name"] == "time__now"
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


@pytest.mark.asyncio
async def test_sends_num_ctx_and_num_predict() -> None:
    """Both ride in `options`. Until 2026-08-17 only `temperature` was sent, so
    every request ran at Ollama's own small default and the prompt was truncated
    from the FRONT -- silently evicting the system prompt."""
    captured: list[dict] = []
    body = _ndjson({"message": {"content": "ok"}, "done": True})
    adapter = OllamaLLM(
        num_ctx=8192,
        num_predict=512,
        temperature=0.0,
        transport=_mock_transport(body, captured=captured),
    )
    await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    options = captured[0]["options"]
    assert options["num_ctx"] == 8192
    assert options["num_predict"] == 512
    assert options["temperature"] == 0.0


@pytest.mark.asyncio
async def test_none_context_options_are_omitted_not_nulled() -> None:
    """`None` must DROP the key, not send null -- Ollama rejects a null option,
    so the escape hatch back to the server's own default has to be an absent
    key. Guards the difference between "unset" and "explicitly nothing"."""
    captured: list[dict] = []
    body = _ndjson({"message": {"content": "ok"}, "done": True})
    adapter = OllamaLLM(
        num_ctx=None,
        num_predict=None,
        transport=_mock_transport(body, captured=captured),
    )
    await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    options = captured[0]["options"]
    assert "num_ctx" not in options
    assert "num_predict" not in options
    assert "temperature" in options


@pytest.mark.asyncio
async def test_warns_when_prompt_approaches_context_limit(caplog) -> None:
    """Front-truncation is invisible on the wire, so prompt pressure is the only
    signal that the system prompt is about to be evicted."""
    body = _ndjson(
        {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 7000}
    )
    adapter = OllamaLLM(num_ctx=8192, transport=_mock_transport(body))
    with caplog.at_level("WARNING"):
        await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    assert any("truncates from the FRONT" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_no_context_warning_when_prompt_is_small(caplog) -> None:
    body = _ndjson(
        {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 100}
    )
    adapter = OllamaLLM(num_ctx=8192, transport=_mock_transport(body))
    with caplog.at_level("WARNING"):
        await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    assert not [r for r in caplog.records if r.levelname == "WARNING"]


@pytest.mark.asyncio
async def test_no_context_warning_when_num_ctx_unset(caplog) -> None:
    """A real prompt count with num_ctx=None must neither warn nor raise. The
    guard this pins is one edit from `int > None`, which would TypeError inside
    the stream loop and kill the turn."""
    body = _ndjson(
        {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 7000}
    )
    adapter = OllamaLLM(num_ctx=None, transport=_mock_transport(body))
    with caplog.at_level("WARNING"):
        await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    assert not [r for r in caplog.records if r.levelname == "WARNING"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_tokens, expect_warning", [(6554, True), (6553, False)]
)
async def test_context_pressure_ratio_boundary(
    caplog, prompt_tokens: int, expect_warning: bool
) -> None:
    """Pin the boundary so _CONTEXT_PRESSURE_RATIO is load-bearing: 0.8 * 8192
    is 6553.6. Without this the constant could be retuned anywhere in a wide
    band with every test still green."""
    body = _ndjson(
        {
            "message": {"content": "ok"},
            "done": True,
            "prompt_eval_count": prompt_tokens,
        }
    )
    adapter = OllamaLLM(num_ctx=8192, transport=_mock_transport(body))
    with caplog.at_level("WARNING"):
        await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    warned = any("truncates from the FRONT" in r.message for r in caplog.records)
    assert warned is expect_warning


@pytest.mark.asyncio
async def test_context_pressure_warning_is_latched(caplog) -> None:
    """Pressure is a standing condition. Warning every turn on a large-tool-list
    workload is how a warning stops being read."""
    body = _ndjson(
        {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 7000}
    )
    adapter = OllamaLLM(num_ctx=8192, transport=_mock_transport(body))
    with caplog.at_level("WARNING"):
        for _ in range(3):
            await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    hits = [r for r in caplog.records if "truncates from the FRONT" in r.message]
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_warns_when_reply_hits_num_predict_cap(caplog) -> None:
    """done_reason="length" means the spoken reply was cut mid-sentence. Silent
    tail-truncation is the same class of bug as the silent front-truncation
    this slice exists to fix."""
    body = _ndjson(
        {
            "message": {"content": "ok"},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 100,
            "eval_count": 512,
        }
    )
    adapter = OllamaLLM(num_ctx=8192, num_predict=512, transport=_mock_transport(body))
    with caplog.at_level("WARNING"):
        await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    assert any("truncated at the num_predict" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_no_truncation_warning_on_normal_stop(caplog) -> None:
    body = _ndjson(
        {
            "message": {"content": "ok"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 40,
        }
    )
    adapter = OllamaLLM(num_ctx=8192, num_predict=512, transport=_mock_transport(body))
    with caplog.at_level("WARNING"):
        await _collect(adapter, [LLMMessage(role="user", content="hi")], [])

    assert not [r for r in caplog.records if r.levelname == "WARNING"]
