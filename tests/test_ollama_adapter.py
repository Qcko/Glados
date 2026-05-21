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
        model="qwen2.5:7b-instruct",
        transport=_mock_transport(body, captured=captured),
    )
    msgs = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
    ]
    await _collect(adapter, msgs, [_now_spec()])

    payload = captured[0]
    assert payload["model"] == "qwen2.5:7b-instruct"
    assert payload["stream"] is True
    assert payload["tools"][0]["function"]["name"] == "time__now"
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
