"""v1 step 5 follow-up: verify OllamaLLM upstream cancellation actually
closes the httpx stream on the wire, not just locally.

Background: the Organizer wraps `llm.chat(...)` in `contextlib.aclosing` so
`CancelledError` propagates into the async generator. That `aclose()` must
in turn exit the `async with client.stream(...)` block in
[OllamaLLM.chat](../src/glados/brain/llm/ollama.py), which httpx implements
by calling `response.aclose()` — which closes the HTTP/1.1 connection so
Ollama stops generating. If that chain were broken (e.g. by a bare-except
or missing `async with`), the generator would close locally but Ollama
would keep streaming tokens we'll never read.

These tests inject a fake `httpx.AsyncBaseTransport` whose response body
records when `aclose()` is called. We run `OllamaLLM.chat` inside
`aclosing`, cancel mid-stream, and assert the body was closed.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing

import httpx
import pytest

from glados.brain.llm.ollama import OllamaLLM
from glados.core.adapters import LLMText


class _RecordingStream(httpx.AsyncByteStream):
    """Streams one chunk then hangs forever, recording aclose()."""

    def __init__(self, first_chunk: bytes) -> None:
        self._first = first_chunk
        self.aclose_calls = 0
        self.first_chunk_sent = asyncio.Event()

    async def __aiter__(self):
        yield self._first
        self.first_chunk_sent.set()
        # Hang until aclose() is called. The cancellation chain must reach
        # here for the test to pass.
        try:
            await asyncio.sleep(3600)
        except BaseException:
            raise
        yield b""  # unreachable

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, body: _RecordingStream) -> None:
        self._body = body

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/x-ndjson"},
            stream=self._body,
            request=request,
        )


def _ndjson_text_chunk(text: str) -> bytes:
    return (json.dumps({"message": {"content": text}, "done": False}) + "\n").encode()


@pytest.mark.asyncio
async def test_cancel_propagates_to_httpx_aclose() -> None:
    body = _RecordingStream(_ndjson_text_chunk("hello "))
    llm = OllamaLLM(transport=_RecordingTransport(body))

    received: list[str] = []

    async def consume() -> None:
        async with aclosing(llm.chat([], [])) as stream:
            async for event in stream:
                if isinstance(event, LLMText):
                    received.append(event.text)

    task = asyncio.create_task(consume())
    await body.first_chunk_sent.wait()
    assert received == ["hello "]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert body.aclose_calls >= 1, (
        "httpx response body was not aclose()'d on cancel — "
        "Ollama would keep generating tokens with nothing reading them"
    )
