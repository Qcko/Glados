"""Integration test against a live Ollama instance.

Skipped when Ollama isn't reachable or the model isn't pulled — so this stays
green in CI while still being runnable locally with one command.
"""

from __future__ import annotations

import os

import httpx
import pytest

from glados.brain.llm.ollama import OllamaLLM
from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, ToolSpec


HOST = os.environ.get("GLADOS_OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("GLADOS_OLLAMA_MODEL", "qwen2.5:7b-instruct")


def _ollama_has_model() -> bool:
    try:
        r = httpx.get(f"{HOST}/api/tags", timeout=2.0)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", [])]
        return any(name == MODEL or name.startswith(f"{MODEL}:") for name in models)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_has_model(),
    reason=f"Ollama not reachable or model {MODEL!r} not pulled",
)


@pytest.mark.asyncio
async def test_real_ollama_responds() -> None:
    adapter = OllamaLLM(host=HOST, model=MODEL, temperature=0.0, timeout=120.0)
    events = [
        e
        async for e in adapter.chat(
            [LLMMessage(role="user", content="Say the single word: pong")], []
        )
    ]
    text = "".join(e.text for e in events if isinstance(e, LLMText)).lower()
    assert "pong" in text


@pytest.mark.asyncio
async def test_real_ollama_calls_tool() -> None:
    spec = ToolSpec(
        server="time",
        name="now",
        description="Return the current local time. No arguments.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    adapter = OllamaLLM(host=HOST, model=MODEL, temperature=0.0, timeout=120.0)
    events = [
        e
        async for e in adapter.chat(
            [
                LLMMessage(
                    role="system",
                    content="You are a helpful assistant. Use tools when relevant.",
                ),
                LLMMessage(role="user", content="What time is it?"),
            ],
            [spec],
        )
    ]
    tool_calls = [e for e in events if isinstance(e, LLMToolCall)]
    assert tool_calls, f"expected a tool call, got: {events}"
    assert tool_calls[0].server == "time" and tool_calls[0].name == "now"
