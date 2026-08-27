"""Integration test against a live `llama-server`.

Skipped when it is unreachable, so this stays green in CI while remaining
runnable locally with one command. Launch it with
`<lab-root>/glados/ministral/serve_llamacpp.sh`.

This exists because the unit fixtures in `test_llamacpp_adapter.py` are
hand-built: they encode what we BELIEVE llama-server's delta shape is, and that
belief is precisely the untested thing. Only a live server can check it -- and
the second-pass case below is unreachable by any response fixture at all, since
the thing under test is how OUR request renders through the model's template.
"""

from __future__ import annotations

import os

import httpx
import pytest

from glados.brain.llm.llamacpp import LlamaCppLLM
from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, ToolSpec


HOST = os.environ.get("GLADOS_LLAMACPP_HOST", "http://127.0.0.1:9090")
API_KEY = os.environ.get("GLADOS_LLAMACPP_KEY")


def _reachable_and_authorised() -> bool:
    """Probe the way the tests actually talk, not just for a listening socket.

    `/health` answers WITHOUT the api key, so a reachability-only check reports
    "up" for a server that 401s every real request -- which is what the launch
    script produces, since it always sets `--api-key`. That made the whole file
    fail instead of skip whenever the key was absent from the environment.
    """
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    try:
        r = httpx.get(f"{HOST}/props", headers=headers, timeout=2.0)
    except Exception:
        return False
    return r.status_code == 200


pytestmark = pytest.mark.skipif(
    not _reachable_and_authorised(),
    reason=(
        f"no reachable+authorised llama-server at {HOST} "
        "(set GLADOS_LLAMACPP_KEY to match the running server)"
    ),
)


def _llm(**kw) -> LlamaCppLLM:
    return LlamaCppLLM(
        host=HOST, model="ministral", max_tokens=256, repeat_penalty=1.1,
        api_key=API_KEY, **kw
    )


def _cart_tool() -> ToolSpec:
    return ToolSpec(
        server="shop",
        name="add_to_cart_by_name",
        description="Add an item to the shopping cart by name",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        mutating=True,
    )


async def test_live_text_turn() -> None:
    llm = _llm()
    try:
        text = "".join(
            [
                e.text
                async for e in llm.chat(
                    [
                        LLMMessage(
                            role="user", content="Say hello in one short sentence."
                        )
                    ],
                    [],
                )
                if isinstance(e, LLMText)
            ]
        )
    finally:
        await llm.aclose()
    assert text.strip()


async def test_live_structured_tool_call() -> None:
    """The whole premise of the migration: the same GGUF that emits PROSE tool
    calls under Ollama's hand-ported template returns STRUCTURE under
    mistralai's canonical one."""
    llm = _llm()
    try:
        calls = [
            e
            async for e in llm.chat(
                [LLMMessage(role="user", content="Add milk to my cart.")],
                [_cart_tool()],
            )
            if isinstance(e, LLMToolCall)
        ]
    finally:
        await llm.aclose()
    assert len(calls) == 1
    assert (calls[0].server, calls[0].name) == ("shop", "add_to_cart_by_name")
    assert calls[0].from_text is False
    assert calls[0].call_id, "the server supplies the id; we should not mint one"


async def test_live_second_pass_renders_our_history() -> None:
    """Unreachable by any response fixture: what is under test is how OUR
    assistant tool_call and tool result render back through the model's
    template. A template that validated tool-call ids (older Mistral templates
    did) would 400 here while every unit test stayed green."""
    llm = _llm()
    tools = [_cart_tool()]
    try:
        messages = [LLMMessage(role="user", content="Add milk to my cart.")]
        call = None
        async for event in llm.chat(messages, tools):
            if isinstance(event, LLMToolCall):
                call = event
        assert call is not None

        messages.append(LLMMessage(role="assistant", content="", tool_calls=[call]))
        messages.append(
            LLMMessage(
                role="tool",
                tool_call_id=call.call_id,
                content='<external>{"added": true, "name": "Avonmore Milk 1L"}</external>',
            )
        )
        reply = "".join(
            [
                e.text
                async for e in llm.chat(messages, tools)
                if isinstance(e, LLMText)
            ]
        )
    finally:
        await llm.aclose()
    assert reply.strip()


async def test_live_unoffered_tool_is_not_reachable() -> None:
    """Offer nothing, ask for a cart write. Whatever the model does, no call to
    a real server may come back."""
    llm = _llm()
    try:
        calls = [
            e
            async for e in llm.chat(
                [LLMMessage(role="user", content="Add milk to my cart.")], []
            )
            if isinstance(e, LLMToolCall)
        ]
    finally:
        await llm.aclose()
    assert all(c.server == "unknown" for c in calls)
