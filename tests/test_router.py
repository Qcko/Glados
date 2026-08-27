"""v2.6 local multi-model router: deterministic rules + organizer routing/escalation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.brain.router import Router
from glados.core.adapters import LLMText, LLMToolCall
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry
from glados.servers.time_server import NowTool


# ---- Deterministic rules ------------------------------------------------


def test_reasoning_markers_route_specialist() -> None:
    r = Router()
    assert r.decide("Why is the sky blue?").target == "specialist"
    assert r.decide("Compare oat milk and soy milk").target == "specialist"
    assert r.decide("Explain how tariffs work").target == "specialist"


def test_short_imperative_routes_primary() -> None:
    r = Router()
    d = r.decide("Add milk to the cart")
    assert d.target == "primary" and d.confidence == "high"


def test_short_utterance_routes_primary() -> None:
    assert Router().decide("What time is it?").target == "primary"


def test_long_request_routes_specialist() -> None:
    r = Router(max_words_local=10)
    long = "please walk me through the entire process of setting up the kitchen speaker"
    assert r.decide(long).target == "specialist"


def test_long_action_request_stays_primary() -> None:
    # The poison seed: a long multi-item add must NOT route to the specialist by
    # word count -- the imperative check now precedes the length gate so it stays
    # on the primary (which doesn't truncate the list). SESSION 2026-06-16.
    r = Router(max_words_local=10)
    long_add = (
        "add two carrots an onion some thyme a bay leaf and a litre of milk "
        "to the cart please"
    )
    d = r.decide(long_add)
    assert d.target == "primary" and d.reason == "tool-trigger imperative"


def test_specialist_markers_beat_action_lead() -> None:
    # Reasoning markers are still checked first, so an action-led request that
    # also asks for open-ended reasoning routes to the specialist.
    assert Router().decide("Add milk but explain why oat is better").target == "specialist"


def test_ambiguous_midlength_is_primary_low_confidence() -> None:
    d = Router().decide("the milk situation in the fridge is getting complicated lately")
    assert d.target == "primary" and d.confidence == "low"


def test_empty_request_is_primary() -> None:
    assert Router().decide("   ").target == "primary"


# ---- Organizer routing + escalation -------------------------------------


class ScriptedLLM:
    """Emits a fixed sequence of events then a final text. `tag` is woven into
    the text so the test can tell primary vs. specialist output apart."""

    def __init__(self, tag: str, *, fail: bool = False) -> None:
        self._tag = tag
        self._fail = fail

    async def chat(self, messages, tools):
        if self._fail and messages[-1].role != "tool":
            # First pass: call a tool that will error (unknown server) so the
            # turn classifies as `failed`.
            yield LLMToolCall(call_id="x", server="nope", name="nope", args={})
            return
        yield LLMText(text=f"{self._tag} reply")


@asynccontextmanager
async def _organizer(*, router, specialist_llm, primary, escalate=True, tmp_path):
    bindings = [
        ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")
    ]
    sink: list[tuple[str, dict]] = []

    async def send(cid, msg: BaseModel):
        sink.append((cid, msg.model_dump()))

    reg = MCPRegistry()
    reg.register(NowTool())
    org = Organizer(
        llm=primary,
        mcp=reg,
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={b.client_id: b for b in bindings}.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        router=router,
        specialist_llm=specialist_llm,
        escalate_on_failed=escalate,
    )
    try:
        yield org, sink
    finally:
        await org.close()


@pytest.mark.asyncio
async def test_router_routes_reasoning_to_specialist(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        specialist_llm=ScriptedLLM("specialist"),
        primary=ScriptedLLM("primary"),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Why is the sky blue?")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        assert notices and notices[0]["target"] == "specialist"
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert any("specialist" in t for t in deltas)


@pytest.mark.asyncio
async def test_router_keeps_action_primary(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        specialist_llm=ScriptedLLM("specialist"),
        primary=ScriptedLLM("primary"),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        assert notices and notices[0]["target"] == "primary"
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert any("primary" in t for t in deltas)


@pytest.mark.asyncio
async def test_failed_primary_turn_escalates_to_specialist(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        specialist_llm=ScriptedLLM("specialist"),
        primary=ScriptedLLM("primary", fail=True),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        # First primary, then an escalation notice to the specialist.
        assert [n["target"] for n in notices] == ["primary", "specialist"]
        assert notices[1]["escalated"] is True
        # Final spoken text is the specialist retry's, and the emitted outcome
        # is the specialist result (done), not the primary failure.
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert any("specialist" in t for t in deltas)
        outcome = next(m for _, m in sink if m["type"] == "turn_outcome")
        assert outcome["outcome"] == "done"


@pytest.mark.asyncio
async def test_no_escalation_when_specialist_absent(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        specialist_llm=None,
        primary=ScriptedLLM("primary", fail=True),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        assert [n["target"] for n in notices] == ["primary"]
        outcome = next(m for _, m in sink if m["type"] == "turn_outcome")
        assert outcome["outcome"] == "failed"


@pytest.mark.asyncio
async def test_failed_turn_with_mutation_does_not_escalate(tmp_path: Path) -> None:
    # A turn that already landed a successful mutating call must not re-run on
    # the specialist even if it classifies `failed` (e.g. loop exhaustion
    # afterwards) -- replaying the user request would fire the side effect twice.
    from glados.core.turn_outcome import TurnRecord

    async with _organizer(
        router=Router(),
        specialist_llm=ScriptedLLM("specialist"),
        primary=ScriptedLLM("primary"),
        tmp_path=tmp_path,
    ) as (org, _sink):
        mutated = TurnRecord(loop_exhausted=True)
        mutated.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
        assert org._should_escalate("primary", mutated) is False

        clean_fail = TurnRecord(loop_exhausted=True)
        clean_fail.record_tool("dunnes.search_products", ok=True, mutating=False)
        assert org._should_escalate("primary", clean_fail) is True


def test_local_specialist_reuses_primary_llm() -> None:
    # provider="local" with no distinct model => the specialist aliases the
    # already-built primary brain (no API key, no cloud opt-in needed).
    from glados.brain.llm.fake import FakeLLM
    from glados.core.config import LLMConfig, RouterConfig
    from glados.core.server import _build_specialist_llm

    primary = FakeLLM()
    cfg = RouterConfig(enabled=True, provider="local")
    assert _build_specialist_llm(cfg, LLMConfig(), primary) is primary


def test_local_specialist_distinct_model_builds_separate_ollama() -> None:
    from glados.brain.llm.fake import FakeLLM
    from glados.brain.llm.ollama import OllamaLLM
    from glados.core.config import LLMConfig, RouterConfig
    from glados.core.server import _build_specialist_llm

    cfg = RouterConfig(
        enabled=True, provider="local", local_smart_model="qwen2.5:14b-instruct"
    )
    specialist = _build_specialist_llm(
        cfg, LLMConfig(model="qwen3:4b", text_tool_format=None), FakeLLM()
    )
    assert isinstance(specialist, OllamaLLM) and specialist is not None


def test_router_disabled_builds_no_specialist_brain() -> None:
    from glados.brain.llm.fake import FakeLLM
    from glados.core.config import LLMConfig, RouterConfig
    from glados.core.server import _build_specialist_llm

    assert (
        _build_specialist_llm(RouterConfig(enabled=False), LLMConfig(), FakeLLM())
        is None
    )


@pytest.mark.asyncio
async def test_no_router_emits_no_route_notice(tmp_path: Path) -> None:
    async with _organizer(
        router=None,
        specialist_llm=None,
        primary=ScriptedLLM("primary"),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello")
        await org.flush()
        assert not [m for _, m in sink if m["type"] == "route_notice"]
