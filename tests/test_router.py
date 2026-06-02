"""v2.6 hybrid router: deterministic rules + organizer routing/escalation."""

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


def test_reasoning_markers_route_cloud() -> None:
    r = Router()
    assert r.decide("Why is the sky blue?").target == "cloud"
    assert r.decide("Compare oat milk and soy milk").target == "cloud"
    assert r.decide("Explain how tariffs work").target == "cloud"


def test_short_imperative_routes_local() -> None:
    r = Router()
    d = r.decide("Add milk to the cart")
    assert d.target == "local" and d.confidence == "high"


def test_short_utterance_routes_local() -> None:
    assert Router().decide("What time is it?").target == "local"


def test_long_request_routes_cloud() -> None:
    r = Router(max_words_local=10)
    long = "please walk me through the entire process of setting up the kitchen speaker"
    assert r.decide(long).target == "cloud"


def test_ambiguous_midlength_is_local_low_confidence() -> None:
    d = Router().decide("the milk situation in the fridge is getting complicated lately")
    assert d.target == "local" and d.confidence == "low"


def test_empty_request_is_local() -> None:
    assert Router().decide("   ").target == "local"


# ---- Organizer routing + escalation -------------------------------------


class ScriptedLLM:
    """Emits a fixed sequence of events then a final text. `tag` is woven into
    the text so the test can tell local vs. cloud output apart."""

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
async def _organizer(*, router, cloud_llm, local, escalate=True, tmp_path):
    bindings = [
        ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")
    ]
    sink: list[tuple[str, dict]] = []

    async def send(cid, msg: BaseModel):
        sink.append((cid, msg.model_dump()))

    reg = MCPRegistry()
    reg.register(NowTool())
    org = Organizer(
        llm=local,
        mcp=reg,
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={b.client_id: b for b in bindings}.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        router=router,
        cloud_llm=cloud_llm,
        escalate_on_failed=escalate,
    )
    try:
        yield org, sink
    finally:
        await org.close()


@pytest.mark.asyncio
async def test_router_routes_reasoning_to_cloud(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        cloud_llm=ScriptedLLM("cloud"),
        local=ScriptedLLM("local"),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Why is the sky blue?")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        assert notices and notices[0]["target"] == "cloud"
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert any("cloud" in t for t in deltas)


@pytest.mark.asyncio
async def test_router_keeps_action_local(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        cloud_llm=ScriptedLLM("cloud"),
        local=ScriptedLLM("local"),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        assert notices and notices[0]["target"] == "local"
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert any("local" in t for t in deltas)


@pytest.mark.asyncio
async def test_failed_local_turn_escalates_to_cloud(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        cloud_llm=ScriptedLLM("cloud"),
        local=ScriptedLLM("local", fail=True),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        # First local, then an escalation notice to cloud.
        assert [n["target"] for n in notices] == ["local", "cloud"]
        assert notices[1]["escalated"] is True
        # Final spoken text is the cloud retry's, and the emitted outcome is
        # the cloud result (done), not the local failure.
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert any("cloud" in t for t in deltas)
        outcome = next(m for _, m in sink if m["type"] == "turn_outcome")
        assert outcome["outcome"] == "done"


@pytest.mark.asyncio
async def test_no_escalation_when_cloud_absent(tmp_path: Path) -> None:
    async with _organizer(
        router=Router(),
        cloud_llm=None,
        local=ScriptedLLM("local", fail=True),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "Roll some dice")
        await org.flush()
        notices = [m for _, m in sink if m["type"] == "route_notice"]
        assert [n["target"] for n in notices] == ["local"]
        outcome = next(m for _, m in sink if m["type"] == "turn_outcome")
        assert outcome["outcome"] == "failed"


@pytest.mark.asyncio
async def test_failed_turn_with_mutation_does_not_escalate(tmp_path: Path) -> None:
    # A turn that already landed a successful mutating call must not re-run on
    # cloud even if it classifies `failed` (e.g. loop exhaustion afterwards) —
    # replaying the user request would fire the side effect twice.
    from glados.core.turn_outcome import TurnRecord

    async with _organizer(
        router=Router(),
        cloud_llm=ScriptedLLM("cloud"),
        local=ScriptedLLM("local"),
        tmp_path=tmp_path,
    ) as (org, _sink):
        mutated = TurnRecord(loop_exhausted=True)
        mutated.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
        assert org._should_escalate("local", mutated) is False

        clean_fail = TurnRecord(loop_exhausted=True)
        clean_fail.record_tool("dunnes.search_products", ok=True, mutating=False)
        assert org._should_escalate("local", clean_fail) is True


def test_local_loopback_cloud_brain_reuses_local_llm() -> None:
    # provider="local" with no distinct model => the smart path aliases the
    # already-built local brain (no API key, no cloud opt-in needed).
    from glados.brain.llm.fake import FakeLLM
    from glados.core.config import LLMConfig, RouterConfig
    from glados.core.server import _build_cloud_llm

    local = FakeLLM()
    cfg = RouterConfig(enabled=True, provider="local")
    assert _build_cloud_llm(cfg, LLMConfig(), local) is local


def test_local_loopback_distinct_model_builds_separate_ollama() -> None:
    from glados.brain.llm.fake import FakeLLM
    from glados.brain.llm.ollama import OllamaLLM
    from glados.core.config import LLMConfig, RouterConfig
    from glados.core.server import _build_cloud_llm

    cfg = RouterConfig(
        enabled=True, provider="local", local_smart_model="qwen2.5:14b-instruct"
    )
    cloud = _build_cloud_llm(cfg, LLMConfig(model="qwen2.5:7b-instruct"), FakeLLM())
    assert isinstance(cloud, OllamaLLM) and cloud is not None


def test_router_disabled_builds_no_cloud_brain() -> None:
    from glados.brain.llm.fake import FakeLLM
    from glados.core.config import LLMConfig, RouterConfig
    from glados.core.server import _build_cloud_llm

    assert _build_cloud_llm(RouterConfig(enabled=False), LLMConfig(), FakeLLM()) is None


@pytest.mark.asyncio
async def test_no_router_emits_no_route_notice(tmp_path: Path) -> None:
    async with _organizer(
        router=None,
        cloud_llm=None,
        local=ScriptedLLM("local"),
        tmp_path=tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello")
        await org.flush()
        assert not [m for _, m in sink if m["type"] == "route_notice"]
