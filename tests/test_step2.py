# rule-guard:allow ascii-source - Thai text is required i18n fixture data for the language path.
"""v0 step 2: Organizer + fake LLM + MCP + traces, end-to-end."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from glados.brain.llm.fake import FakeLLM
from glados.core.adapters import LLMMessage, ToolSpec
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry
from glados.servers.time_server import NowTool


# ---- FakeLLM unit -------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_llm_emits_tool_call_for_time() -> None:
    llm = FakeLLM()
    tools = [NowTool().spec]
    msgs = [LLMMessage(role="user", content="what time is it?")]
    events = [e async for e in llm.chat(msgs, tools)]
    assert len(events) == 1
    assert events[0].type == "tool_call"
    assert events[0].server == "time" and events[0].name == "now"


@pytest.mark.asyncio
async def test_fake_llm_uses_tool_result() -> None:
    llm = FakeLLM()
    msgs = [
        LLMMessage(role="user", content="time?"),
        LLMMessage(
            role="tool",
            tool_call_id="x",
            content=json.dumps({"human": "Friday 14:30", "iso": "2026-05-08T14:30:00"}),
        ),
    ]
    events = [e async for e in llm.chat(msgs, [])]
    assert events[0].type == "text"
    assert "Friday 14:30" in events[0].text


@pytest.mark.asyncio
async def test_fake_llm_echoes_other_text() -> None:
    llm = FakeLLM()
    msgs = [LLMMessage(role="user", content="hello there")]
    events = [e async for e in llm.chat(msgs, [])]
    assert events[0].type == "text"
    assert events[0].text == "echo: hello there"


# ---- MCP dispatch -------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_dispatch_runs_tool() -> None:
    reg = MCPRegistry()
    reg.register(NowTool())
    env = CallEnvelope(session_id="s", room_id="desk", speaker_id="qcko")
    result = await reg.dispatch("time", "now", {}, env)
    assert result.ok
    assert "iso" in result.content


@pytest.mark.asyncio
async def test_mcp_dispatch_unknown_tool() -> None:
    reg = MCPRegistry()
    env = CallEnvelope(session_id="s", room_id="desk", speaker_id="qcko")
    result = await reg.dispatch("nope", "nope", {}, env)
    assert not result.ok
    assert "unknown tool" in result.error


@pytest.mark.asyncio
async def test_mcp_dispatch_timeout() -> None:
    class SlowTool:
        spec = ToolSpec(server="slow", name="hang", description="", parameters={})

        async def call(self, args, envelope):
            await asyncio.sleep(5)
            raise AssertionError("should have been cancelled")

    reg = MCPRegistry()
    reg.register(SlowTool())
    env = CallEnvelope(session_id="s", room_id="desk", speaker_id="qcko")
    result = await reg.dispatch("slow", "hang", {}, env, timeout=0.05)
    assert not result.ok
    assert "timeout" in result.error


@pytest.mark.asyncio
async def test_mcp_dispatch_per_tool_timeout_override() -> None:
    """spec.timeout_s overrides the dispatch default — both shorter
    (forces timeout against a slow tool) and longer (lets a slow tool
    finish despite a tight default)."""

    class SlowTool:
        spec = ToolSpec(
            server="slow",
            name="hang",
            description="",
            parameters={},
            timeout_s=0.02,
        )

        async def call(self, args, envelope):
            await asyncio.sleep(5)
            raise AssertionError("should have been cancelled")

    class QuickButOverBudget:
        spec = ToolSpec(
            server="patient",
            name="wait",
            description="",
            parameters={},
            timeout_s=1.0,
        )

        async def call(self, args, envelope):
            await asyncio.sleep(0.05)
            return MCPCallResult(ok=True, content={"done": True})

    reg = MCPRegistry()
    reg.register(SlowTool())
    reg.register(QuickButOverBudget())
    env = CallEnvelope(session_id="s", room_id="desk", speaker_id="qcko")

    # Spec override (0.02s) wins over the generous dispatch default (5s).
    result = await reg.dispatch("slow", "hang", {}, env, timeout=5.0)
    assert not result.ok
    assert "0.02" in result.error

    # Spec override (1s) wins over the tight dispatch default (0.01s).
    result = await reg.dispatch("patient", "wait", {}, env, timeout=0.01)
    assert result.ok
    assert result.content == {"done": True}


# ---- Organizer ----------------------------------------------------------


@asynccontextmanager
async def _make_organizer(
    bindings: list[ClientBinding], tmp: Path, llm=None, *, extra_tools=(),
    tool_router=None,
):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    by_id = {b.client_id: b for b in bindings}

    def binding_for(cid: str):
        return by_id.get(cid)

    def in_room(rid: str):
        return [b.client_id for b in bindings if b.room_id == rid]

    reg = MCPRegistry()
    reg.register(NowTool())
    for tool in extra_tools:
        reg.register(tool)
    organizer = Organizer(
        llm=llm if llm is not None else FakeLLM(),
        mcp=reg,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=binding_for,
        clients_in_room=in_room,
        tool_router=tool_router,
    )
    try:
        yield organizer, sink
    finally:
        await organizer.close()


@pytest.mark.asyncio
async def test_organizer_runs_tool_loop(tmp_path: Path) -> None:
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what time is it?")
        await org.flush()

        types = [m["type"] for _, m in sink]
        assert types == [
            "welcome", "user_transcript", "tool_call", "tool_result",
            "assistant_delta", "turn_outcome", "done",
        ]
        session_ids = {m["session_id"] for _, m in sink if "session_id" in m}
        assert len(session_ids) == 1


@pytest.mark.asyncio
async def test_organizer_isolates_rooms(tmp_path: Path) -> None:
    async with _make_organizer(
        [
            ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"),
            ClientBinding(client_id="desk2-ui", room_id="desk2", role="ui", default_user="anna"),
        ],
        tmp_path,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what time is it?")
        await org.handle_user_text("desk2-ui", "hello")
        await org.flush()

        desk_msgs = [m for cid, m in sink if cid == "desk-ui"]
        desk2_msgs = [m for cid, m in sink if cid == "desk2-ui"]

        desk_sessions = {m["session_id"] for m in desk_msgs}
        desk2_sessions = {m["session_id"] for m in desk2_msgs}
        assert desk_sessions.isdisjoint(desk2_sessions)
        assert all(s.startswith("desk:qcko") for s in desk_sessions)
        assert all(s.startswith("desk2:anna") for s in desk2_sessions)
        assert "tool_call" in {m["type"] for m in desk_msgs}
        assert "tool_call" not in {m["type"] for m in desk2_msgs}


@pytest.mark.asyncio
async def test_organizer_replays_history_in_same_session(tmp_path: Path) -> None:
    """A follow-up turn in the same session sees the prior turn's user message
    and GLaDOS's reply — the basis for "do that instead" / "add it back"."""
    from glados.core.adapters import LLMText

    class RecordingLLM:
        def __init__(self) -> None:
            self.seen: list[list[tuple[str, str | None]]] = []

        async def chat(self, messages, tools):
            self.seen.append([(m.role, m.content) for m in messages])
            yield LLMText(text="acknowledged")

    llm = RecordingLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=llm,
    ) as (org, _sink):
        # A read (not an action request) so the no-tool "acknowledged" reply
        # stays `done` and is committed verbatim — this test is about replay,
        # not the confabulation rewrite (covered in test_turn_outcome).
        await org.handle_user_text("desk-ui", "what's in my cart")
        await org.flush()
        await org.handle_user_text("desk-ui", "what about eggs")
        await org.flush()

    first_seen, second_seen = llm.seen
    # First turn: just system + the user message, no prior history.
    assert first_seen == [("system", org._system_prompt), ("user", "what's in my cart")]
    # Second turn: the first exchange is replayed before the new user message.
    assert ("user", "what's in my cart") in second_seen
    assert ("assistant", "acknowledged") in second_seen
    assert second_seen[-1] == ("user", "what about eggs")


@pytest.mark.asyncio
async def test_organizer_suppresses_confabulated_action(tmp_path: Path) -> None:
    """An action request answered with a declarative claim but zero tool
    dispatches is a fabricated completion: the outcome is `confabulated`, the
    spoken/correction text is the honest failure line, and the false claim is
    never committed to the hot history buffer (so it can't poison later turns)."""
    from glados.core.adapters import LLMText
    from glados.core.organizer import _CONFABULATION_REPLIES

    class ConfabLLM:
        async def chat(self, messages, tools):
            yield LLMText(text="Done — I've added the milk to your cart.")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=ConfabLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

        outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes == ["confabulated"]
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        # The false claim still streamed live, followed by the honest correction.
        assert "Done — I've added the milk to your cart." in deltas
        assert any(_CONFABULATION_REPLIES[0] in d for d in deltas)

        session_id = next(iter(org._history))
        committed = org._history[session_id]
        assert committed[-1].role == "assistant"
        assert committed[-1].content == _CONFABULATION_REPLIES[0]
        assert "added the milk" not in (committed[-1].content or "")


_THAI = "บริการสภาพอากาศไม่พร้อมใช้งานในขณะนี้"


@pytest.mark.asyncio
async def test_organizer_repairs_language_drift(tmp_path: Path) -> None:
    """A free-form reply that drifts to another language is repaired into the
    configured language by one local re-inference; the repaired text is what
    streams as a correction and what commits to history (the drift never does)."""
    from glados.core.adapters import LLMText

    class DriftLLM:
        """Drifts to Thai on the turn; the repair pass (system says 'translate')
        returns English."""

        async def chat(self, messages, tools):
            if "translate" in (messages[0].content or "").lower():
                yield LLMText(text="The weather service is unavailable right now.")
            else:
                yield LLMText(text=_THAI)

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=DriftLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what is the weather")
        await org.flush()

        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert _THAI in deltas  # the drift streamed live...
        assert any("weather service is unavailable" in d for d in deltas)  # ...then the fix

        session_id = next(iter(org._history))
        committed = org._history[session_id]
        assert committed[-1].role == "assistant"
        assert committed[-1].content == "The weather service is unavailable right now."
        assert _THAI not in (committed[-1].content or "")


@pytest.mark.asyncio
async def test_organizer_falls_back_when_repair_still_drifts(tmp_path: Path) -> None:
    """If the one repair pass still drifts, a deterministic in-language fallback
    line is spoken/committed -- never the drifted text."""
    from glados.core.adapters import LLMText
    from glados.core.language_guard import detect_drift, fallback_line

    class AlwaysDriftLLM:
        async def chat(self, messages, tools):
            yield LLMText(text=_THAI)  # both the turn and the repair drift

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=AlwaysDriftLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what is the weather")
        await org.flush()

        session_id = next(iter(org._history))
        committed = org._history[session_id]
        assert committed[-1].content == fallback_line("en")
        assert detect_drift(committed[-1].content or "", "en") is False


@pytest.mark.asyncio
async def test_confabulation_wins_over_language_guard(tmp_path: Path) -> None:
    """A turn that is BOTH a confabulated action AND drifted is handled by the
    confabulation path only -- the language guard does not also fire (one
    correction, one history rewrite)."""
    from glados.core.adapters import LLMText
    from glados.core.organizer import _CONFABULATION_REPLIES

    class ConfabDriftLLM:
        """An action claim, in Thai, with zero tool calls."""

        async def chat(self, messages, tools):
            yield LLMText(text=_THAI)

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=ConfabDriftLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "add milk to the cart")
        await org.flush()

        outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes == ["confabulated"]
        session_id = next(iter(org._history))
        committed = org._history[session_id]
        assert committed[-1].content in _CONFABULATION_REPLIES


class _FakeTool:
    """Minimal registry tool; succeeds unless given an `error`."""

    def __init__(
        self,
        server: str,
        name: str,
        *,
        mutating: bool = False,
        error: str | None = None,
        properties: dict | None = None,
    ) -> None:
        self.spec = ToolSpec(
            server=server, name=name, description=f"{server}.{name}",
            parameters={"type": "object", "properties": properties or {}},
            mutating=mutating,
        )
        self._error = error

    async def call(self, args, envelope):
        if self._error is not None:
            return MCPCallResult(ok=False, content=None, error=self._error)
        return MCPCallResult(ok=True, content={"ok": True})


def _failing_tool(server: str, name: str) -> _FakeTool:
    """A mutating tool whose call always errors, so a reply claiming success
    is contradicted by the dispatch record."""
    return _FakeTool(
        server, name, mutating=True, error="tool exploded",
        properties={"name": {"type": "string"}},
    )


def _weather_router():
    from glados.brain.tool_router import ServerScope, ToolRouter

    return ToolRouter(scopes={
        "time": ServerScope("time", core=True),
        "weather": ServerScope("weather", intent_keywords=("weather", "forecast")),
    })


@pytest.mark.asyncio
async def test_tool_scope_hides_unmatched_server_from_model(tmp_path: Path) -> None:
    """A turn that matches no server keyword is offered only the core tools --
    the annotated server's tools never reach the model (the overload fix)."""
    from glados.core.adapters import LLMText

    class RecordingLLM:
        def __init__(self) -> None:
            self.offered: list[set[str]] = []

        async def chat(self, messages, tools):
            self.offered.append({t.qualified for t in tools})
            yield LLMText(text="hi there")

    llm = RecordingLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path, llm=llm, extra_tools=[_FakeTool("weather", "get")],
        tool_router=_weather_router(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "say something nice")
        await org.flush()

    assert llm.offered  # the model was called
    assert all("weather.get" not in offered for offered in llm.offered)
    assert any("time.now" in offered for offered in llm.offered)  # core stays


@pytest.mark.asyncio
async def test_tool_scope_offers_matched_server(tmp_path: Path) -> None:
    """A matching keyword brings the server's tools into scope."""
    from glados.core.adapters import LLMText

    class RecordingLLM:
        def __init__(self) -> None:
            self.offered: list[set[str]] = []

        async def chat(self, messages, tools):
            self.offered.append({t.qualified for t in tools})
            yield LLMText(text="it is sunny")

    llm = RecordingLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path, llm=llm, extra_tools=[_FakeTool("weather", "get")],
        tool_router=_weather_router(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what is the weather like")
        await org.flush()

    assert any("weather.get" in offered for offered in llm.offered)


@pytest.mark.asyncio
async def test_misroute_falls_back_to_full_tool_set(tmp_path: Path) -> None:
    """A scoped turn that FAILS without mutating re-drives once on the full tool
    set, so a mis-route (the right tool was scoped out) is recoverable."""
    import uuid

    from glados.core.adapters import LLMText, LLMToolCall

    class MisrouteLLM:
        async def chat(self, messages, tools):
            names = {t.qualified for t in tools}
            if messages[-1].role == "tool":
                yield LLMText(text="done")
                return
            if "weather.get" in names:  # full-tool pass: real call
                yield LLMToolCall(call_id=uuid.uuid4().hex[:8], server="weather",
                                  name="get", args={})
            else:  # scoped pass: weather hidden -> call a missing tool -> error
                yield LLMToolCall(call_id=uuid.uuid4().hex[:8], server="ghost",
                                  name="missing", args={})

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path, llm=MisrouteLLM(), extra_tools=[_FakeTool("weather", "get")],
        tool_router=_weather_router(),
    ) as (org, sink):
        # "tell me about the climate" matches no weather keyword -> weather is
        # scoped out -> scoped pass errors -> full-tool fallback recovers.
        await org.handle_user_text("desk-ui", "tell me about the climate outside")
        await org.flush()

        outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes == ["done"]  # recovered via the fallback
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert any(m["ok"] for m in results)  # the real weather tool ran


@pytest.mark.asyncio
async def test_misroute_fallback_does_not_fire_after_mutation(tmp_path: Path) -> None:
    """The full-tool fallback must NOT re-drive once a mutating call landed --
    re-driving could double-fire the side effect. A scoped turn that mutated and
    then errored stays `failed`, with no second top-level drive."""
    import uuid

    from glados.brain.tool_router import ServerScope, ToolRouter
    from glados.core.adapters import LLMText, LLMToolCall

    drives: list[set[str]] = []

    class MutateThenErrorLLM:
        async def chat(self, messages, tools):
            tool_msgs = [m for m in messages if m.role == "tool"]
            if not tool_msgs:  # first pass of a drive
                drives.append({t.qualified for t in tools})
                yield LLMToolCall(call_id=uuid.uuid4().hex[:8], server="weather",
                                  name="get", args={})  # mutating, lands ok
            elif len(tool_msgs) == 1:
                yield LLMToolCall(call_id=uuid.uuid4().hex[:8], server="ghost",
                                  name="missing", args={})  # unrecovered error
            else:
                yield LLMText(text="done")

    # dunnes is annotated but won't match the prompt -> it's scoped OUT, so the
    # scope is a strict subset and the fallback CONDITION's `scoped` is true;
    # the mutation guard is what must suppress the re-drive.
    router = ToolRouter(scopes={
        "time": ServerScope("time", core=True),
        "weather": ServerScope("weather", intent_keywords=("weather",)),
        "dunnes": ServerScope("dunnes", intent_keywords=("cart",)),
    })
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path, llm=MutateThenErrorLLM(),
        extra_tools=[_FakeTool("weather", "get", mutating=True),
                     _FakeTool("dunnes", "buy", mutating=True)],
        tool_router=router,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "weather please add it")
        await org.flush()

        outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes == ["failed"]  # stayed failed; fallback suppressed
    # Exactly one top-level drive (no full-tool re-drive after the mutation).
    assert sum("weather.get" in d for d in drives) == 1


def test_build_tool_router_none_when_no_scoping(tmp_path: Path) -> None:
    """Scoping is OFF by default: with no server declaring intent_keywords/core,
    _build_tool_router returns None and the organizer offers every tool."""
    from glados.core.config import ServerEntry, ServersConfig
    from glados.core.server import _build_tool_router

    cfg = ServersConfig(server=[
        ServerEntry(id="toy", command="python", args=[]),
        ServerEntry(id="dunnes", command="dotnet", args=[]),
    ])
    assert _build_tool_router(cfg) is None

    scoped = ServersConfig(server=[
        ServerEntry(id="dunnes", command="dotnet", args=[],
                    intent_keywords=["cart"]),
    ])
    assert _build_tool_router(scoped) is not None


@pytest.mark.asyncio
async def test_organizer_forces_time_tool_on_time_question(tmp_path: Path) -> None:
    """A time question forces a deterministic `time.now` dispatch and seeds the
    real time into the turn before the model generates — even when the model
    never asks for a tool (the fabrication bug, SESSION 2026-06-15 Finding 2)."""
    from glados.core.adapters import LLMText

    class NoToolLLM:
        """Answers from prior, never calls a tool — the fabricating model."""

        def __init__(self) -> None:
            self.seen: list[list[tuple[str, str | None]]] = []

        async def chat(self, messages, tools):
            self.seen.append([(m.role, m.content) for m in messages])
            yield LLMText(text="It's twenty past three.")

    llm = NoToolLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=llm,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what time is it?")
        await org.flush()

        tool_calls = [m for _, m in sink if m["type"] == "tool_call"]
        assert any(m["server"] == "time" and m["name"] == "now" for m in tool_calls)
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert any(m["ok"] for m in results)
        # The model's first (and only) pass already saw the forced tool result,
        # so it renders ground truth instead of inventing a time.
        assert any(role == "tool" for role, _ in llm.seen[0])
        outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes == ["done"]


@pytest.mark.asyncio
async def test_organizer_does_not_force_time_on_non_time_turn(tmp_path: Path) -> None:
    """A non-time turn never triggers the forced time dispatch."""
    from glados.core.adapters import LLMText

    class NoToolLLM:
        async def chat(self, messages, tools):
            yield LLMText(text="You have milk.")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=NoToolLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what's in my cart")
        await org.flush()

        assert not any(m["type"] == "tool_call" for _, m in sink)


def test_escape_kind_detection() -> None:
    from glados.core.organizer import _escape_kind

    assert _escape_kind("start over") == "start_over"
    assert _escape_kind("Let's start fresh.") == "start_over"
    assert _escape_kind("clear the chat") == "start_over"
    assert _escape_kind("reset the conversation") == "start_over"
    assert _escape_kind("did that actually work?") == "recheck"
    assert _escape_kind("did it work") == "recheck"
    assert _escape_kind("did that go through?") == "recheck"
    assert _escape_kind("has it actually gone through") == "recheck"
    # Normal turns and mid-sentence mentions must NOT trip an escape hatch.
    assert _escape_kind("add milk") is None
    assert _escape_kind("can you start over from step 2") is None
    assert _escape_kind("what's in my cart") is None
    assert _escape_kind("") is None


@pytest.mark.asyncio
async def test_organizer_start_over_clears_history(tmp_path: Path) -> None:
    """"start over" is a consented history clear: the next turn sees no prior
    exchange, and the clear turn never reaches the LLM."""
    from glados.core.adapters import LLMText

    class RecordingLLM:
        def __init__(self) -> None:
            self.seen: list[list[tuple[str, str | None]]] = []

        async def chat(self, messages, tools):
            self.seen.append([(m.role, m.content) for m in messages])
            yield LLMText(text="ok")

    llm = RecordingLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=llm,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "what's in my cart")
        await org.flush()
        await org.handle_user_text("desk-ui", "start over")
        await org.flush()
        await org.handle_user_text("desk-ui", "what's on sale")
        await org.flush()

    # The clear turn short-circuits the LLM, so only two real turns reached it.
    assert len(llm.seen) == 2
    assert llm.seen[1] == [("system", org._system_prompt), ("user", "what's on sale")]
    deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
    assert any("cleared" in d.lower() for d in deltas)


@pytest.mark.asyncio
async def test_organizer_recheck_reports_no_after_confabulation(tmp_path: Path) -> None:
    """"did that actually work?" reports the dispatch-grounded truth — after a
    confabulated (zero-tool) action turn, the honest answer is No."""
    from glados.core.adapters import LLMText

    class ConfabLLM:
        async def chat(self, messages, tools):
            yield LLMText(text="Done — I've added the milk to your cart.")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=ConfabLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()
        await org.handle_user_text("desk-ui", "did that actually work?")
        await org.flush()

    deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
    assert any(d.strip().lower().startswith("no") for d in deltas)


@pytest.mark.asyncio
async def test_recheck_reports_yes_after_real_mutation(tmp_path: Path) -> None:
    """The truth report says Yes when a mutating call actually landed, and
    reports nothing-to-check for an unknown session."""
    from glados.core.turn_outcome import TurnRecord

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
    ) as (org, _sink):
        rec = TurnRecord()
        rec.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
        org._record_last_turn("s1", rec, "done")
        assert org._describe_last_turn("s1").lower().startswith("yes")
        assert org._describe_last_turn("unknown").startswith("I don't have")


@pytest.mark.asyncio
async def test_organizer_no_history_bleed_across_rooms(tmp_path: Path) -> None:
    """Distinct (room, speaker) sessions never see each other's history."""
    from glados.core.adapters import LLMText

    class RecordingLLM:
        def __init__(self) -> None:
            self.seen: list[list[tuple[str, str | None]]] = []

        async def chat(self, messages, tools):
            self.seen.append([(m.role, m.content) for m in messages])
            yield LLMText(text="ok")

    llm = RecordingLLM()
    async with _make_organizer(
        [
            ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"),
            ClientBinding(client_id="kit-ui", room_id="kitchen", role="ui", default_user="anna"),
        ],
        tmp_path,
        llm=llm,
    ) as (org, _sink):
        await org.handle_user_text("desk-ui", "secret desk thing")
        await org.flush()
        await org.handle_user_text("kit-ui", "kitchen thing")
        await org.flush()

    _desk_seen, kitchen_seen = llm.seen
    assert ("user", "secret desk thing") not in kitchen_seen
    assert kitchen_seen[0] == ("system", org._system_prompt)


def _turn(user: str) -> list[LLMMessage]:
    """A whole turn's worth of history messages: user → assistant → tool."""
    return [
        LLMMessage(role="user", content=user),
        LLMMessage(role="assistant", content=f"did {user}"),
        LLMMessage(role="tool", tool_call_id="c", content="{}"),
    ]


@pytest.mark.asyncio
async def test_cap_history_keeps_last_n_turns_whole(tmp_path: Path) -> None:
    """`_cap_history` slices on user-message boundaries so each kept turn keeps
    its assistant + tool messages — never a turn cut mid-way."""
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
    ) as (org, _sink):
        org._history_max_turns = 2
        history = _turn("one") + _turn("two") + _turn("three")
        capped = org._cap_history(history)
        users = [m.content for m in capped if m.role == "user"]
        assert users == ["two", "three"]  # oldest turn dropped
        # First kept message is the user that opens turn "two" — turn boundary,
        # not a stray assistant/tool from the dropped turn.
        assert capped[0].role == "user" and capped[0].content == "two"
        # Each kept turn still carries its assistant + tool messages.
        assert [m.role for m in capped] == [
            "user", "assistant", "tool", "user", "assistant", "tool",
        ]


@pytest.mark.asyncio
async def test_cap_history_noop_under_limit(tmp_path: Path) -> None:
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
    ) as (org, _sink):
        org._history_max_turns = 8
        history = _turn("one") + _turn("two")
        assert org._cap_history(history) == history


@pytest.mark.asyncio
async def test_commit_history_skips_noop_turn(tmp_path: Path) -> None:
    """An empty/garbled turn (no tools, blank reply) must leave prior history
    untouched rather than diluting the buffer with an empty exchange."""
    from glados.core.turn_outcome import TurnRecord

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
    ) as (org, _sink):
        org._history["s1"] = _turn("kept")
        org._commit_history("s1", _turn("kept") + _turn("garbled"), TurnRecord(), "")
        assert [m.content for m in org._history["s1"] if m.role == "user"] == ["kept"]


@pytest.mark.asyncio
async def test_commit_history_persists_productive_turn(tmp_path: Path) -> None:
    from glados.core.turn_outcome import TurnRecord

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
    ) as (org, _sink):
        org._commit_history("s1", _turn("hello"), TurnRecord(), "hi there")
        assert [m.content for m in org._history["s1"] if m.role == "user"] == ["hello"]


@pytest.mark.asyncio
async def test_commit_history_lru_evicts_idle_not_active(tmp_path: Path) -> None:
    """The history dict is bounded; when full it drops the least-recently
    committed session, and re-committing an existing session moves it back to
    the most-recently-used end so an active conversation isn't evicted."""
    from glados.core.organizer import _MAX_TRACKED_SESSIONS
    from glados.core.turn_outcome import TurnRecord

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
    ) as (org, _sink):
        for i in range(_MAX_TRACKED_SESSIONS):
            org._commit_history(f"s{i}", _turn(f"u{i}"), TurnRecord(), "ok")
        # s0 is the oldest. Touch it so it becomes most-recently-used.
        org._commit_history("s0", _turn("u0-again"), TurnRecord(), "ok")
        # One more new session forces an eviction — the now-idle s1, not s0.
        org._commit_history("new", _turn("n"), TurnRecord(), "ok")
        assert len(org._history) == _MAX_TRACKED_SESSIONS
        assert "s1" not in org._history  # idle one evicted
        assert "s0" in org._history  # touched one survived
        assert "new" in org._history


# ---- Trace writer -------------------------------------------------------


def test_session_registry_reuses_within_idle_window() -> None:
    now = [1000.0]
    reg = SessionRegistry(idle_window_s=180.0, clock=lambda: now[0])
    a = reg.get_or_open("desk", "qcko")
    now[0] += 60.0  # follow-up inside the window
    b = reg.get_or_open("desk", "qcko")
    assert a.session_id == b.session_id  # same conversation, history survives
    now[0] += 1000.0  # long gap — past the idle window
    c = reg.get_or_open("desk", "qcko")
    assert c.session_id != a.session_id  # fresh session, empty history
    assert reg.latest("desk", "qcko").session_id == c.session_id


def test_session_registry_isolates_distinct_keys() -> None:
    reg = SessionRegistry()
    assert (
        reg.get_or_open("desk", "qcko").session_id
        != reg.get_or_open("kitchen", "qcko").session_id
    )


@pytest.mark.asyncio
async def test_organizer_handles_tool_loop_exhaustion(tmp_path: Path) -> None:
    from glados.core.adapters import LLMToolCall

    class LoopingLLM:
        async def chat(self, messages, tools):
            yield LLMToolCall(call_id="x", server="time", name="now", args={})

    bindings = [
        ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")
    ]
    sink: list[tuple[str, dict]] = []

    async def send(cid, msg):
        sink.append((cid, msg.model_dump()))

    reg = MCPRegistry()
    reg.register(NowTool())
    org = Organizer(
        llm=LoopingLLM(),
        mcp=reg,
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={b.client_id: b for b in bindings}.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
    )
    try:
        await org.handle_user_text("desk-ui", "loop forever")
        await org.flush()
        types = [m["type"] for _, m in sink]
        assert types.count("tool_call") >= 8
        assert any(
            m["type"] == "assistant_delta" and "stuck" in m["text"] for _, m in sink
        )
        assert types[-1] == "done"
    finally:
        await org.close()


def test_trace_writes_jsonl(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    w = store.open("desk:qcko:abc")
    w.event("turn_start", room_id="desk")
    w.event("user_text", text="hi")
    w.close()
    lines = (tmp_path / "desk_qcko_abc.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert [p["event"] for p in parsed] == ["turn_start", "user_text"]
    assert parsed[0]["room_id"] == "desk"


def test_trace_reopen_appends(tmp_path: Path) -> None:
    """Multi-turn sessions reopen the same trace file each turn; the
    second open must NOT truncate the first turn's events."""
    store = TraceStore(tmp_path)
    w1 = store.open("desk:qcko:multi")
    w1.event("turn_start", turn=1)
    w1.close()
    w2 = store.open("desk:qcko:multi")
    w2.event("turn_start", turn=2)
    w2.close()
    lines = (tmp_path / "desk_qcko_multi.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert [p["turn"] for p in parsed] == [1, 2]


# ---- End-to-end via FastAPI TestClient ---------------------------------


@pytest.fixture(scope="module")
def http_client():
    """See test_audio_pipeline.py for the rationale on context-managing
    TestClient — needed so each module's room workers are torn down before
    the next module's TestClient creates its own event loop."""
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    from glados.core.server import app

    with TestClient(app) as client:
        yield client


def test_e2e_time_question(http_client: TestClient) -> None:
    with http_client.websocket_connect("/ws/v1") as ws:
        ws.send_json(_hello("desk-ui", "desk", "dev-token-desk"))
        ws.send_json({"type": "user_text", "text": "what time is it?"})
        msgs = [ws.receive_json() for _ in range(8)]
    assert [m["type"] for m in msgs] == [
        "welcome",
        "user_transcript",
        "tool_call",
        "tool_result",
        "assistant_delta",
        "tts_chunk",
        "turn_outcome",
        "done",
    ]
    assert msgs[3]["ok"] is True
    assert "iso" in msgs[3]["content"]
    assert msgs[5]["seq"] == 0
    assert msgs[5]["sample_rate"] == 22_050


def test_e2e_audio_frame_writes_wav(http_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import struct
    import wave

    monkeypatch.setattr(http_client.app.state.glados_cfg.server, "traces_dir", tmp_path)
    samples = [0, 1000, -1000, 32767, -32768]
    pcm = b"".join(struct.pack("<h", s) for s in samples)
    frame = struct.pack(">I", 0) + pcm

    with http_client.websocket_connect("/ws/v1") as ws:
        ws.send_json(_hello("desk-ui", "desk", "dev-token-desk"))
        ws.send_bytes(frame)
        ws.send_bytes(struct.pack(">I", 1) + pcm)

    audio_dir = tmp_path / "audio" / "desk-ui"
    wavs = list(audio_dir.glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.getframerate() == 16_000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == len(samples) * 2


def test_e2e_two_tabs_isolated(http_client: TestClient) -> None:
    with http_client.websocket_connect("/ws/v1") as a, http_client.websocket_connect(
        "/ws/v1"
    ) as b:
        a.send_json(_hello("desk-ui", "desk", "dev-token-desk"))
        b.send_json(_hello("desk2-ui", "desk2", "dev-token-desk2"))
        a.send_json({"type": "user_text", "text": "hello"})
        b.send_json({"type": "user_text", "text": "what time is it?"})

        a_msgs = [a.receive_json() for _ in range(4)]
        b_msgs = [b.receive_json() for _ in range(6)]

    assert {m["session_id"] for m in a_msgs}.isdisjoint(
        {m["session_id"] for m in b_msgs}
    )
    assert "tool_call" not in [m["type"] for m in a_msgs]
    assert "tool_call" in [m["type"] for m in b_msgs]


def _hello(client_id: str, room_id: str, token: str) -> dict:
    return {
        "type": "hello",
        "client_id": client_id,
        "room_id": room_id,
        "role": "ui",
        "token": token,
    }


@pytest.mark.asyncio
async def test_silent_turn_does_not_trigger_the_scope_fallback_redrive(
    tmp_path: Path,
) -> None:
    """A turn that produced no reply classifies `failed`, but re-driving it on
    the FULL tool set is pointless: the failure is the model exhausting its
    token budget before it began replying, and a wider tool set only makes the
    prompt bigger. Measured 2026-08-25 -- one silent turn burned two dead passes
    before this guard, each ~4096 tokens."""
    from glados.core.adapters import LLMText  # noqa: F401  (parity with siblings)

    class SilentLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools):
            self.calls += 1
            return
            yield  # pragma: no cover -- makes this an async generator

    llm = SilentLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path, llm=llm, extra_tools=[_FakeTool("weather", "get")],
        tool_router=_weather_router(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "say something nice")
        await org.flush()

    assert llm.calls == 1, "the silent turn must not be re-driven on the full set"
    outcomes = [m["outcome"] for _, m in sink if m.get("type") == "turn_outcome"]
    assert outcomes == ["failed"]

    traced = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert not any(e.get("event") == "tool_scope_fallback_full" for e in traced)


@pytest.mark.asyncio
async def test_false_claim_after_a_failed_call_is_scrubbed(tmp_path: Path) -> None:
    """An unrecovered tool error classifies `failed`, and `failed` never had its
    reply replaced -- so a model that announced success anyway got its false
    claim both SPOKEN and committed to history. The verdict stays `failed`
    (that is what drives escalation); only the lie is replaced."""
    from glados.core.adapters import LLMText, LLMToolCall
    from glados.core.organizer import _UNBACKED_CLAIM_REPLIES

    class ClaimsSuccessLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield LLMToolCall(
                    call_id="c1", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "milk"},
                )
            else:
                yield LLMText(text="Milk removed from cart.")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=ClaimsSuccessLLM(),
        extra_tools=[_failing_tool("dunnes", "remove_from_cart_by_name")],
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "take the milk off")
        await org.flush()

        outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes == ["failed"]
        deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
        assert any(any(r in d for r in _UNBACKED_CLAIM_REPLIES) for d in deltas)


@pytest.mark.asyncio
async def test_unknown_claim_phrasing_is_logged_for_review(tmp_path: Path, caplog) -> None:
    """The claim vocabulary can only be wrong by omission, so a turn that really
    mutated something but matched no pattern is logged as a candidate phrasing.
    "Took the milk off" is exactly the shape `_CLAIM_RE` does not know."""
    from glados.core.adapters import LLMText, LLMToolCall

    class UnknownPhrasingLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield LLMToolCall(
                    call_id="c1", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "milk"},
                )
            else:
                yield LLMText(text="Took the milk off for you.")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=UnknownPhrasingLLM(),
        extra_tools=[_FakeTool("dunnes", "remove_from_cart_by_name", mutating=True)],
    ) as (org, sink):
        with caplog.at_level("INFO"):
            await org.handle_user_text("desk-ui", "take the milk off")
            await org.flush()

    assert any("claim-vocab" in r.message for r in caplog.records)
    assert any("Took the milk off" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_recognised_phrasing_is_not_logged(tmp_path: Path, caplog) -> None:
    """The log must stay a short weekly list, not a record of every mutation."""
    from glados.core.adapters import LLMText, LLMToolCall

    class KnownPhrasingLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield LLMToolCall(
                    call_id="c1", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "milk"},
                )
            else:
                yield LLMText(text="Removed the milk from your cart.")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=KnownPhrasingLLM(),
        extra_tools=[_FakeTool("dunnes", "remove_from_cart_by_name", mutating=True)],
    ) as (org, sink):
        with caplog.at_level("INFO"):
            await org.handle_user_text("desk-ui", "take the milk off")
            await org.flush()

    assert not any("claim-vocab" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_our_own_canned_line_is_never_logged_as_a_candidate(
    tmp_path: Path, caplog
) -> None:
    """The vocabulary log must harvest the MODEL's phrasings, not ours.

    A silent turn that had already mutated gets one of
    `_SILENT_TURN_AFTER_MUTATION_REPLIES` -- which contains no claim verb, and
    is spoken on a turn where `made_successful_mutation()` is true. Logging
    after the scrub would therefore file our own line as evidence of a missing
    phrasing, every single time it happened."""
    from glados.core.adapters import LLMToolCall

    class MutatesThenSaysNothingLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield LLMToolCall(
                    call_id="c1", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "milk"},
                )
            # Second pass says nothing at all.

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=MutatesThenSaysNothingLLM(),
        extra_tools=[_FakeTool("dunnes", "remove_from_cart_by_name", mutating=True)],
    ) as (org, sink):
        with caplog.at_level("INFO"):
            await org.handle_user_text("desk-ui", "take the milk off")
            await org.flush()

    assert not any("claim-vocab" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_clarifying_question_after_a_mutation_is_not_logged(
    tmp_path: Path, caplog
) -> None:
    """Mutated, then asked something. That is not a change reported in unknown
    words -- it is not a report at all."""
    from glados.core.adapters import LLMText, LLMToolCall

    class AsksAfterMutatingLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield LLMToolCall(
                    call_id="c1", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "milk"},
                )
            else:
                yield LLMText(text="Which milk did you mean?")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=AsksAfterMutatingLLM(),
        extra_tools=[_FakeTool("dunnes", "remove_from_cart_by_name", mutating=True)],
    ) as (org, sink):
        with caplog.at_level("INFO"):
            await org.handle_user_text("desk-ui", "take the milk off")
            await org.flush()

    assert not any("claim-vocab" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_confabulated_turn_is_retried_and_finishes_the_job(tmp_path: Path) -> None:
    """"Show me my cart and then remove the milk" -- the model does the first
    half and narrates the second. Nothing mutated, so the replay cannot double a
    side effect, which is what makes this the one failure worth retrying rather
    than only reporting. Measured two thirds of attempts on qwen3:4b."""
    from glados.core.adapters import LLMText, LLMToolCall

    class HalfDoesItLLM:
        def __init__(self) -> None:
            self.passes = 0
            self.seen: list[list] = []

        async def chat(self, messages, tools):
            self.passes += 1
            self.seen.append([m.model_copy(deep=True) for m in messages])
            if self.passes == 1:
                yield LLMToolCall(call_id="c1", server="dunnes", name="view_cart", args={})
            elif self.passes == 2:
                yield LLMText(text="Milk removed from cart.")  # never dispatched
            elif self.passes == 3:
                yield LLMToolCall(
                    call_id="c2", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "milk"},
                )
            else:
                yield LLMText(text="Removed the milk from your cart.")

    llm = HalfDoesItLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=llm,
        extra_tools=[
            _FakeTool("dunnes", "view_cart"),
            _FakeTool("dunnes", "remove_from_cart_by_name", mutating=True),
        ],
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "show me my cart and then remove the milk")
        await org.flush()

    dispatched = [(m["server"], m["name"]) for _, m in sink if m["type"] == "tool_call"]
    assert ("dunnes", "remove_from_cart_by_name") in dispatched
    outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
    assert outcomes == ["done"]

    # The nudge reaches the model as a SYSTEM message, and the user's message
    # is still the user's own words -- putting harness text in the user role
    # would commit it, and the next turn would read our instruction as
    # something the user said.
    retry = llm.seen[2]
    assert any(
        m.role == "system" and "has not happened" in (m.content or "") for m in retry
    )
    user_says = [m.content for m in retry if m.role == "user"]
    assert user_says == ["show me my cart and then remove the milk"]


@pytest.mark.asyncio
async def test_the_retry_happens_once_and_then_the_reply_is_scrubbed(
    tmp_path: Path,
) -> None:
    """A model that ignores an instruction this explicit will not be talked
    round by repeating it, so the retry is bounded to one."""
    from glados.core.adapters import LLMText, LLMToolCall
    from glados.core.organizer import _CONFABULATION_REPLIES

    class AlwaysConfabulatesLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes in (1, 3):
                yield LLMToolCall(call_id=f"c{self.passes}", server="dunnes",
                                  name="view_cart", args={})
            else:
                yield LLMText(text="Milk removed from cart.")

    llm = AlwaysConfabulatesLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=llm,
        extra_tools=[_FakeTool("dunnes", "view_cart")],
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "show me my cart and then remove the milk")
        await org.flush()

    outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
    assert outcomes == ["confabulated"]
    deltas = [m["text"] for _, m in sink if m["type"] == "assistant_delta"]
    assert any(any(r in d for r in _CONFABULATION_REPLIES) for d in deltas)
    # Two drives, not three: one original plus exactly one retry.
    assert llm.passes == 4


@pytest.mark.asyncio
async def test_a_confabulation_after_a_real_mutation_is_never_retried(
    tmp_path: Path,
) -> None:
    """The interlock. If something DID land, replaying the user's request could
    fire that side effect a second time -- the same reason `_should_escalate`
    refuses to replay a mutating turn."""
    from glados.core.adapters import LLMText, LLMToolCall

    class MutatesThenClaimsMoreLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield LLMToolCall(
                    call_id="c1", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "bananas"},
                )
            else:
                yield LLMText(text="Eggs added to cart.")  # never dispatched

    llm = MutatesThenClaimsMoreLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=llm,
        extra_tools=[_FakeTool("dunnes", "remove_from_cart_by_name", mutating=True)],
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "actually, add eggs instead")
        await org.flush()

    outcomes = [m["outcome"] for _, m in sink if m["type"] == "turn_outcome"]
    assert outcomes == ["confabulated"]
    assert llm.passes == 2, "a turn that already mutated must not be replayed"


@pytest.mark.asyncio
async def test_the_retry_nudge_is_never_committed_to_history(tmp_path: Path) -> None:
    """A successful retry takes no scrub branch, so whatever it returns is what
    gets committed. If the nudge rode along on the user's message, the next turn
    would read "your previous attempt said this was done" as the user's own
    words -- the poisoned history the confabulation handling exists to prevent,
    arriving by another door."""
    from glados.core.adapters import LLMText, LLMToolCall
    from glados.core.organizer import _UNFINISHED_TURN_NUDGE

    class HalfDoesItLLM:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield LLMToolCall(call_id="c1", server="dunnes", name="view_cart", args={})
            elif self.passes == 2:
                yield LLMText(text="Milk removed from cart.")
            elif self.passes == 3:
                yield LLMToolCall(
                    call_id="c2", server="dunnes", name="remove_from_cart_by_name",
                    args={"name": "milk"},
                )
            else:
                yield LLMText(text="Removed the milk from your cart.")

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm=HalfDoesItLLM(),
        extra_tools=[
            _FakeTool("dunnes", "view_cart"),
            _FakeTool("dunnes", "remove_from_cart_by_name", mutating=True),
        ],
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "show me my cart and then remove the milk")
        await org.flush()

        committed = [m for buf in org._history.values() for m in buf]
        assert committed, "a turn that dispatched tools must commit"
        assert not any(
            _UNFINISHED_TURN_NUDGE in (m.content or "") for m in committed
        ), "the harness nudge must not survive into the conversation"
        assert [m.content for m in committed if m.role == "user"] == [
            "show me my cart and then remove the milk"
        ]
