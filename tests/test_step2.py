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
async def _make_organizer(bindings: list[ClientBinding], tmp: Path, llm=None):
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
    organizer = Organizer(
        llm=llm if llm is not None else FakeLLM(),
        mcp=reg,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=binding_for,
        clients_in_room=in_room,
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
        await org.handle_user_text("desk-ui", "add bananas")
        await org.flush()
        await org.handle_user_text("desk-ui", "actually eggs instead")
        await org.flush()

    first_seen, second_seen = llm.seen
    # First turn: just system + the user message, no prior history.
    assert first_seen == [("system", org._system_prompt), ("user", "add bananas")]
    # Second turn: the first exchange is replayed before the new user message.
    assert ("user", "add bananas") in second_seen
    assert ("assistant", "acknowledged") in second_seen
    assert second_seen[-1] == ("user", "actually eggs instead")


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
