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
from glados.mcp.registry import CallEnvelope, MCPRegistry
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


# ---- Organizer ----------------------------------------------------------


@asynccontextmanager
async def _make_organizer(bindings: list[ClientBinding], tmp: Path):
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
        llm=FakeLLM(),
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
        assert types == ["welcome", "tool_call", "tool_result", "assistant_delta", "done"]
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


# ---- Trace writer -------------------------------------------------------


def test_session_registry_opens_distinct_ids() -> None:
    reg = SessionRegistry()
    a = reg.get_or_open("desk", "qcko")
    b = reg.get_or_open("desk", "qcko")
    assert a.session_id != b.session_id
    assert reg.latest("desk", "qcko").session_id == b.session_id


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
        msgs = [ws.receive_json() for _ in range(6)]
    assert [m["type"] for m in msgs] == [
        "welcome",
        "tool_call",
        "tool_result",
        "assistant_delta",
        "tts_chunk",
        "done",
    ]
    assert msgs[2]["ok"] is True
    assert "iso" in msgs[2]["content"]
    assert msgs[4]["seq"] == 0
    assert msgs[4]["sample_rate"] == 22_050


def test_e2e_audio_frame_writes_wav(http_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import struct
    import wave
    from glados.core import server as srv

    monkeypatch.setattr(srv._glados_cfg.server, "traces_dir", tmp_path)
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
