"""v1 step 4: barge-in / interrupt.

The Organizer registers each in-flight turn by session_id so a follow-up
Interrupt cancels just that turn, broadcasts Cancelled to the room, and
suppresses Done. Cross-room interrupts are rejected silently.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from glados.core.adapters import LLMText, TtsChunkOut
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry


class SlowLLM:
    """Yields one delta, then hangs forever -- guaranteed cancel point."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def chat(self, messages, tools):
        yield LLMText(text="hello ")
        self.entered.set()
        await asyncio.sleep(3600)


class SlowTTS:
    """Yields one chunk, then hangs -- cancel point inside _speak."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def synthesize(self, text: str):
        yield TtsChunkOut(pcm=b"\x00\x00", sample_rate=22_050)
        self.entered.set()
        await asyncio.sleep(3600)


@asynccontextmanager
async def _make_organizer(
    bindings: list[ClientBinding], tmp: Path, llm, tts=None
):
    """Yields `(organizer, sink)` and guarantees `organizer.close()` so
    room-queue workers don't leak between tests."""
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    by_id = {b.client_id: b for b in bindings}
    org = Organizer(
        llm=llm,
        tts=tts,
        mcp=MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
    )
    try:
        yield org, sink
    finally:
        await org.close()


@pytest.mark.asyncio
async def test_interrupt_cancels_inflight_turn(tmp_path: Path) -> None:
    llm = SlowLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello")  # enqueue
        await llm.entered.wait()  # worker dequeued; LLM is hanging mid-turn

        welcome = next(m for _, m in sink if m["type"] == "welcome")
        sid = welcome["session_id"]

        await org.handle_interrupt("desk-ui", sid)
        await org.flush()  # wait for the cancelled turn to fully drain

        types = [m["type"] for _, m in sink]
        assert "cancelled" in types
        assert "done" not in types
        cancelled = next(m for _, m in sink if m["type"] == "cancelled")
        assert cancelled["session_id"] == sid


@pytest.mark.asyncio
async def test_interrupt_for_unknown_session_is_noop(tmp_path: Path) -> None:
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        SlowLLM(),
    ) as (org, sink):
        await org.handle_interrupt("desk-ui", "desk:qcko:nosuch")
        assert sink == []


@pytest.mark.asyncio
async def test_interrupt_from_foreign_room_rejected(tmp_path: Path, caplog) -> None:
    llm = SlowLLM()
    async with _make_organizer(
        [
            ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"),
            ClientBinding(client_id="desk2-ui", room_id="desk2", role="ui", default_user="anna"),
        ],
        tmp_path,
        llm,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello")
        await llm.entered.wait()

        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]

        with caplog.at_level("WARNING"):
            await org.handle_interrupt("desk2-ui", sid)
        assert "interrupt rejected" in caplog.text
        assert not any(m["type"] == "cancelled" for _, m in sink)

        await org.handle_interrupt("desk-ui", sid)
        await org.flush()
        assert any(m["type"] == "cancelled" for _, m in sink)


@pytest.mark.asyncio
async def test_interrupt_during_tts_emits_cancelled(tmp_path: Path) -> None:
    from glados.brain.llm.fake import FakeLLM

    tts = SlowTTS()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        FakeLLM(),
        tts=tts,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await tts.entered.wait()  # past the LLM phase, inside _speak

        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        await org.handle_interrupt("desk-ui", sid)
        await org.flush()

        types = [m["type"] for _, m in sink]
        assert "tts_chunk" in types
        assert types[-1] == "cancelled"
        assert "done" not in types


@pytest.mark.asyncio
async def test_interrupt_after_done_is_noop(tmp_path: Path) -> None:
    from glados.brain.llm.fake import FakeLLM

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        FakeLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello")
        await org.flush()
        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        before = len(sink)
        await org.handle_interrupt("desk-ui", sid)
        assert len(sink) == before


# ---- End-to-end via FastAPI TestClient ---------------------------------


@pytest.fixture(scope="module")
def http_client():
    """See test_audio_pipeline.py for the rationale on context-managing
    TestClient -- needed so each module's room workers are torn down before
    the next module's TestClient creates its own event loop."""
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    from glados.core.server import app

    with TestClient(app) as client:
        yield client


def test_e2e_interrupt_emits_cancelled(http_client: TestClient, monkeypatch) -> None:
    # State lives on `app.state.organizer` now, not on the module.
    slow = SlowLLM()
    monkeypatch.setattr(http_client.app.state.organizer, "llm", slow)

    with http_client.websocket_connect("/ws/v1") as ws:
        ws.send_json({
            "type": "hello",
            "client_id": "desk-ui",
            "room_id": "desk",
            "role": "ui",
            "token": "dev-token-desk",
        })
        ws.send_json({"type": "user_text", "text": "long answer please"})

        welcome = ws.receive_json()
        assert welcome["type"] == "welcome"
        sid = welcome["session_id"]
        transcript = ws.receive_json()
        assert transcript["type"] == "user_transcript"
        first_delta = ws.receive_json()
        assert first_delta["type"] == "assistant_delta"

        ws.send_json({"type": "interrupt", "session_id": sid})
        cancelled = ws.receive_json()
        assert cancelled["type"] == "cancelled"
        assert cancelled["session_id"] == sid
