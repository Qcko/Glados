"""TTS unit tests + Organizer integration with FakeTTS.

The Piper smoke test (real model load + voice download) is gated behind
GLADOS_PIPER_SMOKE=1 so normal runs stay fast and offline."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.audio.tts.fake import FakeTTS
from glados.brain.llm.fake import FakeLLM
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry
from glados.servers.time_server import NowTool


# ---- FakeTTS ------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_tts_yields_one_chunk_per_text() -> None:
    tts = FakeTTS(sample_rate=22_050, samples_per_chunk=1_000)
    chunks = [c async for c in tts.synthesize("hello")]
    assert len(chunks) == 1
    assert chunks[0].sample_rate == 22_050
    assert len(chunks[0].pcm) == 2_000  # 1000 int16 samples
    assert tts.calls == ["hello"]


@pytest.mark.asyncio
async def test_fake_tts_skips_empty_text() -> None:
    tts = FakeTTS()
    chunks = [c async for c in tts.synthesize("")]
    assert chunks == []
    assert tts.calls == [""]


# ---- Organizer + FakeTTS -----------------------------------------------


def _make_organizer_with_tts(tts, tmp_path: Path) -> tuple[Organizer, list]:
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    bindings = [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")]
    by_id = {b.client_id: b for b in bindings}

    reg = MCPRegistry()
    reg.register(NowTool())
    organizer = Organizer(
        llm=FakeLLM(),
        tts=tts,
        mcp=reg,
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
    )
    return organizer, sink


@pytest.mark.asyncio
async def test_organizer_emits_tts_chunk_after_assistant_delta(tmp_path: Path) -> None:
    tts = FakeTTS(sample_rate=22_050, samples_per_chunk=2)
    org, sink = _make_organizer_with_tts(tts, tmp_path)

    await org.handle_user_text("desk-ui", "hello there")

    types = [m["type"] for _, m in sink]
    assert types == ["welcome", "assistant_delta", "tts_chunk", "done"]
    tts_msg = next(m for _, m in sink if m["type"] == "tts_chunk")
    assert tts_msg["seq"] == 0
    assert tts_msg["sample_rate"] == 22_050
    assert len(base64.b64decode(tts_msg["pcm_b64"])) == 4  # 2 int16 samples
    assert tts.calls == ["echo: hello there"]


@pytest.mark.asyncio
async def test_organizer_skips_tts_when_no_text(tmp_path: Path) -> None:
    """Empty final_text (rare — would need an LLM that emits no text and no
    tool calls) must not blow up the turn or emit empty tts_chunks."""
    from glados.core.adapters import LLMText

    class SilentLLM:
        async def chat(self, messages, tools):
            yield LLMText(text="")

    tts = FakeTTS()
    org, sink = _make_organizer_with_tts(tts, tmp_path)
    org.llm = SilentLLM()

    await org.handle_user_text("desk-ui", "hi")
    types = [m["type"] for _, m in sink]
    assert "tts_chunk" not in types
    # FakeTTS.calls is appended on first iteration; for a guarded skip we
    # never iterate, so the empty call is filtered before synthesize is
    # reached.
    assert tts.calls == []


@pytest.mark.asyncio
async def test_organizer_with_no_tts_still_completes(tmp_path: Path) -> None:
    org, sink = _make_organizer_with_tts(tts=None, tmp_path=tmp_path)
    await org.handle_user_text("desk-ui", "hello there")
    types = [m["type"] for _, m in sink]
    assert types == ["welcome", "assistant_delta", "done"]


# ---- PiperTTS smoke (env-gated) ----------------------------------------


@pytest.mark.skipif(
    os.environ.get("GLADOS_PIPER_SMOKE") != "1",
    reason="set GLADOS_PIPER_SMOKE=1 to run (downloads ~110 MB voice on first run)",
)
@pytest.mark.asyncio
async def test_piper_smoke_synthesizes_pcm() -> None:
    from glados.audio.tts.piper import PiperTTS

    tts = PiperTTS()
    chunks = [c async for c in tts.synthesize("Hello world.")]
    assert chunks, "piper produced no chunks"
    assert all(len(c.pcm) > 0 for c in chunks)
    assert all(c.sample_rate == 22_050 for c in chunks)
