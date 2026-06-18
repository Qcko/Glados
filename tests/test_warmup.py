"""Warmup hook: STT and TTS are exercised once on server boot so the
first real utterance doesn't pay model-load latency."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glados.audio.stt.fake import FakeSTT
from glados.audio.tts.fake import FakeTTS
from glados.core.server import _WARMUP_PCM, _WARMUP_TEXT, _warmup


@pytest.mark.asyncio
async def test_warmup_exercises_stt_and_tts() -> None:
    stt = FakeSTT()
    tts = FakeTTS()
    await _warmup(stt, tts)
    assert stt.calls == [_WARMUP_PCM]
    assert tts.calls == [_WARMUP_TEXT]


# ---- LLM warm-up + first-turn gate (SESSION 2026-06-15) -----------------

from pydantic import BaseModel

from glados.brain.llm.fake import FakeLLM
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry
from glados.servers.time_server import NowTool


def _organizer(llm, tmp: Path, bindings=None):
    bindings = bindings or []
    by_id = {b.client_id: b for b in bindings}
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    mcp = MCPRegistry()
    mcp.register(NowTool())
    org = Organizer(
        llm=llm,
        tts=None,
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
    )
    return org, sink


@pytest.mark.asyncio
async def test_warm_up_llm_exercises_tool_and_freeform_paths(tmp_path: Path) -> None:
    """The boot warm-up fires TWO inferences: one with the FULL registered tool
    list (primes tool-selection — the cold-model fabrication bug) and one
    tool-free (primes free-form generation — the cold-model language-drift bug),
    then flips the readiness gate."""

    class RecordingLLM:
        def __init__(self) -> None:
            self.calls: list[tuple[list, list]] = []

        async def chat(self, messages, tools):
            self.calls.append((messages, tools))
            if False:  # make this an async generator that yields nothing
                yield

    llm = RecordingLLM()
    org, _ = _organizer(llm, tmp_path)
    org.expect_llm_warmup()  # the server marks it cold before warming
    assert not org._llm_warmed.is_set()

    await org.warm_up_llm()

    assert org._llm_warmed.is_set()
    assert len(llm.calls) == 2
    tool_messages, tool_specs = llm.calls[0]
    assert tool_messages[-1].role == "user" and "time" in tool_messages[-1].content.lower()
    assert any(t.qualified == "time.now" for t in tool_specs)
    # Second shot is free-form: no tools offered, so it primes plain generation.
    free_messages, free_specs = llm.calls[1]
    assert free_specs == []
    assert free_messages[-1].role == "user"


@pytest.mark.asyncio
async def test_warm_up_llm_sets_gate_even_on_failure(tmp_path: Path, caplog) -> None:
    """A warm-up that blows up must still release the gate — otherwise every
    real turn would deadlock behind a model that never warmed."""

    class BrokenLLM:
        async def chat(self, messages, tools):
            raise RuntimeError("ollama unreachable")
            yield  # pragma: no cover

    org, _ = _organizer(BrokenLLM(), tmp_path)
    org.expect_llm_warmup()
    with caplog.at_level(logging.ERROR, logger="glados.core.organizer"):
        await org.warm_up_llm()
    assert "LLM warm-up failed" in caplog.text
    assert org._llm_warmed.is_set()


@pytest.mark.asyncio
async def test_first_turn_waits_for_warm_up_then_proceeds(tmp_path: Path, caplog) -> None:
    """A turn that arrives while the model is cold is held until warm_up_llm
    completes (not dropped), and the race is recorded as an llm_cold_turn trace
    event for observability."""
    binding = ClientBinding(
        client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
    )
    org, sink = _organizer(FakeLLM(), tmp_path, [binding])
    org.expect_llm_warmup()  # cold

    with caplog.at_level(logging.WARNING, logger="glados.core.organizer"):
        await org.handle_user_text("desk-ui", "hello there")
        await asyncio.sleep(0.05)  # let the worker dequeue and hit the gate
        # Gated: nothing broadcast yet (the gate sits before Welcome).
        assert [m["type"] for _, m in sink] == []

        await org.warm_up_llm()  # releases the gate
        await org.flush()

    types = [m["type"] for _, m in sink]
    assert "welcome" in types and "done" in types
    assert "before LLM warm-up finished" in caplog.text
    cold = [
        line
        for f in tmp_path.glob("*.jsonl")
        for line in f.read_text(encoding="utf-8").splitlines()
        if "llm_cold_turn" in line
    ]
    assert cold, "expected an llm_cold_turn trace event"


@pytest.mark.asyncio
async def test_warmup_swallows_stt_failure(caplog) -> None:
    class BrokenSTT:
        async def transcribe(self, pcm: bytes) -> str:
            raise RuntimeError("model unavailable")

    tts = FakeTTS()
    with caplog.at_level(logging.ERROR, logger="glados.core.server"):
        await _warmup(BrokenSTT(), tts)
    assert "STT warmup failed" in caplog.text
    # TTS must still be exercised even when STT blows up — the two
    # backends are independent and one shouldn't gate the other.
    assert tts.calls == [_WARMUP_TEXT]


@pytest.mark.asyncio
async def test_warmup_swallows_tts_failure(caplog) -> None:
    class BrokenTTS:
        async def synthesize(self, text: str):
            raise RuntimeError("voice missing")
            yield  # pragma: no cover  (make this an async generator)

    stt = FakeSTT()
    with caplog.at_level(logging.ERROR, logger="glados.core.server"):
        await _warmup(stt, BrokenTTS())
    assert "TTS warmup failed" in caplog.text
    assert stt.calls == [_WARMUP_PCM]


def test_startup_event_schedules_warmup() -> None:
    """End-to-end: entering the TestClient context manager fires the
    app's lifespan startup, which spawns the warmup task. The lifespan
    reads stt/tts from `app.state` at startup time, so the test can
    swap them after building the app and before the lifespan fires."""
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    from glados.core.server import build_app

    app = build_app()
    stt = FakeSTT()
    tts = FakeTTS()
    app.state.stt = stt
    app.state.tts = tts

    # /healthz is loopback-gated; present as a loopback caller.
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200

    assert stt.calls == [_WARMUP_PCM]
    assert tts.calls == [_WARMUP_TEXT]
