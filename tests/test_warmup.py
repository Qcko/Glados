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


def test_startup_event_schedules_warmup(monkeypatch) -> None:
    """End-to-end via TestClient: entering the context manager fires
    FastAPI's startup, which spawns the warmup task. After the first
    request lands, the task has had a tick to run and both backends
    have been called."""
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    from glados.core import server as srv

    stt = FakeSTT()
    tts = FakeTTS()
    monkeypatch.setattr(srv, "_stt", stt)
    monkeypatch.setattr(srv, "_tts", tts)

    with TestClient(srv.app) as client:
        # First HTTP request gives the scheduled warmup task a chance to
        # run. _on_startup creates the task; the loop runs it before the
        # next await yields back.
        resp = client.get("/healthz")
        assert resp.status_code == 200

    assert stt.calls == [_WARMUP_PCM]
    assert tts.calls == [_WARMUP_TEXT]
