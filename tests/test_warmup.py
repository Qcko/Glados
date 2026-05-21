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

    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200

    assert stt.calls == [_WARMUP_PCM]
    assert tts.calls == [_WARMUP_TEXT]
