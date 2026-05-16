"""Unit + E2E coverage for v1 step 2 plumbing: VAD/STT Protocols, fakes,
and the AudioPipeline that ties them to AudioSink and the Organizer."""

from __future__ import annotations

import asyncio
import os
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from glados.audio.pipeline import AudioPipeline
from glados.audio.stt.fake import FakeSTT
from glados.audio.vad.fake import FakeVAD
from glados.core.adapters import VadEnd, VadStart
from glados.core.audio_sink import AudioSink, FrameTooShort


def _pcm(samples: list[int]) -> bytes:
    return b"".join(struct.pack("<h", s) for s in samples)


def _frame(seq: int, samples: list[int]) -> bytes:
    return struct.pack(">I", seq) + _pcm(samples)


# ---- FakeVAD ------------------------------------------------------------


def test_fake_vad_emits_start_then_end_at_threshold() -> None:
    vad = FakeVAD(utterance_samples=4)
    events = vad.feed(_pcm([1, 2, 3, 4]))
    assert [type(e).__name__ for e in events] == ["VadStart", "VadEnd"]
    assert isinstance(events[1], VadEnd)
    assert events[1].pcm == _pcm([1, 2, 3, 4])


def test_fake_vad_buffers_across_calls() -> None:
    vad = FakeVAD(utterance_samples=4)
    e1 = vad.feed(_pcm([1, 2]))
    assert [type(e).__name__ for e in e1] == ["VadStart"]
    e2 = vad.feed(_pcm([3, 4]))
    assert [type(e).__name__ for e in e2] == ["VadEnd"]
    assert isinstance(e2[0], VadEnd) and e2[0].pcm == _pcm([1, 2, 3, 4])


def test_fake_vad_back_to_back_utterances() -> None:
    vad = FakeVAD(utterance_samples=2)
    events = vad.feed(_pcm([1, 2, 3, 4, 5]))
    kinds = [type(e).__name__ for e in events]
    # start, end(1,2), start, end(3,4), start (5 still buffered)
    assert kinds == ["VadStart", "VadEnd", "VadStart", "VadEnd", "VadStart"]
    ends = [e for e in events if isinstance(e, VadEnd)]
    assert ends[0].pcm == _pcm([1, 2])
    assert ends[1].pcm == _pcm([3, 4])


def test_fake_vad_empty_feed_is_noop() -> None:
    vad = FakeVAD(utterance_samples=4)
    assert vad.feed(b"") == []


def test_fake_vad_reset_clears_buffer() -> None:
    vad = FakeVAD(utterance_samples=4)
    vad.feed(_pcm([1, 2]))
    vad.reset()
    assert vad.feed(_pcm([3, 4])) == [VadStart()]  # buffer was cleared


def test_fake_vad_rejects_zero_samples() -> None:
    with pytest.raises(ValueError):
        FakeVAD(utterance_samples=0)


# ---- FakeSTT ------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_stt_returns_canned_text() -> None:
    stt = FakeSTT("hi there")
    assert await stt.transcribe(_pcm([1])) == "hi there"
    assert stt.calls == [_pcm([1])]


@pytest.mark.asyncio
async def test_fake_stt_accepts_callable() -> None:
    stt = FakeSTT(lambda pcm: f"len={len(pcm)}")
    assert await stt.transcribe(_pcm([1, 2, 3])) == "len=6"


# ---- AudioPipeline ------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_invokes_on_utterance_at_vad_boundary(tmp_path: Path) -> None:
    seen: list[str] = []

    async def on_utterance(text: str) -> None:
        seen.append(text)

    pipe = AudioPipeline(
        sink=AudioSink(tmp_path, "desk-ui"),
        vad=FakeVAD(utterance_samples=4),
        stt=FakeSTT("transcribed"),
        on_utterance=on_utterance,
    )
    await pipe.feed_frame(_frame(0, [1, 2]))
    assert seen == []  # below threshold
    await pipe.feed_frame(_frame(1, [3, 4]))
    await pipe.drain()
    assert seen == ["transcribed"]


@pytest.mark.asyncio
async def test_pipeline_skips_empty_transcripts(tmp_path: Path) -> None:
    seen: list[str] = []

    async def on_utterance(text: str) -> None:
        seen.append(text)

    pipe = AudioPipeline(
        sink=None,
        vad=FakeVAD(utterance_samples=2),
        stt=FakeSTT("   "),
        on_utterance=on_utterance,
    )
    await pipe.feed_frame(_frame(0, [1, 2]))
    await pipe.drain()
    assert seen == []


@pytest.mark.asyncio
async def test_pipeline_with_no_sink_skips_wav(tmp_path: Path) -> None:
    pipe = AudioPipeline(
        sink=None,
        vad=FakeVAD(utterance_samples=2),
        stt=FakeSTT(""),
        on_utterance=lambda _t: asyncio.sleep(0),
    )
    await pipe.feed_frame(_frame(0, [1, 2]))
    await pipe.close()
    assert not (tmp_path / "audio").exists()


@pytest.mark.asyncio
async def test_pipeline_rejects_short_frame() -> None:
    pipe = AudioPipeline(
        sink=None,
        vad=FakeVAD(utterance_samples=2),
        stt=FakeSTT(""),
        on_utterance=lambda _t: asyncio.sleep(0),
    )
    with pytest.raises(FrameTooShort):
        await pipe.feed_frame(b"\x00\x00\x00")


@pytest.mark.asyncio
async def test_pipeline_logs_stt_errors_and_does_not_fire(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[str] = []

    class BrokenSTT:
        async def transcribe(self, pcm: bytes) -> str:
            raise RuntimeError("model exploded")

    pipe = AudioPipeline(
        sink=None,
        vad=FakeVAD(utterance_samples=2),
        stt=BrokenSTT(),
        on_utterance=lambda t: seen.append(t) or asyncio.sleep(0),
    )
    with caplog.at_level("ERROR", logger="glados.audio.pipeline"):
        await pipe.feed_frame(_frame(0, [1, 2]))
        await pipe.drain()
    assert seen == []
    assert any("STT.transcribe failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_pipeline_close_drains_in_flight(tmp_path: Path) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    results: list[str] = []

    class SlowSTT:
        async def transcribe(self, pcm: bytes) -> str:
            started.set()
            await finish.wait()
            return "late"

    async def on_utterance(text: str) -> None:
        results.append(text)

    pipe = AudioPipeline(
        sink=None,
        vad=FakeVAD(utterance_samples=2),
        stt=SlowSTT(),
        on_utterance=on_utterance,
    )
    await pipe.feed_frame(_frame(0, [1, 2]))
    await started.wait()
    close_task = asyncio.create_task(pipe.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    finish.set()
    await close_task
    assert results == ["late"]


# ---- End-to-end through the FastAPI websocket ---------------------------


@pytest.fixture(scope="module")
def http_client() -> TestClient:
    os.environ["GLADOS_CONFIG_DIR"] = str(Path(__file__).parent.parent / "configs")
    os.environ["GLADOS_LLM_BACKEND"] = "fake"
    from glados.core.server import app

    return TestClient(app)


def test_e2e_audio_drives_fake_transcript_into_organizer(
    http_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from glados.audio.vad.fake import FakeVAD as RealFakeVAD
    from glados.core import server as srv

    monkeypatch.setattr(srv._glados_cfg.server, "traces_dir", tmp_path)
    # Shrink the fake utterance to a single frame so the boundary fires
    # without flooding the test with PCM.
    monkeypatch.setattr(
        srv, "_build_vad", lambda _cfg: RealFakeVAD(utterance_samples=4)
    )
    monkeypatch.setattr(srv, "_stt", FakeSTT("what time is it?"))

    frame = _frame(0, [1, 2, 3, 4])
    with http_client.websocket_connect("/ws/v1") as ws:
        ws.send_json(
            {
                "type": "hello",
                "client_id": "desk-ui",
                "room_id": "desk",
                "role": "ui",
                "token": "dev-token-desk",
            }
        )
        ws.send_bytes(frame)
        # FakeLLM + NowTool sequence: welcome, tool_call, tool_result,
        # assistant_delta, done.
        msgs = [ws.receive_json() for _ in range(5)]

    types = [m["type"] for m in msgs]
    assert types == ["welcome", "tool_call", "tool_result", "assistant_delta", "done"]
    assert msgs[1]["server"] == "time" and msgs[1]["name"] == "now"
