"""TTS unit tests + Organizer integration with FakeTTS.

The Piper smoke test (real model load + voice download) is gated behind
GLADOS_PIPER_SMOKE=1 so normal runs stay fast and offline."""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def _make_organizer_with_tts(tts, tmp_path: Path):
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
    try:
        yield organizer, sink
    finally:
        await organizer.close()


@pytest.mark.asyncio
async def test_organizer_emits_tts_chunk_after_assistant_delta(tmp_path: Path) -> None:
    tts = FakeTTS(sample_rate=22_050, samples_per_chunk=2)
    async with _make_organizer_with_tts(tts, tmp_path) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()

        types = [m["type"] for _, m in sink]
        assert types == [
            "welcome", "user_transcript", "assistant_delta", "tts_chunk", "done",
        ]
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
    async with _make_organizer_with_tts(tts, tmp_path) as (org, sink):
        org.llm = SilentLLM()

        await org.handle_user_text("desk-ui", "hi")
        await org.flush()
        types = [m["type"] for _, m in sink]
        assert "tts_chunk" not in types
        # FakeTTS.calls is appended on first iteration; for a guarded skip we
        # never iterate, so the empty call is filtered before synthesize is
        # reached.
        assert tts.calls == []


@pytest.mark.asyncio
async def test_organizer_with_no_tts_still_completes(tmp_path: Path) -> None:
    async with _make_organizer_with_tts(tts=None, tmp_path=tmp_path) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()
        types = [m["type"] for _, m in sink]
        assert types == ["welcome", "user_transcript", "assistant_delta", "done"]


# ---- PiperTTS lazy-load (no real piper / onnx required) ----------------
#
# Construction must be cheap. The voice download (~110 MB cold) and the
# onnx model load happen on first `synthesize()` via background threads,
# so the asyncio loop is never blocked. Tests stub both the `piper`
# module and `_ensure_voice` so they run offline and instantly.


@pytest.fixture
def _fake_piper(monkeypatch):
    """Inject a fake `piper` module into `sys.modules` so
    `from piper import PiperVoice` (deferred inside `_ensure_loaded`)
    finds our stub. Also stubs `_ensure_voice` to skip the disk check
    and `PiperVoice.load` to count calls.

    Returns a small handle exposing call counts on both."""
    import sys
    import types as _types

    from glados.audio.tts import piper as piper_mod
    from glados.core.adapters import TtsChunkOut

    class _FakeVoice:
        def synthesize(self, text):
            class _C:
                audio_int16_bytes = b"\x00\x00"
                sample_rate = 22_050

            yield _C()

    class _CountingLoader:
        load_calls = 0

        @classmethod
        def load(cls, _onnx_path):
            cls.load_calls += 1
            return _FakeVoice()

    fake_module = _types.ModuleType("piper")
    fake_module.PiperVoice = _CountingLoader
    monkeypatch.setitem(sys.modules, "piper", fake_module)

    ensure_calls = []

    def _fake_ensure(voice, voices_dir):
        ensure_calls.append((voice, voices_dir))
        return Path(voices_dir) / f"{voice}.onnx"

    monkeypatch.setattr(piper_mod, "_ensure_voice", _fake_ensure)

    class _Handle:
        @property
        def ensure_calls(self):
            return ensure_calls

        @property
        def load_calls(self):
            return _CountingLoader.load_calls

    return _Handle()


def test_piper_construction_does_no_work(_fake_piper, tmp_path: Path) -> None:
    """`__init__` must not download or load — otherwise the asyncio loop
    is blocked at app-build time. The whole point of this slice."""
    from glados.audio.tts.piper import PiperTTS

    PiperTTS(voice="en_GB-cori-high", voices_dir=tmp_path)
    assert _fake_piper.ensure_calls == [], "construction must not check disk"
    assert _fake_piper.load_calls == 0, "construction must not load model"


@pytest.mark.asyncio
async def test_piper_first_synthesize_triggers_load(_fake_piper, tmp_path: Path) -> None:
    from glados.audio.tts.piper import PiperTTS

    tts = PiperTTS(voice="en_GB-cori-high", voices_dir=tmp_path)
    chunks = [c async for c in tts.synthesize("hi")]
    assert chunks, "fake voice yielded no chunks"
    assert len(_fake_piper.ensure_calls) == 1
    assert _fake_piper.load_calls == 1


@pytest.mark.asyncio
async def test_piper_repeat_synthesize_does_not_reload(_fake_piper, tmp_path: Path) -> None:
    from glados.audio.tts.piper import PiperTTS

    tts = PiperTTS(voice="en_GB-cori-high", voices_dir=tmp_path)
    async for _ in tts.synthesize("hi"):
        pass
    async for _ in tts.synthesize("there"):
        pass
    # Both calls share the loaded voice — no second load.
    assert _fake_piper.load_calls == 1
    assert len(_fake_piper.ensure_calls) == 1


@pytest.mark.asyncio
async def test_piper_concurrent_first_synth_serialises_load(
    _fake_piper, tmp_path: Path
) -> None:
    """Two concurrent first-synthesize calls must load the voice exactly
    once. Without the asyncio.Lock + double-check, both would race past
    the `if self._voice is not None` guard and call `PiperVoice.load`
    twice — wasted work, and on cold start, two parallel ~110 MB
    downloads."""
    import asyncio

    from glados.audio.tts.piper import PiperTTS

    tts = PiperTTS(voice="en_GB-cori-high", voices_dir=tmp_path)

    async def drain(text):
        async for _ in tts.synthesize(text):
            pass

    await asyncio.gather(drain("one"), drain("two"))
    assert _fake_piper.load_calls == 1, (
        f"expected exactly one load under concurrent first-synth, "
        f"got {_fake_piper.load_calls}"
    )
    assert len(_fake_piper.ensure_calls) == 1


@pytest.mark.asyncio
async def test_piper_empty_text_does_not_load(_fake_piper, tmp_path: Path) -> None:
    """`synthesize("")` returns immediately without touching the voice —
    no point in downloading 110 MB to say nothing."""
    from glados.audio.tts.piper import PiperTTS

    tts = PiperTTS(voice="en_GB-cori-high", voices_dir=tmp_path)
    chunks = [c async for c in tts.synthesize("")]
    assert chunks == []
    assert _fake_piper.load_calls == 0
    assert _fake_piper.ensure_calls == []


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
