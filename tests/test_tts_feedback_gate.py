"""TTS feedback gate (server-side mic-mute layer).

While a room's speaker is mid-TTS — or within `tts_cooldown_s` of
finishing — non-barge-in audio transcripts from that room are dropped
to prevent the speaker→mic loop from self-triggering a new turn. Barge-
in regex still passes through so voice-driven interrupt works.

Pairs with the browser's `echoCancellation: true` (mic.ts) as a second
layer — needed for external speakers and Pi clients without
`webrtc-audio-processing`.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.audio.tts.fake import FakeTTS
from glados.brain.llm.fake import FakeLLM
from glados.core.adapters import LLMText, TtsChunkOut
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry


class _GatedTTS:
    """Yields chunks paced by an external Event. Lets the test arrange
    'mid-TTS' state precisely."""

    def __init__(self) -> None:
        self.released = asyncio.Event()
        self.entered = asyncio.Event()

    async def synthesize(self, text: str):
        yield TtsChunkOut(pcm=b"\x00\x00", sample_rate=22_050)
        self.entered.set()
        await self.released.wait()
        yield TtsChunkOut(pcm=b"\x00\x00", sample_rate=22_050)


@asynccontextmanager
async def _make_organizer(bindings, tmp, *, tts=None, llm=None, tts_cooldown_s=0.200):
    sink: list[tuple[str, dict]] = []

    async def send(client_id, msg):
        sink.append((client_id, msg.model_dump()))

    by_id = {b.client_id: b for b in bindings}
    org = Organizer(
        llm=llm if llm is not None else FakeLLM(),
        tts=tts,
        mcp=MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        tts_cooldown_s=tts_cooldown_s,
    )
    try:
        yield org, sink
    finally:
        await org.close()


_DESK = ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")
_KITCHEN = ClientBinding(client_id="kit-ui", room_id="kitchen", role="ui", default_user="anna")


@pytest.mark.asyncio
async def test_gate_drops_non_barge_in_audio_while_speaking(tmp_path: Path) -> None:
    """While `_speaking_rooms` is set for a room, an arbitrary
    transcript from that room must NOT open a new turn — it's almost
    certainly the speaker→mic loop transcribing GLaDOS's own voice."""
    tts = _GatedTTS()
    async with _make_organizer([_DESK], tmp_path, tts=tts) as (org, sink):
        # Fire a turn so the room goes into TTS.
        await org.handle_user_text("desk-ui", "hello there")
        await tts.entered.wait()
        assert "desk" in org._speaking_rooms

        before = len(sink)
        # The TTS audio "loops back" as a transcript. Without the gate
        # this would open a new turn.
        await org.handle_audio_text("desk-ui", "echo: hello there")
        # No new welcome, no new anything — the gate dropped it.
        assert len(sink) == before, (
            f"gate must drop the looped-back transcript, sink grew by "
            f"{len(sink) - before}: {sink[before:]}"
        )

        # Release TTS so the turn can finish.
        tts.released.set()
        await org.flush()


@pytest.mark.asyncio
async def test_gate_passes_barge_in_while_speaking(tmp_path: Path) -> None:
    """The whole point of the gate having a barge-in exception is that
    a voice "stop" mid-TTS still interrupts. If the gate dropped
    barge-in, hands-free interrupt would silently break."""
    tts = _GatedTTS()
    async with _make_organizer([_DESK], tmp_path, tts=tts) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await tts.entered.wait()
        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]

        await org.handle_audio_text("desk-ui", "stop")
        # Voice "stop" cancelled the active turn.
        tts.released.set()  # in case turn is awaiting TTS
        await org.flush()

        types = [m["type"] for _, m in sink]
        assert "cancelled" in types, (
            f"barge-in must still cancel while TTS is mid-flight, got "
            f"{types}"
        )
        assert next(m for _, m in sink if m["type"] == "cancelled")["session_id"] == sid


@pytest.mark.asyncio
async def test_gate_drops_audio_during_cooldown(tmp_path: Path) -> None:
    """Even after TTS finishes, the room stays gated for
    `tts_cooldown_s` to catch the audio tail (decay, reverb, late
    buffered frames) that can race the Done broadcast."""
    async with _make_organizer(
        [_DESK], tmp_path, tts=FakeTTS(), tts_cooldown_s=0.500
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()  # turn complete; cooldown is now active

        assert "desk" not in org._speaking_rooms
        assert org._room_mic_gated("desk"), (
            "room must still be gated in the cooldown window immediately "
            "after TTS ends"
        )

        before = len(sink)
        await org.handle_audio_text("desk-ui", "leftover echo")
        assert len(sink) == before, "cooldown must drop the late echo"


@pytest.mark.asyncio
async def test_gate_releases_after_cooldown(tmp_path: Path) -> None:
    """Once the cooldown expires, the next utterance is processed
    normally — the gate must be transparent during silence."""
    async with _make_organizer(
        [_DESK], tmp_path, tts=FakeTTS(), tts_cooldown_s=0.050
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()

        # Wait past the cooldown.
        await asyncio.sleep(0.080)
        assert not org._room_mic_gated("desk")

        before = len(sink)
        await org.handle_audio_text("desk-ui", "fresh question")
        await org.flush()
        types = [m["type"] for _, m in sink[before:]]
        assert "welcome" in types and "done" in types, (
            f"post-cooldown utterance must run normally, got {types}"
        )


@pytest.mark.asyncio
async def test_gate_is_per_room(tmp_path: Path) -> None:
    """Speaker A speaking in the kitchen must not gate the desk's mic.
    The gate is keyed by room, not global.

    Tested at the predicate level (`_room_mic_gated`) rather than by
    racing two real turns — the shared `_GatedTTS` would block both
    turns on the same release event, and `flush()` would deadlock."""
    tts = _GatedTTS()
    async with _make_organizer(
        [_DESK, _KITCHEN], tmp_path, tts=tts
    ) as (org, sink):
        # Drive kitchen into mid-TTS.
        await org.handle_user_text("kit-ui", "hello there")
        await tts.entered.wait()
        assert "kitchen" in org._speaking_rooms
        assert org._room_mic_gated("kitchen"), "kitchen must be gated"
        assert not org._room_mic_gated("desk"), (
            "kitchen being mid-TTS must NOT gate the desk room"
        )

        # An audio transcript arriving from the desk room while kitchen
        # is mid-TTS must reach `handle_user_text` (i.e. produce an
        # enqueue), not be dropped. We don't drain the queue here because
        # the desk turn would also block on the shared TTS — observing
        # `queue_depth("desk") == 1` proves the gate let it through.
        await org.handle_audio_text("desk-ui", "what time is it")
        assert org._queues.queue_depth("desk") == 1, (
            "desk audio must NOT be gated by kitchen's TTS — turn should "
            "be enqueued"
        )

        # Release kitchen TTS so both turns can drain and the context
        # manager's close() doesn't have to forcibly cancel.
        tts.released.set()
        await org.flush()


@pytest.mark.asyncio
async def test_gate_clears_on_cancellation(tmp_path: Path) -> None:
    """When TTS is cancelled (barge-in or UI interrupt mid-speech),
    `_speaking_rooms` must be cleared in the finally — otherwise the
    gate would stick on forever after a cancelled turn."""

    class HangingTTS:
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def synthesize(self, text: str):
            yield TtsChunkOut(pcm=b"\x00\x00", sample_rate=22_050)
            self.entered.set()
            await asyncio.sleep(3600)

    tts = HangingTTS()
    async with _make_organizer(
        [_DESK], tmp_path, tts=tts, tts_cooldown_s=0.0
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await tts.entered.wait()
        assert "desk" in org._speaking_rooms

        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        await org.handle_interrupt("desk-ui", sid)
        await org.flush()

        # Finally-block must have cleared the speaking flag AND stamped
        # the cooldown timestamp. With cooldown=0, the gate should be
        # off entirely after cancel.
        assert "desk" not in org._speaking_rooms, (
            "cancellation must clear `_speaking_rooms` in the finally"
        )
        assert not org._room_mic_gated("desk"), (
            "gate must be off after cancel + cooldown=0"
        )
