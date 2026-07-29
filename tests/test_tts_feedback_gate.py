"""TTS feedback gate (server-side mic-mute layer).

While a room's speaker is mid-TTS -- or within `tts_cooldown_s` of
finishing -- non-barge-in audio transcripts from that room are dropped
to prevent the speaker->mic loop from self-triggering a new turn. Barge-
in regex still passes through so voice-driven interrupt works.

Pairs with the browser's `echoCancellation: true` (mic.ts) as a second
layer -- needed for external speakers and Pi clients without
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
from glados.core.organizer import Organizer, _TtsGate
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
async def _make_organizer(
    bindings, tmp, *, tts=None, llm=None, tts_cooldown_s=0.200, gate_drain_margin_s=0.0
):
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
        gate_drain_margin_s=gate_drain_margin_s,
    )
    try:
        yield org, sink
    finally:
        await org.close()


_DESK = ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")
_KITCHEN = ClientBinding(client_id="kit-ui", room_id="kitchen", role="ui", default_user="anna")
# A speaker in the room is what makes the post-send DRAINING gate engage (a
# room with no speaker plays nothing, so the gate opens immediately after send).
_DESK_SPK = ClientBinding(
    client_id="desk-spk", room_id="desk", role="speaker", default_user="qcko"
)


@pytest.mark.asyncio
async def test_gate_drops_non_barge_in_audio_while_speaking(tmp_path: Path) -> None:
    """While `_speaking_rooms` is set for a room, an arbitrary
    transcript from that room must NOT open a new turn -- it's almost
    certainly the speaker->mic loop transcribing GLaDOS's own voice."""
    tts = _GatedTTS()
    async with _make_organizer([_DESK], tmp_path, tts=tts) as (org, sink):
        # Fire a turn so the room goes into TTS.
        await org.handle_user_text("desk-ui", "hello there")
        await tts.entered.wait()
        assert org._tts_gate["desk"].phase == "sending"
        assert org._room_mic_gated("desk")

        before = len(sink)
        # The TTS audio "loops back" as a transcript. Without the gate
        # this would open a new turn.
        await org.handle_audio_text("desk-ui", "echo: hello there")
        # No new welcome, no new anything -- the gate dropped it.
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
async def test_gate_drops_audio_after_send_while_draining(tmp_path: Path) -> None:
    """After the server finishes SENDING, the room stays gated while the reply
    is (estimated to be) still playing on the speaker -- this is the fix for the
    feedback loop, where a long reply outlasted the old fixed cooldown."""
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path, tts=FakeTTS(), tts_cooldown_s=0.500,
        gate_drain_margin_s=0.500,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()  # turn complete; draining window now active

        assert org._tts_gate["desk"].phase == "draining"
        assert org._room_mic_gated("desk"), (
            "room must still be gated immediately after TTS finishes sending"
        )

        before = len(sink)
        await org.handle_audio_text("desk-ui", "leftover echo")
        assert len(sink) == before, "the gate must drop the looped-back echo"


@pytest.mark.asyncio
async def test_gate_releases_after_cooldown(tmp_path: Path) -> None:
    """Once the cooldown expires, the next utterance is processed
    normally -- the gate must be transparent during silence."""
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path,
        tts=FakeTTS(samples_per_chunk=1),  # ~0s audio -> cooldown dominates
        tts_cooldown_s=0.050, gate_drain_margin_s=0.0,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()

        # Wait past the (negligible) drain estimate + the cooldown.
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
    racing two real turns -- the shared `_GatedTTS` would block both
    turns on the same release event, and `flush()` would deadlock."""
    tts = _GatedTTS()
    async with _make_organizer(
        [_DESK, _KITCHEN], tmp_path, tts=tts
    ) as (org, sink):
        # Drive kitchen into mid-TTS.
        await org.handle_user_text("kit-ui", "hello there")
        await tts.entered.wait()
        assert org._tts_gate["kitchen"].phase == "sending"
        assert org._room_mic_gated("kitchen"), "kitchen must be gated"
        assert not org._room_mic_gated("desk"), (
            "kitchen being mid-TTS must NOT gate the desk room"
        )

        # An audio transcript arriving from the desk room while kitchen
        # is mid-TTS must reach `handle_user_text` (i.e. produce an
        # enqueue), not be dropped. We don't drain the queue here because
        # the desk turn would also block on the shared TTS -- observing
        # `queue_depth("desk") == 1` proves the gate let it through.
        await org.handle_audio_text("desk-ui", "what time is it")
        assert org._queues.queue_depth("desk") == 1, (
            "desk audio must NOT be gated by kitchen's TTS -- turn should "
            "be enqueued"
        )

        # Release kitchen TTS so both turns can drain and the context
        # manager's close() doesn't have to forcibly cancel.
        tts.released.set()
        await org.flush()


@pytest.mark.asyncio
async def test_gate_clears_on_cancellation(tmp_path: Path) -> None:
    """A cancelled turn was flushed on the client, so it must NOT leave the room
    gated for the full unplayed duration -- that would deafen the mic to the
    user's follow-up. With a speaker present, cancel arms only the short
    cooldown; with cooldown=0 the gate is off immediately after cancel."""

    class HangingTTS:
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def synthesize(self, text: str):
            yield TtsChunkOut(pcm=b"\x00\x00", sample_rate=22_050)
            self.entered.set()
            await asyncio.sleep(3600)

    tts = HangingTTS()
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path, tts=tts, tts_cooldown_s=0.0
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await tts.entered.wait()
        assert org._tts_gate["desk"].phase == "sending"

        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        await org.handle_interrupt("desk-ui", sid)
        await org.flush()

        # Cancel must NOT enter the long DRAINING window (the client flushed).
        # With cooldown=0 the gate is fully off right after cancel.
        assert not org._room_mic_gated("desk"), (
            "cancel must not leave a long gate; cooldown=0 -> off immediately"
        )


@pytest.mark.asyncio
async def test_late_arm_does_not_clobber_successor_turn_gate(tmp_path: Path) -> None:
    """A cancelled turn's _speak finally runs on a later tick -- by then the next
    turn may already own the gate in SENDING. The late arm must NOT overwrite
    it (which would un-gate the room mid-playback of the new turn)."""
    async with _make_organizer([_DESK, _DESK_SPK], tmp_path, tts=FakeTTS()) as (org, _):
        # Simulate: turn N+1 has just claimed the gate (SENDING).
        org._tts_gate["desk"] = _TtsGate(phase="sending", session_id="N+1")
        # Turn N's stale finally arrives late with its own (old) session.
        org._arm_gate_after_send(
            "desk", "N", send_start=0.0, total_samples=4410, sample_rate=22_050,
            cancelled=True,
        )
        gate = org._tts_gate["desk"]
        assert gate.session_id == "N+1" and gate.phase == "sending", (
            "a stale turn's arm must not clobber the successor turn's gate"
        )


@pytest.mark.asyncio
async def test_no_speaker_opens_gate_immediately_after_send(tmp_path: Path) -> None:
    """A room with no connected speaker played nothing -- gating its mic for the
    reply duration would needlessly deafen it. The gate opens at send-end."""
    async with _make_organizer(
        [_DESK], tmp_path, tts=FakeTTS(), tts_cooldown_s=0.500, gate_drain_margin_s=5.0
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()
        assert "desk" not in org._tts_gate
        assert not org._room_mic_gated("desk")


@pytest.mark.asyncio
async def test_playback_done_shortens_gate(tmp_path: Path) -> None:
    """A speaker reporting playback drained collapses the long DRAINING estimate
    to the short cooldown -- the early-release the signal exists for."""
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path,
        tts=FakeTTS(samples_per_chunk=1),  # ~0s audio -> earliest_release ~ now
        tts_cooldown_s=0.0, gate_drain_margin_s=10.0,  # huge estimate -> stays gated
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()
        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        assert org._tts_gate["desk"].phase == "draining"
        assert org._room_mic_gated("desk")

        # Wrong sender role is ignored (only a speaker may drive the gate)...
        await org.handle_playback_done("desk-ui", sid)
        assert org._tts_gate["desk"].phase == "draining"
        # ...a stale session is ignored...
        await org.handle_playback_done("desk-spk", "not-the-session")
        assert org._tts_gate["desk"].phase == "draining"
        # ...the right speaker + session collapses to cooldown (0 -> open).
        await org.handle_playback_done("desk-spk", sid)
        assert not org._room_mic_gated("desk")


@pytest.mark.asyncio
async def test_playback_done_implausibly_early_is_ignored(tmp_path: Path) -> None:
    """A PlaybackDone arriving far before a long reply could have finished is a
    buggy/forged client trying to re-open the gate mid-playback -- ignore it."""
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path,
        tts=FakeTTS(sample_rate=22_050, samples_per_chunk=22_050),  # ~1s of audio
        tts_cooldown_s=0.0, gate_drain_margin_s=0.0,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()  # chunk is sent instantly (no real-time pacing)
        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        assert org._tts_gate["desk"].phase == "draining"

        # Right after send, ~all of the 1s estimate remains -> too early.
        await org.handle_playback_done("desk-spk", sid)
        assert org._tts_gate["desk"].phase == "draining", (
            "an implausibly early PlaybackDone must not collapse the gate"
        )


@pytest.mark.asyncio
async def test_gate_judged_against_capture_time_not_arrival(tmp_path: Path) -> None:
    """The bug this fix exists for: STT takes ~2s, so a transcript of GLaDOS's
    own voice arrives AFTER the gate (sized to playback) has opened. A
    `now`-based check would admit it -> feedback loop. The decision must be made
    against the audio's *capture* time, which fell inside the closed window."""
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path, tts=FakeTTS(),
        tts_cooldown_s=0.050, gate_drain_margin_s=0.050,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()
        closed_until = org._tts_gate["desk"].closed_until

        # Simulate STT latency: real `now` is past the horizon (gate has
        # opened), but the audio was captured just before it closed.
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, closed_until - loop.time()) + 0.020)
        now = loop.time()
        assert now >= closed_until, "precondition: the gate has opened by `now`"
        assert not org._room_mic_gated("desk"), "a now-based check sees it open"

        before = len(sink)
        await org.handle_audio_text(
            "desk-ui", "echo of my own voice", captured_at=closed_until - 0.010
        )
        assert len(sink) == before, (
            "audio captured while the gate was closed must be dropped even "
            "though its transcript arrived after the gate opened"
        )

        # A genuinely fresh utterance captured AFTER the horizon still passes.
        before = len(sink)
        await org.handle_audio_text(
            "desk-ui", "a real new question", captured_at=now
        )
        await org.flush()
        assert any(m["type"] == "welcome" for _, m in sink[before:]), (
            "audio captured after the gate opened must run a normal turn"
        )


@pytest.mark.asyncio
async def test_out_of_order_transcripts_judged_independently(tmp_path: Path) -> None:
    """Two background Whisper tasks can finish out of order. Each transcript
    must be judged against its OWN capture time, not arrival order -- the
    during-playback echo dropped, the post-gate utterance admitted, regardless
    of which `handle_audio_text` call lands first."""
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path, tts=FakeTTS(),
        tts_cooldown_s=0.050, gate_drain_margin_s=0.050,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()
        closed_until = org._tts_gate["desk"].closed_until
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, closed_until - loop.time()) + 0.020)
        after = loop.time()

        before = len(sink)
        # The LATER-captured (fresh) utterance's transcript arrives FIRST...
        await org.handle_audio_text("desk-ui", "fresh", captured_at=after)
        await org.flush()
        assert any(m["type"] == "welcome" for _, m in sink[before:]), (
            "the post-gate utterance must run even when its transcript arrives "
            "before the earlier echo's"
        )
        # ...then the EARLIER-captured echo arrives and is still dropped.
        before = len(sink)
        await org.handle_audio_text(
            "desk-ui", "echo", captured_at=closed_until - 0.010
        )
        assert len(sink) == before, "the during-playback echo is still dropped"


@pytest.mark.asyncio
async def test_stale_playback_done_cannot_reraise_horizon(tmp_path: Path) -> None:
    """A duplicate/late PlaybackDone arriving after the horizon has already
    passed must NOT push `closed_until` back into the future -- that would
    deafen the mic to a fresh user utterance. Early-release only ever lowers."""
    async with _make_organizer(
        [_DESK, _DESK_SPK], tmp_path,
        tts=FakeTTS(samples_per_chunk=1),  # ~0s audio -> earliest_release ~ now
        tts_cooldown_s=0.050, gate_drain_margin_s=0.0,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "hello there")
        await org.flush()
        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]

        # Let the (tiny) horizon pass so the gate is open.
        await asyncio.sleep(0.080)
        assert not org._room_mic_gated("desk")
        opened_horizon = org._tts_gate["desk"].closed_until

        # A late duplicate PlaybackDone must not raise the horizon.
        await org.handle_playback_done("desk-spk", sid)
        assert org._tts_gate["desk"].closed_until == opened_horizon, (
            "a stale PlaybackDone must not push the horizon into the future"
        )
        assert not org._room_mic_gated("desk"), "the gate must stay open"
