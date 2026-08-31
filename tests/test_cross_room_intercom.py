"""Speaking into a room you are not in (`DESIGN-cross-room-delivery.md`, slice 1).

The load-bearing property is that `room.speak_into` NEVER waits on the target
room. Waiting is what would push the call past the registry's dispatch budget
and hand the model an `indeterminate` outcome -- "I do not know whether the
livingroom heard you" -- which is the state `DESIGN-dispatch-cancellation.md`
exists to remove. So the answer is decided synchronously and the target room's
own worker owns the audio from there.
"""

from __future__ import annotations

import asyncio
import json
import tomllib
from contextlib import asynccontextmanager
from datetime import datetime, time as clock_time
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, TtsChunkOut
from glados.core.config import ClientBinding, QuietHours, RoomPolicy, RoomsConfig
from glados.core.protocols import ToolConfirmResponse
from glados.core.organizer import Organizer, _TtsGate
from glados.core.sessions import SessionRegistry
from glados.core.turn_outcome import TurnRecord
from glados.core.traces import TraceStore
from glados.mcp.registry import CallEnvelope, MCPRegistry
from glados.servers.room_intercom import MAX_MESSAGE_CHARS, SPEAK_INTO, SpeakIntoTool


DESK_UI = ClientBinding(
    client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
)
LIVING_SPEAKER = ClientBinding(
    client_id="livingroom-speaker",
    room_id="livingroom",
    role="speaker",
    default_user="qcko",
)
LIVING_MIC = ClientBinding(
    client_id="livingroom-mic", room_id="livingroom", role="mic", default_user="qcko"
)


class _FakeTts:
    """Streams a few chunks per call, after an optional hold so a test can
    catch the audio mid-flight. More than one chunk per call so that two
    interleaved streams are expressible at all -- with one chunk each, two
    streams cannot help but look serialised."""

    CHUNKS = 3

    def __init__(
        self, hold: asyncio.Event | None = None, hold_marker: str = ""
    ) -> None:
        self._hold = hold
        self._hold_marker = hold_marker
        self.spoken: list[str] = []
        self.holding = 0
        self._reached = asyncio.Condition()

    async def synthesize(self, text: str):
        self.spoken.append(text)
        if self._hold is not None and self._hold_marker in text:
            async with self._reached:
                self.holding += 1
                self._reached.notify_all()
            await self._hold.wait()
        for _ in range(self.CHUNKS):
            yield TtsChunkOut(pcm=b"\x00\x00" * 16, sample_rate=22_050)

    async def wait_until_holding(self, count: int = 1) -> None:
        """Block until `count` syntheses have reached the hold. A fixed sleep
        is a guess at how long a worker takes to spawn a task, open a trace on
        disk and enter the fake -- and when the guess is short, the failure is
        an empty-list mismatch that reads like a real bug."""
        async with self._reached:
            await asyncio.wait_for(
                self._reached.wait_for(lambda: self.holding >= count), 2.0
            )


class _SpeakIntoLLM:
    """Calls `room.speak_into` once with the given args, then replies."""

    def __init__(self, args: dict, reply: str = "Passed it on.") -> None:
        self._args = args
        self._reply = reply
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        if len(self.passes) == 1:
            yield LLMToolCall(
                call_id="c1", server="room", name="speak_into", args=self._args
            )
            return
        yield LLMText(text=self._reply)


@asynccontextmanager
async def _organizer(tmp: Path, llm, bindings, *, tts=None, **kwargs):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    by_id = {b.client_id: b for b in bindings}
    mcp = MCPRegistry()
    mcp.register(SpeakIntoTool(["livingroom"]))
    org = Organizer(
        llm=llm,
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        tts=tts,
        # Real length is 2s and every announcement pays it. Shrunk here so the
        # suite measures the ordering the veto window creates, not its duration.
        veto_pause_s=kwargs.pop("veto_pause_s", 0.01),
        **kwargs,
    )
    try:
        yield org, sink
    finally:
        await org.close()


def _tool_result_text(passes: list[list[LLMMessage]]) -> str:
    return next(m.content or "" for m in passes[-1] if m.role == "tool")


def _announcements(sink: list) -> list[str]:
    """What each announcement put into its target room, reassembled from the
    parts it was spoken in.

    Keyed on the synthetic session, which is the only thing that separates an
    announcement from the sending room's own reply -- the two go through the
    same fake TTS, and under the veto pause an announcement is two syntheses
    (the attribution, then the message) with the other room free to speak in
    between. Deduplicated per session because a broadcast lands once per
    client in the room."""
    parts: dict[str, dict[str, list[str]]] = {}
    for cid, m in sink:
        session = str(m.get("session_id", ""))
        if m.get("type") != "assistant_delta" or not session.startswith("intercom-"):
            continue
        parts.setdefault(session, {}).setdefault(cid, []).append(str(m["text"]))
    return [
        " ".join(next(iter(by_client.values()))) for by_client in parts.values()
    ]


async def _grant_confirms(org: Organizer, sink: list, client_id: str) -> None:
    """Approve every confirmation request as it appears, from `client_id`."""
    seen: set[str] = set()
    for _ in range(200):
        for _cid, msg in list(sink):
            rid = msg.get("request_id")
            if msg.get("type") == "tool_confirm_request" and rid not in seen:
                seen.add(rid)
                await org.handle_tool_confirm_response(
                    client_id, ToolConfirmResponse(request_id=rid, granted=True)
                )
        await asyncio.sleep(0.01)


# ---- the spec is a declaration, and its flags are not config ----------


def test_the_intercom_spec_is_gated_in_code() -> None:
    """These flags must not be reachable by the `servers.toml` overlay, which
    only applies to stdio tools and may not exist on the next machine. A
    capability that can put chosen audio into a room nobody is watching does
    not get an optional gate."""
    spec = SpeakIntoTool(["livingroom"]).spec
    assert spec.qualified == SPEAK_INTO
    assert spec.requires_confirmation is True
    assert spec.mutating is True
    assert spec.untrusted is False


def test_the_room_argument_is_pinned_to_configured_rooms() -> None:
    spec = SpeakIntoTool(["livingroom", "kitchen"]).spec
    assert spec.parameters["properties"]["room"]["enum"] == ["livingroom", "kitchen"]


async def test_dispatching_the_stub_speaks_nothing() -> None:
    """It is answered by the Organizer. If a dispatch ever reaches the stub,
    the capability has been demoted to an ordinary tool -- fail loudly rather
    than silently."""
    mcp = MCPRegistry()
    mcp.register(SpeakIntoTool(["livingroom"]))
    result = await mcp.dispatch(
        "room",
        "speak_into",
        {"room": "livingroom", "message": "hi"},
        CallEnvelope(session_id="s", room_id="desk", speaker_id="qcko"),
    )
    assert not result.ok
    assert "never be dispatched" in result.error


# ---- the call returns without waiting on the target room --------------


async def test_speak_into_returns_while_the_target_room_is_still_speaking(
    tmp_path: Path,
) -> None:
    """The property the whole design turns on, tested on the seam itself.

    Room B is held mid-TTS and its worker cannot progress. The call from room A
    must still return, promptly, and say `queued` -- never block, and never come
    back with an unknown outcome. Driven directly rather than through a full
    turn so that what is being measured is the handoff, not the confirmation
    round-trip (covered by the tests above)."""
    hold = asyncio.Event()
    tts = _FakeTts(hold, hold_marker="earlier")
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom",
            lambda: org._deliver_intercom(
                "livingroom", "Message from the desk.", "earlier"
            ),
        )
        await tts.wait_until_holding()
        # The livingroom really is stuck -- held on the message part, its
        # attribution already spoken and its veto window already elapsed.
        assert tts.spoken == ["Message from the desk.", "earlier"]

        call = LLMToolCall(
            call_id="c1",
            server="room",
            name="speak_into",
            args={"room": "livingroom", "message": "dinner is ready"},
        )
        outcome = TurnRecord()
        trace = org.traces.open("desk-test")
        try:
            result = await asyncio.wait_for(
                org._speak_into_room(call, "s1", "desk", trace, outcome),  # noqa: SLF001
                1.0,
            )
            # Handed to the room's FIFO, not spawned alongside it. A bare
            # `create_task` would also return promptly and would also end up
            # spoken -- and would put two streams into one room at once.
            assert org._queues.queue_depth("livingroom") == 1  # noqa: SLF001
            # Only the announcement already in flight, and only as far as it
            # has got -- its message bubble follows its audio. The caller
            # spoke nothing of the new one.
            assert _announcements(sink) == [_PREAMBLE_TEXT]
        finally:
            trace.close()
            hold.set()
            await org.flush()

    assert result.ok
    assert result.content == {"status": "queued", "room": "livingroom"}
    # It was handed over, not spoken by the caller.
    assert _announcements(sink) == [
        "Message from the desk. earlier",
        "Message from the desk. dinner is ready",
    ]


async def test_a_queued_message_is_spoken_into_the_target_room(
    tmp_path: Path,
) -> None:
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    assert _announcements(sink) == ["Message from the desk. dinner is ready"]
    # The handed-over audio reaches the livingroom and nowhere else. Matched on
    # the synthetic session so the desk's own reply chunks cannot be mistaken
    # for it.
    heard_it = {
        cid
        for cid, m in sink
        if m.get("type") == "tts_chunk"
        and str(m.get("session_id", "")).startswith("intercom-")
    }
    # Every recipient is a livingroom client -- the mic is in the room too and
    # simply ignores audio frames. What matters is that the desk heard nothing.
    assert heard_it and all(cid.startswith("livingroom") for cid in heard_it)


async def test_a_denied_confirmation_speaks_nothing(tmp_path: Path) -> None:
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):

        async def deny() -> None:
            for _ in range(200):
                for _cid, msg in list(sink):
                    if msg.get("type") == "tool_confirm_request":
                        await org.handle_tool_confirm_response(
                            "desk-ui",
                            ToolConfirmResponse(
                                request_id=msg["request_id"], granted=False
                            ),
                        )
                        return
                await asyncio.sleep(0.01)

        denier = asyncio.create_task(deny())
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            denier.cancel()

    assert _announcements(sink) == []


# ---- refusals: device facts only, never occupancy ---------------------


async def test_a_room_with_no_speaker_is_refused_not_silently_dropped(
    tmp_path: Path,
) -> None:
    """`_speak` into a speakerless room streams to nobody and returns, so
    without this the user is told the message was passed on when nothing was."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(tmp_path, llm, [DESK_UI], tts=tts) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    result = _tool_result_text(llm.passes)
    assert "refused" in result
    assert "no speaker" in result
    assert _announcements(sink) == []


async def test_speaking_into_your_own_room_is_refused(tmp_path: Path) -> None:
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "desk", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER], tts=tts
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "say it here"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    assert "already in" in _tool_result_text(llm.passes)
    assert _announcements(sink) == []


# ---- what reaches the speaker -----------------------------------------


def test_the_message_is_bounded_and_stripped() -> None:
    from glados.core.organizer import _spoken_message

    assert _spoken_message("hello\x00\x07 there\n\nfriend") == "hello there friend"
    assert len(_spoken_message("x" * 500)) == MAX_MESSAGE_CHARS


def test_the_attribution_names_the_room_never_the_person() -> None:
    """Naming who is at the desk is a presence disclosure into a room they did
    not choose to be in."""
    import inspect

    from glados.core.organizer import Organizer as _Org

    source = inspect.getsource(_Org._speak_into_room)
    assert 'f"Message from the {room_id}' in source
    assert "speaker_id" not in source


# ---- a refusal changed nothing, and must not be recorded as if it did --


async def test_a_refusal_is_not_recorded_as_a_landed_mutation(
    tmp_path: Path,
) -> None:
    """`ok=True` keeps a refusal from failing the turn, but it must not make
    `made_successful_mutation` true -- that drives the "did that work?" answer,
    and it would report success for a message nobody heard. It would also trip
    `may_have_mutated` and switch off this turn's confabulation recovery."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(tmp_path, llm, [DESK_UI], tts=tts) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    recorded = list(org._last_turn.values())  # noqa: SLF001
    assert len(recorded) == 1
    assert recorded[0].mutations == ()


def test_the_spoken_sentence_never_becomes_claim_evidence() -> None:
    """`_subjects` harvests free-text arguments as what a call was ABOUT. The
    intercom's argument is a sentence the model wrote, so harvesting it would
    let its own prose corroborate its own false claim."""
    from glados.core.organizer import _subject_args

    call = LLMToolCall(
        call_id="c1",
        server="room",
        name="speak_into",
        args={"room": "livingroom", "message": "I took the milk out"},
    )
    assert _subject_args(call) is None

    ordinary = LLMToolCall(
        call_id="c2", server="dunnes", name="add", args={"item": "milk"}
    )
    assert _subject_args(ordinary) == {"item": "milk"}


# ---- the target room owns the announcement, and can stop it -----------


def _delta_session(sink: list, prefix: str) -> str:
    session = next(
        (
            m["session_id"]
            for _cid, m in sink
            if m.get("type") == "assistant_delta"
            and str(m.get("text", "")).startswith(prefix)
        ),
        None,
    )
    assert session is not None, f"no assistant_delta starting {prefix!r} in {sink}"
    return session


async def _until(predicate, what: str, timeout: float = 2.0) -> None:
    """Wait for something a cancellation delivers on a later tick. Bounded, so
    a real regression fails on the named condition rather than on a sleep that
    happened to be long enough."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.005)


async def test_stop_in_the_target_room_cancels_only_that_announcement(
    tmp_path: Path,
) -> None:
    """The synthetic session is bound to the room the audio lands in, so a
    voice "stop" there reaches the announcement through the ordinary barge-in
    path -- and reaches nothing else. It must not cancel work in the room that
    sent the message (which is already done and has been told `queued`), and it
    must not fall through into a new turn in the target room."""
    hold = asyncio.Event()
    tts = _FakeTts(hold)  # every synthesis holds
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom",
            lambda: org._deliver_intercom(
                "livingroom", "Message from the desk.", "one"
            ),
        )
        org._queues.enqueue(  # noqa: SLF001
            "desk",
            lambda: org._deliver_intercom(
                "desk", "Message from the livingroom.", "two"
            ),
        )
        await tts.wait_until_holding(2)
        living_session = _delta_session(sink, "Message from the desk")

        await org.handle_audio_text("livingroom-mic", "stop")
        await _until(
            lambda: any(m.get("type") == "cancelled" for _cid, m in sink),
            "the cancelled broadcast",
        )

        cancelled = {
            m["session_id"] for _cid, m in sink if m.get("type") == "cancelled"
        }
        assert cancelled == {living_session}
        # The desk's announcement is untouched and still holding its TTS.
        assert "desk" in {rid for _task, rid in org._inflight.values()}  # noqa: SLF001
        # "stop" ended there -- it did not become a turn in the livingroom.
        assert llm.passes == []

        hold.set()
        await org.flush()


async def test_the_target_rooms_own_playback_done_drives_the_gate(
    tmp_path: Path,
) -> None:
    """`handle_playback_done` matches on `session_id`. Under the design this
    slice replaced -- the caller's session armed the target room's gate -- the
    target's speaker could never match it, and the room stayed deaf for the
    whole estimate. The gate belongs to the synthetic session, so the speaker
    that actually played the audio is the one that can release it.

    Driven through a whole turn on purpose: the regression being pinned is the
    caller's session reaching the target's gate, and a test that only enqueues
    a delivery has no caller session to confuse it with."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

        session = _delta_session(sink, "Message from the desk")
        desk_session = _delta_session(sink, "Passed it on")
        gate = org._tts_gate["livingroom"]  # noqa: SLF001
        assert gate.phase == "draining"
        assert gate.session_id == session
        # The synthetic session, not the desk turn that sent the message.
        assert session.startswith("intercom-")
        assert session != desk_session

        # Another turn's signal is still stale and still dropped.
        await org.handle_playback_done("livingroom-speaker", "some-other-session")
        assert org._tts_gate["livingroom"].phase == "draining"  # noqa: SLF001

        await org.handle_playback_done("livingroom-speaker", session)
        assert org._tts_gate["livingroom"].phase == "cooldown"  # noqa: SLF001


# ---- one turn cannot flood a room -------------------------------------


class _TwiceIntoTheSameRoomLLM:
    """Two `room.speak_into` calls at the same target, in one turn."""

    def __init__(self) -> None:
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        if len(self.passes) == 1:
            for call_id, message in (("c1", "dinner is ready"), ("c2", "and again")):
                yield LLMToolCall(
                    call_id=call_id,
                    server="room",
                    name="speak_into",
                    args={"room": "livingroom", "message": message},
                )
            return
        yield LLMText(text="Passed it on.")


def _tool_result_texts(passes: list[list[LLMMessage]]) -> list[str]:
    return [m.content or "" for m in passes[-1] if m.role == "tool"]


async def test_one_turn_may_announce_into_a_room_only_once(tmp_path: Path) -> None:
    """A per-turn cap is what answers the Security lens's objection to
    queueing: bounded accumulation, so one compromised turn cannot loop a
    room's speaker. The second call is refused, not queued."""
    tts = _FakeTts()
    llm = _TwiceIntoTheSameRoomLLM()
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    results = _tool_result_texts(llm.passes)
    assert len(results) == 2
    assert "queued" in results[0]
    assert "already been passed" in results[1]
    assert _announcements(sink) == ["Message from the desk. dinner is ready"]


async def test_two_announcements_into_one_room_do_not_interleave(
    tmp_path: Path,
) -> None:
    """The reason the handoff is an enqueue rather than a spawn: the room's
    FIFO already owns "two simultaneous TTS streams to one room are
    incoherent". The second announcement waits for the first to finish, and
    the chunks of the two never mix."""
    hold = asyncio.Event()
    tts = _FakeTts(hold, hold_marker="first")
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        for message in ("first", "second"):
            org._queues.enqueue(  # noqa: SLF001
                "livingroom",
                lambda m=message: org._deliver_intercom(
                    "livingroom", "Message from the desk.", m
                ),
            )
        await tts.wait_until_holding()
        # The second has not started synthesising while the first is held --
        # not even its attribution, which is the part that would be audible
        # over the first announcement.
        assert tts.spoken == ["Message from the desk.", "first"]
        hold.set()
        await org.flush()

    assert _announcements(sink) == [
        "Message from the desk. first",
        "Message from the desk. second",
    ]
    streams = [
        m["session_id"] for _cid, m in sink if m.get("type") == "tts_chunk"
    ]
    # Contiguous runs, one per announcement -- never A, B, A.
    runs = [sid for i, sid in enumerate(streams) if i == 0 or streams[i - 1] != sid]
    assert len(runs) == len(set(runs)) == 2


# ---- egress only: nothing of the announcement lands in the target room -


async def test_the_announcement_leaves_no_trace_in_the_target_rooms_history(
    tmp_path: Path,
) -> None:
    """Section 3's "no context bleed" is a session rule, so the intercom is
    egress and nothing more: the sending room keeps the tool result, and the
    room the audio lands in gains no session and no history. This is what the
    synthetic session buys structurally rather than by policy -- and it is the
    accepted cost, too, because "what did you just say?" asked in the target
    room has nothing to answer from."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

        desk = org.sessions.latest("desk", "qcko")
        assert desk is not None
        history = org._history[desk.session_id]  # noqa: SLF001
        assert any("queued" in (m.content or "") for m in history if m.role == "tool")

        # No session was opened in the room the audio played in, and the
        # synthetic session that carried it holds no history of its own.
        assert org.sessions.latest("livingroom", "qcko") is None
        assert list(org._history) == [desk.session_id]  # noqa: SLF001


# ---- slice 2a: the receiving room's own say ---------------------------


def _rooms_toml(text: str) -> RoomsConfig:
    return RoomsConfig(**tomllib.loads(text))


_CLIENTS = """
[[clients]]
client_id = "desk-ui"
room_id = "desk"
role = "ui"

[[clients]]
client_id = "living-speaker"
room_id = "livingroom"
role = "speaker"
"""


def test_a_room_with_no_policy_row_keeps_the_pre_slice_behaviour() -> None:
    """Absent config is the permissive default, which is the whole reason the
    validations below have to be strict: a misspelled row must not be able to
    counterfeit an absence."""
    cfg = _rooms_toml(_CLIENTS)
    assert cfg.policy_for("livingroom") is None


def test_a_duplicate_or_misspelled_policy_row_fails_the_load() -> None:
    """Each of these fails the ACL OPEN if it is allowed through -- a
    duplicate silently loses one row, and a typo leaves the room the operator
    meant to protect running policy-free."""
    duplicate = _CLIENTS + """
[[rooms]]
room_id = "livingroom"

[[rooms]]
room_id = "livingroom"
"""
    with pytest.raises(ValidationError, match="duplicate room policy"):
        _rooms_toml(duplicate)

    unknown_room = _CLIENTS + """
[[rooms]]
room_id = "livingrom"
"""
    with pytest.raises(ValidationError, match="unknown room"):
        _rooms_toml(unknown_room)

    unknown_source = _CLIENTS + """
[[rooms]]
room_id = "livingroom"
announce_sources = ["desk", "kitchen"]
"""
    with pytest.raises(ValidationError, match="unknown room: 'kitchen'"):
        _rooms_toml(unknown_source)


def test_a_misspelled_key_cannot_counterfeit_an_absent_policy() -> None:
    """The nastier half of the same failure, and the reason these models
    forbid extra keys: a singular `announce_source` is DROPPED under
    pydantic's default, leaving a row that exists, passes every validation,
    and permits everything -- a fully open ACL that reads at a glance as a
    configured one. A typo'd table name must fail just as loudly."""
    for typo in (
        'announce_source = ["desk"]',
        'quiet_hour = { start = "22:00", end = "07:00" }',
    ):
        with pytest.raises(ValidationError, match="[Ee]xtra"):
            _rooms_toml(_CLIENTS + f"""
[[rooms]]
room_id = "livingroom"
{typo}
""")

    with pytest.raises(ValidationError, match="[Ee]xtra"):
        _rooms_toml(_CLIENTS + """
[[room]]
room_id = "livingroom"
""")


def test_an_empty_allowlist_is_an_opt_out_and_an_absent_one_is_not() -> None:
    """`[] for none` is what rooms.toml advertises, and the two are one
    keystroke apart in the code: `if not self.announce_sources` would turn a
    room's full opt-out into a full opt-in and nothing else would notice."""
    assert not RoomPolicy(room_id="livingroom", announce_sources=[]).allows_source("desk")
    assert RoomPolicy(room_id="livingroom").allows_source("desk")


def test_an_equal_ended_quiet_window_is_rejected_rather_than_guessed() -> None:
    """"Always" and "never" are equally defensible readings of start == end,
    so the operator says which."""
    with pytest.raises(ValidationError, match="make them differ"):
        _rooms_toml(_CLIENTS + """
[[rooms]]
room_id = "livingroom"
quiet_hours = { start = "22:00", end = "22:00" }
""")


def test_a_quiet_window_wraps_midnight_at_both_edges() -> None:
    """The common shape for a room someone sleeps in. Half-open: the start
    minute is quiet, the end minute is not."""
    hours = QuietHours(start=clock_time(22, 0), end=clock_time(7, 0))
    assert hours.contains(clock_time(22, 0))
    assert hours.contains(clock_time(3, 0))
    assert hours.contains(clock_time(6, 59))
    assert not hours.contains(clock_time(7, 0))
    assert not hours.contains(clock_time(21, 59))

    daytime = QuietHours(start=clock_time(9, 0), end=clock_time(17, 0))
    assert daytime.contains(clock_time(12, 0))
    assert not daytime.contains(clock_time(3, 0))


def _policy(**kwargs) -> RoomPolicy:
    return RoomPolicy(room_id="livingroom", **kwargs)


def _at(hour: int) -> "datetime":
    return _at_minute(hour, 0)


def _at_minute(hour: int, minute: int) -> "datetime":
    return datetime(2026, 8, 29, hour, minute).astimezone()


async def test_a_source_off_the_allowlist_is_refused_and_speaks_nothing(
    tmp_path: Path,
) -> None:
    """The receiving room's say, made as config when nobody is under attack
    rather than as a live prompt under duress."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        room_policy=lambda r: _policy(announce_sources=["desk2"]) if r == "livingroom" else None,
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    assert "not accepting messages right now" in _tool_result_text(llm.passes)
    assert _announcements(sink) == []


async def test_quiet_hours_refuse_in_words_the_allowlist_cannot_be_told_from(
    tmp_path: Path,
) -> None:
    """Both refusals are static config facts, so either is safe to disclose
    alone -- but distinguishable ones let a caller map the policy table room
    by room, and a quiet window is a proxy for where somebody sleeps. The
    real reason lives in the trace, not in what the model can read out."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        now=lambda: _at(23),
        room_policy=lambda r: _policy(
            quiet_hours=QuietHours(start=clock_time(22, 0), end=clock_time(7, 0))
        ) if r == "livingroom" else None,
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    result = _tool_result_text(llm.passes)
    assert "not accepting messages right now" in result
    for leak in ("quiet", "hours", "allow", "asleep", "22:00"):
        assert leak not in result.lower()
    assert _announcements(sink) == []


async def test_an_allowed_source_outside_the_window_still_gets_through(
    tmp_path: Path,
) -> None:
    """The policy must refuse the two cases it names and nothing else."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        now=lambda: _at(19),
        room_policy=lambda r: _policy(
            announce_sources=["desk"],
            quiet_hours=QuietHours(start=clock_time(22, 0), end=clock_time(7, 0)),
        ) if r == "livingroom" else None,
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    assert "queued" in _tool_result_text(llm.passes)
    assert _announcements(sink) == ["Message from the desk. dinner is ready"]


async def test_a_window_that_opens_while_the_message_waits_drops_it(
    tmp_path: Path,
) -> None:
    """The hand-over is an enqueue behind the target room's own turn, which
    the design says can run for tens of seconds -- long enough for the window
    to open under a message already approved. Checked at delivery as well as
    at hand-over, so an announcement cleared at 21:59 cannot speak at 22:00."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    # The window opens BETWEEN the two reads: the hand-over sees 21:59 and
    # clears it, the delivery sees 22:00. Driven off the call order rather
    # than a sleep, so it does not depend on how the worker gets scheduled.
    readings = iter([_at_minute(21, 59)])
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        now=lambda: next(readings, _at_minute(22, 0)),
        room_policy=lambda r: _policy(
            quiet_hours=QuietHours(start=clock_time(22, 0), end=clock_time(7, 0))
        ) if r == "livingroom" else None,
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

        # The sending room was told `queued` and is not told otherwise --
        # `queued` never promised the message was spoken.
        assert "queued" in _tool_result_text(llm.passes)
        # Dropped before anything of it reached the target room: no audio, no
        # bubble in its UI, and no synthetic session left registered.
        assert _announcements(sink) == []
        # Nothing under the synthetic session the announcement would have run
        # as -- the sending room's own reply speaks through the same fake, so
        # the session id is what separates the two.
        assert [
            m
            for _cid, m in sink
            if str(m.get("session_id", "")).startswith("intercom-")
        ] == []
        assert org._inflight == {}  # noqa: SLF001


# ---- slice 2b: the target room's veto ---------------------------------


class _TimedTts:
    """One chunk per synthesis, long enough that its playback estimate is a
    measurable number, and the loop time each synthesis began."""

    def __init__(self, seconds: float) -> None:
        self._samples = int(22_050 * seconds)
        self.calls: list[tuple[str, float]] = []

    async def synthesize(self, text: str):
        self.calls.append((text, asyncio.get_running_loop().time()))
        yield TtsChunkOut(pcm=b"\x00\x00" * self._samples, sample_rate=22_050)


async def test_the_veto_gap_starts_when_the_attribution_stops_being_audible(
    tmp_path: Path,
) -> None:
    """`_speak` returns when the last chunk is handed to the client, not when
    it is heard. A gap measured from that return elapses while the attribution
    is still playing, so nobody could answer inside it and the veto would be
    offered in name only. The hold is measured from the horizon the gate was
    armed with instead -- pinned here with the configured gap set to zero, so
    the only thing separating the two syntheses is the playback estimate."""
    tts = _TimedTts(0.3)
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        veto_pause_s=0.0,
    ) as (org, _sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom",
            lambda: org._deliver_intercom(  # noqa: SLF001
                "livingroom", "Message from the desk.", "dinner is ready"
            ),
        )
        await org.flush()

    spoken = [text for text, _at in tts.calls]
    assert spoken == ["Message from the desk.", "dinner is ready"]
    began = dict((text, at) for text, at in tts.calls)
    assert began["dinner is ready"] - began["Message from the desk."] >= 0.3


_PREAMBLE_TEXT = "Message from the desk."


async def _wait_for_veto_window(tts: "_FakeTts") -> None:
    """Block until an announcement has spoken its attribution and is holding."""
    await _until(
        lambda: _PREAMBLE_TEXT in tts.spoken, "the attribution to be spoken"
    )


@pytest.mark.parametrize("veto", ["stop", "nevermind"])
async def test_a_veto_in_the_target_room_stops_the_message_being_spoken(
    tmp_path: Path, veto: str
) -> None:
    """The whole of slice 2b. The room that is about to be spoken into gets
    the attribution, then a gap, and ending it is the ordinary voice barge-in
    -- no new listener, no reply parser, no consent window.

    Parametrised over two barge-in tokens on purpose: `nevermind` is already
    matched by `_BARGE_IN_RE`, so any later attempt to add an ask-first "no"
    parser has to keep losing this race. Barge-in is checked before the mic
    gate, which is what lets a veto land at all -- the attribution has just
    closed the gate on the room it was spoken into."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        veto_pause_s=5.0,
    ) as (org, sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom",
            lambda: org._deliver_intercom(  # noqa: SLF001
                "livingroom", _PREAMBLE_TEXT, "dinner is ready"
            ),
        )
        await _wait_for_veto_window(tts)

        await org.handle_audio_text("livingroom-mic", veto)
        await _until(
            lambda: any(m.get("type") == "cancelled" for _cid, m in sink),
            "the cancelled broadcast",
        )
        await org.flush()

    # The message never reached the room: not as audio, and not as a bubble.
    assert tts.spoken == [_PREAMBLE_TEXT]
    assert _announcements(sink) == [_PREAMBLE_TEXT]
    # The veto ended there -- it did not fall through into a turn of its own.
    assert llm.passes == []
    assert org._inflight == {}  # noqa: SLF001


async def test_a_room_may_opt_out_of_the_gap_and_hears_one_utterance(
    tmp_path: Path,
) -> None:
    """The opt-out removes the gap, not the announcement, and not the sending
    room's confirmation -- what it costs the room is a veto, which is why it
    is the room's own row that carries it."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        veto_pause_s=5.0,  # long enough that a gap taken would show as a hang
        room_policy=lambda r: _policy(announce_veto_pause=False)
        if r == "livingroom"
        else None,
    ) as (org, sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom",
            lambda: org._deliver_intercom(  # noqa: SLF001
                "livingroom", _PREAMBLE_TEXT, "dinner is ready"
            ),
        )
        await asyncio.wait_for(org.flush(), 2.0)

    assert tts.spoken == ["Message from the desk. dinner is ready"]
    assert _announcements(sink) == ["Message from the desk. dinner is ready"]


async def test_announcements_stop_accumulating_in_a_room_that_is_not_keeping_up(
    tmp_path: Path,
) -> None:
    """A queue is a buffer, and the per-turn cap only bounds one turn. Past a
    depth the room stops taking announcements -- and says so in the words a
    policy block uses, because "that room is busy" is the occupancy fact this
    direction may never disclose."""
    blocked = asyncio.Event()
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, _sink):
        for _ in range(4):  # one runs, three wait
            org._queues.enqueue("livingroom", blocked.wait)  # noqa: SLF001
        await _until(
            lambda: org._queues.queue_depth("livingroom") == 3,  # noqa: SLF001
            "the queue to settle at its bound",
        )

        call = LLMToolCall(
            call_id="c1",
            server="room",
            name="speak_into",
            args={"room": "livingroom", "message": "dinner is ready"},
        )
        trace = org.traces.open("desk-test")
        try:
            result = await asyncio.wait_for(
                org._speak_into_room(  # noqa: SLF001
                    call, "s1", "desk", trace, TurnRecord()
                ),
                1.0,
            )
        finally:
            trace.close()

        assert result.ok
        assert result.content == {
            "status": "refused",
            "reason": "the livingroom is not accepting messages right now",
        }
        assert org._queues.queue_depth("livingroom") == 3  # noqa: SLF001
        # The room's own occupant is never turned away for a depth an
        # announcement from elsewhere put there.
        await org.handle_user_text("livingroom-mic", "what time is it")
        assert org._queues.queue_depth("livingroom") == 4  # noqa: SLF001

        # Dropped rather than run: what is being pinned is that the enqueue
        # was accepted, and letting that turn drive the fake LLM would only
        # measure the warm-up gate.
        org._queues.clear("livingroom")  # noqa: SLF001
        blocked.set()
        await org.flush()

    assert _announcements(_sink) == []


def _trace_events(tmp: Path, session_prefix: str = "intercom-") -> list[str]:
    """Every event kind recorded under an announcement's synthetic session."""
    events: list[str] = []
    for path in sorted(tmp.glob(f"{session_prefix}*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line)["event"])
    return events


async def test_a_veto_and_an_interruption_are_audited_apart(tmp_path: Path) -> None:
    """Both stop an announcement through the same mechanism, and afterwards
    only the audit can say which happened: whether the room refused a message
    it had not heard, or cut one off part-way through. A single `cancelled`
    for both would answer neither question."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        veto_pause_s=5.0,
    ) as (org, sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom",
            lambda: org._deliver_intercom(  # noqa: SLF001
                "livingroom", _PREAMBLE_TEXT, "dinner is ready"
            ),
        )
        await _wait_for_veto_window(tts)
        await org.handle_audio_text("livingroom-mic", "stop")
        await _until(
            lambda: any(m.get("type") == "cancelled" for _cid, m in sink),
            "the cancelled broadcast",
        )
        await org.flush()

    events = _trace_events(tmp_path)
    assert "intercom_vetoed" in events
    assert "cancelled" not in events


async def test_an_interruption_during_the_message_is_not_recorded_as_a_veto(
    tmp_path: Path,
) -> None:
    """The other half of the pair: audio is already out, and the room is
    stopping something it has begun to hear."""
    hold = asyncio.Event()
    tts = _FakeTts(hold, hold_marker="dinner is ready")
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        veto_pause_s=0.01,
    ) as (org, sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom",
            lambda: org._deliver_intercom(  # noqa: SLF001
                "livingroom", _PREAMBLE_TEXT, "dinner is ready"
            ),
        )
        await tts.wait_until_holding()  # held inside the message's synthesis
        await org.handle_audio_text("livingroom-mic", "stop")
        await _until(
            lambda: any(m.get("type") == "cancelled" for _cid, m in sink),
            "the cancelled broadcast",
        )
        hold.set()
        await org.flush()

    events = _trace_events(tmp_path)
    assert "cancelled" in events
    assert "intercom_vetoed" not in events


async def test_the_gap_does_not_borrow_another_sessions_playback_horizon(
    tmp_path: Path,
) -> None:
    """The horizon read back from the gate is only this announcement's if the
    gate is still this announcement's. A speakerless room has no gate at all
    (`_arm_gate_after_send` pops it) and a successor turn overwrites it with
    its own -- and that one can be seconds out, which would hold the message
    for somebody else's audio. Unowned means fall back to the configured gap."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "unused"})
    async with _organizer(
        tmp_path,
        llm,
        [DESK_UI, LIVING_SPEAKER, LIVING_MIC],
        tts=tts,
        veto_pause_s=0.01,
    ) as (org, _sink):
        org._tts_gate["livingroom"] = _TtsGate(  # noqa: SLF001
            phase="draining",
            session_id="somebody-elses-turn",
            closed_until=asyncio.get_running_loop().time() + 30.0,
        )
        trace = org.traces.open("intercom-guard-test")
        try:
            await asyncio.wait_for(
                org._hold_veto_window(  # noqa: SLF001
                    "intercom-mine", "livingroom", trace
                ),
                1.0,
            )
        finally:
            trace.close()

        del org._tts_gate["livingroom"]  # noqa: SLF001
        trace = org.traces.open("intercom-guard-test")
        try:
            await asyncio.wait_for(
                org._hold_veto_window(  # noqa: SLF001
                    "intercom-mine", "livingroom", trace
                ),
                1.0,
            )
        finally:
            trace.close()


# ---- the confirmation gate cannot be reached around --------------------


class _TwiceIdenticallyLLM:
    """The same `room.speak_into` call twice in one turn, argument for
    argument -- the shape `_in_flight_key` canonicalises to one key."""

    ARGS = {"room": "livingroom", "message": "dinner is ready"}

    def __init__(self) -> None:
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        if len(self.passes) == 1:
            for call_id in ("c1", "c2"):
                yield LLMToolCall(
                    call_id=call_id,
                    server="room",
                    name="speak_into",
                    args=dict(self.ARGS),
                )
            return
        yield LLMText(text="Passed it on.")


def _watch_the_intercom(org: Organizer, sink: list) -> list[list[str]]:
    """Record what room A had already been sent each time the intercom was
    entered. Entering it at all is the thing being watched: a gate checked
    after the hand-over would leave the room just as silent on a denial, so
    "nothing was spoken" is not evidence that nothing was reached."""
    seen: list[list[str]] = []
    original = org._speak_into_room  # noqa: SLF001

    async def watched(*args, **kwargs):
        seen.append([msg.get("type") for _cid, msg in sink])
        return await original(*args, **kwargs)

    org._speak_into_room = watched  # noqa: SLF001
    return seen


async def _deny_confirms(org: Organizer, sink: list, client_id: str) -> None:
    """Refuse every confirmation request as it appears, from `client_id`."""
    seen: set[str] = set()
    for _ in range(200):
        for _cid, msg in list(sink):
            rid = msg.get("request_id")
            if msg.get("type") == "tool_confirm_request" and rid not in seen:
                seen.add(rid)
                await org.handle_tool_confirm_response(
                    client_id, ToolConfirmResponse(request_id=rid, granted=False)
                )
        await asyncio.sleep(0.01)


async def test_the_intercom_is_never_entered_without_a_granted_confirmation(
    tmp_path: Path,
) -> None:
    """The gate holds by three facts and none of them is local to the
    intercom: `requires_confirmation` is hardcoded True on the spec, so the
    un-gated arm of `_run_tool_calls` is unreachable for this tool; a denial
    returns before `_dispatch_or_answer`; and the interception is on the
    qualified name inside `_dispatch_or_answer`, downstream of both. Each is
    one edit away from a capability that puts chosen audio into a room nobody
    is watching, and none of the three was pinned."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        entered = _watch_the_intercom(org, sink)
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    # Entered once, and the room had already been asked when it was.
    assert len(entered) == 1
    assert "tool_confirm_request" in entered[0]


async def test_a_denied_confirmation_never_reaches_the_intercom_at_all(
    tmp_path: Path,
) -> None:
    """The stronger form of "a denied confirmation speaks nothing": the
    capability is not invoked, so silence is the gate's doing rather than a
    later refusal's."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        entered = _watch_the_intercom(org, sink)
        denials = asyncio.create_task(_deny_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            denials.cancel()

    assert entered == []
    assert _announcements(sink) == []
    assert org._queues.queue_depth("livingroom") == 0  # noqa: SLF001


async def test_the_intercom_never_answers_indeterminate(tmp_path: Path) -> None:
    """What keeps the in-flight ledger's arm unreachable for this tool.

    That arm answers a re-issued call ahead of the confirmation gate, and a
    key only lands in the ledger when a mutating call comes back
    `indeterminate`. The intercom is intercepted before `mcp.dispatch`, so it
    is never inside the registry's timeout and has no indeterminate outcome to
    return -- which is the fact that keeps the pre-gate arm dead here, and the
    fact a single `return MCPCallResult(indeterminate=True)` would kill."""
    tts = _FakeTts()
    llm = _SpeakIntoLLM({"room": "livingroom", "message": "dinner is ready"})
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, _sink):
        outcome = TurnRecord()
        trace = org.traces.open("desk-test")
        attempts = [
            {"room": "livingroom", "message": "dinner is ready"},  # queued
            {"room": "livingroom", "message": "again"},  # already announced
            {"room": "desk", "message": "to myself"},  # own room
            {"room": "kitchen", "message": "nobody there"},  # no speaker
            {"room": "livingroom", "message": ""},  # nothing to say
        ]
        try:
            for i, args in enumerate(attempts):
                call = LLMToolCall(
                    call_id=f"c{i}", server="room", name="speak_into", args=args
                )
                result = await org._speak_into_room(  # noqa: SLF001
                    call, "s1", "desk", trace, outcome
                )
                assert result.indeterminate is False, args
        finally:
            trace.close()
            await org.flush()

        # Nothing the intercom returned could put a key in the ledger.
        assert outcome.in_flight == set()


async def test_a_re_issued_announcement_is_refused_after_its_gate_not_before(
    tmp_path: Path,
) -> None:
    """A second identical call in one turn is answered by the per-turn cap,
    which sits BELOW the confirmation gate -- not by the in-flight ledger,
    which sits above it. Both refuse, so only the wording and the second
    confirmation prompt tell them apart, and inverting the two would move the
    intercom to the one arm of `_run_tool_calls` that answers without asking
    the room."""
    tts = _FakeTts()
    llm = _TwiceIdenticallyLLM()
    async with _organizer(
        tmp_path, llm, [DESK_UI, LIVING_SPEAKER, LIVING_MIC], tts=tts
    ) as (org, sink):
        confirms = asyncio.create_task(_grant_confirms(org, sink, "desk-ui"))
        try:
            await asyncio.wait_for(org.handle_user_text("desk-ui", "tell them"), 5.0)
            await org.flush()
        finally:
            confirms.cancel()

    asked = [m for _cid, m in sink if m.get("type") == "tool_confirm_request"]
    assert len(asked) == 2  # each attempt asked the room for itself

    results = _tool_result_texts(llm.passes)
    assert "queued" in results[0]
    assert "already been passed" in results[1]
    assert "already attempted" not in results[1].lower()
    assert _announcements(sink) == ["Message from the desk. dinner is ready"]
