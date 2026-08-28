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
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, TtsChunkOut
from glados.core.config import ClientBinding
from glados.core.protocols import ToolConfirmResponse
from glados.core.organizer import Organizer
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
    """Streams one chunk per call, after an optional hold so a test can catch
    the audio mid-flight."""

    def __init__(
        self, hold: asyncio.Event | None = None, hold_marker: str = ""
    ) -> None:
        self._hold = hold
        self._hold_marker = hold_marker
        self.spoken: list[str] = []

    async def synthesize(self, text: str):
        self.spoken.append(text)
        if self._hold is not None and self._hold_marker in text:
            await self._hold.wait()
        yield TtsChunkOut(pcm=b"\x00\x00" * 16, sample_rate=22_050)


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
        **kwargs,
    )
    try:
        yield org, sink
    finally:
        await org.close()


def _tool_result_text(passes: list[list[LLMMessage]]) -> str:
    return next(m.content or "" for m in passes[-1] if m.role == "tool")


def _intercom_lines(tts: "_FakeTts") -> list[str]:
    """Only the handed-over messages. The originating room also speaks the
    turn's own reply through the same TTS, and that is not the intercom."""
    return [t for t in tts.spoken if t.startswith("Message from the")]


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
    ) as (org, _sink):
        org._queues.enqueue(  # noqa: SLF001
            "livingroom", lambda: org._deliver_intercom("livingroom", "earlier")
        )
        await asyncio.sleep(0.05)
        assert tts.spoken == ["earlier"]  # the livingroom really is stuck

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
        finally:
            trace.close()
            hold.set()
            await org.flush()

    assert result.ok
    assert result.content == {"status": "queued", "room": "livingroom"}
    # It was handed over, not spoken by the caller.
    assert _intercom_lines(tts) == ["Message from the desk. dinner is ready"]


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

    assert _intercom_lines(tts) == ["Message from the desk. dinner is ready"]
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

    assert _intercom_lines(tts) == []


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
    assert _intercom_lines(tts) == []


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
    assert _intercom_lines(tts) == []


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
