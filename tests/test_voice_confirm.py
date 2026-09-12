"""The spoken arm of the confirmation gate (DESIGN-voice-confirm.md).

A room with a microphone and a loudspeaker hears the question and answers
by voice; the answer is intercepted at ingress and resolves the same Future
the desk dialog does. Voice source only, room-bound, no LLM in the decision.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMText, LLMToolCall, ToolSpec, TtsChunkOut
from glados.core.config import ClientBinding
from glados.core.confirm_phrase import render_confirm_question
from glados.core.organizer import Organizer
from glados.core.protocols import ToolConfirmResponse
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.core.turn_outcome import TurnRecord, classify
from glados.core.utterance import classify_confirm_answer
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry

SAMPLE_RATE = 22_050
ZERO_WIDTH_SPACE = chr(0x200B)
BELL = chr(7)


class _GatedTool:
    spec = ToolSpec(
        server="t",
        name="boom",
        description="side-effecting test tool",
        parameters={"type": "object"},
        requires_confirmation=True,
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        self.calls.append(args)
        return MCPCallResult(ok=True, content={"did": "the thing"})


class _ToolCallingLLM:
    """First pass: one gated call. Every later pass (the reply, and any
    queued non-answer turn): text only."""

    def __init__(self, args: dict | None = None) -> None:
        self._n = 0
        self._args = args or {"x": 1}

    async def chat(self, messages, tools):
        self._n += 1
        if self._n == 1:
            yield LLMToolCall(call_id="c1", server="t", name="boom", args=self._args)
        else:
            yield LLMText(text="done")


class _FakeTts:
    def __init__(self, seconds: float = 0.2, fail: bool = False) -> None:
        self.spoken: list[str] = []
        self._samples = int(seconds * SAMPLE_RATE)
        self._fail = fail

    async def synthesize(self, text: str):
        self.spoken.append(text)
        if self._fail:
            raise RuntimeError("synth down")
        yield TtsChunkOut(pcm=b"\x00\x00" * self._samples, sample_rate=SAMPLE_RATE)


MIC = ClientBinding(client_id="k-mic", room_id="kitchen", role="mic", default_user="u")
SPEAKER = ClientBinding(
    client_id="k-spk", room_id="kitchen", role="speaker", default_user="u"
)
DESK_UI = ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="u")
DESK_MIC = ClientBinding(client_id="d-mic", room_id="desk", role="mic", default_user="u")


@asynccontextmanager
async def _make_org(
    tmp: Path,
    bindings: list[ClientBinding],
    *,
    tts=None,
    llm=None,
    confirm_timeout_s: float = 30.0,
):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    by_id = {b.client_id: b for b in bindings}
    mcp = MCPRegistry()
    tool = _GatedTool()
    mcp.register(tool)
    org = Organizer(
        llm=llm or _ToolCallingLLM(),
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        tts=tts,
        confirm_timeout_s=confirm_timeout_s,
        tts_cooldown_s=0.0,
        gate_drain_margin_s=0.0,
    )
    try:
        yield org, sink, tool
    finally:
        await org.close()


async def _wait_for(sink: list, kind: str, timeout_s: float = 2.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        for _, msg in sink:
            if msg.get("type") == kind:
                return msg
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {kind} after {timeout_s}s")


async def _wait_until_asked(org: Organizer, room_id: str, timeout_s: float = 2.0) -> None:
    """Block until the room's pending confirm has its voice arm armed AND the
    room's mic has reopened after the question -- the moment a person could
    first answer."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        pending = org._confirm_by_room.get(room_id)
        if pending is not None and pending.voice_armed:
            # Poll rather than one sleep: a timer may fire a clock tick early
            # on Windows, which would put "now" just inside the closed gate.
            while loop.time() < pending.answers_after:
                await asyncio.sleep(0.01)
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the room was never asked aloud")


def _events(tmp: Path) -> list[dict]:
    events: list[dict] = []
    for path in sorted(tmp.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def _kinds(tmp: Path) -> list[str]:
    return [e["event"] for e in _events(tmp)]


def _transcripts(sink: list, client_id: str = "k-spk") -> list[str]:
    """Transcripts as one client in the room saw them (a broadcast reaches
    every client, so counting all of them double-counts a room of two)."""
    return [
        m["text"] for cid, m in sink if m["type"] == "user_transcript" and cid == client_id
    ]


# ---- the lexicon ------------------------------------------------------------


@pytest.mark.parametrize(
    "text, verdict",
    [
        ("yes", "yes"),
        ("Yes.", "yes"),
        ("yes please", "yes"),
        ("please yes", "yes"),
        ("okay yes", "yes"),
        ("yes yes", "yes"),
        ("yes, go ahead", "yes"),
        ("yes glados", "yes"),
        ("no", "no"),
        ("No.", "no"),
        ("nope", "no"),
        ("no thanks", "no"),
        ("don't", "no"),
        ("do not", "no"),
        ("no no no", "no"),
        ("okay", None),
        ("Okay.", None),
        ("sure", None),
        ("correct", None),
        ("do it", None),
        ("go ahead", None),
        ("Thank you.", None),
        ("yes or no", None),
        ("I know", None),
        ("yesterday", None),
        ("yes add two more", None),
        ("no wait add three", None),
        ("", None),
    ],
)
def test_classify_confirm_answer(text: str, verdict: str | None) -> None:
    assert classify_confirm_answer(text) == verdict


# ---- the renderer -----------------------------------------------------------


def test_render_speaks_fixed_args_before_text_and_quotes_text() -> None:
    q = render_confirm_question(
        "dunnes.add_to_cart_by_name",
        {"product_name": "tomatoes, quantity one", "quantity": 4, "fresh": True},
    )
    assert q == (
        "GLaDOS needs a yes: dunnes add to cart by name, quantity 4, fresh yes, "
        "product name, quote, tomatoes, quantity one, unquote -- shall I go ahead?"
    )


def test_render_nested_and_control_characters() -> None:
    q = render_confirm_question(
        "t.boom", {"ids": [1, 2], "note": "a" + ZERO_WIDTH_SPACE + "b" + BELL + "c  d"}
    )
    assert q is not None
    assert "ids, quote, [1, 2], unquote" in q
    assert "note, quote, abc d, unquote" in q


@pytest.mark.parametrize(
    "args",
    [
        {"name": "x" * 61},
        {f"k{i}": i for i in range(9)},
        {f"key_{i}": "v" * 40 for i in range(7)},
    ],
)
def test_render_refuses_to_clip(args: dict) -> None:
    assert render_confirm_question("t.boom", args) is None


def test_the_question_is_never_an_answer() -> None:
    q = render_confirm_question("t.boom", {"x": 1})
    assert q is not None
    assert classify_confirm_answer(q) is None


# ---- who can confirm --------------------------------------------------------


@pytest.mark.parametrize(
    "bindings, with_tts, expected",
    [
        ([DESK_UI], False, True),
        ([MIC, SPEAKER], True, True),
        ([MIC, SPEAKER], False, False),
        ([MIC], True, False),
        ([SPEAKER], True, False),
        ([], True, False),
    ],
)
async def test_room_can_confirm(tmp_path: Path, bindings, with_tts, expected) -> None:
    room = bindings[0].room_id if bindings else "empty"
    async with _make_org(
        tmp_path, bindings, tts=_FakeTts() if with_tts else None
    ) as (org, _, _):
        assert org._room_can_confirm(room) is expected


# ---- the organizer ----------------------------------------------------------


async def test_spoken_yes_grants(tmp_path: Path) -> None:
    tts = _FakeTts()
    async with _make_org(tmp_path, [MIC, SPEAKER], tts=tts) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        assert tts.spoken == [render_confirm_question("t.boom", {"x": 1})]
        await org.handle_audio_text("k-mic", "Yes.", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == [{"x": 1}]
        kinds = _kinds(tmp_path)
        assert "tool_confirm_spoken" in kinds
        assert kinds.index("tool_confirm_voice") < kinds.index("tool_confirm_response")
        voice = next(e for e in _events(tmp_path) if e["event"] == "tool_confirm_voice")
        assert voice["verdict"] == "yes" and voice["client_id"] == "k-mic"
        # The answer was consumed: not a transcript, not a turn.
        assert _transcripts(sink) == ["do it"]


async def test_spoken_no_denies_and_does_not_fail_the_turn(tmp_path: Path) -> None:
    async with _make_org(tmp_path, [MIC, SPEAKER], tts=_FakeTts()) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        await org.handle_audio_text("k-mic", "no thanks", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == []
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert results[0]["error"] == "user denied"
        outcomes = [m for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes[0]["outcome"] == "needs-user"


async def test_typed_yes_is_a_turn_not_an_answer(tmp_path: Path) -> None:
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=_FakeTts(), confirm_timeout_s=0.3
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        await org.handle_user_text("k-mic", "yes")
        await org.flush()
        assert tool.calls == []
        assert "tool_confirm_timeout" in _kinds(tmp_path)
        assert _transcripts(sink) == ["do it", "yes"]


async def test_yes_from_another_room_is_that_rooms_turn(tmp_path: Path) -> None:
    async with _make_org(
        tmp_path, [MIC, SPEAKER, DESK_UI, DESK_MIC], tts=_FakeTts(), confirm_timeout_s=0.3
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        await org.handle_audio_text("d-mic", "yes", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == []
        assert "tool_confirm_timeout" in _kinds(tmp_path)
        assert _transcripts(sink, "desk-ui") == ["yes"]


async def test_non_answers_queue_bounded_and_the_confirm_still_waits(
    tmp_path: Path,
) -> None:
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=_FakeTts(), confirm_timeout_s=0.3
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        now = asyncio.get_running_loop().time()
        await org.handle_audio_text("k-mic", "what did you say", now)
        await org.handle_audio_text("k-mic", "hello", now)
        assert org._queues.queue_depth("kitchen") == 1
        await org.flush()
        assert tool.calls == []
        assert "tool_confirm_timeout" in _kinds(tmp_path)
        assert _transcripts(sink) == ["do it", "what did you say"]


async def test_an_answer_captured_before_the_question_ended_is_not_one(
    tmp_path: Path,
) -> None:
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=_FakeTts(), confirm_timeout_s=0.3
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        early = org._confirm_by_room["kitchen"].answers_after - 0.05
        # Arrives after the mic reopened but BEGAN while the question was
        # still playing; the TTS gate judges capture time too, so bypass it
        # the way a room with no speaker-driven gate would.
        await org.handle_user_text("k-mic", "yes", source="voice", captured_at=early)
        await org.flush()
        assert tool.calls == []
        assert "tool_confirm_timeout" in _kinds(tmp_path)


async def test_the_ttl_runs_from_the_mic_reopening(tmp_path: Path) -> None:
    # A one-second question with a 0.3 s ttl: measured from `_speak`'s
    # return the confirm would time out while the question was still
    # playing; measured from the gate horizon it is still open afterwards.
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=_FakeTts(seconds=1.0), confirm_timeout_s=0.3
    ) as (org, sink, tool):
        started = asyncio.get_running_loop().time()
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        assert asyncio.get_running_loop().time() - started >= 1.0
        assert "tool_confirm_timeout" not in _kinds(tmp_path)
        await org.handle_audio_text("k-mic", "yes", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == [{"x": 1}]


async def test_dialog_answer_during_the_question_cuts_it_short(tmp_path: Path) -> None:
    class _SlowTts(_FakeTts):
        async def synthesize(self, text: str):
            self.spoken.append(text)
            await asyncio.sleep(0.5)
            yield TtsChunkOut(pcm=b"\x00\x00" * 100, sample_rate=SAMPLE_RATE)

    tts = _SlowTts()
    async with _make_org(tmp_path, [DESK_UI, DESK_MIC], tts=tts) as (org, sink, tool):
        await org.handle_user_text("d-mic", "do it")
        req = await _wait_for(sink, "tool_confirm_request")
        await asyncio.sleep(0.1)
        assert tts.spoken, "the question was not being spoken"
        await org.handle_tool_confirm_response(
            "desk-ui", ToolConfirmResponse(request_id=req["request_id"], granted=True)
        )
        await org.flush()
        assert tool.calls == [{"x": 1}]
        kinds = _kinds(tmp_path)
        assert "tool_confirm_voice" not in kinds
        assert "tool_confirm_voice_skipped" not in kinds
        assert kinds.count("tool_confirm_response") == 1
        # Question chunks never streamed; only the reply's did.
        assert kinds.index("tool_confirm_response") < kinds.index("tts_chunk")


async def test_barge_in_during_the_question_leaves_the_gate_in_cooldown(
    tmp_path: Path,
) -> None:
    class _SlowTts(_FakeTts):
        async def synthesize(self, text: str):
            self.spoken.append(text)
            await asyncio.sleep(0.5)
            yield TtsChunkOut(pcm=b"\x00\x00" * 100, sample_rate=SAMPLE_RATE)

    tts = _SlowTts()
    async with _make_org(tmp_path, [MIC, SPEAKER], tts=tts) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_for(sink, "tool_confirm_request")
        await asyncio.sleep(0.1)
        assert tts.spoken
        await org.handle_audio_text("k-mic", "stop", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == []
        assert org._pending_confirms == {} and org._confirm_by_room == {}
        assert "tts_chunk" not in _kinds(tmp_path)
        gate = org._tts_gate.get("kitchen")
        assert gate is None or gate.phase == "cooldown"


async def test_a_speaker_credential_cannot_answer(tmp_path: Path) -> None:
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=_FakeTts(), confirm_timeout_s=0.3
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        await org.handle_audio_text("k-spk", "yes", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == []
        assert "tool_confirm_timeout" in _kinds(tmp_path)


async def test_a_question_nobody_could_hear_leaves_the_voice_arm_off(
    tmp_path: Path,
) -> None:
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=_FakeTts(fail=True), confirm_timeout_s=0.2
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        skipped_at = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < skipped_at:
            if "tool_confirm_voice_skipped" in _kinds(tmp_path):
                break
            await asyncio.sleep(0.01)
        pending = org._confirm_by_room.get("kitchen")
        assert pending is not None and not pending.voice_armed
        await org.handle_audio_text("k-mic", "yes", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == []
        skipped = [e for e in _events(tmp_path) if e["event"] == "tool_confirm_voice_skipped"]
        assert skipped and skipped[0]["reason"] == "no_audio"


async def test_a_clipped_question_is_not_asked_aloud(tmp_path: Path) -> None:
    llm = _ToolCallingLLM(args={"name": "x" * 100})
    tts = _FakeTts()
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=tts, llm=llm, confirm_timeout_s=0.2
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await org.flush()
        assert not any(t.startswith("GLaDOS needs a yes") for t in tts.spoken)
        skipped = [e for e in _events(tmp_path) if e["event"] == "tool_confirm_voice_skipped"]
        assert skipped and skipped[0]["reason"] == "clipped"


async def test_cancellation_clears_both_indexes(tmp_path: Path) -> None:
    async with _make_org(tmp_path, [MIC, SPEAKER], tts=_FakeTts()) as (org, sink, tool):
        await org.handle_user_text("k-mic", "do it")
        await _wait_until_asked(org, "kitchen")
        await org.handle_audio_text("k-mic", "stop", asyncio.get_running_loop().time())
        await org.flush()
        assert org._pending_confirms == {}
        assert org._confirm_by_room == {}
        assert tool.calls == []


# ---- the outcome ------------------------------------------------------------


def test_a_refused_confirm_is_needs_user_not_failed() -> None:
    turn = TurnRecord(final_text="Fine, I won't.", action_intent=True, confirm_refused=True)
    assert classify(turn) == "needs-user"


async def test_a_refused_confirm_never_escalates_even_when_failed(tmp_path: Path) -> None:
    # An earlier unrecovered error keeps the record `failed`; the flag alone
    # must stop the specialist re-driving (and re-asking) the request.
    turn = TurnRecord(final_text="Search failed, and you said no.", confirm_refused=True)
    turn.record_tool("dunnes.search_products", False)
    assert classify(turn) == "failed"
    async with _make_org(tmp_path, [MIC, SPEAKER], tts=_FakeTts()) as (org, _, _):
        org._escalate_on_failed = True
        org._specialist_llm = _ToolCallingLLM()
        assert org._should_escalate("primary", turn) is False
        turn.confirm_refused = False
        assert org._should_escalate("primary", turn) is True


def test_a_refused_confirm_does_not_excuse_a_false_claim() -> None:
    turn = TurnRecord(
        final_text="Added tomatoes to your cart.", action_intent=True, confirm_refused=True
    )
    assert classify(turn) == "confabulated"
