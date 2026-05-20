"""v1 step 5: voice-triggered barge-in.

A short utterance matching `stop|cancel|halt|nevermind|...` arriving via the
audio path (`handle_audio_text`) cancels the speaker's room's in-flight turn
instead of opening a new one. Anything else falls through to
`handle_user_text` unchanged.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMText
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer, _is_barge_in
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry


class SlowLLM:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def chat(self, messages, tools):
        yield LLMText(text="working on it ")
        self.entered.set()
        await asyncio.sleep(3600)


@asynccontextmanager
async def _make_organizer(bindings, tmp, llm):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    by_id = {b.client_id: b for b in bindings}
    org = Organizer(
        llm=llm,
        mcp=MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
    )
    try:
        yield org, sink
    finally:
        await org.close()


# ---- regex unit tests --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "stop",
        "STOP",
        "Stop.",
        "stop!",
        "stop it",
        "stop talking",
        "cancel",
        "halt",
        "nevermind",
        "never mind",
        "shut up",
        "shut up.",
        "be quiet",
        "hey glados stop",
        "glados, stop",
        "stop glados",
        "  stop  ",
        "okay stop",
        "ok stop",
        "alright stop",
        "please stop",
        "stop please",
        "stop, please.",
        "um stop",
    ],
)
def test_barge_in_matches(text: str) -> None:
    assert _is_barge_in(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "stop the timer",
        "what time is it",
        "tell me a story about cancellation",
        "i need to halt the deploy",
        "do not stop",
        "stop everything you are doing right now please",
    ],
)
def test_barge_in_rejects(text: str) -> None:
    assert not _is_barge_in(text)


# ---- handle_audio_text routing -----------------------------------------


@pytest.mark.asyncio
async def test_audio_text_cancels_inflight_turn(tmp_path: Path) -> None:
    llm = SlowLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm,
    ) as (org, sink):
        await org.handle_audio_text("desk-ui", "hello there")
        await llm.entered.wait()
        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]

        # Second utterance arrives via audio: short barge-in. Must cancel sid,
        # not open a new turn.
        await org.handle_audio_text("desk-ui", "stop.")
        await org.flush()

        types = [m["type"] for _, m in sink]
        assert types.count("welcome") == 1, "barge-in must not open a new session"
        assert "cancelled" in types
        assert "done" not in types
        assert next(m for _, m in sink if m["type"] == "cancelled")["session_id"] == sid


@pytest.mark.asyncio
async def test_audio_text_without_active_session_falls_through(tmp_path: Path) -> None:
    """A barge-in utterance with no in-flight turn AND empty queue should
    NOT silently drop — it falls through to a normal turn (Whisper false
    positives are real; better to answer the user than swallow the input)."""
    from glados.brain.llm.fake import FakeLLM

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        FakeLLM(),
    ) as (org, sink):
        await org.handle_audio_text("desk-ui", "stop")
        await org.flush()
        types = [m["type"] for _, m in sink]
        assert "welcome" in types
        assert "done" in types
        assert "cancelled" not in types


@pytest.mark.asyncio
async def test_non_barge_in_does_not_cancel_active_session(tmp_path: Path) -> None:
    """Even with an active session in the same room, a non-barge-in audio
    utterance must NOT cancel it. Proves the regex gate is doing real work
    in the routing layer."""
    llm = SlowLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm,
    ) as (org, sink):
        await org.handle_audio_text("desk-ui", "first question")
        await llm.entered.wait()

        # Second utterance is a normal question. Must NOT cancel; instead
        # queues behind the active turn. The queued turn never runs because
        # the first is still hanging on SlowLLM — that's fine for this assertion.
        await org.handle_audio_text("desk-ui", "what time is it")
        await asyncio.sleep(0.05)
        assert not any(m["type"] == "cancelled" for _, m in sink)
        assert org._queues.queue_depth("desk") == 1, "second turn queued behind first"


@pytest.mark.asyncio
async def test_audio_text_non_match_routes_to_user_text(tmp_path: Path) -> None:
    from glados.brain.llm.fake import FakeLLM

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        FakeLLM(),
    ) as (org, sink):
        await org.handle_audio_text("desk-ui", "what time is it")
        await org.flush()
        types = [m["type"] for _, m in sink]
        assert "welcome" in types
        assert "done" in types


@pytest.mark.asyncio
async def test_audio_text_unknown_client_is_noop(tmp_path: Path) -> None:
    from glados.brain.llm.fake import FakeLLM

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        FakeLLM(),
    ) as (org, sink):
        await org.handle_audio_text("ghost", "stop")
        assert sink == []


@pytest.mark.asyncio
async def test_barge_in_only_cancels_same_room(tmp_path: Path) -> None:
    """A barge-in spoken in room A must not cancel a turn running in
    room B, even if A has no active session of its own."""
    llm = SlowLLM()
    async with _make_organizer(
        [
            ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"),
            ClientBinding(client_id="desk2-ui", room_id="desk2", role="ui", default_user="anna"),
        ],
        tmp_path,
        llm,
    ) as (org, sink):
        await org.handle_audio_text("desk-ui", "hello")
        await llm.entered.wait()
        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]

        # desk2 speaker says "stop" — should NOT cancel desk's session.
        # No turn or queued items in desk2, so it falls through to a fresh
        # turn (also hanging on SlowLLM).
        await org.handle_audio_text("desk2-ui", "stop")
        await asyncio.sleep(0.05)  # let desk2's worker enter its LLM
        assert not any(
            m["type"] == "cancelled" and m["session_id"] == sid for _, m in sink
        )
