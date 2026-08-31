"""Tests for the v2 per-tool permission gate.

Exercises the Organizer's `_await_confirmation` path end-to-end:
- gate-granted: client says yes; dispatch happens; LLM sees the result.
- gate-denied: client says no; dispatch skipped; LLM sees "user denied".
- gate-timeout: no response within ttl_s; same as denied.
- cross-room reply rejected: a client in another room cannot answer.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMText, LLMToolCall, ToolSpec
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.protocols import ToolConfirmResponse
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry


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
    """First turn: emit one tool_call for `t.boom`. Second turn: a final
    text reply. Mimics the loop the Organizer drives until no more tool
    calls arrive."""

    def __init__(self) -> None:
        self._n = 0

    async def chat(self, messages, tools):
        self._n += 1
        if self._n == 1:
            yield LLMToolCall(call_id="c1", server="t", name="boom", args={"x": 1})
        else:
            yield LLMText(text="done")


@asynccontextmanager
async def _make_org(tmp: Path, *, confirm_timeout_s: float = 30.0):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    bindings = [
        ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="u"),
        ClientBinding(client_id="kitchen-ui", room_id="kitchen", role="ui", default_user="u"),
    ]
    by_id = {b.client_id: b for b in bindings}
    mcp = MCPRegistry()
    tool = _GatedTool()
    mcp.register(tool)
    org = Organizer(
        llm=_ToolCallingLLM(),
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        confirm_timeout_s=confirm_timeout_s,
    )
    try:
        yield org, sink, tool
    finally:
        await org.close()


async def _wait_for_confirm_request(sink: list, timeout_s: float = 2.0) -> dict:
    """Spin until a tool_confirm_request lands in the sink, or fail."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        for _, msg in sink:
            if msg.get("type") == "tool_confirm_request":
                return msg
        await asyncio.sleep(0.02)
    raise AssertionError(f"no tool_confirm_request after {timeout_s}s; sink={sink}")


async def test_gate_granted_dispatches(tmp_path: Path) -> None:
    async with _make_org(tmp_path) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        await org.handle_tool_confirm_response(
            "desk-ui",
            ToolConfirmResponse(request_id=req["request_id"], granted=True),
        )
        await org.flush()
        assert tool.calls == [{"x": 1}]
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert results and results[0]["ok"] is True


async def test_gate_denied_skips_dispatch(tmp_path: Path) -> None:
    async with _make_org(tmp_path) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        await org.handle_tool_confirm_response(
            "desk-ui",
            ToolConfirmResponse(request_id=req["request_id"], granted=False),
        )
        await org.flush()
        assert tool.calls == []
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert results and results[0]["ok"] is False
        assert results[0]["error"] == "user denied"


async def test_gate_denied_does_not_mark_turn_failed(tmp_path: Path) -> None:
    # A deliberate user denial is a boundary, not a tool failure: the turn
    # outcome must not be `failed` (which would spuriously escalate to the
    # v2.6 specialist router). The model's final text "done" has no question, so
    # with the denied call skipped the turn classifies as `done`.
    async with _make_org(tmp_path) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        await org.handle_tool_confirm_response(
            "desk-ui",
            ToolConfirmResponse(request_id=req["request_id"], granted=False),
        )
        await org.flush()
        outcomes = [m for _, m in sink if m["type"] == "turn_outcome"]
        assert outcomes and outcomes[0]["outcome"] == "done"


async def test_gate_timeout_denies(tmp_path: Path) -> None:
    async with _make_org(tmp_path, confirm_timeout_s=0.10) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        await org.flush()  # waits past the 100ms timeout
        assert tool.calls == []
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert results and results[0]["ok"] is False
        assert results[0]["error"] == "user denied"


async def test_cross_room_response_is_ignored(tmp_path: Path) -> None:
    async with _make_org(tmp_path, confirm_timeout_s=0.50) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        # kitchen-ui is bound to room=kitchen; the request went to room=desk.
        await org.handle_tool_confirm_response(
            "kitchen-ui",
            ToolConfirmResponse(request_id=req["request_id"], granted=True),
        )
        await org.flush()  # timeout still wins -> denied
        assert tool.calls == []
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert results and results[0]["error"] == "user denied"


# ---- a room with nobody who can answer ---------------------------------


@asynccontextmanager
async def _make_speaker_only_org(tmp: Path, *, confirm_timeout_s: float = 30.0):
    """A livingroom as `configs/rooms.toml` actually ships it: a mic and a
    speaker, no screen. Neither client can send `tool_confirm_response` --
    `client_room/mic.py` and `speaker.py` send `hello` and `playback_done` and
    nothing else."""
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    bindings = [
        ClientBinding(
            client_id="livingroom-mic", room_id="livingroom", role="mic",
            default_user="u",
        ),
        ClientBinding(
            client_id="livingroom-speaker", room_id="livingroom", role="speaker",
            default_user="u",
        ),
    ]
    by_id = {b.client_id: b for b in bindings}
    mcp = MCPRegistry()
    tool = _GatedTool()
    mcp.register(tool)
    org = Organizer(
        llm=_ToolCallingLLM(),
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        confirm_timeout_s=confirm_timeout_s,
    )
    try:
        yield org, sink, tool
    finally:
        await org.close()


async def test_a_room_that_cannot_confirm_is_refused_at_once_not_after_the_ttl(
    tmp_path: Path,
) -> None:
    """A gated call from a speaker-only room was denied either way -- the gate
    is not what this changes. What it changes is that the room used to be MUTE
    for the whole ttl first: the request went out to a microphone and a
    loudspeaker, its FIFO worker held, and the denial arrived half a minute
    later. The ttl here is 30s and the assertion is a 2s bound, so a
    regression fails on the clock rather than on the outcome."""
    async with _make_speaker_only_org(tmp_path) as (org, sink, tool):
        await org.handle_user_text("livingroom-mic", "do it")
        await asyncio.wait_for(org.flush(), 2.0)

        assert tool.calls == []
        results = [m for _, m in sink if m["type"] == "tool_result"]
        assert results and results[0]["ok"] is False
        assert results[0]["error"] == "user denied"


async def test_a_room_that_cannot_confirm_is_never_asked(tmp_path: Path) -> None:
    """Asking is itself the defect, not just the waiting. A request broadcast
    at clients with no way to answer is a prompt nobody will ever see, and on
    a speaker client it is a frame the room cannot even render."""
    async with _make_speaker_only_org(tmp_path, confirm_timeout_s=0.10) as (
        org,
        sink,
        tool,
    ):
        await org.handle_user_text("livingroom-mic", "do it")
        await org.flush()

        assert [m for _, m in sink if m["type"] == "tool_confirm_request"] == []


async def test_a_ui_client_in_the_room_is_what_makes_it_answerable(
    tmp_path: Path,
) -> None:
    """The predicate is "someone here can answer", not "this room has a
    screen-shaped name" -- and it is read from the CONNECTED clients, so a
    room whose only screen has dropped is refused at once for the same reason
    a room that never had one is. Same room as the test above, one client
    added, opposite outcome."""
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    bindings = [
        ClientBinding(
            client_id="livingroom-mic", room_id="livingroom", role="mic",
            default_user="u",
        ),
        ClientBinding(
            client_id="livingroom-tablet", room_id="livingroom", role="ui",
            default_user="u",
        ),
    ]
    by_id = {b.client_id: b for b in bindings}
    mcp = MCPRegistry()
    tool = _GatedTool()
    mcp.register(tool)
    org = Organizer(
        llm=_ToolCallingLLM(),
        mcp=mcp,
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        confirm_timeout_s=30.0,
    )
    try:
        await org.handle_user_text("livingroom-mic", "do it")
        req = await _wait_for_confirm_request(sink)
        await org.handle_tool_confirm_response(
            "livingroom-tablet",
            ToolConfirmResponse(request_id=req["request_id"], granted=True),
        )
        await org.flush()
        assert tool.calls == [{"x": 1}]
    finally:
        await org.close()
