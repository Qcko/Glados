"""Server-side hardening of the confirmation gate (DESIGN-confirm-modal.md).

The desk client's dialog is about to be the only human gate on cart writes.
Two holes the dialog cannot close from the browser: any client in the room
could answer (a mic or speaker token included), and what was approved was
dispatched only by convention -- the same mutable dict, read again after the
await.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMText, LLMToolCall
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.protocols import ToolConfirmResponse
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry
from tests.test_permission_gate import _GatedTool, _wait_for_confirm_request


class _HeldCallLLM:
    """Emits one gated call and keeps hold of the very object it handed over,
    so a test can change its arguments while the confirmation is pending."""

    def __init__(self) -> None:
        self.call = LLMToolCall(call_id="c1", server="t", name="boom", args={"x": 1})
        self._n = 0

    async def chat(self, messages, tools):
        self._n += 1
        if self._n == 1:
            yield self.call
        else:
            yield LLMText(text="done")


class _ReissuingLLM(_HeldCallLLM):
    """Second pass re-issues the call with the arguments the user approved."""

    async def chat(self, messages, tools):
        self._n += 1
        if self._n == 1:
            yield self.call
        elif self._n == 2:
            yield LLMToolCall(call_id="c2", server="t", name="boom", args={"x": 1})
        else:
            yield LLMText(text="done")


class _TimingOutGatedTool(_GatedTool):
    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        self.calls.append(args)
        return MCPCallResult(ok=False, indeterminate=True, error="timeout")


@asynccontextmanager
async def _desk_with_mic(
    tmp: Path,
    llm: _HeldCallLLM,
    *,
    confirm_timeout_s: float = 30.0,
    tool: _GatedTool | None = None,
):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    bindings = [
        ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="u"),
        ClientBinding(client_id="desk-mic", room_id="desk", role="mic", default_user="u"),
        ClientBinding(client_id="desk-speaker", room_id="desk", role="speaker", default_user="u"),
    ]
    by_id = {b.client_id: b for b in bindings}
    mcp = MCPRegistry()
    tool = tool or _GatedTool()
    mcp.register(tool)
    org = Organizer(
        llm=llm,
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


@pytest.mark.parametrize("client_id", ["desk-mic", "desk-speaker"])
async def test_a_room_device_cannot_answer(tmp_path: Path, client_id: str) -> None:
    """Same room, wrong role: the grant is dropped and the timeout denies."""
    llm = _HeldCallLLM()
    async with _desk_with_mic(tmp_path, llm, confirm_timeout_s=0.3) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        await org.handle_tool_confirm_response(
            client_id, ToolConfirmResponse(request_id=req["request_id"], granted=True)
        )
        await org.flush()

    assert tool.calls == []
    results = [m for _, m in sink if m["type"] == "tool_result"]
    assert results and results[0]["error"] == "user denied"


async def test_a_dropped_device_answer_does_not_use_up_the_request(tmp_path: Path) -> None:
    llm = _HeldCallLLM()
    async with _desk_with_mic(tmp_path, llm) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        for client_id in ("desk-mic", "desk-ui"):
            await org.handle_tool_confirm_response(
                client_id, ToolConfirmResponse(request_id=req["request_id"], granted=True)
            )
        await org.flush()

    assert tool.calls == [{"x": 1}]


async def test_the_in_flight_ledger_records_the_approved_arguments(tmp_path: Path) -> None:
    """Args changed during the wait, the approved x=1 went out and timed out.
    Re-issuing x=1 in the same turn must be refused as outstanding, not
    prompted for again as if nothing with those args had been sent."""
    llm = _ReissuingLLM()
    tool = _TimingOutGatedTool()
    async with _desk_with_mic(tmp_path, llm, tool=tool, confirm_timeout_s=0.5) as (org, sink, _):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        llm.call.args["x"] = 99
        await org.handle_tool_confirm_response(
            "desk-ui", ToolConfirmResponse(request_id=req["request_id"], granted=True)
        )
        await org.flush()

    assert tool.calls == [{"x": 1}]
    request_ids = {m["request_id"] for _, m in sink if m["type"] == "tool_confirm_request"}
    assert len(request_ids) == 1


async def test_dispatch_sends_the_arguments_that_were_approved(tmp_path: Path) -> None:
    """The user approved x=1. A change to the call made while the dialog was
    open must not reach the wire."""
    llm = _HeldCallLLM()
    async with _desk_with_mic(tmp_path, llm) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "do it")
        req = await _wait_for_confirm_request(sink)
        assert req["args_summary"] == {"x": 1}
        llm.call.args["x"] = 99
        await org.handle_tool_confirm_response(
            "desk-ui", ToolConfirmResponse(request_id=req["request_id"], granted=True)
        )
        await org.flush()

    assert tool.calls == [{"x": 1}]
